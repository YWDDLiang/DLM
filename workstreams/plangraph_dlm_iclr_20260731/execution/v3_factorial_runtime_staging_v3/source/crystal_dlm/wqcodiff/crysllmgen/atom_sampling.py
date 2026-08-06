"""Attempt-accounted CrysLLMGen atom proposal and CSP refinement baselines."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..contracts import (
    ArtifactLedger,
    AttemptLedger,
    AttemptRecord,
    AttemptStatus,
    SeedDeriver,
    write_json_exclusive,
)
from ..vocabulary import MP20_ATOMIC_NUMBERS
from .atom_text import AtomTextFields, parse_upstream_atom_text_fields
from .gate import GateALock, sha256_file
from .lora import validate_trained_adapter
from .protocol import load_protocol_v4
from .schedules import (
    OFFICIAL_REVERSE_START_TIMESTEP,
    PARENT_RUN_TYPE,
    PARENT_SCHEDULER_TIMESTEPS,
)
from .sft_data import ATOM_SYSTEM_PROMPT, UNCONDITIONAL_USER_PROMPT


OFFICIAL_PROMPT = (
    "Below is a description of a bulk material. "
    "Generate a description of the lengths and angles of the lattice vectors "
    "and then the element type and coordinates for each atom within the lattice:\n"
)


@dataclasses.dataclass(frozen=True, slots=True)
class CrysLLMGenAtomSamplingConfig:
    protocol_path: str
    gate_a_lock: str
    csp_checkpoint: str
    llama_root: str
    llama_adapter: str
    output_jsonl: str
    attempt_ledger: str
    report_path: str
    experiment_id: str
    pairing_id: str
    method: str
    training_seed: int
    sampling_seed: int
    attempts: int
    start_ordinal: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.method not in {"C-ATOM-OFFICIAL", "C-ATOM-MATCHED"}:
            raise ValueError("unsupported registered atom method")
        if self.training_seed not in {11, 23, 47}:
            raise ValueError("training seed is outside 11/23/47")
        if self.method == "C-ATOM-OFFICIAL" and self.training_seed != 11:
            raise ValueError("the frozen official checkpoint has one identity, assigned seed 11")
        if self.attempts <= 0 or self.start_ordinal < 0:
            raise ValueError("invalid attempt denominator or start ordinal")
        if not self.experiment_id or not self.pairing_id:
            raise ValueError("experiment and pairing identities are required")

    @property
    def reverse_steps(self) -> int:
        return (
            OFFICIAL_REVERSE_START_TIMESTEP
            if self.method == "C-ATOM-OFFICIAL"
            else 32
        )

    @property
    def csp_forwards(self) -> int:
        return 2 * self.reverse_steps


class AtomAttemptFailure(RuntimeError):
    def __init__(
        self,
        cause: Exception,
        *,
        calls: Mapping[str, int],
        raw_text: str = "",
        proposal_usage: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{type(cause).__name__}:{cause}")
        self.calls = dict(calls)
        self.raw_text = raw_text
        self.proposal_usage = dict(proposal_usage or {})


def _derived_subseed(seed: int, label: str) -> int:
    raw = f"{seed}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


class AtomLlamaEngine:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        base_root: Path,
        adapter_root: Path,
        official_prompt: bool,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.base_root = base_root
        self.adapter_root = adapter_root
        self.official_prompt = official_prompt
        self.identity = {
            "base_root": str(base_root),
            "adapter_root": str(adapter_root),
            "adapter_model_sha256": sha256_file(
                adapter_root / "adapter_model.safetensors"
            ),
            "adapter_config_sha256": sha256_file(adapter_root / "adapter_config.json"),
            "prompt_mode": "upstream_plain" if official_prompt else "registered_chat",
            "tokenizer_size": len(tokenizer),
            "parameter_count_with_adapter": sum(
                value.numel() for value in model.parameters()
            ),
        }

    @classmethod
    def load(
        cls,
        *,
        base_root: str | Path,
        adapter_root: str | Path,
        official_prompt: bool,
    ) -> "AtomLlamaEngine":
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = Path(base_root).resolve()
        adapter = Path(adapter_root).resolve()
        for path in (
            adapter / "adapter_model.safetensors",
            adapter / "adapter_config.json",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        tokenizer = AutoTokenizer.from_pretrained(
            base,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
            model_max_length=512,
            padding_side="left" if official_prompt else "right",
        )
        if tokenizer.eos_token_id is None:
            raise RuntimeError("registered tokenizer has no EOS token")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if not official_prompt and not tokenizer.chat_template:
            raise RuntimeError("matched atom model requires the registered chat template")
        model = AutoModelForCausalLM.from_pretrained(
            base,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )
        model = PeftModel.from_pretrained(
            model,
            adapter,
            local_files_only=True,
            is_trainable=False,
        )
        model.eval()
        model.config.use_cache = True
        return cls(
            model=model,
            tokenizer=tokenizer,
            base_root=base,
            adapter_root=adapter,
            official_prompt=official_prompt,
        )

    def _prompt(self) -> tuple[torch.Tensor, str]:
        if self.official_prompt:
            rendered = OFFICIAL_PROMPT
            tokens = self.tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"]
        else:
            messages = [
                {"role": "system", "content": ATOM_SYSTEM_PROMPT},
                {"role": "user", "content": UNCONDITIONAL_USER_PROMPT},
            ]
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            tokens = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        return tokens.to(next(self.model.parameters()).device), rendered

    def generate(self, *, seed: int) -> tuple[AtomTextFields, str, dict[str, Any]]:
        prompt, rendered = self._prompt()
        prompt_width = int(prompt.shape[1])
        max_new_tokens = 500 if self.official_prompt else 512 - prompt_width
        if max_new_tokens <= 0:
            raise ValueError("registered atom prompt consumes the complete context")
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=prompt,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=0.9,
                top_p=0.9,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if self.official_prompt:
            decoded = self.tokenizer.batch_decode(
                output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            text = decoded.replace(rendered, "")
        else:
            text = self.tokenizer.decode(
                output[0, prompt_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        usage = {
            "prompt_tokens": prompt_width,
            "generated_tokens": int(output.shape[1] - prompt_width),
            "llama_invocations": 1,
            "walltime_s": time.monotonic() - started,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        try:
            fields = parse_upstream_atom_text_fields(text)
        except Exception as exc:
            raise AtomAttemptFailure(
                exc,
                calls={"llama": 1, "llama_tokens": usage["generated_tokens"], "csp": 0},
                raw_text=text,
                proposal_usage=usage,
            ) from exc
        return fields, text, usage


@dataclasses.dataclass(slots=True)
class _SingleBatch:
    num_graphs: int
    num_nodes: int
    num_atoms: torch.Tensor
    atom_types: torch.Tensor
    frac_coords: torch.Tensor
    lengths: torch.Tensor
    angles: torch.Tensor
    batch: torch.Tensor

    def to(self, device: torch.device) -> "_SingleBatch":
        return _SingleBatch(
            num_graphs=self.num_graphs,
            num_nodes=self.num_nodes,
            num_atoms=self.num_atoms.to(device),
            atom_types=self.atom_types.to(device),
            frac_coords=self.frac_coords.to(device),
            lengths=self.lengths.to(device),
            angles=self.angles.to(device),
            batch=self.batch.to(device),
        )


def expanded_state_to_parent_batch(
    expanded: Any,
    device: torch.device,
) -> _SingleBatch:
    """Convert one expanded WQ proposal into the released parent-CSP contract.

    The conversion deliberately uses the primitive lattice and coordinates
    produced by ``expand_state``.  It does not redetect, canonicalize, relax,
    or otherwise alter the proposal, so a parent-CSP probe and a WQ-refiner
    attempt can share the same proposal seed without a hidden geometry change.
    """

    lattice = np.asarray(expanded.primitive_lattice, dtype=np.float64)
    coordinates = np.asarray(expanded.fractional_coordinates, dtype=np.float64)
    atomic_numbers = np.asarray(expanded.atomic_numbers, dtype=np.int64)
    if lattice.shape != (3, 3):
        raise ValueError("expanded primitive lattice must be 3x3")
    if coordinates.shape != (len(atomic_numbers), 3):
        raise ValueError("expanded primitive coordinates have an invalid shape")
    if not 1 <= len(atomic_numbers) <= 20:
        raise ValueError("expanded proposal is outside the parent MP20 atom range")
    if any(not 1 <= int(value) < 100 for value in atomic_numbers):
        raise ValueError("expanded proposal is outside parent CSP species support")
    if not (
        np.all(np.isfinite(lattice))
        and np.all(np.isfinite(coordinates))
    ):
        raise FloatingPointError("expanded proposal contains non-finite geometry")

    lengths = np.linalg.norm(lattice, axis=1)
    if np.any(lengths <= 1.0e-8) or not np.all(np.isfinite(lengths)):
        raise ValueError("expanded proposal has a degenerate primitive lattice")

    def angle(first: np.ndarray, second: np.ndarray) -> float:
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        cosine = float(np.dot(first, second) / denominator)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    # pymatgen/CrysLLMGen use alpha=(b,c), beta=(a,c), gamma=(a,b).
    angles = np.asarray(
        [
            angle(lattice[1], lattice[2]),
            angle(lattice[0], lattice[2]),
            angle(lattice[0], lattice[1]),
        ],
        dtype=np.float64,
    )
    if np.any(angles <= 0.0) or np.any(angles >= 180.0):
        raise ValueError("expanded proposal has invalid primitive lattice angles")
    return _SingleBatch(
        num_graphs=1,
        num_nodes=len(atomic_numbers),
        num_atoms=torch.tensor(
            [len(atomic_numbers)],
            dtype=torch.long,
            device=device,
        ),
        atom_types=torch.tensor(atomic_numbers, dtype=torch.long, device=device),
        frac_coords=torch.tensor(
            coordinates,
            dtype=torch.float32,
            device=device,
        ),
        lengths=torch.tensor(
            [lengths],
            dtype=torch.float32,
            device=device,
        ),
        angles=torch.tensor(
            [angles],
            dtype=torch.float32,
            device=device,
        ),
        batch=torch.zeros(len(atomic_numbers), dtype=torch.long, device=device),
    )


def _fields_to_batch(fields: AtomTextFields, device: torch.device) -> _SingleBatch:
    from pymatgen.core import Structure
    from pymatgen.core.lattice import Lattice

    if not 1 <= fields.num_atoms <= 20:
        raise ValueError("atom proposal is outside the registered 1-20 atom support")
    structure = Structure(
        lattice=Lattice.from_parameters(*(fields.lengths + fields.angles)),
        species=fields.species,
        coords=fields.frac_coords,
        coords_are_cartesian=False,
    )
    atomic_numbers = structure.atomic_numbers
    if any(not 1 <= int(value) < 100 for value in atomic_numbers):
        raise ValueError("atom proposal is outside parent CSP species support")
    return _SingleBatch(
        num_graphs=1,
        num_nodes=len(structure),
        num_atoms=torch.tensor([len(structure)], dtype=torch.long, device=device),
        atom_types=torch.tensor(atomic_numbers, dtype=torch.long, device=device),
        frac_coords=torch.tensor(
            structure.frac_coords,
            dtype=torch.float32,
            device=device,
        ),
        lengths=torch.tensor([structure.lattice.abc], dtype=torch.float32, device=device),
        angles=torch.tensor(
            [structure.lattice.angles], dtype=torch.float32, device=device
        ),
        batch=torch.zeros(len(structure), dtype=torch.long, device=device),
    )


def load_registered_csp(
    *, snapshot_root: str | Path, checkpoint: str | Path, device: torch.device
) -> tuple[Any, dict[str, Any]]:
    root = Path(snapshot_root).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    sys.path.insert(0, str(root))
    try:
        from models_ddpm.diffusion import CSPDiffusion
    finally:
        sys.path.pop(0)
    model = CSPDiffusion(PARENT_SCHEDULER_TIMESTEPS, PARENT_RUN_TYPE)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("registered CrysLLMGen checkpoint has no model state")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("registered CrysLLMGen checkpoint mapping is not strict")
    model = model.to(device).eval()
    return model, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameters": sum(value.numel() for value in model.parameters()),
        "run_type": "train",
        "scheduler_timesteps": PARENT_SCHEDULER_TIMESTEPS,
        "official_reverse_start_timestep": OFFICIAL_REVERSE_START_TIMESTEP,
        "parent_run_type": PARENT_RUN_TYPE,
    }


def _respaced_timesteps(
    total: int = OFFICIAL_REVERSE_START_TIMESTEP, steps: int = 32
) -> tuple[int, ...]:
    values = np.rint(np.linspace(total, 1, steps)).astype(int)
    result = tuple(int(value) for value in values)
    if len(result) != steps or len(set(result)) != steps or result[0] != total or result[-1] != 1:
        raise RuntimeError("invalid registered respaced timestep grid")
    return result


@torch.no_grad()
def _sample_respaced_32(model: Any, batch: _SingleBatch) -> tuple[dict[str, torch.Tensor], int]:
    """Two parent-CSP calls per skip step over the complete 800->0 horizon."""

    x = batch.frac_coords
    lattice = __import__(
        "models_ddpm.data_utils", fromlist=["lattice_params_to_matrix_torch"]
    ).lattice_params_to_matrix_torch(batch.lengths, batch.angles)
    calls = 0
    grid = _respaced_timesteps()
    for index, timestep in enumerate(grid):
        next_timestep = grid[index + 1] if index + 1 < len(grid) else 0
        times = torch.full((1,), timestep, device=x.device, dtype=torch.long)
        time_embedding = model.time_embedding(times)
        sigma_x = model.sigma_scheduler.sigmas[timestep]
        sigma_norm = model.sigma_scheduler.sigmas_norm[timestep]
        next_sigma_x = model.sigma_scheduler.sigmas[next_timestep]

        random_coordinate = torch.randn_like(x) if next_timestep > 0 else torch.zeros_like(x)
        corrector_step = 1.0e-5 * (
            sigma_x / model.sigma_scheduler.sigma_begin
        ).square()
        coordinate_noise = torch.sqrt(2.0 * corrector_step)
        predicted_lattice, predicted_coordinate = model.decoder(
            time_embedding,
            batch.atom_types,
            x,
            lattice,
            batch.num_atoms,
            batch.batch,
        )
        calls += 1
        predicted_coordinate = predicted_coordinate * torch.sqrt(sigma_norm)
        x_half = x - corrector_step * predicted_coordinate + coordinate_noise * random_coordinate

        predicted_lattice, predicted_coordinate = model.decoder(
            time_embedding,
            batch.atom_types,
            x_half,
            lattice,
            batch.num_atoms,
            batch.batch,
        )
        calls += 1
        predicted_coordinate = predicted_coordinate * torch.sqrt(sigma_norm)
        coordinate_step = sigma_x.square() - next_sigma_x.square()
        coordinate_std = torch.sqrt(
            (next_sigma_x.square() * coordinate_step / sigma_x.square()).clamp_min(0.0)
        )
        random_coordinate = torch.randn_like(x) if next_timestep > 0 else torch.zeros_like(x)
        x = (x_half - coordinate_step * predicted_coordinate + coordinate_std * random_coordinate) % 1.0

        alpha_bar = model.beta_scheduler.alphas_cumprod[timestep]
        next_alpha_bar = model.beta_scheduler.alphas_cumprod[next_timestep]
        clean_lattice = (
            lattice - torch.sqrt(1.0 - alpha_bar) * predicted_lattice
        ) / torch.sqrt(alpha_bar)
        random_lattice = (
            torch.randn_like(lattice) if next_timestep > 0 else torch.zeros_like(lattice)
        )
        lattice = (
            torch.sqrt(next_alpha_bar) * clean_lattice
            + torch.sqrt(1.0 - next_alpha_bar) * random_lattice
        )
        if not torch.isfinite(x).all() or not torch.isfinite(lattice).all():
            raise FloatingPointError("non-finite respaced CSP state")
    if calls != 64:
        raise RuntimeError("matched CSP call contract changed")
    return {
        "num_atoms": batch.num_atoms,
        "atom_types": batch.atom_types,
        "frac_coords": x,
        "lattices": lattice,
    }, calls


def _output_structure(output: Mapping[str, torch.Tensor]) -> Any:
    from pymatgen.core import Structure
    from pymatgen.core.lattice import Lattice

    lattice = output["lattices"][0].detach().double().cpu().numpy()
    coordinates = output["frac_coords"].detach().double().cpu().numpy() % 1.0
    species = [int(value) for value in output["atom_types"].detach().cpu().tolist()]
    if not np.all(np.isfinite(lattice)) or not np.all(np.isfinite(coordinates)):
        raise FloatingPointError("non-finite atom refinement output")
    structure = Structure(
        lattice=Lattice(lattice),
        species=species,
        coords=coordinates,
        coords_are_cartesian=False,
    )
    if not math.isfinite(float(structure.volume)) or structure.volume <= 1.0e-8:
        raise ValueError("refined atom structure has non-positive volume")
    return structure


def _sample_one(
    *,
    config: CrysLLMGenAtomSamplingConfig,
    llama: AtomLlamaEngine,
    csp: Any,
    attempt_id: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    calls = {"llama": 0, "llama_tokens": 0, "csp": 0}
    raw_text = ""
    usage: dict[str, Any] = {}
    try:
        fields, raw_text, usage = llama.generate(
            seed=_derived_subseed(seed, "llama_proposal")
        )
        calls["llama"] = 1
        calls["llama_tokens"] = int(usage["generated_tokens"])
        batch = _fields_to_batch(fields, device)
        torch.manual_seed(_derived_subseed(seed, "csp_refinement"))
        torch.cuda.manual_seed_all(_derived_subseed(seed, "csp_refinement"))
        with torch.inference_mode():
            if config.method == "C-ATOM-OFFICIAL":
                output, _ = csp.sample(
                    batch,
                    step_lr=1.0e-5,
                    diff_steps=OFFICIAL_REVERSE_START_TIMESTEP,
                )
                calls["csp"] = 1600
            else:
                output, count = _sample_respaced_32(csp, batch)
                calls["csp"] = count
        if calls["csp"] != config.csp_forwards:
            raise RuntimeError("atom CSP forward-call contract changed")
        structure = _output_structure(output)
        cif = structure.to(fmt="cif")
    except AtomAttemptFailure:
        raise
    except Exception as exc:
        raise AtomAttemptFailure(
            exc,
            calls=calls,
            raw_text=raw_text,
            proposal_usage=usage,
        ) from exc
    return {
        "schema": "wqcodiff_generation_attempt_v1",
        "producer_schema": "crysllmgen_atom_generation_attempt_v1",
        "attempt_id": attempt_id,
        "method": config.method,
        "training_seed": config.training_seed,
        "sampling_seed": config.sampling_seed,
        "status": AttemptStatus.SUCCEEDED.value,
        "reason": "",
        "proposal_text": raw_text,
        "proposal_usage": usage,
        "proposal_fields": fields.to_dict(),
        "structure": structure.as_dict(),
        "structure_cif_sha256": hashlib.sha256(cif.encode("utf-8")).hexdigest(),
        "atom_count": len(structure),
        "mp20_species_support": all(
            int(value) in MP20_ATOMIC_NUMBERS for value in structure.atomic_numbers
        ),
        "reverse_steps": config.reverse_steps,
        "respaced_timesteps": (
            None if config.method == "C-ATOM-OFFICIAL" else list(_respaced_timesteps())
        ),
        "calls": calls,
    }


def sample(config: CrysLLMGenAtomSamplingConfig) -> dict[str, Any]:
    protocol = load_protocol_v4(config.protocol_path)
    project_root = Path(config.protocol_path).resolve().parents[3]
    gate = GateALock.load(
        config.gate_a_lock,
        project_root=project_root,
        protocol_path=config.protocol_path,
    )
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("registered atom sampling requires CUDA")
    expected_csp = str(protocol.data["assets"]["cspdiffusion"]["sha256"])
    if sha256_file(config.csp_checkpoint) != expected_csp:
        raise ValueError("atom sampler CSP checkpoint differs from the protocol")
    official = config.method == "C-ATOM-OFFICIAL"
    if not official and tuple(
        int(value)
        for value in protocol.data["sampling"]["matched_atom_respacing"]["timesteps"]
    ) != _respaced_timesteps():
        raise ValueError("atom sampler/protocol respaced grids differ")
    if official:
        if Path(config.llama_adapter).resolve() != Path(
            str(protocol.data["assets"]["official_atom_lora"]["path"])
        ).resolve():
            raise ValueError("official atom method must use the frozen official adapter")
        adapter_training_identity: Mapping[str, Any] = {
            "role": "frozen_upstream_official",
            "gate_a_artifact": gate.payload["artifacts"]["llama_offline_forward"],
        }
    else:
        adapter_training_identity = validate_trained_adapter(
            adapter_root=config.llama_adapter,
            gate_a_lock_sha256=gate.sha256,
            source_bundle_sha256=gate.source_bundle_sha256,
            representation="atom",
            training_stage="coarse",
            training_seed=config.training_seed,
        )
    llama = AtomLlamaEngine.load(
        base_root=config.llama_root,
        adapter_root=config.llama_adapter,
        official_prompt=official,
    )
    csp, csp_identity = load_registered_csp(
        snapshot_root=project_root / "crystal_dlm/wqcodiff/crysllmgen/upstream",
        checkpoint=config.csp_checkpoint,
        device=device,
    )
    # The upstream sampler writes one progress bar per attempt. Suppress only
    # that presentation side effect; tensor operations and RNG order are intact.
    if official and "models_ddpm.diffusion" in sys.modules:
        sys.modules["models_ddpm.diffusion"].tqdm = lambda iterable: iterable
    output_path = Path(config.output_jsonl).resolve()
    ledger_path = Path(config.attempt_ledger).resolve()
    report_path = Path(config.report_path).resolve()
    for path in (output_path, ledger_path, report_path):
        if path.exists():
            raise FileExistsError(f"atom sampling evidence is immutable: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactLedger(output_path)
    attempts = AttemptLedger(ledger_path)
    deriver = SeedDeriver(protocol.name, config.experiment_id)
    pairing_deriver = SeedDeriver(protocol.name, config.pairing_id)
    model_identity = {
        **llama.identity,
        "adapter_training": adapter_training_identity,
        "csp": csp_identity,
    }
    parameter_count = int(csp_identity["parameters"])
    expected_ids: list[str] = []
    for ordinal in range(config.start_ordinal, config.start_ordinal + config.attempts):
        attempt_id = deriver.attempt_id(
            training_seed=config.training_seed,
            sampling_seed=config.sampling_seed,
            ordinal=ordinal,
            method=config.method,
        )
        expected_ids.append(attempt_id)
        ledger_seed = deriver.derive(
            training_seed=config.training_seed,
            sampling_seed=config.sampling_seed,
            attempt_id=attempt_id,
            stage="generation",
        )
        attempts.append(
            AttemptRecord(
                attempt_id=attempt_id,
                method=config.method,
                training_seed=config.training_seed,
                sampling_seed=config.sampling_seed,
                stage="generation",
                status=AttemptStatus.SUBMITTED,
                seed=ledger_seed,
                metadata={
                    "ordinal": ordinal,
                    "pair_id": pairing_deriver.pair_id(
                        training_seed=config.training_seed,
                        sampling_seed=config.sampling_seed,
                        ordinal=ordinal,
                    ),
                },
            )
        )
    succeeded = 0
    failed = 0
    started_all = time.monotonic()
    for ordinal, attempt_id in zip(
        range(config.start_ordinal, config.start_ordinal + config.attempts),
        expected_ids,
    ):
        ledger_seed = deriver.derive(
            training_seed=config.training_seed,
            sampling_seed=config.sampling_seed,
            attempt_id=attempt_id,
            stage="generation",
        )
        sampling_seed = pairing_deriver.paired_derive(
            training_seed=config.training_seed,
            sampling_seed=config.sampling_seed,
            ordinal=ordinal,
            stage="generation_sampling",
        )
        started = time.monotonic()
        calls = {"llama": 0, "llama_tokens": 0, "csp": 0}
        try:
            row = _sample_one(
                config=config,
                llama=llama,
                csp=csp,
                attempt_id=attempt_id,
                seed=sampling_seed,
                device=device,
            )
            calls = dict(row["calls"])
            status = AttemptStatus.SUCCEEDED
            reason = ""
            succeeded += 1
        except AtomAttemptFailure as exc:
            calls = dict(exc.calls)
            status = AttemptStatus.FAILED
            reason = str(exc)
            row = {
                "schema": "wqcodiff_generation_attempt_v1",
                "producer_schema": "crysllmgen_atom_generation_attempt_v1",
                "attempt_id": attempt_id,
                "method": config.method,
                "training_seed": config.training_seed,
                "sampling_seed": config.sampling_seed,
                "status": status.value,
                "reason": reason,
                "proposal_text": exc.raw_text,
                "proposal_usage": exc.proposal_usage,
                "calls": calls,
            }
            failed += 1
        elapsed = time.monotonic() - started
        pair_id = pairing_deriver.pair_id(
            training_seed=config.training_seed,
            sampling_seed=config.sampling_seed,
            ordinal=ordinal,
        )
        row.update(
            {
                "ordinal": ordinal,
                "pair_id": pair_id,
                "ledger_seed": ledger_seed,
                "sampling_seed_derived": sampling_seed,
                "paired_seed": sampling_seed,
                "experiment_id": config.experiment_id,
                "pairing_id": config.pairing_id,
                "stage": "raw",
                "backbone_calls": calls.get("csp", 0),
                "generation_flops_lower_bound": float(
                    2 * parameter_count * calls.get("csp", 0)
                    + 2
                    * int(llama.identity["parameter_count_with_adapter"])
                    * calls.get("llama_tokens", 0)
                ),
                "generation_flops_estimator": "2_parameter_count_per_forward_or_decode_token",
                "walltime_s": elapsed,
                "retry_or_replacement_used": False,
                "model_identity": model_identity,
            }
        )
        digest = artifacts.append(row)
        flops = float(
            2 * parameter_count * calls.get("csp", 0)
            + 2
            * int(llama.identity["parameter_count_with_adapter"])
            * calls.get("llama_tokens", 0)
        )
        attempts.append(
            AttemptRecord(
                attempt_id=attempt_id,
                method=config.method,
                training_seed=config.training_seed,
                sampling_seed=config.sampling_seed,
                stage="generation",
                status=status,
                reason=reason,
                artifact_hash=digest,
                seed=ledger_seed,
                calls=calls,
                flops=flops,
                walltime_s=elapsed,
                metadata={
                    "ordinal": ordinal,
                    "pairing_id": config.pairing_id,
                    "pair_id": pair_id,
                    "sampling_seed_derived": sampling_seed,
                },
            )
        )
    audit = attempts.audit(
        seed_deriver=deriver,
        terminal_stage="generation",
        expected_attempt_ids=expected_ids,
    )
    report = {
        "schema": "crysllmgen_atom_sampling_report_v1",
        "ok": audit.ok and succeeded + failed == config.attempts,
        "method": config.method,
        "training_seed": config.training_seed,
        "sampling_seed": config.sampling_seed,
        "submitted": config.attempts,
        "terminal": succeeded + failed,
        "succeeded": succeeded,
        "failed": failed,
        "reverse_steps": config.reverse_steps,
        "expected_csp_forwards_per_success": config.csp_forwards,
        "retry_or_replacement_used": False,
        "audit": dataclasses.asdict(audit),
        "output_jsonl": str(output_path),
        "output_sha256": sha256_file(output_path),
        "attempt_ledger": str(ledger_path),
        "attempt_ledger_sha256": sha256_file(ledger_path),
        "model_identity": model_identity,
        "protocol_sha256": protocol.sha256,
        "gate_a_lock_sha256": gate.sha256,
        "walltime_s": time.monotonic() - started_all,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(0),
    }
    write_json_exclusive(report_path, report)
    if not report["ok"]:
        raise RuntimeError("atom sampling attempt audit failed")
    return report
