"""Deterministic CrysLLMGen atom/Wyckoff SFT example construction."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..bridge import ChartCatalog
from ..dataset import PRIMARY_SYMPREC, tolerance_tag
from ..runtime import compute_geometry_evidence, expand_state
from ..state import OrbitState, StratifiedState
from ..vocabulary import MP20_ATOMIC_NUMBERS
from .wq_text import TopologyEdit, serialize_topology_edit, serialize_wq_proposal


PROTOCOL_NAME = "crysllmgen_wyckoff_georev_v4"
ATOM_SYSTEM_PROMPT = "Return exactly one atom-coordinate crystal record."
WQ_SYSTEM_PROMPT = "Return exactly one crystal record in the registered grammar."
WQ_EDIT_SYSTEM_PROMPT = "Return one topology edit command."
UNCONDITIONAL_USER_PROMPT = (
    "Generate one unconditional MP20 crystal. Return only the record."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _derive_seed(*parts: Any) -> int:
    digest = hashlib.sha256(
        _canonical_json([PROTOCOL_NAME, *parts]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _primary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema") != "mp20_wq_v1" or not bool(payload.get("selected")):
        raise ValueError("SFT input must be a selected mp20_wq_v1 record")
    decompositions = payload.get("decompositions")
    if not isinstance(decompositions, Mapping):
        raise ValueError("SFT input has no decomposition mapping")
    primary = decompositions.get(tolerance_tag(PRIMARY_SYMPREC))
    if not isinstance(primary, Mapping):
        raise ValueError("SFT input has no primary 1e-2 decomposition")
    return primary


def serialize_atom_structure(
    structure: Any,
    *,
    translation: Sequence[float],
) -> str:
    """Preserve CrysLLMGen's atom record precision with deterministic translation."""

    if len(translation) != 3 or not all(math.isfinite(float(value)) for value in translation):
        raise ValueError("atom translation must contain three finite values")
    lengths = tuple(float(value) for value in structure.lattice.abc)
    angles = tuple(float(value) for value in structure.lattice.angles)
    coordinates = (
        structure.frac_coords
        + __import__("numpy").asarray(translation, dtype=float)[None, :]
    ) % 1.0
    lines = [
        " ".join(f"{value:.1f}" for value in lengths),
        " ".join(str(int(value)) for value in angles),
    ]
    for species, coordinate in zip(structure.species, coordinates):
        lines.append(str(species))
        lines.append(" ".join(f"{float(value):.2f}" for value in coordinate))
    return "\n".join(lines)


def _atom_answer(
    primary: Mapping[str, Any], *, material_id: str, epoch: int, training_seed: int
) -> str:
    try:
        from pymatgen.core import Structure
    except ImportError as exc:  # pragma: no cover - registered server dependency
        raise RuntimeError("pymatgen is required to build matched atom SFT data") from exc
    structure = Structure.from_dict(dict(primary["primitive_structure"]))
    rng = random.Random(_derive_seed("atom_translation", training_seed, epoch, material_id))
    translation = tuple(rng.random() for _ in range(3))
    return serialize_atom_structure(structure, translation=translation)


def build_coarse_example(
    payload: Mapping[str, Any],
    *,
    representation: str,
    epoch: int,
    training_seed: int,
    catalog: ChartCatalog | None = None,
    canonical_orbit_order: bool = False,
) -> dict[str, Any]:
    if representation not in {"atom", "wyckoff"}:
        raise ValueError("representation must be atom or wyckoff")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    material_id = str(payload.get("material_id", ""))
    if not material_id:
        raise ValueError("material_id is required")
    primary = _primary(payload)
    state = StratifiedState.from_dict(dict(primary["state"]))
    if representation == "atom":
        answer = _atom_answer(
            primary,
            material_id=material_id,
            epoch=epoch,
            training_seed=training_seed,
        )
        system_prompt = ATOM_SYSTEM_PROMPT
        order_mode = "deterministic_global_translation"
    else:
        if catalog is None:
            raise ValueError("Wyckoff SFT construction requires a chart catalogue")
        if canonical_orbit_order:
            presented = state.replace_orbits(state.canonical_orbits())
            order_mode = "canonical_ablation"
        else:
            rng = random.Random(
                _derive_seed("orbit_permutation", training_seed, epoch, material_id)
            )
            presented = state.permuted(rng)
            order_mode = "seed_derived_epoch_permutation"
        answer = serialize_wq_proposal(presented, catalog)
        system_prompt = WQ_SYSTEM_PROMPT
    identity = {
        "protocol": PROTOCOL_NAME,
        "material_id": material_id,
        "training_seed": int(training_seed),
        "epoch": int(epoch),
        "representation": representation,
        "order_mode": order_mode,
    }
    example_id = "sft-" + hashlib.sha256(
        _canonical_json(identity).encode("utf-8")
    ).hexdigest()[:24]
    return {
        "schema": "crysllmgen_sft_example_v1",
        "example_id": example_id,
        **identity,
        "system_prompt": system_prompt,
        "user_prompt": UNCONDITIONAL_USER_PROMPT,
        "answer": answer,
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "source_topology_hash": state.topology_hash(include_geometry=True),
        "primitive_atom_count": state.atom_count,
        "orbit_count": len(state.orbits),
    }


def iter_selected_records(paths: Sequence[str | Path]) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not bool(payload.get("selected")):
                    continue
                material_id = str(payload.get("material_id", ""))
                if not material_id or material_id in seen:
                    raise ValueError(f"{path}:{line_number}: duplicate/missing material_id")
                seen.add(material_id)
                yield payload


def write_coarse_sft_jsonl(
    *,
    input_paths: Sequence[str | Path],
    output: str | Path,
    manifest: str | Path,
    representation: str,
    epochs: int,
    training_seed: int,
    catalog: ChartCatalog | None = None,
    canonical_orbit_order: bool = False,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    destination = Path(output)
    manifest_path = Path(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or manifest_path.exists():
        raise FileExistsError("SFT outputs are append-only/exclusive")
    records = list(iter_selected_records(input_paths))
    digest = hashlib.sha256()
    counts_by_epoch = {str(epoch): 0 for epoch in range(epochs)}
    answer_bytes = 0
    with destination.open("xb") as handle:
        for epoch in range(epochs):
            for payload in records:
                example = build_coarse_example(
                    payload,
                    representation=representation,
                    epoch=epoch,
                    training_seed=training_seed,
                    catalog=catalog,
                    canonical_orbit_order=canonical_orbit_order,
                )
                encoded = (_canonical_json(example) + "\n").encode("utf-8")
                handle.write(encoded)
                digest.update(encoded)
                counts_by_epoch[str(epoch)] += 1
                answer_bytes += len(example["answer"].encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    report = {
        "schema": "crysllmgen_sft_manifest_v1",
        "protocol": PROTOCOL_NAME,
        "representation": representation,
        "training_seed": int(training_seed),
        "epochs": epochs,
        "canonical_orbit_order": bool(canonical_orbit_order),
        "source_selected_structures": len(records),
        "examples": len(records) * epochs,
        "counts_by_epoch": counts_by_epoch,
        "answer_bytes": answer_bytes,
        "jsonl_bytes": destination.stat().st_size,
        "jsonl_sha256": digest.hexdigest(),
        "input_paths": [str(Path(value).resolve()) for value in input_paths],
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def tokenize_sft_example(
    tokenizer: Any,
    example: Mapping[str, Any],
    *,
    max_length: int = 512,
) -> dict[str, list[int]]:
    """Apply the official chat template and mask every non-answer label."""

    messages = [
        {"role": "system", "content": str(example["system_prompt"])},
        {"role": "user", "content": str(example["user_prompt"])},
    ]
    prompt_ids = list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = list(
        tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": str(example["answer"])}],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("official chat template does not preserve the generation prefix")
    if len(full_ids) > max_length:
        raise ValueError(
            f"SFT example exceeds frozen sequence length: {len(full_ids)} > {max_length}"
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not labels or all(value == -100 for value in labels):
        raise ValueError("SFT example has no assistant supervision")
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def _other_species(rng: random.Random, source: int) -> int:
    candidates = [value for value in MP20_ATOMIC_NUMBERS if value != int(source)]
    return candidates[rng.randrange(len(candidates))]


def _replacement_orbit(
    source: OrbitState,
    *,
    wyckoff_type: int,
    catalog: ChartCatalog,
    space_group: int,
    rng: random.Random,
) -> OrbitState:
    spec = catalog.get(space_group, int(wyckoff_type))
    return OrbitState(
        orbit_id=source.orbit_id,
        wyckoff_type=int(wyckoff_type),
        species=source.species,
        multiplicity=int(spec.multiplicity),
        primitive_multiplicity=int(spec.primitive_multiplicity),
        chart_dimension=int(spec.dimension),
        free_coordinate=tuple(rng.random() for _ in range(int(spec.dimension))),
    )


def _type_choices(
    state: StratifiedState,
    orbit_index: int,
    catalog: ChartCatalog,
) -> tuple[int, ...]:
    source = state.orbits[orbit_index]
    remaining = state.atom_count - int(source.primitive_multiplicity)
    return tuple(
        int(value)
        for value in catalog.types(state.space_group)
        if int(value) != source.wyckoff_type
        and 1
        <= remaining + int(catalog.get(state.space_group, int(value)).primitive_multiplicity)
        <= 20
    )


def _corrupt_for_direct_edit(
    state: StratifiedState,
    *,
    requested_operator: str,
    catalog: ChartCatalog,
    rng: random.Random,
) -> tuple[StratifiedState, TopologyEdit, str]:
    """Construct one supported corruption and its single-step inverse command."""

    operator = requested_operator
    orbits = list(state.orbits)
    if operator == "clean":
        return state, TopologyEdit("noop"), operator
    if operator == "deletion" and len(orbits) > 1:
        index = rng.randrange(len(orbits))
        deleted = orbits.pop(index)
        return (
            state.replace_orbits(orbits),
            TopologyEdit(
                "birth",
                wyckoff_type=deleted.wyckoff_type,
                species=deleted.species,
            ),
            operator,
        )
    if operator == "false_insertion":
        choices = [
            int(value)
            for value in catalog.types(state.space_group)
            if state.atom_count
            + int(catalog.get(state.space_group, int(value)).primitive_multiplicity)
            <= 20
        ]
        if choices:
            wyckoff = choices[rng.randrange(len(choices))]
            spec = catalog.get(state.space_group, wyckoff)
            false = OrbitState(
                orbit_id=f"false-{rng.getrandbits(64):016x}",
                wyckoff_type=wyckoff,
                species=MP20_ATOMIC_NUMBERS[rng.randrange(len(MP20_ATOMIC_NUMBERS))],
                multiplicity=int(spec.multiplicity),
                primitive_multiplicity=int(spec.primitive_multiplicity),
                chart_dimension=int(spec.dimension),
                free_coordinate=tuple(rng.random() for _ in range(int(spec.dimension))),
            )
            position = rng.randrange(len(orbits) + 1)
            orbits.insert(position, false)
            return (
                state.replace_orbits(orbits),
                TopologyEdit("death", orbit_index=position),
                operator,
            )
    if operator in {"wrong_wyckoff", "joint"}:
        candidates = [
            (index, _type_choices(state, index, catalog))
            for index in range(len(orbits))
        ]
        candidates = [(index, values) for index, values in candidates if values]
        if candidates:
            index, values = candidates[rng.randrange(len(candidates))]
            source = orbits[index]
            target = values[rng.randrange(len(values))]
            corrupted = _replacement_orbit(
                source,
                wyckoff_type=target,
                catalog=catalog,
                space_group=state.space_group,
                rng=rng,
            )
            if operator == "joint":
                corrupted = dataclasses.replace(
                    corrupted,
                    species=_other_species(rng, source.species),
                )
            orbits[index] = corrupted
            return (
                state.replace_orbits(orbits),
                TopologyEdit(
                    "type_change",
                    orbit_index=index,
                    wyckoff_type=source.wyckoff_type,
                ),
                operator,
            )
    # Every MP20 orbit has at least one alternative species.  This is a
    # registered support fallback for structurally impossible deletion/type/
    # insertion requests, not a failed-attempt retry.
    index = rng.randrange(len(orbits))
    source = orbits[index]
    orbits[index] = dataclasses.replace(
        source,
        species=_other_species(rng, source.species),
    )
    return (
        state.replace_orbits(orbits),
        TopologyEdit(
            "species_change",
            orbit_index=index,
            species=source.species,
        ),
        "wrong_species" if operator != "clean" else operator,
    )


def _quantize_evidence(values: Sequence[float]) -> str:
    result = []
    for value in values:
        numeric = min(1.0, max(0.0, float(value)))
        result.append(f"{min(15, int(math.floor(numeric * 16.0))):X}")
    return "".join(result)


def serialize_geometry_evidence(values: Sequence[Sequence[float]]) -> str:
    """Encode fixed-width six-nibble signals in orbit presentation order.

    Orbit indices are intentionally omitted: the proposal and evidence share
    the same presentation order, and every evidence row has exactly six hex
    nibbles. This preserves every signal while keeping 20-orbit edit examples
    inside the registered 512-token context.
    """

    return "".join(_quantize_evidence(row) for row in values)


def _evidence_text(
    state: StratifiedState,
    *,
    catalog: ChartCatalog,
    condition: str,
    rng: random.Random,
) -> str:
    if condition == "absent":
        return "-"
    expanded = expand_state(state, catalog, redetect_space_group=True)
    evidence = [list(value.as_tuple()) for value in compute_geometry_evidence(state, expanded)]
    if condition == "noisy":
        evidence = [
            [min(1.0, max(0.0, value + rng.gauss(0.0, 0.10))) for value in row]
            for row in evidence
        ]
    elif condition == "shuffled":
        rng.shuffle(evidence)
    elif condition != "clean":
        raise ValueError(f"unknown geometry condition: {condition}")
    return serialize_geometry_evidence(evidence)


def build_direct_edit_example(
    payload: Mapping[str, Any],
    *,
    ordinal: int,
    training_seed: int,
    catalog: ChartCatalog,
) -> dict[str, Any]:
    material_id = str(payload.get("material_id", ""))
    if not material_id:
        raise ValueError("material_id is required")
    primary = _primary(payload)
    clean = StratifiedState.from_dict(dict(primary["state"]))
    rng = random.Random(_derive_seed("direct_edit", training_seed, ordinal, material_id))
    clean = clean.permuted(rng)
    operators = (
        "clean",
        "deletion",
        "false_insertion",
        "wrong_wyckoff",
        "wrong_species",
        "joint",
    )
    # One in five direct-edit examples is the clean NOOP calibration negative.
    selector = rng.randrange(10)
    requested = "clean" if selector < 2 else operators[1 + (selector - 2) % 5]
    corrupted, edit, actual = _corrupt_for_direct_edit(
        clean,
        requested_operator=requested,
        catalog=catalog,
        rng=rng,
    )
    conditions = ("clean", "noisy", "shuffled", "absent")
    condition = conditions[rng.randrange(len(conditions))]
    proposal = serialize_wq_proposal(corrupted, catalog)
    evidence = _evidence_text(corrupted, catalog=catalog, condition=condition, rng=rng)
    answer = serialize_topology_edit(edit)
    user_prompt = f"P={proposal};G={evidence}"
    identity = {
        "protocol": PROTOCOL_NAME,
        "material_id": material_id,
        "training_seed": int(training_seed),
        "ordinal": int(ordinal),
        "representation": "wyckoff",
        "stage": "direct_edit",
        "requested_operator": requested,
        "actual_operator": actual,
        "geometry_condition": condition,
    }
    example_id = "edit-" + hashlib.sha256(
        _canonical_json(identity).encode("utf-8")
    ).hexdigest()[:24]
    return {
        "schema": "crysllmgen_sft_example_v1",
        "example_id": example_id,
        **identity,
        "system_prompt": WQ_EDIT_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "answer": answer,
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "source_topology_hash": clean.topology_hash(include_geometry=True),
        "input_topology_hash": corrupted.topology_hash(include_geometry=True),
        "primitive_atom_count": corrupted.atom_count,
        "orbit_count": len(corrupted.orbits),
    }


def write_mixed_wq_sft_jsonl(
    *,
    input_paths: Sequence[str | Path],
    output: str | Path,
    manifest: str | Path,
    training_seed: int,
    catalog: ChartCatalog,
) -> dict[str, Any]:
    destination = Path(output)
    manifest_path = Path(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or manifest_path.exists():
        raise FileExistsError("mixed WQ outputs are append-only/exclusive")
    records = list(iter_selected_records(input_paths))
    digest = hashlib.sha256()
    counts = {"coarse_proposal": 0, "direct_edit": 0}
    with destination.open("xb") as handle:
        for ordinal, payload in enumerate(records):
            examples = (
                {
                    **build_coarse_example(
                        payload,
                        representation="wyckoff",
                        epoch=0,
                        training_seed=training_seed,
                        catalog=catalog,
                    ),
                    "stage": "coarse_proposal",
                },
                build_direct_edit_example(
                    payload,
                    ordinal=ordinal,
                    training_seed=training_seed,
                    catalog=catalog,
                ),
            )
            for example in examples:
                encoded = (_canonical_json(example) + "\n").encode("utf-8")
                handle.write(encoded)
                digest.update(encoded)
                counts[str(example["stage"])] += 1
        handle.flush()
        os.fsync(handle.fileno())
    report = {
        "schema": "crysllmgen_sft_manifest_v1",
        "protocol": PROTOCOL_NAME,
        "representation": "wyckoff",
        "stage": "mixed_initial_and_direct_edit",
        "training_seed": int(training_seed),
        "epochs": 1,
        "canonical_orbit_order": False,
        "source_selected_structures": len(records),
        "examples": 2 * len(records),
        "stage_counts": counts,
        "initial_proposal_fraction": 0.5,
        "direct_edit_fraction": 0.5,
        "jsonl_bytes": destination.stat().st_size,
        "jsonl_sha256": digest.hexdigest(),
        "input_paths": [str(Path(value).resolve()) for value in input_paths],
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report
