#!/usr/bin/env python3
"""Build compact PMTR paired repair rows from certified train-only corruptions."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crystal_dlm.dynamic_crystal import parse_dynamic_answer  # noqa: E402
from crystal_dlm.manifold_corruption import (  # noqa: E402
    CorruptionCertificate,
    CorruptionConfig,
    CorruptionProposal,
    CorruptionSelection,
    CrystalGeometry,
    canonical_request_key,
    minimum_image_cartesian_retraction,
    relative_spd_tangent,
    select_first_certified_corruption,
)
from crystal_dlm.spad_program import (  # noqa: E402
    element_position,
    program_from_element_order,
)


_CLOSURE_SPEC = importlib.util.spec_from_file_location(
    "_pmtr_existing_spad_closure_builder",
    PROJECT_ROOT / "scripts" / "build_spad_basin_closure_sft_data.py",
)
if _CLOSURE_SPEC is None or _CLOSURE_SPEC.loader is None:
    raise RuntimeError("cannot load the existing SPAD closure-state builder")
_CLOSURE_MODULE = importlib.util.module_from_spec(_CLOSURE_SPEC)
_CLOSURE_SPEC.loader.exec_module(_CLOSURE_MODULE)
FALLBACK_SOURCE = _CLOSURE_MODULE.FALLBACK_SOURCE
closure_states = _CLOSURE_MODULE.closure_states
deterministic_state_index = _CLOSURE_MODULE.deterministic_state_index
pointer_programs = _CLOSURE_MODULE.pointer_programs


SCHEMA = "rollout_matched_transition_v1"
CLOSURE_SCHEMA = "pmtr_coherent_repair_v1"
MANIFEST_SCHEMA = "pmtr_preflight_manifest_v1"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield value


def request_key_for_source(source: Mapping[str, Any]) -> str:
    if source.get("request_key") is not None:
        return canonical_request_key(source["request_key"])
    split = str(source.get("source_split") or "preflight")
    return f"{split}:{int(source['source_row_idx'])}"


class CertificateLookup:
    """A small request/proposal keyed provider for JSONL or test fixtures."""

    def __init__(self, rows: Iterable[Mapping[str, Any]] = ()) -> None:
        self._values: dict[tuple[str, int], CorruptionCertificate] = {}
        for row in rows:
            self.add(row)

    def add(
        self,
        row: Mapping[str, Any],
        *,
        default_request_key: str | None = None,
    ) -> None:
        request_key = row.get("request_key", default_request_key)
        if request_key is None and row.get("source_row_idx") is not None:
            split = str(row.get("source_split") or "preflight")
            request_key = f"{split}:{int(row['source_row_idx'])}"
        if request_key is None:
            raise ValueError("certificate row lacks request_key or source_row_idx")
        key = (canonical_request_key(request_key), int(row["proposal_index"]))
        if key in self._values:
            raise ValueError(f"duplicate certificate for {key}")
        payload = row.get("certificate")
        certificate = payload if isinstance(payload, Mapping) else row
        self._values[key] = CorruptionCertificate.from_mapping(certificate)

    def get(self, proposal: CorruptionProposal) -> CorruptionCertificate | None:
        return self._values.get((proposal.request_key, int(proposal.proposal_index)))

    def with_embedded(
        self, source: Mapping[str, Any], request_key: str
    ) -> "CertificateLookup":
        combined = CertificateLookup()
        combined._values.update(self._values)
        for row in source.get("pmtr_certificates") or ():
            if not isinstance(row, Mapping):
                raise TypeError("pmtr_certificates entries must be JSON objects")
            combined.add(row, default_request_key=request_key)
        return combined


def _plan_components(plan: Mapping[str, Any]) -> tuple[int, list[str], list[int]]:
    num_atoms = int(plan.get("N") or 0)
    elements = [str(value) for value in plan.get("elements") or ()]
    counts = [int(value) for value in plan.get("counts") or ()]
    if not elements or len(elements) != len(counts):
        raise ValueError("Plan elements/counts are malformed")
    if len(elements) != len(set(elements)):
        raise ValueError("Plan elements must be unique")
    if any(count <= 0 for count in counts) or sum(counts) != num_atoms:
        raise ValueError("Plan counts disagree with exact N")
    return num_atoms, elements, counts


def _expanded_species(elements: Sequence[str], counts: Sequence[int]) -> list[str]:
    return [
        symbol
        for symbol, count in zip(elements, counts, strict=True)
        for _ in range(int(count))
    ]


def resolve_program(
    source: Mapping[str, Any],
    pointer: tuple[list[str], str] | None,
) -> tuple[list[str], str]:
    plan = source["plan_state"]
    _num_atoms, elements, _counts = _plan_components(plan)
    if pointer is not None:
        return list(pointer[0]), str(pointer[1])
    if source.get("species_program"):
        return (
            [str(value) for value in source["species_program"]],
            str(source.get("species_program_source") or "source_species_program"),
        )
    return elements, FALLBACK_SOURCE


def select_source_corruption(
    source: Mapping[str, Any],
    *,
    certificates: CertificateLookup,
    seed: int,
    config: CorruptionConfig,
) -> CorruptionSelection:
    request_key = request_key_for_source(source)
    provider = certificates.with_embedded(source, request_key)
    return select_first_certified_corruption(
        str(source["answer"]),
        request_key=request_key,
        certify=provider.get,
        seed=int(seed),
        config=config,
    )


def _validate_source(source: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    plan = source.get("plan_state")
    if not isinstance(plan, Mapping):
        raise ValueError("source lacks plan_state")
    num_atoms, elements, counts = _plan_components(plan)
    parsed = parse_dynamic_answer(str(source["answer"]), strict=True)
    if int(parsed["num_atoms"]) != num_atoms:
        raise ValueError("source answer N differs from Plan")
    if list(parsed["species"]) != _expanded_species(elements, counts):
        raise ValueError("source answer species slots differ from Plan")
    if len(parsed["tokens"]) != 7 + 4 * num_atoms:
        raise RuntimeError("source answer is not exact 7+4N")
    return dict(plan), num_atoms


def coherent_source_tokens(
    *,
    clean_tokens: Sequence[str],
    corrupted_tokens: Sequence[str],
    repair_order: Sequence[int],
    state_index: int,
    num_atoms: int,
) -> tuple[str, ...]:
    """Keep repaired positions clean and every unrepaired position coherent."""

    clean = tuple(str(value) for value in clean_tokens)
    corrupted = tuple(str(value) for value in corrupted_tokens)
    if len(clean) != 7 + 4 * int(num_atoms) or len(corrupted) != len(clean):
        raise ValueError("clean/corrupted bodies must share exact 7+4N length")
    if len(repair_order) != 6 + 3 * int(num_atoms):
        raise ValueError("repair order must cover exactly 6+3N geometry components")
    if len(set(int(value) for value in repair_order)) != len(repair_order):
        raise ValueError("repair order contains duplicate positions")
    index = int(state_index)
    if not 0 <= index < len(repair_order):
        raise ValueError("state_index lies outside repair order")

    output = list(corrupted)
    protected = {0, *(element_position(slot) for slot in range(int(num_atoms)))}
    repaired = {int(value) for value in repair_order[:index]}
    for position in protected | repaired:
        output[position] = clean[position]
    return tuple(output)


def repair_target_for_state(
    *,
    clean_answer: str,
    source_answer: str,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute the repair target from the exact hybrid state seen at inference."""

    clean = CrystalGeometry.from_mapping(
        parse_dynamic_answer(clean_answer, strict=True)
    )
    current = CrystalGeometry.from_mapping(
        parse_dynamic_answer(source_answer, strict=True)
    )
    if tuple(current.species) != tuple(clean.species):
        raise RuntimeError("repair target source changed species/site order")
    if state["kind"] == "cell_sequential_component":
        tangent = relative_spd_tangent(current.metric, clean.metric)
        return {
            "kind": "cell",
            "lattice_tangent": tangent.tolist(),
            "site_slot_index": None,
            "cartesian_site_delta_A": None,
        }

    if state["kind"] != "reverse_species_block_component":
        raise ValueError(f"unsupported closure state kind {state['kind']!r}")
    slot = int(state["metadata"]["site_slot_index"])
    if not 0 <= slot < len(clean.species):
        raise RuntimeError("active repair site lies outside exact N")
    vectors = minimum_image_cartesian_retraction(
        clean.frac_coords,
        current.frac_coords,
        current.lattice,
        image_radius=2,
    )
    return {
        "kind": "site",
        "lattice_tangent": None,
        "site_slot_index": slot,
        "cartesian_site_delta_A": vectors[slot].tolist(),
    }


def build_repair_row(
    source: Mapping[str, Any],
    *,
    source_idx: int,
    program_order: Sequence[str],
    program_source: str,
    selection: CorruptionSelection,
    seed: int,
    state_index: int | None = None,
) -> dict[str, Any]:
    """Build one inference-matched state from one selected coherent corruption."""

    plan, num_atoms = _validate_source(source)
    program = program_from_element_order(
        plan, program_order, order_source=str(program_source)
    )
    states = closure_states(program)
    split = str(source.get("source_split") or "preflight")
    selected_index = (
        deterministic_state_index(
            source_split=split,
            source_idx=int(source_idx),
            seed=int(seed),
            state_count=len(states),
        )
        if state_index is None
        else int(state_index)
    )
    if not 0 <= selected_index < len(states):
        raise ValueError("state_index lies outside closure states")
    state = states[selected_index]
    repair_order = [int(item["loss"][0]) for item in states]
    if any(len(item["loss"]) != 1 for item in states):
        raise RuntimeError("PMTR requires singleton component closure states")

    clean = tuple(parse_dynamic_answer(selection.clean_body, strict=True)["tokens"])
    corrupted_body = selection.source_body
    corrupted = tuple(parse_dynamic_answer(corrupted_body, strict=True)["tokens"])
    source_tokens = coherent_source_tokens(
        clean_tokens=clean,
        corrupted_tokens=corrupted,
        repair_order=repair_order,
        state_index=selected_index,
        num_atoms=num_atoms,
    )
    source_answer = "".join(source_tokens)
    parsed_source = parse_dynamic_answer(source_answer, strict=True)
    if int(parsed_source["num_atoms"]) != num_atoms or list(parsed_source["species"]) != list(
        parse_dynamic_answer(selection.clean_body, strict=True)["species"]
    ):
        raise RuntimeError("coherent source changed exact composition")

    forced = [int(value) for value in state["forced"]]
    loss = [int(value) for value in state["loss"]]
    protected = {0, *(element_position(slot) for slot in range(num_atoms))}
    if state["kind"] == "cell_sequential_component" and forced != [
        int(value) for value in state["metadata"]["remaining_lattice_positions"]
    ]:
        raise RuntimeError("cell forced mask diverged from closure state")
    if not set(loss) <= set(forced) or protected & set(forced):
        raise RuntimeError("closure masks violate protected exact-composition positions")

    proposal = selection.proposal
    certificate = selection.certificate
    mode = "clean_ce_fallback" if selection.fallback else "manifold_repair"
    repair_target = (
        None
        if selection.fallback
        else repair_target_for_state(
            clean_answer=selection.clean_body,
            source_answer=source_answer,
            state=state,
        )
    )
    closure = dict(state["metadata"])
    closure.update(
        {
            "schedule": "cell_then_reverse_llama_species_blocks_v1",
            "state_index": selected_index,
            "state_count": len(states),
            "program_order": list(program.element_order),
            "earlier_repaired_positions": repair_order[:selected_index],
            "active_position": repair_order[selected_index],
            "same_corruption_for_all_unrepaired_positions": True,
        }
    )
    return {
        "schema": SCHEMA,
        "closure_schema": CLOSURE_SCHEMA,
        "source_row_idx": int(source_idx),
        "source_split": split,
        "prompt": str(source["prompt"]),
        "answer": selection.clean_body,
        "source_answer": source_answer,
        "plan_state": plan,
        "num_atoms": num_atoms,
        "sample_weight": 1.0,
        "loss_profile": "fixed_slot",
        "mask_policy": "normal",
        "spad_mask_class": str(state["kind"]),
        "species_program": list(program.element_order),
        "species_program_source": program.order_source,
        "forced_mask_positions": forced,
        "loss_positions": loss,
        "closure": closure,
        "repair_target": repair_target,
        "pmtr": {
            "mode": mode,
            "request_key": request_key_for_source(source),
            "proposal_index": None if proposal is None else int(proposal.proposal_index),
            "attempted_proposals": int(selection.attempted_proposals),
            "lattice_changed_components": 0
            if proposal is None
            else len(proposal.lattice_changed_positions),
            "coordinate_changed_components": 0
            if proposal is None
            else len(proposal.coordinate_changed_positions),
            "certificate": None if certificate is None else certificate.to_dict(),
        },
        "prospective_outcomes_read": False,
    }


def build_file(
    *,
    input_path: Path,
    output_path: Path,
    certificates: CertificateLookup,
    pointer_path: Path | None,
    seed: int,
    config: CorruptionConfig,
) -> dict[str, Any]:
    pointers = {} if pointer_path is None else pointer_programs(pointer_path)
    seen: set[int] = set()
    modes: Counter[str] = Counter()
    mask_classes: Counter[str] = Counter()
    component_roles: Counter[str] = Counter()
    proposal_indices: Counter[str] = Counter()
    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        for source in iter_jsonl(input_path):
            source_idx = int(source["source_row_idx"])
            if source_idx in seen:
                raise ValueError(f"input duplicates source_row_idx {source_idx}")
            seen.add(source_idx)
            order, order_source = resolve_program(source, pointers.get(source_idx))
            selection = select_source_corruption(
                source,
                certificates=certificates,
                seed=int(seed),
                config=config,
            )
            row = build_repair_row(
                source,
                source_idx=source_idx,
                program_order=order,
                program_source=order_source,
                selection=selection,
                seed=int(seed),
            )
            modes[row["pmtr"]["mode"]] += 1
            mask_classes[row["spad_mask_class"]] += 1
            closure = row["closure"]
            role = str(closure.get("cell_component") or closure.get("coordinate_component"))
            component_roles[role] += 1
            proposal_indices[str(row["pmtr"]["proposal_index"])] += 1
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    unexpected = set(pointers) - seen
    if unexpected:
        raise ValueError(f"pointer rows absent from input: {sorted(unexpected)[:5]}")
    return {
        "sources": len(seen),
        "rows": len(seen),
        "modes": dict(sorted(modes.items())),
        "mask_classes": dict(sorted(mask_classes.items())),
        "component_roles": dict(sorted(component_roles.items())),
        "selected_proposal_indices": dict(sorted(proposal_indices.items())),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--certificates-jsonl", type=Path)
    parser.add_argument("--pointer-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--max-proposals", type=int, default=4)
    parser.add_argument("--max-delta-energy", type=float, default=2.0)
    parser.add_argument("--lattice-log-std", type=float, default=0.055)
    parser.add_argument("--coordinate-std-A", type=float, default=0.18)
    args = parser.parse_args(argv)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    certificate_rows = (
        () if args.certificates_jsonl is None else iter_jsonl(args.certificates_jsonl)
    )
    certificates = CertificateLookup(certificate_rows)
    config = CorruptionConfig(
        max_proposals=int(args.max_proposals),
        max_delta_energy=float(args.max_delta_energy),
        lattice_log_std=float(args.lattice_log_std),
        coordinate_cartesian_std_A=float(args.coordinate_std_A),
    )
    summary = build_file(
        input_path=args.input_jsonl,
        output_path=args.output_dir / "data.jsonl",
        certificates=certificates,
        pointer_path=args.pointer_jsonl,
        seed=int(args.seed),
        config=config,
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "seed": int(args.seed),
        "corruption_space": "SPD(3)xT^(3N)",
        "codec": "dynamic_exact_7_plus_4N",
        "selection": "first_certified_at_most_4_else_clean_ce",
        "schedule": "cell_then_reverse_llama_species_blocks_v1",
        "certificate_provider": "jsonl_or_embedded_mock",
        "config": {
            "max_proposals": int(config.max_proposals),
            "max_delta_energy": float(config.max_delta_energy),
            "lattice_log_std": float(config.lattice_log_std),
            "coordinate_cartesian_std_A": float(config.coordinate_cartesian_std_A),
        },
        **summary,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
