#!/usr/bin/env python3
"""Label retained potential-closure transactions with raw CHGNet E/F/stress."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from crystal_dlm.dynamic_crystal import arrays_to_structure, parse_dynamic_answer


SCHEMA = "potential_closure_candidate_group_v1"
OUTPUT_SCHEMA = "potential_closure_labelled_group_v1"
MIN_SPREAD_EV_PER_ATOM = 1.0e-3
EXPECTED_STRATA = (
    "mp20_clean_cell",
    "mp20_clean_site",
    "on_policy_cell",
    "on_policy_site",
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


def predict_batches(
    model: Any,
    structures: Sequence[Any],
    *,
    batch_size: int,
) -> list[dict[str, Any] | None]:
    output: list[dict[str, Any] | None] = []
    for start in range(0, len(structures), int(batch_size)):
        chunk = list(structures[start : start + int(batch_size)])
        try:
            values = model.predict_structure(
                chunk,
                task="efsm",
                batch_size=int(batch_size),
            )
            if isinstance(values, dict):
                values = [values]
            output.extend(values)
        except Exception:
            for structure in chunk:
                try:
                    output.append(model.predict_structure(structure, task="efsm"))
                except Exception:
                    output.append(None)
    if len(output) != len(structures):
        raise RuntimeError("CHGNet prediction count changed")
    return output


def finite_prediction(value: Mapping[str, Any] | None) -> dict[str, float] | None:
    if value is None:
        return None
    try:
        energy = float(np.asarray(value["e"], dtype=float).reshape(()))
        forces = np.asarray(value["f"], dtype=float)
        stress = np.asarray(value["s"], dtype=float)
        if forces.ndim != 2 or forces.shape[1] != 3 or stress.shape != (3, 3):
            return None
        if not (
            math.isfinite(energy)
            and np.isfinite(forces).all()
            and np.isfinite(stress).all()
        ):
            return None
        norms = np.linalg.norm(forces, axis=1)
        return {
            "raw_chgnet_energy_eV_per_atom": energy,
            "raw_force_rms_eV_per_A": float(np.sqrt(np.mean(norms * norms))),
            "raw_force_max_eV_per_A": float(np.max(norms)),
            "raw_stress_frobenius_GPa": float(np.linalg.norm(stress)),
            "raw_hydrostatic_stress_GPa": float(np.trace(stress) / 3.0),
        }
    except Exception:
        return None


def validate_groups(groups: Sequence[Mapping[str, Any]]) -> None:
    if len(groups) != 2048:
        raise ValueError("potential closure requires exactly 2048 groups")
    if [int(group["group_idx"]) for group in groups] != list(range(2048)):
        raise ValueError("group indices must be contiguous and ordered")
    counts = Counter(str(group.get("stratum")) for group in groups)
    if counts != Counter({name: 512 for name in EXPECTED_STRATA}):
        raise ValueError("potential closure stratum counts changed")
    for group in groups:
        if group.get("schema") != SCHEMA:
            raise ValueError("candidate group schema changed")
        transaction_length = int(group.get("transaction_length", 0))
        if transaction_length not in (3, 6):
            raise ValueError("transaction length must be three or six")
        candidates = group.get("candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 4:
            raise ValueError("candidate count must lie in one through four")
        if candidates[0].get("candidate_source") != "noop":
            raise ValueError("candidate zero must be the no-op")
        actions: list[tuple[str, ...]] = []
        for index, candidate in enumerate(candidates):
            if int(candidate.get("candidate_idx", -1)) != index:
                raise ValueError("candidate indices are not contiguous")
            if candidate.get("valid_action") is not True:
                raise ValueError("retained candidates must be legal")
            tokens = candidate.get("action_tokens")
            if not isinstance(tokens, list) or len(tokens) != transaction_length:
                raise ValueError("candidate action token length changed")
            actions.append(tuple(str(value) for value in tokens))
        if len(actions) != len(set(actions)):
            raise ValueError("candidate group contains unmerged duplicate actions")


def attach_labels(
    groups: list[dict[str, Any]],
    predictions: Sequence[Mapping[str, Any] | None],
) -> list[dict[str, Any]]:
    expected = sum(len(group["candidates"]) for group in groups)
    if len(predictions) != expected:
        raise ValueError("one CHGNet prediction is required per retained candidate")
    cursor = 0
    labelled: list[dict[str, Any]] = []
    for group in groups:
        candidates: list[dict[str, Any]] = []
        for candidate in group["candidates"]:
            prediction = finite_prediction(predictions[cursor])
            cursor += 1
            candidates.append(
                {
                    **candidate,
                    "raw_chgnet_known": prediction is not None,
                    **(prediction or {}),
                }
            )
        known = [
            float(candidate["raw_chgnet_energy_eV_per_atom"])
            for candidate in candidates
            if candidate["raw_chgnet_known"]
        ]
        spread = None if len(known) < 2 else float(max(known) - min(known))
        informative = bool(
            len(known) >= 2
            and candidates[0]["raw_chgnet_known"]
            and spread is not None
            and spread >= MIN_SPREAD_EV_PER_ATOM
        )
        labelled.append(
            {
                **group,
                "schema": OUTPUT_SCHEMA,
                "K": int(len(candidates)),
                "candidates": candidates,
                "legal_energy_known_actions": int(len(known)),
                "energy_spread_eV_per_atom": spread,
                "informative": informative,
                "raw_energy_only": True,
                "same_composition_within_group": True,
                "model494_or_official_outcomes_read": False,
            }
        )
    return labelled


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "median": None, "q10": None, "q90": None}
    return {
        "count": int(len(finite)),
        "median": float(statistics.median(finite)),
        "q10": float(np.quantile(finite, 0.1)),
        "q90": float(np.quantile(finite, 0.9)),
    }


def summarize(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [candidate for group in groups for candidate in group["candidates"]]
    known = [candidate for candidate in candidates if candidate["raw_chgnet_known"]]
    informative_by_stratum = Counter(
        str(group["stratum"]) for group in groups if group["informative"]
    )
    k_histogram = Counter(str(group["K"]) for group in groups)
    coverage = len(known) / len(candidates) if candidates else 0.0
    gates = {
        "groups_2048": len(groups) == 2048,
        "informative_total_ge_1024": sum(informative_by_stratum.values()) >= 1024,
        "each_stratum_informative_ge_256": all(
            informative_by_stratum[name] >= 256 for name in EXPECTED_STRATA
        ),
        "energy_coverage_ge_0p98": coverage >= 0.98,
        "noop_energy_known_all_groups": all(
            group["candidates"][0]["raw_chgnet_known"] for group in groups
        ),
        "candidate_actions_unique": all(
            len(
                {
                    tuple(str(value) for value in candidate["action_tokens"])
                    for candidate in group["candidates"]
                }
            )
            == len(group["candidates"])
            for group in groups
        ),
    }
    return {
        "schema": "potential_closure_label_manifest_v1",
        "groups": int(len(groups)),
        "candidates": int(len(candidates)),
        "K_histogram": dict(sorted(k_histogram.items())),
        "raw_energy_known": int(len(known)),
        "raw_energy_coverage": float(coverage),
        "informative_groups": int(sum(informative_by_stratum.values())),
        "informative_by_stratum": {
            name: int(informative_by_stratum[name]) for name in EXPECTED_STRATA
        },
        "energy_spread_eV_per_atom": describe(
            float(group["energy_spread_eV_per_atom"])
            for group in groups
            if group.get("energy_spread_eV_per_atom") is not None
        ),
        "force_rms_eV_per_A": describe(
            float(candidate["raw_force_rms_eV_per_A"]) for candidate in known
        ),
        "stress_frobenius_GPa": describe(
            float(candidate["raw_stress_frobenius_GPa"]) for candidate in known
        ),
        "gates": gates,
        "formal_action_pool_gate": bool(all(gates.values())),
        "raw_energy_only": True,
        "model494_or_official_outcomes_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-groups", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if int(args.batch_size) <= 0:
        raise ValueError("batch size must be positive")
    groups = list(iter_jsonl(args.candidate_groups.resolve()))
    validate_groups(groups)
    structures = [
        arrays_to_structure(parse_dynamic_answer(str(candidate["answer"]), strict=True))
        for group in groups
        for candidate in group["candidates"]
    ]
    from chgnet.model.model import CHGNet

    model = CHGNet.load(
        use_device=str(args.device),
        check_cuda_mem=False,
        verbose=False,
    )
    predictions = predict_batches(model, structures, batch_size=int(args.batch_size))
    labelled = attach_labels(groups, predictions)
    report = summarize(labelled)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "labelled_groups.jsonl").open("x", encoding="utf-8") as handle:
        for group in labelled:
            handle.write(json.dumps(group, sort_keys=True) + "\n")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / ("_SUCCESS" if report["formal_action_pool_gate"] else "_BLOCKED")).touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

