#!/usr/bin/env python3
"""Freeze the immutable 36+12+16 fixed-topology composition mechanism panel.

The source generation rows contain the original WQ ``proposal_state`` before
parent diffusion.  The taxonomy is used only to identify the three eligibility
groups.  Selection of the 16 valid controls uses proposal-state covariates only;
CHGNet, hull, relaxation, novelty, and post-parent geometry fields are never
read by the matching algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from crystal_dlm.wqcodiff.state import StratifiedState  # noqa: E402


EXPECTED_GROUP_COUNTS = {
    "no_neutral": 36,
    "pauling_only": 12,
    "valid_control": 16,
}
TAXONOMY_TO_PANEL_GROUP = {
    "no_charge_neutral_assignment": "no_neutral",
    "pauling_rejection": "pauling_only",
}
VALID_TAXONOMY_MECHANISM = "valid"
NUMERIC_MATCH_FEATURES = (
    "number_elements",
    "primitive_atom_count",
    "conventional_atom_count",
    "orbit_count",
    "continuous_dimension",
    "lattice_chart_dimension",
    "zero_dim_orbit_count",
    "max_primitive_multiplicity",
    "mean_primitive_multiplicity",
)
POST_OUTCOME_FIELDS_EXCLUDED = (
    "category",
    "density",
    "e_above_hull",
    "energy_per_atom",
    "min_distance",
    "struct_valid",
    "valid",
    "volume_per_atom",
)


class PanelFreezeError(ValueError):
    """Raised before any output when the frozen source contract is violated."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PanelFreezeError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise PanelFreezeError(
                    f"{path}:{line_number}: row must be a JSON object"
                )
            rows.append(row)
    return rows


def _read_taxonomy(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("attempts"), list):
        raise PanelFreezeError("taxonomy must contain an attempts list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload["attempts"], start=1):
        if not isinstance(row, dict):
            raise PanelFreezeError(f"taxonomy attempt {index} must be an object")
        rows.append(dict(row))
    return rows


def _crystal_system(space_group: int) -> str:
    bounds = (
        (2, "triclinic"),
        (15, "monoclinic"),
        (74, "orthorhombic"),
        (142, "tetragonal"),
        (167, "trigonal"),
        (194, "hexagonal"),
        (230, "cubic"),
    )
    for upper, name in bounds:
        if space_group <= upper:
            return name
    raise PanelFreezeError(f"space group outside [1,230]: {space_group}")


def proposal_features(state: StratifiedState) -> dict[str, Any]:
    """Return frozen pre-parent, pre-evaluator matching covariates."""

    primitive = [int(orbit.primitive_multiplicity) for orbit in state.orbits]
    return {
        "crystal_system": _crystal_system(state.space_group),
        "number_elements": len({int(orbit.species) for orbit in state.orbits}),
        "primitive_atom_count": state.atom_count,
        "conventional_atom_count": state.conventional_atom_count,
        "orbit_count": len(state.orbits),
        "continuous_dimension": state.continuous_dimension,
        "lattice_chart_dimension": len(state.lattice_chart),
        "zero_dim_orbit_count": sum(
            int(orbit.chart_dimension == 0) for orbit in state.orbits
        ),
        "max_primitive_multiplicity": max(primitive),
        "mean_primitive_multiplicity": sum(primitive) / len(primitive),
    }


def _feature_ranges(feature_rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    materialized = list(feature_rows)
    ranges: dict[str, float] = {}
    for key in NUMERIC_MATCH_FEATURES:
        values = [float(row[key]) for row in materialized]
        ranges[key] = max(1.0, max(values) - min(values))
    return ranges


def proposal_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    feature_ranges: Mapping[str, float],
) -> float:
    """Manhattan distance with full-panel pre-outcome range normalization."""

    distance = float(left["crystal_system"] != right["crystal_system"])
    for key in NUMERIC_MATCH_FEATURES:
        distance += abs(float(left[key]) - float(right[key])) / float(
            feature_ranges[key]
        )
    if not math.isfinite(distance):
        raise PanelFreezeError("non-finite proposal matching distance")
    return distance


def select_matched_controls(
    *,
    targets: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    all_features: Sequence[Mapping[str, Any]],
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedy facility-location coverage of no-neutral proposal covariates."""

    if count <= 0 or len(candidates) < count or not targets:
        raise PanelFreezeError("invalid matched-control selection dimensions")
    ranges = _feature_ranges(all_features)
    target_rows = sorted(
        (dict(row) for row in targets),
        key=lambda row: str(row["attempt_id"]),
    )
    candidate_rows = sorted(
        (dict(row) for row in candidates),
        key=lambda row: str(row["attempt_id"]),
    )
    distance_table = {
        (str(target["attempt_id"]), str(candidate["attempt_id"])): proposal_distance(
            target["features"],
            candidate["features"],
            feature_ranges=ranges,
        )
        for target in target_rows
        for candidate in candidate_rows
    }

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    best_by_target: dict[str, float] = {}
    steps: list[dict[str, Any]] = []
    for step in range(count):
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for candidate in candidate_rows:
            candidate_id = str(candidate["attempt_id"])
            if candidate_id in selected_ids:
                continue
            total = 0.0
            for target in target_rows:
                target_id = str(target["attempt_id"])
                candidate_distance = distance_table[(target_id, candidate_id)]
                total += min(
                    best_by_target.get(target_id, math.inf),
                    candidate_distance,
                )
            scored.append((total, candidate_id, candidate))
        if not scored:
            raise PanelFreezeError("matched-control candidate set exhausted")
        objective, candidate_id, chosen = min(
            scored,
            key=lambda item: (round(item[0], 15), item[1]),
        )
        selected.append(chosen)
        selected_ids.add(candidate_id)
        for target in target_rows:
            target_id = str(target["attempt_id"])
            best_by_target[target_id] = min(
                best_by_target.get(target_id, math.inf),
                distance_table[(target_id, candidate_id)],
            )
        steps.append(
            {
                "step": step + 1,
                "selected_attempt_id": candidate_id,
                "coverage_objective_after_selection": objective,
            }
        )

    selected_by_id = {str(row["attempt_id"]): row for row in selected}
    coverage: dict[str, list[str]] = {attempt_id: [] for attempt_id in selected_by_id}
    target_distances: list[dict[str, Any]] = []
    for target in target_rows:
        target_id = str(target["attempt_id"])
        distance, control_id = min(
            (
                distance_table[(target_id, control_id)],
                control_id,
            )
            for control_id in selected_by_id
        )
        coverage[control_id].append(target_id)
        target_distances.append(
            {
                "no_neutral_attempt_id": target_id,
                "matched_control_attempt_id": control_id,
                "distance": distance,
            }
        )

    evidence = {
        "algorithm": (
            "deterministic_greedy_facility_location_over_no_neutral_"
            "proposal_covariates_v1"
        ),
        "target_group": "no_neutral",
        "candidate_group": "taxonomy_valid",
        "count": count,
        "numeric_features": list(NUMERIC_MATCH_FEATURES),
        "categorical_features": ["crystal_system"],
        "numeric_feature_ranges": ranges,
        "categorical_mismatch_penalty": 1.0,
        "selection_steps": steps,
        "selected_control_coverage": {
            key: sorted(value) for key, value in sorted(coverage.items())
        },
        "target_matches": target_distances,
        "post_outcome_fields_excluded": list(POST_OUTCOME_FIELDS_EXCLUDED),
    }
    return selected, evidence


def freeze_panel(
    *,
    generation_rows: Sequence[Mapping[str, Any]],
    taxonomy_rows: Sequence[Mapping[str, Any]],
    expected_attempts: int = 256,
    expected_start_ordinal: int = 256,
    expected_group_counts: Mapping[str, int] = EXPECTED_GROUP_COUNTS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(generation_rows) != expected_attempts:
        raise PanelFreezeError("source generation denominator changed")
    if len(taxonomy_rows) != expected_attempts:
        raise PanelFreezeError("taxonomy denominator changed")
    expected_ordinals = list(
        range(expected_start_ordinal, expected_start_ordinal + expected_attempts)
    )
    generation_ordinals = [int(row.get("ordinal", -1)) for row in generation_rows]
    taxonomy_ordinals = [int(row.get("ordinal", -1)) for row in taxonomy_rows]
    if generation_ordinals != expected_ordinals:
        raise PanelFreezeError("source generation ordinal contract changed")
    if taxonomy_ordinals != expected_ordinals:
        raise PanelFreezeError("taxonomy ordinal contract changed")

    generation_by_id: dict[str, dict[str, Any]] = {}
    prepared: list[dict[str, Any]] = []
    for row in generation_rows:
        attempt_id = str(row.get("attempt_id", ""))
        if not attempt_id or attempt_id in generation_by_id:
            raise PanelFreezeError("source attempt IDs are missing or duplicated")
        if row.get("schema") != "wq_parent_csp_probe_attempt_v1":
            raise PanelFreezeError("unexpected source generation schema")
        if row.get("status") != "succeeded":
            raise PanelFreezeError("mechanism panel requires successful WQ proposals")
        if row.get("retry_or_replacement_used") is not False:
            raise PanelFreezeError("retry/replacement source cannot enter panel")
        payload = row.get("proposal_state")
        if not isinstance(payload, Mapping):
            raise PanelFreezeError(f"{attempt_id}: proposal_state missing")
        state = StratifiedState.from_dict(dict(payload))
        if state.attempt_id != attempt_id:
            raise PanelFreezeError(f"{attempt_id}: proposal state identity mismatch")
        canonical_state = state.to_dict(canonical_storage=True)
        generation_by_id[attempt_id] = dict(row)
        prepared.append(
            {
                "attempt_id": attempt_id,
                "ordinal": int(row["ordinal"]),
                "state": canonical_state,
                "features": proposal_features(state),
                "source_generation_row_sha256": _sha256_bytes(_canonical(row)),
                "proposal_state_sha256": _sha256_bytes(_canonical(canonical_state)),
            }
        )

    taxonomy_by_id: dict[str, dict[str, Any]] = {}
    for row in taxonomy_rows:
        attempt_id = str(row.get("attempt_id", ""))
        if not attempt_id or attempt_id in taxonomy_by_id:
            raise PanelFreezeError("taxonomy attempt IDs are missing or duplicated")
        taxonomy_by_id[attempt_id] = dict(row)
    if set(generation_by_id) != set(taxonomy_by_id):
        raise PanelFreezeError("source generation and taxonomy IDs differ")
    for row in prepared:
        taxonomy = taxonomy_by_id[row["attempt_id"]]
        if int(taxonomy["ordinal"]) != row["ordinal"]:
            raise PanelFreezeError("source generation and taxonomy ordinals differ")
        row["validity_mechanism"] = str(taxonomy.get("validity_mechanism", ""))

    no_neutral = [
        row
        for row in prepared
        if row["validity_mechanism"] == "no_charge_neutral_assignment"
    ]
    pauling = [
        row
        for row in prepared
        if row["validity_mechanism"] == "pauling_rejection"
    ]
    valid_candidates = [
        row
        for row in prepared
        if row["validity_mechanism"] == VALID_TAXONOMY_MECHANISM
    ]
    mechanism_counts = Counter(
        str(row["validity_mechanism"]) for row in prepared
    )
    ignored_mechanism_counts = {
        mechanism: count
        for mechanism, count in sorted(mechanism_counts.items())
        if mechanism
        not in {
            "no_charge_neutral_assignment",
            "pauling_rejection",
            VALID_TAXONOMY_MECHANISM,
        }
    }
    expected = {str(key): int(value) for key, value in expected_group_counts.items()}
    if len(no_neutral) != expected["no_neutral"]:
        raise PanelFreezeError("no-neutral count differs from frozen contract")
    if len(pauling) != expected["pauling_only"]:
        raise PanelFreezeError("Pauling-only count differs from frozen contract")
    controls, matching = select_matched_controls(
        targets=no_neutral,
        candidates=valid_candidates,
        all_features=[row["features"] for row in prepared],
        count=expected["valid_control"],
    )
    rows_by_group = (
        ("no_neutral", no_neutral),
        ("pauling_only", pauling),
        ("valid_control", controls),
    )
    panel: list[dict[str, Any]] = []
    for panel_group, rows in rows_by_group:
        for row in sorted(rows, key=lambda value: int(value["ordinal"])):
            panel.append(
                {
                    "schema": "wqcodiff_composition_mechanism_panel_input_row_v1",
                    "attempt_id": row["attempt_id"],
                    "ordinal": row["ordinal"],
                    "panel_group": panel_group,
                    "state": row["state"],
                    "proposal_features": row["features"],
                    "proposal_state_sha256": row["proposal_state_sha256"],
                    "source_generation_row_sha256": row[
                        "source_generation_row_sha256"
                    ],
                }
            )

    observed = Counter(row["panel_group"] for row in panel)
    if dict(observed) != expected or len({row["attempt_id"] for row in panel}) != len(
        panel
    ):
        raise PanelFreezeError("frozen panel group or identity contract failed")
    evidence = {
        "schema": "wqcodiff_composition_mechanism_panel_freeze_evidence_v1",
        "ok": True,
        "source_attempts": expected_attempts,
        "source_start_ordinal": expected_start_ordinal,
        "source_end_ordinal_inclusive": expected_ordinals[-1],
        "taxonomy_mechanism_counts": dict(sorted(mechanism_counts.items())),
        "ignored_taxonomy_mechanism_counts": ignored_mechanism_counts,
        "selection_pool_counts": {
            "no_neutral": len(no_neutral),
            "pauling_only": len(pauling),
            "valid_candidates": len(valid_candidates),
            "ignored_other": sum(ignored_mechanism_counts.values()),
        },
        "expected_group_counts": expected,
        "observed_group_counts": dict(observed),
        "matching": matching,
        "attempt_ids": {
            group: [row["attempt_id"] for row in panel if row["panel_group"] == group]
            for group in expected
        },
        "ordinals": {
            group: [row["ordinal"] for row in panel if row["panel_group"] == group]
            for group in expected
        },
        "scientific_generation_attempts_created": 0,
        "mlip_calls": 0,
        "llm_calls": 0,
        "parent_diffusion_calls": 0,
        "external_api_calls": 0,
        "retry_or_replacement_used": False,
    }
    return panel, evidence


def _write_exclusive_pair(
    *,
    panel_path: Path,
    manifest_path: Path,
    panel: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    descriptors: list[int] = []
    try:
        for path in (panel_path, manifest_path):
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            descriptors.append(descriptor)
            created.append(path)
        with os.fdopen(descriptors.pop(0), "w", encoding="utf-8") as handle:
            for row in panel:
                handle.write(_canonical(row).decode("utf-8") + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        panel_sha = _sha256_file(panel_path)
        complete_manifest = dict(manifest)
        complete_manifest["panel"] = {
            "path": str(panel_path),
            "sha256": panel_sha,
            "bytes": panel_path.stat().st_size,
            "lines": len(panel),
        }
        with os.fdopen(descriptors.pop(0), "w", encoding="utf-8") as handle:
            json.dump(
                complete_manifest,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-generation-jsonl", type=Path, required=True)
    parser.add_argument("--taxonomy-json", type=Path, required=True)
    parser.add_argument("--output-panel-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--expected-attempts", type=int, default=256)
    parser.add_argument("--expected-start-ordinal", type=int, default=256)
    args = parser.parse_args(argv)

    source = args.source_generation_jsonl.resolve()
    taxonomy = args.taxonomy_json.resolve()
    panel_path = args.output_panel_jsonl.resolve()
    manifest_path = args.output_manifest_json.resolve()
    if panel_path == manifest_path:
        raise PanelFreezeError("panel and manifest paths must differ")
    panel, evidence = freeze_panel(
        generation_rows=_read_jsonl(source),
        taxonomy_rows=_read_taxonomy(taxonomy),
        expected_attempts=args.expected_attempts,
        expected_start_ordinal=args.expected_start_ordinal,
    )
    evidence["sources"] = {
        "generation_jsonl": {
            "path": str(source),
            "sha256": _sha256_file(source),
            "bytes": source.stat().st_size,
            "lines": args.expected_attempts,
        },
        "taxonomy_json": {
            "path": str(taxonomy),
            "sha256": _sha256_file(taxonomy),
            "bytes": taxonomy.stat().st_size,
            "attempts": args.expected_attempts,
        },
    }
    _write_exclusive_pair(
        panel_path=panel_path,
        manifest_path=manifest_path,
        panel=panel,
        manifest=evidence,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "panel": str(panel_path),
                "panel_sha256": _sha256_file(panel_path),
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "group_counts": evidence["observed_group_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
