#!/usr/bin/env python3
"""Run the frozen proposal-only fixed-topology composition mechanism panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crystal_dlm.wqcodiff.composition_projection import (  # noqa: E402
    FixedTopologyCompositionProjector,
)
from crystal_dlm.wqcodiff.state import StratifiedState  # noqa: E402


EXPECTED_GROUP_COUNTS = {
    "no_neutral": 36,
    "pauling_only": 12,
    "valid_control": 16,
}
VALID_CONTROL_REASONS = {
    "charge_neutral_pauling_valid",
    "all_metal_shortcut",
    "single_element_shortcut",
}


class ProjectionPanelError(ValueError):
    """Raised before any output when a panel identity is malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProjectionPanelError(
                    f"invalid JSON on input line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ProjectionPanelError(
                    f"input line {line_number} must be a JSON object"
                )
            row["_input_line_number"] = line_number
            rows.append(row)
    return rows


def run_projection_panel(
    rows: Iterable[Mapping[str, Any]],
    *,
    projector: FixedTopologyCompositionProjector,
    expected_group_counts: Mapping[str, int] = EXPECTED_GROUP_COUNTS,
    minimum_no_neutral_recovered: int = 24,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project a predeclared panel and return JSONL rows plus acceptance."""

    materialized = [dict(row) for row in rows]
    attempt_ids: set[str] = set()
    group_counts: Counter[str] = Counter()
    outputs: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    source_reason_counts: Counter[str] = Counter()
    recovered = 0
    controls_identity = 0
    pauling_identity = 0
    source_contract_ok = True

    for row_index, row in enumerate(materialized, start=1):
        attempt_id = str(row.get("attempt_id") or "")
        panel_group = str(row.get("panel_group") or "")
        if not attempt_id:
            raise ProjectionPanelError(f"row {row_index} has no attempt_id")
        if attempt_id in attempt_ids:
            raise ProjectionPanelError(f"duplicate attempt_id: {attempt_id}")
        attempt_ids.add(attempt_id)
        if panel_group not in expected_group_counts:
            raise ProjectionPanelError(
                f"row {row_index} has invalid panel_group {panel_group!r}"
            )
        group_counts[panel_group] += 1
        state_payload = row.get("state")
        if not isinstance(state_payload, Mapping):
            raise ProjectionPanelError(f"row {row_index} has no state mapping")
        state = StratifiedState.from_dict(dict(state_payload))
        if state.attempt_id != attempt_id:
            raise ProjectionPanelError(
                f"row {row_index} attempt_id differs from state.attempt_id"
            )

        result = projector.project(state)
        status_counts[result.status] += 1
        source_reason_counts[result.source_reason] += 1
        if panel_group == "no_neutral":
            source_contract_ok &= result.source_reason == "charge_neutrality_fail"
            recovered += int(result.status == "projected")
        elif panel_group == "pauling_only":
            source_contract_ok &= (
                result.source_reason == "pauling_fail_or_ratio_rejected"
            )
            pauling_identity += int(
                result.status == "identity_protected_reason"
                and not result.changed_orbit_ids
            )
        else:
            source_contract_ok &= result.source_reason in VALID_CONTROL_REASONS
            controls_identity += int(
                result.status == "identity_protected_reason"
                and not result.changed_orbit_ids
            )

        outputs.append(
            {
                "schema": "wqcodiff_composition_mechanism_panel_row_v1",
                "attempt_id": attempt_id,
                "panel_group": panel_group,
                "input_line_number": int(row.get("_input_line_number", row_index)),
                "projection": result.to_dict(include_state=True),
            }
        )

    expected = {str(key): int(value) for key, value in expected_group_counts.items()}
    checks = {
        "attempt_count_exact": len(outputs) == sum(expected.values()),
        "group_counts_exact": dict(group_counts) == expected,
        "source_reason_contract": bool(source_contract_ok),
        "minimum_no_neutral_recovered": recovered
        >= int(minimum_no_neutral_recovered),
        "valid_controls_byte_identical": controls_identity
        == expected.get("valid_control", 0),
        "pauling_only_byte_identical": pauling_identity
        == expected.get("pauling_only", 0),
        "no_classifier_error": status_counts.get("classifier_error", 0) == 0,
        "no_search_budget_exhaustion": status_counts.get("budget_exhausted", 0)
        == 0,
        "one_output_per_attempt": len(outputs) == len(attempt_ids),
    }
    report = {
        "schema": "wqcodiff_composition_mechanism_panel_report_v1",
        "ok": all(checks.values()),
        "attempts": len(outputs),
        "expected_group_counts": expected,
        "observed_group_counts": dict(sorted(group_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "source_reason_counts": dict(sorted(source_reason_counts.items())),
        "no_neutral_recovered": recovered,
        "valid_controls_identity": controls_identity,
        "pauling_only_identity": pauling_identity,
        "minimum_no_neutral_recovered": int(minimum_no_neutral_recovered),
        "checks": checks,
        "scientific_generation_attempts_created": 0,
        "mlip_calls": 0,
        "llm_calls": 0,
        "parent_diffusion_calls": 0,
        "retry_or_replacement_used": False,
    }
    return outputs, report


def _write_outputs_exclusive(
    output_path: Path,
    report_path: Path,
    rows: Iterable[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    descriptors: list[int] = []
    try:
        for path in (output_path, report_path):
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
            descriptors.append(descriptor)
            created.append(path)
        with os.fdopen(descriptors.pop(0), "w", encoding="utf-8") as output_handle:
            for row in rows:
                output_handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
        with os.fdopen(descriptors.pop(0), "w", encoding="utf-8") as report_handle:
            json.dump(report, report_handle, sort_keys=True, indent=2)
            report_handle.write("\n")
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-changed-orbits", type=int, default=6)
    parser.add_argument("--max-candidate-assignments", type=int, default=100_000)
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    if output_path == report_path:
        raise ProjectionPanelError("output and report paths must differ")
    rows = load_panel(input_path)
    outputs, report = run_projection_panel(
        rows,
        projector=FixedTopologyCompositionProjector(
            max_changed_orbits=args.max_changed_orbits,
            max_candidate_assignments=args.max_candidate_assignments,
        ),
    )
    report["input"] = str(input_path)
    report["input_sha256"] = _sha256(input_path)
    report["output"] = str(output_path)
    report["max_changed_orbits"] = args.max_changed_orbits
    report["max_candidate_assignments"] = args.max_candidate_assignments
    _write_outputs_exclusive(output_path, report_path, outputs, report)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["ok"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
