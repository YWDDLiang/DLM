#!/usr/bin/env python3
"""CPU-only G0 audit for frozen dynamic 7+4N crystal generations."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


TAXONOMY_PRECEDENCE = (
    "parse",
    "composition",
    "lattice",
    "pbc_min_distance_lt_0p5A",
    "crystalnn_graph",
    "direct_other",
    "pass",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _attempt_index(row: Mapping[str, Any], fallback: int) -> int:
    for key in ("sample_idx", "source_sample_idx", "ordinal"):
        value = row.get(key)
        if value is not None:
            return int(value)
    match = re.search(r"(\d+)$", str(row.get("attempt_id", "")))
    return int(match.group(1)) if match else int(fallback)


def _terminal_class(flags: Mapping[str, bool]) -> str:
    for name in TAXONOMY_PRECEDENCE[:-1]:
        if flags.get(name, False):
            return name
    return "pass"


def _plan_counter(plan: Mapping[str, Any]) -> Counter[str]:
    return Counter(
        {
            str(symbol): int(count)
            for symbol, count in zip(plan.get("elements", []), plan.get("counts", []))
        }
    )


def _structure_counter(structure: Any) -> Counter[str]:
    return Counter(str(site.specie.symbol) for site in structure.sites)


def _minimum_distance(structure: Any) -> float:
    import numpy as np

    if len(structure) <= 1:
        return math.inf
    matrix = structure.distance_matrix.copy()
    np.fill_diagonal(matrix, np.inf)
    return float(matrix.min())


def _periodic_coordinate_error(left: Any, right: Any) -> float:
    import numpy as np

    delta = np.abs(np.mod(left, 1.0) - np.mod(right, 1.0))
    return float(np.minimum(delta, 1.0 - delta).max())


def _geometry_valid(structure: Any, minimum_distance: float) -> bool:
    return bool(float(structure.volume) >= 0.1 and minimum_distance >= 0.5)


def _audit_row(
    row: Mapping[str, Any], direct: Mapping[str, Any], sample_idx: int
) -> dict[str, Any]:
    from pymatgen.core import Structure
    from crystal_dlm.dynamic_crystal import (
        arrays_to_structure,
        parse_dynamic_answer,
        structure_to_dynamic_answer,
    )

    record: dict[str, Any] = {
        "sample_idx": sample_idx,
        "direct_comp_valid": direct.get("comp_valid") is True,
        "direct_struct_valid": direct.get("struct_valid") is True,
        "direct_valid": direct.get("valid") is True,
        "direct_reason": str(direct.get("reason") or ""),
        "parse_ok": False,
        "composition_exact": False,
        "lattice_ok": False,
        "pbc_minimum_distance_A": None,
        "crystalnn_graph_ok": None,
        "strict_7_plus_4N_pass": False,
        "species_sequence_exact": False,
        "token_reencode_exact": False,
        "token_count": None,
        "expected_token_count": None,
        "length_clip_count": None,
        "angle_clip_count": None,
        "coordinate_clip_count": None,
        "coordinate_wrap_count": None,
        "max_length_error_A": None,
        "max_angle_error_deg": None,
        "max_periodic_fractional_error": None,
        "direct_validity_changed_after_roundtrip": None,
        "error": "",
    }
    flags = {name: False for name in TAXONOMY_PRECEDENCE[:-1]}
    cif = row.get("cif")
    if not cif:
        flags["parse"] = True
        record["error"] = str(row.get("error") or "missing_cif")
        record["terminal_class"] = _terminal_class(flags)
        return record

    try:
        source = Structure.from_str(str(cif), fmt="cif")
        record["parse_ok"] = True
    except Exception as exc:  # pragma: no cover - runtime parser detail
        flags["parse"] = True
        record["error"] = f"{type(exc).__name__}:{exc}"
        record["terminal_class"] = _terminal_class(flags)
        return record

    plan = row.get("plan_state") or {}
    record["composition_exact"] = _structure_counter(source) == _plan_counter(plan)
    flags["composition"] = not record["composition_exact"]

    lengths = tuple(float(value) for value in source.lattice.abc)
    angles = tuple(float(value) for value in source.lattice.angles)
    lattice_finite = all(math.isfinite(value) and value > 0 for value in lengths)
    lattice_finite = lattice_finite and all(math.isfinite(value) for value in angles)
    record["volume_A3"] = float(source.volume)
    record["lattice_ok"] = bool(lattice_finite and source.volume >= 0.1)
    flags["lattice"] = not record["lattice_ok"]

    minimum_distance = _minimum_distance(source)
    record["pbc_minimum_distance_A"] = (
        minimum_distance if math.isfinite(minimum_distance) else None
    )
    flags["pbc_min_distance_lt_0p5A"] = minimum_distance < 0.5

    graph_failure = (
        direct.get("comp_valid") is True
        and direct.get("struct_valid") is True
        and direct.get("valid") is not True
    )
    record["crystalnn_graph_ok"] = not graph_failure
    flags["crystalnn_graph"] = graph_failure
    flags["direct_other"] = bool(
        direct.get("valid") is not True
        and not any(flags[name] for name in TAXONOMY_PRECEDENCE[:-2])
    )

    try:
        answer, diagnostics = structure_to_dynamic_answer(source)
        arrays = parse_dynamic_answer(answer, strict=True)
        rebuilt = arrays_to_structure(arrays)
        answer2, _ = structure_to_dynamic_answer(rebuilt)
        record["strict_7_plus_4N_pass"] = True
        record["token_count"] = len(arrays["tokens"])
        record["expected_token_count"] = 7 + 4 * len(source)
        record["species_sequence_exact"] = [
            site.specie.symbol for site in source.sites
        ] == list(arrays["species"])
        record["token_reencode_exact"] = answer == answer2
        record["length_clip_count"] = int(diagnostics.length_clips)
        record["angle_clip_count"] = int(diagnostics.angle_clips)
        record["coordinate_clip_count"] = int(diagnostics.coord_clips)
        record["coordinate_wrap_count"] = int(diagnostics.coord_wraps)
        record["max_length_error_A"] = max(
            abs(float(a) - float(b))
            for a, b in zip(source.lattice.abc, rebuilt.lattice.abc)
        )
        record["max_angle_error_deg"] = max(
            abs(float(a) - float(b))
            for a, b in zip(source.lattice.angles, rebuilt.lattice.angles)
        )
        record["max_periodic_fractional_error"] = _periodic_coordinate_error(
            source.frac_coords, rebuilt.frac_coords
        )
        rebuilt_minimum = _minimum_distance(rebuilt)
        record["direct_validity_changed_after_roundtrip"] = (
            _geometry_valid(source, minimum_distance)
            != _geometry_valid(rebuilt, rebuilt_minimum)
        )
    except Exception as exc:  # pragma: no cover - runtime codec detail
        record["error"] = f"codec:{type(exc).__name__}:{exc}"

    record["terminal_class"] = _terminal_class(flags)
    return record


def _max_present(records: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in records if row.get(key) is not None]
    return max(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-generation", type=Path, required=True)
    parser.add_argument("--direct-attempts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")

    import crystal_dlm.dynamic_crystal as dynamic_codec
    import crystal_dlm.fixed_slot as fixed_slot_codec

    raw_rows = _read_jsonl(args.raw_generation)
    direct_rows = _read_jsonl(args.direct_attempts)
    raw_by_idx = {_attempt_index(row, pos): row for pos, row in enumerate(raw_rows)}
    direct_by_idx = {
        _attempt_index(row, pos): row for pos, row in enumerate(direct_rows)
    }
    requested = sorted(set(raw_by_idx) | set(direct_by_idx))
    if requested != list(range(len(requested))):
        raise ValueError("sample_idx coverage is not contiguous from zero")

    records = [
        _audit_row(raw_by_idx.get(idx, {}), direct_by_idx.get(idx, {}), idx)
        for idx in requested
    ]
    counts = Counter(str(row["terminal_class"]) for row in records)
    parsed = [row for row in records if row["parse_ok"]]
    report = {
        "schema": "g0_raw_failure_quantization_audit_v1",
        "inputs": {
            "raw_generation": {
                "path": str(args.raw_generation),
                "sha256": _sha256(args.raw_generation),
                "rows": len(raw_rows),
            },
            "direct_attempts": {
                "path": str(args.direct_attempts),
                "sha256": _sha256(args.direct_attempts),
                "rows": len(direct_rows),
            },
        },
        "implementation": {
            "audit_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "dynamic_codec": {
                "path": str(Path(dynamic_codec.__file__).resolve()),
                "sha256": _sha256(Path(dynamic_codec.__file__).resolve()),
            },
            "fixed_slot_codec": {
                "path": str(Path(fixed_slot_codec.__file__).resolve()),
                "sha256": _sha256(Path(fixed_slot_codec.__file__).resolve()),
            },
        },
        "identity": {
            "key": "sample_idx",
            "unique": len(requested),
            "coverage_closed": len(requested) == len(raw_rows) == len(direct_rows),
            "file_order_used_as_identity": False,
        },
        "taxonomy": {
            "precedence": list(TAXONOMY_PRECEDENCE),
            "counts": {name: counts.get(name, 0) for name in TAXONOMY_PRECEDENCE},
            "denominator_closed": sum(counts.values()) == len(requested),
        },
        "quantization": {
            "cif_rows": len(parsed),
            "strict_7_plus_4N_pass": sum(row["strict_7_plus_4N_pass"] for row in parsed),
            "species_sequence_exact": sum(row["species_sequence_exact"] for row in parsed),
            "token_reencode_exact": sum(row["token_reencode_exact"] for row in parsed),
            "clip_counts": {
                "length": sum(int(row["length_clip_count"] or 0) for row in parsed),
                "angle": sum(int(row["angle_clip_count"] or 0) for row in parsed),
                "coordinate": sum(int(row["coordinate_clip_count"] or 0) for row in parsed),
                "coordinate_wrap": sum(int(row["coordinate_wrap_count"] or 0) for row in parsed),
            },
            "max_errors": {
                "length_A": _max_present(parsed, "max_length_error_A"),
                "angle_deg": _max_present(parsed, "max_angle_error_deg"),
                "periodic_fractional": _max_present(parsed, "max_periodic_fractional_error"),
            },
            "direct_validity_changed_after_roundtrip": sum(
                row["direct_validity_changed_after_roundtrip"] is True for row in parsed
            ),
        },
        "energy_taxonomy": {
            "status": "not_available_no_CHGNet_run",
            "valid_but_high_energy_count": None,
        },
    }
    args.output_dir.mkdir(parents=True)
    records_path = args.output_dir / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "REPORT.md").write_text(
        "# G0 raw geometry and quantization audit\n\n"
        f"- requested: `{len(requested)}`\n"
        f"- terminal taxonomy: `{dict(report['taxonomy']['counts'])}`\n"
        f"- strict 7+4N: `{report['quantization']['strict_7_plus_4N_pass']}/{len(parsed)}`\n"
        f"- token re-encode exact: `{report['quantization']['token_reencode_exact']}/{len(parsed)}`\n"
        f"- round-trip validity flips: `{report['quantization']['direct_validity_changed_after_roundtrip']}`\n"
        "- energy class: unavailable because this frozen raw screen did not run CHGNet.\n"
    )
    output_files = [args.output_dir / name for name in ("report.json", "records.jsonl", "records.csv", "REPORT.md")]
    (args.output_dir / "OUTPUTS.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in output_files)
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
