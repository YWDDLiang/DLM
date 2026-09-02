#!/usr/bin/env python3
"""Audit the deployed SSCD DLM tokenizer, checkpoint and MP20 coverage."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.dynamic_crystal import (  # noqa: E402
    arrays_to_dynamic_answer,
    dynamic_answer_token_count,
    parse_dynamic_answer,
)
from crystal_dlm.fixed_slot import (  # noqa: E402
    MASK_TOKEN_ID,
    FixedSlotConfig,
    SYMBOL_TO_Z,
    build_special_tokens,
)


FAMILY_PATTERNS = (
    ("N", re.compile(r"^<N_\d{3}>$")),
    ("length", re.compile(r"^<L[ABC]_\d{3}>$")),
    ("angle", re.compile(r"^<A[ABG]_\d{3}>$")),
    ("element", re.compile(r"^<E_[A-Z][a-z]?>$")),
    ("coordinate", re.compile(r"^<[XYZ]_\d{3}>$")),
    ("fixed_only", re.compile(r"^<(?:S\d{2}|EMPTY|[XYZ]_PAD)>$")),
)


def token_family(token: str) -> str:
    for name, pattern in FAMILY_PATTERNS:
        if pattern.fullmatch(str(token)):
            return name
    return "unknown"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


def _minimum_off_diagonal(matrix: Any) -> float | None:
    import numpy as np

    values = np.asarray(matrix, dtype=float)
    if values.shape[0] < 2:
        return None
    copy = values.copy()
    np.fill_diagonal(copy, np.inf)
    result = float(np.min(copy))
    return result if math.isfinite(result) else None


def _audit_structure(payload: tuple[int, str]) -> dict[str, Any]:
    row_index, cif = payload
    try:
        from pymatgen.core import Lattice, Structure

        structure = Structure.from_str(cif, fmt="cif")
        if not bool(structure.is_ordered):
            return {"row_index": row_index, "status": "unordered"}
        n_atom = int(len(structure))
        species = [str(site.specie.symbol) for site in structure.sites]
        unsupported = sorted({symbol for symbol in species if symbol not in SYMBOL_TO_Z})
        if unsupported:
            return {
                "row_index": row_index,
                "status": "unsupported_element",
                "unsupported": unsupported,
            }
        lengths = [float(value) for value in structure.lattice.abc]
        angles = [float(value) for value in structure.lattice.angles]
        coords = [[float(value) for value in row] for row in structure.frac_coords]
        answer, diagnostics = arrays_to_dynamic_answer(
            lengths=lengths,
            angles=angles,
            species=species,
            frac_coords=coords,
        )
        parsed = parse_dynamic_answer(answer, strict=True)
        tokens = list(parsed["tokens"])
        coord_100 = sum(
            token.startswith(("<X_", "<Y_", "<Z_")) and token.endswith("_100>")
            for token in tokens
        )
        length_000 = sum(
            token.startswith(("<LA_", "<LB_", "<LC_")) and token.endswith("_000>")
            for token in tokens
        )
        length_500 = sum(
            token.startswith(("<LA_", "<LB_", "<LC_")) and token.endswith("_500>")
            for token in tokens
        )
        coord_keys = [
            tuple(int(round(float(value) * 100.0)) % 100 for value in row)
            for row in parsed["frac_coords"]
        ]
        quantized_duplicates = len(coord_keys) - len(set(coord_keys))
        lattice = Lattice.from_parameters(
            *(
                [float(value) for value in parsed["lengths"]]
                + [float(value) for value in parsed["angles"]]
            )
        )
        quantized = Structure(
            lattice=lattice,
            species=parsed["species"],
            coords=parsed["frac_coords"],
            coords_are_cartesian=False,
        )
        return {
            "row_index": row_index,
            "status": "ok",
            "N": n_atom,
            "elements": sorted(set(species)),
            "length_min": min(lengths),
            "length_max": max(lengths),
            "angle_min": min(angles),
            "angle_max": max(angles),
            "raw_length_gt_50": int(any(value > 50.0 for value in lengths)),
            "coord_100": int(coord_100),
            "length_000": int(length_000),
            "length_500": int(length_500),
            "length_clips": int(diagnostics.length_clips),
            "angle_clips": int(diagnostics.angle_clips),
            "coord_clips": int(diagnostics.coord_clips),
            "coord_wraps": int(diagnostics.coord_wraps),
            "semantic_length": len(tokens),
            "expected_length": dynamic_answer_token_count(n_atom),
            "answer": answer,
            "quantized_duplicates": int(quantized_duplicates),
            "quantized_volume": float(quantized.volume),
            "quantized_min_distance": _minimum_off_diagonal(
                quantized.distance_matrix
            ),
        }
    except Exception as exc:
        return {
            "row_index": row_index,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:240],
        }


def audit_tokenizer(tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = build_special_tokens()
    vocab = tokenizer.get_vocab()
    rows: list[dict[str, Any]] = []
    seen_ids: dict[int, str] = {}
    failures: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = {}
    for token in expected:
        family = token_family(token)
        counters = family_counts.setdefault(family, Counter())
        counters["expected"] += 1
        present = token in vocab
        token_id = int(vocab[token]) if present else None
        encoded = list(
            tokenizer(token, add_special_tokens=False).get("input_ids", [])
        )
        atomic = bool(present and encoded == [token_id])
        decoded = (
            str(
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            if present
            else None
        )
        decode_exact = bool(decoded == token)
        unique_id = token_id is not None and token_id not in seen_ids
        if token_id is not None and unique_id:
            seen_ids[token_id] = token
        if present:
            counters["present"] += 1
        if atomic:
            counters["atomic"] += 1
        if decode_exact:
            counters["decode_exact"] += 1
        if unique_id:
            counters["unique_id"] += 1
        for key, passed in (
            ("missing", present),
            ("non_atomic", atomic),
            ("decode_mismatch", decode_exact),
            ("duplicate_id", unique_id),
        ):
            if not passed:
                failures[key] += 1
        rows.append(
            {
                "family": family,
                "token": token,
                "token_id": token_id,
                "present": present,
                "atomic": atomic,
                "decode_exact": decode_exact,
                "unique_id": unique_id,
            }
        )

    crystal_ids = {int(row["token_id"]) for row in rows if row["token_id"] is not None}
    special_ids = {
        name: getattr(tokenizer, name, None)
        for name in (
            "pad_token_id",
            "eos_token_id",
            "bos_token_id",
            "unk_token_id",
            "mask_token_id",
        )
    }
    conflicts = {
        name: int(value)
        for name, value in special_ids.items()
        if value is not None and int(value) in crystal_ids
    }
    non_mask_special = {
        int(value)
        for name, value in special_ids.items()
        if name != "mask_token_id" and value is not None
    }
    mask_contract = {
        "configured_mask_id": int(MASK_TOKEN_ID),
        "tokenizer_mask_token_id": special_ids["mask_token_id"],
        "configured_mask_token": tokenizer.convert_ids_to_tokens(int(MASK_TOKEN_ID)),
        "within_tokenizer": int(MASK_TOKEN_ID) < len(tokenizer),
        "distinct_from_crystal_ids": int(MASK_TOKEN_ID) not in crystal_ids,
        "distinct_from_pad_eos_bos_unk": int(MASK_TOKEN_ID)
        not in non_mask_special,
    }
    summary = {
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "expected_special_tokens": len(expected),
        "dynamic_special_tokens": sum(
            1 for token in expected if token_family(token) != "fixed_only"
        ),
        "families": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(family_counts.items())
        },
        "failures": dict(sorted(failures.items())),
        "special_ids": special_ids,
        "special_crystal_id_conflicts": conflicts,
        "mask_contract": mask_contract,
    }
    return rows, summary


def inspect_adapter(path: Path | None, *, tokenizer_size: int) -> dict[str, Any]:
    if path is None:
        return {"requested": False}
    if not path.is_file():
        raise FileNotFoundError(path)
    from safetensors import safe_open

    shapes: dict[str, list[int]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            if any(name in key for name in ("wte", "ff_out")):
                shapes[key] = [int(value) for value in handle.get_slice(key).get_shape()]
    vocab_axes: list[dict[str, Any]] = []
    for key, shape in shapes.items():
        matching_axes = [
            axis for axis, size in enumerate(shape) if int(size) == int(tokenizer_size)
        ]
        vocab_axes.append(
            {"key": key, "shape": shape, "tokenizer_sized_axes": matching_axes}
        )
    return {
        "requested": True,
        "path": str(path.resolve()),
        "matched_keys": vocab_axes,
        "has_input_embedding": any("wte" in row["key"] for row in vocab_axes),
        "has_output_head": any("ff_out" in row["key"] for row in vocab_axes),
        "all_matched_modules_have_vocab_axis": bool(vocab_axes)
        and all(row["tokenizer_sized_axes"] for row in vocab_axes),
    }


def _csv_payloads(path: Path) -> Iterable[tuple[int, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        field = "cif" if "cif" in reader.fieldnames else "cif_str"
        if field not in reader.fieldnames:
            raise ValueError(f"{path} has neither cif nor cif_str")
        for index, row in enumerate(reader):
            yield index, str(row[field])


def audit_mp20(
    *,
    label: str,
    path: Path,
    tokenizer: Any,
    workers: int,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    n_hist: Counter[str] = Counter()
    elements: set[str] = set()
    examples: list[dict[str, Any]] = []
    numeric: dict[str, list[float]] = {
        "length_min": [],
        "length_max": [],
        "angle_min": [],
        "angle_max": [],
        "quantized_volume": [],
        "quantized_min_distance": [],
    }
    payloads = _csv_payloads(path)
    if int(workers) > 1:
        executor: Any = ProcessPoolExecutor(max_workers=int(workers))
        results = executor.map(_audit_structure, payloads, chunksize=32)
    else:
        executor = None
        results = map(_audit_structure, payloads)
    try:
        for result in results:
            counts["rows"] += 1
            status = str(result["status"])
            counts[f"status:{status}"] += 1
            if status != "ok":
                if len(examples) < 20:
                    examples.append(result)
                continue
            n_atom = int(result["N"])
            n_hist[str(n_atom)] += 1
            elements.update(str(value) for value in result["elements"])
            for key in (
                "raw_length_gt_50",
                "coord_100",
                "length_000",
                "length_500",
                "length_clips",
                "angle_clips",
                "coord_clips",
                "coord_wraps",
                "quantized_duplicates",
            ):
                counts[key] += int(result[key])
            semantic_ok = int(result["semantic_length"]) == int(
                result["expected_length"]
            )
            counts["semantic_7plus4N"] += int(semantic_ok)
            encoded = list(
                tokenizer(
                    str(result["answer"]), add_special_tokens=False
                ).get("input_ids", [])
            )
            token_ok = len(encoded) == int(result["expected_length"])
            counts["tokenizer_7plus4N"] += int(token_ok)
            minimum = result.get("quantized_min_distance")
            if minimum is not None and float(minimum) < 0.5:
                counts["quantized_min_distance_lt_0p5"] += 1
            for key in numeric:
                value = result.get(key)
                if value is not None and math.isfinite(float(value)):
                    numeric[key].append(float(value))
            if (not semantic_ok or not token_ok) and len(examples) < 20:
                examples.append(
                    {
                        "row_index": result["row_index"],
                        "semantic_length": result["semantic_length"],
                        "tokenizer_length": len(encoded),
                        "expected_length": result["expected_length"],
                    }
                )
    finally:
        if executor is not None:
            executor.shutdown()
    return {
        "label": label,
        "path": str(path.resolve()),
        "counts": dict(sorted(counts.items())),
        "N_histogram": dict(sorted(n_hist.items(), key=lambda item: int(item[0]))),
        "elements": sorted(elements, key=lambda value: SYMBOL_TO_Z.get(value, 10**9)),
        "ranges": {
            key: {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for key, values in numeric.items()
        },
        "examples": examples,
    }


def audit_context(label: str, path: Path, tokenizer: Any, max_length: int) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total_lengths: list[int] = []
    max_row: dict[str, Any] | None = None
    for index, row in enumerate(iter_jsonl(path)):
        counts["rows"] += 1
        prompt = str(row.get("prompt") or "")
        answer = str(row.get("answer") or "")
        prompt_ids = tokenizer(prompt.rstrip() + "\n", add_special_tokens=False)[
            "input_ids"
        ]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        total = len(prompt_ids) + len(answer_ids)
        total_lengths.append(total)
        if total > int(max_length):
            counts["over_limit"] += 1
        plan = row.get("plan_state") or row.get("r5_plan_state") or {}
        n_atom = int(row.get("num_atoms") or plan.get("N") or 0)
        exact = n_atom > 0 and len(answer_ids) == dynamic_answer_token_count(n_atom)
        counts["answer_7plus4N"] += int(exact)
        if max_row is None or total > int(max_row["total"]):
            max_row = {
                "row_index": index,
                "prompt": len(prompt_ids),
                "answer": len(answer_ids),
                "total": total,
                "N": n_atom,
            }
    return {
        "label": label,
        "path": str(path.resolve()),
        "counts": dict(sorted(counts.items())),
        "max_length": int(max_length),
        "total_min": min(total_lengths) if total_lengths else None,
        "total_max": max(total_lengths) if total_lengths else None,
        "minimum_margin": (
            int(max_length) - max(total_lengths) if total_lengths else None
        ),
        "max_row": max_row,
    }


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected LABEL=PATH, got {value!r}")
    label, raw_path = value.split("=", 1)
    if not label.strip():
        raise ValueError("path label is empty")
    return label.strip(), Path(raw_path)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--adapter-safetensors", type=Path)
    parser.add_argument("--mp20", action="append", default=[], help="LABEL=CSV")
    parser.add_argument("--context-jsonl", action="append", default=[], help="LABEL=JSONL")
    parser.add_argument("--max-length", type=int, default=382)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path), trust_remote_code=True
    )
    token_rows, tokenizer_report = audit_tokenizer(tokenizer)
    checkpoint_report = inspect_adapter(
        args.adapter_safetensors,
        tokenizer_size=len(tokenizer),
    )
    mp20_reports = [
        audit_mp20(
            label=label,
            path=path,
            tokenizer=tokenizer,
            workers=int(args.workers),
        )
        for label, path in (parse_labeled_path(value) for value in args.mp20)
    ]
    context_reports = [
        audit_context(
            label,
            path,
            tokenizer,
            int(args.max_length),
        )
        for label, path in (
            parse_labeled_path(value) for value in args.context_jsonl
        )
    ]

    report = {
        "schema": "sscd_token_checkpoint_audit_v1",
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "tokenizer": tokenizer_report,
        "checkpoint": checkpoint_report,
        "mp20": mp20_reports,
        "contexts": context_reports,
    }
    gates = {
        "all_expected_tokens_atomic": not bool(
            tokenizer_report["failures"].get("missing")
            or tokenizer_report["failures"].get("non_atomic")
            or tokenizer_report["failures"].get("decode_mismatch")
            or tokenizer_report["failures"].get("duplicate_id")
        ),
        "mask_contract": all(
            bool(tokenizer_report["mask_contract"][key])
            for key in (
                "within_tokenizer",
                "distinct_from_crystal_ids",
                "distinct_from_pad_eos_bos_unk",
            )
        ),
        "checkpoint_vocab_rows": (
            not checkpoint_report.get("requested")
            or (
                bool(checkpoint_report.get("has_input_embedding"))
                and bool(checkpoint_report.get("has_output_head"))
                and bool(
                    checkpoint_report.get("all_matched_modules_have_vocab_axis")
                )
            )
        ),
        "all_mp20_rows_tokenizer_7plus4N": all(
            int(item["counts"].get("status:ok", 0))
            == int(item["counts"].get("tokenizer_7plus4N", 0))
            for item in mp20_reports
        ),
        "all_mp20_rows_parsed": all(
            int(item["counts"].get("rows", 0))
            == int(item["counts"].get("status:ok", 0))
            for item in mp20_reports
        ),
        "expected_mp20_row_counts": all(
            int(item["counts"].get("rows", 0))
            == {"train": 27136, "val": 9047}.get(
                str(item["label"]).lower(),
                int(item["counts"].get("rows", 0)),
            )
            for item in mp20_reports
        ),
        "all_contexts_fit": all(
            int(item["counts"].get("over_limit", 0)) == 0
            for item in context_reports
        ),
        "all_context_answers_7plus4N": all(
            int(item["counts"].get("rows", 0))
            == int(item["counts"].get("answer_7plus4N", 0))
            for item in context_reports
        ),
    }
    report["gates"] = gates
    (output / "SSCD_TOKEN_CHECKPOINT_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "TOKEN_FAMILY_AUDIT.csv", token_rows)
    write_csv(
        output / "MP20_TOKEN_COVERAGE.csv",
        [
            {
                "split": item["label"],
                **item["counts"],
                "elements": ",".join(item["elements"]),
                "N_histogram": json.dumps(item["N_histogram"], sort_keys=True),
            }
            for item in mp20_reports
        ],
    )
    write_csv(
        output / "CONTEXT_AUDIT.csv",
        [
            {
                "dataset": item["label"],
                **item["counts"],
                "total_min": item["total_min"],
                "total_max": item["total_max"],
                "minimum_margin": item["minimum_margin"],
            }
            for item in context_reports
        ],
    )
    (output / "CHECKPOINT_ROUNDTRIP.json").write_text(
        json.dumps(
            {
                "tokenizer": tokenizer_report,
                "checkpoint": checkpoint_report,
                "gates": gates,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# SSCD token/checkpoint coverage audit",
        "",
        f"- tokenizer size: {tokenizer_report['vocab_size']}",
        f"- expected special tokens: {tokenizer_report['expected_special_tokens']}",
        f"- dynamic special tokens: {tokenizer_report['dynamic_special_tokens']}",
        f"- configured mask token: {tokenizer_report['mask_contract']['configured_mask_token']!r}",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in gates.items())
    lines.extend(["", "## MP20", ""])
    for item in mp20_reports:
        counts = item["counts"]
        lines.append(
            f"- {item['label']}: rows={counts.get('rows', 0)}, "
            f"ok={counts.get('status:ok', 0)}, "
            f"tokenizer_7+4N={counts.get('tokenizer_7plus4N', 0)}, "
            f"length_clips={counts.get('length_clips', 0)}, "
            f"coord_100={counts.get('coord_100', 0)}, "
            f"quantized_duplicate_rows={counts.get('quantized_duplicates', 0)}"
        )
    (output / "SSCD_TOKEN_CHECKPOINT_AUDIT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    if all(gates.values()):
        (output / "_SUCCESS").touch()
    else:
        (output / "_AUDIT_FINDINGS").touch()
    print(json.dumps({"output_dir": str(output), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
