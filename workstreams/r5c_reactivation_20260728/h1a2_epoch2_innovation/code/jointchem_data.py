#!/usr/bin/env python3
"""Deterministic H1-A2 JointChem data construction.

This module reads the frozen H1-A2 rich-Plan JSONL files.  It emits an
immutable row ledger, a 3,200-row training stream, and a held-out validation
panel.  It never queries an API, loads an energy model, filters generated
crystals, or performs inference-time retry/repair/reranking.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


THIS_DIR = Path(__file__).resolve().parent
REACTIVATION_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[3]
RESTORED_BASELINE_ROOT = REACTIVATION_ROOT / "baseline"
RUNTIME_ROOT = RESTORED_BASELINE_ROOT if (RESTORED_BASELINE_ROOT / "crystal_dlm").is_dir() else PROJECT_ROOT
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from crystal_dlm.composition_validity import classify_smact_validity  # noqa: E402
from crystal_dlm.fixed_slot import SYMBOL_TO_Z  # noqa: E402
from crystal_dlm.r5_plan_body import (  # noqa: E402
    H1_RICH_PLAN_FORMAT,
    format_composition_plan,
    formula_from_symbol_counts,
    parse_composition_plan,
)


EXPECTED_SOURCE_SHA256 = {
    "train": "d431dfec1de8c3240dbc5648867be1b4b676fd85276e805a177b9944f3a1a157",
    "val": "59327aa789ae5d2bbb66d8a8f0dc882d594bcc14623aa96ce95076ed1b6fc540",
    "test": "032845826acf1fcb9e7893fc91da05bc0cd9d2363c80ae459d5c446c3c6d5ea8",
}

ALLOWED_INVALID_REASONS = {
    "charge_neutrality_fail",
    "pauling_fail_or_ratio_rejected",
}

BANNED_SERIALIZED_MARKERS = (
    "e_hull",
    "ehull",
    "formation_energy",
    "chgnet",
    "mattersim",
    "mlip",
    "sun_label",
    "strict_sun",
    "meta_sun",
    "mp_api_key",
    "mp_query",
)


Classifier = Callable[[Sequence[int], Sequence[int]], Mapping[str, Any]]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_key(seed: int, namespace: str, value: str) -> str:
    return sha256_bytes(f"{int(seed)}\0{namespace}\0{value}".encode("utf-8"))


def iter_jsonl_with_hash(path: Path) -> Iterable[tuple[int, str, dict[str, Any]]]:
    with path.open("rb") as handle:
        for row_idx, raw_line in enumerate(handle):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{row_idx + 1} is not a JSON object")
            yield row_idx, sha256_bytes(raw_line), payload


def default_classifier(elems: Sequence[int], counts: Sequence[int]) -> Mapping[str, Any]:
    return classify_smact_validity(tuple(int(v) for v in elems), tuple(int(v) for v in counts))


def plan_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    answer = str(record.get("answer") or record.get("plan_text") or "").strip()
    if not answer:
        raise ValueError("record has no rich-Plan answer")
    return parse_composition_plan(answer, plan_style=H1_RICH_PLAN_FORMAT, max_atoms=20)


def classify_plan(
    plan: Mapping[str, Any],
    *,
    classifier: Classifier = default_classifier,
) -> dict[str, Any]:
    symbols = [str(value) for value in plan.get("elements") or []]
    counts = [int(value) for value in plan.get("counts") or []]
    elems = [int(SYMBOL_TO_Z[symbol]) for symbol in symbols]
    result = dict(classifier(elems, counts))
    if result.get("valid") not in (True, False):
        raise RuntimeError(f"composition validator returned non-boolean validity: {result}")
    return result


def _count_mutation_candidates(counts: Sequence[int], *, row_key: str, seed: int) -> list[list[int]]:
    source = [int(value) for value in counts]
    candidates: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for src_idx, src_count in enumerate(source):
        for dst_idx in range(len(source)):
            if src_idx == dst_idx:
                continue
            for delta in range(1, int(src_count)):
                candidate = list(source)
                candidate[src_idx] -= delta
                candidate[dst_idx] += delta
                key = tuple(candidate)
                if min(candidate) <= 0 or key == tuple(source) or key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
    candidates.sort(
        key=lambda value: deterministic_key(
            seed,
            "chemistry-negative",
            row_key + "\0" + ",".join(str(v) for v in value),
        )
    )
    return candidates


def make_chemistry_negative(
    plan: Mapping[str, Any],
    *,
    row_key: str,
    seed: int,
    classifier: Classifier = default_classifier,
    max_candidates: int = 256,
) -> tuple[str | None, dict[str, Any]]:
    """Mutate counts while preserving elements, N, arity, and rich fields."""

    symbols = [str(value) for value in plan.get("elements") or []]
    counts = [int(value) for value in plan.get("counts") or []]
    elems = [int(SYMBOL_TO_Z[symbol]) for symbol in symbols]
    source_n = int(sum(counts))
    source_arity = len(symbols)
    for candidate_idx, candidate_counts in enumerate(
        _count_mutation_candidates(counts, row_key=row_key, seed=seed)[: int(max_candidates)]
    ):
        classification = dict(classifier(elems, candidate_counts))
        if classification.get("valid") is not False:
            continue
        reason = str(classification.get("reason"))
        if reason not in ALLOWED_INVALID_REASONS:
            continue
        negative = dict(plan)
        negative["counts"] = candidate_counts
        negative["N"] = int(sum(candidate_counts))
        negative["formula"] = formula_from_symbol_counts(symbols, candidate_counts)
        negative["reduced_formula"] = negative["formula"]
        if int(negative["N"]) != source_n or len(candidate_counts) != source_arity:
            raise AssertionError("chemistry negative changed N or arity")
        text = format_composition_plan(negative, plan_style=H1_RICH_PLAN_FORMAT)
        return text, {
            "available": True,
            "candidate_index": candidate_idx,
            "reason": reason,
            "source_counts": counts,
            "negative_counts": candidate_counts,
            "N": source_n,
            "arity": source_arity,
        }
    return None, {
        "available": False,
        "reason": "no_invalid_same_elements_same_N_count_mutation",
        "source_counts": counts,
        "N": source_n,
        "arity": source_arity,
    }


def n_bucket(num_atoms: int) -> str:
    value = int(num_atoms)
    lower = 1 + 5 * ((value - 1) // 5)
    upper = min(20, lower + 4)
    return f"N_{lower:02d}_{upper:02d}"


def donor_bucket(plan: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        n_bucket(int(plan["N"])),
        len(plan.get("elements") or []),
        str(plan.get("anion_framework", "other")),
    )


def geometry_tuple(plan: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(plan.get("lattice_system")),
        str(plan.get("spacegroup_bucket")),
        str(plan.get("volume_per_atom_bin")),
    )


def composition_signature(plan: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    counts = [int(value) for value in plan.get("counts") or []]
    divisor = 0
    for count in counts:
        divisor = math.gcd(divisor, count)
    divisor = max(1, divisor)
    return tuple(
        sorted(
            zip(
                (str(value) for value in plan.get("elements") or []),
                (count // divisor for count in counts),
            )
        )
    )


def make_joint_negative(
    plan: Mapping[str, Any],
    donor: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Swap only lattice/space-group/volume; formula/anion/charge stay fixed."""

    if donor_bucket(plan) != donor_bucket(donor):
        raise ValueError("joint donor is outside the registered matched bucket")
    source_tuple = geometry_tuple(plan)
    donor_tuple = geometry_tuple(donor)
    if source_tuple == donor_tuple:
        raise ValueError("joint donor has the same geometry tuple")
    negative = dict(plan)
    negative["lattice_system"] = donor_tuple[0]
    negative["spacegroup_bucket"] = donor_tuple[1]
    negative["volume_per_atom_bin"] = donor_tuple[2]
    for field in ("formula", "anion_framework", "charge_bucket"):
        if str(negative.get(field)) != str(plan.get(field)):
            raise AssertionError(f"joint negative changed frozen field {field}")
    return format_composition_plan(negative, plan_style=H1_RICH_PLAN_FORMAT), {
        "available": True,
        "bucket": list(donor_bucket(plan)),
        "source_geometry": list(source_tuple),
        "negative_geometry": list(donor_tuple),
    }


def _copy_model_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": record.get("messages"),
        "prompt": record.get("prompt"),
        "answer": str(record.get("answer") or record.get("plan_text") or "").strip(),
    }
    if not payload["answer"]:
        raise ValueError("source row has no answer")
    if payload["prompt"] is None and not isinstance(payload["messages"], list):
        raise ValueError("source row has neither prompt nor messages")
    return payload


def enrich_rows(
    path: Path,
    *,
    split: str,
    seed: int,
    classifier: Classifier = default_classifier,
    max_negative_candidates: int = 256,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_idx, line_sha, source in iter_jsonl_with_hash(path):
        plan = plan_from_record(source)
        classification = classify_plan(plan, classifier=classifier)
        row_key = f"{split}:{row_idx}:{line_sha}"
        chemistry_text = None
        chemistry_audit: dict[str, Any] = {"available": False, "reason": "positive_not_eligible"}
        if classification.get("valid") is True and classification.get("reason") == "charge_neutral_pauling_valid":
            chemistry_text, chemistry_audit = make_chemistry_negative(
                plan,
                row_key=row_key,
                seed=seed,
                classifier=classifier,
                max_candidates=max_negative_candidates,
            )
        model_fields = _copy_model_fields(source)
        if "sample_id:" in canonical_json(model_fields).lower():
            raise ValueError(f"{split}:{row_idx} contains sample_id in historical no-ID mode")
        rows.append(
            {
                **model_fields,
                "schema": "h1a2_jointchem_enriched_row_v1",
                "split": split,
                "row_idx": int(row_idx),
                "source_line_sha256": line_sha,
                "row_key": row_key,
                "formula": str(plan["formula"]),
                "elements": [str(value) for value in plan["elements"]],
                "counts": [int(value) for value in plan["counts"]],
                "N": int(plan["N"]),
                "arity": len(plan["elements"]),
                "anion_framework": str(plan.get("anion_framework")),
                "charge_bucket": str(plan.get("charge_bucket")),
                "lattice_system": str(plan.get("lattice_system")),
                "spacegroup_bucket": str(plan.get("spacegroup_bucket")),
                "volume_per_atom_bin": str(plan.get("volume_per_atom_bin")),
                "composition_valid": bool(classification["valid"]),
                "composition_reason": str(classification.get("reason")),
                "chemistry_negative_answer": chemistry_text,
                "chemistry_negative_audit": chemistry_audit,
                "_plan": plan,
            }
        )
    return rows


def attach_joint_negatives(rows: list[dict[str, Any]], *, seed: int) -> None:
    buckets: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[donor_bucket(row["_plan"])].append(row)
    for row in rows:
        if row.get("composition_valid") is not True:
            row["joint_negative_answer"] = None
            row["joint_negative_audit"] = {
                "available": False,
                "reason": "positive_not_composition_valid",
                "bucket": list(donor_bucket(row["_plan"])),
            }
            continue
        candidates = [
            donor
            for donor in buckets[donor_bucket(row["_plan"])]
            if donor["row_key"] != row["row_key"]
            and donor.get("composition_valid") is True
            and composition_signature(donor["_plan"]) != composition_signature(row["_plan"])
            and geometry_tuple(donor["_plan"]) != geometry_tuple(row["_plan"])
        ]
        candidates.sort(
            key=lambda donor: deterministic_key(
                seed,
                "joint-donor",
                row["row_key"] + "\0" + donor["row_key"],
            )
        )
        if not candidates:
            row["joint_negative_answer"] = None
            row["joint_negative_audit"] = {
                "available": False,
                "reason": "no_distinct_geometry_donor_in_bucket",
                "bucket": list(donor_bucket(row["_plan"])),
            }
            continue
        donor = candidates[0]
        answer, audit = make_joint_negative(row["_plan"], donor["_plan"])
        audit["donor_row_key"] = donor["row_key"]
        audit["donor_formula"] = str(donor["_plan"].get("formula"))
        row["joint_negative_answer"] = answer
        row["joint_negative_audit"] = audit


def _stratified_valid_selection(
    rows: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    valid = [row for row in rows if row["composition_valid"]]
    if len(valid) < int(count):
        raise ValueError(f"need {count} valid rows, found {len(valid)}")
    source_reason_counts = Counter(row["composition_reason"] for row in rows)
    valid_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        valid_by_reason[row["composition_reason"]].append(row)
    for reason, values in valid_by_reason.items():
        values.sort(key=lambda row: deterministic_key(seed, f"valid-{reason}", row["row_key"]))

    # Cap shortcuts at their share of the full source population, not their
    # inflated share after invalid compositions are removed.
    quotas: dict[str, int] = {}
    for shortcut in ("all_metal_shortcut", "single_element_shortcut"):
        quotas[shortcut] = min(
            len(valid_by_reason.get(shortcut, [])),
            int(math.floor(int(count) * source_reason_counts.get(shortcut, 0) / max(1, len(rows)))),
        )
    selected: list[dict[str, Any]] = []
    for reason, quota in quotas.items():
        selected.extend(valid_by_reason.get(reason, [])[:quota])
    selected_keys = {row["row_key"] for row in selected}
    shortcut_reasons = set(quotas)
    remaining = [
        row
        for row in valid
        if row["row_key"] not in selected_keys
        and row["composition_reason"] not in shortcut_reasons
    ]
    remaining.sort(key=lambda row: deterministic_key(seed, "valid-remainder", row["row_key"]))
    selected.extend(remaining[: int(count) - len(selected)])
    if len(selected) < int(count):
        # Only used if non-shortcut valid rows are genuinely insufficient.
        fallback = [
            row
            for row in valid
            if row["row_key"] not in {value["row_key"] for value in selected}
        ]
        fallback.sort(key=lambda row: deterministic_key(seed, "valid-shortcut-fallback", row["row_key"]))
        selected.extend(fallback[: int(count) - len(selected)])
    if len(selected) != int(count):
        raise AssertionError("stratified valid selection did not reach requested count")
    return selected


def replay_stratum(row: Mapping[str, Any]) -> tuple[str, int, str, str, str, str, str]:
    """Frozen anti-drift stratum used by epoch-2 anchor replay."""

    num_atoms = max(1, int(row.get("N", 1)))
    return (
        n_bucket(num_atoms),
        int(row.get("arity", 0)),
        str(row.get("anion_framework", "missing")),
        str(row.get("charge_bucket", "missing")),
        str(row.get("lattice_system", "missing")),
        str(row.get("spacegroup_bucket", "missing")),
        str(row.get("volume_per_atom_bin", "missing")),
    )


def _proportional_stratified_selection(
    rows: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    """Select an exact proportional sample across the frozen Plan strata."""

    if len(rows) < int(count):
        raise ValueError(f"need {count} rows, found {len(rows)}")
    buckets: dict[tuple[str, int, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[replay_stratum(row)].append(row)
    total = len(rows)
    quotas: dict[tuple[str, int, str, str, str, str, str], int] = {}
    remainders: list[tuple[float, str, tuple[str, int, str, str, str, str, str]]] = []
    for stratum, values in buckets.items():
        values.sort(
            key=lambda row: deterministic_key(seed, f"{namespace}-within", row["row_key"])
        )
        exact = int(count) * len(values) / total
        quotas[stratum] = min(len(values), int(math.floor(exact)))
        remainders.append(
            (
                exact - math.floor(exact),
                deterministic_key(seed, f"{namespace}-remainder", canonical_json(stratum)),
                stratum,
            )
        )
    remaining = int(count) - sum(quotas.values())
    for _, _, stratum in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if quotas[stratum] < len(buckets[stratum]):
            quotas[stratum] += 1
            remaining -= 1
    if remaining:
        raise AssertionError(f"proportional selection left {remaining} unassigned rows")
    selected = [
        row
        for stratum, values in buckets.items()
        for row in values[: quotas[stratum]]
    ]
    selected.sort(key=lambda row: deterministic_key(seed, f"{namespace}-final", row["row_key"]))
    if len(selected) != int(count):
        raise AssertionError("proportional stratified selection has the wrong row count")
    return selected


def build_training_stream(
    rows: Sequence[dict[str, Any]],
    *,
    total_rows: int,
    positive_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    positive_count = int(round(int(total_rows) * float(positive_fraction)))
    anchor_count = int(total_rows) - positive_count
    positives = _stratified_valid_selection(rows, count=positive_count, seed=seed)
    anchors = _proportional_stratified_selection(
        rows,
        count=anchor_count,
        seed=seed,
        namespace="epoch2-anchor",
    )
    stream: list[dict[str, Any]] = []
    for stream_role, values in (("chemistry_valid_positive", positives), ("epoch2_anchor", anchors)):
        for row in values:
            stream.append({**row, "stream_role": stream_role})
    stream.sort(
        key=lambda row: deterministic_key(
            seed,
            "training-stream",
            row["stream_role"] + "\0" + row["row_key"],
        )
    )
    if len(stream) != int(total_rows):
        raise AssertionError("training stream has the wrong row count")
    return stream


def build_validation_panel(
    rows: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(rows) < int(count):
        raise ValueError(f"need {count} validation rows, found {len(rows)}")
    selected = sorted(
        rows,
        key=lambda row: deterministic_key(seed, "validation-panel", row["row_key"]),
    )[: int(count)]
    return [{**row, "stream_role": "validation"} for row in selected]


def public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "_plan"}


def assert_no_leakage(payload: Mapping[str, Any]) -> None:
    serialized = canonical_json(payload).lower()
    found = [marker for marker in BANNED_SERIALIZED_MARKERS if marker in serialized]
    if found:
        raise ValueError(f"forbidden training/evaluation markers found: {found}")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = public_row(row)
            assert_no_leakage(payload)
            handle.write(canonical_json(payload) + "\n")
    return sha256_file(path)


def reason_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["composition_reason"]) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-stream-rows", type=int, default=3200)
    parser.add_argument("--positive-fraction", type=float, default=0.8)
    parser.add_argument("--validation-rows", type=int, default=1024)
    parser.add_argument("--max-negative-candidates", type=int, default=256)
    parser.add_argument("--execution-manifest-sha256", required=True)
    parser.add_argument("--allow-source-sha-mismatch", action="store_true")
    args = parser.parse_args()

    if int(args.train_stream_rows) != 3200:
        raise ValueError("JointChem v1 requires exactly 3200 training microbatches")
    if abs(float(args.positive_fraction) - 0.8) > 1e-12:
        raise ValueError("JointChem v1 requires positive_fraction=0.8")

    source_sha: dict[str, str] = {}
    enriched: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        path = args.input_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        source_sha[split] = observed
        if not args.allow_source_sha_mismatch and observed != EXPECTED_SOURCE_SHA256[split]:
            raise ValueError(
                f"{split} source SHA mismatch: expected {EXPECTED_SOURCE_SHA256[split]}, observed {observed}"
            )
        enriched[split] = enrich_rows(
            path,
            split=split,
            seed=int(args.seed),
            max_negative_candidates=int(args.max_negative_candidates),
        )
        attach_joint_negatives(enriched[split], seed=int(args.seed))

    train_stream = build_training_stream(
        enriched["train"],
        total_rows=int(args.train_stream_rows),
        positive_fraction=float(args.positive_fraction),
        seed=int(args.seed),
    )
    validation_panel = build_validation_panel(
        enriched["val"],
        count=int(args.validation_rows),
        seed=int(args.seed),
    )
    ledger = [
        {
            "schema": "h1a2_jointchem_row_ledger_v1",
            "split": split,
            "row_idx": row["row_idx"],
            "source_line_sha256": row["source_line_sha256"],
            "formula": row["formula"],
            "N": row["N"],
            "arity": row["arity"],
            "composition_valid": row["composition_valid"],
            "composition_reason": row["composition_reason"],
        }
        for split in ("train", "val", "test")
        for row in enriched[split]
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_sha = {
        "train_stream": write_jsonl(args.output_dir / "train.jsonl", train_stream),
        "validation_panel": write_jsonl(args.output_dir / "val.jsonl", validation_panel),
        "row_ledger": write_jsonl(args.output_dir / "row_ledger.jsonl", ledger),
    }
    manifest = {
        "schema": "h1a2_jointchem_data_manifest_v1",
        "seed": int(args.seed),
        "source_sha256": source_sha,
        "expected_source_sha256": EXPECTED_SOURCE_SHA256,
        "source_rows": {split: len(rows) for split, rows in enriched.items()},
        "source_reason_counts": {split: reason_counts(rows) for split, rows in enriched.items()},
        "train_stream_rows": len(train_stream),
        "train_stream_role_counts": dict(
            sorted(Counter(row["stream_role"] for row in train_stream).items())
        ),
        "train_stream_reason_counts": reason_counts(train_stream),
        "all_metal_positive_fraction": (
            sum(
                row["stream_role"] == "chemistry_valid_positive"
                and row["composition_reason"] == "all_metal_shortcut"
                for row in train_stream
            )
            / max(
                1,
                sum(row["stream_role"] == "chemistry_valid_positive" for row in train_stream),
            )
        ),
        "chemistry_negative_available": sum(
            row["chemistry_negative_answer"] is not None for row in train_stream
        ),
        "joint_negative_available": sum(
            row["joint_negative_answer"] is not None for row in train_stream
        ),
        "validation_rows": len(validation_panel),
        "positive_fraction": float(args.positive_fraction),
        "max_negative_candidates": int(args.max_negative_candidates),
        "execution_manifest_sha256": str(args.execution_manifest_sha256),
        "forbidden_marker_scan_passed": True,
        "retry_or_replacement_used": False,
        "energy_or_stability_label_used": False,
        "output_sha256": output_sha,
    }
    assert_no_leakage(manifest)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    success = {
        "schema": "h1a2_jointchem_data_success_v1",
        "manifest_sha256": sha256_file(manifest_path),
        "complete": True,
    }
    (args.output_dir / "_SUCCESS").write_text(
        json.dumps(success, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
