#!/usr/bin/env python3
"""Augment frozen listwise groups with raw-CIF structural safety labels."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence
import warnings


SCHEMA = "h1a2_listwise_safety_group_v2"
MANIFEST_SCHEMA = "h1a2_listwise_safety_manifest_v2"
MINIMUM_DISTANCE = 0.5
MINIMUM_VOLUME = 0.1


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def composition_counts(identity: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in str(identity).split("|"):
        symbol, separator, raw_count = item.partition(":")
        if not separator or not symbol:
            raise ValueError(f"malformed composition identity: {identity}")
        count = int(raw_count)
        if count <= 0 or symbol in counts:
            raise ValueError(f"malformed composition identity: {identity}")
        counts[symbol] = count
    if not counts:
        raise ValueError("empty composition identity")
    return counts


def raw_structure_safety(task: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "raw_parse": False,
        "raw_composition_exact": False,
        "raw_min_periodic_distance": None,
        "raw_volume": None,
        "raw_structure_valid": False,
        "raw_direct_joint_valid": False,
        "raw_energy_known": False,
        "raw_energy_per_atom": None,
        "raw_missing_reason": None,
    }
    try:
        import numpy as np
        from pymatgen.core import Structure

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = Structure.from_str(str(task["cif"]), fmt="cif")
        expected = composition_counts(str(task["composition_id"]))
        observed_float = structure.composition.get_el_amt_dict()
        observed = {
            str(symbol): int(round(float(count)))
            for symbol, count in observed_float.items()
        }
        exact = (
            observed == expected
            and all(abs(float(observed_float[symbol]) - expected[symbol]) < 1e-6 for symbol in expected)
        )
        if len(structure) <= 1:
            minimum_distance = float("inf")
        else:
            distances = np.asarray(structure.distance_matrix, dtype=np.float64)
            distances = distances + np.eye(len(structure)) * (MINIMUM_DISTANCE + 10.0)
            minimum_distance = float(distances.min())
        volume = float(structure.volume)
        structure_valid = (
            math.isfinite(minimum_distance)
            and minimum_distance >= MINIMUM_DISTANCE
            and math.isfinite(volume)
            and volume >= MINIMUM_VOLUME
        ) or (
            math.isinf(minimum_distance)
            and math.isfinite(volume)
            and volume >= MINIMUM_VOLUME
        )
        result.update(
            {
                "raw_parse": True,
                "raw_composition_exact": bool(exact),
                "raw_min_periodic_distance": (
                    None if math.isinf(minimum_distance) else minimum_distance
                ),
                "raw_volume": volume,
                "raw_structure_valid": bool(structure_valid),
                "raw_direct_joint_valid": bool(exact and structure_valid),
            }
        )
        if not exact:
            result["raw_missing_reason"] = "composition_mismatch"
        elif not structure_valid:
            result["raw_missing_reason"] = "structure_invalid"
        return result
    except Exception as exc:  # noqa: BLE001 - failures remain explicit labels.
        result["raw_missing_reason"] = f"{type(exc).__name__}:{str(exc)[:300]}"
        return result


def _source_name_from_role(role: str) -> str | None:
    text = str(role)
    if text.startswith("noisy_stream") and text.endswith("_body"):
        return text[: -len("_body")]
    if text.startswith("retired_l7_") and text.endswith("_body"):
        return "l7_" + text[len("retired_l7_") : -len("_body")]
    return None


def load_source_index(source_records: Sequence[Mapping[str, Any]]) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    verified: list[dict[str, Any]] = []
    for record in source_records:
        path = Path(str(record["path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_sha = sha256_file(path)
        if observed_sha != str(record["sha256"]):
            raise ValueError(f"frozen source changed: {path}")
        verified.append(
            {
                "role": str(record["role"]),
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": observed_sha,
            }
        )
        source_name = _source_name_from_role(str(record["role"]))
        if source_name is None:
            continue
        ordinal_field = "sample_idx" if source_name.startswith("noisy_stream") else "ordinal"
        for row in read_jsonl(path):
            key = (source_name, int(row[ordinal_field]))
            if key in index:
                raise ValueError(f"duplicate raw source identity: {key}")
            index[key] = row
    return index, verified


def augment_groups(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    source_index: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    workers: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not 1 <= int(workers) <= 48:
        raise ValueError("workers must lie in 1..48")
    tasks: list[dict[str, Any]] = []
    locations: list[tuple[str, int, int]] = []
    copied: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split in ("train", "validation"):
        for group_index, raw_group in enumerate(rows_by_split[split]):
            group = json.loads(json.dumps(raw_group))
            group["schema"] = SCHEMA
            copied[split].append(group)
            for candidate_index, candidate in enumerate(group["candidates"]):
                source = str(candidate["source"])
                source_ordinal = candidate.get("source_ordinal")
                if source_ordinal is None:
                    raise ValueError("candidate source ordinal is missing")
                key = (source, int(source_ordinal))
                raw = source_index.get(key)
                if raw is None:
                    raise KeyError(f"raw source not found: {key}")
                answer = str(raw.get("text") or "")
                cif = str(raw.get("cif") or "")
                if hashlib.sha256(answer.encode("utf-8")).hexdigest() != str(candidate["answer_sha256"]):
                    raise ValueError(f"candidate answer identity changed: {key}")
                if hashlib.sha256(cif.encode("utf-8")).hexdigest() != str(candidate["cif_sha256"]):
                    raise ValueError(f"candidate CIF identity changed: {key}")
                tasks.append(
                    {
                        "cif": cif,
                        "composition_id": str(group["composition_id"]),
                    }
                )
                locations.append((split, group_index, candidate_index))
    if int(workers) == 1:
        safety_rows = [raw_structure_safety(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            safety_rows = list(executor.map(raw_structure_safety, tasks, chunksize=16))
    for (split, group_index, candidate_index), safety in zip(locations, safety_rows):
        copied[split][group_index]["candidates"][candidate_index].update(safety)

    audit = Counter()
    for split in ("train", "validation"):
        for group in copied[split]:
            candidates = group["candidates"]
            order = sorted(
                range(len(candidates)),
                key=lambda index: (
                    not bool(candidates[index]["raw_direct_joint_valid"]),
                    float(candidates[index]["post_model494_energy_per_atom"]),
                    str(candidates[index]["answer_sha256"]),
                ),
            )
            valid = [index for index in order if candidates[index]["raw_direct_joint_valid"]]
            best_valid = valid[0] if valid else None
            for rank, index in enumerate(order):
                candidates[index]["safety_rank"] = int(rank)
                candidates[index]["refined_energy_rank"] = int(
                    sorted(
                        range(len(candidates)),
                        key=lambda other: (
                            float(candidates[other]["post_model494_energy_per_atom"]),
                            str(candidates[other]["answer_sha256"]),
                        ),
                    ).index(index)
                )
                candidates[index]["is_best_valid_anchor"] = index == best_valid
            group["best_valid_candidate_index"] = best_valid
            group["raw_valid_count"] = len(valid)
            group["raw_invalid_count"] = len(candidates) - len(valid)
            group["raw_energy_known_count"] = 0
            group["safety_policy"] = (
                "raw Direct joint valid before invalid; valid candidates ordered by "
                "post-model494 energy; answer SHA tie break"
            )
            group["safety_candidate_set_sha256"] = canonical_sha256(
                [
                    {
                        "answer_sha256": candidate["answer_sha256"],
                        "raw_direct_joint_valid": candidate["raw_direct_joint_valid"],
                        "refined_energy": candidate["post_model494_energy_per_atom"],
                        "safety_rank": candidate["safety_rank"],
                    }
                    for candidate in candidates
                ]
            )
            audit[f"{split}_groups"] += 1
            audit[f"{split}_candidates"] += len(candidates)
            audit[f"{split}_raw_valid"] += len(valid)
            audit[f"{split}_groups_without_valid_anchor"] += int(best_valid is None)
            for candidate in candidates:
                audit[f"{split}_raw_parse"] += int(candidate["raw_parse"])
                if candidate.get("raw_missing_reason"):
                    audit[f"{split}_missing:{candidate['raw_missing_reason']}"] += 1
    return copied, dict(sorted(audit.items()))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def build_dataset(listwise_dir: Path, output_dir: Path, *, workers: int) -> dict[str, Any]:
    source = listwise_dir.resolve()
    manifest_path = source / "D3PO_LISTWISE_GROUP_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_outputs = manifest["output_hashes"]
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation"):
        path = source / f"{split}.jsonl"
        if sha256_file(path) != str(expected_outputs[split]):
            raise ValueError(f"frozen listwise {split} changed")
        rows_by_split[split] = read_jsonl(path)
    if manifest.get("main_holdout_outcomes_read") is not False or manifest.get("sealed_holdout_outcomes_read") is not False:
        raise ValueError("listwise source read a forbidden holdout")
    source_index, verified_sources = load_source_index(manifest["source_files"])
    augmented, audit = augment_groups(rows_by_split, source_index, workers=workers)
    final = output_dir.resolve()
    if final.exists():
        raise FileExistsError(final)
    preparing = final.with_name(final.name + ".preparing")
    failed = final.with_name(final.name + ".FAILED")
    if preparing.exists() or failed.exists():
        raise FileExistsError(preparing if preparing.exists() else failed)
    preparing.mkdir(parents=True, exist_ok=False)
    try:
        output_hashes = {
            split: _write_jsonl(preparing / f"{split}.jsonl", rows)
            for split, rows in augmented.items()
        }
        output_manifest = {
            "schema": MANIFEST_SCHEMA,
            "minimum_distance_angstrom": MINIMUM_DISTANCE,
            "minimum_volume_angstrom3": MINIMUM_VOLUME,
            "raw_energy_source": "unavailable; explicit missing",
            "raw_invalid_is_lexicographically_worst": True,
            "group_weight": 1.0,
            "source_listwise": {
                "path": str(source),
                "manifest_sha256": sha256_file(manifest_path),
                "train_sha256": expected_outputs["train"],
                "validation_sha256": expected_outputs["validation"],
            },
            "verified_source_files": verified_sources,
            "main_holdout_outcomes_read": False,
            "sealed_holdout_outcomes_read": False,
            "prospective_outcomes_read": False,
            "workers": int(workers),
            "audit": audit,
            "code_sha256": sha256_file(Path(__file__).resolve()),
            "output_hashes": output_hashes,
            "gpu_jobs_used": 0,
        }
        output_manifest_path = preparing / "LISTWISE_SAFETY_V2_MANIFEST.json"
        output_manifest_path.write_text(
            json.dumps(output_manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_sha = sha256_file(output_manifest_path)
        (preparing / "_SUCCESS").write_text(manifest_sha + "\n", encoding="utf-8")
        preparing.rename(final)
        return {**output_manifest, "manifest_sha256": manifest_sha}
    except Exception:
        if preparing.exists():
            shutil.move(str(preparing), str(failed))
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listwise-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_dataset(args.listwise_dir, args.output_dir, workers=args.workers)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
