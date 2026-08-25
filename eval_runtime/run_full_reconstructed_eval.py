#!/usr/bin/env python3
"""Compute exact frozen N/U and CHGNet energy for every reconstructed row.

Unlike the legacy S.U.N. wrapper, this module does not use N/U as a gate for
relaxation. Official E_hull is joined later, then S, N, and U are intersected.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import protocol


def _load(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--r03e-root", type=Path, required=True)
    parser.add_argument("--working-relax-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    arm = protocol.validate_arm(args.arm)
    repeat = protocol.validate_repeat(args.repeat)
    config = protocol.read_json(args.config.resolve())
    protocol.validate_config(config)
    assets = config["assets"]
    frozen = config["frozen_code"]
    generation_path = args.generation_jsonl.resolve()
    generation = protocol.ordered_rows(
        protocol.read_jsonl(generation_path), ordinal_field="ordinal"
    )
    if (
        {str(row.get("arm")) for row in generation} != {arm}
        or {int(row.get("repeat", -1)) for row in generation} != {repeat}
        or any(row.get("retry_or_replacement_used") is not False for row in generation)
    ):
        raise ValueError("generation ledger contract changed")

    r03e = args.r03e_root.resolve()
    runtime = r03e / "runtime"
    a100_path = runtime / "crystal_dlm/wqcodiff/crysllmgen/a100_sun.py"
    protocol.require_file(
        a100_path,
        frozen["a100_adapter_sha256"],
        "frozen attempt-preserving A100 adapter",
    )
    sys.path.insert(0, str(runtime))
    from crystal_dlm.wqcodiff.crysllmgen import a100_sun  # noqa: PLC0415

    eval_sun_path = protocol.require_file(
        assets["eval_sun_py"], frozen["eval_sun_sha256"], "frozen eval_sun.py"
    )
    resumable_path = protocol.require_file(
        assets["eval_sun_resumable_py"],
        frozen["eval_sun_resumable_sha256"],
        "frozen eval_sun_resumable.py",
    )
    train_csv = protocol.require_file(
        assets["train_csv"], a100_sun.MP20_TRAIN_CSV_SHA256, "MP20 train CSV"
    )
    protocol.require_file(
        assets["training_index_cache"],
        a100_sun.MP20_TRAINING_INDEX_CACHE_SHA256,
        "frozen training index cache",
    )
    base_cache = protocol.require_file(
        assets["base_chgnet_relax_cache"],
        a100_sun.CHGNET_RELAX_CACHE_SHA256,
        "frozen base CHGNet cache",
    )
    protocol.require_file(
        assets["chgnet_model_asset"],
        a100_sun.CHGNET_0P3P0_SHA256,
        "CHGNet 0.3.0 model asset",
    )
    protocol.require_file(
        assets["chgnet_runtime_checkpoint"],
        a100_sun.CHGNET_0P3P0_SHA256,
        "CHGNet runtime checkpoint",
    )
    working_cache = args.working_relax_cache.resolve()
    if not working_cache.is_file() or working_cache.stat().st_size < base_cache.stat().st_size:
        raise ValueError("working CHGNet cache is absent or smaller than frozen base")
    working_cache_sha_before = protocol.sha256_file(working_cache)

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    input_pt = output / "all_attempts.pt"
    input_manifest_path = output / "input_manifest.json"
    manifest = a100_sun.prepare_a100_input(
        generation_jsonl=generation_path,
        output_pt=input_pt,
        output_manifest=input_manifest_path,
        expected_attempts=protocol.RAW_DENOMINATOR,
    )

    eval_sun = _load(eval_sun_path, "eval_sun")
    resumable = _load(resumable_path, "eval_sun_resumable_full_reconstructed")
    structures, loader_total = eval_sun.load_generated_structures(input_pt)
    if (
        loader_total != protocol.RAW_DENOMINATOR
        or len(structures) != int(manifest["reconstructed_structures"])
    ):
        raise ValueError("frozen loader disagrees with all-attempt manifest")
    train_structures, train_formula_idx = eval_sun.load_training_index(train_csv)
    matcher = eval_sun.StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    novel_mask = eval_sun.compute_novelty(
        structures, train_structures, train_formula_idx, matcher
    )
    eq_class, n_unique = eval_sun.compute_uniqueness(structures, matcher)
    seen_classes: set[int] = set()
    unique_representatives: set[int] = set()
    for index in range(len(structures)):
        class_id = int(eq_class[index])
        if class_id not in seen_classes:
            seen_classes.add(class_id)
            unique_representatives.add(index)
    if len(unique_representatives) != int(n_unique):
        raise ValueError("frozen uniqueness representative count changed")

    relax_path = output / "all_reconstructed_relax_results.jsonl"
    energies_compositions = resumable.relax_missing(
        structures,
        relax_path,
        "cuda",
        working_cache,
    )
    if len(energies_compositions) != len(structures):
        raise ValueError("full reconstructed relaxation ledger is incomplete")

    reconstructed_by_attempt = {
        str(row["attempt_id"]): int(row["reconstructed_index"])
        for row in manifest["attempt_records"]
        if row.get("reconstructed_index") is not None
    }
    if set(reconstructed_by_attempt) - {str(row["attempt_id"]) for row in generation}:
        raise ValueError("manifest contains an unknown generation attempt")
    labels: list[dict[str, Any]] = []
    for ordinal, generation_row in enumerate(generation):
        attempt_id = str(generation_row["attempt_id"])
        reconstructed_index = reconstructed_by_attempt.get(attempt_id)
        record: dict[str, Any] = {
            "schema": "h1_full_reconstructed_preofficial_attempt_v1",
            "repeat": repeat,
            "arm": arm,
            "ordinal": ordinal,
            "attempt_id": attempt_id,
            "generation_status": generation_row["status"],
            "reconstructed": reconstructed_index is not None,
            "reconstructed_index": reconstructed_index,
            "novel": False,
            "unique_representative": False,
            "novel_unique": False,
            "chgnet_energy_per_atom": None,
            "chgnet_composition": None,
            "reduced_formula": None,
            "chemsys": None,
            "chgnet_relaxation_known": False,
            "official_hull_pending": reconstructed_index is not None,
            "retry_or_replacement_used": False,
        }
        if reconstructed_index is not None:
            energy_value, composition = energies_compositions[reconstructed_index]
            energy = None if energy_value is None else float(energy_value)
            if energy is not None and not math.isfinite(energy):
                raise ValueError("CHGNet returned a nonfinite energy")
            is_novel = bool(novel_mask[reconstructed_index])
            is_unique = reconstructed_index in unique_representatives
            record.update(
                {
                    "novel": is_novel,
                    "unique_representative": is_unique,
                    "novel_unique": is_novel and is_unique,
                    "chgnet_energy_per_atom": energy,
                    "chgnet_composition": composition.as_dict(),
                    "reduced_formula": composition.reduced_formula,
                    "chemsys": "-".join(
                        sorted(element.symbol for element in composition.elements)
                    ),
                    "chgnet_relaxation_known": energy is not None,
                }
            )
        labels.append(record)

    label_path = output / "attempt_labels_preofficial.jsonl"
    protocol.write_jsonl_exclusive(label_path, labels)
    reconstructed = len(structures)
    energy_known = sum(row["chgnet_relaxation_known"] for row in labels)
    novel = sum(row["novel"] for row in labels)
    unique = sum(row["unique_representative"] for row in labels)
    novel_unique = sum(row["novel_unique"] for row in labels)
    report = {
        "schema": "h1_full_reconstructed_preofficial_summary_v1",
        "status": "complete",
        "ok": True,
        "repeat": repeat,
        "arm": arm,
        "raw_attempts": protocol.RAW_DENOMINATOR,
        "generation_succeeded": sum(row["status"] == "succeeded" for row in generation),
        "reconstructed": reconstructed,
        "novel": novel,
        "unique_representatives": unique,
        "novel_unique": novel_unique,
        "chgnet_relaxation_known": energy_known,
        "chgnet_relaxation_unknown": reconstructed - energy_known,
        "official_hull_target": reconstructed,
        "stability_scope": "all_reconstructed_before_NU_intersection",
        "legacy_novel_unique_relaxation_gate_used": False,
        "target_hull_known_reconstructed": int(
            config["evaluation"]["target_hull_known_reconstructed_per_cell"]
        ),
        "target_is_nonblocking": True,
        "attempt_labels_sha256": protocol.sha256_file(label_path),
        "relax_results_sha256": protocol.sha256_file(relax_path),
        "input_manifest_sha256": protocol.sha256_file(input_manifest_path),
        "working_relax_cache_sha256_before": working_cache_sha_before,
        "working_relax_cache_sha256_after": protocol.sha256_file(working_cache),
        "base_relax_cache_sha256": protocol.sha256_file(base_cache),
        "eval_sun_sha256": protocol.sha256_file(eval_sun_path),
        "eval_sun_resumable_sha256": protocol.sha256_file(resumable_path),
        "retry_replacement_repair_filter_rerank": False,
    }
    protocol.write_json_exclusive(output / "summary.json", report)
    (output / "_SUCCESS").touch(exist_ok=False)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
