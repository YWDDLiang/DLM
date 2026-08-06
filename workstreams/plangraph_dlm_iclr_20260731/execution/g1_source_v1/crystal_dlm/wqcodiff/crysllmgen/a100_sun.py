"""Attempt-preserving adapter for the frozen R5-C A100 S.U.N. evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from ..contracts import write_json_exclusive
from .epoch_training import sha256_file


EVAL_SUN_SHA256 = "564b4490f01464012277653951f8a55b5c1575bc78091f5a06db25ca9339852b"
EVAL_SUN_RESUMABLE_SHA256 = (
    "44c7d9adf01de29d5bdd0eb6a0e6e5d77f1b47b47f0e4bcd9b35cd6c51e19baa"
)
MP20_TRAIN_CSV_SHA256 = (
    "9b8031cf4ea7bb62709c74735da7ec11d00e367c5eaa05658fad5b5e7a530dde"
)
MP20_TRAINING_INDEX_CACHE_SHA256 = (
    "f26ea30d6f529cca2d743401049e0328227faa156c95af3e0641d35fe03ffc62"
)
MP_HULL_CACHE_SHA256 = (
    "93d6532cd93c1cfebcbc969d0299852359d6a2950b66b259c028e971f8f7e4ff"
)
CHGNET_RELAX_CACHE_SHA256 = (
    "9d29489fcf61544ed2420fee7f78c6d59a0dca4bc55f989223bb7000f4961c71"
)
CHGNET_0P3P0_SHA256 = (
    "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1"
)
STRICT_THRESHOLD = 0.0
META_LIKE_THRESHOLD = 0.1


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not a mapping")
            rows.append(value)
    return rows


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _expect_sha(path: str | Path, expected: str, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = sha256_file(resolved)
    if observed != expected:
        raise ValueError(f"{label} changed: expected={expected}, observed={observed}")
    return resolved


def _expected_training_index_cache(train_csv: Path) -> Path:
    """Reproduce the frozen evaluator's exact cache-path fingerprint."""

    stat = train_csv.stat()
    payload = f"{train_csv.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    fingerprint = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return (
        train_csv.parent
        / ".cache"
        / f"{train_csv.name}.{fingerprint}.training_index.pkl"
    )


def verify_a100_assets(
    *,
    eval_sun_py: str | Path,
    eval_sun_resumable_py: str | Path,
    train_csv: str | Path,
    training_index_cache: str | Path,
    mp_hull_cache: str | Path,
    chgnet_relax_cache: str | Path,
    chgnet_model_asset: str | Path,
    chgnet_runtime_checkpoint: str | Path,
) -> dict[str, dict[str, Any]]:
    """Verify the exact historical scripts, references, caches, and model."""

    train_csv_path = _expect_sha(
        train_csv, MP20_TRAIN_CSV_SHA256, "MP20 train CSV"
    )
    training_index_path = _expect_sha(
        training_index_cache,
        MP20_TRAINING_INDEX_CACHE_SHA256,
        "MP20 training-index cache",
    )
    if training_index_path != _expected_training_index_cache(train_csv_path).resolve():
        raise ValueError(
            "MP20 training-index cache is not the exact cache selected by "
            "the frozen A100 evaluator"
        )
    paths = {
        "eval_sun_py": _expect_sha(eval_sun_py, EVAL_SUN_SHA256, "eval_sun.py"),
        "eval_sun_resumable_py": _expect_sha(
            eval_sun_resumable_py,
            EVAL_SUN_RESUMABLE_SHA256,
            "eval_sun_resumable.py",
        ),
        "train_csv": train_csv_path,
        "training_index_cache": training_index_path,
        "mp_hull_cache": _expect_sha(
            mp_hull_cache, MP_HULL_CACHE_SHA256, "MP hull cache"
        ),
        "chgnet_relax_cache": _expect_sha(
            chgnet_relax_cache, CHGNET_RELAX_CACHE_SHA256, "CHGNet relax cache"
        ),
        "chgnet_model_asset": _expect_sha(
            chgnet_model_asset, CHGNET_0P3P0_SHA256, "CHGNet model asset"
        ),
        "chgnet_runtime_checkpoint": _expect_sha(
            chgnet_runtime_checkpoint,
            CHGNET_0P3P0_SHA256,
            "CHGNet runtime checkpoint",
        ),
    }
    return {name: _identity(path) for name, path in paths.items()}


def _stable_structure_hash(structure: Any) -> str:
    payload = json.dumps(
        structure.as_dict(),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_a100_input(
    *,
    generation_jsonl: str | Path,
    output_pt: str | Path,
    output_manifest: str | Path,
    expected_attempts: int,
) -> dict[str, Any]:
    """Convert every generation attempt to the legacy A100 ``.pt`` layout.

    Failed attempts receive a deliberately zero-volume placeholder.  The frozen
    loader therefore keeps ``Total generated`` equal to the attempt denominator
    while omitting the placeholder from its reconstructed survivor list.
    """

    import torch
    from pymatgen.core import Structure

    generation_path = Path(generation_jsonl).resolve()
    pt_path = Path(output_pt).resolve()
    manifest_path = Path(output_manifest).resolve()
    for path in (pt_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    rows = _jsonl(generation_path)
    if len(rows) != int(expected_attempts):
        raise ValueError("A100 input must retain the registered attempt denominator")
    attempt_ids = [str(row.get("attempt_id", "")) for row in rows]
    if any(not value for value in attempt_ids) or len(set(attempt_ids)) != len(rows):
        raise ValueError("generation attempt IDs are missing or duplicated")
    methods = {str(row.get("method", "")) for row in rows}
    if len(methods) != 1 or not next(iter(methods)):
        raise ValueError("A100 input must contain exactly one method")
    if any(bool(row.get("retry_or_replacement_used")) for row in rows):
        raise ValueError("retry/replacement evidence cannot enter A100 selection")

    frac_blocks: list[Any] = []
    atom_blocks: list[Any] = []
    num_atoms: list[int] = []
    lengths: list[list[float]] = []
    angles: list[list[float]] = []
    attempt_records: list[dict[str, Any]] = []
    reconstructed_index = 0
    for ordinal, row in enumerate(rows):
        attempt_id = attempt_ids[ordinal]
        structure = None
        reason = ""
        if row.get("status") != "succeeded" or not isinstance(
            row.get("structure"), Mapping
        ):
            reason = "generation:" + str(row.get("reason", row.get("status", "")))
        else:
            try:
                candidate = Structure.from_dict(dict(row["structure"]))
                scalar_values = [
                    *candidate.lattice.abc,
                    *candidate.lattice.angles,
                    *candidate.frac_coords.reshape(-1).tolist(),
                ]
                if (
                    candidate.num_sites <= 0
                    or candidate.volume < 0.1
                    or any(not math.isfinite(float(value)) for value in scalar_values)
                ):
                    raise ValueError("legacy A100 reconstruction support failure")
                structure = candidate
            except Exception as exc:
                reason = f"reconstruction:{type(exc).__name__}:{exc}"

        if structure is None:
            # One atom advances the legacy loader offset; a zero-volume lattice
            # guarantees that this record never reaches novelty/stability.
            num_atoms.append(1)
            frac_blocks.append(torch.zeros((1, 3), dtype=torch.float32))
            atom_blocks.append(torch.ones((1,), dtype=torch.long))
            lengths.append([0.0, 0.0, 0.0])
            angles.append([90.0, 90.0, 90.0])
            attempt_records.append(
                {
                    "attempt_id": attempt_id,
                    "generation_ordinal": ordinal,
                    "pt_record_ordinal": ordinal,
                    "reconstructed_index": None,
                    "status": "failed",
                    "reason": reason,
                }
            )
            continue

        sites = int(structure.num_sites)
        num_atoms.append(sites)
        frac_blocks.append(
            torch.tensor(structure.frac_coords, dtype=torch.float32)
        )
        atom_blocks.append(
            torch.tensor(structure.atomic_numbers, dtype=torch.long)
        )
        lengths.append([float(value) for value in structure.lattice.abc])
        angles.append([float(value) for value in structure.lattice.angles])
        attempt_records.append(
            {
                "attempt_id": attempt_id,
                "generation_ordinal": ordinal,
                "pt_record_ordinal": ordinal,
                "reconstructed_index": reconstructed_index,
                "status": "succeeded",
                "reason": "",
                "structure_sha256": _stable_structure_hash(structure),
            }
        )
        reconstructed_index += 1

    payload = {
        "frac_coords": [torch.cat(frac_blocks, dim=0)],
        "num_atoms": [torch.tensor(num_atoms, dtype=torch.long)],
        "atom_types": [torch.cat(atom_blocks, dim=0)],
        "lengths": [torch.tensor(lengths, dtype=torch.float32)],
        "angles": [torch.tensor(angles, dtype=torch.float32)],
    }
    with pt_path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    manifest = {
        "schema": "crysllmgen_r5c_a100_input_manifest_v1",
        "method": next(iter(methods)),
        "total_attempts": len(rows),
        "pt_records": len(rows),
        "reconstructed_structures": reconstructed_index,
        "attempt_records": attempt_records,
        "generation_jsonl": str(generation_path),
        "generation_jsonl_sha256": sha256_file(generation_path),
        "generated_pt": str(pt_path),
        "generated_pt_sha256": sha256_file(pt_path),
        "retry_or_replacement_used": False,
    }
    write_json_exclusive(manifest_path, manifest)
    return manifest


_SUMMARY_PATTERNS = {
    "total_generated": r"^Total generated:\s*(\d+)\s*$",
    "reconstructed": r"^Reconstructed:\s*(\d+)\s*$",
    "novel": r"^Novel:\s*(\d+)/(\d+)",
    "unique": r"^Unique:\s*(\d+)/(\d+)",
    "novel_unique": r"^Novel \+ Unique:\s*(\d+)/(\d+)",
    "e_hull_evaluated": r"^E_hull evaluated:\s*(\d+)/(\d+)\s*$",
    "e_hull_unknown": r"^E_hull unknown:\s*(\d+)/(\d+)\s*$",
    "stable": r"^Stable:\s*(\d+)/(\d+) evaluated",
    "full_sun_lower_bound_percent": r"^Full S\.U\.N\. lower-bound:\s*([0-9.]+)%\s*$",
    "coverage_adjusted_percent": r"^Coverage-adjusted S\.U\.N\. estimate:\s*([0-9.]+)%\s*$",
}


def parse_a100_summary(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    parsed: dict[str, Any] = {}
    for key, pattern in _SUMMARY_PATTERNS.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match is None:
            raise ValueError(f"A100 summary is missing {key}")
        values = match.groups()
        if key.endswith("_percent"):
            parsed[key] = float(values[0])
        elif len(values) == 1:
            parsed[key] = int(values[0])
        else:
            parsed[key] = {"numerator": int(values[0]), "denominator": int(values[1])}
    return parsed


def _load_eval_sun(path: Path) -> ModuleType:
    name = "_wqcodiff_frozen_r5c_a100_eval_sun"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def aggregate_a100_outputs(
    *,
    input_manifest: str | Path,
    strict_summary: str | Path,
    meta_summary: str | Path,
    strict_relax_results: str | Path,
    meta_relax_results: str | Path,
    eval_sun_py: str | Path,
    eval_sun_resumable_py: str | Path,
    train_csv: str | Path,
    training_index_cache: str | Path,
    source_mp_hull_cache: str | Path,
    source_chgnet_relax_cache: str | Path,
    working_mp_hull_cache: str | Path,
    chgnet_model_asset: str | Path,
    chgnet_runtime_checkpoint: str | Path,
    output_jsonl: str | Path,
    output_summary: str | Path,
    base_source_bundle_sha256: str,
    execution_patch_sha256: str,
) -> dict[str, Any]:
    """Map exact A100 survivor calculations back to every attempt."""

    for label, value in (
        ("base source bundle", base_source_bundle_sha256),
        ("execution patch", execution_patch_sha256),
    ):
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{label} must be one lowercase SHA256")

    assets = verify_a100_assets(
        eval_sun_py=eval_sun_py,
        eval_sun_resumable_py=eval_sun_resumable_py,
        train_csv=train_csv,
        training_index_cache=training_index_cache,
        mp_hull_cache=source_mp_hull_cache,
        chgnet_relax_cache=source_chgnet_relax_cache,
        chgnet_model_asset=chgnet_model_asset,
        chgnet_runtime_checkpoint=chgnet_runtime_checkpoint,
    )
    manifest_path = Path(input_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "crysllmgen_r5c_a100_input_manifest_v1":
        raise ValueError("invalid A100 input manifest")
    if manifest.get("retry_or_replacement_used"):
        raise ValueError("A100 input manifest used retry/replacement")
    generated_pt = Path(str(manifest["generated_pt"])).resolve()
    if sha256_file(generated_pt) != manifest.get("generated_pt_sha256"):
        raise ValueError("A100 generated .pt changed")

    strict = parse_a100_summary(strict_summary)
    meta = parse_a100_summary(meta_summary)
    total_attempts = int(manifest["total_attempts"])
    reconstructed_expected = int(manifest["reconstructed_structures"])
    for label, summary in (("strict", strict), ("meta", meta)):
        if summary["total_generated"] != total_attempts:
            raise ValueError(f"{label} A100 total-generated denominator changed")
        if summary["reconstructed"] != reconstructed_expected:
            raise ValueError(f"{label} A100 reconstruction count changed")

    evaluator = _load_eval_sun(Path(eval_sun_py).resolve())
    structures, loader_total = evaluator.load_generated_structures(generated_pt)
    if loader_total != total_attempts or len(structures) != reconstructed_expected:
        raise ValueError("frozen A100 loader disagrees with input manifest")
    train_structures, train_formula_idx = evaluator.load_training_index(
        Path(train_csv).resolve()
    )
    matcher = evaluator.StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    novel_mask = evaluator.compute_novelty(
        structures, train_structures, train_formula_idx, matcher
    )
    eq_class, n_unique = evaluator.compute_uniqueness(structures, matcher)
    seen_classes: set[int] = set()
    novel_unique_indices: list[int] = []
    unique_representatives: set[int] = set()
    for index in range(len(structures)):
        cls = int(eq_class[index])
        if cls in seen_classes:
            continue
        seen_classes.add(cls)
        unique_representatives.add(index)
        if bool(novel_mask[index]):
            novel_unique_indices.append(index)

    if (
        strict["novel"]["numerator"] != int(sum(bool(value) for value in novel_mask))
        or strict["unique"]["numerator"] != int(n_unique)
        or strict["novel_unique"]["numerator"] != len(novel_unique_indices)
        or meta["novel"]["numerator"] != strict["novel"]["numerator"]
        or meta["unique"]["numerator"] != strict["unique"]["numerator"]
        or meta["novel_unique"]["numerator"] != len(novel_unique_indices)
    ):
        raise ValueError("exact A100 novelty/uniqueness parity failed")

    from pymatgen.core import Composition

    relax_rows = _jsonl(strict_relax_results)
    meta_relax_rows = _jsonl(meta_relax_results)
    if relax_rows != meta_relax_rows:
        raise ValueError("strict/meta A100 relaxation records differ")
    relax_by_index: dict[int, tuple[float | None, Any]] = {}
    for row in relax_rows:
        local_index = int(row["local_index"])
        if local_index in relax_by_index:
            raise ValueError("duplicate A100 relaxation local index")
        energy_value = row.get("energy_per_atom")
        energy = None if energy_value is None else float(energy_value)
        relax_by_index[local_index] = (energy, Composition(row["composition"]))
    if set(relax_by_index) != set(range(len(novel_unique_indices))):
        raise ValueError("A100 relaxation records are incomplete")
    energies_comps = [relax_by_index[index] for index in range(len(relax_by_index))]
    e_hull_results = evaluator.compute_e_hull_batch(
        energies_comps,
        Path(train_csv).resolve(),
        mp_api_key=None,
        mp_cache_path=Path(working_mp_hull_cache).resolve(),
    )

    stability_by_reconstructed: dict[int, dict[str, Any]] = {}
    for local_index, reconstructed_index in enumerate(novel_unique_indices):
        energy, composition = energies_comps[local_index]
        e_hull = None
        if energy is not None:
            value = e_hull_results.get((energy, composition.reduced_formula))
            if value is not None and math.isfinite(float(value)):
                e_hull = float(value)
        stability_by_reconstructed[reconstructed_index] = {
            "energy_per_atom": energy,
            "e_above_hull": e_hull,
            "strict_full_sun": e_hull is not None and e_hull <= STRICT_THRESHOLD,
            "meta_full_sun": e_hull is not None and e_hull <= META_LIKE_THRESHOLD,
        }

    strict_count = sum(
        int(value["strict_full_sun"]) for value in stability_by_reconstructed.values()
    )
    meta_count = sum(
        int(value["meta_full_sun"]) for value in stability_by_reconstructed.values()
    )
    if strict["stable"]["numerator"] != strict_count:
        raise ValueError("strict A100 stable numerator parity failed")
    if meta["stable"]["numerator"] != meta_count:
        raise ValueError("meta-like A100 stable numerator parity failed")
    if meta_count < strict_count:
        raise ValueError("meta-like A100 stable set is smaller than strict")
    for label, summary, count in (
        ("strict", strict, strict_count),
        ("meta", meta, meta_count),
    ):
        expected_percent = 100.0 * count / len(structures)
        if not math.isclose(
            float(summary["full_sun_lower_bound_percent"]),
            expected_percent,
            rel_tol=0.0,
            abs_tol=0.0050000001,
        ):
            raise ValueError(f"{label} A100 legacy lower-bound percentage changed")

    mapping = {
        int(row["reconstructed_index"]): row
        for row in manifest["attempt_records"]
        if row.get("reconstructed_index") is not None
    }
    if set(mapping) != set(range(len(structures))):
        raise ValueError("A100 reconstructed-index mapping is incomplete")
    rows_by_attempt = {
        str(row["attempt_id"]): row for row in manifest["attempt_records"]
    }
    if len(rows_by_attempt) != total_attempts:
        raise ValueError("A100 attempt mapping is incomplete")
    reconstructed_by_attempt = {
        str(row["attempt_id"]): int(row["reconstructed_index"])
        for row in manifest["attempt_records"]
        if row.get("reconstructed_index") is not None
    }

    output_path = Path(output_jsonl).resolve()
    summary_path = Path(output_summary).resolve()
    for path in (output_path, summary_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    unknown_hull = 0
    with output_path.open("x", encoding="utf-8") as handle:
        for attempt in sorted(
            manifest["attempt_records"], key=lambda value: int(value["generation_ordinal"])
        ):
            attempt_id = str(attempt["attempt_id"])
            reconstructed_index = reconstructed_by_attempt.get(attempt_id)
            metrics: dict[str, Any] = {
                "novel": False,
                "unique_representative": False,
                "novel_unique": False,
                "strict_full_sun": False,
                "meta_full_sun": False,
                "energy_per_atom": None,
                "e_above_hull": None,
            }
            evaluation_status = "generation_or_reconstruction_failed"
            if reconstructed_index is not None:
                is_novel = bool(novel_mask[reconstructed_index])
                is_representative = reconstructed_index in unique_representatives
                is_novel_unique = reconstructed_index in stability_by_reconstructed
                metrics.update(
                    {
                        "novel": is_novel,
                        "unique_representative": is_representative,
                        "novel_unique": is_novel_unique,
                    }
                )
                evaluation_status = "not_novel_unique"
                if is_novel_unique:
                    metrics.update(stability_by_reconstructed[reconstructed_index])
                    if metrics["e_above_hull"] is None:
                        evaluation_status = "relaxation_or_hull_unknown"
                        unknown_hull += 1
                    else:
                        evaluation_status = "evaluated"
            record = {
                "schema": "crysllmgen_r5c_a100_sun_attempt_v1",
                "attempt_id": attempt_id,
                "method": manifest["method"],
                "generation_ordinal": int(attempt["generation_ordinal"]),
                "generation_status": attempt["status"],
                "generation_reason": attempt.get("reason", ""),
                "evaluation_status": evaluation_status,
                "metrics": metrics,
                "base_source_bundle_sha256": base_source_bundle_sha256,
                "execution_patch_sha256": execution_patch_sha256,
                "retry_or_replacement_used": False,
            }
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    result = {
        "schema": "crysllmgen_r5c_a100_sun_summary_v1",
        "ok": True,
        "method": manifest["method"],
        "base_source_bundle_sha256": base_source_bundle_sha256,
        "execution_patch_sha256": execution_patch_sha256,
        "definition": "frozen R5-C A100 CHGNet-0.3.0 plus MP hull cache",
        "thresholds_ev_per_atom": {
            "strict": STRICT_THRESHOLD,
            "meta_like": META_LIKE_THRESHOLD,
        },
        "counts": {
            "total_attempts": total_attempts,
            "reconstructed": len(structures),
            "novel": int(sum(bool(value) for value in novel_mask)),
            "unique": int(n_unique),
            "novel_unique": len(novel_unique_indices),
            "strict_full_sun": strict_count,
            "meta_full_sun": meta_count,
            "relaxation_or_hull_unknown": unknown_hull,
        },
        "rates": {
            "attempt_strict_full_sun_lower_bound": strict_count / total_attempts,
            "attempt_meta_full_sun_lower_bound": meta_count / total_attempts,
            "attempt_novel_unique": len(novel_unique_indices) / total_attempts,
        },
        "exact_legacy_r5c_a100": {
            "strict": strict,
            "meta_like": meta,
            "denominator": "reconstructed_structures",
            "selection_role": "report_only",
        },
        "coverage_adjusted_selection_role": "report_only_never_checkpoint_selection",
        "denominator": "all_generation_attempts",
        "assets": assets,
        "working_mp_hull_cache": _identity(working_mp_hull_cache),
        "input_manifest": _identity(manifest_path),
        "strict_summary": _identity(strict_summary),
        "meta_summary": _identity(meta_summary),
        "strict_relax_results": _identity(strict_relax_results),
        "meta_relax_results": _identity(meta_relax_results),
        "attempt_results": _identity(output_path),
        "retry_or_replacement_used": False,
    }
    write_json_exclusive(summary_path, result)
    return result
