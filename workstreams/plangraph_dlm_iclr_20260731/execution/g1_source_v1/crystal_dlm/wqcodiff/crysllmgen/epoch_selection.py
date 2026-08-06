"""Validation-only selection across the three mixed-edit epoch checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import write_json_exclusive
from .epoch_training import MixedEditEpochContract, sha256_file


DIRECT_HIGHER = (
    "comp_valid",
    "struct_valid",
    "valid",
    "cov_recall",
    "cov_precision",
)
DIRECT_LOWER = ("wdist_density", "wdist_num_elems")


def _json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence is not a mapping: {path}")
    return payload


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: row is not a mapping")
            rows.append(payload)
    return rows


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _metric(report: Mapping[str, Any], key: str) -> float:
    unchanged = report.get("metrics_unchanged_upstream") or {}
    if key in unchanged:
        value = float(unchanged[key])
    else:
        count_key = {
            "comp_valid": "comp_valid_count",
            "struct_valid": "struct_valid_count",
            "valid": "valid_count",
        }.get(key)
        if count_key is None:
            raise ValueError(f"unchanged CrysLLMGen report is missing {key}")
        value = 100.0 * int(report[count_key]) / int(report["attempts"])
    if not math.isfinite(value):
        raise ValueError(f"non-finite CrysLLMGen metric: {key}")
    return value


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sample")
    position = (len(sorted_values) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower]) * (1.0 - fraction) + float(
        sorted_values[upper]
    ) * fraction


def paired_bootstrap_difference_ci(
    left: Sequence[bool],
    right: Sequence[bool],
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    if len(left) != len(right) or not left:
        raise ValueError("paired SUN vectors must share one nonempty denominator")
    if int(draws) <= 0:
        raise ValueError("bootstrap draws must be positive")
    differences = [float(a) - float(b) for a, b in zip(left, right, strict=True)]
    rng = random.Random(int(seed))
    size = len(differences)
    samples = []
    for _ in range(int(draws)):
        samples.append(
            sum(differences[rng.randrange(size)] for _ in range(size)) / size
        )
    samples.sort()
    point = sum(differences) / size
    return {
        "attempts": size,
        "draws": int(draws),
        "seed": int(seed),
        "difference": point,
        "ci95_lower": _quantile(samples, 0.025),
        "ci95_upper": _quantile(samples, 0.975),
    }


def _candidate(
    entry: Mapping[str, Any],
    *,
    contract: MixedEditEpochContract,
    expected_attempts: int,
    expected_training_seed: int,
    expected_base_source_bundle_sha256: str,
    expected_adapter_training_execution_patch_sha256: str,
    expected_refiner_training_execution_patch_sha256: str,
    expected_evaluation_execution_patch_sha256: str,
    require_separated_execution_identity: bool,
) -> dict[str, Any]:
    required = {
        "logical_epoch",
        "training_report",
        "nll_report",
        "generation_jsonl",
        "crysllmgen_metrics_report",
        "a100_attempt_jsonl",
        "a100_summary",
    }
    if not required <= set(entry):
        raise ValueError("epoch selection candidate lacks registered evidence")
    epoch = int(entry["logical_epoch"])
    if epoch not in contract.required_checkpoint_epochs:
        raise ValueError("unregistered logical epoch")

    training_path = Path(str(entry["training_report"])).resolve()
    training = _json(training_path)
    if (
        training.get("schema") != "crysllmgen_lora_training_report_v1"
        or int(training.get("logical_epoch", -1)) != epoch
        or int(training.get("training_seed", -1)) != expected_training_seed
        or training.get("training_stage") != "mixed_edit"
        or training.get("source_bundle_sha256")
        != expected_base_source_bundle_sha256
        or training.get("execution_patch_sha256")
        != expected_adapter_training_execution_patch_sha256
        or (training.get("training_amendment") or {}).get("sha256")
        != contract.sha256
    ):
        raise ValueError(f"epoch {epoch}: training identity mismatch")
    adapter_sha = str((training.get("model") or {}).get("adapter_sha256", ""))
    if len(adapter_sha) != 64:
        raise ValueError(f"epoch {epoch}: adapter hash is missing")

    nll_path = Path(str(entry["nll_report"])).resolve()
    nll = _json(nll_path)
    nll_adapter = nll.get("adapter") or {}
    if (
        nll.get("schema") != "crysllmgen_lora_validation_nll_v1"
        or not nll.get("ok")
        or int(nll.get("training_seed", -1)) != expected_training_seed
        or nll.get("training_stage") != "mixed_edit"
        or str(nll_adapter.get("adapter_model_sha256", "")) != adapter_sha
        or str(nll_adapter.get("training_report_sha256", ""))
        != sha256_file(training_path)
        or (
            require_separated_execution_identity
            and (
                nll.get("adapter_training_execution_patch_sha256")
                != expected_adapter_training_execution_patch_sha256
                or nll.get("evaluation_execution_patch_sha256")
                != expected_evaluation_execution_patch_sha256
            )
        )
    ):
        raise ValueError(f"epoch {epoch}: NLL evidence mismatch")
    adapter_nll = float(nll.get("adapter_nll", float("nan")))
    if not math.isfinite(adapter_nll):
        raise ValueError(f"epoch {epoch}: validation NLL is non-finite")

    generation_path = Path(str(entry["generation_jsonl"])).resolve()
    generation = _jsonl(generation_path)
    if len(generation) != expected_attempts:
        raise ValueError(f"epoch {epoch}: generation denominator changed")
    attempt_ids = [str(row.get("attempt_id", "")) for row in generation]
    pair_ids = [str(row.get("pair_id", "")) for row in generation]
    paired_seeds = [int(row.get("paired_seed", -1)) for row in generation]
    if (
        any(not value for value in attempt_ids + pair_ids)
        or len(set(attempt_ids)) != expected_attempts
        or len(set(pair_ids)) != expected_attempts
        or any(bool(row.get("retry_or_replacement_used")) for row in generation)
        or {int(row.get("training_seed", -1)) for row in generation}
        != {expected_training_seed}
    ):
        raise ValueError(f"epoch {epoch}: invalid generation attempt identity")
    generation_methods = {str(row.get("method", "")) for row in generation}
    if len(generation_methods) != 1:
        raise ValueError(f"epoch {epoch}: generation methods differ")
    generation_adapter_hashes = {
        str(
            ((row.get("model_identity") or {}).get("adapter_training") or {}).get(
                "adapter_model_sha256", ""
            )
        )
        for row in generation
    }
    if generation_adapter_hashes != {adapter_sha}:
        raise ValueError(f"epoch {epoch}: generation adapter identity mismatch")
    if require_separated_execution_identity:
        generation_identity = {
            (
                row.get("adapter_training_execution_patch_sha256"),
                row.get("refiner_training_execution_patch_sha256"),
                row.get("evaluation_execution_patch_sha256"),
                (row.get("model_identity") or {}).get(
                    "refiner_training_execution_patch_sha256"
                ),
                (row.get("model_identity") or {}).get(
                    "evaluation_execution_patch_sha256"
                ),
                (
                    (row.get("model_identity") or {}).get("adapter_training")
                    or {}
                ).get("execution_patch_sha256"),
            )
            for row in generation
        }
        expected_generation_identity = {
            (
                expected_adapter_training_execution_patch_sha256,
                expected_refiner_training_execution_patch_sha256,
                expected_evaluation_execution_patch_sha256,
                expected_refiner_training_execution_patch_sha256,
                expected_evaluation_execution_patch_sha256,
                expected_adapter_training_execution_patch_sha256,
            )
        }
        if generation_identity != expected_generation_identity:
            raise ValueError(
                f"epoch {epoch}: separated generation execution identity mismatch"
            )

    metric_path = Path(str(entry["crysllmgen_metrics_report"])).resolve()
    metric_report = _json(metric_path)
    if (
        metric_report.get("schema")
        != "crysllmgen_generation_metrics_report_v1"
        or not metric_report.get("ok")
        or int(metric_report.get("attempts", -1)) != expected_attempts
        or metric_report.get("denominator") != "all_generation_attempts"
        or metric_report.get("generation_jsonl_sha256")
        != sha256_file(generation_path)
        or bool(metric_report.get("retry_or_replacement_used"))
    ):
        raise ValueError(f"epoch {epoch}: CrysLLMGen metric evidence mismatch")
    direct = {key: _metric(metric_report, key) for key in (*DIRECT_HIGHER, *DIRECT_LOWER)}

    r5c_path = Path(str(entry["a100_attempt_jsonl"])).resolve()
    r5c = _jsonl(r5c_path)
    if len(r5c) != expected_attempts:
        raise ValueError(f"epoch {epoch}: R5-C A100 denominator changed")
    by_id: dict[str, dict[str, Any]] = {}
    for row in r5c:
        attempt_id = str(row.get("attempt_id", ""))
        if (
            row.get("schema") != "crysllmgen_r5c_a100_sun_attempt_v1"
            or not attempt_id
            or attempt_id in by_id
            or bool(row.get("retry_or_replacement_used"))
            or row.get("base_source_bundle_sha256")
            != expected_base_source_bundle_sha256
            or row.get("execution_patch_sha256")
            != expected_evaluation_execution_patch_sha256
        ):
            raise ValueError(f"epoch {epoch}: invalid R5-C A100 attempt identity")
        by_id[attempt_id] = row
    if set(by_id) != set(attempt_ids):
        raise ValueError(f"epoch {epoch}: generation/R5-C A100 attempts differ")
    strict_sun_vector = [
        bool(
            (by_id[attempt_id].get("metrics") or {}).get(
                "strict_full_sun", False
            )
        )
        for attempt_id in attempt_ids
    ]
    meta_sun_vector = [
        bool(
            (by_id[attempt_id].get("metrics") or {}).get(
                "meta_full_sun", False
            )
        )
        for attempt_id in attempt_ids
    ]
    novel_unique = [
        bool((by_id[attempt_id].get("metrics") or {}).get("novel_unique", False))
        for attempt_id in attempt_ids
    ]
    r5c_summary_path = Path(str(entry["a100_summary"])).resolve()
    summary = _json(r5c_summary_path)
    counts = summary.get("counts") or {}
    rates = summary.get("rates") or {}
    sun_contract = contract.data["evaluation"]["selection"]["r5c_a100_sun"]
    exact_contract = sun_contract["exact_executor"]
    reference_contract = sun_contract["frozen_references"]
    assets = summary.get("assets") or {}
    if (
        summary.get("schema") != "crysllmgen_r5c_a100_sun_summary_v1"
        or not summary.get("ok")
        or summary.get("method") != next(iter(generation_methods))
        or summary.get("denominator") != "all_generation_attempts"
        or summary.get("coverage_adjusted_selection_role")
        != "report_only_never_checkpoint_selection"
        or summary.get("base_source_bundle_sha256")
        != expected_base_source_bundle_sha256
        or summary.get("execution_patch_sha256")
        != expected_evaluation_execution_patch_sha256
        or bool(summary.get("retry_or_replacement_used"))
        or int(counts.get("total_attempts", -1)) != expected_attempts
        or not math.isclose(
            float(rates.get("attempt_strict_full_sun_lower_bound", -1.0)),
            sum(strict_sun_vector) / expected_attempts,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(rates.get("attempt_meta_full_sun_lower_bound", -1.0)),
            sum(meta_sun_vector) / expected_attempts,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(rates.get("attempt_novel_unique", -1.0)),
            sum(novel_unique) / expected_attempts,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or ((summary.get("attempt_results") or {}).get("sha256"))
        != sha256_file(r5c_path)
        or ((assets.get("eval_sun_py") or {}).get("sha256"))
        != exact_contract["eval_sun_py_sha256"]
        or ((assets.get("eval_sun_resumable_py") or {}).get("sha256"))
        != exact_contract["eval_sun_resumable_py_sha256"]
        or ((assets.get("chgnet_model_asset") or {}).get("sha256"))
        != exact_contract["chgnet_model_sha256"]
        or ((assets.get("chgnet_runtime_checkpoint") or {}).get("sha256"))
        != exact_contract["chgnet_model_sha256"]
        or ((assets.get("train_csv") or {}).get("sha256"))
        != reference_contract["mp20_train_csv_sha256"]
        or ((assets.get("training_index_cache") or {}).get("sha256"))
        != reference_contract["mp20_training_index_cache_sha256"]
        or ((assets.get("mp_hull_cache") or {}).get("sha256"))
        != reference_contract["mp_hull_cache_sha256"]
        or ((assets.get("chgnet_relax_cache") or {}).get("sha256"))
        != reference_contract["chgnet_relax_cache_sha256"]
    ):
        raise ValueError(f"epoch {epoch}: R5-C A100 summary mismatch")
    return {
        "logical_epoch": epoch,
        "adapter_sha256": adapter_sha,
        "adapter_nll": adapter_nll,
        "method": next(iter(generation_methods)),
        "attempt_ids": attempt_ids,
        "pair_ids": pair_ids,
        "paired_seeds": paired_seeds,
        "strict_sun_vector": strict_sun_vector,
        "meta_sun_vector": meta_sun_vector,
        "attempt_strict_full_sun_lower_bound": sum(strict_sun_vector)
        / expected_attempts,
        "attempt_meta_full_sun_lower_bound": sum(meta_sun_vector)
        / expected_attempts,
        "novel_unique": sum(novel_unique) / expected_attempts,
        "exact_legacy_r5c_a100": summary["exact_legacy_r5c_a100"],
        "crysllmgen": direct,
        "artifacts": {
            "training_report": _identity(training_path),
            "nll_report": _identity(nll_path),
            "generation": _identity(generation_path),
            "crysllmgen_metrics": _identity(metric_path),
            "a100_attempts": _identity(r5c_path),
            "a100_summary": _identity(r5c_summary_path),
        },
    }


def _public(candidate: Mapping[str, Any]) -> dict[str, Any]:
    hidden = {
        "attempt_ids",
        "pair_ids",
        "paired_seeds",
        "strict_sun_vector",
        "meta_sun_vector",
    }
    return {key: value for key, value in candidate.items() if key not in hidden}


def _absolute_guard(candidate: Mapping[str, Any], guards: Mapping[str, Any]) -> dict[str, bool]:
    metric = candidate["crysllmgen"]
    return {
        "comp_valid": metric["comp_valid"] >= float(guards["comp_valid_min_percent"]),
        "struct_valid": metric["struct_valid"] >= float(guards["struct_valid_min_percent"]),
        "valid": metric["valid"] >= float(guards["valid_min_percent"]),
        "cov_recall": metric["cov_recall"] >= float(guards["cov_recall_min_percent"]),
        "cov_precision": metric["cov_precision"] >= float(guards["cov_precision_min_percent"]),
        "wdist_density": metric["wdist_density"] <= float(guards["wdist_density_max"]),
        "wdist_num_elems": metric["wdist_num_elems"] <= float(guards["wdist_num_elems_max"]),
    }


def _cov_hmean(candidate: Mapping[str, Any]) -> float:
    recall = float(candidate["crysllmgen"]["cov_recall"])
    precision = float(candidate["crysllmgen"]["cov_precision"])
    return 0.0 if recall + precision <= 0.0 else 2.0 * recall * precision / (recall + precision)


def _direct_rank(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    metric = candidate["crysllmgen"]
    return (
        -float(metric["valid"]),
        -_cov_hmean(candidate),
        -float(metric["comp_valid"]),
        -float(metric["struct_valid"]),
        float(metric["wdist_density"]),
        float(metric["wdist_num_elems"]),
        -float(candidate["novel_unique"]),
        float(candidate["adapter_nll"]),
        float(candidate["logical_epoch"]),
    )


def select_epoch_checkpoint(
    *,
    contract: MixedEditEpochContract,
    evidence_manifest_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(evidence_manifest_path).resolve()
    manifest = _json(manifest_path)
    manifest_schema = manifest.get("schema")
    if manifest_schema not in {
        "crysllmgen_epoch_selection_evidence_v1",
        "crysllmgen_epoch_selection_evidence_v2",
    }:
        raise ValueError("invalid epoch-selection evidence manifest")
    if manifest.get("training_amendment_sha256") != contract.sha256:
        raise ValueError("epoch-selection manifest/amendment mismatch")
    training_seed = int(manifest.get("training_seed", -1))
    if training_seed not in (11, 23, 47):
        raise ValueError("unregistered training seed")
    base_source_bundle_sha256 = str(
        manifest.get("base_source_bundle_sha256", "")
    )
    separated_identity = manifest_schema.endswith("_v2")
    evaluation_execution_patch_sha256 = str(
        manifest.get(
            "evaluation_execution_patch_sha256",
            manifest.get("execution_patch_sha256", ""),
        )
    )
    adapter_training_execution_patch_sha256 = str(
        manifest.get(
            "adapter_training_execution_patch_sha256",
            evaluation_execution_patch_sha256,
        )
    )
    refiner_training_execution_patch_sha256 = str(
        manifest.get(
            "refiner_training_execution_patch_sha256",
            evaluation_execution_patch_sha256,
        )
    )
    for label, value in (
        ("base source bundle", base_source_bundle_sha256),
        (
            "adapter training execution patch",
            adapter_training_execution_patch_sha256,
        ),
        (
            "refiner training execution patch",
            refiner_training_execution_patch_sha256,
        ),
        ("evaluation execution patch", evaluation_execution_patch_sha256),
    ):
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"epoch-selection {label} is not a lowercase SHA256")
    if (
        manifest.get("execution_patch_sha256")
        != evaluation_execution_patch_sha256
    ):
        raise ValueError("epoch-selection evaluation patch alias mismatch")
    execution_supersession = manifest.get("execution_supersession")
    if separated_identity:
        if (
            not isinstance(execution_supersession, Mapping)
            or not str(
                execution_supersession.get(
                    "supersedes_epoch_evaluation_array_job_id", ""
                )
            ).isdigit()
            or manifest.get("retry_or_replacement_used") is not True
            or manifest.get("attempt_retry_or_replacement_used") is not False
        ):
            raise ValueError("epoch-selection supersession identity is incomplete")
    elif (
        execution_supersession is not None
        or bool(manifest.get("retry_or_replacement_used"))
    ):
        raise ValueError("legacy epoch-selection evidence cannot be a supersession")
    selection = contract.data["evaluation"]["selection"]
    sun_contract = selection["r5c_a100_sun"]
    expected_attempts = int(sun_contract["attempt_denominator"])
    candidates = [
        _candidate(
            entry,
            contract=contract,
            expected_attempts=expected_attempts,
            expected_training_seed=training_seed,
            expected_base_source_bundle_sha256=base_source_bundle_sha256,
            expected_adapter_training_execution_patch_sha256=(
                adapter_training_execution_patch_sha256
            ),
            expected_refiner_training_execution_patch_sha256=(
                refiner_training_execution_patch_sha256
            ),
            expected_evaluation_execution_patch_sha256=(
                evaluation_execution_patch_sha256
            ),
            require_separated_execution_identity=separated_identity,
        )
        for entry in manifest.get("epochs", ())
    ]
    if (
        {value["logical_epoch"] for value in candidates} != {1, 2, 3}
        or len(candidates) != 3
    ):
        raise ValueError("epoch selection requires exactly epochs 1, 2, and 3")
    candidates.sort(key=lambda value: int(value["logical_epoch"]))
    identity = (
        candidates[0]["attempt_ids"],
        candidates[0]["pair_ids"],
        candidates[0]["paired_seeds"],
        candidates[0]["method"],
    )
    for candidate in candidates[1:]:
        if (
            candidate["attempt_ids"],
            candidate["pair_ids"],
            candidate["paired_seeds"],
            candidate["method"],
        ) != identity:
            raise ValueError("epoch checkpoints do not share paired attempts and noise")

    metric_contract = selection["crysllmgen_direct_metrics"]
    guards = metric_contract["absolute_collapse_guards"]
    absolute_checks = {
        value["logical_epoch"]: _absolute_guard(value, guards) for value in candidates
    }
    absolute_pass = [
        value
        for value in candidates
        if all(absolute_checks[value["logical_epoch"]].values())
    ]
    if not absolute_pass:
        raise ValueError("every epoch checkpoint failed an absolute collapse guard")

    best_high = {
        key: max(float(value["crysllmgen"][key]) for value in absolute_pass)
        for key in DIRECT_HIGHER
    }
    best_low = {
        key: min(float(value["crysllmgen"][key]) for value in absolute_pass)
        for key in DIRECT_LOWER
    }
    margins = metric_contract["noninferiority_from_metric_best"]
    margin_by_metric = {
        "comp_valid": float(margins["comp_valid_drop_max_pp"]),
        "struct_valid": float(margins["struct_valid_drop_max_pp"]),
        "valid": float(margins["valid_drop_max_pp"]),
        "cov_recall": float(margins["cov_recall_drop_max_pp"]),
        "cov_precision": float(margins["cov_precision_drop_max_pp"]),
        "wdist_density": float(margins["wdist_density_increase_max"]),
        "wdist_num_elems": float(margins["wdist_num_elems_increase_max"]),
    }
    noninferiority: dict[int, dict[str, bool]] = {}
    for value in absolute_pass:
        metric = value["crysllmgen"]
        checks = {
            key: float(metric[key]) >= best_high[key] - margin_by_metric[key]
            for key in DIRECT_HIGHER
        }
        checks.update(
            {
                key: float(metric[key]) <= best_low[key] + margin_by_metric[key]
                for key in DIRECT_LOWER
            }
        )
        noninferiority[int(value["logical_epoch"])] = checks
    eligible = [
        value
        for value in absolute_pass
        if all(noninferiority[int(value["logical_epoch"])].values())
    ]
    if not eligible:
        raise ValueError("no epoch is jointly noninferior on direct CrysLLMGen metrics")

    strict_point_best = min(
        eligible,
        key=lambda value: (
            -float(value["attempt_strict_full_sun_lower_bound"]),
            int(value["logical_epoch"]),
        ),
    )
    bootstrap_seed_root = hashlib.sha256(
        (contract.sha256 + "|" + "|".join(identity[0])).encode("utf-8")
    ).hexdigest()
    strict_pairwise: dict[str, dict[str, float | int]] = {}
    strict_equivalent_epochs = {int(strict_point_best["logical_epoch"])}
    for value in eligible:
        if value is strict_point_best:
            continue
        epoch = int(value["logical_epoch"])
        seed = int(
            hashlib.sha256(
                (
                    f"{bootstrap_seed_root}|strict|"
                    f"{strict_point_best['logical_epoch']}|{epoch}"
                ).encode("utf-8")
            ).hexdigest()[:16],
            16,
        )
        comparison = paired_bootstrap_difference_ci(
            strict_point_best["strict_sun_vector"],
            value["strict_sun_vector"],
            draws=int(sun_contract["paired_bootstrap_draws"]),
            seed=seed,
        )
        strict_pairwise[
            f"epoch_{strict_point_best['logical_epoch']}_minus_epoch_{epoch}"
        ] = comparison
        if float(comparison["ci95_lower"]) <= 0.0 <= float(
            comparison["ci95_upper"]
        ):
            strict_equivalent_epochs.add(epoch)
    strict_equivalent = [
        value
        for value in eligible
        if int(value["logical_epoch"]) in strict_equivalent_epochs
    ]
    strict_decisive = len(strict_equivalent) == 1

    meta_point_best = min(
        strict_equivalent,
        key=lambda value: (
            -float(value["attempt_meta_full_sun_lower_bound"]),
            int(value["logical_epoch"]),
        ),
    )
    meta_pairwise: dict[str, dict[str, float | int]] = {}
    meta_equivalent_epochs = {int(meta_point_best["logical_epoch"])}
    for value in strict_equivalent:
        if value is meta_point_best:
            continue
        epoch = int(value["logical_epoch"])
        seed = int(
            hashlib.sha256(
                (
                    f"{bootstrap_seed_root}|meta|"
                    f"{meta_point_best['logical_epoch']}|{epoch}"
                ).encode("utf-8")
            ).hexdigest()[:16],
            16,
        )
        comparison = paired_bootstrap_difference_ci(
            meta_point_best["meta_sun_vector"],
            value["meta_sun_vector"],
            draws=int(sun_contract["paired_bootstrap_draws"]),
            seed=seed,
        )
        meta_pairwise[
            f"epoch_{meta_point_best['logical_epoch']}_minus_epoch_{epoch}"
        ] = comparison
        if float(comparison["ci95_lower"]) <= 0.0 <= float(
            comparison["ci95_upper"]
        ):
            meta_equivalent_epochs.add(epoch)
    meta_equivalent = [
        value
        for value in strict_equivalent
        if int(value["logical_epoch"]) in meta_equivalent_epochs
    ]
    meta_decisive = len(meta_equivalent) == 1
    selected = (
        strict_point_best
        if strict_decisive
        else meta_point_best
        if meta_decisive
        else min(meta_equivalent, key=_direct_rank)
    )
    result = {
        "schema": "crysllmgen_epoch_selection_lock_v1",
        "ok": True,
        "training_seed": training_seed,
        "base_source_bundle_sha256": base_source_bundle_sha256,
        "execution_patch_sha256": evaluation_execution_patch_sha256,
        "adapter_training_execution_patch_sha256": (
            adapter_training_execution_patch_sha256
        ),
        "refiner_training_execution_patch_sha256": (
            refiner_training_execution_patch_sha256
        ),
        "evaluation_execution_patch_sha256": (
            evaluation_execution_patch_sha256
        ),
        "selection_scope": selection["scope"],
        "selection_policy": selection["policy"],
        "selected_epoch": int(selected["logical_epoch"]),
        "selected_adapter_sha256": selected["adapter_sha256"],
        "strict_sun_point_best_epoch": int(strict_point_best["logical_epoch"]),
        "strict_sun_point_best_decisive": strict_decisive,
        "strict_sun_equivalent_epochs": sorted(strict_equivalent_epochs),
        "paired_strict_sun_bootstrap": strict_pairwise,
        "meta_sun_point_best_epoch_within_strict_equivalent": int(
            meta_point_best["logical_epoch"]
        ),
        "meta_sun_point_best_decisive": meta_decisive,
        "meta_sun_equivalent_epochs": sorted(meta_equivalent_epochs),
        "paired_meta_sun_bootstrap": meta_pairwise,
        "absolute_guard_checks": {
            str(epoch): checks for epoch, checks in sorted(absolute_checks.items())
        },
        "noninferiority_checks": {
            str(epoch): checks for epoch, checks in sorted(noninferiority.items())
        },
        "candidates": [_public(value) for value in candidates],
        "training_amendment": {
            "path": str(contract.path),
            "sha256": contract.sha256,
        },
        "evidence_manifest": _identity(manifest_path),
        "coverage_adjusted_sun_used_for_selection": False,
        "legacy_reconstructed_denominator_sun_used_for_selection": False,
        "final_test_or_final_mlip_used_for_selection": False,
        "execution_supersession": execution_supersession,
        "retry_or_replacement_used": separated_identity,
        "attempt_retry_or_replacement_used": False,
    }
    write_json_exclusive(output, result)
    return result
