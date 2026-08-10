#!/usr/bin/env python3
"""Assemble the complete three-batch R03/B3 pre/post evaluation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from protocol import (
    ARMS,
    DENOMINATOR,
    REPEATS,
    ordered_rows,
    read_json,
    read_jsonl,
    require_source_manifest,
    sha256_file,
    validate_config,
    write_json_exclusive,
)


STAGES = ("pre_model494", "post_model494")
CONDITIONAL_SUN_ENDPOINTS = (
    "novel",
    "unique_representative",
    "novel_unique",
    "strict_full_sun",
    "meta_full_sun",
)


def identity(path: Path) -> dict[str, Any]:
    location = path.resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    return {"path": str(location), "bytes": location.stat().st_size, "sha256": sha256_file(location)}


def generation_complete(row: Mapping[str, Any]) -> bool:
    return row.get("status") == "succeeded"


def reconstructed(row: Mapping[str, Any]) -> bool:
    return row.get("evaluation_status") != "generation_or_reconstruction_failed"


def sun_value(row: Mapping[str, Any], endpoint: str) -> bool:
    metrics = row.get("metrics") or {}
    if endpoint == "reconstructed":
        return reconstructed(row)
    if endpoint == "hull_evaluated":
        return metrics.get("e_above_hull") is not None
    return bool(metrics.get(endpoint))


def exact_mcnemar(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    baseline_only = int(np.count_nonzero(baseline & ~candidate))
    candidate_only = int(np.count_nonzero(candidate & ~baseline))
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(baseline_only, candidate_only)
        numerator = sum(math.comb(discordant, index) for index in range(lower + 1))
        p_value = min(1.0, 2.0 * numerator / (2**discordant))
    return {
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "discordant": discordant,
        "two_sided_exact_p_value": p_value,
    }


def paired_summary(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    attempts = int(baseline.size)
    baseline_count = int(np.count_nonzero(baseline))
    candidate_count = int(np.count_nonzero(candidate))
    return {
        "attempts": attempts,
        "baseline_count": baseline_count,
        "candidate_count": candidate_count,
        "difference_count": candidate_count - baseline_count,
        "baseline_rate": baseline_count / attempts,
        "candidate_rate": candidate_count / attempts,
        "difference_percentage_points": 100.0 * (candidate_count - baseline_count) / attempts,
        "exact_mcnemar": exact_mcnemar(baseline, candidate),
    }


def hierarchical_difference_bootstrap(
    differences: np.ndarray, *, draws: int, seed: int
) -> dict[str, Any]:
    if differences.shape != (len(REPEATS), DENOMINATOR):
        raise ValueError("hierarchical difference matrix shape changed")
    probabilities: list[np.ndarray] = []
    for repeat in REPEATS:
        counts = np.asarray(
            [
                np.count_nonzero(differences[repeat] == -1),
                np.count_nonzero(differences[repeat] == 0),
                np.count_nonzero(differences[repeat] == 1),
            ],
            dtype=np.float64,
        )
        probabilities.append(counts / DENOMINATOR)
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    batch_size = 1000
    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        size = stop - start
        selected = rng.integers(0, len(REPEATS), size=(size, len(REPEATS)))
        block_sums = np.empty((size, len(REPEATS)), dtype=np.int32)
        for repeat in REPEATS:
            positions = np.argwhere(selected == repeat)
            if not len(positions):
                continue
            sampled = rng.multinomial(DENOMINATOR, probabilities[repeat], size=len(positions))
            sums = sampled[:, 2] - sampled[:, 0]
            block_sums[positions[:, 0], positions[:, 1]] = sums
        values[start:stop] = block_sums.sum(axis=1) * (100.0 / (len(REPEATS) * DENOMINATOR))
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "design": "resample three plan-batch blocks, then paired ordinals within selected blocks",
        "implementation": "exact multinomial representation of ordinal bootstrap",
        "draws": draws,
        "seed": seed,
        "observed_mean_difference_percentage_points": float(differences.mean() * 100.0),
        "ci95_lower_percentage_points": float(lower),
        "ci95_upper_percentage_points": float(upper),
        "probability_difference_gt_zero": float(np.mean(values > 0.0)),
        "probability_difference_lt_zero": float(np.mean(values < 0.0)),
    }


def conditional_bootstrap(
    baseline_reconstructed: np.ndarray,
    baseline_numerator: np.ndarray,
    candidate_reconstructed: np.ndarray,
    candidate_numerator: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    expected = (len(REPEATS), DENOMINATOR)
    arrays = (
        baseline_reconstructed,
        baseline_numerator,
        candidate_reconstructed,
        candidate_numerator,
    )
    if any(array.shape != expected for array in arrays):
        raise ValueError("conditional bootstrap matrix shape changed")
    if np.any(baseline_numerator & ~baseline_reconstructed) or np.any(
        candidate_numerator & ~candidate_reconstructed
    ):
        raise ValueError("S.U.N. numerator escaped reconstructed denominator")

    probabilities: list[np.ndarray] = []
    for repeat in REPEATS:
        codes = (
            baseline_reconstructed[repeat].astype(np.uint8)
            | (baseline_numerator[repeat].astype(np.uint8) << 1)
            | (candidate_reconstructed[repeat].astype(np.uint8) << 2)
            | (candidate_numerator[repeat].astype(np.uint8) << 3)
        )
        probabilities.append(np.bincount(codes, minlength=16).astype(np.float64) / DENOMINATOR)
    code_values = np.arange(16, dtype=np.uint8)
    br_bit = ((code_values >> 0) & 1).astype(np.int32)
    bn_bit = ((code_values >> 1) & 1).astype(np.int32)
    cr_bit = ((code_values >> 2) & 1).astype(np.int32)
    cn_bit = ((code_values >> 3) & 1).astype(np.int32)

    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    batch_size = 1000
    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        size = stop - start
        selected = rng.integers(0, len(REPEATS), size=(size, len(REPEATS)))
        aggregate = np.zeros((size, 4), dtype=np.int32)
        for repeat in REPEATS:
            positions = np.argwhere(selected == repeat)
            if not len(positions):
                continue
            sampled = rng.multinomial(DENOMINATOR, probabilities[repeat], size=len(positions))
            block = np.column_stack(
                (sampled @ br_bit, sampled @ bn_bit, sampled @ cr_bit, sampled @ cn_bit)
            )
            np.add.at(aggregate, positions[:, 0], block)
        if np.any(aggregate[:, 0] == 0) or np.any(aggregate[:, 2] == 0):
            raise ValueError("bootstrap produced an empty reconstructed denominator")
        values[start:stop] = 100.0 * (
            aggregate[:, 3] / aggregate[:, 2] - aggregate[:, 1] / aggregate[:, 0]
        )
    observed = 100.0 * (
        candidate_numerator.sum() / candidate_reconstructed.sum()
        - baseline_numerator.sum() / baseline_reconstructed.sum()
    )
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "design": "hierarchical paired bootstrap of conditional reconstructed-structure rates",
        "draws": draws,
        "seed": seed,
        "observed_difference_percentage_points": float(observed),
        "ci95_lower_percentage_points": float(lower),
        "ci95_upper_percentage_points": float(upper),
        "probability_difference_gt_zero": float(np.mean(values > 0.0)),
        "probability_difference_lt_zero": float(np.mean(values < 0.0)),
    }


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            result.update(flatten(item, child))
    else:
        result[prefix] = value
    return result


def escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(escape(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def rate_cell(count: int, denominator: int = DENOMINATOR) -> str:
    return f"{count}/{denominator} ({100.0 * count / denominator:.3f}%)"


def find_numeric(native: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    flat = flatten(native)
    normalized = {key.lower().replace("-", "_"): value for key, value in flat.items()}
    for name in names:
        for key, value in normalized.items():
            if key.endswith(name) and isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def percent_display(value: float | None) -> str:
    if value is None:
        return "NA"
    percent = value * 100.0 if abs(value) <= 1.0 else value
    return f"{percent:.4f}"


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P0 Plan1200 × R03/B3 × pre/post model_494 — Complete Results",
        "",
        "## 结论与统计口径",
        "",
        "三批 P0 plan 独立采样，每批 raw 1,200；按 planner ordinal 冻结前 1,000 个 parse-success。每批同一组 1,000 plan、body seed 和 refiner seed 同时送入 R03 与 B3。pre 指 safe-axis body 直接输出，post 指同一输出经过冻结 model_494、800 步 refine。",
        "",
        "S.U.N. headline 严格采用本地论文复现脚本的 `reconstructed_structures` 分母；固定 all-1,000 attempt 口径作为保守 secondary。McNemar 与普通配对 bootstrap 使用固定 all-attempt 口径；条件 S.U.N. bootstrap 单独对 reconstructed 分母做比率推断，二者不混算。CrysLLMGen/Direct 指标不归因于 `comp_valid` 与 DLM 的训练关系。",
        "",
        "## Planner 三批冻结证据",
        "",
    ]
    planner_rows = []
    for item in report["planner"]["repeats"]:
        planner_rows.append(
            (
                item["repeat"],
                item["planner_seed"],
                item["raw_attempts"],
                item["parse_successes"],
                item["parse_failures"],
                item["reserve_parse_success_count"],
                item["cohort1000"]["sha256"],
            )
        )
    lines.extend(
        [
            table(
                ["Repeat", "P0 seed", "Raw", "Parse-success", "Parse-fail", "Reserve", "Frozen cohort SHA256"],
                planner_rows,
            ),
            "",
            "## 每臂、每批、pre/post 主表",
            "",
        ]
    )
    stage_rows = []
    for row in report["stage_rows"]:
        direct = row["direct_counts"]
        sun = row["sun_counts"]
        stage_rows.append(
            (
                row["arm"],
                row["repeat"],
                row["stage"],
                rate_cell(row["generation_succeeded"]),
                rate_cell(direct["composition_valid"]),
                rate_cell(direct["structure_valid"]),
                rate_cell(direct["joint_valid"]),
                row["cov_precision_percent"],
                row["cov_recall_percent"],
                rate_cell(sun["reconstructed"]),
                rate_cell(sun["novel"]),
                rate_cell(sun["unique"]),
                rate_cell(sun["novel_unique"]),
                rate_cell(sun["strict_full_sun"]),
                rate_cell(sun["meta_full_sun"]),
                f"{sun['hull_evaluated']}/{DENOMINATOR}",
                row["hull_unknown"],
            )
        )
    lines.extend(
        [
            table(
                [
                    "Arm", "Repeat", "Stage", "Gen", "Comp", "Struct", "Joint", "COV-P %", "COV-R %",
                    "Reconstructed", "Novel", "Unique", "Novel∩Unique", "Strict S.U.N.", "Meta S.U.N.",
                    "Hull evaluated", "Hull unknown",
                ],
                stage_rows,
            ),
            "",
            "## S.U.N. reconstructed-structure headline（完整细分字段）",
            "",
        ]
    )
    legacy_rows = []
    for row in report["stage_rows"]:
        for path, value in sorted(flatten(row["sun_exact_legacy_reconstructed_denominator"]).items()):
            legacy_rows.append((row["arm"], row["repeat"], row["stage"], path, value))
    lines.extend(
        [
            table(["Arm", "Repeat", "Stage", "Exact-legacy field", "Value"], legacy_rows),
            "",
            "## CrysLLMGen 完整 upstream 数值指标",
            "",
        ]
    )
    direct_rows = []
    for row in report["stage_rows"]:
        native = row["direct_native_report_complete"].get("metrics_unchanged_upstream") or {}
        for path, value in sorted(flatten(native).items()):
            direct_rows.append((row["arm"], row["repeat"], row["stage"], path, value))
    lines.extend(
        [
            table(["Arm", "Repeat", "Stage", "Upstream metric", "Value"], direct_rows),
            "",
            "## 固定 all-1,000 attempt 的逐批精确 McNemar",
            "",
        ]
    )
    mcnemar_rows = []
    for comparison, endpoints in report["paired_all_attempt_inference"].items():
        for endpoint, result in endpoints.items():
            for repeat in result["per_repeat"]:
                exact = repeat["exact_mcnemar"]
                mcnemar_rows.append(
                    (
                        comparison,
                        endpoint,
                        repeat["repeat"],
                        repeat["baseline_count"],
                        repeat["candidate_count"],
                        f"{repeat['difference_percentage_points']:.3f}",
                        exact["baseline_only"],
                        exact["candidate_only"],
                        f"{exact['two_sided_exact_p_value']:.6g}",
                    )
                )
    lines.extend(
        [
            table(
                ["Comparison", "Endpoint", "Repeat", "Baseline", "Candidate", "Δ pp", "Baseline-only", "Candidate-only", "Exact p"],
                mcnemar_rows,
            ),
            "",
            "## 固定 all-attempt 分层配对 bootstrap（50k）",
            "",
        ]
    )
    bootstrap_rows = []
    for comparison, endpoints in report["paired_all_attempt_inference"].items():
        for endpoint, result in endpoints.items():
            boot = result["hierarchical_paired_bootstrap"]
            signs = result["repeat_sign_stability"]
            bootstrap_rows.append(
                (
                    comparison,
                    endpoint,
                    f"{boot['observed_mean_difference_percentage_points']:.3f}",
                    f"[{boot['ci95_lower_percentage_points']:.3f}, {boot['ci95_upper_percentage_points']:.3f}]",
                    f"{boot['probability_difference_gt_zero']:.4f}",
                    f"+{signs['positive']}/0{signs['zero']}/-{signs['negative']}",
                )
            )
    lines.extend(
        [
            table(["Comparison", "Endpoint", "Observed Δ pp", "95% CI pp", "P(Δ>0)", "Repeat signs"], bootstrap_rows),
            "",
            "## S.U.N. reconstructed 分母的条件率分层配对 bootstrap（50k）",
            "",
        ]
    )
    conditional_rows = []
    for comparison, endpoints in report["paired_reconstructed_denominator_inference"].items():
        for endpoint, result in endpoints.items():
            boot = result["hierarchical_paired_bootstrap"]
            conditional_rows.append(
                (
                    comparison,
                    endpoint,
                    f"{100.0 * result['pooled_descriptive']['baseline_rate']:.3f}",
                    f"{100.0 * result['pooled_descriptive']['candidate_rate']:.3f}",
                    f"{boot['observed_difference_percentage_points']:.3f}",
                    f"[{boot['ci95_lower_percentage_points']:.3f}, {boot['ci95_upper_percentage_points']:.3f}]",
                    f"{boot['probability_difference_gt_zero']:.4f}",
                )
            )
    lines.extend(
        [
            table(["Comparison", "Endpoint / reconstructed", "Baseline %", "Candidate %", "Δ pp", "95% CI pp", "P(Δ>0)"], conditional_rows),
            "",
            "## Pooled 3,000 描述性汇总（不作独立样本推断）",
            "",
        ]
    )
    pooled_rows = []
    for comparison, endpoints in report["pooled_3000_descriptive"].items():
        for endpoint, result in endpoints.items():
            pooled_rows.append(
                (
                    comparison,
                    endpoint,
                    result["baseline_count"],
                    result["candidate_count"],
                    f"{result['difference_percentage_points']:.3f}",
                )
            )
    lines.extend(
        [
            table(["Comparison", "Endpoint", "Baseline / 3000", "Candidate / 3000", "Δ pp"], pooled_rows),
            "",
            "## 失败、coverage 与能量诊断",
            "",
        ]
    )
    diagnostic_rows = []
    for row in report["stage_rows"]:
        diagnostics = row["sun_diagnostics"]
        diagnostic_rows.append(
            (
                row["arm"], row["repeat"], row["stage"],
                json.dumps(row["direct_failure_reasons"], sort_keys=True),
                json.dumps(diagnostics["evaluation_status_counts"], sort_keys=True),
                json.dumps(diagnostics["generation_failure_reasons"], sort_keys=True),
                diagnostics["hull_evaluated_denominator"],
                diagnostics["hull_unknown"],
                json.dumps(diagnostics["e_above_hull_ev_per_atom"], sort_keys=True),
                json.dumps(diagnostics["energy_per_atom_ev"], sort_keys=True),
            )
        )
    lines.extend(
        [
            table(
                ["Arm", "Repeat", "Stage", "Direct failures", "S.U.N. statuses", "Generation failures", "Hull eval", "Hull unknown", "E-hull summary", "Energy summary"],
                diagnostic_rows,
            ),
            "",
            "## 完整 native summary 附录",
            "",
            "以下 JSON 是每个 stage 的原生 CrysLLMGen report 与原生 S.U.N. summary，未删减字段；逐 attempt ledger 以 SHA256 路径登记在 terminal JSON 中。",
            "",
        ]
    )
    for row in report["stage_rows"]:
        lines.extend(
            [
                f"### {row['arm']} / repeat {row['repeat']} / {row['stage']}",
                "",
                "```json",
                json.dumps(
                    {
                        "direct_native_report_complete": row["direct_native_report_complete"],
                        "sun_native_summary_complete": row["sun_native_summary_complete"],
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 科学与工程边界",
            "",
            "- 结果仅比较冻结 P0 plan cohort 下 R03/B3，以及 model_494 refine 前后；不执行训练、checkpoint 重选、promotion 或 RL。",
            "- `comp_valid`/Direct 是输出描述指标，不被解释为 DLM 训练目标的直接结果。",
            "- 逐批是主要证据；pooled 3,000 只作描述。",
            "- 所有 raw failure 均留在固定 1,000 分母中，无 retry、replacement、repair、filter 或 rerank。",
            "",
        ]
    )
    return "\n".join(lines)


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    validate_config(config)
    run_root = args.run_root.resolve()
    if run_root != Path(config["run_root"]).resolve():
        raise ValueError("assembly run root changed")
    if args.output.resolve() != run_root / "terminal_report.json":
        raise ValueError("terminal report output changed")
    if args.markdown.resolve() != run_root / "RESULTS_COMPLETE.md":
        raise ValueError("Markdown output changed")

    submission = read_json(run_root / "status/body_submission_record.json")
    preflight = read_json(run_root / "status/body_preflight_report.json")
    planner = read_json(run_root / "planner_terminal_report.json")
    completion = read_json(run_root / "mp_cache/completion_manifest.json")
    if (
        preflight.get("status") != "pass"
        or preflight.get("source_manifest_sha256") != args.source_manifest_sha256
        or planner.get("status") != "complete"
        or completion.get("status") != "complete_all_wanted_chemsys_resolved"
        or submission.get("status") != "complete"
        or submission.get("separate_arm_arrays") is not True
    ):
        raise ValueError("preflight/planner/cache/submission evidence changed")

    vectors: dict[tuple[str, int, str, str], np.ndarray] = {}
    reconstructed_vectors: dict[tuple[str, int, str], np.ndarray] = {}
    stage_rows: list[dict[str, Any]] = []
    repeat_reports: list[dict[str, Any]] = []
    cohort_body_hashes: dict[tuple[str, int], str] = {}

    endpoints = list(config["statistics"]["binary_endpoints"])
    for arm in ARMS:
        for repeat in REPEATS:
            root = run_root / "arms" / arm / "repeats" / str(repeat)
            exit_code = run_root / f"status/body_{arm}_repeat_{repeat}_exit_code.txt"
            report_path = root / "repeat_report.json"
            if (
                not (root / "_SUCCESS").is_file()
                or (root / "_FAILURE").exists()
                or not exit_code.is_file()
                or exit_code.read_text(encoding="ascii").strip() != "0"
            ):
                raise ValueError(f"{arm} repeat {repeat} is not an engineering success")
            repeat_report = read_json(report_path)
            if repeat_report.get("status") != "complete" or repeat_report.get("ok") is not True:
                raise ValueError(f"{arm} repeat {repeat} report changed")
            repeat_reports.append(repeat_report)
            body_input = read_json(root / "body/input_report.json")
            cohort_body_hashes[(arm, repeat)] = str(body_input["cohort1000_sha256"])

            for stage in STAGES:
                stage_root = root / "stages" / stage
                generation = ordered_rows(
                    read_jsonl(stage_root / "generation/generation.jsonl"), ordinal_field="ordinal"
                )
                direct = read_jsonl(stage_root / "evaluation/crysllmgen_metrics/attempt_metrics.jsonl")
                sun = sorted(
                    read_jsonl(stage_root / "evaluation/r5c_a100_sun/attempt_results.jsonl"),
                    key=lambda row: int(row["generation_ordinal"]),
                )
                stage_report = read_json(stage_root / "evaluation/stage_report.json")
                if len(direct) != DENOMINATOR or len(sun) != DENOMINATOR:
                    raise ValueError(f"{arm} repeat {repeat} {stage} ledger length changed")
                expected_ids = [str(row["attempt_id"]) for row in generation]
                if (
                    [str(row.get("attempt_id")) for row in direct] != expected_ids
                    or [str(row.get("attempt_id")) for row in sun] != expected_ids
                ):
                    raise ValueError(f"{arm} repeat {repeat} {stage} attempt alignment changed")

                direct_map = {
                    "composition_valid": np.asarray([bool(row["comp_valid"]) for row in direct]),
                    "structure_valid": np.asarray([bool(row["struct_valid"]) for row in direct]),
                    "joint_valid": np.asarray([bool(row["valid"]) for row in direct]),
                }
                reconstructed_array = np.asarray([reconstructed(row) for row in sun])
                reconstructed_vectors[(arm, repeat, stage)] = reconstructed_array
                vectors[(arm, repeat, stage, "generation_complete")] = np.asarray(
                    [generation_complete(row) for row in generation]
                )
                for endpoint, values in direct_map.items():
                    vectors[(arm, repeat, stage, endpoint)] = values
                for endpoint in endpoints:
                    if endpoint in {"generation_complete", *direct_map.keys()}:
                        continue
                    vectors[(arm, repeat, stage, endpoint)] = np.asarray(
                        [sun_value(row, endpoint) for row in sun]
                    )

                native_direct = stage_report["direct_native_report_complete"]
                cov_precision = find_numeric(
                    native_direct.get("metrics_unchanged_upstream") or {},
                    ("cov_precision", "coverage_precision", "cov_p"),
                )
                cov_recall = find_numeric(
                    native_direct.get("metrics_unchanged_upstream") or {},
                    ("cov_recall", "coverage_recall", "cov_r"),
                )
                stage_rows.append(
                    {
                        "arm": arm,
                        "repeat": repeat,
                        "stage": stage,
                        "generation_succeeded": int(stage_report["generation_succeeded"]),
                        "direct_counts": stage_report["direct_counts"],
                        "direct_rates_all_attempts": stage_report["direct_rates_all_attempts"],
                        "direct_failure_reasons": stage_report["direct_failure_reasons"],
                        "direct_native_report_complete": native_direct,
                        "cov_precision_raw": cov_precision,
                        "cov_recall_raw": cov_recall,
                        "cov_precision_percent": percent_display(cov_precision),
                        "cov_recall_percent": percent_display(cov_recall),
                        "sun_counts": stage_report["sun_counts"],
                        "sun_rates_all_attempts": stage_report["sun_rates_all_attempts"],
                        "sun_exact_legacy_reconstructed_denominator": stage_report[
                            "sun_exact_legacy_reconstructed_denominator"
                        ],
                        "sun_native_summary_complete": stage_report["sun_native_summary_complete"],
                        "sun_diagnostics": stage_report["sun_diagnostics"],
                        "hull_unknown": int(
                            stage_report["sun_diagnostics"]["hull_unknown"]
                        ),
                        "artifacts": {
                            "repeat_report": identity(report_path),
                            "generation": identity(stage_root / "generation/generation.jsonl"),
                            "direct_attempts": identity(
                                stage_root / "evaluation/crysllmgen_metrics/attempt_metrics.jsonl"
                            ),
                            "sun_attempts": identity(
                                stage_root / "evaluation/r5c_a100_sun/attempt_results.jsonl"
                            ),
                            "stage_report": identity(
                                stage_root / "evaluation/stage_report.json"
                            ),
                        },
                    }
                )

    for repeat in REPEATS:
        if cohort_body_hashes[("R03", repeat)] != cohort_body_hashes[("B3", repeat)]:
            raise ValueError(f"repeat {repeat} R03/B3 cohort identity diverged")
    if len({cohort_body_hashes[("R03", repeat)] for repeat in REPEATS}) != 3:
        raise ValueError("three process repeats reuse a cohort")

    comparison_specs = {
        "B3_minus_R03_pre_model494": (("R03", "pre_model494"), ("B3", "pre_model494")),
        "B3_minus_R03_post_model494": (("R03", "post_model494"), ("B3", "post_model494")),
        "post_minus_pre_R03": (("R03", "pre_model494"), ("R03", "post_model494")),
        "post_minus_pre_B3": (("B3", "pre_model494"), ("B3", "post_model494")),
    }
    if list(comparison_specs) != list(config["statistics"]["comparisons"]):
        raise ValueError("comparison order changed")
    draws = int(config["statistics"]["hierarchical_paired_bootstrap_draws"])
    seed = int(config["statistics"]["bootstrap_seed"])
    all_attempt_inference: dict[str, Any] = {}
    conditional_inference: dict[str, Any] = {}
    pooled: dict[str, Any] = {}

    for comparison_index, (comparison, (baseline_spec, candidate_spec)) in enumerate(
        comparison_specs.items()
    ):
        all_attempt_inference[comparison] = {}
        conditional_inference[comparison] = {}
        pooled[comparison] = {}
        baseline_arm, baseline_stage = baseline_spec
        candidate_arm, candidate_stage = candidate_spec
        for endpoint_index, endpoint in enumerate(endpoints):
            baseline_matrix = np.stack(
                [vectors[(baseline_arm, repeat, baseline_stage, endpoint)] for repeat in REPEATS]
            )
            candidate_matrix = np.stack(
                [vectors[(candidate_arm, repeat, candidate_stage, endpoint)] for repeat in REPEATS]
            )
            differences = candidate_matrix.astype(np.int8) - baseline_matrix.astype(np.int8)
            per_repeat = []
            repeat_difference_counts = []
            for repeat in REPEATS:
                summary = paired_summary(baseline_matrix[repeat], candidate_matrix[repeat])
                per_repeat.append({"repeat": repeat, **summary})
                repeat_difference_counts.append(summary["difference_count"])
            all_attempt_inference[comparison][endpoint] = {
                "baseline": {"arm": baseline_arm, "stage": baseline_stage},
                "candidate": {"arm": candidate_arm, "stage": candidate_stage},
                "denominator": "all_1000_attempts_per_repeat",
                "per_repeat": per_repeat,
                "repeat_sign_stability": {
                    "positive": sum(value > 0 for value in repeat_difference_counts),
                    "zero": sum(value == 0 for value in repeat_difference_counts),
                    "negative": sum(value < 0 for value in repeat_difference_counts),
                },
                "hierarchical_paired_bootstrap": hierarchical_difference_bootstrap(
                    differences,
                    draws=draws,
                    seed=seed + comparison_index * 1000 + endpoint_index,
                ),
            }
            pooled[comparison][endpoint] = {
                **paired_summary(baseline_matrix.ravel(), candidate_matrix.ravel()),
                "interpretation": "descriptive only; three plan batches are the replication unit",
            }

        baseline_reconstructed = np.stack(
            [reconstructed_vectors[(baseline_arm, repeat, baseline_stage)] for repeat in REPEATS]
        )
        candidate_reconstructed = np.stack(
            [reconstructed_vectors[(candidate_arm, repeat, candidate_stage)] for repeat in REPEATS]
        )
        for endpoint_index, endpoint in enumerate(CONDITIONAL_SUN_ENDPOINTS):
            baseline_numerator = np.stack(
                [vectors[(baseline_arm, repeat, baseline_stage, endpoint)] for repeat in REPEATS]
            )
            candidate_numerator = np.stack(
                [vectors[(candidate_arm, repeat, candidate_stage, endpoint)] for repeat in REPEATS]
            )
            per_repeat = []
            for repeat in REPEATS:
                br = int(np.count_nonzero(baseline_reconstructed[repeat]))
                cr = int(np.count_nonzero(candidate_reconstructed[repeat]))
                bn = int(np.count_nonzero(baseline_numerator[repeat]))
                cn = int(np.count_nonzero(candidate_numerator[repeat]))
                per_repeat.append(
                    {
                        "repeat": repeat,
                        "baseline_numerator": bn,
                        "baseline_reconstructed_denominator": br,
                        "baseline_rate": bn / br,
                        "candidate_numerator": cn,
                        "candidate_reconstructed_denominator": cr,
                        "candidate_rate": cn / cr,
                        "difference_percentage_points": 100.0 * (cn / cr - bn / br),
                    }
                )
            br_total = int(baseline_reconstructed.sum())
            cr_total = int(candidate_reconstructed.sum())
            bn_total = int(baseline_numerator.sum())
            cn_total = int(candidate_numerator.sum())
            conditional_inference[comparison][endpoint] = {
                "baseline": {"arm": baseline_arm, "stage": baseline_stage},
                "candidate": {"arm": candidate_arm, "stage": candidate_stage},
                "denominator": "reconstructed_structures_exact_legacy",
                "per_repeat": per_repeat,
                "pooled_descriptive": {
                    "baseline_numerator": bn_total,
                    "baseline_reconstructed_denominator": br_total,
                    "baseline_rate": bn_total / br_total,
                    "candidate_numerator": cn_total,
                    "candidate_reconstructed_denominator": cr_total,
                    "candidate_rate": cn_total / cr_total,
                    "interpretation": "descriptive point estimate; inference uses hierarchical bootstrap",
                },
                "hierarchical_paired_bootstrap": conditional_bootstrap(
                    baseline_reconstructed,
                    baseline_numerator,
                    candidate_reconstructed,
                    candidate_numerator,
                    draws=draws,
                    seed=seed + 10000 + comparison_index * 1000 + endpoint_index,
                ),
                "mcnemar_not_used_for_conditional_rate": True,
            }

    report = {
        "schema": "h1_plan1200_r03_b3_prepost_repeats3_terminal_v3",
        "status": "complete",
        "run_id": config["run_id"],
        "paired_seed_namespace": config["paired_seed_namespace"],
        "scientific_question": "R03 versus B3 and pre versus post model_494 under three independently sampled P0 plan cohorts",
        "planner": planner,
        "attempts_per_arm_repeat_stage": DENOMINATOR,
        "process_repeats": 3,
        "raw_planner_attempts_per_repeat": 1200,
        "stage_rows": stage_rows,
        "repeat_reports": repeat_reports,
        "paired_all_attempt_inference": all_attempt_inference,
        "paired_reconstructed_denominator_inference": conditional_inference,
        "pooled_3000_descriptive": pooled,
        "statistical_contract": {
            "primary_replication_unit": "independently sampled P0 plan batch",
            "paired_ordinals_within_repeat": True,
            "per_repeat_exact_mcnemar": True,
            "mcnemar_denominator": "all_1000_attempts",
            "hierarchical_paired_bootstrap_draws": draws,
            "headline_sun_denominator": "reconstructed_structures_exact_legacy",
            "secondary_sun_denominator": "all_1000_attempts",
            "conditional_sun_ratio_bootstrap": True,
            "pooled_3000_is_descriptive_only": True,
        },
        "prompt_contract": {
            "P0_raw_rich_seven_line_forwarded": False,
            "model_visible_prompt": "historical_r5c_plan_state_json_exact_length",
            "canonical_charge_bucket_visible": True,
            "B3_training_rich_plan_visibility": "collator_sidecar_only",
        },
        "artifacts": {
            "planner_terminal": identity(run_root / "planner_terminal_report.json"),
            "body_preflight": identity(run_root / "status/body_preflight_report.json"),
            "body_submission": identity(run_root / "status/body_submission_record.json"),
            "mp_completion": identity(run_root / "mp_cache/completion_manifest.json"),
            "mp_cache": identity(run_root / "mp_cache/completed_mp_hull_cache.jsonl"),
        },
        "source_manifest_sha256": args.source_manifest_sha256,
        "retry_replacement_repair_filter_rerank": False,
        "formal_promotion": False,
        "automatic_training": False,
        "automatic_checkpoint_reselection": False,
        "automatic_rl": False,
    }
    return report


def write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        if not value.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = assemble(args)
        write_json_exclusive(args.output.resolve(), report)
        write_text_exclusive(args.markdown.resolve(), markdown(report))
        with (args.run_root.resolve() / "status/body_assembly_SUCCESS").open(
            "x", encoding="ascii"
        ) as handle:
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"status": "complete", "output": str(args.output), "markdown": str(args.markdown)}, sort_keys=True))
    except Exception as exc:  # noqa: BLE001 - terminal evidence must survive.
        failure = {
            "schema": "h1_plan1200_r03_b3_prepost_repeats3_terminal_failure_v3",
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "automatic_retry": False,
            "automatic_training": False,
            "automatic_rl": False,
        }
        if not args.output.resolve().exists():
            write_json_exclusive(args.output.resolve(), failure)
        if not args.markdown.resolve().exists():
            write_text_exclusive(
                args.markdown.resolve(),
                "# P0 Plan1200 × R03/B3 pre/post — Engineering failure\n\n"
                + f"- Error: `{type(exc).__name__}: {exc}`\n"
                + "- Fail-closed: no retry, replacement, repair, training, or RL was launched.\n",
            )
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        raise SystemExit(3) from None


if __name__ == "__main__":
    main()
