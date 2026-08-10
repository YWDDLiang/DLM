#!/usr/bin/env python3
"""Assemble the complete CrysLLMGen-native post-refine 1,000 panel."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from native_protocol import (
    ARMS,
    NATIVE_DENOMINATOR,
    REPEATS,
    identity,
    read_json,
    read_jsonl,
    write_json_exclusive,
)
from protocol import require_source_manifest


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    rows: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.update(flatten(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            rows.update(flatten(item, path))
    else:
        rows[prefix] = value
    return rows


def find_numeric(native: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    normalized = {
        key.lower().replace("-", "_"): value for key, value in flatten(native).items()
    }
    for name in names:
        for key, value in normalized.items():
            if (
                key.endswith(name)
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                return float(value)
    return None


def percent_display(value: float | None) -> str:
    if value is None:
        return "NA"
    percent = value * 100.0 if abs(value) <= 1.0 else value
    return f"{percent:.4f}"


def escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(escape(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def rate_cell(count: int, denominator: int = NATIVE_DENOMINATOR) -> str:
    return f"{count}/{denominator} ({100.0 * count / denominator:.3f}%)"


def write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        if not value.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(args.config.resolve())
    run_root = args.run_root.resolve()
    native_root = run_root / "crysllmgen_native1000"
    if (
        config.get("schema")
        != "h1_p0_plan1200_r03_b3_crysllmgen_native_post1000_config_v2"
        or run_root != Path(str(config.get("run_root"))).resolve()
        or args.output.resolve() != native_root / "terminal_report.json"
        or args.markdown.resolve()
        != run_root / "CRYSLLMGEN_NATIVE1000_RESULTS_COMPLETE.md"
        or args.combined_markdown.resolve()
        != run_root / "RESULTS_COMPLETE_WITH_NATIVE1000.md"
    ):
        raise ValueError("native assembly identity changed")

    submission = read_json(run_root / "status/native1000_submission_record.json")
    preflight = read_json(run_root / "status/native1000_preflight_report.json")
    completion = read_json(run_root / "native_mp_cache/completion_manifest.json")
    main_terminal = read_json(run_root / "terminal_report.json")
    main_markdown = run_root / "RESULTS_COMPLETE.md"
    if (
        submission.get("status") != "complete"
        or submission.get("source_manifest_sha256") != args.source_manifest_sha256
        or preflight.get("status") != "pass"
        or preflight.get("native_source_manifest_sha256")
        != args.source_manifest_sha256
        or completion.get("status") != "complete_all_wanted_chemsys_resolved"
        or main_terminal.get("status") != "complete"
        or not main_markdown.is_file()
    ):
        raise ValueError("main/native preflight, cache, submission, or terminal evidence changed")

    stage_rows: list[dict[str, Any]] = []
    repeat_reports: list[dict[str, Any]] = []
    selected_ranks: dict[tuple[str, int], set[int]] = {}
    for arm in ARMS:
        for repeat in REPEATS:
            root = native_root / "arms" / arm / "repeats" / str(repeat)
            exit_code = run_root / f"status/native1000_{arm}_repeat_{repeat}_exit_code.txt"
            if (
                not (root / "_SUCCESS").is_file()
                or (root / "_FAILURE").exists()
                or not exit_code.is_file()
                or exit_code.read_text(encoding="ascii").strip() != "0"
            ):
                raise ValueError(f"native {arm} repeat {repeat} is not an engineering success")
            repeat_report = read_json(root / "native_repeat_report.json")
            stage = read_json(root / "post_model494/evaluation/stage_report.json")
            selection = read_json(root / "selected_body1000/selection_report.json")
            mapping = sorted(
                read_jsonl(root / "selected_body1000/selection_mapping.jsonl"),
                key=lambda row: int(row["native_ordinal"]),
            )
            ranks = {int(row["candidate_rank"]) for row in mapping}
            if (
                repeat_report.get("status") != "complete"
                or repeat_report.get("ok") is not True
                or int(repeat_report.get("refined_structures", -1))
                != NATIVE_DENOMINATOR
                or stage.get("status") != "complete"
                or int(stage.get("generation_succeeded", -1))
                != NATIVE_DENOMINATOR
                or int(selection.get("selected_body_successes", -1))
                != NATIVE_DENOMINATOR
                or len(mapping) != NATIVE_DENOMINATOR
                or len(ranks) != NATIVE_DENOMINATOR
            ):
                raise ValueError(f"native {arm} repeat {repeat} report changed")
            selected_ranks[(arm, repeat)] = ranks
            repeat_reports.append(repeat_report)
            direct_native = stage["direct_native_report_complete"]
            cov_precision = find_numeric(
                direct_native.get("metrics_unchanged_upstream") or {},
                ("cov_precision", "coverage_precision", "cov_p"),
            )
            cov_recall = find_numeric(
                direct_native.get("metrics_unchanged_upstream") or {},
                ("cov_recall", "coverage_recall", "cov_r"),
            )
            stage_rows.append(
                {
                    "arm": arm,
                    "repeat": repeat,
                    "candidate_pool_count": int(selection["candidate_pool_count"]),
                    "candidate_attempts_through_completion": int(
                        selection["candidate_attempts_through_completion"]
                    ),
                    "body_failures_before_completion": int(
                        selection["body_failures_before_completion"]
                    ),
                    "selected_prefix_count": int(selection["selected_prefix_count"]),
                    "selected_reserve_count": int(selection["selected_reserve_count"]),
                    "unused_candidates_after_completion": int(
                        selection["unused_candidates_after_completion"]
                    ),
                    "refined_structures": NATIVE_DENOMINATOR,
                    "direct_counts": stage["direct_counts"],
                    "direct_rates_all_1000": stage["direct_rates_all_attempts"],
                    "direct_native_report_complete": direct_native,
                    "cov_precision_raw": cov_precision,
                    "cov_recall_raw": cov_recall,
                    "cov_precision_percent": percent_display(cov_precision),
                    "cov_recall_percent": percent_display(cov_recall),
                    "sun_counts": stage["sun_counts"],
                    "sun_rates_all_1000": stage["sun_rates_all_attempts"],
                    "sun_exact_legacy_reconstructed_denominator": stage[
                        "sun_exact_legacy_reconstructed_denominator"
                    ],
                    "sun_native_summary_complete": stage["sun_native_summary_complete"],
                    "sun_diagnostics": stage["sun_diagnostics"],
                    "selection_failure_classes": selection[
                        "failure_classes_before_completion"
                    ],
                    "artifacts": {
                        "native_repeat_report": identity(
                            root / "native_repeat_report.json"
                        ),
                        "selection_mapping": identity(
                            root / "selected_body1000/selection_mapping.jsonl"
                        ),
                        "generation": identity(
                            root / "post_model494/generation/generation.jsonl"
                        ),
                        "direct_attempts": identity(
                            root
                            / "post_model494/evaluation/crysllmgen_metrics/attempt_metrics.jsonl"
                        ),
                        "sun_attempts": identity(
                            root
                            / "post_model494/evaluation/r5c_a100_sun/attempt_results.jsonl"
                        ),
                        "stage_report": identity(
                            root / "post_model494/evaluation/stage_report.json"
                        ),
                    },
                }
            )

    overlaps: list[dict[str, Any]] = []
    for repeat in REPEATS:
        r03 = selected_ranks[("R03", repeat)]
        b3 = selected_ranks[("B3", repeat)]
        intersection = len(r03 & b3)
        union = len(r03 | b3)
        overlaps.append(
            {
                "repeat": repeat,
                "R03_selected": len(r03),
                "B3_selected": len(b3),
                "intersection": intersection,
                "R03_only": len(r03 - b3),
                "B3_only": len(b3 - r03),
                "union": union,
                "jaccard": intersection / union,
                "paired_cross_arm_native_inference_valid": r03 == b3,
            }
        )

    aggregates: list[dict[str, Any]] = []
    for arm in ARMS:
        rows = [row for row in stage_rows if row["arm"] == arm]
        direct_keys = tuple(rows[0]["direct_counts"])
        sun_keys = tuple(rows[0]["sun_counts"])
        cov_p = [row["cov_precision_raw"] for row in rows if row["cov_precision_raw"] is not None]
        cov_r = [row["cov_recall_raw"] for row in rows if row["cov_recall_raw"] is not None]
        aggregates.append(
            {
                "arm": arm,
                "refined_structures": 3 * NATIVE_DENOMINATOR,
                "candidate_attempts_total": sum(
                    row["candidate_attempts_through_completion"] for row in rows
                ),
                "body_failures_total": sum(
                    row["body_failures_before_completion"] for row in rows
                ),
                "direct_counts_pooled_descriptive": {
                    key: sum(int(row["direct_counts"][key]) for row in rows)
                    for key in direct_keys
                },
                "sun_counts_pooled_descriptive": {
                    key: sum(int(row["sun_counts"][key]) for row in rows)
                    for key in sun_keys
                },
                "cov_precision_repeat_mean_raw": statistics.fmean(cov_p) if cov_p else None,
                "cov_precision_repeat_range_raw": [min(cov_p), max(cov_p)] if cov_p else None,
                "cov_recall_repeat_mean_raw": statistics.fmean(cov_r) if cov_r else None,
                "cov_recall_repeat_range_raw": [min(cov_r), max(cov_r)] if cov_r else None,
                "interpretation": "descriptive; independently sampled plan batch is the replication unit",
            }
        )

    report = {
        "schema": "h1_plan1200_crysllmgen_native_post1000_terminal_v2",
        "status": "complete",
        "run_id": config["run_id"],
        "scientific_question": "R03 versus B3 under the upstream CrysLLMGen denominator after model_494 refine",
        "sampling_contract": "first_1000_process_one_successes_in_frozen_candidate_order_then_refine_all_1000",
        "refined_structures_per_arm_repeat": NATIVE_DENOMINATOR,
        "stage": "post_model494",
        "stage_rows": stage_rows,
        "repeat_reports": repeat_reports,
        "selection_overlap": overlaps,
        "pooled_3000_descriptive": aggregates,
        "statistical_contract": {
            "primary_all_attempt_inference": "retained_in_V4_all_attempt_terminal_report",
            "native_cross_arm_selection_can_differ": True,
            "native_cross_arm_values": "descriptive_unless_selected_candidate_sets_are_identical",
            "sun_headline_denominator": "reconstructed_structures_exact_legacy",
            "sun_secondary_denominator": "all_1000_refined_structures",
            "pooled_3000_is_descriptive_only": True,
        },
        "upstream_sampling_contract": config["upstream_sampling_contract"],
        "artifacts": {
            "main_V4_all_attempt_terminal": identity(run_root / "terminal_report.json"),
            "main_V4_all_attempt_markdown": identity(main_markdown),
            "native_preflight": identity(
                run_root / "status/native1000_preflight_report.json"
            ),
            "native_submission": identity(
                run_root / "status/native1000_submission_record.json"
            ),
            "native_mp_completion": identity(
                run_root / "native_mp_cache/completion_manifest.json"
            ),
            "native_mp_cache": identity(
                run_root / "native_mp_cache/completed_mp_hull_cache.jsonl"
            ),
        },
        "source_manifest_sha256": args.source_manifest_sha256,
        "same_plan_retry": False,
        "stochastic_replacement": False,
        "additional_planner_draw": False,
        "repair_filter_rerank": False,
        "automatic_training": False,
        "automatic_rl": False,
    }
    return report


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# CrysLLMGen-native：P0 Plan1200 × R03/B3 × post-model_494 完整 1000",
        "",
        "## 口径",
        "",
        "本补充表严格复现上游 `crysllmgen_sample.py` 的计数方式：只有经 `process_one` 成功的 body 候选才推进 `num_samples`；每个 arm×repeat 按冻结候选顺序累计恰好 1,000 个成功候选，并让这完整 1,000 个全部经过 model_494、800-step diffusion refine，再计算完整 CrysLLMGen 与 S.U.N.。",
        "",
        "它不替代 V4 的 all-attempt 主视图：该视图固定 1,000 次 body attempt 并保留失败，适合无偏配对推断；native1000 是论文/上游基数对齐视图。由于 R03/B3 的 body failure 位置可能不同，native1000 的入选 plan 集合可能不同，因此跨臂差值只作描述；逐批精确 McNemar 与 50k hierarchical paired bootstrap 仍以 all-attempt 视图为准。",
        "",
        "## 生成与完整 refine 审计",
        "",
    ]
    audit_rows = []
    for row in report["stage_rows"]:
        audit_rows.append(
            (
                row["arm"],
                row["repeat"],
                row["candidate_pool_count"],
                row["candidate_attempts_through_completion"],
                row["body_failures_before_completion"],
                row["selected_prefix_count"],
                row["selected_reserve_count"],
                row["unused_candidates_after_completion"],
                rate_cell(row["refined_structures"]),
            )
        )
    lines.extend(
        [
            table(
                [
                    "Arm", "Repeat", "Frozen candidates", "Tried through completion",
                    "Body failures", "Selected prefix", "Selected reserve", "Unused reserve",
                    "Diffusion-refined",
                ],
                audit_rows,
            ),
            "",
            "## post-model_494 完整主表",
            "",
        ]
    )
    metric_rows = []
    for row in report["stage_rows"]:
        direct = row["direct_counts"]
        sun = row["sun_counts"]
        metric_rows.append(
            (
                row["arm"],
                row["repeat"],
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
                f"{sun['hull_evaluated']}/1000",
                row["sun_diagnostics"]["hull_unknown"],
            )
        )
    lines.extend(
        [
            table(
                [
                    "Arm", "Repeat", "Comp", "Struct", "Joint", "COV-P %", "COV-R %",
                    "Reconstructed", "Novel", "Unique", "Novel∩Unique", "Strict S.U.N.",
                    "Meta S.U.N.", "Hull evaluated", "Hull unknown",
                ],
                metric_rows,
            ),
            "",
            "## R03/B3 入选 plan 集合重叠",
            "",
            table(
                ["Repeat", "Intersection", "R03-only", "B3-only", "Union", "Jaccard", "Paired native?"],
                (
                    (
                        row["repeat"], row["intersection"], row["R03_only"], row["B3_only"],
                        row["union"], f"{row['jaccard']:.6f}",
                        row["paired_cross_arm_native_inference_valid"],
                    )
                    for row in report["selection_overlap"]
                ),
            ),
            "",
            "## 三批汇总（pooled 3,000 仅描述）",
            "",
        ]
    )
    aggregate_rows = []
    for item in report["pooled_3000_descriptive"]:
        d = item["direct_counts_pooled_descriptive"]
        s = item["sun_counts_pooled_descriptive"]
        aggregate_rows.append(
            (
                item["arm"], item["refined_structures"], item["candidate_attempts_total"],
                item["body_failures_total"], f"{d['composition_valid']}/3000",
                f"{d['structure_valid']}/3000", f"{d['joint_valid']}/3000",
                f"{s['strict_full_sun']}/3000", f"{s['meta_full_sun']}/3000",
            )
        )
    lines.extend(
        [
            table(
                ["Arm", "Refined", "Candidate attempts", "Body failures", "Comp", "Struct", "Joint", "Strict", "Meta"],
                aggregate_rows,
            ),
            "",
            "## S.U.N. reconstructed-structure headline：全部细分字段",
            "",
        ]
    )
    legacy_rows = []
    for row in report["stage_rows"]:
        for path, value in sorted(
            flatten(row["sun_exact_legacy_reconstructed_denominator"]).items()
        ):
            legacy_rows.append((row["arm"], row["repeat"], path, value))
    lines.extend(
        [
            table(["Arm", "Repeat", "Exact-legacy field", "Value"], legacy_rows),
            "",
            "## CrysLLMGen upstream report：全部字段",
            "",
        ]
    )
    direct_rows = []
    for row in report["stage_rows"]:
        for path, value in sorted(flatten(row["direct_native_report_complete"]).items()):
            direct_rows.append((row["arm"], row["repeat"], path, value))
    lines.extend(
        [
            table(["Arm", "Repeat", "CrysLLMGen field", "Value"], direct_rows),
            "",
            "## S.U.N. native summary：全部字段",
            "",
        ]
    )
    sun_rows = []
    for row in report["stage_rows"]:
        for path, value in sorted(flatten(row["sun_native_summary_complete"]).items()):
            sun_rows.append((row["arm"], row["repeat"], path, value))
    lines.extend(
        [
            table(["Arm", "Repeat", "S.U.N. field", "Value"], sun_rows),
            "",
            "## S.U.N. hull / energy 诊断：全部字段",
            "",
        ]
    )
    diagnostic_rows = []
    for row in report["stage_rows"]:
        for path, value in sorted(flatten(row["sun_diagnostics"]).items()):
            diagnostic_rows.append((row["arm"], row["repeat"], path, value))
    lines.extend(
        [
            table(["Arm", "Repeat", "Diagnostic field", "Value"], diagnostic_rows),
            "",
            "## 边界",
            "",
            "- 每个冻结 plan 最多生成一次；没有 same-plan retry、随机 replacement、新 planner draw、repair、filter 或 rerank。",
            "- 所有入选的 1,000 个 body 成功候选都必须完成 diffusion refine；任何 refiner failure 都是工程失败，不补样。",
            "- S.U.N. headline 使用复现脚本的 reconstructed-structures exact-legacy 分母；all-refined-1000 同时作为保守 secondary。",
            "- V4 的固定 all-attempt 结果和统计推断保留为主证据；本表只回答 CrysLLMGen-native 基数下的表现。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--combined-markdown", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = assemble(args)
        native_text = markdown(report)
        main_text = (args.run_root.resolve() / "RESULTS_COMPLETE.md").read_text(
            encoding="utf-8"
        )
        write_json_exclusive(args.output.resolve(), report)
        write_text_exclusive(args.markdown.resolve(), native_text)
        write_text_exclusive(
            args.combined_markdown.resolve(),
            main_text.rstrip()
            + "\n\n---\n\n"
            + native_text,
        )
        marker = args.run_root.resolve() / "status/native1000_panel_SUCCESS"
        with marker.open("x", encoding="ascii") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(args.output),
                    "markdown": str(args.markdown),
                    "combined_markdown": str(args.combined_markdown),
                },
                sort_keys=True,
            )
        )
    except Exception as exc:  # noqa: BLE001
        failure = {
            "schema": "h1_plan1200_crysllmgen_native_post1000_failure_v2",
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
                "# CrysLLMGen-native post-refine 1000 — Engineering failure\n\n"
                f"- Error: `{type(exc).__name__}: {exc}`\n"
                "- Fail-closed: no retry, replacement, repair, training, or RL was launched.\n",
            )
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        raise SystemExit(3) from None


if __name__ == "__main__":
    main()
