#!/usr/bin/env python3
"""Build the single complete V4 all-attempt/native1000 Markdown report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from native_protocol import identity, read_json, write_json_exclusive
from protocol import require_source_manifest


FIXED_DENOMINATOR = 1000
MAIN_PROTOCOL = "V4_REPLAY_OF_V3_ALL_ATTEMPT_1000"
NATIVE_PROTOCOL = "V4_CRYSLLMGEN_NATIVE_SUCCESS1000"


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


def escape(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    materialized = [tuple(row) for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(escape(value) for value in row) + " |"
        for row in materialized
    )
    return "\n".join(lines)


def count_cell(count: int, denominator: int = FIXED_DENOMINATOR) -> str:
    return f"{count}/{denominator} ({100.0 * count / denominator:.3f}%)"


def reconstructed_rate_cell(count: int, denominator: int) -> str:
    if denominator <= 0:
        return "NA"
    return f"{count}/{denominator} ({100.0 * count / denominator:.3f}%)"


def legacy_component(row: Mapping[str, Any], name: str) -> dict[str, Any]:
    exact = row["sun_exact_legacy_reconstructed_denominator"]
    component = exact.get(name)
    if not isinstance(component, Mapping):
        raise ValueError(f"missing exact-legacy {name} object")
    return dict(component)


def normalized_rows(
    main: Mapping[str, Any], native: Mapping[str, Any]
) -> list[dict[str, Any]]:
    planner_by_repeat = {
        int(item["repeat"]): item for item in main["planner"]["repeats"]
    }
    if set(planner_by_repeat) != {0, 1, 2}:
        raise ValueError("planner repeat coverage changed")
    rows: list[dict[str, Any]] = []
    for source in main["stage_rows"]:
        repeat = int(source["repeat"])
        stage = str(source["stage"])
        generation_succeeded = int(source["generation_succeeded"])
        rows.append(
            {
                "protocol_type": MAIN_PROTOCOL,
                "protocol_label": "V3 fixed-all-attempt semantics; immutable V4 repair execution",
                "arm": str(source["arm"]),
                "repeat": repeat,
                "stage": stage,
                "planner_raw": int(main["raw_planner_attempts_per_repeat"]),
                "planner_parse_successes": int(
                    planner_by_repeat[repeat]["parse_successes"]
                ),
                "candidate_pool_count": int(
                    planner_by_repeat[repeat]["parse_successes"]
                ),
                "candidate_attempts_through_completion": FIXED_DENOMINATOR,
                "terminal_generation_ledger": FIXED_DENOMINATOR,
                "body_successes": generation_succeeded,
                "body_failures": FIXED_DENOMINATOR - generation_succeeded,
                "selected_body_successes": None,
                "diffusion_inputs": generation_succeeded
                if stage == "post_model494"
                else 0,
                "post_refine_ledger": FIXED_DENOMINATOR
                if stage == "post_model494"
                else 0,
                "direct_counts": source["direct_counts"],
                "cov_precision_percent": source["cov_precision_percent"],
                "cov_recall_percent": source["cov_recall_percent"],
                "sun_counts": source["sun_counts"],
                "sun_exact_legacy_reconstructed_denominator": source[
                    "sun_exact_legacy_reconstructed_denominator"
                ],
                "direct_native_report_complete": source[
                    "direct_native_report_complete"
                ],
                "sun_native_summary_complete": source["sun_native_summary_complete"],
                "sun_diagnostics": source["sun_diagnostics"],
                "failure_detail": source["direct_failure_reasons"],
                "artifacts": source["artifacts"],
            }
        )
    for source in native["stage_rows"]:
        repeat = int(source["repeat"])
        rows.append(
            {
                "protocol_type": NATIVE_PROTOCOL,
                "protocol_label": "upstream CrysLLMGen first-1000 body successes; all 1000 refined",
                "arm": str(source["arm"]),
                "repeat": repeat,
                "stage": "post_model494",
                "planner_raw": int(main["raw_planner_attempts_per_repeat"]),
                "planner_parse_successes": int(
                    planner_by_repeat[repeat]["parse_successes"]
                ),
                "candidate_pool_count": int(source["candidate_pool_count"]),
                "candidate_attempts_through_completion": int(
                    source["candidate_attempts_through_completion"]
                ),
                "terminal_generation_ledger": int(
                    source["candidate_attempts_through_completion"]
                ),
                "body_successes": FIXED_DENOMINATOR,
                "body_failures": int(source["body_failures_before_completion"]),
                "selected_body_successes": FIXED_DENOMINATOR,
                "diffusion_inputs": FIXED_DENOMINATOR,
                "post_refine_ledger": int(source["refined_structures"]),
                "selected_prefix_count": int(source["selected_prefix_count"]),
                "selected_reserve_count": int(source["selected_reserve_count"]),
                "unused_candidates_after_completion": int(
                    source["unused_candidates_after_completion"]
                ),
                "direct_counts": source["direct_counts"],
                "cov_precision_percent": source["cov_precision_percent"],
                "cov_recall_percent": source["cov_recall_percent"],
                "sun_counts": source["sun_counts"],
                "sun_exact_legacy_reconstructed_denominator": source[
                    "sun_exact_legacy_reconstructed_denominator"
                ],
                "direct_native_report_complete": source[
                    "direct_native_report_complete"
                ],
                "sun_native_summary_complete": source["sun_native_summary_complete"],
                "sun_diagnostics": source["sun_diagnostics"],
                "failure_detail": source["selection_failure_classes"],
                "artifacts": source["artifacts"],
            }
        )
    if len(rows) != 18:
        raise ValueError(f"combined row count changed: {len(rows)}")
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    expected_main = {
        (arm, repeat, stage)
        for arm in ("R03", "B3")
        for repeat in range(3)
        for stage in ("pre_model494", "post_model494")
    }
    expected_native = {
        (arm, repeat, "post_model494")
        for arm in ("R03", "B3")
        for repeat in range(3)
    }
    observed_main = {
        (row["arm"], row["repeat"], row["stage"])
        for row in rows
        if row["protocol_type"] == MAIN_PROTOCOL
    }
    observed_native = {
        (row["arm"], row["repeat"], row["stage"])
        for row in rows
        if row["protocol_type"] == NATIVE_PROTOCOL
    }
    if observed_main != expected_main or observed_native != expected_native:
        raise ValueError("combined protocol/arm/repeat/stage coverage changed")
    for row in rows:
        direct = row["direct_counts"]
        sun = row["sun_counts"]
        strict = legacy_component(row, "strict")
        meta = legacy_component(row, "meta_like")
        reconstructed = int(sun["reconstructed"])
        if (
            int(strict["reconstructed"]) != reconstructed
            or int(meta["reconstructed"]) != reconstructed
            or int(strict["stable"]["numerator"])
            != int(sun["strict_full_sun"])
            or int(meta["stable"]["numerator"])
            != int(sun["meta_full_sun"])
            or int(row["post_refine_ledger"])
            not in {0, FIXED_DENOMINATOR}
            or any(int(direct[key]) not in range(FIXED_DENOMINATOR + 1) for key in (
                "composition_valid", "structure_valid", "joint_valid"
            ))
        ):
            raise ValueError(
                f"metric/denominator contract changed for {row['protocol_type']} "
                f"{row['arm']} repeat {row['repeat']} {row['stage']}"
            )
        if row["protocol_type"] == NATIVE_PROTOCOL and (
            int(row["selected_body_successes"]) != FIXED_DENOMINATOR
            or int(row["diffusion_inputs"]) != FIXED_DENOMINATOR
            or int(row["post_refine_ledger"]) != FIXED_DENOMINATOR
        ):
            raise ValueError("native full-1000 refine denominator changed")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    run_root = args.run_root.resolve()
    config = read_json(args.config.resolve())
    if (
        config.get("schema")
        != "h1_p0_plan1200_r03_b3_crysllmgen_native_post1000_config_v2"
        or run_root != Path(str(config["run_root"])).resolve()
        or args.output.resolve() != run_root / "COMBINED_TERMINAL_REPORT.json"
        or args.markdown.resolve() != run_root / "COMBINED_RESULTS_COMPLETE.md"
    ):
        raise ValueError("combined output identity changed")
    main_path = run_root / "terminal_report.json"
    native_path = run_root / "crysllmgen_native1000/terminal_report.json"
    main = read_json(main_path)
    native = read_json(native_path)
    if (
        main.get("status") != "complete"
        or native.get("status") != "complete"
        or not (run_root / "status/native1000_panel_SUCCESS").is_file()
        or not (run_root / "status/body_assembly_exit_code.txt").is_file()
        or (run_root / "status/body_assembly_exit_code.txt").read_text(
            encoding="ascii"
        ).strip()
        != "0"
    ):
        raise ValueError("main/native terminal evidence is incomplete")
    rows = normalized_rows(main, native)
    validate_rows(rows)
    return {
        "schema": "h1_plan1200_v4_combined_all_attempt_native1000_terminal_v1",
        "status": "complete",
        "run_id": config["run_id"],
        "protocol_definitions": {
            MAIN_PROTOCOL: {
                "lineage": "V3 scientific protocol executed in immutable V4 after schema-only consumer repair",
                "denominator": "all_1000_body_generation_attempts",
                "stages": ["pre_model494", "post_model494"],
                "paired_inference": True,
            },
            NATIVE_PROTOCOL: {
                "lineage": "upstream CrysLLMGen success counter reproduced in immutable V4",
                "denominator": "first_1000_body_successes_then_refine_all_1000",
                "stages": ["post_model494"],
                "paired_inference": "only_if_R03_B3_selected_sets_identical",
            },
        },
        "rows": rows,
        "planner": main["planner"],
        "all_attempt_statistics": {
            "paired_all_attempt_inference": main["paired_all_attempt_inference"],
            "paired_reconstructed_denominator_inference": main[
                "paired_reconstructed_denominator_inference"
            ],
            "pooled_3000_descriptive": main["pooled_3000_descriptive"],
            "statistical_contract": main["statistical_contract"],
        },
        "native_statistics": {
            "selection_overlap": native["selection_overlap"],
            "pooled_3000_descriptive": native["pooled_3000_descriptive"],
            "statistical_contract": native["statistical_contract"],
        },
        "repeat_reports_complete": {
            "all_attempt": main["repeat_reports"],
            "native1000": native["repeat_reports"],
        },
        "artifacts": {
            "main_terminal": identity(main_path),
            "native_terminal": identity(native_path),
            "main_markdown": identity(run_root / "RESULTS_COMPLETE.md"),
            "native_markdown": identity(
                run_root / "CRYSLLMGEN_NATIVE1000_RESULTS_COMPLETE.md"
            ),
            "input_import_contract": identity(
                run_root / "INPUT_IMPORT_CONTRACT.json"
            ),
            "input_import_report": identity(
                run_root / "status/v4_input_import_report.json"
            ),
        },
        "source_manifest_sha256": args.source_manifest_sha256,
        "headline_sun_denominator": "reconstructed_structures_exact_legacy",
        "secondary_sun_denominator": "fixed_1000",
        "sample_retry_replacement_repair_filter_rerank": False,
        "automatic_training": False,
        "automatic_rl": False,
    }


def main_metric_table(rows: list[dict[str, Any]]) -> str:
    output = []
    for row in rows:
        direct = row["direct_counts"]
        sun = row["sun_counts"]
        output.append(
            (
                row["protocol_type"],
                row["arm"],
                row["repeat"],
                row["stage"],
                row["candidate_pool_count"],
                row["candidate_attempts_through_completion"],
                row["body_successes"],
                row["body_failures"],
                row["selected_body_successes"],
                row["diffusion_inputs"],
                row["post_refine_ledger"],
                count_cell(int(direct["composition_valid"])),
                count_cell(int(direct["structure_valid"])),
                count_cell(int(direct["joint_valid"])),
                row["cov_precision_percent"],
                row["cov_recall_percent"],
                count_cell(int(sun["reconstructed"])),
                int(sun["hull_evaluated"]),
                int(row["sun_diagnostics"]["hull_unknown"]),
            )
        )
    return table(
        [
            "Protocol", "Arm", "Repeat", "Stage", "Candidate pool",
            "Tried through completion", "Body success", "Body failure",
            "Selected success", "Diffusion inputs", "Post ledger", "Comp",
            "Struct", "Joint", "COV-P %", "COV-R %", "Reconstructed",
            "Hull evaluated", "Hull unknown",
        ],
        output,
    )


def sun_table(rows: list[dict[str, Any]], component_name: str) -> str:
    output = []
    count_key = (
        "strict_full_sun" if component_name == "strict" else "meta_full_sun"
    )
    for row in rows:
        component = legacy_component(row, component_name)
        count = int(row["sun_counts"][count_key])
        reconstructed = int(component["reconstructed"])
        output.append(
            (
                row["protocol_type"],
                row["arm"],
                row["repeat"],
                row["stage"],
                int(component["novel"]["numerator"]),
                int(component["unique"]["numerator"]),
                int(component["novel_unique"]["numerator"]),
                int(component["e_hull_evaluated"]["numerator"]),
                int(component["e_hull_unknown"]["numerator"]),
                count,
                reconstructed_rate_cell(count, reconstructed),
                component["full_sun_lower_bound_percent"],
                component["coverage_adjusted_percent"],
                count_cell(count),
            )
        )
    return table(
        [
            "Protocol", "Arm", "Repeat", "Stage", "Novel", "Unique",
            "Novel∩Unique", "Hull evaluated", "Hull unknown", "Stable numerator",
            "Stable / reconstructed headline", "Exact lower-bound %",
            "Coverage-adjusted %", "Stable / fixed1000 secondary",
        ],
        output,
    )


def flattened_rows(
    rows: list[dict[str, Any]], field: str
) -> list[tuple[Any, ...]]:
    output: list[tuple[Any, ...]] = []
    for row in rows:
        for path, value in sorted(flatten(row[field]).items()):
            output.append(
                (
                    row["protocol_type"],
                    row["arm"],
                    row["repeat"],
                    row["stage"],
                    path,
                    value,
                )
            )
    return output


def markdown(report: Mapping[str, Any]) -> str:
    rows = list(report["rows"])
    planner_rows = [
        (
            item["repeat"],
            item["planner_seed"],
            item["raw_attempts"],
            item["parse_successes"],
            item["parse_failures"],
            item["reserve_parse_success_count"],
            item["cohort1000"]["sha256"],
        )
        for item in report["planner"]["repeats"]
    ]
    lines = [
        "# P0 Plan1200 × R03/B3：V4 完整合并结果",
        "",
        "## 执行与统计口径",
        "",
        "原 V3 在 generation 前因 consumer 错误要求不存在的顶层 `parsed` 字段而 fail-closed，没有产生科学指标。本报告中的 all-attempt 数值来自新的不可变 V4；V4 byte-for-byte 复用 V3 的三批 planner cohort、候选池与两套 completed MP cache，只修复 consumer/preflight 的 cohort schema 契约。",
        "",
        f"- `{MAIN_PROTOCOL}`：V3 的科学协议；每个 arm×repeat 固定 1,000 次 body attempt，失败保留，pre/post-model494 都评估。正式配对 McNemar 与 50k hierarchical paired bootstrap 以此为准。",
        f"- `{NATIVE_PROTOCOL}`：按上游 CrysLLMGen `process_one` success counter，沿冻结候选顺序累计前 1,000 个 body success，并让这 1,000 个全部进入 model494 refine；只报告 post-model494。若 R03/B3 入选集合不同，跨臂只作描述。",
        "- S.U.N. headline 始终是复现脚本的 reconstructed-structures exact-legacy 分母；固定 1,000 同时作为保守 secondary。evaluated/stable 分母只作诊断。",
        "",
        "## 三批 P0 planner 冻结证据",
        "",
        table(
            [
                "Repeat", "Seed", "Raw", "Parse success", "Parse failure",
                "Reserve", "Cohort SHA256",
            ],
            planner_rows,
        ),
        "",
        "## 合并主表",
        "",
        main_metric_table(rows),
        "",
        "## Strict S.U.N. 独立表",
        "",
        sun_table(rows, "strict"),
        "",
        "## Meta S.U.N. 独立表",
        "",
        sun_table(rows, "meta_like"),
        "",
        "## Native1000 入选 plan 集合重叠",
        "",
        table(
            [
                "Repeat", "R03 selected", "B3 selected", "Intersection",
                "R03 only", "B3 only", "Union", "Jaccard", "Paired native?",
            ],
            (
                (
                    item["repeat"], item["R03_selected"], item["B3_selected"],
                    item["intersection"], item["R03_only"], item["B3_only"],
                    item["union"], item["jaccard"],
                    item["paired_cross_arm_native_inference_valid"],
                )
                for item in report["native_statistics"]["selection_overlap"]
            ),
        ),
        "",
        "## All-attempt 配对统计：全部字段",
        "",
        table(
            ["Statistical object", "Field", "Value"],
            (
                (object_name, path, value)
                for object_name in (
                    "paired_all_attempt_inference",
                    "paired_reconstructed_denominator_inference",
                    "pooled_3000_descriptive",
                    "statistical_contract",
                )
                for path, value in sorted(
                    flatten(report["all_attempt_statistics"][object_name]).items()
                )
            ),
        ),
        "",
        "## Native1000 描述统计：全部字段",
        "",
        table(
            ["Statistical object", "Field", "Value"],
            (
                (object_name, path, value)
                for object_name in (
                    "selection_overlap",
                    "pooled_3000_descriptive",
                    "statistical_contract",
                )
                for path, value in sorted(
                    flatten(report["native_statistics"][object_name]).items()
                )
            ),
        ),
        "",
        "## CrysLLMGen complete：全部字段",
        "",
        table(
            ["Protocol", "Arm", "Repeat", "Stage", "Field", "Value"],
            flattened_rows(rows, "direct_native_report_complete"),
        ),
        "",
        "## S.U.N. exact legacy：全部字段",
        "",
        table(
            ["Protocol", "Arm", "Repeat", "Stage", "Field", "Value"],
            flattened_rows(
                rows, "sun_exact_legacy_reconstructed_denominator"
            ),
        ),
        "",
        "## S.U.N. native summary：全部字段",
        "",
        table(
            ["Protocol", "Arm", "Repeat", "Stage", "Field", "Value"],
            flattened_rows(rows, "sun_native_summary_complete"),
        ),
        "",
        "## Hull / energy / failure 诊断：全部字段",
        "",
        table(
            ["Protocol", "Arm", "Repeat", "Stage", "Object", "Field", "Value"],
            (
                (
                    row["protocol_type"], row["arm"], row["repeat"], row["stage"],
                    object_name, path, value,
                )
                for row in rows
                for object_name in ("sun_diagnostics", "failure_detail")
                for path, value in sorted(flatten(row[object_name]).items())
            ),
        ),
        "",
        "## Attempt ledger 与证据 SHA：全部字段",
        "",
        table(
            ["Protocol", "Arm", "Repeat", "Stage", "Artifact field", "Value"],
            flattened_rows(rows, "artifacts"),
        ),
        "",
        "## 每臂每重复终态报告：全部字段",
        "",
        table(
            ["Protocol group", "Field", "Value"],
            (
                (group, path, value)
                for group, value_object in report["repeat_reports_complete"].items()
                for path, value in sorted(flatten(value_object).items())
            ),
        ),
        "",
        "## 不可越界项",
        "",
        "- 三批 plan 不复用；同批 R03/B3 共享相同 all-attempt cohort。",
        "- 无 same-plan retry、replacement、repair、filter、rerank、新 planner draw、training、promotion 或 RL。",
        "- Pooled3000 只作描述；三批独立 planner draw 才是重复单位。",
        "- CrysLLMGen/Direct 的 comp_valid 只作输出描述，不归因于 DLM 训练标签。",
        "",
    ]
    return "\n".join(lines)


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
        report = build_report(args)
        text = markdown(report)
        write_json_exclusive(args.output.resolve(), report)
        write_text_exclusive(args.markdown.resolve(), text)
        marker = args.run_root.resolve() / "status/combined_assembly_SUCCESS"
        with marker.open("x", encoding="ascii") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        print(
            json.dumps(
                {
                    "status": "complete",
                    "rows": len(report["rows"]),
                    "output": str(args.output),
                    "markdown": str(args.markdown),
                },
                sort_keys=True,
            )
        )
    except Exception as exc:  # noqa: BLE001
        failure = {
            "schema": "h1_plan1200_v4_combined_failure_v1",
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
                "# V4 combined report — Engineering failure\n\n"
                f"- Error: `{type(exc).__name__}: {exc}`\n"
                "- Fail-closed: no retry, replacement, repair, training, or RL.\n",
            )
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        raise SystemExit(3) from None


if __name__ == "__main__":
    main()
