#!/usr/bin/env python3
"""Write the Chinese-primary report for the structure-aware doping trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric(payload: Dict[str, Any] | None, key: str, default: Any = "NA") -> Any:
    if not payload:
        return default
    return payload.get(key, default)


def make_section(title: str, sample_metrics: Dict[str, Any] | None, eval_metrics: Dict[str, Any] | None) -> str:
    lines = [
        f"## {title}",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| parse_rate | {metric(sample_metrics, 'parse_rate')} |",
        f"| composition_exact_rate | {metric(sample_metrics, 'composition_exact_rate')} |",
        f"| graph_build_rate | {metric(sample_metrics, 'graph_build_rate')} |",
        f"| unique_expanded_structure_count | {metric(sample_metrics, 'unique_expanded_structure_count')} |",
        f"| hidden structural near-hit | {metric(metric(eval_metrics, 'generated_summary', {}), 'structural_near_hit_count') if eval_metrics else 'NA'} |",
        f"| compact baseline near-hit | {metric(metric(eval_metrics, 'compact_baseline_summary', {}), 'structural_near_hit_count') if eval_metrics else 'NA'} |",
        f"| offline_preliminary_pass | {metric(eval_metrics, 'offline_preliminary_pass')} |",
        "",
    ]
    if eval_metrics:
        lines.extend(
            [
                "### 结论边界",
                "",
                str(eval_metrics.get("claim_status")),
                "",
                "注意：本轮没有 DFT relaxation，因此 `final_property_claim_allowed=false`；不能写成“性质已经更好”。",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, default=Path("reports/20260520_structure20_vs_full80_trial_report.md"))
    args = parser.parse_args()

    struct_sample = load_json(args.run_dir / "outputs/structure20_sample/sample_metrics.json")
    struct_eval = load_json(args.run_dir / "outputs/structure20_sample/structure_similarity_eval.json")
    full80_sample = load_json(args.run_dir / "outputs/full80_sample/sample_metrics.json")
    full80_eval = load_json(args.run_dir / "outputs/full80_sample/structure_similarity_eval.json")

    lines = [
        "# Structure20 vs Full80 Doping Trial Report",
        "",
        "## 总结",
        "",
        "本报告对应 `reports/20260519_structure_aware_doping_validation_experiment_plan.md` 的一次实验尝试，比较 `DOPING_STRUCT20` compressed structural code 与 direct `DOPING_FULL80`。",
        "",
        "当前评价是 offline structural validation：检查生成结构能否展开为完整 80 原子 CIF，组成是否精确，是否与 hidden verified-good structures 在元素和结构上相似。没有统一 DFT relaxation，因此不能声称模型已经证明生成结构的性质更好。",
        "",
        "## Baseline",
        "",
        "compact-template baseline 使用已有 `runs/20260519_210000-doping-good-holdout/outputs/compact_sample_goodholdout_2048`，它只生成元素组合并套用固定模板，用作结构相似度的下限对照。",
        "",
        make_section("实验 A：DOPING_STRUCT20", struct_sample, struct_eval),
        make_section("实验 B：DOPING_FULL80", full80_sample, full80_eval),
        "## 方法评估",
        "",
        "- 如果 `DOPING_STRUCT20` 的 hidden-good structural near-hit 高于 compact-template baseline，可以写“模型有结构感知生成 verified-good-like 80 原子结构的初步证据”。",
        "- 如果 direct `DOPING_FULL80` 失败，应记录为 407-token 长序列在 225 条训练数据上的高风险对照，不影响 compressed20 主线。",
        "- 只有后续对 20-30 个候选做统一 DFT relaxation，并且 relaxed properties 优于 template/random baseline，才能写“模型已经证明可以生成更好的晶体结构”。",
        "",
        "## 下一步",
        "",
        "若 offline gate 通过，输出 `dft_candidate_manifest.json` 并选择 compressed20 top candidates、compact-template matched candidates、random/site baseline candidates 进入统一 DFT relaxation。",
        "",
    ]
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines), encoding="utf-8")
    print(args.output_report)


if __name__ == "__main__":
    main()
