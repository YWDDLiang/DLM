#!/usr/bin/env python3
"""Classify run failures and write Chinese-primary AI debug artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


CLASSIFIERS = {
    "environment": re.compile(r"ImportError|ModuleNotFoundError|CUDA error|CUDA out of memory|driver|conda|No module named", re.I),
    "data": re.compile(r"csv|CIF|Structure.from_str|material_id|dataset", re.I),
    "tokenizer": re.compile(r"tokenizer|pad_token|mask token|special token|vocab", re.I),
    "parser": re.compile(r"FixedSlot|Expected .* token|parse|slot|coordinate|element", re.I),
    "sampling": re.compile(r"generate|remasking|gen_length|block_length|temperature", re.I),
    "graph build": re.compile(r"CrystalNN|process_one|edge_indices|to_jimages|graph", re.I),
    "diffusion": re.compile(r"CSPDiffusion|diffusion|diff_steps|state_dict|lattices", re.I),
    "metrics": re.compile(r"compute_metrics|smact|coverage|fingerprint|eval_model", re.I),
}


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_artifact(run_dir: Path, filename: str) -> Path:
    direct = run_dir / filename
    if direct.exists():
        return direct
    matches = sorted(run_dir.rglob(filename))
    return matches[0] if matches else direct


def collect_evidence(run_dir: Path) -> str:
    chunks = []
    for filename in [
        "failure_cases.jsonl",
        "sample_metrics.json",
        "eval_metrics.json",
        "tokenizer_report.json",
        "training_log.jsonl",
    ]:
        path = find_artifact(run_dir, filename)
        chunks.append(read_text(path)[-30000:])
    for log_path in sorted((run_dir / "logs").glob("*")):
        if log_path.is_file():
            chunks.append(read_text(log_path)[-30000:])
    return "\n".join(chunks)


def classify(evidence: str) -> Counter:
    counts: Counter = Counter()
    for label, pattern in CLASSIFIERS.items():
        counts[label] += len(pattern.findall(evidence))
    if not counts:
        counts["unknown"] = 1
    return counts


def suggest_changes(run_dir: Path, counts: Counter) -> List[Dict[str, str]]:
    sample_metrics = read_json(find_artifact(run_dir, "sample_metrics.json"))
    eval_metrics = read_json(find_artifact(run_dir, "eval_metrics.json"))
    sample_analysis = read_json(run_dir / "notes" / "sample_failure_analysis.json")
    failure_text = read_text(find_artifact(run_dir, "failure_cases.jsonl"))
    suggestions: List[Dict[str, str]] = []
    top = counts.most_common(1)[0][0]

    parse_rate = sample_metrics.get("parse_rate")
    if isinstance(parse_rate, (int, float)) and parse_rate < 0.7:
        if "<|endoftext|>" in failure_text or "<|eot_id|>" in failure_text:
            suggestions.append(
                {
                    "axis": "training",
                    "change": "当前 32-sample SFT 后 raw generation 塌缩到结束 token；下一轮只扩大 SFT smoke 到 256 条，保持 schema、prompt、采样长度和 temperature=0.0 不变。",
                    "risk": "仍可能欠拟合；若 256 条仍塌缩，再考虑 constrained vocabulary 或更长训练。",
                }
            )
            return suggestions[:3]
        if "Expected at least 107 tokens, got 0" in failure_text:
            suggestions.append(
                {
                    "axis": "training",
                    "change": "当前采样没有产生 fixed-slot special tokens，说明 SFT 仍欠拟合；下一轮进入 full MP-20 SFT，保持 parser、schema、prompt 和 sampling 协议不变。",
                    "risk": "full SFT 仍可能需要更多 epoch；若 full SFT 后仍 parse=0，再考虑 constrained vocabulary sampling。",
                }
            )
            return suggestions[:3]
        if "Expected at least 107 tokens, got 54" in failure_text and "<N_" in failure_text:
            suggestions.append(
                {
                    "axis": "data representation",
                    "change": "当前输出已经包含 <N_...> 但只有约 54 个 schema tokens，说明 spaced answer 的空格消耗了 tokenizer 生成长度；下一轮改用 compact answer 重建数据并重训，保持 gen_length=107、schema 和 parser 不变。",
                    "risk": "需要重建 SFT 数据并重新训练；不能同时改变模型家族、fixed-slot schema 或评估协议。",
                }
            )
            return suggestions[:3]

    graph_rate = sample_metrics.get("graph_rate")
    duplicate_failures = sample_analysis.get("duplicate_exact_site_failures")
    high_symmetry_mean = sample_analysis.get("high_symmetry_coord_fraction_mean")
    if (
        isinstance(graph_rate, (int, float))
        and graph_rate < 0.5
        and isinstance(duplicate_failures, int)
        and duplicate_failures > 0
    ):
        suggestions.append(
            {
                "axis": "sampling",
                "change": (
                    "主导失败是 DLM 输出的精确重复 fractional coordinates，而不是 fixed-slot 语法解析。"
                    "已收集到 graph-valid proposals 时可以继续 DLM-only metrics 和 diffusion refinement；"
                    "下一轮采样只加入 duplicate-coordinate 预检查/早拒绝，并保留所有失败 attempt 记录。"
                ),
                "risk": (
                    "这不会提升模型原生 graph rate；不能把 rejection collection 误报为模型天然有效率。"
                    "在 graph acceptance 明显偏低时不要直接扩大到 10000。"
                ),
            }
        )
        if isinstance(high_symmetry_mean, (int, float)) and high_symmetry_mean > 0.8:
            suggestions.append(
                {
                    "axis": "training",
                    "change": (
                        "高对称坐标比例仍然过高；在完成 1000 条 refinement 后，再单独开一个受控 SFT/解码实验，"
                        "比较更多 epoch 或坐标去塌缩约束，当前不要同时改变 schema 和 diffusion。"
                    ),
                    "risk": "更多 epoch 可能继续强化高对称原型塌缩，必须用 raw-output duplicate rate 验证。",
                }
            )
        return suggestions[:3]

    if top == "environment":
        suggestions.append(
            {
                "axis": "environment",
                "change": "先运行 scripts/a800/env_doctor.py；只有当 env_doctor 明确报告缺包或依赖异常时，才重新运行 scripts/a800/bootstrap_diff_meets_diff.sh。",
                "risk": "环境修复可能改变依赖版本；必须记录 pip freeze 和 env_fix.md。",
            }
        )
    elif top == "parser":
        suggestions.append(
            {
                "axis": "parser",
                "change": "检查 failure_cases.jsonl 前 50 个样本；只有当失败来自 decoded text 噪声时，才添加不改变 schema 的解析归一化规则。",
                "risk": "不能修改 fixed-slot schema；只能修复解析鲁棒性。",
            }
        )
    elif top == "graph build":
        suggestions.append(
            {
                "axis": "graph build",
                "change": "用最多 32 个已解析结构运行 proposal-only graph smoke test，并保存失败 CIF 供 CrystalNN 排查。",
                "risk": "Graph failure 可能来自 DLM 结构质量，而不一定是代码 bug。",
            }
        )
    elif top == "diffusion":
        suggestions.append(
            {
                "axis": "diffusion",
                "change": "先确认 MP-20 diffusion checkpoint 路径，再用 8 个 proposal graphs 做 refinement smoke，然后再重试 1000 条。",
                "risk": "不要切换 diffusion architecture 或复现旧 baseline。",
            }
        )
    elif top == "metrics":
        suggestions.append(
            {
                "axis": "metrics",
                "change": "分别对 8-sample raw_dlm_samples.pt 和 refined pt 运行 scripts/run_crysllmgen_metrics.py，以隔离评估失败位置。",
                "risk": "不要改变 compute_metrics.py 的指标口径。",
            }
        )
    else:
        if isinstance(parse_rate, (int, float)) and parse_rate < 0.7:
            suggestions.append(
                {
                    "axis": "sampling",
                    "change": "保持训练 checkpoint 不变，只做一个 sampling 轴改动：temperature=0.0，remasking=low_confidence，steps=107。",
                "risk": "只能改 sampling 轴，避免同时改训练和 schema。",
                }
            )
        metrics = eval_metrics.get("metrics", eval_metrics)
        valid = metrics.get("valid") if isinstance(metrics, dict) else None
        if isinstance(valid, (int, float)) and valid < 85:
            suggestions.append(
                {
                    "axis": "training",
                    "change": "在不改变 tokenizer 或 schema 的前提下，从同一 checkpoint 继续一个短 SFT controlled run。",
                    "risk": "可能增加过拟合；必须保留同一 1000-sample evaluation protocol。",
                }
            )

    if not suggestions:
        suggestions.append(
            {
                "axis": "monitoring",
                "change": "没有发现单一主导失败；先补齐日志重新运行 monitor，暂时不要改变实验设置。",
                "risk": "证据不足时贸然改配置会破坏可比性。",
            }
        )
    return suggestions[:3]


def choose_status(run_dir: Path) -> str:
    sample_metrics = read_json(find_artifact(run_dir, "sample_metrics.json"))
    eval_metrics = read_json(find_artifact(run_dir, "eval_metrics.json"))
    if sample_metrics:
        if sample_metrics.get("target_mode") and not sample_metrics.get("target_reached"):
            return "STOP"
        parse_rate = sample_metrics.get("parse_rate")
        graph_rate = sample_metrics.get("graph_rate")
        if isinstance(parse_rate, (int, float)) and parse_rate < 0.7:
            return "PAUSE"
        if isinstance(graph_rate, (int, float)) and graph_rate < 0.5:
            return "WATCH"
    metrics = eval_metrics.get("metrics", eval_metrics)
    if isinstance(metrics, dict):
        valid = metrics.get("valid")
        if isinstance(valid, (int, float)) and valid == 0:
            return "STOP"
        if isinstance(valid, (int, float)) and valid < 85:
            return "WATCH"
    return "CONTINUE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    evidence = collect_evidence(args.run_dir)
    counts = classify(evidence)
    suggestions = suggest_changes(args.run_dir, counts)
    status = choose_status(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    notes_dir = args.run_dir / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    report = [
        "# AI Debug Review",
        "",
        f"状态：`{status}`",
        "",
        "## 失败分类",
    ]
    for label, count in counts.most_common():
        report.append(f"- {label}: {count}")
    report.extend(["", "## 建议改动"])
    for item in suggestions:
        report.append(f"- 轴：`{item['axis']}`；建议：{item['change']}；风险：{item['risk']}")
    report.extend(
        [
            "",
            "## 变更控制",
            "- 下一轮默认只接受一个主要轴的改动。",
            "- fixed-slot schema、canonical prompt、模型家族、模型路径、评估协议需要人工确认。",
        ]
    )
    (args.run_dir / "ai_review.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (args.run_dir / "next_run_suggestion.json").write_text(
        json.dumps({"status": status, "failure_counts": counts, "suggestions": suggestions}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    accepted_path = notes_dir / "accepted_fixes.md"
    if not accepted_path.exists():
        accepted_path.write_text(
            "# Accepted Fixes\n\n尚未接受任何自动 debug 建议。\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
