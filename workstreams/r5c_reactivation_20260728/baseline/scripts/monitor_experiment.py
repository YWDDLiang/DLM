#!/usr/bin/env python3
"""Chinese-primary monitor for active or completed experiment runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

BAD_LOG_PATTERNS = [
    ("OOM", re.compile(r"out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED", re.I)),
    ("NaN/Inf", re.compile(r"\bnan\b|\binf\b", re.I)),
    ("Traceback", re.compile(r"Traceback \(most recent call last\)", re.I)),
    (
        "Checkpoint failure",
        re.compile(
            r"checkpoint (failure|failed|error)|failed to load checkpoint|"
            r"state_dict.*(missing|unexpected|error|mismatch)|SafetensorError|HeaderTooLarge",
            re.I,
        ),
    ),
    ("Dataloader failure", re.compile(r"DataLoader|worker.*exited|BrokenPipe", re.I)),
]


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_jsonl_tail(path: Path, limit: int = 200) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def find_artifact(run_dir: Path, filename: str) -> Path:
    direct = run_dir / filename
    if direct.exists():
        return direct
    matches = sorted(run_dir.rglob(filename))
    return matches[0] if matches else direct


def iter_log_text(run_dir: Path) -> Iterable[str]:
    log_roots = [run_dir / "logs"]
    log_roots.extend(path for path in sorted(run_dir.rglob("logs")) if path.is_dir() and path != run_dir / "logs")
    seen = set()
    for log_root in log_roots:
        if not log_root.exists():
            continue
        for path in sorted(log_root.rglob("*")):
            if path in seen:
                continue
            seen.add(path)
            if path.is_file() and path.suffix in {".log", ".out", ".err"}:
                yield path.read_text(encoding="utf-8", errors="replace")[-20000:]


def inspect_training(rows: List[Dict[str, Any]], findings: List[str]) -> str:
    losses = [row.get("loss") for row in rows if isinstance(row.get("loss"), (int, float))]
    val_losses = [row.get("val_loss") for row in rows if isinstance(row.get("val_loss"), (int, float))]
    if any(loss is not None and (math.isnan(loss) or math.isinf(loss)) for loss in losses + val_losses):
        findings.append("训练日志中出现 NaN/Inf loss。")
        return "STOP"
    if len(losses) >= 20 and losses[-1] > max(10.0, losses[0] * 5):
        findings.append(f"训练 loss 明显爆炸：first={losses[0]:.4f}, latest={losses[-1]:.4f}。")
        return "PAUSE"
    if not rows:
        findings.append("未发现 training_log.jsonl，可能尚未开始训练或日志路径不完整。")
        return "WATCH"
    findings.append("训练日志未发现 NaN/Inf 或明显 loss 爆炸。")
    return "CONTINUE"


def inspect_metrics(run_dir: Path, findings: List[str]) -> str:
    sample_metrics = read_json(find_artifact(run_dir, "sample_metrics.json"))
    eval_metrics = read_json(find_artifact(run_dir, "eval_metrics.json"))
    status = "CONTINUE"
    if sample_metrics:
        parse_rate = sample_metrics.get("parse_rate")
        graph_rate = sample_metrics.get("graph_rate")
        if parse_rate is not None and parse_rate < 0.5:
            findings.append(f"raw DLM parse rate 偏低：{parse_rate:.4f}。")
            status = "PAUSE"
        elif parse_rate is not None:
            findings.append(f"raw DLM parse rate={parse_rate:.4f}。")
        if graph_rate is not None and graph_rate < 0.5:
            findings.append(f"CrysLLMGen graph build rate 偏低：{graph_rate:.4f}。")
            status = "PAUSE"
    if eval_metrics:
        metrics = eval_metrics.get("metrics", eval_metrics)
        valid = metrics.get("valid") if isinstance(metrics, dict) else None
        if isinstance(valid, (int, float)) and valid == 0:
            findings.append("评估 valid=0，说明采样或评估链路已经系统性失败。")
            status = "STOP"
        elif isinstance(valid, (int, float)):
            findings.append(f"当前评估 valid={valid}。")
    return status


def inspect_logs(run_dir: Path, findings: List[str]) -> str:
    status = "CONTINUE"
    for text in iter_log_text(run_dir):
        for name, pattern in BAD_LOG_PATTERNS:
            if pattern.search(text):
                findings.append(f"日志命中异常模式：{name}。")
                if name in {"OOM", "Traceback"}:
                    status = "STOP"
                elif status != "STOP":
                    status = "PAUSE"
    if status == "CONTINUE":
        findings.append("Slurm/full logs 未命中高危异常模式。")
    return status


def merge_status(*statuses: str) -> str:
    rank = {"CONTINUE": 0, "WATCH": 1, "PAUSE": 2, "STOP": 3}
    return max(statuses, key=lambda item: rank[item])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    findings: List[str] = []
    training_status = inspect_training(read_jsonl_tail(find_artifact(args.run_dir, "training_log.jsonl")), findings)
    metrics_status = inspect_metrics(args.run_dir, findings)
    logs_status = inspect_logs(args.run_dir, findings)
    status = merge_status(training_status, metrics_status, logs_status)

    payload = {"status": status, "findings": findings}
    report_lines = [
        f"# AI Monitor Report",
        "",
        f"状态：`{status}`",
        "",
        "## 证据",
    ]
    report_lines.extend(f"- {item}" for item in findings)
    report_lines.extend(
        [
            "",
            "## 建议",
            "- 若状态为 `CONTINUE`，继续当前 run。",
            "- 若状态为 `WATCH`，保留任务但缩短下一次检查间隔。",
            "- 若状态为 `PAUSE`，优先检查最近一次配置、数据和环境变更。",
            "- 若状态为 `STOP`，停止启动新任务，先运行 `scripts/auto_debug_run.py` 分类失败原因。",
        ]
    )
    report = "\n".join(report_lines) + "\n"
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report, encoding="utf-8")
    else:
        print(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if status in {"CONTINUE", "WATCH"} else 1)


if __name__ == "__main__":
    main()
