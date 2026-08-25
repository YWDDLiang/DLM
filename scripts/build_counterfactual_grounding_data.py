#!/usr/bin/env python3
"""Add a composition-matched counterfactual prompt to exact-length DLM rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crystal_dlm.r5_plan_state import build_body_prompt  # noqa: E402
from h1a2_repro.counterfactual import (  # noqa: E402
    build_counterfactual_plan,
    choose_donors,
    plan_key,
    structural_tuple,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def extract_plan(row: dict) -> dict:
    explicit = row.get("plan_state") or row.get("r5_plan_state")
    if isinstance(explicit, dict):
        return explicit
    prompt = str(row.get("prompt") or "")
    marker = "plan_state:"
    body_marker = "\ndynamic_crystal_body:"
    if marker not in prompt or body_marker not in prompt:
        return {}
    payload = prompt.split(marker, 1)[1].split(body_marker, 1)[0].strip()
    value = json.loads(payload)
    return value if isinstance(value, dict) else {}


def build_split(source: Path, destination: Path, *, seed: int) -> dict:
    rows = read_jsonl(source)
    plans = [extract_plan(row) for row in rows]
    donor_indices = choose_donors(plans, seed=seed)
    grounded = 0
    for row, plan, donor_index in zip(rows, plans, donor_indices):
        row["counterfactual_grounding_eligible"] = False
        if donor_index is None or not plan:
            continue
        donor = plans[donor_index]
        counterfactual = build_counterfactual_plan(plan, donor)
        row.update(
            {
                "counterfactual_grounding_eligible": True,
                "counterfactual_plan_state": counterfactual,
                "counterfactual_prompt": build_body_prompt(counterfactual),
                "counterfactual_donor_plan_key": plan_key(donor),
                "counterfactual_structural_tuple": list(structural_tuple(counterfactual)),
            }
        )
        grounded += 1
    write_jsonl(destination, rows)
    return {"rows": len(rows), "grounding_eligible": grounded, "coverage": grounded / max(1, len(rows))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "composition-matched-counterfactual-grounding-data@1", "seed": args.seed, "splits": {}}
    for split in ("train", "val", "test"):
        source = args.input_dir / f"{split}.jsonl"
        if source.exists():
            report["splits"][split] = build_split(source, args.output_dir / source.name, seed=args.seed)
    for name in ("stats.json", "feasibility_metrics.json", "vocab_tokens.txt", "prompt_pool.json"):
        source = args.input_dir / name
        if source.exists():
            shutil.copyfile(source, args.output_dir / name)
    (args.output_dir / "counterfactual_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
