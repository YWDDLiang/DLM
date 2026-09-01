#!/usr/bin/env python3
"""Freeze matched baseline/BTRD raw generations for the first256 Plans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(run: Path) -> list[dict]:
    rows = read_jsonl(run / "main0/generation/generation.jsonl")
    rows += read_jsonl(run / "main1/generation/generation.jsonl")[:56]
    if len(rows) != 256:
        raise ValueError("first256 generation extraction changed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--plans-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    plans_path = args.plans_jsonl.resolve()
    plans = read_jsonl(plans_path)
    if len(plans) != 256:
        raise ValueError("frozen BTRD development Plan count changed")
    arms = {
        "BASE": selected(args.baseline_run.resolve()),
        "BTRD": selected(args.candidate_run.resolve()),
    }
    output.mkdir(parents=True)
    hashes = {}
    for arm, rows in arms.items():
        target = output / arm
        target.mkdir()
        frozen = []
        for index, (row, plan) in enumerate(zip(rows, plans, strict=True)):
            if row.get("plan_state") != plan.get("plan_state"):
                raise ValueError(f"{arm} Plan changed at {index}")
            if arms["BASE"][index].get("plan_state") != arms["BTRD"][index].get("plan_state"):
                raise ValueError(f"paired Plan mismatch at {index}")
            item = dict(row)
            item["ordinal"] = index
            item["sample_idx"] = index
            item["attempt_id"] = f"btrd-first256-{arm.lower()}-{index:04d}"
            item["pair_id"] = f"btrd-first256-{index:04d}"
            item["arm"] = "control" if arm == "BASE" else "candidate"
            frozen.append(item)
        path = target / "generation.jsonl"
        path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in frozen))
        hashes[arm] = sha256_file(path)
    manifest = {
        "schema": "btrd_first256_matched_raw_v1",
        "status": "complete",
        "requested_per_arm": 256,
        "plans_sha256": sha256_file(plans_path),
        "generation_sha256": hashes,
        "selection_outcomes_read": False,
        "body_rerun": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
