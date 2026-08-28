#!/usr/bin/env python3
"""Record frozen teacher-trajectory coverage for the v2.5 online canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_c3fd_v23_pauling_witness import teacher_witness_counts  # noqa: E402
from crystal_dlm.composition_pair_prior import ValenceNode  # noqa: E402
from crystal_dlm.family_reachability import PaulingBitsetReachability  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.perf_counter()
    vocabulary = json.loads(
        (args.data_dir / "vocabulary.json").read_text(encoding="utf-8")
    )
    family_values = list(vocabulary["soft_vocabulary"]["anion_framework"])
    nodes = [
        ValenceNode(int(row["atomic_number"]), int(row["oxidation_state"]))
        for row in sorted(vocabulary["species"], key=lambda value: int(value["id"]))
    ]
    oracle = PaulingBitsetReachability(nodes)
    teacher = {
        split: teacher_witness_counts(
            args.data_dir / f"{split}.jsonl",
            nodes=nodes,
            family_values=family_values,
            oracle=oracle,
        )
        for split in ("train", "val")
    }
    elapsed = time.perf_counter() - started
    gate = {
        "train_teacher_witness_100pct": teacher["train"]["rate"] == 1.0,
        "val_teacher_witness_100pct": teacher["val"]["rate"] == 1.0,
        "model_weights_unused": True,
        "outcome_labels_unused": True,
    }
    gate["teacher_gate_pass"] = all(gate.values())
    payload = {
        "schema": "h1a2_c3fd_v25_teacher_witness_audit_v1",
        "data_dir": str(args.data_dir.resolve()),
        "teacher_witness": teacher,
        "elapsed_sec": elapsed,
        "gate": gate,
    }
    args.output_dir.mkdir(parents=True)
    stem = "C3FD_V25_TEACHER_WITNESS_AUDIT"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# C³FD-v2.5 teacher-witness audit",
        "",
        f"Gate pass: **{gate['teacher_gate_pass']}**",
        f"Train: `{teacher['train']['valid']}/{teacher['train']['total']}`; "
        f"validation: `{teacher['val']['valid']}/{teacher['val']['total']}`; "
        f"elapsed `{elapsed:.3f}` seconds.",
        "",
        *[f"- {key}: `{value}`" for key, value in gate.items()],
    ]
    (args.output_dir / f"{stem}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
