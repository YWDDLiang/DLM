#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_fixed_adapter(training_dir: Path, candidate: str) -> Path:
    terminal_path = training_dir / "terminal_report.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if (
        terminal.get("status") != "pass"
        or terminal.get("candidate") != candidate
        or terminal.get("generated_metric_checkpoint_selection") is not False
        or terminal.get("intermediate_checkpoint_count") != 0
        or (terminal.get("conditional_structural_anchor_nll_gate") or {}).get("passed")
        is not True
    ):
        raise ValueError("training terminal is not a passing fixed endpoint")
    endpoint = training_dir / str(terminal["fixed_endpoint"])
    relative = Path(str(terminal.get("fixed_adapter_relative", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("fixed adapter relative path is unsafe")
    adapter_dir = training_dir / relative
    if endpoint not in (adapter_dir, *adapter_dir.parents):
        raise ValueError("fixed adapter is outside the registered endpoint")
    location = json.loads(
        (endpoint / "adapter_location.json").read_text(encoding="utf-8")
    )
    location_relative = Path(str(location["adapter_directory_relative"]))
    if adapter_dir != endpoint / location_relative:
        raise ValueError("terminal and adapter-location paths disagree")
    weight_file = str(terminal.get("fixed_adapter_weight_file", ""))
    if weight_file != location.get("weight_file"):
        raise ValueError("terminal and adapter-location weight files disagree")
    weight_path = adapter_dir / weight_file
    if not weight_path.is_file() or sha256_file(weight_path) != terminal.get(
        "fixed_adapter_weight_sha256"
    ):
        raise ValueError("fixed endpoint adapter weight identity mismatch")
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--candidate", choices=("sft_v2", "sft_v2_c"), required=True)
    args = parser.parse_args()
    try:
        endpoint = resolve_fixed_adapter(args.training_dir, args.candidate)
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(endpoint)


if __name__ == "__main__":
    main()
