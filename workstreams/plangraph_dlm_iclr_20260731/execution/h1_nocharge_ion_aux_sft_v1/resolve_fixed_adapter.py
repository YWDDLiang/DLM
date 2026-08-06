#!/usr/bin/env python3
"""Resolve only the frozen checkpoint-0400 candidate adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    args = parser.parse_args()
    training = args.training_dir.resolve()
    terminal = json.loads((training / "terminal_report.json").read_text(encoding="utf-8"))
    if (
        terminal.get("status") != "pass"
        or terminal.get("fixed_endpoint") != "checkpoint-0400"
        or int(terminal.get("global_step", -1)) != 400
        or not (training / "_SUCCESS").is_file()
    ):
        raise RuntimeError("training terminal does not authorize fixed endpoint sampling")
    checkpoint = (training / "checkpoint-0400").resolve()
    location = json.loads((checkpoint / "adapter_location.json").read_text(encoding="utf-8"))
    relative = Path(str(location["adapter_directory_relative"]))
    adapter = (checkpoint / relative).resolve()
    try:
        adapter.relative_to(checkpoint)
    except ValueError as exc:
        raise RuntimeError("adapter location escapes checkpoint-0400") from exc
    weight = adapter / str(location["weight_file"])
    if not (adapter / "adapter_config.json").is_file() or not weight.is_file():
        raise FileNotFoundError("fixed candidate adapter is incomplete")
    print(adapter)


if __name__ == "__main__":
    main()
