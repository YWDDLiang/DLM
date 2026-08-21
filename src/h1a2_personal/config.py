"""Load and display a personal H1-A2 JSON configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"personal config is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("paths"), dict):
        raise ValueError("personal config requires a 'paths' object")
    if not isinstance(payload.get("science"), dict):
        raise ValueError("personal config requires a 'science' object")
    if not isinstance(payload.get("runtime"), dict):
        raise ValueError("personal config requires a 'runtime' object")
    return payload


def resolve_relative_paths(payload: dict[str, Any]) -> dict[str, Any]:
    resolved = json.loads(json.dumps(payload))
    for key, raw in resolved["paths"].items():
        path = Path(raw)
        if path.is_absolute():
            raise ValueError(f"personal project paths must be relative: {key}={raw}")
        resolved["paths"][key] = str((REPO_ROOT / path).resolve())
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "configs" / "personal.example.json",
    )
    args = parser.parse_args()
    payload = load_config(args.config.resolve())
    print(json.dumps(resolve_relative_paths(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

