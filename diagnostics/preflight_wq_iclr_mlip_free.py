#!/usr/bin/env python3
"""Run the dependency-light preflight for the MLIP-free WQ experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crystal_dlm.wqcodiff.mlip_free_contract import (  # noqa: E402
    MLIPFreeContractError,
    load_and_validate_mlip_free_contract,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "wq_iclr_mlip_free_v1.json"
)


def _write_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--exclusive-output", type=Path)
    args = parser.parse_args(argv)

    try:
        acceptance = load_and_validate_mlip_free_contract(args.config)
    except (MLIPFreeContractError, KeyError, TypeError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "schema": "wqcodiff_iclr_mlip_free_preflight_v1",
                    "ok": False,
                    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config": str(args.config.resolve()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 2

    result: dict[str, object] = {
        "schema": "wqcodiff_iclr_mlip_free_preflight_v1",
        "ok": True,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_attempts_created": 0,
        "remote_submission_performed": False,
        "acceptance": acceptance,
    }
    if args.exclusive_output is not None:
        try:
            _write_exclusive(args.exclusive_output.resolve(), result)
        except FileExistsError:
            print(
                json.dumps(
                    {
                        "schema": "wqcodiff_iclr_mlip_free_preflight_v1",
                        "ok": False,
                        "error": "exclusive output already exists",
                        "exclusive_output": str(args.exclusive_output.resolve()),
                    },
                    sort_keys=True,
                    indent=2,
                )
            )
            return 3
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
