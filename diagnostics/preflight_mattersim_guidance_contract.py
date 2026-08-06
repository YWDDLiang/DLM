#!/usr/bin/env python3
"""Validate the local MatterSim-guide / CHGNet-evaluator preparation contract."""

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

from crystal_dlm.wqcodiff.guidance_contract import (  # noqa: E402
    GuidanceContractError,
    load_and_validate_guidance_contract,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "mattersim_guidance_chgnet_eval_v1.json"
)


def _write_exclusive_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
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
    parser.add_argument(
        "--exclusive-output",
        type=Path,
        help="Optionally write one immutable preflight record; existing paths fail.",
    )
    args = parser.parse_args(argv)

    try:
        result = load_and_validate_guidance_contract(args.config)
    except (GuidanceContractError, KeyError, TypeError, ValueError, OSError) as exc:
        failure = {
            "schema": "wqcodiff_guidance_contract_preflight_v1",
            "ok": False,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": str(args.config.resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(failure, sort_keys=True, indent=2))
        return 2

    output: dict[str, object] = {
        "schema": "wqcodiff_guidance_contract_preflight_v1",
        "ok": True,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "remote_submission_performed": False,
        "scientific_attempts_created": 0,
        "acceptance": result,
    }
    if args.exclusive_output is not None:
        try:
            _write_exclusive_json(args.exclusive_output.resolve(), output)
        except FileExistsError:
            print(
                json.dumps(
                    {
                        "schema": "wqcodiff_guidance_contract_preflight_v1",
                        "ok": False,
                        "error": "exclusive output already exists",
                        "exclusive_output": str(args.exclusive_output.resolve()),
                    },
                    sort_keys=True,
                    indent=2,
                )
            )
            return 3
    print(json.dumps(output, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
