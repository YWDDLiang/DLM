"""Resolve paper assets without absolute project paths or hash gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PlanChoice:
    path: Path
    resampled: bool
    reason: str


def repo_path(value: str | Path) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        raise ValueError(f"project asset paths must be relative: {raw}")
    return (REPO_ROOT / raw).resolve()


def choose_plans(*, resample_requested: bool) -> PlanChoice:
    frozen = repo_path("data/plans/r03_parsed_256.jsonl")
    planner = repo_path("checkpoints/planner/adapter_model.safetensors")
    sampled = repo_path("runs/quick_256x4/plans/resampled_256.jsonl")

    if resample_requested and planner.is_file():
        return PlanChoice(sampled, True, "Planner checkpoint is available")
    if resample_requested and not planner.is_file():
        return PlanChoice(
            frozen,
            False,
            "Planner checkpoint is missing; using frozen Plans",
        )
    return PlanChoice(frozen, False, "frozen Plans are the default")


def asset_status() -> dict[str, object]:
    paths = {
        "planner_checkpoint": "checkpoints/planner/adapter_model.safetensors",
        "dlm_checkpoint": "checkpoints/dlm/adapter_model.safetensors",
        "diffusion_checkpoint": "checkpoints/diffusion/model_494.pt",
        "mp20_train": "data/mp20/train.csv",
        "mp20_validation": "data/mp20/val.csv",
        "mp20_test": "data/mp20/test.csv",
        "frozen_plans": "data/plans/r03_parsed_256.jsonl",
        "seed_ledger": "data/plans/r03_seed_ledger_256.jsonl",
    }
    return {
        name: {"path": value, "present": repo_path(value).is_file()}
        for name, value in paths.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resample-plans", action="store_true")
    args = parser.parse_args()
    choice = choose_plans(resample_requested=args.resample_plans)
    try:
        displayed_path = choice.path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        displayed_path = str(choice.path)
    payload = {
        "assets": asset_status(),
        "plan_choice": {
            "path": displayed_path,
            "resampled": choice.resampled,
            "reason": choice.reason,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
