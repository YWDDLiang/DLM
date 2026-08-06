#!/usr/bin/env python3
"""Generate the preregistered 512-entry CR-Plan science ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from crystal_dlm.ordinal_rng import derive_ordinal_seed


IDENTITY = "h1_crplan_fourarm512_route_amendment_v1"
ATTEMPTS = 512
BASE_SEED = 1187798901
DERIVATION_PHRASE = f"{IDENTITY}|science_ledger_v1"
DERIVATION_SHA256 = (
    "46cc5f7595aa311b5fea7d8fbd49619c37b39b86772f99b21292ab2fc6a76412"
)


def build_ledger() -> dict[str, object]:
    observed = hashlib.sha256(DERIVATION_PHRASE.encode("utf-8")).hexdigest()
    if observed != DERIVATION_SHA256:
        raise RuntimeError("seed-derivation phrase identity mismatch")
    rows = [
        {
            "ordinal": ordinal,
            "stage": "planner_sampling",
            "role": "shared",
            "planner_sampling_seed": derive_ordinal_seed(
                BASE_SEED,
                sample_idx=ordinal,
                stage="planner_sampling",
                role="shared",
            ),
        }
        for ordinal in range(ATTEMPTS)
    ]
    return {
        "schema": "h1_crplan_fourarm512_science_ledger_v1",
        "identity": IDENTITY,
        "attempts_per_arm": ATTEMPTS,
        "ordinals": [0, ATTEMPTS - 1],
        "base_seed": BASE_SEED,
        "seed_mode": "stateless_ordinal_v1",
        "seed_derivation_phrase": DERIVATION_PHRASE,
        "seed_derivation_phrase_sha256": DERIVATION_SHA256,
        "rows": rows,
        "independent_of_paired32": True,
        "independent_of_e1": True,
        "sample_retry_or_replacement": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(
        json.dumps(build_ledger(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
