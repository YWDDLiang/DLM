#!/usr/bin/env python3
"""Static fail-closed checks for the zero-Slurm finalization continuation."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


EXPECTED_UNCHANGED = {
    "protocol.py": "a528128f1cb8fb5b949ba8d59d76085799748768735dff005adcccc3df3e3c00",
    "collect_official_inputs.py": "8969dd77ce6a3ceaa1ea15d21f506d96a4b107744ab7370e8cdda3e6477e5544",
    "assemble_preliminary.py": "4efc8737deeb207dc358d89fff0cb5992426fe1102b993a911ce7e923983bb00",
    "adopt_precompleted_cache.py": "9feac2a2ccf48416230ccc619bc0014696cff95742b610fb089d59032b6ea63a",
    "finalize_postonly.py": "62e3ff2883e6544ff2df86181ba40a8287ffc52faf89c2c3109933a44dd6ba4f",
    "run_frozen_official.py": "087e22ba960e9a9616bd5e709d5340057a043f2efc2183c7a8c9dca00188d4dc",
    "preflight.py": "6abc702e39373332b7b7372e80acc7f846f15c72b377671186ed1123a22fd710",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = Path(__file__).resolve().parent
    config = json.loads((source / "CONFIG.json").read_text(encoding="utf-8"))
    continuation = config["continuation"]
    scheduler = config["scheduler"]
    if (
        config.get("run_id")
        != "20260813_h1a2_retrained_world2_r03_sun_official_v8_finalization_continuation_v5"
        or int(continuation.get("parent_slurm_job_id", -1)) != 32049
        or continuation.get("parent_terminal_state") != "FAILED"
        or int(continuation.get("parent_completed_preliminary_cells", -1)) != 9
        or int(continuation.get("reused_preliminary_cells", -1)) != 9
        or continuation.get("reused_precompleted_official_cache") is not True
        or continuation.get("generation_or_refinement_rerun") is not False
        or continuation.get("preliminary_evaluation_rerun") is not False
        or continuation.get("mp_query_rerun") is not False
        or int(continuation.get("slurm_jobs", -1)) != 0
        or int(scheduler.get("slurm_job_count", -1)) != 0
        or int(scheduler.get("array_jobs", -1)) != 0
        or int(scheduler.get("gpus", -1)) != 0
    ):
        raise ValueError("continuation scope changed")
    for relative, expected in EXPECTED_UNCHANGED.items():
        path = source / relative
        if sha256(path) != expected:
            raise ValueError(f"scientific implementation changed: {relative}")
        ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    orchestration = (source / "prepare_and_finalize_once.sh").read_text(
        encoding="utf-8"
    )
    forbidden = ("sbatch", "srun", "run_crysllmgen", "complete_official_cache.py")
    if any(token in orchestration for token in forbidden):
        raise ValueError("continuation attempts forbidden compute or submission")
    required = (
        "collect_official_inputs.py",
        "--audit-only",
        "assemble_preliminary.py",
        "adopt_precompleted_cache.py",
        "finalize_postonly.py",
        "cp -al",
    )
    if any(token not in orchestration for token in required):
        raise ValueError("continuation stages changed")
    print("self_test: PASS")


if __name__ == "__main__":
    main()
