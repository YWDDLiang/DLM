#!/usr/bin/env python3
"""Static fail-closed checks for the nine-cell post-only official evaluator."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = Path(__file__).resolve().parent
    config = json.loads((source / "CONFIG.json").read_text(encoding="utf-8"))
    upstream = config["upstream_generation"]
    if (
        config.get("schema")
        != "h1a2_retrained_postonly_sun256_official_recovery_config_v1"
    ):
        raise ValueError("official evaluator config schema changed")
    if re.fullmatch(r"[0-9a-f]{64}", str(upstream["source_manifest_sha256"])) is None:
        raise ValueError("upstream source manifest is not frozen")
    expected_ids = [
        "fresh_0",
        "fresh_1",
        "fresh_2",
        "fresh_3",
        "topology_repeat_0",
        "topology_repeat_1",
        "topology_repeat_2",
        "topology_repeat_3",
        "h1a2_b0_d1_once",
    ]
    cells = list(upstream["expected_cells"])
    if (
        [row["cell_id"] for row in cells] != expected_ids
        or {row["stage"] for row in cells} != {"post_model494"}
        or upstream.get("evaluated_stage") != "post_model494_only"
        or upstream.get("pre_refine_role") != "intermediate_only_not_scored"
        or config["inference"].get("pre_post_inference") is not False
    ):
        raise ValueError("post-only nine-cell layout changed")
    if [row["panel"] for row in cells].count("fresh_cohort") != 4:
        raise ValueError("fresh panel shape changed")
    if [row["panel"] for row in cells].count("topology_match") != 4:
        raise ValueError("topology panel shape changed")
    if [row["panel"] for row in cells].count("historical_h1a2_dlm_control") != 1:
        raise ValueError("historical H1-A2 DLM control shape changed")

    prequery = config["prequery_all_reconstructed"]
    scheduler = config["scheduler"]
    if (
        int(prequery.get("count", -1)) != 1076
        or prequery.get("canonical_sha256")
        != "d0183a73c03a65d675a052ad157114c9cf51288b6876957564b44286681b7a25"
        or int(scheduler.get("slurm_job_count", -1)) != 1
        or int(scheduler.get("array_jobs", -1)) != 0
        or int(scheduler.get("gpus", -1)) != 4
        or int(scheduler.get("cpus", -1)) != 32
        or scheduler.get("cell_waves")
        != [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
    ):
        raise ValueError("V8 prequery or one-job scheduler contract changed")

    frozen = Path(config["frozen_evaluator"]["source_root"])
    if (
        sha256_file(frozen / "SOURCE_SHA256.txt")
        != config["frozen_evaluator"]["source_manifest_sha256"]
    ):
        raise ValueError("frozen official evaluator identity changed")
    expected = {
        "run_completed_cache_sun.py": "585496be8fde0a50bc09f178ba456c50cfed4e2e612ef7678a3be09106936937",
        "audit_official_cache.py": "02c21066e60b49552b70c2adfa9a1b6186af1cdc014eb79717bc6ea327b07cf5",
        "complete_official_cache.py": "ff5e82de653e192a4bb27e6005adc5c02ff976f36ca57ad82f72c0a2398b5a85",
        "finalize_official.py": "17ac3657b268fcb690767d9168593cfa6c17141a09612995f66a4f758c102ff4",
    }
    for relative, digest in expected.items():
        if sha256_file(frozen / relative) != digest:
            raise ValueError(f"frozen official evaluator changed: {relative}")
    finalizer_text = (frozen / "finalize_official.py").read_text(encoding="utf-8")
    if finalizer_text.count("def evaluate_cell(") != 1 or finalizer_text.count(
        "def exact_mcnemar("
    ) != 1:
        raise ValueError("frozen official core anchors changed")
    adapter = config["finalizer_adapter"]
    if (
        adapter["base_finalizer_sha256"] != expected["finalize_official.py"]
        or adapter["stability_phase_diagram_and_cell_evaluation_code_changed"]
        is not False
        or "evaluate_cell" not in adapter["unchanged_official_core"]
    ):
        raise ValueError("post-only finalizer adapter scope changed")

    reference = config["historical_h1a2_planner_reference"]
    reference_path = Path(reference["terminal_report"])
    if sha256_file(reference_path) != reference["terminal_report_sha256"]:
        raise ValueError("historical H1-A2 planner terminal changed")
    reference_terminal = json.loads(reference_path.read_text(encoding="utf-8"))
    matches = [
        row
        for row in reference_terminal.get("planner_distributions", [])
        if row.get("cohort_id") == reference["cohort_id"]
    ]
    if (
        len(matches) != 1
        or matches[0].get("cohort256_sha256") != reference["cohort256_sha256"]
        or int(matches[0].get("attempts", -1)) != 256
    ):
        raise ValueError("historical H1-A2 planner reference changed")

    for name in (
        "protocol.py",
        "cell_descriptor.py",
        "preflight.py",
        "validate_preliminary.py",
        "collect_official_inputs.py",
        "assemble_preliminary.py",
        "finalize_postonly.py",
        "prepare_prequery_inputs.py",
        "adopt_precompleted_cache.py",
    ):
        ast.parse((source / name).read_text(encoding="utf-8"), filename=name)
    sbatch = (source / "sun_cell.sbatch").read_text(encoding="utf-8")
    if "#SBATCH --array=0-8%9" not in sbatch or "pre_model494" in sbatch:
        raise ValueError("active preliminary array is not post-only 0-8")
    combined = (source / "combined_official.sbatch").read_text(encoding="utf-8")
    submit = (source / "submit_combined_official_once.sh").read_text(
        encoding="utf-8"
    )
    precomplete = (source / "precomplete_official_cache_once.sh").read_text(
        encoding="utf-8"
    )
    if (
        "#SBATCH --array" in combined
        or "#SBATCH --cpus-per-task=32" not in combined
        or "#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4" not in combined
        or "run_wave 0 1 2 3" not in combined
        or "run_wave 4 5 6 7" not in combined
        or "run_wave 8" not in combined
        or "pre_model494" in combined
        or submit.count("sbatch --parsable") != 1
        or "MP_API_KEY" not in precomplete
        or "unset MP_API_KEY" not in precomplete
        or "--key-file \"$KEY_FILE\"" not in precomplete
    ):
        raise ValueError("combined V8 official implementation changed")
    print("self_test: PASS")


if __name__ == "__main__":
    main()
