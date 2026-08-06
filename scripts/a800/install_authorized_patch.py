#!/usr/bin/env python3
"""Verify and atomically overlay a user-authorized post-bundle source patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath


ALLOWED_TOP_LEVEL = {
    "configs",
    "crystal_dlm",
    "diagnostics",
    "docs",
    "scripts",
    "tests",
}
PRESERVED_TOP_LEVEL = {"data", "reference", "runs", "reports", "archive", ".secrets", ".git"}
ALLOWED_AUTHORIZATIONS = {
    "user_explicit_chgnet_waiver_2026-07-17",
    "user_environment_scope_explicit_2026-07-17",
    "user_r5c_a100_sun_and_three_epoch_checkpoint_selection_2026-07-21",
    "user_r5c_a100_sun_three_epoch_two_gpu_ddp_2026-07-21",
    "user_thread_env_superseding_diagnostics_2026-07-22",
    "user_ddp_profile_matrix_2026-07-22",
    "user_flash_profile_formal_progression_2026-07-23",
    "user_refiner_supersession26955_v2_2026-07-23",
    "user_epoch_eval_selection_supersession27104_v1_2026-07-23",
    "user_epoch_eval_selection_supersession27104_v2_bash42_2026-07-24",
    "user_wq_parent_csp_probe_and_bridge_v2_2026-07-24",
    "user_wq_parent_csp_probe_sup27407_v1_2026-07-24",
    "user_wq_parent_csp_sun256_v1_2026-07-24",
    "user_iclr_mlip_free_wq_mechanism64_2026-07-25",
    "user_wq_existing22_chgnet_sun_v1_2026-07-25",
    "user_wq_existing22_mp_completion_v1_2026-07-25",
    "user_wq_schedule_correct_bridge_parity_v1_2026-07-26",
    "user_wq_schedule_correct_bridge_parity_sup28054_v1_2026-07-26",
    "user_wq_wyckoff_tangent_bridge_preflight_v1_remote_install_2026-07-26",
    "user_wq_wyckoff_tangent_bridge_preflight_v1_audit_amendment_and_submit_2026-07-26",
    "user_wq_wyckoff_chart_retraction_preflight_sup28185_v2_2026-07-26",
    "user_wq_wyckoff_chart_retraction_confirmatory256_v1_local_preparation_2026-07-26",
    "user_wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1_2026-07-27",
    "user_wq_wyckoff_identity_mechanics_sup28195_v1_2026-07-27",
    "user_wq_llm_charge_stop_paired64_v1_2026-07-27",
    "user_wq_formula_plan_sft_pilot_v1_2026-07-27",
    "user_wq_formula_plan_sft_one_epoch_v2_2026-07-27",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    staging = args.staging.resolve()
    target = args.target.resolve()
    if _sha(manifest_path) != args.expected_manifest_sha256:
        raise ValueError("authorized patch manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "wqcodiff_authorized_patch_v1":
        raise ValueError("invalid authorized patch schema")
    if manifest.get("authorization") not in ALLOWED_AUTHORIZATIONS:
        raise ValueError("patch lacks a registered exact user authorization")
    if len(str(manifest.get("base_source_bundle_sha256", ""))) != 64:
        raise ValueError("patch lacks its base source bundle hash")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("authorized patch has no files")
    seen: set[str] = set()
    for entry in entries:
        relative = str(entry.get("path", ""))
        path = PurePosixPath(relative)
        if (
            not relative
            or path.is_absolute()
            or ".." in path.parts
            or path.parts[0] not in ALLOWED_TOP_LEVEL
            or set(path.parts) & PRESERVED_TOP_LEVEL
            or relative in seen
        ):
            raise ValueError(f"unsafe or duplicate authorized-patch path: {relative}")
        seen.add(relative)
        source = staging / relative
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"authorized patch source is missing or unsafe: {source}")
        if source.stat().st_size != int(entry["bytes"]) or _sha(source) != entry["sha256"]:
            raise ValueError(f"authorized patch source hash/size mismatch: {relative}")
    for entry in entries:
        relative = str(entry["path"])
        source = staging / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".authorized-patch-tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    record_dir = target / ".artifacts" / "source_sync"
    record_dir.mkdir(parents=True, exist_ok=True)
    record = record_dir / f"authorized_patch_{args.expected_manifest_sha256}.json"
    with record.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                **manifest,
                "manifest_sha256": args.expected_manifest_sha256,
                "target": str(target),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(
        json.dumps(
            {
                "ok": True,
                "files_installed": len(entries),
                "manifest_sha256": args.expected_manifest_sha256,
                "record": str(record),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
