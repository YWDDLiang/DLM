#!/usr/bin/env python3
"""Rebuild WTB-32 v2 from the latest authorized tangent-submit parent."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = (
    ROOT
    / "diagnostics"
    / "build_wq_wyckoff_chart_retraction_preflight_sup28185_v2_archive.py"
)
PARENT_ROOT = (
    ROOT
    / "runs"
    / "remote_audit"
    / "20260726_wq_wyckoff_tangent_bridge_preflight_v1"
    / "build_submit_amendment_v1"
    / "archive_root_5b61c3c2915b"
)
PARENT_MANIFEST_SHA256 = (
    "5b61c3c2915b0e9d97736ad3020f1e37059fd901ae05213acd13a520915872f4"
)
BUILD = (
    ROOT
    / "runs"
    / "remote_audit"
    / "20260726_wq_wyckoff_chart_retraction_preflight_sup28185_v2"
    / "build_transfer_preservation_v2"
)
THIS_BUILDER = (
    "diagnostics/"
    "build_wq_wyckoff_chart_retraction_preflight_sup28185_v2_preservation_archive.py"
)


def main() -> None:
    spec = importlib.util.spec_from_file_location("wtb32_v2_base_builder", BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen WTB-32 v2 archive builder")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    builder.BUILD = BUILD
    builder.PARENT_ROOT = PARENT_ROOT
    builder.PARENT_MANIFEST = PARENT_ROOT / "patch_manifest.json"
    builder.PARENT_MANIFEST_SHA256 = PARENT_MANIFEST_SHA256
    builder.CURRENT_PATHS = set(builder.CURRENT_PATHS) | {THIS_BUILDER}
    builder.main()


if __name__ == "__main__":
    main()
