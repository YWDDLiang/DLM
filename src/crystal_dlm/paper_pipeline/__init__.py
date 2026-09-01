"""Stable paper-facing interfaces for the coupled C3FD–G2 pipeline."""

from .manifest import (
    MAINLINE_STAGE_ORDER,
    ManifestError,
    command_for_stage,
    load_and_validate,
    stage_spec,
    validate_manifest,
)

__all__ = [
    "MAINLINE_STAGE_ORDER",
    "ManifestError",
    "command_for_stage",
    "load_and_validate",
    "stage_spec",
    "validate_manifest",
]
