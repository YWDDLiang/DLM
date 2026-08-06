"""Read-only path guard for the frozen H1 submission fallback.

The local planning mirror does not currently have usable Git metadata, so new
experiment entry points must reject output paths that overlap the frozen H1
source, bundles, or run identities.  The check deliberately works for both the
local checkout and an execution-cluster checkout with a different absolute
project root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


H1_RESERVED_RELATIVE_ROOTS: tuple[str, ...] = (
    "workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation",
    "workstreams/r5c_reactivation_20260728/r5c_frozen_baseline_20260728",
    "runs/20260603_034533-h1a2-epoch2-3-fullmetrics",
    "runs/20260729_h1a2c_jointchem_v1",
    "runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1",
    "runs/20260731_h1a2c_p0_p1_sun256_mpcomplete_v4",
)

H1_RESERVED_RELATIVE_FILES: tuple[str, ...] = (
    "workstreams/r5c_reactivation_20260728/r5c_frozen_baseline_20260728.tar.gz",
    "workstreams/r5c_reactivation_20260728/r5c_frozen_baseline_20260728.tar.gz.sha256",
    "workstreams/r5c_reactivation_20260728/r5c_reactivation_bundle_20260728.tar.gz",
    "workstreams/r5c_reactivation_20260728/r5c_reactivation_bundle_20260728.tar.gz.sha256",
)


class H1ReadOnlyViolation(ValueError):
    """Raised when a proposed output path overlaps a frozen H1 identity."""


def default_project_root() -> Path:
    """Return the source checkout root containing ``crystal_dlm``."""

    return Path(__file__).resolve().parents[1]


def _contains_path_parts(candidate: Path, relative_path: str) -> bool:
    """Return whether ``relative_path`` occurs as whole components.

    This catches a remote path such as
    ``/public/home/.../project/runs/<frozen-run>`` even when the local project
    root is ``/mnt/d/.../project``.  Whole-component matching avoids rejecting
    innocent names that merely contain an H1 run name as a substring.
    """

    candidate_parts = candidate.parts
    target_parts = Path(relative_path).parts
    width = len(target_parts)
    if width == 0 or width > len(candidate_parts):
        return False
    return any(
        candidate_parts[offset : offset + width] == target_parts
        for offset in range(len(candidate_parts) - width + 1)
    )


def _equal_or_descendant(candidate: Path, reserved: Path) -> bool:
    return candidate == reserved or reserved in candidate.parents


def frozen_h1_match(
    output_path: str | Path,
    *,
    project_root: str | Path | None = None,
    reserved_roots: Sequence[str] = H1_RESERVED_RELATIVE_ROOTS,
    reserved_files: Sequence[str] = H1_RESERVED_RELATIVE_FILES,
) -> str | None:
    """Return the matching frozen relative identity, if any.

    Existing symlink components are resolved before comparison.  Non-existing
    paths are still normalized, so ``..`` cannot be used to bypass the guard.
    """

    root = Path(project_root or default_project_root()).expanduser().resolve()
    candidate = Path(output_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=False)

    for relative_root in reserved_roots:
        local_reserved = (root / relative_root).resolve(strict=False)
        if _equal_or_descendant(candidate, local_reserved):
            return relative_root
        if _contains_path_parts(candidate, relative_root):
            return relative_root

    for relative_file in reserved_files:
        local_reserved = (root / relative_file).resolve(strict=False)
        if candidate == local_reserved:
            return relative_file
        if _contains_path_parts(candidate, relative_file):
            return relative_file
    return None


def assert_writable_output_path(
    output_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Normalize and return an allowed output path.

    Raises:
        H1ReadOnlyViolation: if the path overlaps a frozen H1 identity.
    """

    root = Path(project_root or default_project_root()).expanduser().resolve()
    candidate = Path(output_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=False)
    matched = frozen_h1_match(candidate, project_root=root)
    if matched is not None:
        raise H1ReadOnlyViolation(
            f"Refusing output path {candidate}: frozen H1 identity {matched!r} is read-only"
        )
    return candidate


def assert_writable_output_paths(
    output_paths: Iterable[str | Path],
    *,
    project_root: str | Path | None = None,
) -> list[Path]:
    """Validate multiple output paths with the same project-root policy."""

    return [
        assert_writable_output_path(path, project_root=project_root)
        for path in output_paths
    ]


__all__ = [
    "H1_RESERVED_RELATIVE_FILES",
    "H1_RESERVED_RELATIVE_ROOTS",
    "H1ReadOnlyViolation",
    "assert_writable_output_path",
    "assert_writable_output_paths",
    "default_project_root",
    "frozen_h1_match",
]
