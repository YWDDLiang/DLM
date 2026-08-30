"""Small, rich-Plan-style interface between C3FD and the crystal DLM."""

from __future__ import annotations

import json
from typing import Any, Mapping

from crystal_dlm.composition_identity import canonical_symbol_counts


C3FD_NATIVE_PLAN_VERSION = "C3FD_NATIVE_PLAN_V2"
SOFT_FIELD_KEYS = (
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
)
ALLOWED_ANION_FRAMEWORKS = {
    "oxide",
    "halide",
    "sulfide",
    "chalcogenide",
    "nitride",
    "phosphide_or_phosphate",
    "other",
}
REQUIRED_FIELDS = (
    "schema",
    "N",
    "elements",
    "counts",
    "anion_framework",
    *SOFT_FIELD_KEYS,
)


def _payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    n_value = int(plan.get("N") or 0)
    if n_value < 1 or n_value > 20:
        raise ValueError("native Plan N must be in 1..20")
    composition = canonical_symbol_counts(
        [str(value) for value in (plan.get("elements") or ())],
        [int(value) for value in (plan.get("counts") or ())],
    )
    if not composition or sum(count for _symbol, count in composition) != n_value:
        raise ValueError("native Plan violates exact N/count conservation")
    family = str(plan.get("anion_framework") or "").strip()
    if family not in ALLOWED_ANION_FRAMEWORKS:
        raise ValueError(f"unsupported native anion framework {family!r}")
    soft: dict[str, str] = {}
    for field in SOFT_FIELD_KEYS:
        value = str(plan.get(field) or "").strip()
        if not value:
            raise ValueError(f"native Plan lacks {field}")
        soft[field] = value
    return {
        "schema": C3FD_NATIVE_PLAN_VERSION,
        "N": n_value,
        "elements": [symbol for symbol, _count in composition],
        "counts": [int(count) for _symbol, count in composition],
        "anion_framework": family,
        **soft,
    }


def serialize_native_plan(plan: Mapping[str, Any]) -> str:
    """Serialize one portable C3FD Plan in the established rich-JSON style."""

    return json.dumps(
        _payload(plan),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_native_plan_line(line: str) -> dict[str, Any]:
    """Parse an unmasked native Plan and validate exact composition."""

    text = str(line).strip().splitlines()[0] if str(line).strip() else ""
    if text.startswith("c3fd_native_plan:"):
        text = text.split(":", 1)[1].strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("native Plan is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise TypeError("native Plan JSON must be an object")
    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"native Plan is missing fields: {','.join(missing)}")
    if str(raw.get("schema")) != C3FD_NATIVE_PLAN_VERSION:
        raise ValueError("native Plan schema marker changed")
    parsed = _payload(raw)
    parsed["native_plan_version"] = C3FD_NATIVE_PLAN_VERSION
    parsed["native_line"] = text
    return parsed


def mask_native_soft_fields(line: str) -> str:
    """Mask only uncertain structural hints while preserving composition."""

    parsed = parse_native_plan_line(line)
    for field in SOFT_FIELD_KEYS:
        parsed[field] = "<SOFT_MASK>"
    parsed.pop("native_plan_version", None)
    parsed.pop("native_line", None)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def build_native_body_prompt(
    plan: Mapping[str, Any],
    *,
    mask_soft_fields: bool = False,
) -> str:
    line = serialize_native_plan(plan)
    if mask_soft_fields:
        line = mask_native_soft_fields(line)
    return (
        "Generate only the exact-length dynamic crystal body for this C3FD Plan. "
        "N and element counts are hard constraints; lattice system, space-group "
        "bucket, and volume-per-atom bin are soft structural hints.\n"
        f"c3fd_native_plan: {line}\n"
        "dynamic_crystal_body:"
    )


__all__ = [
    "ALLOWED_ANION_FRAMEWORKS",
    "C3FD_NATIVE_PLAN_VERSION",
    "SOFT_FIELD_KEYS",
    "build_native_body_prompt",
    "mask_native_soft_fields",
    "parse_native_plan_line",
    "serialize_native_plan",
]
