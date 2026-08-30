"""Typed, deployment-native interface between C3FD and the crystal DLM."""

from __future__ import annotations

import re
from typing import Any, Mapping

from crystal_dlm.r5_plan_state import (
    parse_countvalence_plan_state,
    plan_state_to_countvalencefields,
)


C3FD_NATIVE_PLAN_VERSION = "C3FD_NATIVE_PLAN_V1"
SOFT_FIELD_KEYS = ("LS", "SG", "VP")
ALLOWED_ANION_FRAMEWORKS = {
    "oxide",
    "halide",
    "sulfide",
    "chalcogenide",
    "nitride",
    "phosphide_or_phosphate",
    "other",
}


def _field_map(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for chunk in str(line).strip().split(";"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip().upper()
        if key in fields:
            raise ValueError(f"native Plan duplicates field {key}")
        fields[key] = value.strip()
    return fields


def serialize_native_plan(plan: Mapping[str, Any]) -> str:
    """Serialize one current C3FD Plan without legacy H1 prompt fields."""

    n_value = int(plan.get("N") or 0)
    if n_value < 1 or n_value > 20:
        raise ValueError("native Plan N must be in 1..20")
    family = str(plan.get("anion_framework") or "").strip()
    if family not in ALLOWED_ANION_FRAMEWORKS:
        raise ValueError(f"unsupported native anion framework {family!r}")
    core = plan_state_to_countvalencefields(plan)
    line = f"{C3FD_NATIVE_PLAN_VERSION};N=N{n_value:03d};AF={family};{core}"
    parsed = parse_native_plan_line(line)
    if int(parsed["N"]) != n_value:
        raise ValueError("native Plan serialization changed N")
    return line


def parse_native_plan_line(line: str) -> dict[str, Any]:
    """Parse an unmasked native Plan and validate exact typed composition."""

    text = str(line).strip().splitlines()[0] if str(line).strip() else ""
    if not text.startswith(C3FD_NATIVE_PLAN_VERSION + ";"):
        raise ValueError("native Plan version marker is missing")
    fields = _field_map(text)
    n_token = fields.get("N", "")
    match = re.fullmatch(r"N(\d{3})", n_token.upper())
    if match is None:
        raise ValueError("native Plan N token is malformed")
    declared_n = int(match.group(1))
    family = fields.get("AF", "")
    if family not in ALLOWED_ANION_FRAMEWORKS:
        raise ValueError("native Plan AF field is unsupported")
    core_keys = [f"P{index:02d}" for index in range(1, 8)] + ["CB", "LS", "SG", "VP"]
    missing = [key for key in core_keys if key not in fields]
    if missing:
        raise ValueError(f"native Plan is missing fields: {','.join(missing)}")
    core = ";".join(f"{key}={fields[key]}" for key in core_keys)
    parsed = parse_countvalence_plan_state(core)
    if int(parsed["N"]) != declared_n:
        raise ValueError("native Plan declared N disagrees with species counts")
    parsed["native_plan_version"] = C3FD_NATIVE_PLAN_VERSION
    parsed["anion_framework"] = family
    parsed["native_line"] = text
    return parsed


def mask_native_soft_fields(line: str) -> str:
    """Mask only uncertain structure hints while preserving chemistry fields."""

    text = str(line).strip().splitlines()[0] if str(line).strip() else ""
    if not text.startswith(C3FD_NATIVE_PLAN_VERSION + ";"):
        raise ValueError("native Plan version marker is missing")
    chunks = text.split(";")
    output = []
    seen: set[str] = set()
    for chunk in chunks:
        if "=" not in chunk:
            output.append(chunk)
            continue
        key, _value = chunk.split("=", 1)
        key_upper = key.strip().upper()
        if key_upper in SOFT_FIELD_KEYS:
            output.append(f"{key_upper}=<SOFT_MASK>")
            seen.add(key_upper)
        else:
            output.append(chunk)
    if seen != set(SOFT_FIELD_KEYS):
        raise ValueError("native Plan does not contain all soft fields")
    return ";".join(output)


def build_native_body_prompt(
    plan: Mapping[str, Any],
    *,
    mask_soft_fields: bool = False,
) -> str:
    line = serialize_native_plan(plan)
    if mask_soft_fields:
        line = mask_native_soft_fields(line)
    return (
        "Generate only the exact-length dynamic crystal body for this C3FD-native Plan. "
        "N and typed species counts are hard constraints; LS, SG, and VP are soft structural hints.\n"
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
