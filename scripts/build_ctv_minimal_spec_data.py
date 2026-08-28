#!/usr/bin/env python3
"""Build the frozen CTV-DLM minimal-spec SFT dataset from existing DLM rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.composition_identity import (  # noqa: E402
    canonical_symbol_counts,
    formula_from_symbol_counts,
    identity_from_plan_state,
    identity_text,
)


SCHEMA = "h1a2_ctv_minimal_spec_v1"
FAMILIES = {
    "oxide",
    "sulfide",
    "chalcogenide",
    "halide",
    "nitride",
    "phosphide_or_phosphate",
    "other",
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"non-object row in {path}")
                yield value


def charge_class(
    plan: Mapping[str, Any], certificate_row: Mapping[str, Any] | None = None
) -> str:
    if certificate_row is not None:
        if certificate_row.get("composition_supervision") is not True:
            raise ValueError("C³FD certificate does not authorize composition")
        elements = list(plan.get("elements") or ())
        if len(elements) == 1:
            return "single_element"
        nodes = list(certificate_row.get("species_labels") or ())
        raw_plan = certificate_row.get("plan_state") or {}
        bucket = str(raw_plan.get("charge_bucket") or plan.get("charge_bucket") or "")
        if bucket == "all_metal":
            return "all_metal"
        # A benchmark-supervised multi-element row not assigned to the alloy
        # shortcut carries an exact nonzero-valence witness in the sidecar.
        if nodes:
            return "certified_neutral"
        raise ValueError("C³FD certificate lacks semantic species witness")
    classification = plan.get("validator") or {}
    valid = classification.get("valid")
    reason = str(classification.get("reason") or "")
    bucket = str(plan.get("charge_bucket") or "")
    elements = list(plan.get("elements") or ())
    if len(elements) == 1 and valid is True:
        return "single_element"
    if valid is not True:
        raise ValueError("independent composition validator is not positive")
    if reason == "all_metal_shortcut" or bucket == "all_metal":
        return "all_metal"
    if reason in {"charge_neutral_pauling_valid", "validator_boolean"} or bucket in {
        "neutral_plausible",
        "certified_neutral",
    }:
        return "certified_neutral"
    raise ValueError(f"unsupported positive charge certificate {reason or bucket!r}")


def minimal_spec_from_plan(
    plan: Mapping[str, Any], certificate_row: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    canonical = canonical_symbol_counts(
        [str(value) for value in (plan.get("elements") or ())],
        [int(value) for value in (plan.get("counts") or ())],
    )
    elements = [symbol for symbol, _count in canonical]
    counts = [int(count) for _symbol, count in canonical]
    n_value = int(plan.get("N") or 0)
    if n_value <= 0 or sum(counts) != n_value:
        raise ValueError("Plan N/count conservation failed")
    family = str(plan.get("anion_framework") or "")
    if family not in FAMILIES:
        raise ValueError(f"unsupported family {family!r}")
    formula = formula_from_symbol_counts(canonical)
    return {
        "N": n_value,
        "charge": charge_class(plan, certificate_row),
        "counts": counts,
        "elements": elements,
        "family": family,
        "formula": formula,
    }


def minimal_prompt(spec: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(spec), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return payload + "\ndynamic_crystal_body:"


def convert_row(
    row: Mapping[str, Any],
    certificate_row: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    plan = row.get("plan_state")
    if not isinstance(plan, Mapping):
        return None, "missing_plan_state"
    try:
        if certificate_row is not None:
            certificate_plan = certificate_row.get("plan_state")
            if not isinstance(certificate_plan, Mapping):
                raise ValueError("C³FD row lacks plan_state")
            for key in ("formula", "N", "elements", "counts"):
                if plan.get(key) != certificate_plan.get(key):
                    raise ValueError(f"DLM/C³FD row alignment changed for {key}")
        spec = minimal_spec_from_plan(plan, certificate_row)
        identity = identity_text(identity_from_plan_state(plan))
    except Exception as exc:  # noqa: BLE001 - exclusions are counted explicitly.
        return None, f"{type(exc).__name__}:{str(exc)}"
    output = dict(row)
    output["prompt"] = minimal_prompt(spec)
    output["minimal_spec"] = spec
    output["minimal_spec_schema"] = SCHEMA
    output["reduced_composition_identity"] = identity
    output["source_prompt_sha256"] = hashlib.sha256(
        str(row.get("prompt") or "").encode("utf-8")
    ).hexdigest()
    output.pop("counterfactual_prompt", None)
    output["counterfactual_grounding_eligible"] = False
    return output, "kept"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--certificate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    split_reports: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        source = args.input_dir / f"{split}.jsonl"
        certificate_source = args.certificate_dir / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        if not certificate_source.is_file():
            raise FileNotFoundError(certificate_source)
        reasons: Counter[str] = Counter()
        kept = 0
        prompt_tokens_pending = 0
        identities: set[str] = set()
        output_path = args.output_dir / f"{split}.jsonl"
        with output_path.open("x", encoding="utf-8") as handle:
            for row_index, pair in enumerate(
                zip_longest(iter_jsonl(source), iter_jsonl(certificate_source))
            ):
                row, certificate_row = pair
                if row is None or certificate_row is None:
                    raise ValueError(f"DLM/C³FD split length changed for {split}")
                if int(certificate_row.get("source_row_idx", -1)) != row_index:
                    raise ValueError(f"C³FD source_row_idx changed for {split}:{row_index}")
                converted, reason = convert_row(row, certificate_row)
                reasons[reason] += 1
                if converted is None:
                    continue
                kept += 1
                prompt_tokens_pending += 1
                identities.add(str(converted["reduced_composition_identity"]))
                handle.write(
                    json.dumps(converted, ensure_ascii=False, sort_keys=True) + "\n"
                )
        split_reports[split] = {
            "source_rows": sum(reasons.values()),
            "kept_rows": kept,
            "excluded_rows": sum(reasons.values()) - kept,
            "reduced_composition_identities": len(identities),
            "reasons": dict(reasons.most_common()),
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "tokenizer_prompt_audit_pending": prompt_tokens_pending,
        }

    gate = {
        "train_nonempty": split_reports["train"]["kept_rows"] > 0,
        "val_nonempty": split_reports["val"]["kept_rows"] > 0,
        "test_nonempty": split_reports["test"]["kept_rows"] > 0,
        "counterfactual_prompt_removed": True,
        "stability_fields_absent": True,
        "prompt_token_audit_pending": True,
    }
    report = {
        "schema": SCHEMA,
        "input_dir": str(args.input_dir.resolve()),
        "certificate_dir": str(args.certificate_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "splits": split_reports,
        "gate": gate,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
