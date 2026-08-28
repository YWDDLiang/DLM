#!/usr/bin/env python3
"""Audit the frozen CTV minimal prompt with the exact DLM tokenizer."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.composition_identity import (  # noqa: E402
    identity_from_plan_state,
    identity_text,
)

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_ctv_minimal_spec_data import minimal_prompt, minimal_spec_from_plan  # noqa: E402


FORBIDDEN_KEYS = {
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
    "prototype_key",
    "target_stability",
    "energy",
    "E0",
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def validate_row(row: Mapping[str, Any], tokenizer) -> dict[str, Any]:
    plan = row.get("plan_state")
    spec = row.get("minimal_spec")
    if not isinstance(plan, Mapping) or not isinstance(spec, Mapping):
        raise ValueError("minimal row requires plan_state and minimal_spec")
    expected = minimal_spec_from_plan(plan)
    if dict(spec) != expected:
        raise ValueError("minimal_spec does not match deterministic Plan projection")
    expected_prompt = minimal_prompt(expected)
    if str(row.get("prompt")) != expected_prompt:
        raise ValueError("minimal prompt serialization changed")
    if FORBIDDEN_KEYS.intersection(spec):
        raise ValueError("minimal spec leaked a forbidden soft/stability field")
    identity = identity_text(identity_from_plan_state(plan))
    if str(row.get("reduced_composition_identity")) != identity:
        raise ValueError("reduced composition identity changed")
    token_ids = tokenizer(expected_prompt, add_special_tokens=False)["input_ids"]
    if not row.get("answer"):
        raise ValueError("minimal row lacks DLM answer")
    return {"prompt_tokens": len(token_ids), "identity": identity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), trust_remote_code=True
    )
    split_reports: dict[str, Any] = {}
    all_failures: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        path = args.data_dir / f"{split}.jsonl"
        lengths: list[int] = []
        identities: set[str] = set()
        failures: Counter[str] = Counter()
        rows = 0
        for row in iter_jsonl(path):
            rows += 1
            try:
                result = validate_row(row, tokenizer)
                lengths.append(int(result["prompt_tokens"]))
                identities.add(str(result["identity"]))
            except Exception as exc:  # noqa: BLE001 - every failure is audited.
                failures[f"{type(exc).__name__}:{str(exc)}"] += 1
        all_failures.update(failures)
        split_reports[split] = {
            "rows": rows,
            "valid_rows": len(lengths),
            "reduced_identities": len(identities),
            "prompt_tokens_min": min(lengths) if lengths else None,
            "prompt_tokens_median": statistics.median(lengths) if lengths else None,
            "prompt_tokens_max": max(lengths) if lengths else None,
            "failures": dict(failures.most_common()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    gate = {
        "all_rows_valid": not all_failures,
        "all_prompt_lengths_within_128": all(
            int(report["prompt_tokens_max"] or 10**9) <= int(args.max_prompt_tokens)
            for report in split_reports.values()
        ),
        "train_nonempty": split_reports["train"]["valid_rows"] > 0,
        "val_nonempty": split_reports["val"]["valid_rows"] > 0,
        "test_nonempty": split_reports["test"]["valid_rows"] > 0,
        "soft_and_stability_fields_absent": True,
    }
    gate["minimal_spec_data_authorized"] = all(gate.values())
    report = {
        "schema": "h1a2_ctv_minimal_spec_audit_v1",
        "data_dir": str(args.data_dir.resolve()),
        "model_path": str(args.model_path.resolve()),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "splits": split_reports,
        "gate": gate,
    }
    args.output_dir.mkdir(parents=True)
    stem = "CTV_MINIMAL_SPEC_DATA_AUDIT"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# CTV-DLM minimal-spec data audit",
        "",
        f"Authorized: **{gate['minimal_spec_data_authorized']}**",
        f"Tokenizer: `{type(tokenizer).__name__}`; vocab `{len(tokenizer)}`.",
        "",
    ]
    for split, value in split_reports.items():
        lines.append(
            f"- {split}: rows `{value['rows']}`, identities `{value['reduced_identities']}`, "
            f"prompt tokens min/median/max `{value['prompt_tokens_min']}/"
            f"{value['prompt_tokens_median']}/{value['prompt_tokens_max']}`"
        )
    (args.output_dir / f"{stem}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
