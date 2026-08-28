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


def validate_row(
    row: Mapping[str, Any],
    tokenizer,
    certificate_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = row.get("plan_state")
    spec = row.get("minimal_spec")
    if not isinstance(plan, Mapping) or not isinstance(spec, Mapping):
        raise ValueError("minimal row requires plan_state and minimal_spec")
    if certificate_row is not None:
        source_row_idx = int(row.get("c3fd_certificate_source_row_idx", -1))
        if source_row_idx < 0:
            raise ValueError("minimal row lacks C³FD certificate source index")
        if int(certificate_row.get("source_row_idx", -2)) != source_row_idx:
            raise ValueError("C³FD certificate source index changed")
        certificate_plan = certificate_row.get("plan_state")
        if not isinstance(certificate_plan, Mapping):
            raise ValueError("C³FD certificate lacks plan_state")
        for key in ("formula", "N", "elements", "counts"):
            if plan.get(key) != certificate_plan.get(key):
                raise ValueError(f"DLM/C³FD row alignment changed for {key}")
    expected = minimal_spec_from_plan(plan, certificate_row)
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
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    static_assets = manifest.get("static_assets") or {}
    vocab_path = args.data_dir / "vocab_tokens.txt"
    expected_vocab_sha = str(
        static_assets.get("vocab_tokens.txt", {}).get("sha256") or ""
    )
    actual_vocab_sha = (
        hashlib.sha256(vocab_path.read_bytes()).hexdigest()
        if vocab_path.is_file()
        else ""
    )
    certificate_dir = Path(str(manifest.get("certificate_dir") or ""))
    if not certificate_dir.is_dir():
        raise FileNotFoundError(
            f"frozen C³FD certificate directory is unavailable: {certificate_dir}"
        )
    split_reports: dict[str, Any] = {}
    all_failures: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        path = args.data_dir / f"{split}.jsonl"
        certificate_path = certificate_dir / f"{split}.jsonl"
        expected_certificate_sha = str(
            manifest.get("splits", {}).get(split, {}).get("certificate_sha256") or ""
        )
        actual_certificate_sha = hashlib.sha256(certificate_path.read_bytes()).hexdigest()
        if not expected_certificate_sha or actual_certificate_sha != expected_certificate_sha:
            raise ValueError(f"frozen C³FD certificate hash changed for {split}")
        certificates = {
            int(value.get("source_row_idx", -1)): value
            for value in iter_jsonl(certificate_path)
        }
        lengths: list[int] = []
        identities: set[str] = set()
        failures: Counter[str] = Counter()
        rows = 0
        for row in iter_jsonl(path):
            rows += 1
            try:
                source_row_idx = int(row.get("c3fd_certificate_source_row_idx", -1))
                certificate_row = certificates.get(source_row_idx)
                if certificate_row is None:
                    raise ValueError("C³FD certificate source row is unavailable")
                result = validate_row(row, tokenizer, certificate_row)
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
            "certificate_sha256": actual_certificate_sha,
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
        "vocab_tokens_present_and_hash_matched": bool(expected_vocab_sha)
        and actual_vocab_sha == expected_vocab_sha,
    }
    gate["minimal_spec_data_authorized"] = all(gate.values())
    report = {
        "schema": "h1a2_ctv_minimal_spec_audit_v1",
        "data_dir": str(args.data_dir.resolve()),
        "model_path": str(args.model_path.resolve()),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "vocab_tokens_sha256": actual_vocab_sha,
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
