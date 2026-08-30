#!/usr/bin/env python3
"""Freeze schema-faithful H0/R0S development diagnostic views.

This builder is deliberately outcome-blind.  It repairs only deterministic
legacy prompt fields and creates a current-runtime replay of the first 256
parsed historical H1-A2 Plans.  The outputs are development diagnostics, not a
prospective or official evaluation cohort.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
for value in (PROJECT_ROOT, SRC, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import freeze_rich_recovery_cohort as RICH  # noqa: E402
from crystal_dlm.r5_plan_state import canonical_plan_state  # noqa: E402


SCHEMA = "h1a2_faithful_rich_diagnostic_v1"
ALLOWED_R0S_PLAN_CHANGES = ("oxidation_candidates", "prototype_key")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise TypeError(f"non-object at {path}:{line_number}")
            yield value


def changed_plan_fields(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[str]:
    keys = sorted(set(before) | set(after))
    return [
        key
        for key in keys
        if canonical_json(before.get(key)) != canonical_json(after.get(key))
    ]


def source_indices(row: Mapping[str, Any], ordinal: int) -> tuple[int, int]:
    source_sample_idx = int(row.get("source_sample_idx", row.get("sample_idx", ordinal)))
    source_ordinal = int(row.get("source_ordinal", ordinal))
    return source_sample_idx, source_ordinal


def composition_metadata(plan: Mapping[str, Any]) -> dict[str, str]:
    return {
        "exact_composition_identity": RICH.exact_identity(plan),
        "reduced_composition_identity": RICH.reduced_identity(plan),
        "chemsys": RICH.chemsys(plan),
    }


def validate_canonical_prompt_plan(plan: Mapping[str, Any], *, label: str) -> None:
    payload = canonical_plan_state(plan)
    null_fields = [key for key, value in payload.items() if value is None]
    if null_fields:
        raise ValueError(f"{label} canonical rich plan contains null fields: {null_fields}")
    if not isinstance(payload["N"], int) or isinstance(payload["N"], bool):
        raise TypeError(f"{label} N must be int")
    if not isinstance(payload["elements"], list) or not all(
        isinstance(value, str) for value in payload["elements"]
    ):
        raise TypeError(f"{label} elements must be list[str]")
    if not isinstance(payload["counts"], list) or not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in payload["counts"]
    ):
        raise TypeError(f"{label} counts must be list[int]")
    if len(payload["elements"]) != len(payload["counts"]):
        raise ValueError(f"{label} elements/counts length mismatch")
    if sum(payload["counts"]) != payload["N"]:
        raise ValueError(f"{label} counts do not sum to N")
    for field in (
        "formula",
        "reduced_formula",
        "charge_bucket",
        "anion_framework",
        "lattice_system",
        "spacegroup_bucket",
        "volume_per_atom_bin",
        "prototype_key",
    ):
        if not isinstance(payload[field], str) or not payload[field]:
            raise TypeError(f"{label} {field} must be a non-empty string")
    if not isinstance(payload["oxidation_candidates"], (str, list)):
        raise TypeError(f"{label} oxidation_candidates must be str or list")


def freeze_r0s(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(source_rows) < int(count):
        raise RuntimeError(f"R0 source has {len(source_rows)} rows, need {count}")
    output: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    exact_mismatches = 0
    soft_mismatches = 0
    change_counts = {field: 0 for field in ALLOWED_R0S_PLAN_CHANGES}
    for ordinal, source in enumerate(source_rows[: int(count)]):
        original = RICH.find_plan_state(source)
        if original is None:
            raise ValueError(f"R0 source ordinal {ordinal} has no plan_state")
        repaired = RICH.canonicalize_rich_plan(original)
        validate_canonical_prompt_plan(repaired, label=f"R0S[{ordinal}]")
        changed = changed_plan_fields(original, repaired)
        unexpected = sorted(set(changed) - set(ALLOWED_R0S_PLAN_CHANGES))
        if unexpected:
            raise RuntimeError(f"R0S changed non-schema fields at {ordinal}: {unexpected}")
        for field in changed:
            change_counts[field] += 1
        exact_before = RICH.exact_identity(original)
        exact_after = RICH.exact_identity(repaired)
        soft_before = RICH.soft_tuple(original)
        soft_after = RICH.soft_tuple(repaired)
        exact_mismatches += int(exact_before != exact_after)
        soft_mismatches += int(soft_before != soft_after)
        source_sample_idx, source_ordinal = source_indices(source, ordinal)
        prompt = RICH.build_body_prompt(repaired).rstrip() + "\n"
        row = {
            **deepcopy(dict(source)),
            "schema": SCHEMA,
            "view": "R0S",
            "sample_idx": ordinal,
            "source_sample_idx": source_sample_idx,
            "source_ordinal": source_ordinal,
            "plan_state": repaired,
            "prompt": prompt,
            **composition_metadata(repaired),
            "diagnostic_role": "schema_corrected_current_c3fd_rich",
        }
        output.append(row)
        ledger.append(
            {
                "schema": SCHEMA,
                "view": "R0S",
                "sample_idx": ordinal,
                "source_sample_idx": source_sample_idx,
                "source_ordinal": source_ordinal,
                "exact_composition_identity": exact_after,
                "soft_tuple": list(soft_after),
                "allowed_plan_changes": changed,
                "source_prompt_sha256": hashlib.sha256(
                    str(source.get("prompt", "")).encode("utf-8")
                ).hexdigest(),
                "output_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    if exact_mismatches or soft_mismatches:
        raise RuntimeError("R0S schema repair changed composition or sampled soft fields")
    return output, ledger, {
        "rows": len(output),
        "exact_identity_mismatches": exact_mismatches,
        "soft_tuple_mismatches": soft_mismatches,
        "schema_change_counts": change_counts,
        "unique_exact_identities": len(
            {RICH.exact_identity(row["plan_state"]) for row in output}
        ),
    }


def freeze_h0(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(source_rows) < int(count):
        raise RuntimeError(f"H0 source has {len(source_rows)} rows, need {count}")
    output: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    prompt_matches = 0
    changed_counts = {field: 0 for field in ALLOWED_R0S_PLAN_CHANGES}
    for ordinal, source in enumerate(source_rows[: int(count)]):
        original = RICH.find_plan_state(source)
        if original is None:
            raise ValueError(f"H0 source ordinal {ordinal} has no plan_state")
        canonical = RICH.canonicalize_rich_plan(original)
        validate_canonical_prompt_plan(canonical, label=f"H0[{ordinal}]")
        changed = changed_plan_fields(original, canonical)
        unexpected = sorted(set(changed) - set(ALLOWED_R0S_PLAN_CHANGES))
        if unexpected:
            raise RuntimeError(f"H0 canonicalization changed fields at {ordinal}: {unexpected}")
        for field in changed:
            changed_counts[field] += 1
        prompt = RICH.build_body_prompt(canonical).rstrip() + "\n"
        source_prompt = str(source.get("prompt", "")).rstrip() + "\n"
        exact_match = prompt == source_prompt
        prompt_matches += int(exact_match)
        source_sample_idx = int(source.get("sample_idx", ordinal))
        row = {
            **deepcopy(dict(source)),
            "schema": SCHEMA,
            "view": "H0",
            "sample_idx": ordinal,
            "source_sample_idx": source_sample_idx,
            "source_ordinal": ordinal,
            "plan_state": canonical,
            "prompt": prompt,
            **composition_metadata(canonical),
            "diagnostic_role": "historical_h1_first256_current_runtime",
        }
        output.append(row)
        ledger.append(
            {
                "schema": SCHEMA,
                "view": "H0",
                "sample_idx": ordinal,
                "source_sample_idx": source_sample_idx,
                "source_ordinal": ordinal,
                "exact_composition_identity": RICH.exact_identity(canonical),
                "soft_tuple": list(RICH.soft_tuple(canonical)),
                "canonical_plan_changes": changed,
                "source_prompt_exact_match": exact_match,
                "source_prompt_sha256": hashlib.sha256(source_prompt.encode("utf-8")).hexdigest(),
                "output_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    return output, ledger, {
        "rows": len(output),
        "source_prompt_exact_matches": prompt_matches,
        "source_prompt_exact_match_rate": prompt_matches / len(output),
        "canonical_change_counts": changed_counts,
        "unique_exact_identities": len(
            {RICH.exact_identity(row["plan_state"]) for row in output}
        ),
    }


def length_stats(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("length statistics require rows")
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": float(statistics.median(values)),
        "min": min(values),
        "max": max(values),
    }


def prompt_stats(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any | None,
) -> dict[str, Any]:
    prompts = []
    for row in rows:
        if row.get("prompt") is not None:
            prompts.append(str(row["prompt"]))
            continue
        plan = RICH.find_plan_state(row)
        if plan is None:
            raise ValueError("prompt statistics row has neither prompt nor plan_state")
        prompts.append(RICH.build_body_prompt(RICH.canonicalize_rich_plan(plan)))
    result: dict[str, Any] = {
        "characters": length_stats([len(prompt) for prompt in prompts]),
    }
    if tokenizer is not None:
        result["tokens"] = length_stats(
            [
                len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                for prompt in prompts
            ]
        )
    return result


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
    return sha256_file(path)


def render_markdown(manifest: Mapping[str, Any]) -> str:
    r0s = manifest["views"]["R0S"]
    h0 = manifest["views"]["H0"]
    lines = [
        "# Faithful rich-interface diagnostic cohort",
        "",
        "This is an outcome-blind, post-outcome development diagnosis. It is not",
        "a prospective or official scientific result.",
        "",
        "## Frozen views",
        "",
        f"- R0S rows: {r0s['rows']}; exact mismatches: {r0s['exact_identity_mismatches']}; soft-tuple mismatches: {r0s['soft_tuple_mismatches']}.",
        f"- R0S schema repairs: `{canonical_json(r0s['schema_change_counts'])}`.",
        f"- H0 rows: {h0['rows']}; source prompt exact matches: {h0['source_prompt_exact_matches']}/{h0['rows']}.",
        "- Neither view reads energy, stability, hull, validity, or prior canary outcomes.",
        "",
        "## Prompt lengths",
        "",
    ]
    for name, stats in manifest["prompt_stats"].items():
        token_text = "not loaded"
        if "tokens" in stats:
            token = stats["tokens"]
            token_text = f"mean {token['mean']:.2f}, median {token['median']:.1f}, range {token['min']}--{token['max']}"
        lines.append(f"- {name}: tokens {token_text}; chars mean {stats['characters']['mean']:.2f}.")
    lines.extend(
        [
            "",
            "## Interpretation contract",
            "",
            "- H0 tests current-runtime compatibility on the historical first256 parsed Plan sequence.",
            "- R0S tests deterministic schema repair on the frozen seed19 C3FD predictions.",
            "- No official MP query is permitted for this development diagnostic.",
            "",
        ]
    )
    return "\n".join(lines)


def freeze_to_directory(
    *,
    r0_source: Sequence[Mapping[str, Any]],
    h0_source: Sequence[Mapping[str, Any]],
    output_dir: Path,
    count: int,
    input_provenance: Mapping[str, Any],
    tokenizer: Any | None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    preparing = output_dir.with_name(output_dir.name + ".preparing")
    if preparing.exists():
        raise FileExistsError(preparing)
    preparing.mkdir(parents=True)
    try:
        r0s_rows, r0s_ledger, r0s_report = freeze_r0s(r0_source, count=count)
        h0_rows, h0_ledger, h0_report = freeze_h0(h0_source, count=count)
        ledgers = r0s_ledger + h0_ledger
        output_hashes = {
            "R0S.jsonl": write_jsonl(preparing / "R0S.jsonl", r0s_rows),
            "H0.jsonl": write_jsonl(preparing / "H0.jsonl", h0_rows),
            "DIAGNOSTIC_LEDGER.jsonl": write_jsonl(
                preparing / "DIAGNOSTIC_LEDGER.jsonl", ledgers
            ),
        }
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "frozen_outcome_blind_development_diagnostic",
            "requested_per_view": int(count),
            "input_provenance": dict(input_provenance),
            "views": {"R0S": r0s_report, "H0": h0_report},
            "prompt_stats": {
                "R0_source": prompt_stats(r0_source[: int(count)], tokenizer),
                "R0S": prompt_stats(r0s_rows, tokenizer),
                "H0_source": prompt_stats(h0_source[: int(count)], tokenizer),
                "H0": prompt_stats(h0_rows, tokenizer),
            },
            "allowed_r0s_plan_changes": list(ALLOWED_R0S_PLAN_CHANGES),
            "outcome_fields_read": [],
            "official_query_permitted": False,
            "confirmatory_claim_permitted": False,
            "output_hashes": output_hashes,
        }
        manifest_path = preparing / "FAITHFUL_RICH_DIAGNOSTIC_MANIFEST.json"
        manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        markdown_path = preparing / "FAITHFUL_RICH_DIAGNOSTIC_MANIFEST.md"
        markdown_path.write_text(render_markdown(manifest), encoding="utf-8")
        checks = {
            **output_hashes,
            manifest_path.name: sha256_file(manifest_path),
            markdown_path.name: sha256_file(markdown_path),
        }
        (preparing / "OUTPUTS.sha256").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(checks.items())),
            encoding="utf-8",
        )
        (preparing / "_SUCCESS").write_text(
            canonical_json({"schema": SCHEMA, "status": "success"}) + "\n",
            encoding="utf-8",
        )
        os.replace(preparing, output_dir)
        return manifest
    except Exception:
        # Preserve any failed preparation for root-cause analysis.
        if preparing.exists():
            (preparing / "_FAILED").touch(exist_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-source", type=Path, required=True)
    parser.add_argument("--h0-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--tokenizer-path", type=Path)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    tokenizer = None
    if args.tokenizer_path is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer_path),
            local_files_only=True,
        )
    r0_rows = list(iter_jsonl(args.r0_source))
    h0_rows = list(iter_jsonl(args.h0_source))
    provenance = {
        "r0": {
            "path": str(args.r0_source.resolve()),
            "sha256": sha256_file(args.r0_source),
            "rows": len(r0_rows),
        },
        "h0": {
            "path": str(args.h0_source.resolve()),
            "sha256": sha256_file(args.h0_source),
            "rows": len(h0_rows),
        },
        "tokenizer": None
        if args.tokenizer_path is None
        else {"path": str(args.tokenizer_path.resolve())},
    }
    manifest = freeze_to_directory(
        r0_source=r0_rows,
        h0_source=h0_rows,
        output_dir=args.output_dir,
        count=args.count,
        input_provenance=provenance,
        tokenizer=tokenizer,
    )
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
