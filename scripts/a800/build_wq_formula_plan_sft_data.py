#!/usr/bin/env python3
"""Build chemistry-plan/body continuation SFT from the frozen WQ train set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.formula_plan import (  # noqa: E402
    FORMULA_BODY_SYSTEM_PROMPT,
    FORMULA_PLAN_SYSTEM_PROMPT,
    FORMULA_PLAN_USER_PROMPT,
    formula_body_user_prompt,
    formula_plan_from_state,
    serialize_formula_plan,
)
from crystal_dlm.wqcodiff.crysllmgen.lora import sha256_file  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.sft_data import (  # noqa: E402
    tokenize_sft_example,
)
from crystal_dlm.wqcodiff.crysllmgen.wq_text import parse_wq_proposal  # noqa: E402


IDENTITY = "wq_formula_plan_sft_pilot_v1"
EXAMPLE_SCHEMA = "crysllmgen_formula_plan_sft_example_v1"
MANIFEST_SCHEMA = "crysllmgen_formula_plan_sft_manifest_v1"
TOKEN_AUDIT_SCHEMA = "crysllmgen_formula_plan_sft_token_audit_v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _derived_id(source_id: str, task: str) -> str:
    digest = hashlib.sha256(f"{IDENTITY}|{source_id}|{task}".encode("utf-8"))
    return f"fpsft-{digest.hexdigest()[:24]}"


def formula_plan_training_examples(
    source: Mapping[str, Any],
    *,
    catalog: Any,
    replay_direct_edit: bool,
    require_composition_valid: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Transform one frozen source row without using evaluation information."""

    if source.get("schema") != "crysllmgen_sft_example_v1":
        raise ValueError("unexpected source SFT example schema")
    source_id = str(source.get("example_id", ""))
    stage = str(source.get("stage", ""))
    if not source_id:
        raise ValueError("source SFT row has no example_id")
    common = {
        "schema": EXAMPLE_SCHEMA,
        "identity": IDENTITY,
        "source_example_id": source_id,
        "source_stage": stage,
        "material_id": str(source.get("material_id", "")),
        "training_seed": int(source.get("training_seed", -1)),
        "representation": "wyckoff_formula_plan",
        "source_topology_hash": source.get("source_topology_hash"),
    }
    if stage == "direct_edit":
        if not replay_direct_edit:
            return ()
        return (
            {
                **common,
                "example_id": _derived_id(source_id, "direct_edit_replay"),
                "stage": "direct_edit_replay",
                "system_prompt": str(source["system_prompt"]),
                "user_prompt": str(source["user_prompt"]),
                "answer": str(source["answer"]),
                "answer_sha256": str(source["answer_sha256"]),
            },
        )
    if stage != "coarse_proposal":
        raise ValueError(f"unsupported source SFT stage: {stage}")

    state = parse_wq_proposal(str(source["answer"]), catalog)
    plan = formula_plan_from_state(state)
    if require_composition_valid and not plan.composition_valid:
        return ()
    plan_text = serialize_formula_plan(plan)
    plan_record = {
        **common,
        "formula_plan": plan.as_dict(include_composition_valid=False),
        "primitive_atom_count": int(state.atom_count),
        "orbit_count": len(state.orbits),
    }
    planner_answer_sha = hashlib.sha256(plan_text.encode("utf-8")).hexdigest()
    return (
        {
            **plan_record,
            "example_id": _derived_id(source_id, "formula_plan"),
            "stage": "formula_plan",
            "system_prompt": FORMULA_PLAN_SYSTEM_PROMPT,
            "user_prompt": FORMULA_PLAN_USER_PROMPT,
            "answer": plan_text,
            "answer_sha256": planner_answer_sha,
        },
        {
            **plan_record,
            "example_id": _derived_id(source_id, "formula_conditioned_body"),
            "stage": "formula_conditioned_body",
            "system_prompt": FORMULA_BODY_SYSTEM_PROMPT,
            "user_prompt": formula_body_user_prompt(plan),
            "answer": str(source["answer"]),
            "answer_sha256": str(source["answer_sha256"]),
        },
    )


def _iter_source(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def build_dataset(
    *,
    source_path: Path,
    output_dir: Path,
    expected_source_sha256: str,
    tokenizer: Any,
    max_length: int,
    expected_source_examples: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if sha256_file(source_path) != expected_source_sha256:
        raise ValueError("frozen mixed-edit SFT source changed")
    output_dir.mkdir(parents=True)
    data_path = output_dir / "formula_plan_sft.jsonl"
    manifest_path = output_dir / "manifest.json"
    token_audit_path = output_dir / "token_audit.json"
    catalog = PyXtalChartCatalog()

    source_counts = {"coarse_proposal": 0, "direct_edit": 0}
    stage_counts = {
        "formula_plan": 0,
        "formula_conditioned_body": 0,
        "direct_edit_replay": 0,
    }
    source_examples = 0
    valid_material_ids: set[str] = set()
    valid_direct_edit_index = 0
    output_digest = hashlib.sha256()
    with data_path.open("xb") as output:
        for source in _iter_source(source_path):
            source_examples += 1
            stage = str(source.get("stage", ""))
            if stage not in source_counts:
                raise ValueError(f"unexpected frozen source stage: {stage}")
            source_counts[stage] += 1
            material_id = str(source.get("material_id", ""))
            replay = bool(
                stage == "direct_edit"
                and material_id in valid_material_ids
                and valid_direct_edit_index % 2 == 0
            )
            examples = formula_plan_training_examples(
                source,
                catalog=catalog,
                replay_direct_edit=replay,
                require_composition_valid=True,
            )
            if stage == "coarse_proposal" and examples:
                valid_material_ids.add(material_id)
            if stage == "direct_edit" and material_id in valid_material_ids:
                valid_direct_edit_index += 1
            for example in examples:
                encoded = (_canonical_json(example) + "\n").encode("utf-8")
                output.write(encoded)
                output_digest.update(encoded)
                stage_counts[str(example["stage"])] += 1
        output.flush()
        os.fsync(output.fileno())

    if source_examples != expected_source_examples:
        raise ValueError("frozen source SFT denominator changed")
    if (
        source_counts["coarse_proposal"] != source_counts["direct_edit"]
        or stage_counts["formula_plan"] != stage_counts["formula_conditioned_body"]
        or not 20_000
        <= stage_counts["formula_plan"]
        <= source_counts["coarse_proposal"]
        or stage_counts["direct_edit_replay"]
        != (stage_counts["formula_plan"] + 1) // 2
    ):
        raise ValueError("formula-plan SFT mixture denominator changed")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "identity": IDENTITY,
        "source_path": str(source_path.resolve()),
        "source_sha256": expected_source_sha256,
        "source_examples": source_examples,
        "source_stage_counts": source_counts,
        "source_composition_valid_structures": stage_counts["formula_plan"],
        "source_composition_invalid_structures_excluded": (
            source_counts["coarse_proposal"] - stage_counts["formula_plan"]
        ),
        "examples": sum(stage_counts.values()),
        "stage_counts": stage_counts,
        "target_mixture_ratio": {
            "formula_plan": 2,
            "formula_conditioned_body": 2,
            "direct_edit_replay": 1,
        },
        "chemistry_filter": "training_only_same_reduced_SMACT_Pauling_validity",
        "heldout_metrics_or_generations_used": False,
        "jsonl_sha256": output_digest.hexdigest(),
        "jsonl_bytes": data_path.stat().st_size,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    lengths: list[int] = []
    token_stage_counts = {key: 0 for key in stage_counts}
    for example in _iter_source(data_path):
        tokenized = tokenize_sft_example(
            tokenizer,
            example,
            max_length=max_length,
        )
        lengths.append(len(tokenized["input_ids"]))
        token_stage_counts[str(example["stage"])] += 1
    if len(lengths) != manifest["examples"] or token_stage_counts != stage_counts:
        raise ValueError("formula-plan token audit denominator changed")
    token_audit = {
        "schema": TOKEN_AUDIT_SCHEMA,
        "identity": IDENTITY,
        "ok": True,
        "data_sha256": manifest["jsonl_sha256"],
        "examples_tokenized": len(lengths),
        "stage_counts": token_stage_counts,
        "max_length": int(max_length),
        "maximum_observed_tokens": max(lengths),
        "minimum_observed_tokens": min(lengths),
        "mean_observed_tokens": sum(lengths) / len(lengths),
        "overflow_count": 0,
        "fixed_padded_optimizer_tokens": len(lengths) * int(max_length),
    }
    with token_audit_path.open("x", encoding="utf-8") as handle:
        json.dump(token_audit, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "token_audit_path": str(token_audit_path),
        "token_audit_sha256": sha256_file(token_audit_path),
        **manifest,
        "token_audit": token_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-examples", type=int, default=54_270)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=640)
    args = parser.parse_args()
    for name in ("source_data", "output_dir", "llama_root"):
        setattr(args, name, getattr(args, name).resolve())

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.llama_root,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
        model_max_length=args.max_length,
        padding_side="right",
    )
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise RuntimeError("registered tokenizer lacks EOS or chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    report = build_dataset(
        source_path=args.source_data,
        output_dir=args.output_dir,
        expected_source_sha256=args.source_sha256,
        tokenizer=tokenizer,
        max_length=args.max_length,
        expected_source_examples=args.source_examples,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
