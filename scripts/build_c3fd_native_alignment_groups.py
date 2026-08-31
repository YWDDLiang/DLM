#!/usr/bin/env python3
"""Build policy-specific K-way safety groups from the fresh on-policy pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "c3fd_native_alignment_group_v1"
MANIFEST_SCHEMA = "c3fd_native_alignment_group_manifest_v1"
POLICIES = (82017, 82018)
SAFETY_ENERGY_FLOOR = 0.05


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def attempt_index(row: Mapping[str, Any]) -> int:
    return int(str(row["attempt_id"]).rsplit("-", 1)[1])


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_eval(eval_run: Path, policy: int) -> dict[str, dict[int, dict[str, Any]]]:
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for stage, arm in (("raw", f"raw_policy{policy}"), ("refined", f"policy{policy}")):
        root = eval_run / arm / "evaluation"
        labels = read_jsonl(root / "full_reconstructed/attempt_labels_preofficial.jsonl")
        direct = read_jsonl(root / "direct/attempt_metrics.jsonl")
        label_map = {int(row["ordinal"]): row for row in labels}
        direct_map = {attempt_index(row): row for row in direct}
        if set(label_map) != set(range(256)) or set(direct_map) != set(range(256)):
            raise ValueError(f"{stage} policy{policy} does not cover fixed256")
        output[f"{stage}_labels"] = label_map
        output[f"{stage}_direct"] = direct_map
    return output


def build_policy_groups(
    *,
    policy: int,
    plans: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    raw_attempts: Mapping[int, Mapping[str, Any]],
    evaluation: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan_map = {int(row["sample_idx"]): row for row in plans}
    if set(plan_map) != set(raw_attempts):
        raise ValueError("pool plans and raw attempts have different sample indices")
    output: list[dict[str, Any]] = []
    trainable = 0
    groups_without_anchor = 0
    raw_valid_candidates = 0
    raw_invalid_candidates = 0
    refined_energy_known = 0
    raw_energy_known = 0
    groups_with_validity_variation = 0
    for group in groups:
        sample_indices = [int(value) for value in group["sample_indices"]]
        if len(sample_indices) != 4 or int(group["K"]) != 4:
            raise ValueError("alignment group K changed")
        group_plans = [plan_map[index] for index in sample_indices]
        if len({str(row["prompt"]) for row in group_plans}) != 1:
            raise ValueError("alignment group prompt changed")
        candidates: list[dict[str, Any]] = []
        valid_with_energy: list[int] = []
        for local_index, sample_idx in enumerate(sample_indices):
            plan = plan_map[sample_idx]
            attempt = raw_attempts[sample_idx]
            raw_direct = evaluation["raw_direct"][sample_idx]
            raw_label = evaluation["raw_labels"][sample_idx]
            refined_label = evaluation["refined_labels"][sample_idx]
            answer = str(attempt.get("text") or "")
            if not answer:
                raise ValueError("every on-policy attempt must retain answer text")
            raw_energy = finite_or_none(raw_label.get("chgnet_energy_per_atom"))
            refined_energy = finite_or_none(refined_label.get("chgnet_energy_per_atom"))
            direct_valid = raw_direct.get("valid") is True
            eligible = bool(direct_valid and refined_energy is not None)
            if eligible:
                valid_with_energy.append(local_index)
            raw_valid_candidates += int(direct_valid)
            raw_invalid_candidates += int(not direct_valid)
            raw_energy_known += int(raw_energy is not None)
            refined_energy_known += int(refined_energy is not None)
            candidates.append(
                {
                    "candidate_index": local_index,
                    "sample_idx": sample_idx,
                    "answer": answer,
                    "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                    "raw_parsed": attempt.get("parsed") is True,
                    "raw_comp_valid": raw_direct.get("comp_valid") is True,
                    "raw_struct_valid": raw_direct.get("struct_valid") is True,
                    "raw_direct_joint_valid": direct_valid,
                    "raw_energy_per_atom": raw_energy,
                    "refined_energy_per_atom": refined_energy,
                    "refined_reconstructed": refined_label.get("reconstructed") is True,
                    "eligible_valid_energy": eligible,
                    "target_energy_per_atom": None,
                    "safety_rank": None,
                    "is_best_valid_anchor": False,
                }
            )
        valid_flags = {candidate["raw_direct_joint_valid"] for candidate in candidates}
        groups_with_validity_variation += int(len(valid_flags) > 1)
        best_valid: int | None = None
        penalty: float | None = None
        if valid_with_energy:
            valid_energies = [
                float(candidates[index]["refined_energy_per_atom"])
                for index in valid_with_energy
            ]
            best_valid = min(
                valid_with_energy,
                key=lambda index: (
                    float(candidates[index]["refined_energy_per_atom"]),
                    index,
                ),
            )
            spread = max(valid_energies) - min(valid_energies)
            penalty = max(float(spread), SAFETY_ENERGY_FLOOR)
            worst_valid = max(valid_energies)
            invalid_target = worst_valid + penalty
            for index, candidate in enumerate(candidates):
                candidate["target_energy_per_atom"] = (
                    float(candidate["refined_energy_per_atom"])
                    if index in valid_with_energy
                    else invalid_target
                )
                candidate["is_best_valid_anchor"] = index == best_valid
            order = sorted(
                range(4),
                key=lambda index: (
                    float(candidates[index]["target_energy_per_atom"]),
                    index,
                ),
            )
            for rank, index in enumerate(order):
                candidates[index]["safety_rank"] = rank
            trainable += 1
        else:
            groups_without_anchor += 1
        output.append(
            {
                "schema": SCHEMA,
                "policy_seed": policy,
                "group_id": str(group["group_id"]),
                "group_ordinal": int(group["group_ordinal"]),
                "composition_ordinal": int(group["composition_ordinal"]),
                "composition_id": str(group["reduced_composition_identity"]),
                "prediction_checkpoint": str(group["prediction_checkpoint"]),
                "prompt": str(group_plans[0]["prompt"]),
                "plan_state": dict(group_plans[0]["plan_state"]),
                "K": 4,
                "group_weight": 1.0,
                "trainable": best_valid is not None,
                "best_valid_candidate_index": best_valid,
                "raw_valid_count": sum(
                    candidate["raw_direct_joint_valid"] for candidate in candidates
                ),
                "valid_energy_count": len(valid_with_energy),
                "invalid_energy_penalty": penalty,
                "candidates": candidates,
            }
        )
    audit = {
        "groups": len(output),
        "candidates": sum(len(group["candidates"]) for group in output),
        "trainable_groups": trainable,
        "groups_without_valid_anchor": groups_without_anchor,
        "groups_with_validity_variation": groups_with_validity_variation,
        "raw_valid_candidates": raw_valid_candidates,
        "raw_invalid_candidates": raw_invalid_candidates,
        "raw_energy_known": raw_energy_known,
        "refined_energy_known": refined_energy_known,
    }
    return output, audit


def build(*, cohort: Path, generation_run: Path, eval_run: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    plans_path = cohort / "pool_plans.jsonl"
    groups_path = cohort / "groups.jsonl"
    plans = read_jsonl(plans_path)
    groups = read_jsonl(groups_path)
    if len(plans) != 256 or len(groups) != 64:
        raise ValueError("frozen alignment cohort shape changed")
    results: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Any] = {}
    for policy in POLICIES:
        attempts_path = generation_run / f"policy{policy}/body/raw_generations.jsonl"
        attempts = {int(row["sample_idx"]): row for row in read_jsonl(attempts_path)}
        evaluation = load_eval(eval_run, policy)
        results[str(policy)], audit[str(policy)] = build_policy_groups(
            policy=policy,
            plans=plans,
            groups=groups,
            raw_attempts=attempts,
            evaluation=evaluation,
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    output_hashes: dict[str, str] = {}
    for policy in POLICIES:
        path = output_dir / f"policy{policy}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in results[str(policy)]:
                handle.write(canonical_json(row) + "\n")
        output_hashes[path.name] = sha256_file(path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source": "fresh_MP20_train_on_policy_pool_only",
        "policies": list(POLICIES),
        "groups_per_policy": 64,
        "candidates_per_group": 4,
        "group_weight": 1.0,
        "raw_invalid_is_lexicographically_worst": True,
        "valid_order": "lower_same_group_refined_CHGNet_energy_is_better",
        "best_valid_anchor": "minimum_refined_energy_among_raw_Direct_valid",
        "groups_without_valid_anchor": "preserved_but_fail_closed_not_trainable",
        "cross_composition_energy_comparison": False,
        "historical_3614_used": False,
        "prospective_outcomes_read": False,
        "safety_energy_floor": SAFETY_ENERGY_FLOOR,
        "input_sha256": {
            "plans": sha256_file(plans_path),
            "groups": sha256_file(groups_path),
            "generation_final": sha256_file(
                generation_run / "C3FD_NATIVE_ALIGNMENT_POOL_GENERATION_FINAL.json"
            ),
            "eval_outputs": sha256_file(eval_run / "OUTPUTS.sha256"),
        },
        "audit": audit,
        "output_sha256": output_hashes,
        "gates": {
            "all_groups_preserved": all(audit[str(policy)]["groups"] == 64 for policy in POLICIES),
            "all_candidates_preserved": all(audit[str(policy)]["candidates"] == 256 for policy in POLICIES),
            "fixed_k": all(all(group["K"] == 4 for group in results[str(policy)]) for policy in POLICIES),
            "group_weight_one": all(all(group["group_weight"] == 1.0 for group in results[str(policy)]) for policy in POLICIES),
            "old_3614_unused": True,
            "prospective_unread": True,
        },
    }
    if not all(manifest["gates"].values()):
        raise RuntimeError("alignment group builder gates failed")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    output_hashes["manifest.json"] = sha256_file(manifest_path)
    with (output_dir / "SHA256SUMS").open("x", encoding="utf-8", newline="\n") as handle:
        for name in sorted(output_hashes):
            handle.write(f"{output_hashes[name]}  {name}\n")
    (output_dir / "_SUCCESS").touch()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--generation-run", type=Path, required=True)
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        cohort=args.cohort.resolve(),
        generation_run=args.generation_run.resolve(),
        eval_run=args.eval_run.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
