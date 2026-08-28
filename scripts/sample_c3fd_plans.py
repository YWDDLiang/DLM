#!/usr/bin/env python3
"""Sample one trajectory per request from the full C³FD-v2 Planner."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from crystal_dlm.c3fd_planner_model import C3FDPlannerConfig, C3FDPlannerModel  # noqa: E402
from crystal_dlm.c3fd_calibration import (  # noqa: E402
    StratumInteraction,
    calibrated_top_p_probabilities,
)
from crystal_dlm.ccfd import FormulaToken  # noqa: E402
from crystal_dlm.ccfd_v2 import (  # noqa: E402
    BenchmarkReachability,
    CCFDv2State,
    SetAtomCount,
    render_rich_plan,
    state_to_plan_state,
)
from crystal_dlm.composition_pair_prior import ValenceNode  # noqa: E402
from crystal_dlm.family_reachability import (  # noqa: E402
    FamilyAwareBenchmarkReachability,
    PaulingWitnessReachability,
    element_allowed_for_family,
    family_prefix_reachable,
    state_symbols,
)
from crystal_dlm.fixed_slot import Z_TO_SYMBOL  # noqa: E402
from crystal_dlm.r5_plan_state import anion_framework_from_symbols  # noqa: E402
from crystal_dlm.semantic_composition_head import SemanticHeadFlags  # noqa: E402


LATTICE_TO_SPACEGROUP = {
    "triclinic": "sg_001_002",
    "monoclinic": "sg_003_015",
    "orthorhombic": "sg_016_074",
    "tetragonal": "sg_075_142",
    "trigonal": "sg_143_167",
    "hexagonal": "sg_168_194",
    "cubic": "sg_195_230",
}

def sample_index(
    logits: torch.Tensor,
    *,
    rng: torch.Generator,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    values = logits.detach().float().cpu().clone()
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        raise ValueError("no finite sampling action")
    values = values / max(float(temperature), 1e-6)
    if int(top_k) > 0 and int(top_k) < int(finite.sum()):
        threshold = torch.topk(values, int(top_k)).values[-1]
        values[values < threshold] = float("-inf")
    probabilities = torch.softmax(values, dim=-1)
    if 0.0 < float(top_p) < 1.0:
        sorted_prob, sorted_idx = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_prob, dim=-1)
        remove = cumulative - sorted_prob > float(top_p)
        sorted_prob[remove] = 0.0
        probabilities.zero_().scatter_(0, sorted_idx, sorted_prob)
        probabilities /= probabilities.sum().clamp_min(1e-12)
    return int(torch.multinomial(probabilities, 1, generator=rng).item())


def semantic_inputs(
    target_n: int,
    species_ids: Sequence[int],
    counts: Sequence[int],
    *,
    state_history: Sequence[CCFDv2State],
    target_arity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if len(species_ids) != len(counts):
        raise ValueError("semantic history mismatch")
    target_position = len(species_ids) + 1
    width = target_position + 1
    previous_species = torch.full((1, width), -1, dtype=torch.long)
    previous_count = torch.zeros((1, width), dtype=torch.long)
    previous_n = torch.zeros((1, width), dtype=torch.long)
    ledger_features = torch.zeros((1, width, 6), dtype=torch.float32)
    previous_n[0, 1] = int(target_n)
    for index, (species_id, count) in enumerate(zip(species_ids, counts)):
        position = index + 2
        previous_species[0, position] = int(species_id)
        previous_count[0, position] = int(count)
    if len(state_history) != len(species_ids) + 1:
        raise ValueError("state history must contain post-N and post-species states")
    for index, state in enumerate(state_history, start=1):
        branch_vector = {
            None: (1.0, 0.0, 0.0),
            "ionic": (0.0, 1.0, 0.0),
            "alloy": (0.0, 0.0, 1.0),
        }.get(state.branch)
        if branch_vector is None:
            raise ValueError(f"unknown state branch {state.branch!r}")
        ledger_features[0, index] = torch.tensor(
            [
                float(state.remaining_atoms or 0) / 20.0,
                float(state.net_charge) / 160.0,
                float(int(target_arity) - len(state.tokens)) / 7.0,
                *branch_vector,
            ],
            dtype=torch.float32,
        )
    return previous_species, previous_count, previous_n, ledger_features, target_position


def charge_bucket(certificate: Mapping[str, Any]) -> str:
    reason = str(certificate.get("benchmark_reason") or "")
    if reason == "single_element_shortcut":
        return "single_element"
    if reason == "all_metal_shortcut":
        return "all_metal"
    return "neutral_plausible"


def sample_soft_value(
    logits: torch.Tensor,
    values: Sequence[str],
    *,
    rng: torch.Generator,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    masked = logits.detach().clone()
    for index, value in enumerate(values):
        if str(value) == "<UNKNOWN>":
            masked[index] = float("-inf")
    return str(values[sample_index(masked, rng=rng, temperature=temperature, top_p=top_p, top_k=top_k)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--pair-prior-weight", type=float, default=0.0)
    parser.add_argument("--max-species", type=int, default=7)
    parser.add_argument(
        "--reachability-mode",
        choices=("family_exact", "pauling_witness"),
        default="family_exact",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if int(args.top_k) != 0:
        raise ValueError("C3FD-v2.1 freezes species top_k=0")
    if float(args.pair_prior_weight) != 0.0:
        raise ValueError("C3FD-v2.1 freezes global pair-prior weight=0")
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary_bytes = (args.data_dir / "vocabulary.json").read_bytes()
    vocabulary = json.loads(vocabulary_bytes)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    expected_hash = hashlib.sha256(vocabulary_bytes).hexdigest()
    if checkpoint.get("vocabulary_sha256") != expected_hash:
        raise RuntimeError("C3FD checkpoint/vocabulary mismatch")
    calibration = checkpoint.get("calibration") or {}
    if set(calibration) != {"family", "n", "arity", "species", "count"}:
        raise RuntimeError("C3FD-v2.1 checkpoint lacks complete head calibration")
    interaction = StratumInteraction.from_dict(checkpoint["stratum_interaction"])
    sampling_contract = checkpoint.get("sampling_contract") or {}
    if (
        int(sampling_contract.get("species_top_k", -1)) != 0
        or float(sampling_contract.get("pair_prior_weight", -1.0)) != 0.0
    ):
        raise RuntimeError("C3FD-v2.1 checkpoint sampling contract changed")
    config = C3FDPlannerConfig(**checkpoint["config"])
    physics = torch.tensor(vocabulary["physics"]["matrix"], dtype=torch.float32)
    model = C3FDPlannerModel(config, physics_features=physics)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    context = torch.as_tensor(checkpoint["context"], dtype=torch.float32, device=device)
    species_rows = sorted(vocabulary["species"], key=lambda row: int(row["id"]))
    nodes = [
        ValenceNode(int(row["atomic_number"]), int(row["oxidation_state"]))
        for row in species_rows
    ]
    node_to_id = {node: index for index, node in enumerate(nodes)}
    if args.reachability_mode == "family_exact":
        reachability = FamilyAwareBenchmarkReachability(nodes)
    else:
        reachability = PaulingWitnessReachability(nodes)
    soft_values = vocabulary["soft_vocabulary"]
    eos_id = int(vocabulary["species_eos_id"])

    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / "raw_generations.jsonl"
    plan_path = args.output_dir / "plans_for_dlm.jsonl"
    parsed = 0
    failures: Counter[str] = Counter()
    n_counts: Counter[int] = Counter()
    arity_counts: Counter[int] = Counter()
    family_counts: Counter[str] = Counter()
    certificate_counts: Counter[str] = Counter()
    with raw_path.open("w", encoding="utf-8") as raw_handle, plan_path.open(
        "w", encoding="utf-8"
    ) as plan_handle:
        for local_idx in range(int(args.num_samples)):
            sample_idx = int(args.start_index) + local_idx
            rng = torch.Generator(device="cpu")
            rng.manual_seed(int(args.seed) * 1_000_003 + sample_idx)
            record: dict[str, Any] = {
                "sample_idx": sample_idx,
                "parsed": False,
                "plan_text": None,
                "plan_state": None,
                "failure": None,
                "semantic_trace": [],
                "target_proposal": None,
            }
            try:
                sentinel_species = torch.tensor([[-1]], device=device)
                sentinel_count = torch.tensor([[0]], device=device)
                sentinel_n = torch.tensor([[0]], device=device)
                sentinel_ledger = torch.zeros((1, 1, 6), device=device)
                with torch.inference_mode():
                    first = model(
                        context,
                        previous_species_indices=sentinel_species,
                        previous_count_values=sentinel_count,
                        previous_n_values=sentinel_n,
                        ledger_features=sentinel_ledger,
                        flags=SemanticHeadFlags(use_physics=True),
                    )
                proposal_scores = interaction.joint_scores(
                    first.family_logits[0],
                    first.n_logits[0],
                    first.arity_logits[0],
                    family_temperature=float(calibration["family"]["temperature"]),
                    n_temperature=float(calibration["n"]["temperature"]),
                    arity_temperature=float(calibration["arity"]["temperature"]),
                )
                proposal_probabilities = calibrated_top_p_probabilities(
                    proposal_scores,
                    temperature=1.0,
                    top_p=float(args.top_p),
                ).cpu()
                proposal_index = int(
                    torch.multinomial(
                        proposal_probabilities, 1, generator=rng
                    ).item()
                )
                target_family_id, target_n, target_arity = interaction.strata[
                    proposal_index
                ]
                target_family = str(
                    soft_values["anion_framework"][int(target_family_id)]
                )
                if target_family == "<UNKNOWN>":
                    raise ValueError("unsupported_unknown_family")
                state = CCFDv2State.start().apply(SetAtomCount(target_n))
                state_history = [state]
                sampled_species: list[int] = []
                sampled_counts: list[int] = []
                semantic_trace: list[dict[str, int | str]] = [
                    {
                        "action": "proposal",
                        "family": target_family,
                        "N": int(target_n),
                        "arity": int(target_arity),
                    }
                ]
                record["target_proposal"] = dict(semantic_trace[0])
                record["semantic_trace"] = semantic_trace
                last_output = None
                last_position = None
                while True:
                    (
                        previous_species,
                        previous_count,
                        previous_n,
                        ledger_features,
                        target_position,
                    ) = semantic_inputs(
                        target_n,
                        sampled_species,
                        sampled_counts,
                        state_history=state_history,
                        target_arity=target_arity,
                    )
                    with torch.inference_mode():
                        output = model(
                            context,
                            previous_species_indices=previous_species.to(device),
                            previous_count_values=previous_count.to(device),
                            previous_n_values=previous_n.to(device),
                            ledger_features=ledger_features.to(device),
                            flags=SemanticHeadFlags(use_physics=True),
                        )
                    species_logits = output.species_logits[0, target_position]
                    count_logits = output.count_logits[0, target_position]
                    legal_mask = torch.zeros(
                        model.head.num_joint_actions, dtype=torch.bool, device=device
                    )
                    if state.eos_legal and len(state.tokens) == int(target_arity):
                        certificate = state.end().certificate()
                        if (
                            certificate.benchmark_compatible
                            and anion_framework_from_symbols(state_symbols(state))
                            == target_family
                        ):
                            legal_mask[model.head.eos_action_index] = True
                    else:
                        legal_tokens = reachability.legal_species_counts(
                            state,
                            family=target_family,
                            max_species=int(args.max_species),
                            target_arity=int(target_arity),
                        )
                        for token in legal_tokens:
                            node = ValenceNode(token.atomic_number, token.oxidation_state)
                            species_id = node_to_id.get(node)
                            if species_id is None:
                                continue
                            action_index = int(species_id) * model.head.max_count + (
                                int(token.count) - 1
                            )
                            legal_mask[action_index] = True
                    if not bool(legal_mask.any().item()):
                        raise ValueError("semantic_dead_end")
                    joint = model.head.joint_action_scores(
                        (
                            species_logits
                            / float(calibration["species"]["temperature"])
                        ).unsqueeze(0),
                        (
                            count_logits
                            / float(calibration["count"]["temperature"])
                        ).unsqueeze(0),
                        legal_action_mask=legal_mask,
                        flags=SemanticHeadFlags(
                            use_pair_prior=False,
                            use_hard_mask=True,
                        ),
                    )[0]
                    action_probabilities = calibrated_top_p_probabilities(
                        joint,
                        temperature=1.0,
                        top_p=float(args.top_p),
                    )
                    action = int(
                        torch.multinomial(
                            action_probabilities.cpu(), 1, generator=rng
                        ).item()
                    )
                    last_output = output
                    last_position = target_position
                    if action == model.head.eos_action_index:
                        state = state.end()
                        semantic_trace.append({"action": "EOS"})
                        break
                    species_id = action // model.head.max_count
                    count = action % model.head.max_count + 1
                    node = nodes[species_id]
                    token = FormulaToken(node.atomic_number, node.oxidation_state, count)
                    state = state.apply(token, max_species=int(args.max_species))
                    state_history.append(state)
                    sampled_species.append(species_id)
                    sampled_counts.append(count)
                    semantic_trace.append(
                        {
                            "action": "species",
                            "atomic_number": int(node.atomic_number),
                            "oxidation_state": int(node.oxidation_state),
                            "count": int(count),
                        }
                    )
                    record["semantic_trace"] = semantic_trace
                certificate = state.certificate()
                if not certificate.benchmark_compatible:
                    raise ValueError("terminal_certificate_not_benchmark")
                if len(state.tokens) != int(target_arity):
                    raise ValueError("terminal_arity_mismatch")
                if (
                    anion_framework_from_symbols(state_symbols(state))
                    != target_family
                ):
                    raise ValueError("terminal_family_mismatch")
                if last_output is None or last_position is None:
                    raise RuntimeError("missing terminal model output")
                lattice = sample_soft_value(
                    last_output.rich_logits["lattice_system"][0, last_position],
                    soft_values["lattice_system"],
                    rng=rng,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                )
                volume = sample_soft_value(
                    last_output.rich_logits["volume_per_atom_bin"][0, last_position],
                    soft_values["volume_per_atom_bin"],
                    rng=rng,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                )
                soft = {
                    "anion_framework": target_family,
                    "charge_bucket": charge_bucket(certificate.to_dict()),
                    "lattice_system": lattice,
                    "spacegroup_bucket": LATTICE_TO_SPACEGROUP[lattice],
                    "volume_per_atom_bin": volume,
                }
                plan_state = state_to_plan_state(state, soft_fields=soft)
                plan_text = render_rich_plan(state, soft_fields=soft)
                record.update(
                    {
                        "parsed": True,
                        "plan_text": plan_text,
                        "plan_state": plan_state,
                        "semantic_trace": semantic_trace,
                        "certificate": certificate.to_dict(),
                    }
                )
                parsed += 1
                n_counts[target_n] += 1
                arity_counts[int(target_arity)] += 1
                family_counts[target_family] += 1
                certificate_counts[certificate.certificate_class] += 1
                plan_handle.write(
                    json.dumps(
                        {
                            "sample_idx": sample_idx,
                            "plan_text": plan_text,
                            "plan_state": plan_state,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            except Exception as exc:  # noqa: BLE001 - one request remains one failed request.
                reason = f"{type(exc).__name__}:{str(exc)}"
                record["failure"] = reason
                failures[reason] += 1
            raw_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = {
        "schema": "h1a2_c3fd_sampling_metrics_v1",
        "requested_samples": int(args.num_samples),
        "start_index": int(args.start_index),
        "end_index_exclusive": int(args.start_index) + int(args.num_samples),
        "parsed_samples": parsed,
        "all_request_benchmark_comp_valid": parsed,
        "parse_rate": parsed / int(args.num_samples),
        "formula_bpe": False,
        "repair": False,
        "replacement": False,
        "rerank": False,
        "rl": False,
        "pair_prior_weight": float(args.pair_prior_weight),
        "species_top_k": int(args.top_k),
        "calibration": calibration,
        "seed": int(args.seed),
        "N": {str(key): value for key, value in sorted(n_counts.items())},
        "arity": {str(key): value for key, value in sorted(arity_counts.items())},
        "family": dict(sorted(family_counts.items())),
        "certificate_classes": dict(sorted(certificate_counts.items())),
        "failures": dict(failures.most_common()),
        "joint_reachability": dict(reachability.stats()),
        "reachability_mode": str(args.reachability_mode),
    }
    (args.output_dir / "sample_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
