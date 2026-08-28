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
from crystal_dlm.ccfd import FormulaToken  # noqa: E402
from crystal_dlm.ccfd_v2 import (  # noqa: E402
    BenchmarkReachability,
    CCFDv2State,
    SetAtomCount,
    render_rich_plan,
    state_to_plan_state,
)
from crystal_dlm.composition_pair_prior import (  # noqa: E402
    CompositionPairPrior,
    ValenceNode,
)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if len(species_ids) != len(counts):
        raise ValueError("semantic history mismatch")
    target_position = len(species_ids) + 1
    width = target_position + 1
    previous_species = torch.full((1, width), -1, dtype=torch.long)
    previous_count = torch.zeros((1, width), dtype=torch.long)
    previous_n = torch.zeros((1, width), dtype=torch.long)
    previous_n[0, 1] = int(target_n)
    for index, (species_id, count) in enumerate(zip(species_ids, counts)):
        position = index + 2
        previous_species[0, position] = int(species_id)
        previous_count[0, position] = int(count)
    return previous_species, previous_count, previous_n, target_position


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
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--pair-prior-weight", type=float, default=0.25)
    parser.add_argument("--max-species", type=int, default=7)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary_bytes = (args.data_dir / "vocabulary.json").read_bytes()
    vocabulary = json.loads(vocabulary_bytes)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    expected_hash = hashlib.sha256(vocabulary_bytes).hexdigest()
    if checkpoint.get("vocabulary_sha256") != expected_hash:
        raise RuntimeError("C3FD checkpoint/vocabulary mismatch")
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
    reachability = BenchmarkReachability(
        tuple((node.atomic_number, node.oxidation_state) for node in nodes)
    )
    pair_prior = CompositionPairPrior.from_dict(vocabulary["pair_prior"])
    soft_values = vocabulary["soft_vocabulary"]
    eos_id = int(vocabulary["species_eos_id"])
    legal_n = []
    for n_value in range(1, 21):
        state = CCFDv2State.start().apply(SetAtomCount(n_value))
        if reachability.legal_species_counts(state, max_species=int(args.max_species)):
            legal_n.append(n_value)
    if not legal_n:
        raise RuntimeError("no N value has a reachable benchmark composition")

    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / "raw_generations.jsonl"
    plan_path = args.output_dir / "plans_for_dlm.jsonl"
    parsed = 0
    failures: Counter[str] = Counter()
    n_counts: Counter[int] = Counter()
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
            }
            try:
                sentinel_species = torch.tensor([[-1]], device=device)
                sentinel_count = torch.tensor([[0]], device=device)
                sentinel_n = torch.tensor([[0]], device=device)
                with torch.inference_mode():
                    first = model(
                        context,
                        previous_species_indices=sentinel_species,
                        previous_count_values=sentinel_count,
                        previous_n_values=sentinel_n,
                        flags=SemanticHeadFlags(use_physics=True),
                    )
                n_logits = first.n_logits[0].clone()
                legal_n_mask = torch.zeros_like(n_logits, dtype=torch.bool)
                legal_n_mask[[value - 1 for value in legal_n]] = True
                n_logits[~legal_n_mask] = float("-inf")
                target_n = sample_index(
                    n_logits,
                    rng=rng,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                ) + 1
                state = CCFDv2State.start().apply(SetAtomCount(target_n))
                sampled_species: list[int] = []
                sampled_counts: list[int] = []
                semantic_trace: list[dict[str, int | str]] = [
                    {"action": "N", "value": target_n}
                ]
                last_output = None
                last_position = None
                while True:
                    previous_species, previous_count, previous_n, target_position = semantic_inputs(
                        target_n, sampled_species, sampled_counts
                    )
                    with torch.inference_mode():
                        output = model(
                            context,
                            previous_species_indices=previous_species.to(device),
                            previous_count_values=previous_count.to(device),
                            previous_n_values=previous_n.to(device),
                            flags=SemanticHeadFlags(use_physics=True),
                        )
                    species_logits = output.species_logits[0, target_position]
                    count_logits = output.count_logits[0, target_position]
                    legal_mask = torch.zeros(
                        model.head.num_joint_actions, dtype=torch.bool, device=device
                    )
                    if state.eos_legal:
                        certificate = state.end().certificate()
                        if certificate.benchmark_compatible:
                            legal_mask[model.head.eos_action_index] = True
                    else:
                        legal_tokens = reachability.legal_species_counts(
                            state, max_species=int(args.max_species)
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
                    prior_scores = torch.tensor(
                        [
                            float(args.pair_prior_weight)
                            * pair_prior.context_score(
                                node,
                                [nodes[index] for index in sampled_species],
                            )
                            for node in nodes
                        ],
                        dtype=species_logits.dtype,
                        device=device,
                    )
                    joint = model.head.joint_action_scores(
                        species_logits.unsqueeze(0),
                        count_logits.unsqueeze(0),
                        pair_prior_scores=prior_scores,
                        legal_action_mask=legal_mask,
                        flags=SemanticHeadFlags(
                            use_pair_prior=True,
                            use_hard_mask=True,
                        ),
                    )[0]
                    action = sample_index(
                        joint,
                        rng=rng,
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        top_k=int(args.top_k),
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
                certificate = state.certificate()
                if not certificate.benchmark_compatible:
                    raise ValueError("terminal_certificate_not_benchmark")
                if last_output is None or last_position is None:
                    raise RuntimeError("missing terminal model output")
                symbols = [
                    str(value)
                    for value in state_to_plan_state(
                        state,
                        soft_fields={
                            "anion_framework": "other",
                            "lattice_system": "triclinic",
                            "spacegroup_bucket": "sg_001_002",
                            "volume_per_atom_bin": "volpa_000_004",
                        },
                    )["elements"]
                ]
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
                    "anion_framework": anion_framework_from_symbols(symbols),
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
        "seed": int(args.seed),
        "N": {str(key): value for key, value in sorted(n_counts.items())},
        "certificate_classes": dict(sorted(certificate_counts.items())),
        "failures": dict(failures.most_common()),
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
