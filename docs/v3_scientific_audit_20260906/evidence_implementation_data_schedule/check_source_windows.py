"""Read-only CPU audit of source windows, validation seeds, and padded losses.

This does not load a neural network or use distributed/GPU execution. The exact
gradient accounting below is an oracle for a fixed parameter value, not a claim
that sequential Adam updates or different world sizes give identical weights.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import torch

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens
from crystal_dlm.manifold_corruption import lattice_matrix_from_parameters
from crystal_dlm.mixed_geometry_diffusion import LatticeNormalizer
from crystal_dlm.mixed_geometry_training_data import (
    MixedBatchSchedule, MixedGeometrySources, VALIDATION_TIME_BANDS,
    materialize_mixed_batch,
)
from crystal_dlm.periodic_v2_objective import PeriodicV2Objective
from scripts.train_mixed_geometry_dlm import make_sources, mixed_batch_loss
from test_state_programmed_runtime import TinyTokenizer, constraints

torch.set_num_threads(1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accounting(count, world):
    schedule = MixedBatchSchedule(count, world)
    real = Counter()
    padding = Counter()
    mode_by_micro = {}
    windows = {}
    for rank in range(world):
        for window, micro, mode, slots in schedule.rank_batches(rank=rank, seed=202609062, epoch=1):
            key = window, micro
            assert mode_by_micro.setdefault(key, mode) == mode
            for offset, (index, padded) in enumerate(slots):
                order = (micro // 2 * world + rank) * schedule.microbatch + offset
                windows.setdefault((window, mode), []).append((order, index, padded))
                if padded:
                    padding[mode] += 1
                else:
                    real[mode, index] += 1
    ordered = {}
    for (window, mode), values in windows.items():
        ordered[window, mode] = [(index, padded) for _, index, padded in sorted(values)]
    for window in range(schedule.updates_per_epoch):
        assert ordered[window, "token"] == ordered[window, "geometry"]
        assert len(ordered[window, "token"]) == 12
    assert set(real) == {(mode, index) for mode in ("token", "geometry") for index in range(count)}
    assert set(real.values()) == {1}
    assert padding["token"] == padding["geometry"] == schedule.padded_sources - count
    # Local mean x DDP rank mean x accumulation scaling. Exact rational checks
    # every real state coefficient, including sources in the short last window.
    per_window_coefficient = Fraction(schedule.padded_sources, count) / (
        schedule.accumulation * world * schedule.microbatch)
    epoch_mean_coefficient = per_window_coefficient / schedule.updates_per_epoch
    assert epoch_mean_coefficient == Fraction(1, 2 * count)
    rows = [{"window": window, "sources": ordered[window, "token"]}
            for window in range(schedule.updates_per_epoch)]
    return {
        "source_count": count, "world_size": world,
        "manifest": schedule.manifest(2),
        "real_coverage_min": min(real.values()), "real_coverage_max": max(real.values()),
        "padding_by_mode": dict(padding),
        "last_window_real_sources": sum(not padded for _, padded in ordered[schedule.updates_per_epoch - 1, "token"]),
        "per_real_state_window_coefficient": str(per_window_coefficient),
        "per_real_state_epoch_mean_coefficient": str(epoch_mean_coefficient),
        "source_window_sha256": hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest(),
    }


class Tokenizer(TinyTokenizer):
    def __init__(self):
        super().__init__()
        self.pad_token_id = self.vocab["<PAD>"]
        self.mask_token_id = self.mask_id

    def __call__(self, text, **kwargs):
        return {"input_ids": [self.vocab["<E_H>"]]}


def fixture(split, index):
    fractional = [[.003137 + .001 * index, .202314, .398721], [.507913, .604822, .796132]]
    lengths, angles = [4.0234, 5.0432, 6.0321], [85.213, 95.327, 105.413]
    tokens, _ = arrays_to_dynamic_tokens(lengths, angles, ["H", "H"], fractional)
    answer = " ".join(tokens)
    row = {"source_row_idx": index, "source_split": split, "prompt": "retained Plan",
           "answer": answer, "source_answer": answer,
           "plan_state": {"N": 2, "elements": ["H"], "counts": [2]},
           "species_program": ["H"], "species_program_source": "frozen_planner",
           "sample_weight": 1., "energy_above_hull": -999., "forced_mask_positions": [1]}
    identity = {"source_split": split, "source_row_idx": index, "identity_verified": True,
                "target_answer_sha256_bytes": hashlib.sha256(answer.encode()).hexdigest(),
                "continuous_aligned": {"lattice_matrix_A": lattice_matrix_from_parameters(lengths, angles).tolist(),
                                       "species": ["H", "H"], "fractional": fractional}}
    return row, identity


def same_noise(left, right):
    return all(torch.equal(getattr(left.state, name), getattr(right.state, name))
               for name in ("z", "fractional", "t", "atom_mask")) and all(
                   torch.equal(getattr(left, name), getattr(right, name)) for name in ("v_target", "u_target"))


def data_checks():
    tokenizer = Tokenizer()
    support = constraints(tokenizer)
    support.update(canonicalize_periodic_alias=True)
    normalizer = LatticeNormalizer((0.,) * 6, (1.,) * 6, (1.,) * 6, 1e-6, 1., 2)
    train_pairs = [fixture("train", i) for i in range(2)]
    val_pairs = [fixture("val", i) for i in range(2)]
    args = SimpleNamespace(train_data=[x[0] for x in train_pairs], train_identity=[x[1] for x in train_pairs],
                           val_data=[x[0] for x in val_pairs], val_identity=[x[1] for x in val_pairs],
                           data_seed=202609062, validation_seed=202609063,
                           expected_train_sources=2, expected_val_sources=2)
    train, val = make_sources(args, tokenizer, support, normalizer, max_sites=20)
    assert train.seed == 202609062 and val.seed == 202609063
    reordered = MixedGeometrySources(args.train_data[::-1], args.train_identity[::-1], tokenizer, support,
                                     normalizer, split="train", seed=args.data_seed)
    original = train.example(0, mode="geometry", epoch=1)["noisy_geometry"]
    assert same_noise(original, reordered.example(1, mode="geometry", epoch=1, is_padding=True)["noisy_geometry"])
    assert not same_noise(original, train.example(0, mode="geometry", epoch=0)["noisy_geometry"])
    val_before = val.example(0, mode="geometry", epoch=0, fixed_time=.05,
                             noise_stream="validation_band_0")["noisy_geometry"]
    for i in (1, 0, 1):
        train.example(i, mode="geometry", epoch=9)
    assert same_noise(val_before, val.example(0, mode="geometry", epoch=0, fixed_time=.05,
                                              noise_stream="validation_band_0")["noisy_geometry"])
    assert not same_noise(val_before, val.example(0, mode="geometry", epoch=0, fixed_time=.05,
                                                  noise_stream="validation_band_1")["noisy_geometry"])
    assert float(train.fractional[0, 0, 0]) == .003137
    g_example = train.example(0, mode="geometry", epoch=0)
    assert "energy_above_hull" not in g_example and "forced_mask_positions" not in g_example
    assert all(g_example["input_body"][p] == tokenizer.mask_id for p in train.sources[0].transaction_positions)
    t_example = train.example(0, mode="token", epoch=0)
    assert t_example["view"] == 0 and not t_example["old_admission_required"]
    zero_results = {}
    for mode in ("token", "geometry"):
        examples = [train.example(i, mode=mode, epoch=0, is_padding=True) for i in range(2)]
        batch = materialize_mixed_batch(examples, tokenizer, device="cpu")
        if mode == "geometry":
            v = torch.ones(2, 6, requires_grad=True)
            u = torch.ones(2, 20, 3, requires_grad=True)
            output = SimpleNamespace(v_prediction=v, u_prediction=u)
            parameters = [v, u]
            objective = None
        else:
            logits = torch.zeros(2, batch["input_ids"].shape[1], len(tokenizer.vocab), requires_grad=True)
            output = SimpleNamespace(logits=logits)
            parameters = [logits]
            objective = PeriodicV2Objective(tokenizer, support, "cpu")
        loss, _, _ = mixed_batch_loss(output, batch, objective)
        assert loss.requires_grad and float(loss.detach()) == 0.
        loss.backward()
        assert all(p.grad is not None and float(p.grad.abs().sum()) == 0. for p in parameters)
        zero_results[mode] = {"loss": float(loss.detach()), "connected_zero_gradient": True}
    return {
        "train_seed": train.seed, "validation_seed": val.seed,
        "validation_seed_regression_closed": True,
        "source_reorder_and_padding_preserve_noise": True,
        "epoch_changes_noise": True, "validation_noise_is_fixed_and_band_separated": True,
        "continuous_fractional_example": float(train.fractional[0, 0, 0]),
        "geometry_scaffold_has_no_clean_numeric_tokens": True,
        "outcomes_and_old_rollout_masks_not_copied": True,
        "token_view_is_construction": True,
        "wholly_padded_microbatches": zero_results,
        "monitor_times": [math.sqrt(a * b) for a, b in VALIDATION_TIME_BANDS],
        "monitor_measure": "equal weight at six fixed geometric band centers; not a LogUniform integral estimator",
    }


results = [accounting(count, world) for count in (1, 11, 12, 13, 27, 27136) for world in (4, 6)]
for i in range(0, len(results), 2):
    assert results[i]["source_window_sha256"] == results[i + 1]["source_window_sha256"]
report = {
    "status": "pass", "torch": torch.__version__, "model_forwards": 0, "GPU_calls": 0,
    "actual_DDP_execution": False,
    "scope": "exact schedule accounting, synthetic data constructor checks, tensor-only padding autograd",
    "source_sha256": {name: sha(ROOT / name) for name in (
        "src/crystal_dlm/mixed_geometry_training_data.py", "src/scripts/train_mixed_geometry_dlm.py",
        "src/crystal_dlm/mixed_geometry_diffusion.py", "src/crystal_dlm/periodic_v2_objective.py",
        "docs/v3_scientific_audit_20260906/V3_H_P33_SPECIFICATION.md")},
    "accounting": results, "data": data_checks(),
}
destination = Path(__file__).with_suffix(".json")
destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"status": report["status"], "schedule_cases": len(results), "data": report["data"],
                  "output": str(destination)}, allow_nan=False))
