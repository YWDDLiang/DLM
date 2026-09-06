"""CPU fixtures for H-P33 occurrence, numerical-failure and representation ledgers."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crystal_dlm.fixed_slot import build_special_tokens
from crystal_dlm.mixed_geometry_diffusion import DEFAULT_CONFIG, GeometrySamplingError, LatticeNormalizer
import crystal_dlm.mixed_geometry_sampling as sampling
from crystal_dlm.mixed_geometry_sampling import (
    METHOD, keyed_cpu_prior, paired_representations, sample_compiled_batch, validate_final_policy,
)
from crystal_dlm.programmed_path_data import compile_condition, path_seed
from crystal_dlm.sampling_layout import sampling_batches

SPEC = importlib.util.spec_from_file_location("mixed_geometry_sample_cli", ROOT / "src/scripts/sample_mixed_geometry_paths.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class Tokenizer:
    def __init__(self):
        self.vocab = {token: index+1 for index, token in enumerate(build_special_tokens())}
        self.vocab["<PAD>"] = 0
        self.mask_token_id = len(self.vocab)
        self.vocab["<MASK>"] = self.mask_token_id

    def get_vocab(self):
        return self.vocab

    def __call__(self, text, **_kwargs):
        return {"input_ids": [1]*len(text.split())}


def condition(index, *, count=1, words=2):
    return {"sample_idx": index, "group_id": f"eval:{index}", "source_split": "evaluation",
            "prompt": "frozen "*words, "plan_state": {"N": count, "elements": ["H"], "counts": [count]},
            "species_program": ["H"], "species_program_source": "frozen_pointer"}


def normalizer(length=4.):
    return LatticeNormalizer.fit(torch.eye(3, dtype=torch.float64)[None]*length, torch.tensor([1]))


class ZeroFieldModel(nn.Module):
    def __init__(self, tokenizer, *, norm=None, bad_row=None, failure_call=5, raise_batch=False):
        super().__init__()
        self.mask_id = tokenizer.mask_token_id
        self.state_config = SimpleNamespace(max_sites=20)
        self.diffusion_config = DEFAULT_CONFIG
        self.normalizer = norm or normalizer()
        self.bad_row, self.failure_call, self.raise_batch = bad_row, failure_call, raise_batch
        self.calls = []

    def forward(self, input_ids, *, attention_mask, mode, geometry_context, geometry_state, species):
        assert mode == "geometry"
        assert geometry_state.z.dtype == geometry_state.fractional.dtype == geometry_state.t.dtype == torch.float32
        assert input_ids.shape == attention_mask.shape == geometry_context.old_token_ids.shape
        assert torch.equal(geometry_state.atom_mask.sum(-1), geometry_context.num_sites)
        assert species.dtype == torch.long
        self.calls.append({"time": float(geometry_state.t[0]), "z": geometry_state.z.clone(),
                           "fractional": geometry_state.fractional.clone()})
        if self.raise_batch and len(self.calls) == self.failure_call:
            raise RuntimeError("unattributed fixture model error")
        v, u = torch.zeros_like(geometry_state.z), torch.zeros_like(geometry_state.fractional)
        if self.bad_row is not None and len(self.calls) >= self.failure_call:
            u[self.bad_row, 0, 0] = float("nan")
        return SimpleNamespace(v_prediction=v, u_prediction=u)


def compiled_batch(tokenizer, indices=(0, 2)):
    return [(index, 0, compile_condition(condition(index), tokenizer, mask_id=tokenizer.mask_token_id, purpose="evaluation")) for index in indices]


@pytest.fixture(autouse=True)
def cif_dependency():
    pytest.importorskip("pymatgen.core")


def test_source_prior_is_cpu_fp64_and_independent_of_packing_padding_and_64bit_json():
    seed = path_seed(20260905, "eval:0", 0, 0)
    assert seed > 2**53
    state, receipts = keyed_cpu_prior([seed, seed+1], [1, 3], max_sites=20, device="cpu")
    single, one = keyed_cpu_prior([seed], [1], max_sites=1, device="cpu")
    torch.testing.assert_close(state.z[0], single.z[0], atol=0, rtol=0)
    torch.testing.assert_close(state.fractional[0, :1], single.fractional[0], atol=0, rtol=0)
    assert receipts[0]["sha256"] == one[0]["sha256"]
    assert state.z.dtype == state.fractional.dtype == torch.float64
    assert state.z.device.type == "cpu" and not bool(state.fractional[0, 1:].any())
    assert json.loads(json.dumps(receipts))[0]["seed"] == seed


def test_frozen_logical_membership_is_identical_on_two_four_and_six_workers():
    tokenizer = Tokenizer()
    items = [(i, 0, compile_condition(condition(i, count=1+i%3, words=2+i%2), tokenizer,
                                     mask_id=tokenizer.mask_token_id, purpose="evaluation")) for i in range(30)]
    layouts = []
    for world in (2, 4, 6):
        layout = {}
        for rank in range(world):
            for batch in sampling_batches(items, batch_size=4, rank=rank, world_size=world, layout_world_size=2):
                members = tuple(item[0] for item in batch)
                for index in members:
                    assert index not in layout
                    layout[index] = members
        layouts.append(layout)
    assert layouts[0] == layouts[1] == layouts[2]
    assert len(layouts[0]) == len(items)


def test_real_33rd_readout_float_output_and_decoded_q_are_separate():
    from pymatgen.core import Structure
    tokenizer = Tokenizer()
    varying_volume = LatticeNormalizer.fit(torch.stack([torch.eye(3)*4., torch.eye(3)*8.]).double(),
                                           torch.tensor([1, 1]))
    model = ZeroFieldModel(tokenizer, norm=varying_volume).eval()
    batch = compiled_batch(tokenizer, (0,))
    native, quantized, statistics = sample_compiled_batch(model, tokenizer, batch, device="cpu", checkpoint_path="fixture")
    assert len(model.calls) == statistics["model_forward_calls"] == 33
    assert model.calls[-1]["time"] == pytest.approx(.002, abs=1e-9)
    assert model.calls[-2]["time"] > model.calls[-1]["time"]
    a, b = native[0], quantized[0]
    assert a["success"] and b["success"] and a["sampling_nfe"] == b["sampling_nfe"] == 33
    assert a["body"] is None and b["body"] is not None and b["additional_neural_nfe"] == 0
    continuous, discrete = Structure.from_dict(a["structure"]), Structure.from_dict(b["structure"])
    prior_z = torch.tensor(a["initial_geometry_prior"]["z"], dtype=torch.float64)
    readout_z = np.cos(np.pi*DEFAULT_CONFIG.epsilon/2)*prior_z
    np.testing.assert_allclose(continuous.lattice.matrix, model.normalizer.decode(readout_z, 1).numpy(),
                               atol=1e-12, rtol=1e-12)
    assert not np.allclose(continuous.lattice.matrix, model.normalizer.decode(prior_z, 1).numpy(),
                           atol=1e-9, rtol=1e-9)
    assert not np.allclose(continuous.frac_coords, discrete.frac_coords, atol=1e-8, rtol=0)
    assert np.max(np.abs(discrete.frac_coords*100-np.rint(discrete.frac_coords*100))) < 1e-12
    assert a["sampling_seed"] == b["sampling_seed"] == path_seed(20260905, "eval:0", 0, 0)
    assert a["initial_geometry_prior"] == b["initial_geometry_prior"]
    json.dumps([native, quantized, statistics], allow_nan=False)


def test_row_failure_is_quarantined_without_retry_or_changing_healthy_sample():
    tokenizer = Tokenizer()
    batch = compiled_batch(tokenizer)
    model = ZeroFieldModel(tokenizer, bad_row=0, failure_call=5).eval()
    native, quantized, statistics = sample_compiled_batch(model, tokenizer, batch, device="cpu", checkpoint_path="fixture")
    assert len(model.calls) == statistics["model_forward_calls"] == 33
    assert not native[0]["success"] and not quantized[0]["success"]
    assert native[0]["structure"] is None and native[0]["body"] is None
    assert native[0]["sampling_nfe"] == 5 and native[0]["sampling_failure"]["attempted_field_nfe"] == 5
    assert native[0]["sampling_failure"]["scope"] == "row"
    assert native[0]["failed_row_padding_evaluations"] == 28
    assert native[1]["success"] and quantized[1]["success"] and native[1]["sampling_nfe"] == 33
    assert statistics["model_row_evaluations"] == 66
    assert statistics["live_row_model_evaluations"] == 38 and statistics["failed_row_padding_evaluations"] == 28
    assert not bool(model.calls[5]["z"][0].any()) and not bool(model.calls[5]["fractional"][0].any())
    alone, _, _ = sample_compiled_batch(ZeroFieldModel(tokenizer).eval(), tokenizer, [batch[1]],
                                        device="cpu", checkpoint_path="fixture")
    assert native[1]["structure"] == alone[0]["structure"]
    assert native[1]["initial_geometry_prior"] == alone[0]["initial_geometry_prior"]


def test_unknown_model_exception_retains_every_batch_member_as_failed():
    tokenizer = Tokenizer()
    model = ZeroFieldModel(tokenizer, raise_batch=True, failure_call=2).eval()
    native, quantized, statistics = sample_compiled_batch(model, tokenizer, compiled_batch(tokenizer),
                                                          device="cpu", checkpoint_path="fixture")
    assert len(native) == len(quantized) == 2 and len(model.calls) == statistics["model_forward_calls"] == 2
    assert statistics["model_row_evaluations"] == 4
    for record in native+quantized:
        assert not record["success"] and record["body"] is None and record["structure"] is None
        assert record["sampling_nfe"] == 2 and record["sampling_failure"]["scope"] == "logical_batch"
        assert "unattributed fixture model error" in record["failure"]
    json.dumps([native, quantized, statistics], allow_nan=False)


def test_bad_geometry_input_is_attributed_before_neural_call_and_preserves_other_row():
    class RejectPositiveFirstZ:
        def __init__(self):
            self.base = normalizer()

        def decode(self, z, count):
            if bool((z[..., 0] > 0).any()):
                raise FloatingPointError("fixture per-row lattice failure")
            return self.base.decode(z, count)

    tokenizer = Tokenizer()
    model = ZeroFieldModel(tokenizer, norm=RejectPositiveFirstZ()).eval()
    native, quantized, statistics = sample_compiled_batch(model, tokenizer, compiled_batch(tokenizer),
                                                          device="cpu", checkpoint_path="fixture")
    failed, healthy = native
    assert not failed["success"] and not quantized[0]["success"] and healthy["success"]
    assert failed["sampling_failure"]["scope"] == "row" and failed["sampling_failure"]["stage"] == "geometry_input"
    assert failed["sampling_failure"]["attempted_field_nfe"] == 1
    assert failed["sampling_nfe"] == failed["sampling_failure"]["model_nfe_at_failure"] == 0
    assert failed["failed_row_padding_evaluations"] == 33 and healthy["sampling_nfe"] == 33
    assert len(model.calls) == statistics["model_forward_calls"] == 33
    assert statistics["live_row_model_evaluations"] == statistics["failed_row_padding_evaluations"] == 33
    json.dumps([native, quantized, statistics], allow_nan=False)


def test_unattributed_integrator_failure_retains_batch_without_neural_retry(monkeypatch):
    def failed_step(field, initial, *, config):
        field(initial)
        try:
            raise FloatingPointError("fixture integrator step failure")
        except FloatingPointError as error:
            raise GeometrySamplingError(str(error), nfe=1, time=1.) from error

    monkeypatch.setattr(sampling, "sample_geometry", failed_step)
    tokenizer = Tokenizer()
    model = ZeroFieldModel(tokenizer).eval()
    native, quantized, statistics = sample_compiled_batch(model, tokenizer, compiled_batch(tokenizer),
                                                          device="cpu", checkpoint_path="fixture")
    assert len(native) == len(quantized) == 2 and len(model.calls) == statistics["model_forward_calls"] == 1
    for record in native+quantized:
        assert not record["success"] and record["body"] is None and record["structure"] is None
        assert record["sampling_nfe"] == 1 and record["sampling_failure"]["scope"] == "logical_batch"
        assert record["sampling_failure"]["stage"] == "sampler_step"
        assert "fixture integrator step failure" in record["failure"]
    json.dumps([native, quantized, statistics], allow_nan=False)


def test_q_clipping_is_failure_with_preview_outside_primary_fields():
    tokenizer = Tokenizer()
    model = ZeroFieldModel(tokenizer, norm=normalizer(50.3)).eval()
    native, quantized, _ = sample_compiled_batch(model, tokenizer, compiled_batch(tokenizer, (0,)),
                                                device="cpu", checkpoint_path="fixture")
    assert native[0]["success"] and not quantized[0]["success"]
    q = quantized[0]
    assert q["continuous_native_success"] and not q["native_execution_success"]
    assert q["body"] is None and q["structure"] is None and q["native_structure"] is None
    assert q["quantization"]["length_clips"] > 0 and q["diagnostic_candidate"]["body"]


def test_angle_clipping_and_canonical_periodic_alias_are_explicit():
    from pymatgen.core import Lattice, Structure
    tokenizer = Tokenizer()
    clipped = Structure(Lattice.from_parameters(4.,4.,4.,90.,90.,.1), ["H"], [[.2,.3,.4]])
    failed = paired_representations(clipped, ["H"], tokenizer)
    assert not failed["success"] and failed["quantization"]["angle_clips"] > 0
    periodic = Structure(Lattice.cubic(4.), ["H"], [[.9999,.23456,.45678]])
    result = paired_representations(periodic, ["H"], tokenizer)
    assert result["success"] and "<X_000>" in result["body"] and "<X_100>" not in result["body"]
    assert result["quantization"]["periodic_alias_tokens_canonicalized"] == 1
    assert Structure.from_dict(result["structure"]).frac_coords[0, 0] == 0


def test_final_policy_gate_uses_completion_contract_not_a_hardcoded_step(tmp_path):
    train = tmp_path / "train"
    checkpoint = train / "checkpoints" / "step-7"
    checkpoint.mkdir(parents=True)
    checkpoint_report = {"method": METHOD, "eligible_policy": True, "completed_epochs": 2, "expected_epochs": 2, "completed_step": 7}
    (checkpoint/"CHECKPOINT_FINAL.json").write_text(json.dumps(checkpoint_report), encoding="utf-8")
    (train/"TRAIN_FINAL.json").write_text(json.dumps({"method": METHOD, "eligible_policy": True, "epochs": 2,
                                                      "updates": 7, "policy_path": str(checkpoint)}), encoding="utf-8")
    (train/"_SUCCESS").touch()
    assert validate_final_policy(checkpoint)["checkpoint"]["completed_step"] == 7
    checkpoint_report["eligible_policy"] = False
    (checkpoint/"CHECKPOINT_FINAL.json").write_text(json.dumps(checkpoint_report), encoding="utf-8")
    with pytest.raises(ValueError, match="final policy"):
        validate_final_policy(checkpoint)


def test_merge_retains_failed_q_and_exact_64bit_seeds(tmp_path):
    tokenizer = Tokenizer()
    conditions = [condition(0), condition(1)]
    source = tmp_path / "conditions.jsonl"
    source.write_text("".join(json.dumps(row)+"\n" for row in conditions), encoding="utf-8")
    output = tmp_path / "samples"
    output.mkdir()
    args = SimpleNamespace(conditions=source, condition_start=0, condition_stop=2, world_size=2,
                           seed=20260905, collection_round=0, checkpoint_path=Path("fixture"),
                           sampling_layout_world_size=2, batch_size=4, output_dir=output)
    for rank, length in ((0,4.),(1,50.3)):
        batch = compiled_batch(tokenizer, (rank,))
        native, q, statistics = sample_compiled_batch(ZeroFieldModel(tokenizer, norm=normalizer(length)).eval(), tokenizer, batch,
                                                     device="cpu", checkpoint_path="fixture")
        for name, rows in (("native",native),("quantized",q)):
            (output/f"{name}.records.rank{rank}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
        report = {"rank":rank,"conditions_sha256":CLI.file_sha256(source),"seed":args.seed,"collection_round":0,
                  "checkpoint":"fixture","sampling_layout_world_size":2,"logical_batch_cap":4,**statistics}
        (output/f"SAMPLING.rank{rank}.json").write_text(json.dumps(report), encoding="utf-8")
        (output/f"_SUCCESS.rank{rank}").touch()
    report = CLI.merge_outputs(args)
    assert report["requested"] == 2 and report["counts"]["native"]["successful"] == 2
    assert report["counts"]["quantized"]["successful"] == 1 and report["model_forward_calls"] == 66
    native = [json.loads(line) for line in (output/"native/paths.jsonl").read_text().splitlines()]
    q = [json.loads(line) for line in (output/"quantized/paths.jsonl").read_text().splitlines()]
    assert [row["sample_idx"] for row in q] == [0,1]
    assert native[0]["sampling_seed"] > 2**53 and native[0]["sampling_seed"] == q[0]["sampling_seed"]
    assert q[1]["structure"] is None and q[1]["body"] is None
    assert (output/"_SUCCESS").is_file()
