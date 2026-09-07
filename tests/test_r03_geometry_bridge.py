"""CPU geometry/call contracts; no learned model or physical evaluator runs."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crystal_dlm.fixed_slot import MASK_TOKEN_ID, build_special_tokens
from crystal_dlm.r03_geometry_bridge import (
    ConstructionGeometryMonitor, GeometryBridgeContractError, GeometryNoLegalSupport,
    construction_geometry_bridge,
)
from crystal_dlm.r03_physics_transfer import build_repair_constraints


class TinyTokenizer:
    def __init__(self):
        self.vocab = {token: index + 2 for index, token in enumerate(build_special_tokens())}
    def get_vocab(self):
        return self.vocab


def groups(n):
    return [[0, *[7 + 4 * i for i in range(n)]], [1, 2, 3, 4, 5, 6],
            [8 + 4 * i for i in range(n)], [9 + 4 * i for i in range(n)], [10 + 4 * i for i in range(n)]]


def native_constraints(tokenizer):
    full = build_repair_constraints(tokenizer)
    original_keys = ("representation", "max_atoms", "coord_period", "duplicate_coordinate_mask",
                     "lattice_volume_mask", "min_lattice_rad", "count_token_to_n", "coord_token_to_bin",
                     "z_bin_to_token_id", "angle_token_to_bin", "gamma_bin_to_token_id",
                     "zero_length_token_ids_by_position")
    return {key: full[key] for key in original_keys}


def canvas(tokenizer, n=2, *, length=40, coords=None, angles=(90, 90, 90)):
    v = tokenizer.vocab
    tokens = [v[f"<N_{n:03d}>"], *[v[f"<{axis}_{length:03d}>"] for axis in ("LA", "LB", "LC")],
              *[v[f"<{axis}_{angle:03d}>"] for axis, angle in zip(("AA", "AB", "AG"), angles)]]
    coords = coords or [(0, 0, 0)] * n
    for xyz in coords:
        tokens.append(v["<E_Si>"])
        tokens.extend(MASK_TOKEN_ID if value is None else v[f"<{axis}_{value:03d}>"]
                      for axis, value in zip("XYZ", xyz))
    return torch.tensor([[1, 1, *tokens]], dtype=torch.long)


def lattice_logits(tokenizer, x, active_positions):
    width = max(tokenizer.vocab.values()) + 1
    minimum = torch.finfo(torch.float32).min
    logits = torch.full((1, x.shape[1], width), minimum)
    for position in active_positions:
        axis = "XYZ"[(position - 8) % 4]
        logits[0, 2 + position, [tokenizer.vocab[f"<{axis}_{i:03d}>"] for i in range(101)]] = 0
    return logits


def load_frozen_pair_fixture():
    noise_spec = importlib.util.spec_from_file_location("_geometry_noise_fixture", ROOT / "src/r03/paired_noise.py")
    noise = importlib.util.module_from_spec(noise_spec)
    noise_spec.loader.exec_module(noise)
    pair_spec = importlib.util.spec_from_file_location("_geometry_pair_fixture", ROOT / "src/r03/paired_llada.py")
    paired = importlib.util.module_from_spec(pair_spec)
    with patch.dict(sys.modules, {"paired_noise": noise}):
        pair_spec.loader.exec_module(paired)
    return paired


class GeometryMaskContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = TinyTokenizer()
        cls.native = native_constraints(cls.tokenizer)

    def monitor(self, n=2):
        return ConstructionGeometryMonitor(enabled=True, tokenizer=self.tokenizer,
            generation_position_groups=groups(n), native_constraints=self.native)

    def apply(self, monitor, logits, x, group):
        return monitor.apply_logits(logits, current_tokens=x, prompt_length=2, gen_length=x.shape[1] - 2,
                                    semantic_group=group, step_in_group=0)

    def test_alias_mass_is_merged_only_at_active_masked_coordinates(self):
        x = canvas(self.tokenizer, 1, coords=[(None, None, None)])
        logits = lattice_logits(self.tokenizer, x, [8])
        zero, alias = self.tokenizer.vocab["<X_000>"], self.tokenizer.vocab["<X_100>"]
        logits[0, 10, zero], logits[0, 10, alias] = torch.log(torch.tensor(2.)), torch.log(torch.tensor(3.))
        before, original_x = logits.clone(), x.clone()
        monitor = self.monitor(1)
        self.apply(monitor, logits, x, 2)
        self.assertAlmostEqual(float(logits[0, 10, zero]), float(torch.log(torch.tensor(5.))), places=6)
        self.assertEqual(float(logits[0, 10, alias]), torch.finfo(logits.dtype).min)
        self.assertTrue(torch.equal(x, original_x))
        self.assertTrue(torch.equal(logits[:, :10], before[:, :10]))
        self.assertTrue(torch.equal(logits[:, 11:], before[:, 11:]))
        self.assertEqual(monitor.report()["active_alias_vectors"], 1)

    def test_pbc_masks_close_distinct_positions_across_periodic_boundary(self):
        x = canvas(self.tokenizer, 2, coords=[(99, 0, 0), (1, 0, None)])
        logits = lattice_logits(self.tokenizer, x, [14])
        monitor = self.monitor(2)
        self.apply(monitor, logits, x, 4)
        for z in (0, 100):
            self.assertEqual(float(logits[0, 16, self.tokenizer.vocab[f"<Z_{z:03d}>"]]), torch.finfo(logits.dtype).min)
        self.assertGreater(float(logits[0, 16, self.tokenizer.vocab["<Z_050>"]]), torch.finfo(logits.dtype).min)
        self.assertGreater(monitor.report()["newly_masked_legal_tokens"], 1)
        self.assertEqual(monitor.report()["protocol"]["periodic_image_count"], 125)

    def test_pbc_empty_support_raises_before_native_argmax_and_preserves_partial_canvas(self):
        x = canvas(self.tokenizer, 3, length=10, coords=[(0, 0, 0), (0, 0, 50), (0, 0, None)])
        logits = lattice_logits(self.tokenizer, x, [18])
        called = []
        def original(*args, **kwargs):
            called.append(True)
            raise AssertionError("empty support must never reach candidate selection")
        paired = SimpleNamespace(_paired_suffix_candidates=original)
        with self.assertRaises(GeometryNoLegalSupport) as raised:
            with construction_geometry_bridge(paired, tokenizer=self.tokenizer, generation_position_groups=groups(3),
                                              native_constraints=self.native, enabled=True) as monitor:
                paired._paired_suffix_candidates(logits, current_tokens=x, prompt_length=2, gen_length=19,
                    semantic_group=4, step_in_group=2)
        self.assertEqual(called, [])
        self.assertIs(paired._paired_suffix_candidates, original)
        record = raised.exception.to_dict()
        self.assertEqual(record["reason"], "pbc_no_legal_completion")
        self.assertEqual(record["failure_positions"], [18])
        self.assertEqual(record["partial_body_token_ids"][0][18], MASK_TOKEN_ID)
        self.assertEqual(record["step_in_group"], 2)
        self.assertTrue(monitor.report()["failed"])

    def test_incomplete_xy_or_cell_is_engineering_error_not_silent_mask_skip(self):
        for x, group, positions in (
            (canvas(self.tokenizer, 2, coords=[(0, 0, 0), (None, 0, None)]), 4, [14]),
            (canvas(self.tokenizer, 1, coords=[(None, None, None)]), 2, [8]),
        ):
            if group == 2:
                x[0, 3] = MASK_TOKEN_ID
            with self.subTest(group=group), self.assertRaises(GeometryBridgeContractError):
                self.apply(self.monitor((x.shape[1] - 9) // 4), lattice_logits(self.tokenizer, x, positions), x, group)

    def test_complete_degenerate_lattice_is_explicit_constraint_failure(self):
        x = canvas(self.tokenizer, 1, coords=[(None, None, None)], angles=(30, 30, 100))
        with self.assertRaises(GeometryNoLegalSupport) as raised:
            self.apply(self.monitor(1), lattice_logits(self.tokenizer, x, [8]), x, 2)
        self.assertIn("nondegenerate_lattice", raised.exception.to_dict()["reason"])

    def test_all_masked_legacy_lattice_support_cannot_sample_token_zero(self):
        x = canvas(self.tokenizer, 1, coords=[(None, None, None)])
        x[0, 3:9] = MASK_TOKEN_ID
        logits = lattice_logits(self.tokenizer, x, [])
        with self.assertRaises(GeometryNoLegalSupport) as raised:
            self.apply(self.monitor(1), logits, x, 1)
        self.assertEqual(raised.exception.to_dict()["failure_positions"], [1, 2, 3, 4, 5, 6])

    def test_nonfinite_model_logits_are_engineering_failures(self):
        x = canvas(self.tokenizer, 1, coords=[(None, None, None)])
        logits = lattice_logits(self.tokenizer, x, [8])
        logits[0, 10, self.tokenizer.vocab["<X_010>"]] = torch.nan
        with self.assertRaises(GeometryBridgeContractError):
            self.apply(self.monitor(1), logits, x, 2)

    def test_mixed_axis_schedule_and_multirow_enabled_calls_are_rejected(self):
        invalid = groups(1)
        invalid[2], invalid[3] = invalid[3], invalid[2]
        with self.assertRaises(GeometryBridgeContractError):
            ConstructionGeometryMonitor(enabled=True, tokenizer=self.tokenizer,
                generation_position_groups=invalid, native_constraints=self.native)
        x = canvas(self.tokenizer, 1, coords=[(None, None, None)])
        with self.assertRaises(GeometryBridgeContractError):
            self.apply(self.monitor(1), lattice_logits(self.tokenizer, x, [8]).repeat(2, 1, 1), x.repeat(2, 1), 2)


class FrozenSamplerHookTest(unittest.TestCase):
    def test_off_is_exact_noop_even_with_unusable_optional_inputs(self):
        paired = load_frozen_pair_fixture()
        original = paired._paired_suffix_candidates
        logits = torch.arange(33, dtype=torch.float32).reshape(1, 3, 11) / 20
        kwargs = dict(current_tokens=torch.tensor([[1, MASK_TOKEN_ID, MASK_TOKEN_ID]]), prompt_length=1,
                      gen_length=2, temperature=.7, remasking="low_confidence", base_seeds=[17], semantic_group=3, step_in_group=2)
        expected = original(logits.clone(), **kwargs)
        with construction_geometry_bridge(paired, tokenizer=object(), generation_position_groups=[], native_constraints=None) as monitor:
            self.assertIs(paired._paired_suffix_candidates, original)
            actual = paired._paired_suffix_candidates(logits.clone(), **kwargs)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
        self.assertEqual(monitor.report()["candidate_calls"], 0)
        self.assertFalse(monitor.report()["enabled"])

    def test_noop_geometry_preserves_frozen_full_generation_and_noise_calls(self):
        tokenizer = TinyTokenizer()
        native = native_constraints(tokenizer)
        desired = canvas(tokenizer, 1, coords=[(10, 20, 30)])[0, 2:]
        width = max(tokenizer.vocab.values()) + 1
        class FixedLogitFixture:
            device = torch.device("cpu")
            def get_output_embeddings(self):
                return SimpleNamespace(weight=torch.zeros(width, 1))
            def __call__(self, x, **_kwargs):
                scores = torch.full((*x.shape, width), torch.finfo(torch.float32).min)
                for position, token in enumerate(desired):
                    scores[:, 2 + position, token] = 1.0
                return SimpleNamespace(logits=scores)
        paired = load_frozen_pair_fixture()
        original = paired._paired_suffix_candidates
        schema = [[int(token)] for token in desired]
        arguments = dict(base_seeds=[91117], attention_mask=torch.ones(1, 2, dtype=torch.long), gen_length=11,
            temperature=.7, cfg_scale=0., remasking="low_confidence", mask_id=MASK_TOKEN_ID,
            allowed_token_ids_by_generation_pos=schema,
            prefill_token_ids_by_generation_pos={0: [int(desired[0])], 7: [int(desired[7])]},
            generation_position_groups=groups(1), lightweight_decoding_constraints=native)
        with patch.object(paired, "paired_uniform", wraps=paired.paired_uniform) as spy:
            expected = paired.generate_paired_exact_plan(FixedLogitFixture(), torch.tensor([[1, 1]]), **arguments)
            expected_noise = list(spy.call_args_list)
        with patch.object(paired, "paired_uniform", wraps=paired.paired_uniform) as spy:
            with construction_geometry_bridge(paired, tokenizer=tokenizer, generation_position_groups=groups(1),
                                              native_constraints=native, enabled=True) as monitor:
                actual = paired.generate_paired_exact_plan(FixedLogitFixture(), torch.tensor([[1, 1]]), **arguments)
            actual_noise = list(spy.call_args_list)
        self.assertIs(paired._paired_suffix_candidates, original)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual_noise, expected_noise)
        self.assertTrue(torch.equal(actual[0, 2:], desired))
        self.assertEqual(monitor.report()["candidate_calls"], 9)
        self.assertFalse(monitor.report()["failed"])

    def test_frozen_sequential_z_reveal_uses_new_distance_support(self):
        tokenizer = TinyTokenizer()
        native = native_constraints(tokenizer)
        desired = canvas(tokenizer, 2, coords=[(0, 0, 0), (0, 0, 0)])[0, 2:]
        width = max(tokenizer.vocab.values()) + 1
        z_ids = [tokenizer.vocab[f"<Z_{value:03d}>"] for value in range(101)]
        class CrowdedLogitFixture:
            device = torch.device("cpu")
            def get_output_embeddings(self):
                return SimpleNamespace(weight=torch.zeros(width, 1))
            def __call__(self, x, **_kwargs):
                scores = torch.full((*x.shape, width), torch.finfo(torch.float32).min)
                for position, token in enumerate(desired):
                    scores[:, 2 + position, token] = 1.0
                for position in (10, 14):
                    scores[:, 2 + position, z_ids] = torch.tensor([-min(z, 100 - z) / 10 for z in range(101)])
                return SimpleNamespace(logits=scores)
        schema = [[int(token)] for token in desired]
        schema[10] = schema[14] = z_ids
        args = dict(base_seeds=[17], attention_mask=torch.ones(1, 2, dtype=torch.long), gen_length=15,
            temperature=0., cfg_scale=0., remasking="low_confidence", mask_id=MASK_TOKEN_ID,
            allowed_token_ids_by_generation_pos=schema,
            prefill_token_ids_by_generation_pos={0: [int(desired[0])], 7: [int(desired[7])], 11: [int(desired[11])]},
            generation_position_groups=groups(2), lightweight_decoding_constraints=native)
        paired = load_frozen_pair_fixture()
        with construction_geometry_bridge(paired, tokenizer=tokenizer, generation_position_groups=groups(2),
                                          native_constraints=native, enabled=True) as monitor:
            result = paired.generate_paired_exact_plan(CrowdedLogitFixture(), torch.tensor([[1, 1]]), **args)[0, 2:]
        bins = [z_ids.index(int(result[position])) for position in (10, 14)]
        separation = min(abs(bins[0] - bins[1]), 100 - abs(bins[0] - bins[1])) / 100 * 4
        self.assertGreaterEqual(separation, .5)
        z_events = [event for event in monitor.report()["events"] if event.get("stage") == "Z"]
        self.assertEqual([len(event["active_positions"]) for event in z_events], [2, 1])
        self.assertGreater(monitor.report()["newly_masked_legal_tokens"], 1)
        for position in (0, 7, 11):
            self.assertEqual(int(result[position]), int(desired[position]))


if __name__ == "__main__":
    unittest.main()
