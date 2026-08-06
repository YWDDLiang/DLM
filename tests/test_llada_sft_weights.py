import json
from pathlib import Path
import re
import tempfile
import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer
from crystal_dlm.planned_corruption import (
    corruption_key_for_record,
    current_order_groups,
    position_group_ids,
    stateless_uniform,
)

try:
    import torch
except (
    ModuleNotFoundError
):  # pragma: no cover - local planning mirror may not have torch.
    torch = None

if torch is not None:
    from scripts.llada_sft import (
        _stateless_uniform_tensor,
        DataCollator,
        JsonlSftDataset,
        LOSS_PROFILE_TO_ID,
        MASK_POLICY_TO_ID,
        build_answer_position_weights,
        build_loss_config,
        compute_loss,
        DistributedWeightedSampler,
        forward_process,
        parse_sample_weight_multipliers,
        sample_weight_multiplier_for_row,
        semantic_init_element_tokens,
        summarize_sample_weight_multipliers,
        summarize_sample_weights,
        validate_corruption_mix,
    )
else:
    build_answer_position_weights = None


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class LLaDASftWeightTest(unittest.TestCase):
    def test_slot_weight_groups(self):
        empty_id = 10
        x_pad_id, y_pad_id, z_pad_id = 11, 12, 13
        element_id = 20
        coord_id = 30
        input_ids = torch.full((1, 114), coord_id, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        prompt_lengths = torch.tensor([7], dtype=torch.long)

        # Answer position 0 is <N_...>; position 7 starts slot 00.
        input_ids[0, 7] = 101
        input_ids[0, 14] = 201
        input_ids[0, 15] = element_id
        input_ids[0, 16] = coord_id
        input_ids[0, 17] = coord_id
        input_ids[0, 18] = coord_id
        input_ids[0, 19] = 202
        input_ids[0, 20] = empty_id
        input_ids[0, 21] = x_pad_id
        input_ids[0, 22] = y_pad_id
        input_ids[0, 23] = z_pad_id
        input_ids[0, 24] = 203
        input_ids[0, 25] = element_id
        input_ids[0, 26] = coord_id
        input_ids[0, 27] = coord_id
        input_ids[0, 28] = coord_id

        loss_config = {
            "atom_count_loss_weight": 3.0,
            "slot_marker_loss_weight": 0.25,
            "empty_slot_loss_weight": 0.5,
            "nonempty_slot_loss_weight": 2.0,
            "late_slot_start": 1,
            "late_nonempty_slot_loss_weight": 4.0,
            "coordinate_loss_weight": 1.0,
            "pad_coordinate_loss_weight": 0.1,
            "empty_token_id": empty_id,
            "pad_coord_token_ids": [x_pad_id, y_pad_id, z_pad_id],
        }
        weights = build_answer_position_weights(
            input_ids,
            attention_mask,
            prompt_lengths,
            loss_config,
        )

        self.assertAlmostEqual(weights[0, 7].item(), 3.0)
        self.assertAlmostEqual(weights[0, 14].item(), 0.25)
        self.assertAlmostEqual(weights[0, 15].item(), 2.0)
        self.assertAlmostEqual(weights[0, 16].item(), 1.0)
        self.assertAlmostEqual(weights[0, 20].item(), 0.5)
        self.assertAlmostEqual(weights[0, 21].item(), 0.1)
        self.assertAlmostEqual(weights[0, 25].item(), 4.0)

    def test_missing_fixed_slot_tokens_use_sentinel(self):
        class ToyTokenizer:
            def get_vocab(self):
                return {"<DOPANT_Al>": 1}

            def convert_tokens_to_ids(self, token):
                return self.get_vocab().get(token)

        class Args:
            atom_count_loss_weight = 1.0
            slot_marker_loss_weight = 1.0
            empty_slot_loss_weight = 1.0
            nonempty_slot_loss_weight = 1.0
            late_slot_start = 4
            late_nonempty_slot_loss_weight = None
            coordinate_loss_weight = 1.0
            pad_coordinate_loss_weight = 1.0

        config = build_loss_config(ToyTokenizer(), Args())
        self.assertEqual(config["empty_token_id"], -1)
        self.assertEqual(config["pad_coord_token_ids"], [-1, -1, -1])

    def test_dynamic_weight_groups(self):
        element_id = 20
        coord_id = 30
        input_ids = torch.full((1, 18), coord_id, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        prompt_lengths = torch.tensor([3], dtype=torch.long)
        input_ids[0, 3] = 101
        input_ids[0, 10] = element_id
        input_ids[0, 11] = coord_id
        input_ids[0, 12] = coord_id
        input_ids[0, 13] = coord_id
        loss_config = {
            "representation": "dynamic_v1",
            "answer_token_count": 87,
            "atom_count_loss_weight": 3.0,
            "nonempty_slot_loss_weight": 2.5,
            "coordinate_loss_weight": 1.2,
            "dynamic_lattice_length_loss_weight": 1.7,
            "dynamic_lattice_angle_loss_weight": 1.8,
            "dynamic_coord_loss_weight": 1.2,
            "empty_token_id": -1,
            "pad_coord_token_ids": [-1, -1, -1],
        }

        weights = build_answer_position_weights(
            input_ids,
            attention_mask,
            prompt_lengths,
            loss_config,
        )

        self.assertAlmostEqual(weights[0, 3].item(), 3.0)
        self.assertAlmostEqual(weights[0, 4].item(), 1.7)
        self.assertAlmostEqual(weights[0, 7].item(), 1.8)
        self.assertAlmostEqual(weights[0, 10].item(), 2.5)
        self.assertAlmostEqual(weights[0, 11].item(), 1.2)
        self.assertAlmostEqual(weights[0, 13].item(), 1.2)

    def test_text_loss_profile_uses_plain_answer_weights(self):
        input_ids = torch.full((2, 20), 30, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        prompt_lengths = torch.tensor([3, 3], dtype=torch.long)
        input_ids[0, 3] = 101
        input_ids[1, 3] = 101
        loss_config = {
            "representation": "fixed_slot",
            "answer_token_count": 107,
            "atom_count_loss_weight": 3.0,
            "slot_marker_loss_weight": 0.25,
            "empty_slot_loss_weight": 0.5,
            "nonempty_slot_loss_weight": 2.0,
            "late_slot_start": 4,
            "late_nonempty_slot_loss_weight": 4.0,
            "coordinate_loss_weight": 1.0,
            "pad_coordinate_loss_weight": 0.1,
            "empty_token_id": 10,
            "pad_coord_token_ids": [11, 12, 13],
        }

        weights = build_answer_position_weights(
            input_ids,
            attention_mask,
            prompt_lengths,
            loss_config,
            loss_profile_ids=torch.tensor(
                [LOSS_PROFILE_TO_ID["fixed_slot"], LOSS_PROFILE_TO_ID["text"]],
                dtype=torch.long,
            ),
        )

        self.assertAlmostEqual(weights[0, 3].item(), 3.0)
        self.assertAlmostEqual(weights[1, 3].item(), 1.0)
        self.assertAlmostEqual(weights[1, 10].item(), 1.0)

    def test_cif_lite_module_weights(self):
        input_ids = torch.ones((3, 8), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        prompt_lengths = torch.tensor([2, 2, 2], dtype=torch.long)
        loss_config = {
            "representation": "cif_lite_modular",
            "answer_token_count": 16,
            "composition_module_loss_weight": 2.0,
            "lattice_module_loss_weight": 0.75,
            "sites_module_loss_weight": 1.25,
        }

        weights = build_answer_position_weights(
            input_ids,
            attention_mask,
            prompt_lengths,
            loss_config,
            module_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        )

        self.assertAlmostEqual(weights[0, 2].item(), 2.0)
        self.assertAlmostEqual(weights[1, 2].item(), 0.75)
        self.assertAlmostEqual(weights[2, 2].item(), 1.25)
        self.assertAlmostEqual(weights[0, 1].item(), 0.0)

    def test_active_element_mask_policy_masks_only_occupancy_targets(self):
        empty_id = 10
        element_id = 20
        input_ids = torch.full((1, 114), 30, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        prompt_lengths = torch.tensor([7], dtype=torch.long)
        input_ids[0, 7] = 101
        input_ids[0, 15] = element_id
        input_ids[0, 20] = empty_id
        processed = forward_process(
            input_ids,
            attention_mask,
            prompt_lengths,
            mask_policy_ids=torch.tensor([MASK_POLICY_TO_ID["active_element"]]),
            empty_token_id=empty_id,
            eps=1.0,
        )
        masked_rel = (
            torch.nonzero(processed["masked_indices"][0], as_tuple=False).flatten() - 7
        )
        self.assertEqual(masked_rel.tolist(), [8])

    def test_planned_fraction_zero_preserves_original_forward_path(self):
        input_ids = torch.tensor(
            [
                [99, 11, 12, 13, 14, 15],
                [99, 21, 22, 23, 24, 25],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        prompt_lengths = torch.tensor([1, 1], dtype=torch.long)
        torch.manual_seed(314159)
        baseline = forward_process(
            input_ids,
            attention_mask,
            prompt_lengths,
        )
        torch.manual_seed(314159)
        explicit_zero = forward_process(
            input_ids,
            attention_mask,
            prompt_lengths,
            iid_fraction=0.0,
            planned_fraction=0.0,
            planned_group_ids=torch.full((2, 5), 999, dtype=torch.long),
            planned_group_counts=torch.tensor([-1, -1], dtype=torch.long),
            corruption_keys=torch.tensor([1, 2], dtype=torch.long),
            corruption_step=999,
            corruption_seed=888,
        )
        for key in (
            "noisy",
            "masked_indices",
            "input_masked_indices",
            "p_mask",
            "answer_mask",
            "candidate_mask",
        ):
            self.assertTrue(torch.equal(baseline[key], explicit_zero[key]), key)

    def test_stateless_iid_fixed_panel_ignores_global_torch_rng(self):
        input_ids = torch.stack(
            [
                torch.arange(8, dtype=torch.long) + 100,
                torch.arange(8, dtype=torch.long) + 200,
            ]
        )
        attention_mask = torch.ones_like(input_ids)
        prompt_lengths = torch.tensor([2, 2], dtype=torch.long)
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_lengths": prompt_lengths,
            "iid_fraction": 1.0,
            "planned_fraction": 0.0,
            "corruption_keys": torch.tensor([4321, 8765], dtype=torch.long),
            "corruption_step": 0,
            "corruption_seed": 20260731,
            "stateless_iid": True,
        }
        torch.manual_seed(1)
        first = forward_process(**kwargs)
        torch.rand(1000)
        torch.manual_seed(999)
        second = forward_process(**kwargs)
        for key in (
            "noisy",
            "masked_indices",
            "p_mask",
            "candidate_mask",
        ):
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_torch_stateless_uniform_matches_cpu_formula(self):
        keys = torch.tensor([1, 123456789, 2**62 + 17], dtype=torch.long)
        positions = torch.tensor([0, 9, 87], dtype=torch.long)
        actual = _stateless_uniform_tensor(
            keys,
            step=17,
            seed=20260731,
            stream=2,
            positions=positions,
        )
        expected = torch.tensor(
            [
                stateless_uniform(
                    int(key),
                    step=17,
                    seed=20260731,
                    stream=2,
                    position=int(position),
                )
                for key, position in zip(keys.tolist(), positions.tolist())
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(actual.cpu(), expected, atol=1e-7, rtol=0.0))

    def test_planned_forward_masks_future_but_supervises_only_active_group(self):
        groups = current_order_groups(1)
        encoded = position_group_ids(groups, answer_length=11)
        input_ids = torch.arange(13, dtype=torch.long).unsqueeze(0) + 100
        attention_mask = torch.ones_like(input_ids)
        prompt_lengths = torch.tensor([2], dtype=torch.long)
        processed = forward_process(
            input_ids,
            attention_mask,
            prompt_lengths,
            iid_fraction=0.0,
            planned_fraction=1.0,
            planned_group_ids=torch.tensor([encoded], dtype=torch.long),
            planned_group_counts=torch.tensor([len(groups)], dtype=torch.long),
            corruption_keys=torch.tensor([123456789], dtype=torch.long),
            corruption_step=5,
            corruption_seed=20260731,
        )
        self.assertTrue(bool(processed["planned_sample_mask"][0]))
        active_group = int(processed["active_group_indices"][0])
        active_positions = set(groups[active_group].positions)
        prerequisite_positions = {
            position for group in groups[:active_group] for position in group.positions
        }
        future_positions = {
            position
            for group in groups[active_group + 1 :]
            for position in group.positions
        }
        supervised = {
            int(position) - 2
            for position in torch.nonzero(
                processed["masked_indices"][0],
                as_tuple=False,
            ).flatten()
        }
        input_masked = {
            int(position) - 2
            for position in torch.nonzero(
                processed["input_masked_indices"][0],
                as_tuple=False,
            ).flatten()
        }
        self.assertTrue(supervised)
        self.assertTrue(supervised.issubset(active_positions))
        self.assertTrue(future_positions.issubset(input_masked))
        self.assertFalse(prerequisite_positions.intersection(input_masked))
        self.assertFalse(future_positions.intersection(supervised))

    def test_stateless_mixture_is_invariant_to_batch_order(self):
        groups = current_order_groups(1)
        encoded = position_group_ids(groups, answer_length=11)
        input_ids = torch.stack(
            [
                torch.arange(13, dtype=torch.long) + 100,
                torch.arange(13, dtype=torch.long) + 200,
            ]
        )
        attention_mask = torch.ones_like(input_ids)
        prompt_lengths = torch.tensor([2, 2], dtype=torch.long)
        group_ids = torch.tensor([encoded, encoded], dtype=torch.long)
        group_counts = torch.tensor([len(groups), len(groups)], dtype=torch.long)
        keys = torch.tensor([111, 222], dtype=torch.long)

        first = forward_process(
            input_ids,
            attention_mask,
            prompt_lengths,
            iid_fraction=2.0,
            planned_fraction=1.0,
            planned_group_ids=group_ids,
            planned_group_counts=group_counts,
            corruption_keys=keys,
            corruption_step=9,
            corruption_seed=20260731,
        )
        order = torch.tensor([1, 0], dtype=torch.long)
        second = forward_process(
            input_ids[order],
            attention_mask[order],
            prompt_lengths[order],
            iid_fraction=2.0,
            planned_fraction=1.0,
            planned_group_ids=group_ids[order],
            planned_group_counts=group_counts[order],
            corruption_keys=keys[order],
            corruption_step=9,
            corruption_seed=20260731,
        )
        for key in (
            "noisy",
            "masked_indices",
            "input_masked_indices",
            "p_mask",
            "candidate_mask",
            "planned_sample_mask",
            "active_group_indices",
        ):
            self.assertTrue(torch.equal(first[key], second[key][order]), key)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_stateless_planned_forward_matches_cpu_on_cuda(self):
        groups = current_order_groups(2)
        encoded = position_group_ids(groups, answer_length=15)
        cpu_inputs = {
            "input_ids": torch.stack(
                [
                    torch.arange(17, dtype=torch.long) + 100,
                    torch.arange(17, dtype=torch.long) + 200,
                ]
            ),
            "attention_mask": torch.ones((2, 17), dtype=torch.long),
            "prompt_lengths": torch.tensor([2, 2], dtype=torch.long),
            "planned_group_ids": torch.tensor(
                [encoded, encoded],
                dtype=torch.long,
            ),
            "planned_group_counts": torch.tensor(
                [len(groups), len(groups)],
                dtype=torch.long,
            ),
            "corruption_keys": torch.tensor([12345, 67890], dtype=torch.long),
        }
        cpu = forward_process(
            **cpu_inputs,
            iid_fraction=2.0,
            planned_fraction=1.0,
            corruption_step=21,
            corruption_seed=20260731,
        )
        cuda = forward_process(
            **{key: value.cuda() for key, value in cpu_inputs.items()},
            iid_fraction=2.0,
            planned_fraction=1.0,
            corruption_step=21,
            corruption_seed=20260731,
        )
        for key in (
            "noisy",
            "masked_indices",
            "input_masked_indices",
            "p_mask",
            "candidate_mask",
            "planned_sample_mask",
            "active_group_indices",
        ):
            self.assertTrue(torch.equal(cpu[key], cuda[key].cpu()), key)

    def test_dynamic_dataset_and_collator_pad_planned_group_ids(self):
        class TinyTokenizer:
            pad_token_id = 0

            def __init__(self):
                self.vocab = {}

            def __call__(
                self,
                text,
                add_special_tokens=False,
                truncation=False,
                max_length=None,
            ):
                del add_special_tokens
                pieces = re.findall(r"<[^>]+>|.", text, flags=re.DOTALL)
                ids = []
                for piece in pieces:
                    if piece not in self.vocab:
                        self.vocab[piece] = len(self.vocab) + 1
                    ids.append(self.vocab[piece])
                if truncation and max_length is not None:
                    ids = ids[:max_length]
                return {"input_ids": ids}

        first_answer, _ = arrays_to_dynamic_answer(
            lengths=[3.0, 3.0, 3.0],
            angles=[90.0, 90.0, 90.0],
            species=["Li"],
            frac_coords=[[0.0, 0.0, 0.0]],
        )
        second_answer, _ = arrays_to_dynamic_answer(
            lengths=[3.0, 3.0, 4.0],
            angles=[90.0, 90.0, 90.0],
            species=["Li", "O"],
            frac_coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )
        rows = [
            {
                "prompt": "Generate:",
                "answer": first_answer,
                "num_atoms": 1,
                "answer_semantic_length": 11,
                "answer_token_count": 11,
            },
            {
                "prompt": "Generate:",
                "answer": second_answer,
                "num_atoms": 2,
                "answer_semantic_length": 15,
                "answer_token_count": 15,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "train.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            tokenizer = TinyTokenizer()
            dataset = JsonlSftDataset(
                path,
                tokenizer,
                max_length=128,
                planned_corruption_policy="d1",
            )
            batch = DataCollator(tokenizer)([dataset[0], dataset[1]])

        self.assertEqual(batch["planned_group_ids"].shape, (2, 15))
        self.assertEqual(batch["planned_group_counts"].tolist(), [6, 6])
        self.assertTrue((batch["planned_group_ids"][0, 11:] == -1).all())
        self.assertEqual(
            int(batch["corruption_keys"][0]),
            corruption_key_for_record(rows[0]),
        )

    def test_corruption_mix_validation_enforces_dynamic_planned_policy(self):
        registered = validate_corruption_mix(
            iid_fraction=2.0,
            planned_fraction=1.0,
            planned_policy="d2",
            representation="dynamic_v1",
        )
        self.assertAlmostEqual(float(registered["planned_probability"]), 1.0 / 3.0)
        with self.assertRaises(ValueError):
            validate_corruption_mix(
                iid_fraction=2.0,
                planned_fraction=1.0,
                planned_policy="none",
                representation="dynamic_v1",
            )
        with self.assertRaises(ValueError):
            validate_corruption_mix(
                iid_fraction=2.0,
                planned_fraction=1.0,
                planned_policy="d1",
                representation="fixed_slot",
            )

    def test_semantic_init_element_tokens_only_when_new_tokens_added(self):
        class ToyTokenizer:
            def __init__(self):
                self.vocab = {
                    "O": 0,
                    "oxygen": 1,
                    " oxide": 2,
                    "<E_O>": 3,
                }

            def __call__(self, text, add_special_tokens=False):
                if text in self.vocab:
                    return {"input_ids": [self.vocab[text]]}
                return {"input_ids": [0]}

            def convert_tokens_to_ids(self, token):
                return self.vocab.get(token)

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(4, 2)
                self.out = torch.nn.Linear(2, 4, bias=False)
                with torch.no_grad():
                    self.emb.weight.zero_()
                    self.emb.weight[0] = torch.tensor([1.0, 0.0])
                    self.emb.weight[1] = torch.tensor([0.0, 1.0])
                    self.emb.weight[2] = torch.tensor([1.0, 1.0])
                    self.out.weight.zero_()

            def get_input_embeddings(self):
                return self.emb

            def get_output_embeddings(self):
                return self.out

        model = ToyModel()
        report = semantic_init_element_tokens(
            ToyTokenizer(),
            model,
            enabled=True,
            num_new_tokens=1,
        )
        self.assertTrue(report["attempted"])
        self.assertGreaterEqual(report["initialized_count"], 1)
        self.assertFalse(torch.allclose(model.emb.weight[3], torch.zeros(2)))

        skipped = semantic_init_element_tokens(
            ToyTokenizer(),
            model,
            enabled=True,
            num_new_tokens=0,
        )
        self.assertEqual(skipped["skipped_reason"], "no_new_tokens_added")

    def test_compute_loss_applies_batch_sample_weights(self):
        class ToyOutput:
            def __init__(self, logits):
                self.logits = logits

        class ToyModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None):
                logits = torch.zeros((*input_ids.shape, 3), dtype=torch.float32)
                logits[:, :, 0] = 4.0
                return ToyOutput(logits)

        batch = {
            "input_ids": torch.tensor([[2, 1, 1, 1], [2, 0, 0, 0]], dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "prompt_lengths": torch.tensor([1, 1], dtype=torch.long),
            "mask_policy_ids": torch.tensor(
                [MASK_POLICY_TO_ID["normal"], MASK_POLICY_TO_ID["normal"]]
            ),
            "loss_profile_ids": torch.tensor(
                [LOSS_PROFILE_TO_ID["text"], LOSS_PROFILE_TO_ID["text"]]
            ),
            "module_ids": torch.tensor([0, 0]),
        }

        class TinyTokenizer:
            def get_vocab(self):
                return {}

            def convert_tokens_to_ids(self, token):
                return None

        config = build_loss_config(
            TinyTokenizer(),
            type(
                "Args",
                (),
                {
                    "physical_header_loss_weight": 1.0,
                    "atom_count_loss_weight": 1.0,
                    "slot_marker_loss_weight": 1.0,
                    "empty_slot_loss_weight": 1.0,
                    "nonempty_slot_loss_weight": 1.0,
                    "late_nonempty_slot_loss_weight": None,
                    "late_slot_start": 10,
                    "coordinate_loss_weight": 1.0,
                    "pad_coordinate_loss_weight": 1.0,
                    "composition_module_loss_weight": 1.0,
                    "lattice_module_loss_weight": 1.0,
                    "sites_module_loss_weight": 1.0,
                    "crysllmgen_lattice_loss_weight": 1.0,
                    "crysllmgen_composition_loss_weight": 1.0,
                    "crysllmgen_species_loss_weight": 1.0,
                    "crysllmgen_coords_loss_weight": 1.0,
                    "crysllmgen_site_coord_loss_weight": 1.0,
                    "fixed_plain_count_loss_weight": 1.0,
                    "fixed_plain_lattice_loss_weight": 1.0,
                    "fixed_plain_elements_loss_weight": 1.0,
                    "fixed_plain_coords_loss_weight": 1.0,
                    "representation": "r5_plan_state",
                    "empty_token_id": -1,
                    "train_prefill_slot_tokens": False,
                    "fixed_slot_body_offset": 0,
                },
            )(),
        )
        torch.manual_seed(7)
        high_loss = compute_loss(
            ToyModel(), {**batch, "sample_weights": torch.tensor([1.0, 0.0])}, config
        )
        torch.manual_seed(7)
        low_loss = compute_loss(
            ToyModel(), {**batch, "sample_weights": torch.tensor([0.0, 1.0])}, config
        )
        self.assertGreater(float(high_loss), float(low_loss))

    def test_compute_loss_planned_mixture_has_finite_backward(self):
        class ToyOutput:
            def __init__(self, logits):
                self.logits = logits

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab_logits = torch.nn.Parameter(torch.zeros(64))

            def forward(self, input_ids, attention_mask=None):
                del attention_mask
                logits = self.vocab_logits.view(1, 1, -1).expand(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    -1,
                )
                return ToyOutput(logits)

        groups = current_order_groups(1)
        encoded = position_group_ids(groups, answer_length=11)
        batch = {
            "input_ids": torch.tensor(
                [[63, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones((1, 12), dtype=torch.long),
            "prompt_lengths": torch.tensor([1], dtype=torch.long),
            "mask_policy_ids": torch.tensor(
                [MASK_POLICY_TO_ID["normal"]],
                dtype=torch.long,
            ),
            "loss_profile_ids": torch.tensor(
                [LOSS_PROFILE_TO_ID["fixed_slot"]],
                dtype=torch.long,
            ),
            "module_ids": torch.tensor([0], dtype=torch.long),
            "sample_weights": torch.tensor([1.0]),
            "planned_group_ids": torch.tensor([encoded], dtype=torch.long),
            "planned_group_counts": torch.tensor([len(groups)], dtype=torch.long),
            "corruption_keys": torch.tensor([998877], dtype=torch.long),
        }
        config = {
            "representation": "dynamic_v1",
            "answer_token_count": 11,
            "fixed_slot_body_offset": 0,
            "train_prefill_slot_tokens": False,
            "empty_token_id": -1,
            "pad_coord_token_ids": [-1, -1, -1],
            "atom_count_loss_weight": 1.0,
            "nonempty_slot_loss_weight": 1.0,
            "coordinate_loss_weight": 1.0,
            "dynamic_lattice_length_loss_weight": 1.0,
            "dynamic_lattice_angle_loss_weight": 1.0,
            "dynamic_coord_loss_weight": 1.0,
            "iid_fraction": 2.0,
            "planned_fraction": 1.0,
            "planned_corruption_policy": "d1",
            "corruption_seed": 20260731,
        }
        model = ToyModel()
        loss = compute_loss(
            model,
            batch,
            config,
            corruption_step=13,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(model.vocab_logits.grad)
        self.assertTrue(torch.isfinite(model.vocab_logits.grad).all())

    def test_distributed_weighted_sampler_respects_zero_weights(self):
        sampler = DistributedWeightedSampler(
            [0.0, 0.0, 1.0],
            num_replicas=1,
            rank=0,
            seed=123,
        )
        self.assertEqual(list(sampler), [2, 2, 2])

    def test_sample_weight_summary_marks_nonuniform(self):
        summary = summarize_sample_weights([1.0, 0.5, 0.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["positive_count"], 2)
        self.assertTrue(summary["has_nonuniform_weights"])

    def test_sample_weight_multipliers_match_bucket_and_reason(self):
        multipliers = parse_sample_weight_multipliers(
            "strict=1.0,all_metal=0.7,single_element=0.05,invalid=0.6"
        )
        self.assertEqual(multipliers["single_element"], 0.05)
        self.assertEqual(
            sample_weight_multiplier_for_row(
                {
                    "composition_bucket": "single_element",
                    "composition_reason": "single_element_shortcut",
                },
                multipliers,
            ),
            0.05,
        )
        self.assertEqual(
            sample_weight_multiplier_for_row(
                {"composition_reason": "charge_neutral_pauling_valid"},
                multipliers,
            ),
            1.0,
        )
        summary = summarize_sample_weight_multipliers(
            [
                {"composition_bucket": "strict", "sample_weight": 1.0},
                {"composition_bucket": "single_element", "sample_weight": 0.7},
            ],
            multipliers,
        )
        self.assertEqual(summary["matched_counts"]["single_element"], 1)

    def test_jsonl_dataset_can_ignore_base_sample_weight(self):
        class TinyDataset:
            rows = [{"sample_weight": 0.1, "composition_bucket": "strict"}]

        from scripts.llada_sft import JsonlSftDataset

        dataset = TinyDataset()
        weights = JsonlSftDataset.sample_weights(
            dataset,
            {"strict": 2.0},
            use_jsonl_sample_weight=False,
        )
        self.assertEqual(weights, [2.0])


if __name__ == "__main__":
    unittest.main()
