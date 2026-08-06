import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local planning mirror may not have torch.
    torch = None

if torch is not None:
    from scripts.llada_sft import (
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
        masked_rel = torch.nonzero(processed["masked_indices"][0], as_tuple=False).flatten() - 7
        self.assertEqual(masked_rel.tolist(), [8])

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
            "mask_policy_ids": torch.tensor([MASK_POLICY_TO_ID["normal"], MASK_POLICY_TO_ID["normal"]]),
            "loss_profile_ids": torch.tensor([LOSS_PROFILE_TO_ID["text"], LOSS_PROFILE_TO_ID["text"]]),
            "module_ids": torch.tensor([0, 0]),
        }
        class TinyTokenizer:
            def get_vocab(self):
                return {}

            def convert_tokens_to_ids(self, token):
                return None

        config = build_loss_config(
            TinyTokenizer(),
            type("Args", (), {
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
            })()
        )
        torch.manual_seed(7)
        high_loss = compute_loss(ToyModel(), {**batch, "sample_weights": torch.tensor([1.0, 0.0])}, config)
        torch.manual_seed(7)
        low_loss = compute_loss(ToyModel(), {**batch, "sample_weights": torch.tensor([0.0, 1.0])}, config)
        self.assertGreater(float(high_loss), float(low_loss))

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
                {"composition_bucket": "single_element", "composition_reason": "single_element_shortcut"},
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
