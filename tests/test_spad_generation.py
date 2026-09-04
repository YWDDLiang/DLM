from types import SimpleNamespace
import unittest


try:
    import torch
    from crystal_dlm.llada_generation import generate
    from crystal_dlm.spad_generation import (
        Model494ResponseConfig,
        _bounded_translation_free_response,
        _kl_bounded_gain_bias,
        _minimum_image_vector,
        _spad_basin_closure_block_salt,
        _transaction_candidate_tokens,
        continue_spad_species_blocks_from_cursor,
        revise_spad_anchors,
        revise_spad_cell,
        revise_spad_species_blocks,
    )
except ModuleNotFoundError:
    torch = None
    generate = None
    Model494ResponseConfig = None
    revise_spad_anchors = None
    revise_spad_cell = None
    revise_spad_species_blocks = None
    continue_spad_species_blocks_from_cursor = None
    _transaction_candidate_tokens = None
    _spad_basin_closure_block_salt = None


_TorchModuleBase = torch.nn.Module if torch is not None else object


@unittest.skipIf(torch is None, "torch unavailable")
class SPADGenerationTest(unittest.TestCase):
    class TinyModel(_TorchModuleBase):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.output = torch.nn.Embedding(128, 1)
            self.device = torch.device("cpu")
            self.attention_masks = []
            self.token_inputs = []

        def get_output_embeddings(self):
            return self.output

        def forward(self, token_ids, attention_mask=None):
            self.token_inputs.append(token_ids.detach().clone())
            if attention_mask is not None:
                self.attention_masks.append(attention_mask.detach().clone())
            batch, length = token_ids.shape
            logits = torch.zeros((batch, length, 128), dtype=torch.float32)
            logits[..., 21] = 2.0
            return SimpleNamespace(logits=logits)

    def test_row_local_predictor_schedules_batch_without_permuting_canvas(self):
        model = self.TinyModel()
        prompt = torch.tensor([[1], [1]], dtype=torch.long)
        allowed = [[10, 21] for _ in range(5)]
        result = generate(
            model,
            prompt,
            attention_mask=torch.ones_like(prompt),
            steps=5,
            gen_length=5,
            block_length=1,
            temperature=0.0,
            mask_id=127,
            allowed_token_ids_by_generation_pos=allowed,
            generation_position_groups_by_batch=[
                [[0], [4], [1], [2], [3]],
                [[4], [3], [2], [1], [0]],
            ],
        )
        self.assertEqual(tuple(result.shape), (2, 6))
        self.assertFalse(bool((result[:, 1:] == 127).any()))
        self.assertTrue(bool((result[:, 1:] == 21).all()))

    def test_row_keyed_transaction_sampling_is_batch_order_invariant(self):
        logits = torch.zeros((2, 1, 128), dtype=torch.float32)
        logits[:, :, 20:40] = 1.0
        first = _transaction_candidate_tokens(
            logits,
            active_absolute_positions={0: 0, 1: 0},
            temperature=0.7,
            remasking="low_confidence",
            sampling_seeds_by_batch=[123, 456],
            salt=17,
        )
        swapped = _transaction_candidate_tokens(
            logits.flip(0),
            active_absolute_positions={0: 0, 1: 0},
            temperature=0.7,
            remasking="low_confidence",
            sampling_seeds_by_batch=[456, 123],
            salt=17,
        )
        self.assertEqual(int(first[0, 0]), int(swapped[1, 0]))
        self.assertEqual(int(first[1, 0]), int(swapped[0, 0]))

    def test_revision_changes_only_anchor_and_can_hide_literal_suffix(self):
        model = self.TinyModel()
        prompt_length = 2
        gen_length = 15
        values = torch.arange(prompt_length + gen_length).reshape(1, -1) % 50
        allowed = [list(range(50)) for _ in range(gen_length)]
        output, logs = revise_spad_anchors(
            model,
            values,
            prompt_length=prompt_length,
            gen_length=gen_length,
            revision_slots_by_batch=[[0]],
            attention_mask=torch.ones((1, prompt_length), dtype=torch.long),
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=127,
            allowed_token_ids_by_generation_pos=allowed,
            atom_count_grammar=None,
            lightweight_decoding_constraints=None,
            suffix_visible=False,
        )
        active = [prompt_length + position for position in (8, 9, 10)]
        inactive = [index for index in range(output.shape[1]) if index not in active]
        self.assertTrue(torch.equal(output[0, inactive], values[0, inactive]))
        self.assertEqual(output[0, active].tolist(), [21, 21, 21])
        self.assertEqual(logs[0][0]["changed_components"], 3)
        self.assertTrue(logs[0][0]["no_op_was_in_schema"])
        self.assertTrue(any(mask[0, -1].item() == 0 for mask in model.attention_masks))

    def test_cell_closure_changes_only_complete_lattice_transaction(self):
        model = self.TinyModel()
        prompt_length = 2
        gen_length = 15
        values = torch.arange(prompt_length + gen_length).reshape(1, -1) % 50
        allowed = [list(range(50)) for _ in range(gen_length)]
        output, logs = revise_spad_cell(
            model,
            values,
            prompt_length=prompt_length,
            gen_length=gen_length,
            attention_mask=torch.ones((1, prompt_length), dtype=torch.long),
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=127,
            allowed_token_ids_by_generation_pos=allowed,
            atom_count_grammar=None,
            lightweight_decoding_constraints=None,
            strict_geometry_fallback=False,
        )
        active = [prompt_length + position for position in range(1, 7)]
        inactive = [index for index in range(output.shape[1]) if index not in active]
        self.assertTrue(torch.equal(output[0, inactive], values[0, inactive]))
        self.assertEqual(output[0, active].tolist(), [21] * 6)
        self.assertEqual(logs[0]["changed_components"], 6)
        self.assertTrue(logs[0]["all_sites_visible"])

    def test_cell_closure_restores_noop_when_new_cell_causes_pbc_collision(self):
        model = self.TinyModel()
        prompt_length = 1
        suffix = torch.tensor(
            [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                20,
                30,
                40,
                50,
                20,
                31,
                41,
                51,
            ],
            dtype=torch.long,
        )
        complete = torch.cat((torch.tensor([1]), suffix)).reshape(1, -1)
        allowed = [
            [10],
            [11, 21],
            [12, 21],
            [13, 21],
            [14, 21],
            [15, 21],
            [16, 21],
            [20],
            [30],
            [40],
            [50],
            [20],
            [31],
            [41],
            [51],
        ]
        constraints = {
            "representation": "dynamic_v1",
            "body_offset": 0,
            "max_atoms": 2,
            "coord_period": 100,
            "count_token_to_n": {10: 2},
            "length_step": 0.1,
            "length_token_to_bin": {
                "LA": {11: 40, 21: 1},
                "LB": {12: 40, 21: 1},
                "LC": {13: 40, 21: 1},
            },
            "angle_token_to_bin": {
                "AA": {14: 90, 21: 90},
                "AB": {15: 90, 21: 90},
                "AG": {16: 90, 21: 90},
            },
            "gamma_bin_to_token_id": {90: 21},
            "coord_token_to_bin": {
                "X": {30: 0, 31: 50},
                "Y": {40: 0, 41: 50},
                "Z": {50: 0, 51: 50},
            },
            "lattice_volume_mask": True,
            "pbc_min_distance_mask": True,
            "pbc_min_distance_A": 0.5,
            "pbc_image_radius": 2,
        }
        output, logs = revise_spad_cell(
            model,
            complete,
            prompt_length=prompt_length,
            gen_length=15,
            attention_mask=torch.ones((1, prompt_length), dtype=torch.long),
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=127,
            allowed_token_ids_by_generation_pos=allowed,
            atom_count_grammar=None,
            lightweight_decoding_constraints=constraints,
            strict_geometry_fallback=True,
        )
        self.assertTrue(torch.equal(output, complete))
        self.assertTrue(logs[0]["restored_complete_noop"])
        self.assertFalse(logs[0]["geometry_supported_before_restore"])

    def test_species_block_masks_whole_block_then_commits_with_future_visible(self):
        model = self.TinyModel()
        prompt_length = 2
        gen_length = 19
        values = torch.arange(prompt_length + gen_length).reshape(1, -1) % 50
        allowed = [list(range(50)) for _ in range(gen_length)]
        output, logs = revise_spad_species_blocks(
            model,
            values,
            prompt_length=prompt_length,
            gen_length=gen_length,
            revision_blocks_by_batch=[[[1, 0]]],
            attention_mask=torch.ones((1, prompt_length), dtype=torch.long),
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=127,
            allowed_token_ids_by_generation_pos=allowed,
            atom_count_grammar=None,
            lightweight_decoding_constraints=None,
        )
        active_generation = [
            position
            for slot in (1, 0)
            for position in (8 + 4 * slot, 9 + 4 * slot, 10 + 4 * slot)
        ]
        active_absolute = [prompt_length + position for position in active_generation]
        inactive = [
            index for index in range(output.shape[1]) if index not in active_absolute
        ]
        self.assertTrue(torch.equal(output[0, inactive], values[0, inactive]))
        self.assertEqual(output[0, active_absolute].tolist(), [21] * 6)
        self.assertFalse(bool((output[:, prompt_length:] == 127).any()))

        first_input = model.token_inputs[0][0]
        self.assertEqual(first_input[active_absolute].tolist(), [127] * 6)
        future_absolute = prompt_length + 18
        self.assertEqual(int(first_input[future_absolute]), int(values[0, future_absolute]))
        self.assertTrue(all(bool(mask[0, future_absolute]) for mask in model.attention_masks))

        block_log = logs[0][0]
        self.assertEqual(block_log["slot_indices"], [1, 0])
        self.assertTrue(block_log["all_block_sites_masked_initially"])
        self.assertTrue(block_log["suffix_visible"])
        self.assertTrue(block_log["non_active_tokens_unchanged"])
        self.assertEqual(
            [site["slot_index"] for site in block_log["site_revisions"]],
            [1, 0],
        )

    def test_species_block_rolls_back_only_site_with_no_legal_z(self):
        model = self.TinyModel()
        prompt_length = 1
        gen_length = 15
        suffix = torch.tensor(
            [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                20,
                30,
                40,
                50,
                20,
                31,
                41,
                51,
            ],
            dtype=torch.long,
        )
        complete = torch.cat((torch.tensor([1]), suffix)).reshape(1, -1)
        allowed = [
            [10],
            [11],
            [12],
            [13],
            [14],
            [15],
            [16],
            [20],
            [30, 31],
            [40, 41],
            [50, 51],
            [20],
            [30, 31],
            [40, 41],
            [50, 51],
        ]
        constraints = {
            "representation": "dynamic_v1",
            "body_offset": 0,
            "max_atoms": 2,
            "coord_period": 100,
            "count_token_to_n": {10: 2},
            "length_step": 0.1,
            "length_token_to_bin": {
                "LA": {11: 10},
                "LB": {12: 10},
                "LC": {13: 10},
            },
            "angle_token_to_bin": {
                "AA": {14: 90},
                "AB": {15: 90},
                "AG": {16: 90},
            },
            "coord_token_to_bin": {
                "X": {30: 0, 31: 1},
                "Y": {40: 0, 41: 1},
                "Z": {50: 0, 51: 1},
            },
            "coord_bin_to_token_id": {
                "X": {0: 30, 1: 31},
                "Y": {0: 40, 1: 41},
                "Z": {0: 50, 1: 51},
            },
            "pbc_min_distance_mask": True,
            "pbc_min_distance_A": 0.5,
            "pbc_image_radius": 2,
        }
        output, logs = revise_spad_species_blocks(
            model,
            complete,
            prompt_length=prompt_length,
            gen_length=gen_length,
            revision_blocks_by_batch=[[[1]]],
            attention_mask=torch.ones((1, prompt_length), dtype=torch.long),
            temperature=0.7,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=127,
            allowed_token_ids_by_generation_pos=allowed,
            atom_count_grammar=None,
            lightweight_decoding_constraints=constraints,
            sampling_seeds_by_batch=[123],
        )
        self.assertTrue(torch.equal(output, complete))
        self.assertEqual(logs[0][0]["restored_site_count"], 1)
        self.assertTrue(
            logs[0][0]["site_revisions"][0]["restored_site_no_legal_z"]
        )
        self.assertFalse(logs[0][0]["final_geometry_supported"])
        self.assertFalse(bool((output[:, prompt_length:] == 127).any()))

    def test_species_block_restores_complete_block_after_local_site_rollback(self):
        model = self.TinyModel()
        prompt_length = 1
        gen_length = 19
        suffix = torch.tensor(
            [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                20,
                30,
                40,
                50,
                20,
                31,
                41,
                51,
                20,
                32,
                42,
                52,
            ],
            dtype=torch.long,
        )
        complete = torch.cat((torch.tensor([1]), suffix)).reshape(1, -1)
        allowed = [
            [10],
            [11],
            [12],
            [13],
            [14],
            [15],
            [16],
            [20],
            [30],
            [40],
            [50],
            [20],
            [31, 21],
            [41, 21],
            [51, 21],
            [20],
            [32, 21],
            [42, 21],
            [52, 21],
        ]
        constraints = {
            "representation": "dynamic_v1",
            "body_offset": 0,
            "max_atoms": 3,
            "coord_period": 100,
            "count_token_to_n": {10: 3},
            "length_step": 0.1,
            "length_token_to_bin": {
                "LA": {11: 100},
                "LB": {12: 100},
                "LC": {13: 100},
            },
            "angle_token_to_bin": {
                "AA": {14: 90},
                "AB": {15: 90},
                "AG": {16: 90},
            },
            "coord_token_to_bin": {
                "X": {30: 0, 31: 30, 32: 60, 21: 60},
                "Y": {40: 0, 41: 30, 42: 60, 21: 60},
                "Z": {50: 0, 51: 30, 52: 60, 21: 60},
            },
            "coord_bin_to_token_id": {
                "X": {0: 30, 30: 31, 60: 21},
                "Y": {0: 40, 30: 41, 60: 21},
                "Z": {60: 21},
            },
            "pbc_min_distance_mask": True,
            "pbc_min_distance_A": 0.5,
            "pbc_image_radius": 2,
        }
        output, logs = revise_spad_species_blocks(
            model,
            complete,
            prompt_length=prompt_length,
            gen_length=gen_length,
            revision_blocks_by_batch=[[[1, 2]]],
            attention_mask=torch.ones((1, prompt_length), dtype=torch.long),
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=127,
            allowed_token_ids_by_generation_pos=allowed,
            atom_count_grammar=None,
            lightweight_decoding_constraints=constraints,
        )
        self.assertTrue(torch.equal(output, complete))
        block_log = logs[0][0]
        self.assertFalse(block_log["geometry_supported_before_restore"])
        self.assertTrue(block_log["restored_complete_block"])
        self.assertEqual(block_log["restored_site_count"], 1)
        self.assertEqual(block_log["new_token_ids"], block_log["previous_token_ids"])
        self.assertNotEqual(
            block_log["proposed_token_ids"], block_log["previous_token_ids"]
        )
        self.assertEqual(
            [
                site["restored_site_no_legal_z"]
                for site in block_log["site_revisions"]
            ],
            [False, True],
        )
        self.assertTrue(block_log["final_geometry_supported"])

    def test_species_block_row_local_sampling_is_batch_order_invariant(self):
        class FlatSamplingModel(self.TinyModel):
            def forward(self, token_ids, attention_mask=None):
                self.token_inputs.append(token_ids.detach().clone())
                if attention_mask is not None:
                    self.attention_masks.append(attention_mask.detach().clone())
                batch, length = token_ids.shape
                logits = torch.full(
                    (batch, length, 128),
                    -torch.inf,
                    dtype=torch.float32,
                )
                logits[..., 20:40] = 0.0
                return SimpleNamespace(logits=logits)

        model = FlatSamplingModel()
        prompt_length = 1
        gen_length = 15
        first_input = torch.stack(
            (
                torch.arange(prompt_length + gen_length) % 20,
                (torch.arange(prompt_length + gen_length) + 3) % 20,
            )
        )
        kwargs = {
            "prompt_length": prompt_length,
            "gen_length": gen_length,
            "attention_mask": torch.ones((2, prompt_length), dtype=torch.long),
            "temperature": 0.7,
            "cfg_scale": 0.0,
            "remasking": "low_confidence",
            "mask_id": 127,
            "allowed_token_ids_by_generation_pos": [
                list(range(20, 40)) for _ in range(gen_length)
            ],
            "atom_count_grammar": None,
            "lightweight_decoding_constraints": None,
        }
        first, _ = revise_spad_species_blocks(
            model,
            first_input,
            revision_blocks_by_batch=[[[1], [0]], [[1], [0]]],
            sampling_seeds_by_batch=[123, 456],
            **kwargs,
        )
        swapped, _ = revise_spad_species_blocks(
            FlatSamplingModel(),
            first_input.flip(0),
            revision_blocks_by_batch=[[[1], [0]], [[1], [0]]],
            sampling_seeds_by_batch=[456, 123],
            **kwargs,
        )
        self.assertTrue(torch.equal(first[0], swapped[1]))

    def test_cursor_continuation_replays_reference_with_global_rng_salts(self):
        class FlatSamplingModel(self.TinyModel):
            def forward(self, token_ids, attention_mask=None):
                batch, length = token_ids.shape
                logits = torch.full(
                    (batch, length, 128), -torch.inf, dtype=torch.float32
                )
                logits[..., 20:40] = 0.0
                return SimpleNamespace(logits=logits)

        prompt_length = 1
        gen_length = 19
        initial = (torch.arange(prompt_length + gen_length) % 20).reshape(1, -1)
        blocks = [[2, 1], [0]]
        kwargs = {
            "prompt_length": prompt_length,
            "gen_length": gen_length,
            "attention_mask": torch.ones((1, prompt_length), dtype=torch.long),
            "temperature": 0.7,
            "cfg_scale": 0.0,
            "remasking": "low_confidence",
            "mask_id": 127,
            "allowed_token_ids_by_generation_pos": [
                list(range(20, 40)) for _ in range(gen_length)
            ],
            "atom_count_grammar": None,
            "lightweight_decoding_constraints": None,
        }
        expected, logs = revise_spad_species_blocks(
            FlatSamplingModel(),
            initial,
            revision_blocks_by_batch=[blocks],
            sampling_seeds_by_batch=[123],
            **kwargs,
        )
        first_site = logs[0][0]["site_revisions"][0]
        state = initial.clone()
        masked_positions = [
            prompt_length + position
            for slot in blocks[0]
            for position in (8 + 4 * slot, 9 + 4 * slot, 10 + 4 * slot)
        ]
        state[0, torch.tensor(masked_positions)] = 127
        actual, report = continue_spad_species_blocks_from_cursor(
            FlatSamplingModel(),
            state,
            block_entry_tokens=initial,
            revision_blocks=blocks,
            block_index=0,
            site_order_index=0,
            action_token_ids=first_site["new_token_ids"],
            sampling_seed=123,
            **kwargs,
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(report["block_index"], 0)
        self.assertEqual(
            [row["block_index"] for row in report["later_block_revisions"]], [1]
        )
        self.assertTrue(torch.equal(first[1], swapped[0]))

    def test_species_block_mixed_radix_salts_are_unique(self):
        salts = {
            _spad_basin_closure_block_salt(block, site, component)
            for block in range(4)
            for site in range(20)
            for component in range(3)
        }
        self.assertEqual(len(salts), 4 * 20 * 3)
        self.assertEqual(_spad_basin_closure_block_salt(0, 0, 0), 0)
        self.assertNotEqual(
            _spad_basin_closure_block_salt(0, 1, 0),
            _spad_basin_closure_block_salt(1, 0, 0),
        )

    def test_response_bias_respects_kl_and_absolute_logit_budgets(self):
        config = Model494ResponseConfig(kl_budget_nats=0.05, max_abs_logit_bias=2.0)
        base = torch.tensor([-1.0, 0.2, 0.8, -0.4], dtype=torch.float64)
        gains = torch.tensor([-2.0, -0.1, 1.0, 3.0], dtype=torch.float64)
        bias, report = _kl_bounded_gain_bias(base, gains, config)
        log_p = torch.log_softmax(base, dim=0)
        log_q = torch.log_softmax(base + bias, dim=0)
        q = torch.exp(log_q)
        divergence = torch.sum(q * (log_q - log_p)).item()
        self.assertLessEqual(divergence, 0.0500001)
        self.assertLessEqual(torch.abs(bias).max().item(), 2.0000001)
        self.assertAlmostEqual(divergence, report["kl_nats"], places=8)
        self.assertGreater(torch.dot(q, gains).item(), torch.dot(torch.exp(log_p), gains).item())

    def test_periodic_alias_and_global_translation_have_identical_geometry(self):
        lattice = torch.eye(3, dtype=torch.float64) * 4.0
        zero = _minimum_image_vector(
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
            lattice,
            image_radius=2,
        )
        alias = _minimum_image_vector(
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            lattice,
            image_radius=2,
        )
        self.assertTrue(torch.allclose(zero, alias))
        source = torch.tensor(
            [[0.05, 0.10, 0.15], [0.60, 0.70, 0.80]], dtype=torch.float64
        )
        translated = source + torch.tensor([0.25, -0.20, 0.30], dtype=torch.float64)
        response = _bounded_translation_free_response(
            source,
            translated,
            lattice,
            Model494ResponseConfig(),
        )
        self.assertTrue(torch.allclose(response, torch.zeros_like(response), atol=1.0e-12))

    def test_response_path_restores_complete_xyz_when_z_has_no_legal_completion(self):
        model = self.TinyModel()
        prompt_length = 1
        gen_length = 15
        # N=2, unit cubic lattice, then two sites separated by only 0.01 A.
        suffix = torch.tensor(
            [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                20,
                30,
                40,
                50,
                20,
                31,
                41,
                51,
            ],
            dtype=torch.long,
        )
        complete = torch.cat((torch.tensor([1]), suffix)).reshape(1, -1)
        allowed = [
            [10],
            [11],
            [12],
            [13],
            [14],
            [15],
            [16],
            [20],
            [30, 31],
            [40, 41],
            [50, 51],
            [20],
            [30, 31],
            [40, 41],
            [50, 51],
        ]
        constraints = {
            "representation": "dynamic_v1",
            "body_offset": 0,
            "max_atoms": 2,
            "coord_period": 100,
            "count_token_to_n": {10: 2},
            "length_step": 0.1,
            "length_token_to_bin": {
                "LA": {11: 10},
                "LB": {12: 10},
                "LC": {13: 10},
            },
            "angle_token_to_bin": {
                "AA": {14: 90},
                "AB": {15: 90},
                "AG": {16: 90},
            },
            "coord_token_to_bin": {
                "X": {30: 0, 31: 1},
                "Y": {40: 0, 41: 1},
                "Z": {50: 0, 51: 1},
            },
            "coord_bin_to_token_id": {
                "X": {0: 30, 1: 31},
                "Y": {0: 40, 1: 41},
                "Z": {0: 50, 1: 51},
            },
            "pbc_min_distance_mask": True,
            "pbc_min_distance_A": 0.5,
            "pbc_image_radius": 2,
        }
        output, logs = revise_spad_anchors(
            model,
            complete,
            prompt_length=prompt_length,
            gen_length=gen_length,
            revision_slots_by_batch=[[1]],
            attention_mask=torch.ones((1, prompt_length), dtype=torch.long),
            temperature=0.7,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=127,
            allowed_token_ids_by_generation_pos=allowed,
            atom_count_grammar=None,
            lightweight_decoding_constraints=constraints,
            suffix_visible=True,
            model494_target_frac_coords_by_batch=[
                [[0.0, 0.0, 0.0], [0.02, 0.02, 0.02]]
            ],
        )
        self.assertTrue(torch.equal(output, complete))
        self.assertEqual(
            logs[0][0]["guidance_status"],
            "guidance_skipped_no_legal_completion",
        )
        strict_output, strict_logs = revise_spad_anchors(
            model,
            complete,
            prompt_length=prompt_length,
            gen_length=gen_length,
            revision_slots_by_batch=[[1]],
            attention_mask=torch.ones((1, prompt_length), dtype=torch.long),
            temperature=0.7,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=127,
            allowed_token_ids_by_generation_pos=allowed,
            atom_count_grammar=None,
            lightweight_decoding_constraints=constraints,
            suffix_visible=True,
            strict_pbc_no_legal_fallback=True,
        )
        self.assertTrue(torch.equal(strict_output, complete))
        self.assertEqual(
            strict_logs[0][0]["guidance_status"],
            "guidance_skipped_no_legal_completion",
        )


if __name__ == "__main__":
    unittest.main()
