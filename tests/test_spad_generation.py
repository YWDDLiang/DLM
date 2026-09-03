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
        revise_spad_anchors,
    )
except ModuleNotFoundError:
    torch = None
    generate = None
    Model494ResponseConfig = None
    revise_spad_anchors = None


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

        def get_output_embeddings(self):
            return self.output

        def forward(self, token_ids, attention_mask=None):
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


if __name__ == "__main__":
    unittest.main()
