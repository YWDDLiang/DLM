from types import SimpleNamespace
import unittest


try:
    import torch
    from crystal_dlm.llada_generation import generate
    from crystal_dlm.spad_generation import revise_spad_anchors
except ModuleNotFoundError:
    torch = None
    generate = None
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


if __name__ == "__main__":
    unittest.main()
