import unittest
import torch

from crystal_dlm.programmed_path_reference import close_reference
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints
from test_state_programmed_runtime import TinyTokenizer, PreferredModel, body, constraints, program


class ReferenceClosureTest(unittest.TestCase):
    def test_reference_really_executes_cell_before_species(self):
        tok = TinyTokenizer()
        source, target = body(tok, length=20, x=30), body(tok, length=30, x=40)
        model = PreferredModel(tok, target)
        x = torch.tensor([[0] + source])
        result, logs = close_reference(model, x, [{"success": True}], programs=[program()], seeds=[21],
            prompt_length=1, attention_mask=torch.ones_like(x), allowed=exact_dynamic_schema_constraints(tok, 2),
            constraints=constraints(tok), temperature=0., mask_id=tok.mask_id)
        self.assertEqual(result[0, 1:].tolist(), target)
        self.assertEqual(logs[0]["stage_order"], ["predictor", "cell", "reverse_species_blocks"])
        self.assertEqual(logs[0]["cell"]["changed_components"], 3)

    def test_reference_retains_the_separate_cell_rejection_boundary(self):
        tok = TinyTokenizer()
        source, target = body(tok, length=20, x=30), body(tok, length=10, x=50)
        x = torch.tensor([[0] + source])
        result, logs = close_reference(PreferredModel(tok, target), x, [{"success": True}],
            programs=[program()], seeds=[21], prompt_length=1, attention_mask=torch.ones_like(x),
            allowed=exact_dynamic_schema_constraints(tok, 2), constraints=constraints(tok), temperature=0., mask_id=tok.mask_id)
        self.assertEqual(result[0, 2:8].tolist(), source[1:7])
        self.assertTrue(logs[0]["cell"]["restored_complete_noop"])
        self.assertTrue(logs[0]["common_final_support"])


if __name__ == "__main__":
    unittest.main()
