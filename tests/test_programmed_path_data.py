import unittest

from crystal_dlm.fixed_slot import build_special_tokens
from crystal_dlm.programmed_path_data import compile_condition, path_seed, trace_terminal_body, validate_completed_body


class Tokenizer:
    def __init__(self):
        self.vocab = {token: i + 1 for i, token in enumerate(build_special_tokens())}
    def get_vocab(self):
        return self.vocab
    def __call__(self, prompt, **kwargs):
        self.prompt = prompt
        return {"input_ids": [0]}


class PathDataTest(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.row = {"group_id": "23", "source_split": "train", "prompt": "native:\n",
                    "plan_state": {"N": 3, "elements": ["Fe", "O"], "counts": [1, 2]},
                    "species_program": ["Fe", "O"], "species_program_source": "test_pointer"}

    def test_compiler_prefills_canonical_slots_and_keeps_prompt_newline(self):
        compiled = compile_condition(self.row, self.tok, mask_id=99999)
        self.assertEqual(self.tok.prompt, "native:\n")
        self.assertEqual(compiled["initial_body"][7::4], [self.tok.vocab["<E_O>"]] * 2 + [self.tok.vocab["<E_Fe>"]])
        self.assertEqual(compiled["program"].element_order, ("Fe", "O"))
        self.assertEqual(len(compiled["initial_body"]), 19)

    def test_heldout_and_invalid_permutation_rejected(self):
        with self.assertRaises(ValueError):
            compile_condition(dict(self.row, source_split="test"), self.tok, mask_id=99999)
        with self.assertRaises(ValueError):
            compile_condition(dict(self.row, species_program=["O", "O"]), self.tok, mask_id=99999)

    def test_completed_body_must_match_canonical_identity(self):
        c = compile_condition(self.row, self.tok, mask_id=99999)
        cell = "<N_003><LA_040><LB_040><LC_040><AA_090><AB_090><AG_090>"
        sites = [f"<E_{s}><X_{v:03d}><Y_000><Z_000>" for s, v in zip(("O", "O", "Fe"), (0, 30, 60))]
        validate_completed_body(cell + "".join(sites), c)
        with self.assertRaises(ValueError):
            validate_completed_body(cell + "".join(reversed(sites)), c)

    def test_occurrence_seeds_do_not_depend_on_shard_or_packing(self):
        seeds = [path_seed(23, "c", r, j) for r in range(2) for j in range(4)]
        self.assertEqual(len(set(seeds)), 8)
        self.assertEqual(seeds[2], path_seed(23, "c", 0, 2))

    def test_final_rollback_reconstructs_old_body_without_extra_draw(self):
        trace = {"initial_body": [1, 2, 3], "mask_id": 9, "events": [
            {"op": "begin", "positions": [1, 2]}, {"op": "draw", "position": 1, "token": 8},
            {"op": "no_support", "position": 2}, {"op": "rollback", "positions": [1, 2]}, {"op": "end"}]}
        self.assertEqual(trace_terminal_body(trace), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
