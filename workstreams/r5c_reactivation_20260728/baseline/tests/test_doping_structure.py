import unittest
from pathlib import Path

from crystal_dlm.doping import read_jsonl
from crystal_dlm.doping_structure import (
    bsite_dopants_from_arrays,
    compress_full80_arrays,
    expand_structure20_arrays,
    full80_composition_is_exact,
    load_full80_template,
    parse_full80_answer,
    parse_structure20_answer,
)


class DopingStructureTests(unittest.TestCase):
    def test_structure20_round_trip_expands_to_full80(self):
        rows = read_jsonl(Path("data/doping_crystal/full80_success.jsonl"))
        full80 = parse_full80_answer(rows[0]["answer"], strict=True)
        compressed = compress_full80_arrays(full80)
        parsed = parse_structure20_answer(compressed["answer"], strict=True)
        template = load_full80_template(Path("data/doping_crystal/full80_success.jsonl"))
        expanded = expand_structure20_arrays(parsed, template)
        self.assertEqual(parsed["num_atoms"], 20)
        self.assertEqual(len(parsed["tokens"]), 107)
        self.assertEqual(expanded["num_atoms"], 80)
        self.assertTrue(full80_composition_is_exact(expanded))
        self.assertEqual(sorted(bsite_dopants_from_arrays(expanded)), sorted(rows[0]["metadata"]["dopants"]))

    def test_full80_answer_is_407_tokens(self):
        rows = read_jsonl(Path("data/doping_full80_holdout/train.jsonl"))
        parsed = parse_full80_answer(rows[0]["answer"], strict=True)
        self.assertEqual(parsed["num_atoms"], 80)
        self.assertEqual(len(parsed["tokens"]), 407)
        self.assertTrue(full80_composition_is_exact(parsed))


if __name__ == "__main__":
    unittest.main()
