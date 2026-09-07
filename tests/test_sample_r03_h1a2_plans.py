"""Native Planner sampling contracts with CPU fixture outputs only."""
from pathlib import Path
import tempfile
import json
from types import SimpleNamespace
import unittest

from crystal_dlm.r03_formula_bridge import C3FDFormulaOracle
from scripts.sample_r03_h1a2_plans import (
    NativePlanEndStoppingCriteria, SAMPLING_DEFAULTS, decode_record,
    generation_arguments, read_domain, sample_batch, summarize,
)


VALID = "\n".join(("formula: NaCl", "anion: halide", "charge: neutral_plausible",
                   "lattice: cubic", "spacegroup: sg_195_230", "volume: volpa_015_019", "end: plan"))
MISSING = VALID.replace("volume: volpa_015_019\n", "")
INVALID_CHEMISTRY = VALID.replace("formula: NaCl", "formula: Na2Cl")


class FixtureTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self):
        self.fragments = {0: "", 1: "PROMPT", 10: VALID, 11: MISSING, 12: INVALID_CHEMISTRY,
                          13: "\ntrailing original generation", 14: "end: plan", 15: "formula: Na",
                          16: VALID.replace("end: plan", "formula: Na2Cl\nend: plan")}
        self.prompts = []

    def decode(self, ids, **kwargs):
        return "".join(self.fragments[int(value)] for value in ids)

    def batch_decode(self, ids, **kwargs):
        return [self.decode(row, **kwargs) for row in ids]

    def __call__(self, texts, **kwargs):
        import torch
        self.prompts.extend(texts)
        return {"input_ids": torch.ones(len(texts), 2, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 2, dtype=torch.long)}


def tiny_oracle():
    return C3FDFormulaOracle(nodes={"Na": [0, 1], "Cl": [-1]},
                            electronegativities={"Na": 0.93, "Cl": 3.16}, metal_symbols=["Na"],
                            max_atoms=20, max_species=7, allowed_strata=[(2, 2, "halide")])


class RichPlanLedgerTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FixtureTokenizer()

    def record(self, token, *, oracle=None, processor=None):
        return decode_record(tokenizer=self.tokenizer, prompt_ids=[1, 1], generated_ids=[token, 0],
                             sample_idx=502, local_idx=2, seed=29, oracle=oracle, processor=processor)

    def test_control_preserves_native_fields_tokens_and_global_ids(self):
        row = self.record(10)
        self.assertTrue(row["parsed"])
        self.assertEqual(row["sample_idx"], 502)
        self.assertEqual(row["local_sample_idx"], 2)
        self.assertEqual(row["generated_token_ids"], [10, 0])
        self.assertEqual(row["prompt_input_ids"], [1, 1])
        self.assertEqual(row["raw_model_text"], VALID)
        self.assertEqual(row["body_prompt"], row["prompt"])
        self.assertIn('"charge_bucket":"neutral_plausible"', row["body_prompt"].replace(" ", ""))
        self.assertFalse(row["include_sample_id"])
        self.assertIsNone(row["c3fd_terminal_valid"])

    def test_terminal_oracle_is_checked_independently_after_generation(self):
        valid = self.record(10, oracle=tiny_oracle())
        invalid = self.record(12, oracle=tiny_oracle())
        self.assertTrue(valid["c3fd_terminal_valid"])
        self.assertTrue(valid["body_eligible"])
        self.assertTrue(invalid["plan_parse_success"])
        self.assertFalse(invalid["parsed"])
        self.assertFalse(invalid["body_eligible"])
        self.assertEqual(invalid["attempt_status"], "c3fd_terminal_failure")
        self.assertEqual(invalid["c3fd_failure_reason"], "terminal_formula_outside_declared_support")

    def test_missing_rich_field_is_failure_without_invented_condition(self):
        row = self.record(11, oracle=tiny_oracle())
        self.assertTrue(row["c3fd_terminal_valid"])
        self.assertFalse(row["parsed"])
        self.assertEqual(row["attempt_status"], "planner_parse_failure")
        self.assertNotIn("plan_state", row)
        self.assertIn("volume", row["message"])

    def test_later_unmasked_duplicate_formula_cannot_bypass_terminal_certificate(self):
        row = self.record(16, oracle=tiny_oracle())
        self.assertTrue(row["plan_parse_success"])
        self.assertEqual(row["c3fd_checked_formula"], "NaCl")
        self.assertFalse(row["c3fd_parsed_formula_terminal_valid"])
        self.assertEqual(row["c3fd_failure_reason"], "parsed_formula_outside_declared_support")
        self.assertFalse(row["body_eligible"])

    def test_original_tail_and_actual_generated_tokens_remain_auditable(self):
        row = decode_record(tokenizer=self.tokenizer, prompt_ids=[1, 1], generated_ids=[10, 13, 0],
                            sample_idx=0, local_idx=0, seed=17, oracle=None)
        self.assertIn("trailing original generation", row["decoded_continuation"])
        self.assertIn("trailing original generation", row["raw_model_text"])
        self.assertNotIn("trailing original generation", row["raw_plan_text"])
        self.assertEqual(row["generated_token_ids"], [10, 13, 0])
        self.assertTrue(row["parsed"])

    def test_failed_rich_and_chemical_rows_remain_in_request_denominator(self):
        rows = [self.record(10, oracle=tiny_oracle()), self.record(11, oracle=tiny_oracle()),
                self.record(12, oracle=tiny_oracle())]
        metrics = summarize(rows, requested=3, elapsed=1)
        self.assertEqual(metrics["requested_samples"], 3)
        self.assertEqual(metrics["completed_request_rows"], 3)
        self.assertEqual(metrics["body_eligible"], 1)
        self.assertEqual(metrics["plan_parse_rate"], 1 / 3)
        self.assertFalse(metrics["replacement_sampling"])

    def test_domain_loading_uses_declared_strata_and_alias_keys(self):
        domain = {"nodes": {"Na": [0, 1], "Cl": [-1]}, "eneg": {"Na": 0.93, "Cl": 3.16},
                  "metals": ["Na"], "max": 20, "max_species": 7, "allowed_strata": [[2, 2, "halide"]]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "domain.json"
            path.write_text(json.dumps(domain), encoding="utf-8")
            oracle, identity = read_domain(path)
        self.assertTrue(oracle.is_terminal_valid("NaCl"))
        self.assertFalse(oracle.is_terminal_valid("Na2Cl"))
        self.assertEqual(identity["declared_strata"], 1)


try:
    import torch
except ModuleNotFoundError:
    torch = None


if torch is not None:
    class FixturePlanner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            continuation = torch.tensor([[10, 0], [11, 0]])[:kwargs["input_ids"].shape[0]]
            return torch.cat((kwargs["input_ids"], continuation), dim=1)


@unittest.skipIf(torch is None, "CPU torch required")
class NativeSamplingCallTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FixtureTokenizer()
        self.containers = SimpleNamespace(LogitsProcessorList=list, StoppingCriteriaList=list)

    def test_control_kwargs_preserve_defaults_eos_and_have_no_formula_processor(self):
        ids = torch.ones(2, 2, dtype=torch.long)
        kwargs = generation_arguments(self.tokenizer, ids, ids, containers=self.containers)
        self.assertTrue(all(kwargs[key] == value for key, value in SAMPLING_DEFAULTS.items()))
        self.assertEqual(kwargs["pad_token_id"], 0)
        self.assertEqual(kwargs["eos_token_id"], 0)
        self.assertNotIn("logits_processor", kwargs)
        self.assertIsInstance(kwargs["stopping_criteria"][0], NativePlanEndStoppingCriteria)

    def test_old_end_marker_stops_only_after_every_batch_row_has_marker(self):
        criteria = NativePlanEndStoppingCriteria(self.tokenizer, 2)
        self.assertFalse(criteria(torch.tensor([[1, 1], [1, 1]]), None))
        self.assertFalse(criteria(torch.tensor([[1, 1, 14], [1, 1, 15]]), None))
        self.assertTrue(criteria(torch.tensor([[1, 1, 14], [1, 1, 10]]), None))

    def test_one_batch_returns_success_and_failure_without_replacement(self):
        model = FixturePlanner()
        rows = sample_batch(model, self.tokenizer, native_prompt="NATIVE EXACT STRING\n",
                            global_ids=[501, 502], local_ids=[1, 2], seed=71,
                            oracle=None, containers=self.containers)
        self.assertEqual([row["sample_idx"] for row in rows], [501, 502])
        self.assertEqual([row["parsed"] for row in rows], [True, False])
        self.assertEqual(self.tokenizer.prompts, ["NATIVE EXACT STRING\n", "NATIVE EXACT STRING\n"])
        self.assertNotIn("logits_processor", model.kwargs)

    def test_method_attaches_one_processor_and_checks_its_prompt_boundary(self):
        model = FixturePlanner()
        processor = SimpleNamespace(start_length=2, failures={})
        sample_batch(model, self.tokenizer, native_prompt="NATIVE", global_ids=[1], local_ids=[0], seed=1,
                     oracle=tiny_oracle(), processor=processor, containers=self.containers)
        self.assertEqual(model.kwargs["logits_processor"], [processor])
        with self.assertRaisesRegex(ValueError, "boundary"):
            generation_arguments(self.tokenizer, torch.ones(1, 3), torch.ones(1, 3),
                                 processor=processor, containers=self.containers)


if __name__ == "__main__":
    unittest.main()
