from pathlib import Path
import hashlib
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "slurm/111_sample_c3fd_llama_typed_planner.sbatch").read_text(
    encoding="utf-8"
)


class Slurm111StaticTest(unittest.TestCase):
    def test_resources_and_fixed_sampling_contract(self):
        self.assertIn("#SBATCH --cpus-per-task=8", SOURCE)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", SOURCE)
        self.assertIn("sampling_seed\t21", SOURCE)
        self.assertIn("requested\t256", SOURCE)
        self.assertIn("--seed 21", SOURCE)
        self.assertIn("--temperature 0.9 --top-p 0.95 --top-k 0", SOURCE)

    def test_inputs_and_frozen_source_are_hash_pinned(self):
        self.assertIn("C3FD_LLAMA_TYPED_SOURCE_LEDGER_SHA256", SOURCE)
        self.assertIn("C3FD_LLAMA_TYPED_CONFIG_SHA256", SOURCE)
        self.assertIn("C3FD_LLAMA_TYPED_STATE_SHA256", SOURCE)
        self.assertIn("C3FD_LLAMA_TYPED_ADAPTER_MODEL_SHA256", SOURCE)
        self.assertIn("INPUTS.sha256", SOURCE)
        self.assertIn('$(dirname "${SOURCE_LEDGER}")/_SUCCESS', SOURCE)
        match = re.search(r"^readonly SAMPLER_SHA=([0-9a-f]{64})$", SOURCE, re.M)
        self.assertIsNotNone(match)
        observed = hashlib.sha256(
            (ROOT / "src/scripts/sample_c3fd_llama_typed_planner.py").read_bytes()
        ).hexdigest()
        self.assertEqual(match.group(1), observed)
        self.assertNotIn("TO_PIN", SOURCE)

    def test_one_trajectory_no_selection_and_comp_valid_gate(self):
        self.assertIn("trajectory_per_ordinal\t1", SOURCE)
        self.assertIn("retry_filter_replacement_rerank_best_of_n\tfalse", SOURCE)
        self.assertIn("--minimum-comp-valid 0.95", SOURCE)
        self.assertIn('metrics["comp_valid_rate_requested_denominator"] >= 0.95', SOURCE)
        self.assertNotIn("official", SOURCE.lower())

    def test_final_output_is_immutable_and_complete(self):
        self.assertIn('mkdir "${RUN}"', SOURCE)
        self.assertIn('[[ -f "${OUTPUT}/_SUCCESS" ]]', SOURCE)
        self.assertIn('wc -l <"${OUTPUT}/raw_generations.jsonl"', SOURCE)
        self.assertIn('touch "${RUN}/_SUCCESS"', SOURCE)


if __name__ == "__main__":
    unittest.main()
