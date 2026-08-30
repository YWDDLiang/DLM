from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "slurm" / "82_freeze_faithful_rich_diagnostic.sbatch"


class Slurm82FaithfulRichDiagnosticTest(unittest.TestCase):
    def test_contract_is_cpu_only_outcome_blind_and_immutable(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn("outcomes_read=false", text)
        self.assertIn("official_query=false", text)
        self.assertIn('[[ ! -e "${OUTPUT}" ]]', text)
        self.assertIn("R0S.jsonl", text)
        self.assertIn("H0.jsonl", text)
        self.assertIn("--count 256", text)
        self.assertIn("oxidation_candidates,prototype_key", text)


if __name__ == "__main__":
    unittest.main()
