from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "slurm/177_spad_response_fast_eval.sbatch"
).read_text(encoding="utf-8")


class SPADResponseFastEvalSlurmTest(unittest.TestCase):
    def test_two_gpu_two_stream_raw_refined_chgnet(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2", SOURCE)
        self.assertIn("run_stream 17", SOURCE)
        self.assertIn("run_stream 18", SOURCE)
        self.assertIn('for endpoint in ("raw", "refined")', SOURCE)
        self.assertIn("run_full_reconstructed_eval.py", SOURCE)

    def test_direct_and_official_are_explicitly_deferred(self):
        self.assertIn("DEFERRED_COST", SOURCE)
        self.assertNotIn("run_crysllmgen_metrics.py", SOURCE)
        self.assertNotIn("query_official_mp.py", SOURCE)
        self.assertIn('"official_query": False', SOURCE)


if __name__ == "__main__":
    unittest.main()
