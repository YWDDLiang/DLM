from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1] / "slurm/179_spad_energy_train.sbatch"
).read_text(encoding="utf-8")


class SPADEnergyTrainSlurmTest(unittest.TestCase):
    def test_equal_compute_control_and_energy_cells(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2", SOURCE)
        self.assertIn('run_cell control "${allocated[0]}"', SOURCE)
        self.assertIn('run_cell energy "${allocated[1]}"', SOURCE)
        self.assertIn("--seed 98017", SOURCE)
        self.assertIn('final["updates"] == 348', SOURCE)
        self.assertIn('"same_seed_data_updates": True', SOURCE)

    def test_no_inference_critic_or_selection(self):
        self.assertNotIn("query_official", SOURCE)
        self.assertNotIn("--rerank", SOURCE)
        self.assertIn('"validity_before_energy": True', SOURCE)
        self.assertIn('"inference_time_critic": False', SOURCE)


if __name__ == "__main__":
    unittest.main()
