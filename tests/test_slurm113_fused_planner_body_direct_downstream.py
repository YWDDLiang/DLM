from pathlib import Path
import shutil
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=(ROOT/'slurm/113_fused_planner_body_direct_downstream.sbatch').read_text()

class Slurm113Test(unittest.TestCase):
    def test_cpu_downstream_only(self):
        self.assertIn('#SBATCH --cpus-per-task=8',SOURCE)
        self.assertNotIn('#SBATCH --gres',SOURCE)
        self.assertIn('source_body_job\t39096',SOURCE)
        self.assertIn('body_rerun\tfalse',SOURCE)
        self.assertNotIn('sample_sgtc_l6.py',SOURCE)
    def test_only_assembly_and_direct(self):
        self.assertIn('assemble_raw_body_repeat.py',SOURCE)
        self.assertIn('run_crysllmgen_metrics.py',SOURCE)
        self.assertIn('model494\tfalse',SOURCE)
        self.assertIn('chgnet\tfalse',SOURCE)
        self.assertNotIn('run_full_reconstructed_eval',SOURCE)
    def test_bash(self):
        bash=shutil.which('bash')
        if bash is None:self.skipTest('bash unavailable')
        subprocess.run([bash,'-n','slurm/113_fused_planner_body_direct_downstream.sbatch'],cwd=ROOT,check=True,capture_output=True)

if __name__=='__main__':unittest.main()
