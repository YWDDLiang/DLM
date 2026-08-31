from pathlib import Path
import shutil,subprocess,unittest
ROOT=Path(__file__).resolve().parents[1]
SOURCE=(ROOT/'slurm/114_fused_planner_direct_only.sbatch').read_text()
class Slurm114Test(unittest.TestCase):
 def test_direct_only(self):
  self.assertIn('#SBATCH --cpus-per-task=8',SOURCE);self.assertNotIn('#SBATCH --gres',SOURCE)
  self.assertIn('source_body_job\t39096',SOURCE);self.assertIn('source_assembly_job\t39097',SOURCE)
  self.assertIn('CRYSLLMGEN_METRICS_NUM_CPUS=1',SOURCE)
  for name in ('OPENBLAS_NUM_THREADS=8','OMP_NUM_THREADS=8','MKL_NUM_THREADS=8','NUMEXPR_NUM_THREADS=8'):self.assertIn(name,SOURCE)
  self.assertNotIn('sample_sgtc_l6',SOURCE);self.assertNotIn('assemble_raw_body_repeat',SOURCE)
 def test_bash(self):
  b=shutil.which('bash')
  if b is None:self.skipTest('bash unavailable')
  subprocess.run([b,'-n','slurm/114_fused_planner_direct_only.sbatch'],cwd=ROOT,check=True,capture_output=True)
if __name__=='__main__':unittest.main()
