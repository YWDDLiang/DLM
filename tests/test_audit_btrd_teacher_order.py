from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_btrd_teacher_order.py"


class BtrdTeacherOrderAuditTest(unittest.TestCase):
    def test_passes_exact_order_and_rejects_permutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal = root / "proposal.pt"
            refined = root / "refined.pt"
            accounting = root / "accounting.jsonl"
            torch.save(
                [
                    {
                        "btrd_index": 0,
                        "n_atom": torch.tensor([2]),
                        "a_type": torch.tensor([3, 8]),
                    }
                ],
                proposal,
            )
            accounting.write_text(
                json.dumps({"btrd_index": 0, "parsed": True}) + "\n"
                + json.dumps({"btrd_index": 1, "parsed": False}) + "\n"
            )
            payload = {
                "sample_indices": torch.tensor([0]),
                "num_atoms": torch.tensor([[2]]),
                "atom_types": torch.tensor([[3, 8]]),
            }
            torch.save(payload, refined)
            output = root / "pass"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--proposal-graphs",
                    str(proposal),
                    "--refined-pt",
                    str(refined),
                    "--accounting-jsonl",
                    str(accounting),
                    "--output-dir",
                    str(output),
                    "--expected-requested",
                    "2",
                ],
                check=True,
            )
            self.assertTrue((output / "_SUCCESS").is_file())

            payload["atom_types"] = torch.tensor([[8, 3]])
            torch.save(payload, refined)
            failed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--proposal-graphs",
                    str(proposal),
                    "--refined-pt",
                    str(refined),
                    "--accounting-jsonl",
                    str(accounting),
                    "--output-dir",
                    str(root / "fail"),
                    "--expected-requested",
                    "2",
                ],
                check=False,
            )
            self.assertNotEqual(failed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
