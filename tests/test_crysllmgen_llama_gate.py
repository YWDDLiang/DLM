from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.a800.run_crysllmgen_llama_gate_a import _verify_registered_files


class CrysLLMGenLlamaGateTests(unittest.TestCase):
    def test_registered_file_identity_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "weight.bin"
            path.write_bytes(b"registered-weight")
            record = [
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ]
            observed = _verify_registered_files(root, record)
            self.assertEqual(observed, record)
            path.write_bytes(b"mutated-weight")
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                _verify_registered_files(root, record)


if __name__ == "__main__":
    unittest.main()

