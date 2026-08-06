import hashlib
from pathlib import Path
import tempfile
import unittest

from crystal_dlm.h1_integrity import (
    H1IntegrityError,
    verify_sha256_manifest,
)


class H1IntegrityTests(unittest.TestCase):
    def test_manifest_verification_detects_success_and_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.txt"
            payload.write_text("frozen\n", encoding="utf-8")
            expected = hashlib.sha256(payload.read_bytes()).hexdigest()
            manifest = root / "SHA256.txt"
            manifest.write_text(
                f"{expected}  payload.txt\n",
                encoding="utf-8",
            )
            valid = verify_sha256_manifest(manifest)
            self.assertTrue(valid["ok"])
            self.assertEqual(valid["entry_count"], 1)

            payload.write_text("changed\n", encoding="utf-8")
            changed = verify_sha256_manifest(manifest)
            self.assertFalse(changed["ok"])
            self.assertNotEqual(
                changed["entries"][0]["actual_sha256"],
                expected,
            )

    def test_manifest_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "SHA256.txt"
            manifest.write_text(
                f"{'0' * 64}  ../outside.txt\n",
                encoding="utf-8",
            )
            with self.assertRaises(H1IntegrityError):
                verify_sha256_manifest(manifest)

    def test_manifest_allows_parent_reference_inside_explicit_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = root / "manifests"
            payloads = root / "payloads"
            manifests.mkdir()
            payloads.mkdir()
            payload = payloads / "frozen.txt"
            payload.write_text("frozen\n", encoding="utf-8")
            expected = hashlib.sha256(payload.read_bytes()).hexdigest()
            manifest = manifests / "SHA256.txt"
            manifest.write_text(
                f"{expected}  ../payloads/frozen.txt\n",
                encoding="utf-8",
            )
            report = verify_sha256_manifest(
                manifest,
                allowed_root=root,
            )
            self.assertTrue(report["ok"])


if __name__ == "__main__":
    unittest.main()
