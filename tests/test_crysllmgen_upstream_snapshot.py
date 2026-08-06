from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORK_ROOT = PROJECT_ROOT / "crystal_dlm" / "wqcodiff" / "crysllmgen"
SNAPSHOT_ROOT = FORK_ROOT / "upstream"
MANIFEST_PATH = FORK_ROOT / "UPSTREAM_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_hash_manifest(root: Path) -> str:
    records = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        records.append(f"{sha256(path)}  {relative}\n")
    return hashlib.sha256("".join(records).encode("utf-8")).hexdigest()


class CrysLLMGenUpstreamSnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_snapshot_identity(self) -> None:
        self.assertEqual(
            self.manifest["upstream_commit"],
            "94bb287751cd20a882c7c1df7ca736633d78e5e1",
        )
        files = [item for item in SNAPSHOT_ROOT.rglob("*") if item.is_file()]
        self.assertEqual(len(files), self.manifest["source_file_count"])
        self.assertEqual(
            relative_hash_manifest(SNAPSHOT_ROOT),
            self.manifest["relative_file_hash_manifest_sha256"],
        )

    def test_key_file_hashes(self) -> None:
        for relative, expected in self.manifest["key_files"].items():
            self.assertEqual(sha256(SNAPSHOT_ROOT / relative), expected, relative)

    def test_snapshot_is_source_only_and_licensed(self) -> None:
        forbidden_suffixes = {".pt", ".pth", ".safetensors", ".csv", ".pyc"}
        for path in SNAPSHOT_ROOT.rglob("*"):
            if path.is_file():
                self.assertNotIn(path.suffix.lower(), forbidden_suffixes, path)
        license_text = (SNAPSHOT_ROOT / "LICENSE").read_text(
            encoding="utf-8", errors="replace"
        )
        self.assertIn("MIT License", license_text)
        self.assertIn("Copyright (c) 2025 Kishalay Das", license_text)

    def test_required_upstream_interfaces_are_present(self) -> None:
        diffusion = (SNAPSHOT_ROOT / "models_ddpm" / "diffusion.py").read_text(
            encoding="utf-8", errors="replace"
        )
        cspnet = (SNAPSHOT_ROOT / "models_ddpm" / "cspnet.py").read_text(
            encoding="utf-8", errors="replace"
        )
        sampler = (SNAPSHOT_ROOT / "crysllmgen_sample.py").read_text(
            encoding="utf-8", errors="replace"
        )
        self.assertIn("class CSPDiffusion", diffusion)
        self.assertIn("def sample(self, batch", diffusion)
        self.assertIn("class CSPNet", cspnet)
        self.assertIn("def unconditional_sample", sampler)
        self.assertIn("diffusion_model.sample", sampler)


if __name__ == "__main__":
    unittest.main()
