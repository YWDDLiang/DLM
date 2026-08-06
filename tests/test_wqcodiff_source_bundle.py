from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/a800/build_source_bundle.py"
INSTALLER = ROOT / "scripts/a800/install_source_bundle.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceBundleTests(unittest.TestCase):
    def test_bundle_is_deterministic_and_installer_preserves_remote_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            target = root / "remote"
            first.mkdir()
            second.mkdir()
            (target / "data").mkdir(parents=True)
            preserved = target / "data/keep.txt"
            preserved.write_text("server-owned\n", encoding="utf-8")
            for output in (first, second):
                subprocess.run(
                    [
                        sys.executable,
                        str(BUILDER),
                        "--root",
                        str(ROOT),
                        "--output-dir",
                        str(output),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            bundle = first / "wqcodiff_source.tar.gz"
            self.assertEqual(_sha(bundle), _sha(second / bundle.name))
            subprocess.run(
                [
                    sys.executable,
                    str(INSTALLER),
                    "--bundle",
                    str(bundle),
                    "--expected-sha256",
                    _sha(bundle),
                    "--target",
                    str(target),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(preserved.read_text(encoding="utf-8"), "server-owned\n")
            self.assertTrue((target / "crystal_dlm/wqcodiff/model.py").is_file())
            self.assertTrue((target / "scripts/run_mattergen_sun_eval.py").is_file())
            self.assertTrue((target / ".artifacts/source_sync" / f"{_sha(bundle)}.json").is_file())
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(target)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import crystal_dlm; import crystal_dlm.wqcodiff",
                ],
                cwd=target,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()
