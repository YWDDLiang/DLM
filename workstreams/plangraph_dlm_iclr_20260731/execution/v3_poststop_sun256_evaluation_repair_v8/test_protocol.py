from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from protocol import sha256_file, verify_frozen_arm  # noqa: E402


class FrozenV7EvidenceTests(unittest.TestCase):
    def _write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _identity(self, path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    def _fixture(self, root: Path) -> tuple[Path, list[dict], list[dict]]:
        method = "fixture-method"
        generation = []
        direct = []
        for ordinal in range(256):
            attempt_id = f"fixture:{ordinal:04d}:M00"
            succeeded = ordinal < 2
            generation.append(
                {
                    "attempt_id": attempt_id,
                    "ordinal": ordinal,
                    "method": method,
                    "status": "succeeded" if succeeded else "failed",
                    "reason": "" if succeeded else "planner",
                    "diffusion_refinement_applied": succeeded,
                    "diffusion_refinement_steps": 800 if succeeded else 0,
                    "retry_or_replacement_used": False,
                }
            )
            direct.append(
                {
                    "schema": "crysllmgen_metric_attempt_v1",
                    "attempt_id": attempt_id,
                    "method": method,
                    "comp_valid": ordinal == 0,
                    "struct_valid": succeeded,
                    "valid": ordinal == 0,
                }
            )
        generation_path = root / "generation.jsonl"
        generation_report_path = root / "generation_report.json"
        generation_success_path = root / "_SUCCESS"
        direct_path = root / "attempt_metrics.jsonl"
        direct_report_path = root / "report.json"
        self._write_jsonl(generation_path, generation)
        self._write_json(
            generation_report_path,
            {
                "ok": True,
                "all_successes_diffusion_refined": True,
                "diffusion_steps": 800,
            },
        )
        generation_success_path.write_bytes(b"")
        self._write_jsonl(direct_path, direct)
        self._write_json(
            direct_report_path,
            {
                "ok": True,
                "attempts": 256,
                "denominator": "all_generation_attempts",
                "method": method,
                "comp_valid_count": 1,
                "struct_valid_count": 2,
                "valid_count": 1,
            },
        )
        manifest = {
            "schema": "h1a2_v3_poststop_sun256_evaluation_repair_input_v1",
            "status": "frozen",
            "attempts_per_arm": 256,
            "denominator": "raw_all_attempt",
            "retry": False,
            "replacement": False,
            "repair": False,
            "filter": False,
            "rerank": False,
            "arms": {
                "M00": {
                    "method": method,
                    "generation_jsonl": self._identity(generation_path),
                    "generation_report": self._identity(generation_report_path),
                    "generation_success": self._identity(generation_success_path),
                    "direct_attempt_metrics": self._identity(direct_path),
                    "direct_report": self._identity(direct_report_path),
                    "expected": {
                        "generation_succeeded": 2,
                        "composition_valid": 1,
                        "structure_valid": 2,
                        "joint_valid": 1,
                    },
                }
            },
        }
        manifest_path = root / "input_manifest.json"
        self._write_json(manifest_path, manifest)
        return manifest_path, generation, direct

    def _refresh_identity(
        self, manifest_path: Path, field: str, path: Path
    ) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["arms"]["M00"][field] = self._identity(path)
        self._write_json(manifest_path, manifest)

    def test_exact_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _ = self._fixture(Path(temporary))
            evidence = verify_frozen_arm(manifest, "M00")
            self.assertEqual(evidence["counts"]["generation_succeeded"], 2)
            self.assertEqual(evidence["counts"]["joint_valid"], 1)

    def test_unrefined_success_fails_even_when_hash_is_refrozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, generation, _ = self._fixture(root)
            generation[0]["diffusion_refinement_applied"] = False
            self._write_jsonl(root / "generation.jsonl", generation)
            self._refresh_identity(
                manifest, "generation_jsonl", root / "generation.jsonl"
            )
            with self.assertRaisesRegex(ValueError, "generation/refine800"):
                verify_frozen_arm(manifest, "M00")

    def test_reordered_direct_attempts_fail_when_hash_is_refrozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, direct = self._fixture(root)
            direct[0], direct[1] = direct[1], direct[0]
            self._write_jsonl(root / "attempt_metrics.jsonl", direct)
            self._refresh_identity(
                manifest, "direct_attempt_metrics", root / "attempt_metrics.jsonl"
            )
            with self.assertRaisesRegex(ValueError, "direct attempt mapping"):
                verify_frozen_arm(manifest, "M00")


if __name__ == "__main__":
    unittest.main()
