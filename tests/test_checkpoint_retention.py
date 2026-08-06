from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crystal_dlm.checkpoint_retention import (
    apply_retention_plan,
    build_retention_plan,
)


class CheckpointRetentionTest(unittest.TestCase):
    def _make_checkpoints(
        self,
        root: Path,
        steps: tuple[int, ...],
        *,
        prefix: str = "step",
    ) -> dict[int, Path]:
        result: dict[int, Path] = {}
        for step in steps:
            checkpoint = root / f"{prefix}-{step}"
            checkpoint.mkdir()
            (checkpoint / "weights.bin").write_bytes(bytes([step % 251]) * 4096)
            result[step] = checkpoint
        return result

    def test_keeps_older_best_plus_latest_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._make_checkpoints(root, (50, 100, 150, 200))
            plan = build_retention_plan(root, paths[50])
            self.assertEqual([entry.step for entry in plan.keep], [50, 150, 200])
            self.assertEqual([entry.step for entry in plan.delete], [100])

    def test_best_overlapping_latest_two_keeps_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._make_checkpoints(
                root, (100, 200, 300), prefix="checkpoint"
            )
            plan = build_retention_plan(root, paths[300])
            self.assertEqual([entry.step for entry in plan.keep], [200, 300])
            self.assertEqual([entry.step for entry in plan.delete], [100])

    def test_dry_run_does_not_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._make_checkpoints(root, (1, 2, 3, 4))
            plan = build_retention_plan(root, paths[1])
            self.assertTrue(paths[2].is_dir())
            self.assertEqual(len(plan.delete), 1)

    def test_apply_deletes_only_planned_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._make_checkpoints(root, (1, 2, 3, 4))
            unrelated = root / "training_report.json"
            unrelated.write_text("{}", encoding="utf-8")
            plan = build_retention_plan(root, paths[1])
            deleted = apply_retention_plan(plan)
            self.assertEqual(deleted, (str(paths[2].resolve()),))
            self.assertFalse(paths[2].exists())
            self.assertTrue(paths[1].is_dir())
            self.assertTrue(paths[3].is_dir())
            self.assertTrue(paths[4].is_dir())
            self.assertTrue(unrelated.is_file())

    def test_rejects_best_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "checkpoints"
            root.mkdir()
            self._make_checkpoints(root, (1, 2))
            outside = base / "step-3"
            outside.mkdir()
            with self.assertRaisesRegex(ValueError, "direct child"):
                build_retention_plan(root, outside)

    def test_ignores_unselected_symlink_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "checkpoints"
            root.mkdir()
            paths = self._make_checkpoints(root, (1, 2))
            (root / "step-3").symlink_to(paths[2], target_is_directory=True)
            plan = build_retention_plan(root, paths[1])
            self.assertEqual([entry.step for entry in plan.keep], [1, 2])

    def test_rejects_symlink_as_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "checkpoints"
            root.mkdir()
            paths = self._make_checkpoints(root, (1, 2))
            best_link = root / "step-3"
            best_link.symlink_to(paths[2], target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "may not be a symlink"):
                build_retention_plan(root, best_link)


if __name__ == "__main__":
    unittest.main()
