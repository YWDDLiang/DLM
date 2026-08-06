import json
from pathlib import Path
import tempfile
import unittest

from crystal_dlm.h1_readonly_guard import (
    H1_RESERVED_RELATIVE_ROOTS,
    H1ReadOnlyViolation,
    assert_writable_output_path,
    frozen_h1_match,
)


class H1ReadOnlyGuardTests(unittest.TestCase):
    def test_exact_frozen_root_and_descendant_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = H1_RESERVED_RELATIVE_ROOTS[0]
            with self.assertRaises(H1ReadOnlyViolation):
                assert_writable_output_path(frozen, project_root=root)
            with self.assertRaises(H1ReadOnlyViolation):
                assert_writable_output_path(
                    Path(frozen) / "new_checkpoint",
                    project_root=root,
                )

    def test_remote_checkout_path_is_rejected_by_component_identity(self):
        candidate = (
            "/public/home/user/project/"
            "runs/20260729_h1a2c_jointchem_v1/arms/P0/report.json"
        )
        self.assertEqual(
            frozen_h1_match(candidate, project_root="/tmp/local-project"),
            "runs/20260729_h1a2c_jointchem_v1",
        )

    def test_new_plangraph_run_and_similar_substring_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new_run = assert_writable_output_path(
                "runs/20260801_plangraph_dlm_v1",
                project_root=root,
            )
            similar = assert_writable_output_path(
                "runs/archive_20260729_h1a2c_jointchem_v1_copy",
                project_root=root,
            )
            self.assertEqual(new_run, root / "runs/20260801_plangraph_dlm_v1")
            self.assertEqual(
                similar,
                root / "runs/archive_20260729_h1a2c_jointchem_v1_copy",
            )

    def test_parent_traversal_cannot_bypass_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = (
                "scratch/../runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1/"
                "replacement.json"
            )
            with self.assertRaises(H1ReadOnlyViolation):
                assert_writable_output_path(candidate, project_root=root)

    def test_existing_symlink_into_frozen_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / H1_RESERVED_RELATIVE_ROOTS[0]
            frozen.mkdir(parents=True)
            link = root / "apparently_new"
            try:
                link.symlink_to(frozen, target_is_directory=True)
            except (
                OSError
            ) as exc:  # pragma: no cover - platforms without symlink support.
                self.skipTest(f"symlink unavailable: {exc}")
            with self.assertRaises(H1ReadOnlyViolation):
                assert_writable_output_path(link / "output.json", project_root=root)

    def test_registry_read_only_roots_match_guard(self):
        root = Path(__file__).resolve().parents[1]
        registry_path = (
            root
            / "workstreams"
            / "plangraph_dlm_iclr_20260731"
            / "EXPERIMENT_REGISTRY_V1.json"
        )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(registry["h1_fallback"]["read_only_roots"]),
            set(H1_RESERVED_RELATIVE_ROOTS),
        )


if __name__ == "__main__":
    unittest.main()
