from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from crystal_dlm.wqcodiff.dependency_waiver import (
    AUTHORIZATION,
    CHGNET_WAIVER_VALIDATION,
    CHGNET_NON_TORCH_REQUIREMENTS,
    MATTERSIM_EXCLUDED_NONINFERENCE_DISTRIBUTIONS,
    MATTERSIM_COMPATIBILITY_PINS,
    MATTERSIM_INFERENCE_REQUIREMENTS,
    MATTERSIM_VERSION,
    MATTERSIM_WHEEL_SHA256,
    CHGNET_VERSION,
    RETAINED_TORCH_BASE,
    SOURCE_SDISTS_DIRNAME,
    WAIVED_REQUIREMENT,
    WAIVER_SCHEMA,
    WHEELHOUSE_DIRNAME,
    WHEELHOUSE_LOCK_FILENAME,
    build_isolated_runtime_resolver_inputs,
    build_runtime_tree_manifest,
    build_waiver_resolver_input,
    load_chgnet_torch_waiver,
    load_mattersim_inference_waiver,
    load_wheel_distribution_metadata,
    validate_runtime_tree_manifest,
)


class DependencyWaiverTests(unittest.TestCase):
    def test_active_wheelhouse_stack_uses_versioned_paths(self) -> None:
        self.assertEqual(WHEELHOUSE_DIRNAME, "wheelhouse_v4")
        self.assertEqual(WHEELHOUSE_LOCK_FILENAME, "wheelhouse_lock_v4.json")
        self.assertEqual(SOURCE_SDISTS_DIRNAME, "source_sdists_v4")

    def test_wheel_metadata_ignores_vendored_dist_info(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "setuptools-81.0.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    "setuptools-81.0.0.dist-info/METADATA",
                    "Name: setuptools\nVersion: 81.0.0\n",
                )
                archive.writestr(
                    "setuptools/_vendor/example-1.0.dist-info/METADATA",
                    "Name: example\nVersion: 1.0\n",
                )
            metadata = load_wheel_distribution_metadata(wheel)
            self.assertEqual(metadata["Name"], "setuptools")
            self.assertEqual(metadata["Version"], "81.0.0")

    def test_conflicting_e3nn_evaluators_are_resolved_into_separate_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "requirements.txt"
            core_path = root / "core.txt"
            mattersim_path = root / "mattersim.txt"
            source.write_text(
                "pyxtal==1.1.4\nchgnet==0.4.2\n"
                "mattersim==1.1.2\nmace-torch==0.3.13\n"
            )
            core, mattersim = build_isolated_runtime_resolver_inputs(
                source, core_path, mattersim_path
            )
            self.assertNotIn("mace-torch==0.3.13", core)
            self.assertIn("e3nn==0.4.4", core)
            self.assertNotIn("python-hostlist", core)
            self.assertNotIn("mattersim==1.1.2", core)
            self.assertNotIn("chgnet==0.4.2", core)
            self.assertIn("e3nn>=0.5.0", mattersim)
            self.assertIn("ase==3.27.0", mattersim)
            self.assertIn("setuptools==81.0.0", mattersim)
            self.assertNotIn("mace-torch==0.3.13", mattersim)
            self.assertEqual(core_path.read_text().splitlines(), list(core))
            self.assertEqual(mattersim_path.read_text().splitlines(), list(mattersim))

    def test_resolver_input_removes_only_chgnet_and_exposes_its_other_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "requirements.txt"
            destination = root / "resolver.txt"
            source.write_text("pyxtal==1.1.4\nchgnet==0.4.2\nmattersim==1.1.2\n")
            values = build_waiver_resolver_input(source, destination)
            self.assertNotIn("chgnet==0.4.2", values)
            self.assertEqual(values[-len(CHGNET_NON_TORCH_REQUIREMENTS) :], CHGNET_NON_TORCH_REQUIREMENTS)
            self.assertEqual(destination.read_text().splitlines(), list(values))

    def test_loader_rejects_any_broader_or_tampered_waiver(self) -> None:
        line = (
            "chgnet 0.4.2 has requirement torch>=2.4.1, "
            "but you have torch 2.4.0+cu121."
        )
        import hashlib

        payload = {
            "schema": WAIVER_SCHEMA,
            "authorization": AUTHORIZATION,
            "scope": "chgnet_0p4p2_torch_minimum_metadata_only",
            "package": "chgnet",
            "package_version": CHGNET_VERSION,
            "waived_requirement": WAIVED_REQUIREMENT,
            "retained_torch": "2.4.0+cu121",
            "retained_torch_base": RETAINED_TORCH_BASE,
            "non_torch_requirements": list(CHGNET_NON_TORCH_REQUIREMENTS),
            "pip_check_output": [line],
            "pip_check_output_sha256": hashlib.sha256((line + "\n").encode()).hexdigest(),
            "authorized_chgnet_pip_check_line": line,
            "preexisting_pip_check_status": 0,
            "preexisting_pip_check_output": [],
            "preexisting_pip_check_output_sha256": hashlib.sha256(b"").hexdigest(),
            "validation": CHGNET_WAIVER_VALIDATION,
            "source_bundle_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "waiver.json"
            path.write_text(json.dumps(payload))
            loaded = load_chgnet_torch_waiver(path, installed_torch="2.4.0+cu121")
            self.assertEqual(loaded["waived_requirement"], WAIVED_REQUIREMENT)
            payload["waived_requirement"] = "torch>=2"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_chgnet_torch_waiver(path, installed_torch="2.4.0+cu121")
            payload["waived_requirement"] = WAIVED_REQUIREMENT
            payload["pip_check_output"].append("unrelated 1.0 requires missing-package")
            payload["pip_check_output_sha256"] = hashlib.sha256(
                ("\n".join(payload["pip_check_output"]) + "\n").encode()
            ).hexdigest()
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_chgnet_torch_waiver(path, installed_torch="2.4.0+cu121")

    def test_mattersim_loader_rejects_runtime_tree_tampering(self) -> None:
        payload = {
            "schema": WAIVER_SCHEMA,
            "authorization": "user_environment_scope_explicit_2026-07-17",
            "scope": "mattersim_1p1p2_forcefield_inference_runtime_only",
            "package": "mattersim",
            "package_version": MATTERSIM_VERSION,
            "official_wheel_sha256": MATTERSIM_WHEEL_SHA256,
            "retained_torch": "2.4.0+cu121",
            "retained_torch_base": RETAINED_TORCH_BASE,
            "inference_requirements": list(MATTERSIM_INFERENCE_REQUIREMENTS),
            "excluded_noninference_distributions": list(
                MATTERSIM_EXCLUDED_NONINFERENCE_DISTRIBUTIONS
            ),
            "compatibility_pins": MATTERSIM_COMPATIBILITY_PINS,
            "runtime_tree_sha256": "b" * 64,
            "source_bundle_sha256": "a" * 64,
            "validation": "reviewed_forcefield_import_closure_satisfied_in_isolated_target",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mattersim-waiver.json"
            path.write_text(json.dumps(payload))
            load_mattersim_inference_waiver(
                path, installed_torch="2.4.0+cu121"
            )
            payload["runtime_tree_sha256"] = "not-a-digest"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_mattersim_inference_waiver(
                    path, installed_torch="2.4.0+cu121"
                )

    def test_runtime_tree_manifest_detects_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "package").mkdir()
            target = root / "package/module.py"
            target.write_text("VALUE = 1\n")
            manifest = build_runtime_tree_manifest(root)
            self.assertEqual(validate_runtime_tree_manifest(root, manifest), manifest["tree_sha256"])
            target.write_text("VALUE = 2\n")
            with self.assertRaises(RuntimeError):
                validate_runtime_tree_manifest(root, manifest)


if __name__ == "__main__":
    unittest.main()
