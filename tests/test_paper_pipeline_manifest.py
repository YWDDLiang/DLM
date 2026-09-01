import contextlib
import io
import json
from pathlib import Path
import unittest

from crystal_dlm.paper_pipeline.cli import main
from crystal_dlm.paper_pipeline.manifest import (
    MAINLINE_STAGE_ORDER,
    command_for_stage,
    load_and_validate,
)


ROOT = Path(__file__).resolve().parents[1]


class PaperPipelineManifestTest(unittest.TestCase):
    def test_mainline_manifest_and_every_repository_path_validate(self) -> None:
        manifest, report = load_and_validate(root=ROOT)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(tuple(report["stage_order"]), MAINLINE_STAGE_ORDER)
        self.assertEqual(report["stage_count"], 10)
        self.assertEqual(report["component_count"], 6)
        self.assertTrue(report["checked_repository_paths"])
        self.assertEqual(
            manifest["profiles"]["prospective_headline"]["periodic_checkpoint"],
            "registered G2 step348",
        )
        self.assertEqual(
            manifest["profiles"]["full_epoch_mechanism"]["periodic_checkpoint"],
            "G2-PBC-R step1696",
        )

    def test_selected_periodic_stage_uses_a_without_uncertainty_gate(self) -> None:
        manifest, _ = load_and_validate(root=ROOT)
        command = command_for_stage(manifest, "train-periodic-dlm")
        self.assertEqual(command["command"], ["sbatch", "slurm/129_train_g2_full_epoch_ab.sbatch"])
        self.assertEqual(command["contract_environment"], {"G2_FULL_METHOD": "A"})
        g2 = json.loads((ROOT / "configs/paper/g2_pbc_r_v1.json").read_text())
        self.assertFalse(g2["relation_residual"]["uncertainty_gate"])
        self.assertEqual(g2["periodic_geometry"]["image_count"], 125)

    def test_config_files_are_portable_and_contain_no_machine_paths(self) -> None:
        for path in sorted((ROOT / "configs/paper").glob("*.json")):
            text = path.read_text(encoding="utf-8")
            json.loads(text)
            self.assertNotIn("/public/home/", text)
            self.assertNotIn("D:\\\\", text)
            self.assertNotIn("MP_API_KEY", text)

    def test_cli_is_read_only_and_reports_stage_contract(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["stage", "sample-plan"])
        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], ["sbatch", "slurm/125_sample_final_fused_planner.sbatch"])
        self.assertEqual(payload["execution"], "not_started_by_this_read_only_command")


if __name__ == "__main__":
    unittest.main()
