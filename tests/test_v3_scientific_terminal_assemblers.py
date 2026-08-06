from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLANNER_ASSEMBLER = (
    REPOSITORY_ROOT
    / "workstreams/plangraph_dlm_iclr_20260731/execution"
    / "v3_pstar_scientific400_v1/assemble_planner_terminal.py"
)
DLM_ASSEMBLER = (
    REPOSITORY_ROOT
    / "workstreams/plangraph_dlm_iclr_20260731/execution"
    / "v3_dlm_scientific1epoch_lr5e5_v1/assemble_dlm_terminal.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


class PlannerTerminalAssemblerTest(unittest.TestCase):
    def test_registered_checkpoint_selection(self) -> None:
        module = load_module("planner_scientific_assembler_test", PLANNER_ASSEMBLER)
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            arm_root = run_root / "arms/pstar"
            checkpoints = []
            for step in range(50, 401, 50):
                checkpoint = arm_root / f"checkpoints/step_{step:04d}"
                manifest = checkpoint / "checkpoint_manifest.json"
                write_json(manifest, {"arm": "pstar", "step": step})
                manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
                checkpoints.append(
                    {
                        "step": step,
                        "path": str(checkpoint.relative_to(arm_root)),
                        "manifest_sha256": manifest_sha,
                        "metrics": {
                            "target_nll": 1.005 if step != 100 else 1.0,
                            "field_loss": 2.0 - (0.5 if step == 100 else step / 1000),
                        },
                    }
                )
            write_json(
                arm_root / "training_report.json",
                {
                    "status": "complete",
                    "arm": "pstar",
                    "microbatches": 3200,
                    "optimizer_updates": 400,
                    "checkpoints": checkpoints,
                },
            )
            result = module.select_arm(run_root, "pstar", p0_target_nll=1.0)
            self.assertEqual(result["eligible_count"], 8)
            self.assertEqual(result["selected"]["step"], 100)
            self.assertFalse(result["selection_uses_generation_sun_or_energy"])
            self.assertFalse(result["automatic_promotion"])


class DlmTerminalAssemblerTest(unittest.TestCase):
    def test_complete_one_epoch_arm_audit(self) -> None:
        module = load_module("dlm_scientific_assembler_test", DLM_ASSEMBLER)
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            output = run_root / "arms/B2/output"
            write_json(
                output / "run_config.json",
                {
                    "planned_corruption_policy": "d2",
                    "epochs": 1,
                    "max_train_steps": 1696,
                    "batch_size": 1,
                    "grad_accum": 8,
                    "lr": 5e-5,
                    "lr_scheduler": "cosine",
                    "warmup_steps": 100,
                    "min_lr_ratio": 0.2,
                    "eval_before_train": True,
                    "eval_steps": 212,
                    "eval_max_batches": 50,
                    "distributed": True,
                    "world_size": 2,
                },
            )
            write_json(
                output / "distributed_runtime_report.json",
                {
                    "status": "complete",
                    "runtime_gate_passed": True,
                    "world_size": 2,
                    "global_effective_batch": 16,
                    "global_train_sequences": 27136,
                    "optimizer_updates": 1696,
                    "final_checkpoint_saved": True,
                },
            )
            write_json(
                output / "validation_sampler_report.json",
                {
                    "gate_passed": True,
                    "dataset_length": 9047,
                    "duplicate_count": 0,
                    "missing_count": 0,
                    "per_rank": [{"count": 4524}, {"count": 4523}],
                },
            )
            output.mkdir(parents=True, exist_ok=True)
            with (output / "training_log.jsonl").open("w", encoding="utf-8") as handle:
                for offset, step in enumerate(module.VALIDATION_STEPS):
                    handle.write(
                        json.dumps(
                            {
                                "event": "eval",
                                "step": step,
                                "val_loss": 2.0 - offset * 0.01,
                            }
                        )
                        + "\n"
                    )
            final = output / "final"
            final.mkdir()
            (final / "adapter_config.json").write_text("{}\n", encoding="utf-8")

            result = module.audit_arm(run_root, "B2")
            self.assertEqual(result["optimizer_updates"], 1696)
            self.assertEqual(result["validation_steps"], module.VALIDATION_STEPS)
            self.assertTrue(result["nll_noninferior_to_initial"])
            self.assertEqual(result["final_inventory"][0]["path"], "adapter_config.json")
            self.assertFalse(result["automatic_promotion"])


if __name__ == "__main__":
    unittest.main()
