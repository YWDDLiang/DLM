import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "workstreams"
    / "plangraph_dlm_iclr_20260731"
    / "execution"
    / "v3_dlm_b1_b2_2xa800_smoke32_v1"
    / "assemble_arm_report.py"
)
SPEC = importlib.util.spec_from_file_location("dlm_smoke_assembler", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


class DlmTwoA800SmokeAssemblerTests(unittest.TestCase):
    def build_fixture(self, root: Path):
        output = root / "output"
        output.mkdir()
        write_json(
            output / "run_config.json",
            {
                "representation": "dynamic_v1",
                "planned_corruption_policy": "d1",
                "iid_fraction": 2.0,
                "planned_fraction": 1.0,
                "corruption_seed": 20260731,
                "data_seed": 20260515,
                "max_length": 382,
                "answer_token_count": 87,
                "limit_train": 32,
                "limit_val": 32,
                "epochs": 1,
                "max_train_steps": 2,
                "batch_size": 1,
                "grad_accum": 8,
                "lr": 5e-05,
                "lr_scheduler": "constant",
                "warmup_steps": 0,
                "weight_decay": 0.0,
                "logging_steps": 1,
                "eval_before_train": True,
                "eval_steps": 2,
                "eval_max_batches": 16,
                "dataloader_num_workers": 0,
                "engineering_only": True,
                "skip_final_save": True,
                "distributed": True,
                "world_size": 2,
            },
        )
        write_json(
            output / "validation_sampler_report.json",
            {
                "gate_passed": True,
                "dataset_length": 32,
                "world_size": 2,
                "total_assigned": 32,
                "unique_assigned": 32,
                "duplicate_count": 0,
                "missing_count": 0,
                "rank_mapping_exact": True,
                "sampler": "DistributedNoPaddingSampler",
                "per_rank": [{"count": 16}, {"count": 16}],
            },
        )
        rank_runtime = []
        for rank in range(2):
            rank_runtime.append(
                {
                    "rank": rank,
                    "train_microbatches": 16,
                    "optimizer_updates": 2,
                    "task_loss_count": 16,
                    "task_loss_min": 0.5,
                    "task_loss_max": 1.5,
                    "gradient_norm_count": 2,
                    "gradient_norm_min": 1.0,
                    "gradient_norm_max": 2.0,
                    "evaluation_loss_count": 2,
                    "evaluation_losses": [1.2, 1.1],
                    "cuda_peak_allocated_bytes": 100,
                    "cuda_peak_reserved_bytes": 200,
                }
            )
        write_json(
            output / "distributed_runtime_report.json",
            {
                "runtime_gate_passed": True,
                "distributed": True,
                "world_size": 2,
                "batch_size_per_rank": 1,
                "gradient_accumulation": 8,
                "global_effective_batch": 16,
                "global_train_sequences": 32,
                "optimizer_updates": 2,
                "rank0_only_checkpoint_and_report_publication": True,
                "final_checkpoint_saved": False,
                "engineering_only": True,
                "eligible_for_checkpoint_selection": False,
                "eligible_for_later_initialization": False,
                "automatic_downstream": False,
                "scientific_training_authorized": False,
                "rank_runtime": rank_runtime,
            },
        )
        events = [
            {"event": "eval", "step": 0, "val_loss": 1.2},
            {
                "event": "train",
                "step": 1,
                "task_loss": 1.0,
                "pre_clip_gradient_norm": 2.0,
            },
            {
                "event": "train",
                "step": 2,
                "task_loss": 0.9,
                "pre_clip_gradient_norm": 1.8,
            },
            {"event": "eval", "step": 2, "val_loss": 1.1},
        ]
        with (output / "training_log.jsonl").open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        gpu_csv = root / "gpu.csv"
        gpu_csv.write_text(
            "2026-08-01, 0, NVIDIA A800-SXM4-80GB, 100, 81920, 50, 10\n"
            "2026-08-01, 1, NVIDIA A800-SXM4-80GB, 101, 81920, 51, 11\n",
            encoding="utf-8",
        )
        return output, gpu_csv

    def test_complete_fixture_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, gpu_csv = self.build_fixture(Path(tmp))
            report = MODULE.assemble("B1", output, gpu_csv)
        self.assertTrue(report["engineering_gate_passed"])
        self.assertEqual(report["optimizer_updates"], 2)
        self.assertEqual(report["validation_rows"], 32)

    def test_validation_duplicate_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, gpu_csv = self.build_fixture(Path(tmp))
            validation_path = output / "validation_sampler_report.json"
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            validation["duplicate_count"] = 1
            write_json(validation_path, validation)
            with self.assertRaises(MODULE.ArmGateError):
                MODULE.assemble("B1", output, gpu_csv)


if __name__ == "__main__":
    unittest.main()
