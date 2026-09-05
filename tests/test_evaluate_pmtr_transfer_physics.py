import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer


SPEC = importlib.util.spec_from_file_location(
    "evaluate_pmtr_transfer_physics",
    ROOT / "scripts" / "evaluate_pmtr_transfer_physics.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import PMTR transfer evaluator")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def body(length, coordinate=0.35):
    value, _ = arrays_to_dynamic_answer(
        lengths=[length, length + 0.2, length + 0.4],
        angles=[82.0, 91.0, 104.0],
        species=["Na", "Cl"],
        frac_coords=[[0.05, 0.15, 0.25], [coordinate, 0.65, 0.75]],
        separator="",
    )
    return value


def prediction(value, num_sites=2):
    return {
        "e": float(value),
        "f": np.full((num_sites, 3), float(value)),
        "s": np.eye(3) * float(value),
        "m": np.zeros(num_sites),
    }


class SequenceFakeEvaluator:
    def __init__(self, values):
        self.values = list(values)
        self.batch_sizes = []

    def evaluate(self, geometries, *, batch_size):
        self.batch_sizes.append(batch_size)
        self.site_counts = [len(geometry.species) for geometry in geometries]
        if len(self.values) != len(geometries):
            raise AssertionError("fake EFSM sequence cardinality changed")
        return self.values


class GeometryFakeEvaluator:
    def __init__(self):
        self.batch_sizes = []

    def evaluate(self, geometries, *, batch_size):
        self.batch_sizes.append(batch_size)
        return [prediction(float(geometry.lattice[0, 0])) for geometry in geometries]


class EvaluatePMTRTransferPhysicsTest(unittest.TestCase):
    def test_paired_metrics_keep_invalid_and_unknown_rows(self):
        rows = [
            {
                "source_row_idx": 7,
                "before_body": body(5.0),
                "after_body": body(4.0),
            },
            {
                "source_row_idx": 8,
                "before_body": body(5.0),
                "after_body": "not-a-dynamic-body",
            },
            {
                "source_row_idx": 9,
                "before_body": body(5.0),
                "after_body": body(4.5),
            },
        ]
        evaluator = SequenceFakeEvaluator(
            [
                prediction(2.0),
                prediction(1.0),
                prediction(2.0),
                prediction(2.0),
                None,
            ]
        )
        records = MODULE.evaluate_rows(
            rows,
            input_indices=[0, 1, 2],
            evaluator=evaluator,
            batch_size=16,
        )
        summary = MODULE.summarize_pairs(records)
        self.assertEqual(len(records), 3)
        self.assertEqual(evaluator.batch_sizes, [16])
        self.assertTrue(records[0]["before"]["efsm_known"])
        self.assertAlmostEqual(
            records[0]["delta_after_minus_before"]["energy_eV_per_atom"], -1.0
        )
        self.assertFalse(records[1]["after"]["body_valid"])
        self.assertFalse(records[2]["after"]["efsm_known"])
        self.assertEqual(summary["pairs"], 3)
        self.assertEqual(summary["body_valid_pairs"], 2)
        self.assertEqual(summary["efsm_known_pairs"], 1)
        self.assertEqual(summary["after_invalid"], 1)
        self.assertEqual(summary["after_efsm_unknown"], 1)
        for metric in MODULE.METRICS:
            self.assertEqual(summary["metrics"][metric]["wins"], 1)
            self.assertEqual(summary["metrics"][metric]["losses"], 0)

    def test_optional_bootstrap_is_small_and_deterministic(self):
        records = []
        for index, delta in enumerate((-2.0, -1.0, 1.0, 2.0)):
            record = {
                "before": {"body_valid": True, "efsm_known": True},
                "after": {"body_valid": True, "efsm_known": True},
                "delta_after_minus_before": {
                    metric: delta for metric in MODULE.METRICS
                },
            }
            records.append(record)
        plain = MODULE.summarize_pairs(records, bootstrap_samples=0)
        bootstrapped = MODULE.summarize_pairs(
            records, bootstrap_samples=100, bootstrap_seed=11
        )
        self.assertNotIn(
            "bootstrap_mean_delta_95_ci",
            plain["metrics"]["energy_eV_per_atom"],
        )
        interval = bootstrapped["metrics"]["energy_eV_per_atom"][
            "bootstrap_mean_delta_95_ci"
        ]
        self.assertEqual(len(interval), 2)
        self.assertLessEqual(interval[0], interval[1])

    def test_manual_shards_merge_in_original_order_without_selection(self):
        rows = [
            {
                "source_row_idx": index,
                "before_body": body(5.0 + 0.1 * index),
                "after_body": body(4.0 + 0.1 * index),
            }
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "fixed.jsonl"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            shard_dirs = [root / f"shard{rank}" for rank in range(4)]
            for rank, shard_dir in enumerate(shard_dirs):
                MODULE.evaluate_file(
                    input_path=input_path,
                    output_dir=shard_dir,
                    evaluator=GeometryFakeEvaluator(),
                    shard_rank=rank,
                    shard_count=4,
                    batch_size=16,
                )
            merged_dir = root / "merged"
            summary = MODULE.merge_shards(
                shard_dirs=shard_dirs,
                output_dir=merged_dir,
                bootstrap_samples=0,
            )
            merged = list(MODULE.iter_jsonl(merged_dir / "pairs.jsonl"))
            self.assertEqual(
                [row["input_row_index"] for row in merged], [0, 1, 2, 3]
            )
            self.assertEqual(summary["pairs"], 4)
            self.assertEqual(summary["merged_shards"], 4)
            self.assertEqual(summary["selection"], "none")
            self.assertFalse(summary["direct"])
            self.assertFalse(summary["materials_project_query"])

    def test_cli_with_fake_evaluator_never_loads_chgnet(self):
        evaluator = GeometryFakeEvaluator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "fixed.jsonl"
            output_dir = root / "output"
            input_path.write_text(
                json.dumps(
                    {
                        "sample_idx": 3,
                        "before": {"body": body(5.0)},
                        "after": {"body": body(4.0)},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                MODULE,
                "load_default_evaluator",
                side_effect=AssertionError("CHGNet loader must remain lazy"),
            ):
                MODULE.main(
                    [
                        "--input-jsonl",
                        str(input_path),
                        "--output-dir",
                        str(output_dir),
                    ],
                    evaluator=evaluator,
                )
            output = list(MODULE.iter_jsonl(output_dir / "pairs.jsonl"))
            self.assertEqual(len(output), 1)
            self.assertTrue(output[0]["before"]["efsm_known"])
            self.assertEqual(evaluator.batch_sizes, [16])

    def test_batch_size_is_fixed_at_sixteen(self):
        with self.assertRaisesRegex(ValueError, "fixed at 16"):
            MODULE.evaluate_rows(
                [], input_indices=[], evaluator=GeometryFakeEvaluator(), batch_size=8
            )


if __name__ == "__main__":
    unittest.main()
