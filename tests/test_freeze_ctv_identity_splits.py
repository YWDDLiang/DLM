import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_ctv_identity_splits",
    ROOT / "scripts" / "freeze_ctv_identity_splits.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import freeze_ctv_identity_splits.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def plan(left: str, left_count: int, right: str, right_count: int):
    return {
        "N": left_count + right_count,
        "elements": [left, right],
        "counts": [left_count, right_count],
        "formula": f"{left}{left_count}{right}{right_count}",
    }


def write(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class FreezeCTVIdentitySplitsTest(unittest.TestCase):
    def test_unique_selection_rejects_within_and_cross_split_duplicates(self):
        rows = []
        for index, identity in enumerate(("a", "a", "b", "c")):
            rows.append(
                {
                    "sample_idx": index,
                    "reduced_composition_identity": identity,
                    "plan_state": plan("B", index + 1, "C", 1),
                }
            )
        selected, rejected = MODULE.select_unique_rows(
            rows, count=2, blocked_identities={"c"}
        )
        self.assertEqual(
            [row["reduced_composition_identity"] for row in selected],
            ["a", "b"],
        )
        self.assertEqual(len(rejected), 1)
        self.assertEqual(
            rejected[0]["ctv_identity_rejection"],
            "duplicate_or_cross_split_identity",
        )

    def test_certificate_filter_happens_before_selection(self):
        rows = [
            {"sample_idx": index, "plan_state": plan("B", index + 1, "C", 1)}
            for index in range(4)
        ]

        def compile_fn(_row, row_index):
            return {
                "composition_supervision": row_index % 2 == 0,
                "certificate_class": "benchmark_compatible"
                if row_index % 2 == 0
                else "extended_only",
            }

        accepted, rejected = MODULE.filter_branch_by_certificate(rows, compile_fn)
        self.assertEqual([row["sample_idx"] for row in accepted], [0, 2])
        self.assertEqual([row["sample_idx"] for row in rejected], [1, 3])
        self.assertTrue(
            all(row["ctv_certificate_rejection"] == "extended_only" for row in rejected)
        )

    def test_freezes_pairwise_disjoint_sets_without_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed17 = [
                {"sample_idx": index, "plan_state": plan("H", index + 1, "He", 1)}
                for index in range(1000)
            ]
            seed18 = [
                {"sample_idx": index, "plan_state": plan("Li", index + 1, "Be", 1)}
                for index in range(1000)
            ]
            branch = []
            for index in range(256):
                branch.append(
                    {
                        "sample_idx": index,
                        "pair_split": "train" if index < 192 else "validation",
                        "plan_state": plan("B", index + 1, "C", 1),
                    }
                )
            # One Branch row intentionally overlaps seed18 and must be excluded.
            branch[0]["plan_state"] = dict(seed18[0]["plan_state"])
            write(root / "seed17.jsonl", seed17)
            write(root / "seed18.jsonl", seed18)
            write(root / "branch.jsonl", branch)
            output = root / "output"
            argv = [
                "freeze_ctv_identity_splits.py",
                "--branch-cohort",
                str(root / "branch.jsonl"),
                "--c3fd-seed17",
                str(root / "seed17.jsonl"),
                "--c3fd-seed18",
                str(root / "seed18.jsonl"),
                "--output-dir",
                str(output),
                "--canary-plans",
                "8",
                "--branch-train-plans",
                "128",
                "--branch-val-plans",
                "32",
                "--l6-plans",
                "256",
            ]
            previous = sys.argv
            try:
                sys.argv = argv
                with contextlib.redirect_stdout(io.StringIO()):
                    MODULE.main()
            finally:
                sys.argv = previous
            report = json.loads(
                (output / "CTV_IDENTITY_FREEZE_MANIFEST.json").read_text()
            )
            self.assertTrue(report["gate"]["identity_freeze_authorized"])
            self.assertEqual(report["counts"]["CTV_DLM_L7_PLANS.jsonl"], 1000)
            self.assertEqual(report["counts"]["CTV_DLM_L6_PLANS.jsonl"], 256)
            self.assertEqual(report["source_diagnostics"]["branch_vs_seed18"], 1)
            self.assertTrue(all(value == 0 for value in report["overlap"].values()))
            self.assertTrue(report["gate"]["branch_frozen_identities_unique"])
            self.assertFalse(
                report["configuration"]["require_c3fd_certification"]
            )


if __name__ == "__main__":
    unittest.main()
