from pathlib import Path
import importlib.util
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_c3fd_native_sft_canary.py"
SPEC = importlib.util.spec_from_file_location("freeze_c3fd_native_sft_canary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
freeze = MODULE.freeze


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def teacher(split, index, element):
    return {
        "source_split": split,
        "source_row_idx": index,
        "reduced_composition_identity": f"{element}:2|O:1",
        "plan_state": {
            "N": 3,
            "elements": [element, "O"],
            "counts": [2, 1],
            "anion_framework": "oxide",
            "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230",
            "volume_per_atom_bin": "volpa_015_019",
        },
        "prompt_schema": "C3FD_NATIVE_PLAN_V2",
        "prompt": "teacher prompt",
        "answer": "must never be copied",
    }


def predicted(split, index, element):
    payload = {}
    for seed, lattice in (("seed17", "tetragonal"), ("seed18", "hexagonal")):
        payload[seed] = {
            "lattice_system": {"prediction": lattice, "confidence": 0.8},
            "spacegroup_bucket": {"prediction": "sg_075_142", "confidence": 0.7},
            "volume_per_atom_bin": {"prediction": "volpa_020_024", "confidence": 0.6},
        }
    return {
        "split": split,
        "source_row_idx": index,
        "N": 3,
        "elements": [element, "O"],
        "counts": [2, 1],
        "predictions_by_checkpoint": payload,
    }


class FreezeC3FDNativeSFTCanaryTest(unittest.TestCase):
    def test_freezes_balanced_outcome_blind_two_planner_views(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            teacher_dir = root / "teacher"
            predicted_dir = root / "predicted"
            teacher_dir.mkdir()
            predicted_dir.mkdir()
            for split, elements in (("train", ["Li", "Na", "K"]), ("val", ["Mg", "Ca", "Sr"])):
                write_jsonl(
                    teacher_dir / f"{split}.jsonl",
                    [teacher(split, idx, element) for idx, element in enumerate(elements)],
                )
                write_jsonl(
                    predicted_dir / f"{split}.jsonl",
                    [predicted(split, idx, element) for idx, element in enumerate(elements)],
                )
            output = root / "cohort"
            manifest = freeze(
                teacher_dir=teacher_dir,
                predicted_dir=predicted_dir,
                output_dir=output,
                per_split=2,
                freeze_seed=17,
            )
            self.assertEqual(manifest["requested"], 4)
            self.assertEqual(manifest["split_counts"], {"train": 2, "val": 2})
            self.assertTrue(all(manifest["gates"].values()))
            self.assertFalse(manifest["policy_or_test_outcomes_read"])
            seed17 = [json.loads(line) for line in (output / "planner_seed17.jsonl").read_text().splitlines()]
            seed18 = [json.loads(line) for line in (output / "planner_seed18.jsonl").read_text().splitlines()]
            self.assertEqual([row["sample_idx"] for row in seed17], list(range(4)))
            self.assertEqual(
                [row["reduced_composition_identity"] for row in seed17],
                [row["reduced_composition_identity"] for row in seed18],
            )
            self.assertTrue(all("answer" not in row for row in seed17 + seed18))
            self.assertTrue(all(row["prompt"].endswith("dynamic_crystal_body:") for row in seed17 + seed18))
            self.assertNotEqual(
                seed17[0]["plan_state"]["lattice_system"],
                seed18[0]["plan_state"]["lattice_system"],
            )
            for line in (output / "SHA256SUMS").read_text().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(digest, __import__("hashlib").sha256((output / name).read_bytes()).hexdigest())
            self.assertTrue((output / "_SUCCESS").is_file())

    def test_rejects_hard_composition_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "teacher").mkdir()
            (root / "predicted").mkdir()
            for split, element in (("train", "Li"), ("val", "Na")):
                write_jsonl(root / "teacher" / f"{split}.jsonl", [teacher(split, 0, element)])
                row = predicted(split, 0, element)
                if split == "train":
                    row["counts"] = [1, 2]
                write_jsonl(root / "predicted" / f"{split}.jsonl", [row])
            with self.assertRaisesRegex(ValueError, "counts differs"):
                freeze(
                    teacher_dir=root / "teacher",
                    predicted_dir=root / "predicted",
                    output_dir=root / "cohort",
                    per_split=1,
                    freeze_seed=17,
                )


if __name__ == "__main__":
    unittest.main()
