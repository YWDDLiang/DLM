from pathlib import Path
import importlib.util
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_c3fd_native_alignment_pool.py"
SPEC = importlib.util.spec_from_file_location("freeze_native_alignment_pool", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def teacher(index, element):
    return {
        "source_split": "train",
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
        "answer": "must not enter pool",
    }


def predicted(index, element):
    def fields(lattice):
        return {
            "lattice_system": {"prediction": lattice, "confidence": 0.8},
            "spacegroup_bucket": {"prediction": "sg_075_142", "confidence": 0.7},
            "volume_per_atom_bin": {"prediction": "volpa_020_024", "confidence": 0.6},
        }
    return {
        "split": "train",
        "source_row_idx": index,
        "N": 3,
        "elements": [element, "O"],
        "counts": [2, 1],
        "predictions_by_checkpoint": {
            "seed17": fields("tetragonal"),
            "seed18": fields("hexagonal"),
        },
    }


class FreezeC3FDNativeAlignmentPoolTest(unittest.TestCase):
    def test_train_only_fixed_k_same_prompt_and_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            teacher_dir = root / "teacher"
            predicted_dir = root / "predicted"
            exclude_root = root / "cohorts"
            teacher_dir.mkdir()
            predicted_dir.mkdir()
            exclude_root.mkdir()
            elements = [
                "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
                "B", "Al", "Ga", "In", "C", "Si", "Ge", "Sn", "N", "P",
                "As", "Sb", "F", "Cl", "Br", "I", "Sc", "Ti", "V", "Cr",
                "Mn", "Fe", "Co", "Ni",
            ]
            write_jsonl(teacher_dir / "train.jsonl", [teacher(i, e) for i, e in enumerate(elements)])
            write_jsonl(predicted_dir / "train.jsonl", [predicted(i, e) for i, e in enumerate(elements)])
            write_jsonl(exclude_root / "old.jsonl", [{"reduced_composition_identity": "Li:2|O:1"}])
            output = root / "output"
            manifest = MODULE.freeze(
                teacher_dir=teacher_dir,
                predicted_dir=predicted_dir,
                exclude_cohort_root=exclude_root,
                output_dir=output,
                compositions=32,
                candidates_per_group=4,
                selection_seed=19,
            )
            self.assertEqual(manifest["groups"], 64)
            self.assertEqual(manifest["rows"], 256)
            self.assertTrue(all(manifest["gates"].values()))
            rows = [json.loads(line) for line in (output / "pool_plans.jsonl").read_text().splitlines()]
            groups = [json.loads(line) for line in (output / "groups.jsonl").read_text().splitlines()]
            self.assertNotIn("Li:2|O:1", {row["reduced_composition_identity"] for row in rows})
            self.assertTrue(all("answer" not in row for row in rows))
            for group in groups:
                selected = [row for row in rows if row["group_id"] == group["group_id"]]
                self.assertEqual(len(selected), 4)
                self.assertEqual(len({row["prompt"] for row in selected}), 1)

    def test_production_shape_is_fixed256(self):
        self.assertEqual(32 * 2 * 4, 256)


if __name__ == "__main__":
    unittest.main()
