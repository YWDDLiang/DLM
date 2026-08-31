import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_c3fd_llama_expander_data",
    ROOT / "scripts" / "build_c3fd_llama_expander_data.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import expander data builder")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class BuildC3FDLlamaExpanderDataTest(unittest.TestCase):
    def test_top_level_builder_adds_source_root(self):
        source = (ROOT / "scripts/build_c3fd_llama_expander_data.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('SOURCE_ROOT = PROJECT_ROOT / "src"', source)

    def test_builds_matched_F_and_M_without_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            semantic = root / "semantic"
            predicted = root / "predicted"
            semantic.mkdir()
            predicted.mkdir()
            vocabulary = {
                "species": [
                    {"id": 0, "atomic_number": 11, "oxidation_state": 1},
                    {"id": 1, "atomic_number": 17, "oxidation_state": -1},
                ],
                "soft_vocabulary": {
                    "lattice_system": ["cubic"],
                    "spacegroup_bucket": ["sg_195_230"],
                    "volume_per_atom_bin": ["volpa_020_024"],
                },
            }
            (semantic / "vocabulary.json").write_text(
                json.dumps(vocabulary), encoding="utf-8"
            )
            plan = {
                "N": 2,
                "elements": ["Na", "Cl"],
                "counts": [1, 1],
                "anion_framework": "halide",
                "charge_bucket": "neutral_plausible",
                "lattice_system": "cubic",
                "spacegroup_bucket": "sg_195_230",
                "volume_per_atom_bin": "volpa_020_024",
                "e_above_hull": -9.0,
            }
            semantic_row = {
                "source_row_idx": 5,
                "plan_state": plan,
                "certificate_class": "benchmark_compatible",
                "composition_supervision": True,
                "proposal_supervision": True,
                "proposal_targets": {"N": 2, "arity": 2, "family": 1},
                "species_labels": [0, 1],
                "count_targets": [1, 1],
                "ledger_steps": [{"net_charge": 0, "branch": "ionic"}],
            }
            predicted_row = {
                "source_row_idx": 5,
                "predictions_by_checkpoint": {
                    checkpoint: {
                        "lattice_system": {
                            "prediction": "cubic",
                            "confidence": 0.9,
                        },
                        "spacegroup_bucket": {
                            "prediction": "sg_195_230",
                            "confidence": 0.8,
                        },
                        "volume_per_atom_bin": {
                            "prediction": "volpa_020_024",
                            "confidence": 0.7,
                        },
                    }
                    for checkpoint in ("seed17", "seed18")
                },
            }
            for split in ("train", "val"):
                write_jsonl(semantic / f"{split}.jsonl", [semantic_row])
                write_jsonl(predicted / f"{split}.jsonl", [predicted_row])
            output = root / "output"
            manifest = MODULE.build_dataset(
                semantic_dir=semantic,
                predicted_dir=predicted,
                output_dir=output,
            )
            f_row = json.loads((output / "F/train.jsonl").read_text())
            m_row = json.loads((output / "M/train.jsonl").read_text())
            self.assertEqual(f_row["answer"], m_row["answer"])
            self.assertEqual(
                f_row["expander_plan_state"], m_row["expander_plan_state"]
            )
            self.assertNotIn("e_above_hull", f_row["expander_plan_state"])
            self.assertEqual(
                len(m_row["soft_prefix_features"]), manifest["feature_dim"]
            )
            self.assertTrue(manifest["outcomes_read"] is False)


if __name__ == "__main__":
    unittest.main()
