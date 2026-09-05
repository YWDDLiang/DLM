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
from crystal_dlm.manifold_corruption import (
    CorruptionCertificate,
    minimum_image_cartesian_retraction,
    relative_spd_tangent,
)


SPEC = importlib.util.spec_from_file_location(
    "certify_pmtr_corruptions",
    ROOT / "scripts" / "certify_pmtr_corruptions.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import PMTR certification script")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source(source_idx):
    body, _ = arrays_to_dynamic_answer(
        lengths=[4.2, 5.1, 6.3],
        angles=[78.0, 91.0, 103.0],
        species=["O", "O", "Na", "Cl"],
        frac_coords=[
            [0.03, 0.17, 0.29],
            [0.41, 0.53, 0.67],
            [0.72, 0.11, 0.84],
            [0.26, 0.78, 0.45],
        ],
        separator="",
    )
    return {
        "source_row_idx": int(source_idx),
        "source_split": "train",
        "answer": body,
    }


class FakeEvaluator:
    def __init__(self, clean):
        self.clean = clean
        self.batch_sizes = []

    def evaluate(self, geometries, *, batch_size):
        self.batch_sizes.append(batch_size)
        output = []
        for geometry in geometries:
            vectors = minimum_image_cartesian_retraction(
                self.clean.frac_coords,
                geometry.frac_coords,
                geometry.lattice,
                image_radius=2,
            )
            tangent = relative_spd_tangent(geometry.metric, self.clean.metric)
            output.append(
                {
                    "e": float(
                        np.mean(np.sum(vectors * vectors, axis=1))
                        + 0.05 * np.sum(tangent * tangent)
                    ),
                    "f": 2.0 * vectors,
                    "s": np.zeros((3, 3)),
                    "m": np.zeros(len(geometry.species)),
                }
            )
        return output


class CertifyPMTRCorruptionsScriptTest(unittest.TestCase):
    def test_chgnet_adapter_batches_without_importing_chgnet(self):
        class FakeStructure:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeModel:
            def __init__(self):
                self.chunk_sizes = []

            def predict_structure(self, structures, *, task, batch_size=None):
                if not isinstance(structures, list):
                    structures = [structures]
                self.chunk_sizes.append(len(structures))
                self.asserted_task = task
                return [
                    {
                        "e": 0.0,
                        "f": np.zeros((len(structure.kwargs["species"]), 3)),
                        "s": np.zeros((3, 3)),
                        "m": np.zeros(len(structure.kwargs["species"])),
                    }
                    for structure in structures
                ]

        clean = MODULE.CrystalGeometry.from_mapping(
            MODULE.parse_dynamic_answer(source(0)["answer"], strict=True)
        )
        model = FakeModel()
        evaluator = MODULE.CHGNetEFSMEvaluator(model, FakeStructure)
        predictions = evaluator.evaluate([clean] * 18, batch_size=8)
        self.assertEqual(len(predictions), 18)
        self.assertEqual(model.chunk_sizes, [8, 8, 2])
        self.assertEqual(model.asserted_task, "efsm")

    def test_fake_evaluator_supports_shard_limit_and_compact_manifest(self):
        clean = MODULE.CrystalGeometry.from_mapping(
            MODULE.parse_dynamic_answer(source(0)["answer"], strict=True)
        )
        evaluator = FakeEvaluator(clean)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "train.jsonl"
            output_dir = root / "out"
            input_path.write_text(
                "".join(json.dumps(source(index)) + "\n" for index in range(4)),
                encoding="utf-8",
            )
            with mock.patch.object(
                MODULE,
                "load_chgnet_evaluator",
                side_effect=AssertionError("lazy CHGNet loader must not run"),
            ):
                MODULE.main(
                    [
                        "--input-jsonl",
                        str(input_path),
                        "--output-dir",
                        str(output_dir),
                        "--seed",
                        "17",
                        "--max-proposals",
                        "2",
                        "--batch-size",
                        "8",
                        "--lattice-log-std",
                        "0.10",
                        "--coordinate-std-A",
                        "0.28",
                        "--max-delta-energy",
                        "10.0",
                        "--shard-rank",
                        "1",
                        "--shard-count",
                        "2",
                        "--limit",
                        "1",
                    ],
                    evaluator=evaluator,
                )
            rows = list(MODULE.iter_jsonl(output_dir / "certificates.jsonl"))
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["source_row_idx"] for row in rows}, {1})
            self.assertEqual([row["proposal_index"] for row in rows], [0, 1])
            self.assertTrue(all(row["certified"] for row in rows))
            self.assertEqual(manifest["sources"], 1)
            self.assertEqual(manifest["proposals"], 2)
            self.assertEqual(manifest["batch_size"], 8)
            self.assertEqual(manifest["selection"].split(";")[0], "none")
            self.assertNotIn("sha256", json.dumps(manifest).lower())
            self.assertTrue(evaluator.batch_sizes)
            config = MODULE.CorruptionConfig(
                max_proposals=2,
                lattice_log_std=0.10,
                coordinate_cartesian_std_A=0.28,
                max_delta_energy=10.0,
            )
            regenerated = MODULE._source_proposals(
                source(1), seed=17, config=config
            )
            for row, proposal in zip(rows, regenerated, strict=True):
                certificate = CorruptionCertificate.from_mapping(row["certificate"])
                self.assertEqual(
                    row["certified"], certificate.accepts(proposal, config)
                )

    def test_non_train_source_is_rejected_before_evaluation(self):
        row = source(0)
        row["source_split"] = "val"
        clean = MODULE.CrystalGeometry.from_mapping(
            MODULE.parse_dynamic_answer(row["answer"], strict=True)
        )
        with self.assertRaisesRegex(ValueError, "MP20-train"):
            MODULE._source_proposals(
                row,
                seed=17,
                config=MODULE.CorruptionConfig(max_proposals=1),
            )


if __name__ == "__main__":
    unittest.main()
