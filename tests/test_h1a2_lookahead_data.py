import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_h1a2_lookahead_planner_data import (
    build,
    formula_shape,
    proportional_quotas,
)


def answer(index: int) -> str:
    anion = ("oxide", "sulfide")[index % 2]
    charge = ("neutral_plausible", "all_metal")[index % 2]
    lattice = ("hexagonal", "cubic")[index % 2]
    spacegroup = ("sg_168_194", "sg_195_230")[index % 2]
    volume = ("volpa_005_009", "volpa_010_014")[index % 2]
    formula = ("Li2O", "Fe2")[index % 2]
    return (
        f"formula: {formula}\n"
        f"anion: {anion}\n"
        f"charge: {charge}\n"
        f"lattice: {lattice}\n"
        f"spacegroup: {spacegroup}\n"
        f"volume: {volume}\n"
        "end: plan"
    )


def write_source(path: Path, count: int) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "prompt": "fixed prompt",
                        "answer": answer(index),
                        "fixture_index": index,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()


class H1A2LookaheadDataTests(unittest.TestCase):
    def test_formula_shape_is_strict_and_merges_repeated_symbols(self):
        self.assertEqual(formula_shape("Li2O"), (2, 3))
        self.assertEqual(formula_shape("LiOLi"), (2, 3))
        with self.assertRaisesRegex(ValueError, "flat integer-count formula"):
            formula_shape("Li(OH)")
        with self.assertRaisesRegex(ValueError, "unknown element"):
            formula_shape("Xx2O")
        with self.assertRaisesRegex(ValueError, "must be positive"):
            formula_shape("Li0O")

    def test_largest_remainder_is_exact_and_bounded(self):
        quotas = proportional_quotas({("a",): 7, ("b",): 3}, target_count=6)
        self.assertEqual(sum(quotas.values()), 6)
        self.assertLessEqual(quotas[("a",)], 7)
        self.assertLessEqual(quotas[("b",)], 3)

    def test_builder_is_deterministic_and_shared_between_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            train_sha = write_source(source / "train.jsonl", 20)
            val_sha = write_source(source / "val.jsonl", 12)

            manifests = []
            outputs = []
            for version in (1, 2):
                output = root / f"out{version}"
                manifest = build(
                    argparse.Namespace(
                        source_dir=source,
                        output_dir=output,
                        train_sha256=train_sha,
                        val_sha256=val_sha,
                        train_count=8,
                        val_count=4,
                        seed=17,
                    )
                )
                manifests.append(manifest)
                outputs.append(output)

            self.assertEqual(
                (outputs[0] / "train.jsonl").read_bytes(),
                (outputs[1] / "train.jsonl").read_bytes(),
            )
            self.assertEqual(
                (outputs[0] / "val.jsonl").read_bytes(),
                (outputs[1] / "val.jsonl").read_bytes(),
            )
            self.assertEqual(manifests[0]["selection"]["train_rows"], 8)
            self.assertEqual(manifests[0]["selection"]["val_rows"], 4)
            self.assertEqual(
                manifests[0]["selection"]["same_order_for_arms"],
                ["P-control", "Pstar"],
            )
            rows = [
                json.loads(line)
                for line in (outputs[0] / "train.jsonl").read_text().splitlines()
            ]
            identities = [
                row["v3_planner_stream"]["source_line_sha256"] for row in rows
            ]
            self.assertEqual(len(identities), len(set(identities)))
            self.assertTrue((outputs[0] / "_SUCCESS").is_file())

    def test_builder_fails_closed_on_source_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            train_sha = write_source(source / "train.jsonl", 4)
            val_sha = write_source(source / "val.jsonl", 4)
            with self.assertRaisesRegex(ValueError, "train SHA mismatch"):
                build(
                    argparse.Namespace(
                        source_dir=source,
                        output_dir=root / "out",
                        train_sha256="0" * 64,
                        val_sha256=val_sha,
                        train_count=2,
                        val_count=2,
                        seed=17,
                    )
                )
            self.assertNotEqual(train_sha, "0" * 64)


if __name__ == "__main__":
    unittest.main()
