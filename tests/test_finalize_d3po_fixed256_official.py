from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/finalize_d3po_fixed256_official.py"
SPEC = importlib.util.spec_from_file_location("finalize_d3po_fixed256_official", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def endpoint(
    stream_means: tuple[float, float, float, float],
    seed_means: tuple[float, float],
    *,
    ci_upper: float = -0.01,
) -> dict[str, object]:
    return {
        "per_stream": [
            {"mean_delta": value} for value in stream_means
        ],
        "per_training_seed": [
            {"training_seed": 81017, "mean_delta": seed_means[0]},
            {"training_seed": 81018, "mean_delta": seed_means[1]},
        ],
        "fixed_two_training_seed_cohort": {
            "bootstrap": {"ci95_upper": ci_upper}
        },
    }


def binary(
    *,
    meta: tuple[float, float] = (0.01, 0.01),
    direct: tuple[float, float] = (0.0, 0.0),
    nu: tuple[float, float] = (0.0, 0.0),
) -> dict[str, object]:
    return {
        str(seed): {
            "training_seed_mean_of_stream_rate_deltas": {
                "meta_sun": meta[index],
                "strict_sun": 0.0,
                "direct_joint": direct[index],
                "novel_unique": nu[index],
            }
        }
        for index, seed in enumerate((81017, 81018))
    }


class FakeRuntime:
    denominator = 256

    @staticmethod
    def phase_diagrams(_path: Path) -> dict[str, object]:
        return {}

    @staticmethod
    def evaluate_cell(**kwargs):
        labels = MODULE.read_jsonl(kwargs["labels_path"])
        direct_source = MODULE.read_json(kwargs["direct_path"])
        evaluated = []
        for row in labels:
            e_hull = None if row.get("fixture_unknown") else float(row["fixture_e_hull"])
            known = e_hull is not None
            novel = bool(row.get("novel"))
            unique = bool(row.get("unique_representative"))
            strict = known and e_hull <= 0.0
            meta = known and e_hull <= 0.1
            evaluated.append(
                {
                    **row,
                    "official_hull_status": "known" if known else "official_hull_unknown",
                    "official_e_above_hull": e_hull,
                    "novel_unique": novel and unique,
                    "strict_stable": strict,
                    "strict_sun": strict and novel and unique,
                    "meta_stable": meta,
                    "meta_sun": meta and novel and unique,
                }
            )
        output = kwargs["output_dir"]
        output.mkdir(parents=True, exist_ok=False)
        write_jsonl(output / "attempt_results_official.jsonl", evaluated)
        report = {
            "direct": {
                "composition_valid": int(direct_source["comp_valid_count"]),
                "structure_valid": int(direct_source["struct_valid_count"]),
                "joint_valid": int(direct_source["valid_count"]),
            }
        }
        write_json(output / "report.json", report)
        (output / "_SUCCESS").touch()
        return evaluated, report


class StatisticsTest(unittest.TestCase):
    def test_streams_are_averaged_within_composition_before_bootstrap(self):
        averaged = MODULE.average_delta_maps(
            [{0: -1.0, 1: 1.0}, {0: -3.0, 1: 1.0}]
        )
        self.assertEqual(averaged, {0: -2.0, 1: 1.0})
        first = MODULE.cluster_bootstrap_summary(
            averaged, label="fixed", replicates=200
        )
        second = MODULE.cluster_bootstrap_summary(
            averaged, label="fixed", replicates=200
        )
        self.assertEqual(first, second)
        self.assertEqual(first["compositions_observed"], 2)
        self.assertAlmostEqual(first["mean_delta"], -0.5)

    def test_unknown_hull_values_are_missing(self):
        distribution = MODULE.continuous_distribution([0.0, 0.05])
        self.assertEqual(distribution["known"], 2)
        self.assertEqual(distribution["ecdf"]["le_0"]["count"], 1)
        self.assertEqual(distribution["ecdf"]["le_0.05"]["count"], 2)


class ClassificationTest(unittest.TestCase):
    def classify(self, refined, raw, *, official=None, binary_values=None):
        return MODULE.classify_result(
            refined=refined,
            official=official or endpoint((-0.1,) * 4, (-0.1, -0.1)),
            raw=raw,
            binary=binary_values or binary(),
        )["code"]

    def test_assigns_every_preregistered_class(self):
        positive = endpoint((-0.1,) * 4, (-0.1, -0.1))
        self.assertEqual(self.classify(positive, positive), "P")
        weak_ci = endpoint((-0.1,) * 4, (-0.1, -0.1), ci_upper=0.01)
        self.assertEqual(self.classify(weak_ci, positive), "M")
        erased = endpoint((-0.1, 0.1, -0.1, 0.1), (0.0, 0.0))
        self.assertEqual(self.classify(erased, positive), "I")
        unstable_raw = endpoint((-0.1, 0.1, 0.1, 0.1), (-0.1, 0.1))
        self.assertEqual(self.classify(erased, unstable_raw), "U")
        stream_unstable_raw = endpoint((-0.3, 0.1, -0.3, 0.1), (-0.1, -0.1))
        self.assertEqual(self.classify(erased, stream_unstable_raw), "U")
        negative_raw = endpoint((0.1,) * 4, (0.1, 0.1))
        self.assertEqual(self.classify(erased, negative_raw), "N")
        engineering = MODULE.classify_result(engineering_failure="pre-science failure")
        self.assertEqual(engineering["code"], "E")
        self.assertFalse(engineering["hard_gate"])

    def test_p_requires_official_meta_and_safety_evidence(self):
        positive = endpoint((-0.1,) * 4, (-0.1, -0.1))
        adverse_official = endpoint((0.1,) * 4, (0.1, 0.1))
        self.assertEqual(
            self.classify(positive, positive, official=adverse_official), "M"
        )
        self.assertEqual(
            self.classify(
                positive,
                positive,
                binary_values=binary(direct=(-0.01, 0.0)),
            ),
            "M",
        )


class FinalizerIntegrationTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> argparse.Namespace:
        generation = root / "generation"
        evaluation = root / "evaluation"
        cache_root = root / "official"
        cache = cache_root / "official_mp_cache"
        runtime = root / "runtime"
        output = root / "final"
        for marker in (
            generation / "_SUCCESS",
            evaluation / "_OFFLINE_SUCCESS",
            cache / "completion_SUCCESS",
        ):
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
        write_jsonl(cache / "official_slim_cache.jsonl", [])
        write_jsonl(cache / "unresolved_chemsys.jsonl", [])
        runtime.mkdir()

        for stream in MODULE.STREAMS:
            for arm in MODULE.ARMS:
                gen_cell = generation / f"stream{stream}/{arm}"
                write_json(
                    gen_cell / "body/SGTC_BODY_MANIFEST.json",
                    {"parsed": 256, "graphs": 256},
                )
                write_json(
                    gen_cell / "refine/refinement_metrics.json",
                    {"num_proposals": 256},
                )
                for stage in MODULE.STAGES:
                    eval_arm = arm if stage == "refined" else f"raw_{arm}"
                    cell = evaluation / f"stream{stream}/{eval_arm}"
                    energy_shift = {
                        "base": 0.0,
                        "d3po_seed81017": -0.10,
                        "d3po_seed81018": -0.08,
                    }[arm]
                    hull_shift = {
                        "base": 0.0,
                        "d3po_seed81017": -0.04,
                        "d3po_seed81018": -0.03,
                    }[arm]
                    rows = []
                    generations = []
                    for ordinal in range(256):
                        base_hull = 0.12 if ordinal < 64 else 0.05
                        rows.append(
                            {
                                "ordinal": ordinal,
                                "attempt_id": f"{stage}-{stream}-{arm}-{ordinal}",
                                "reconstructed": True,
                                "chemsys": f"A{ordinal}-B",
                                "chgnet_composition": {f"A{ordinal}": 1, "B": 1},
                                "chgnet_energy_per_atom": -1.0 + energy_shift,
                                "chgnet_relaxation_known": True,
                                "fixture_e_hull": base_hull + hull_shift,
                                "novel": True,
                                "unique_representative": True,
                            }
                        )
                        generations.append(
                            {
                                "ordinal": ordinal,
                                "attempt_id": f"{stage}-{stream}-{arm}-{ordinal}",
                                "status": "succeeded",
                            }
                        )
                    write_jsonl(
                        cell / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
                        rows,
                    )
                    write_jsonl(cell / "generation/generation.jsonl", generations)
                    write_json(
                        cell / "evaluation/direct/report.json",
                        {
                            "attempts": 256,
                            "comp_valid_count": 256,
                            "struct_valid_count": 256,
                            "valid_count": 256,
                        },
                    )
                    success = cell / "evaluation/full_reconstructed/_SUCCESS"
                    success.parent.mkdir(parents=True, exist_ok=True)
                    success.touch()
        return argparse.Namespace(
            generation_run=generation,
            eval_run=evaluation,
            official_cache_run=cache_root,
            eval_runtime=runtime,
            output_dir=output,
        )

    def test_writes_immutable_complete_package(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_fixture(Path(directory))
            runtime = MODULE.RuntimeAPI(
                denominator=FakeRuntime.denominator,
                phase_diagrams=FakeRuntime.phase_diagrams,
                evaluate_cell=FakeRuntime.evaluate_cell,
            )
            report = MODULE.finalize(
                args, runtime=runtime, bootstrap_replicates=200
            )
            self.assertEqual(report["classification"]["code"], "P")
            self.assertEqual(report["endpoint_hierarchy"], list(MODULE.ENDPOINT_HIERARCHY))
            self.assertEqual(len(report["cells"]), 12)
            pooled = report["continuous_endpoints"]["refined_chgnet"][
                "fixed_two_training_seed_cohort"
            ]
            self.assertTrue(pooled["does_not_estimate_training_seed_population_variance"])
            self.assertEqual(pooled["compositions_observed"], 256)
            output = args.output_dir
            for name in (
                f"{MODULE.STEM}.md",
                f"{MODULE.STEM}.json",
                f"{MODULE.STEM}.csv",
                "INPUTS.sha256",
                "OUTPUTS.sha256",
                "_SUCCESS",
            ):
                self.assertTrue((output / name).is_file(), name)
            markdown = (output / f"{MODULE.STEM}.md").read_text(encoding="utf-8")
            self.assertLess(markdown.index("Refined CHGNet"), markdown.index("Official e_hull"))
            self.assertLess(markdown.index("Official e_hull"), markdown.index("Raw CHGNet"))
            self.assertIn("not a hard promotion gate", markdown)
            self.assertIn("never mapped to unstable", markdown)
            with (output / f"{MODULE.STEM}.csv").open(encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
            self.assertIn("classification", {row["record_type"] for row in records})
            for raw in (output / "OUTPUTS.sha256").read_text(encoding="utf-8").splitlines():
                digest, relative = raw.split("  ", 1)
                self.assertEqual(digest, MODULE.sha256_file(output / relative))
            with self.assertRaises(FileExistsError):
                MODULE.finalize(args, runtime=runtime, bootstrap_replicates=10)


if __name__ == "__main__":
    unittest.main()
