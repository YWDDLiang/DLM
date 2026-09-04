from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "label_full_mp20_terminal_values",
    ROOT / "scripts" / "label_full_mp20_terminal_values.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import full MP20 terminal value labeler")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeStructure:
    def __init__(self, candidate_id: int, sites: int = 2):
        self.candidate_id = candidate_id
        self.sites = sites

    def __len__(self):
        return self.sites


class FakePredictor:
    def __init__(self, energies):
        self.energies = dict(energies)
        self.calls = []

    def predict_structure(self, structures, *, task, batch_size=None):
        self.calls.append((task, batch_size))
        values = structures if isinstance(structures, list) else [structures]
        output = [
            {
                "e": self.energies[value.candidate_id],
                "f": np.asarray([[0.05, 0.0, 0.0]] * len(value)),
                "s": np.eye(3) * 0.01,
            }
            for value in values
        ]
        return output if isinstance(structures, list) else output[0]


class FakeRelaxer:
    def __init__(self, energies_by_steps):
        self.energies_by_steps = {
            int(steps): dict(values) for steps, values in energies_by_steps.items()
        }
        self.calls = []

    def relax(self, structure, *, relax_cell, fmax, steps, verbose):
        self.calls.append(
            {
                "candidate_id": structure.candidate_id,
                "relax_cell": relax_cell,
                "fmax": fmax,
                "steps": steps,
                "verbose": verbose,
            }
        )
        energy_per_atom = self.energies_by_steps[steps][structure.candidate_id]
        frames = 12 if steps == 64 else 22
        trajectory = SimpleNamespace(
            energies=[energy_per_atom * len(structure)] * frames,
            forces=[np.asarray([[0.02, 0.0, 0.0]] * len(structure))] * frames,
            stresses=[np.eye(3) * 0.001] * frames,
            steps_taken=frames - 2,
        )
        return {"final_structure": structure, "trajectory": trajectory}


def candidate(candidate_id: int, *, valid: bool = True):
    value = {
        "candidate_idx": candidate_id,
        "valid_terminal": valid,
        "payload_that_must_survive": f"candidate-{candidate_id}",
    }
    if valid:
        value["terminal_answer"] = f"answer-{candidate_id}"
    else:
        value["failure"] = "upstream_terminal_failure"
    return value


def group(group_idx: int, candidates, *, stage: str = "anchor_first"):
    return {
        "group_idx": group_idx,
        "source": {"split": "train", "source_row_idx": group_idx},
        "stage": stage,
        "state": {"program": ["A", "B"]},
        "candidates": candidates,
    }


def loader(row):
    return FakeStructure(int(row["candidate_idx"]))


class TerminalValueLabelerTest(unittest.TestCase):
    def test_shared_candidates_receive_both_values_with_fixed_short_steps(self):
        groups = [group(0, [candidate(0), candidate(1)])]
        predictor = FakePredictor({0: -1.0, 1: -2.0})
        relaxer = FakeRelaxer({64: {0: -3.0, 1: -4.0}})

        result = MODULE.label_groups(
            groups,
            predictor=predictor,
            relaxer=relaxer,
            structure_loader=loader,
        )

        labelled = result[0]["candidates"]
        self.assertEqual([row["candidate_idx"] for row in labelled], [0, 1])
        self.assertEqual(labelled[0]["payload_that_must_survive"], "candidate-0")
        self.assertEqual(labelled[0]["terminal_single_point_energy_eV_per_atom"], -1.0)
        self.assertEqual(labelled[0]["terminal_basin_energy_eV_per_atom"], -3.0)
        self.assertTrue(labelled[0]["terminal_single_point_known"])
        self.assertTrue(labelled[0]["terminal_basin_known"])
        self.assertTrue(result[0]["terminal_value_labels_shared_candidates"])
        self.assertEqual({call["steps"] for call in relaxer.calls}, {64})
        self.assertTrue(all(call["relax_cell"] for call in relaxer.calls))
        self.assertTrue(all(call["fmax"] == 0.1 for call in relaxer.calls))
        self.assertEqual(predictor.calls, [("efsm", 16)])

    def test_invalid_and_parse_failure_are_retained_as_unknown(self):
        groups = [group(0, [candidate(0, valid=False), candidate(1)])]

        def failing_loader(row):
            raise ValueError(f"bad {row['candidate_idx']}")

        result = MODULE.label_groups(
            groups,
            predictor=FakePredictor({}),
            relaxer=FakeRelaxer({64: {}}),
            structure_loader=failing_loader,
        )
        rows = result[0]["candidates"]
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]["terminal_single_point_known"])
        self.assertFalse(rows[0]["terminal_basin_known"])
        self.assertIn("upstream_terminal_failure", rows[0]["failure"])
        self.assertIn("terminal_parse_failed", rows[1]["terminal_basin_error"])

    def test_shards_are_disjoint_and_cover_every_source(self):
        groups = [group(index, [candidate(index)]) for index in range(9)]
        partitions = [
            MODULE.select_groups_for_shard(groups, shard_rank=rank, shard_count=3)
            for rank in range(3)
        ]
        sets = [{int(row["group_idx"]) for row in part} for part in partitions]
        self.assertTrue(sets[0].isdisjoint(sets[1]))
        self.assertTrue(sets[0].isdisjoint(sets[2]))
        self.assertTrue(sets[1].isdisjoint(sets[2]))
        self.assertEqual(set.union(*sets), set(range(9)))

    def test_pairwise_agreement_and_ties(self):
        rows = [
            {
                "terminal_calibration_energy_eV_per_atom": value,
                "terminal_single_point_energy_eV_per_atom": e0,
                "terminal_basin_energy_eV_per_atom": e64,
            }
            for value, e0, e64 in (
                (0.0, 0.0, 0.0),
                (1.0, 2.0, 1.0),
                (2.0, 1.0, 2.0),
            )
        ]
        groups = [{"candidates": rows}]
        e0 = MODULE.pairwise_agreement(
            groups, metric_field="terminal_single_point_energy_eV_per_atom"
        )
        e64 = MODULE.pairwise_agreement(
            groups, metric_field="terminal_basin_energy_eV_per_atom"
        )
        self.assertEqual((e0["agreements"], e0["disagreements"]), (2, 1))
        self.assertAlmostEqual(e0["directional_accuracy_excluding_metric_ties"], 2 / 3)
        self.assertEqual((e64["agreements"], e64["disagreements"]), (3, 0))

        tied = [
            {
                "candidates": [
                    {
                        "terminal_calibration_energy_eV_per_atom": 1.0,
                        "terminal_single_point_energy_eV_per_atom": 2.0,
                    },
                    {
                        "terminal_calibration_energy_eV_per_atom": 1.0,
                        "terminal_single_point_energy_eV_per_atom": 2.0,
                    },
                ]
            }
        ]
        tie_report = MODULE.pairwise_agreement(
            tied, metric_field="terminal_single_point_energy_eV_per_atom"
        )
        self.assertEqual(tie_report["reference_ties"], 1)
        self.assertEqual(tie_report["reference_tie_agreements"], 1)

        tied[0]["candidates"][1]["terminal_single_point_energy_eV_per_atom"] = None
        missing_report = MODULE.pairwise_agreement(
            tied, metric_field="terminal_single_point_energy_eV_per_atom"
        )
        self.assertEqual(missing_report["reference_known_pairs"], 1)
        self.assertEqual(missing_report["reference_ties"], 1)
        self.assertEqual(missing_report["jointly_known_pairs"], 0)
        self.assertEqual(missing_report["coverage_of_reference_known"], 0.0)

    def test_calibration_runs_e64_and_e500_from_same_candidate(self):
        groups = [group(0, [candidate(0), candidate(1)], stage="cell")]
        relaxer = FakeRelaxer(
            {
                64: {0: -1.0, 1: -2.0},
                500: {0: -1.5, 1: -2.5},
            }
        )
        result = MODULE.label_groups(
            groups,
            predictor=FakePredictor({0: -0.5, 1: -0.6}),
            relaxer=relaxer,
            structure_loader=loader,
            calibration_full_steps=500,
        )
        rows = result[0]["candidates"]
        self.assertEqual(rows[0]["terminal_basin_energy_eV_per_atom"], -1.0)
        self.assertEqual(rows[0]["terminal_calibration_energy_eV_per_atom"], -1.5)
        self.assertEqual(
            [(call["candidate_id"], call["steps"]) for call in relaxer.calls],
            [(0, 64), (0, 500), (1, 64), (1, 500)],
        )
        report = MODULE.calibration_report(result)
        self.assertEqual(report["calibration_full_steps"], 500)
        self.assertIn("cell", report["by_stage"])
        self.assertEqual(report["by_stage"]["cell"]["variation"]["E500"]["groups_with_variation"], 1)

    def test_calibration_rejects_non_train_or_unfixed_steps(self):
        test_group = group(0, [candidate(0)])
        test_group["source"]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "train-only"):
            MODULE.label_groups(
                [test_group],
                predictor=FakePredictor({0: -1.0}),
                relaxer=FakeRelaxer({64: {0: -1.0}, 500: {0: -1.0}}),
                structure_loader=loader,
                calibration_full_steps=500,
            )
        with self.assertRaisesRegex(ValueError, "fixed at 500"):
            MODULE.label_groups(
                [group(0, [candidate(0)])],
                predictor=FakePredictor({0: -1.0}),
                relaxer=FakeRelaxer({64: {0: -1.0}}),
                structure_loader=loader,
                calibration_full_steps=499,
            )

    def test_validation_requires_terminal_geometry_but_not_schema_name(self):
        value = group(7, [candidate(0)])
        value["arbitrary_upstream_schema"] = "anything"
        MODULE.validate_groups([value])
        del value["candidates"][0]["terminal_answer"]
        with self.assertRaisesRegex(ValueError, "terminal_answer or terminal_cif"):
            MODULE.validate_groups([value])


if __name__ == "__main__":
    unittest.main()
