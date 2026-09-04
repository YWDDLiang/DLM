from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "label_spad_basin_preflight_values",
    ROOT / "scripts" / "label_spad_basin_preflight_values.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import SPAD basin preflight value labeler")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeStructure:
    def __init__(self, candidate_id: int, sites: int = 2):
        self.candidate_id = candidate_id
        self.sites = sites

    def __len__(self):
        return self.sites


class FakePredictor:
    def __init__(self, energies: dict[int, float]):
        self.energies = energies
        self.calls: list[tuple[str, int | None, int]] = []

    def predict_structure(self, structures, *, task, batch_size=None):
        values = structures if isinstance(structures, list) else [structures]
        self.calls.append((task, batch_size, len(values)))
        rows = [
            {
                "e": self.energies[value.candidate_id],
                "f": np.asarray([[0.03, 0.0, 0.0]] * len(value)),
                "s": np.eye(3) * 0.01,
            }
            for value in values
        ]
        return rows if isinstance(structures, list) else rows[0]


class FakeRelaxer:
    def __init__(self, frame_energies: dict[int, list[float]], *, steps_taken: int):
        self.frame_energies = frame_energies
        self.steps_taken = steps_taken
        self.calls: list[dict[str, object]] = []

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
        per_atom = self.frame_energies[structure.candidate_id]
        energies = [value * len(structure) for value in per_atom]
        frames = len(energies)
        trajectory = SimpleNamespace(
            energies=energies,
            forces=[np.asarray([[0.02, 0.0, 0.0]] * len(structure))] * frames,
            stresses=[np.eye(3) * 0.001] * frames,
            steps_taken=self.steps_taken,
        )
        return {"trajectory": trajectory, "final_structure": structure}


def candidate(
    candidate_idx: int,
    *,
    source: str,
    legal: bool = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_idx": candidate_idx,
        "action_id": f"action-{candidate_idx}",
        "action_source": source,
        "terminal_legal": legal,
        "payload": f"keep-{candidate_idx}",
    }
    if legal:
        row["terminal_structure"] = {
            "lengths": [4.0, 4.0, 4.0],
            "angles": [90.0, 90.0, 90.0],
            "species": ["Li", "O"],
            "frac_coords": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        }
        row["terminal_token_ids"] = [candidate_idx, 10, 20]
    else:
        row["failure"] = "continuation_failed"
    return row


def group(sample_idx: int, candidates: list[dict[str, object]]):
    return {
        "schema": MODULE.INPUT_SCHEMA,
        "sample_idx": sample_idx,
        "state_type": "cell" if sample_idx % 2 == 0 else "xyz",
        "cursor": {"bucket": "early", "site": 0},
        "state_diagnostics": {"must_survive": True},
        "candidates": candidates,
    }


def loader(row):
    return FakeStructure(int(row["candidate_idx"]))


def make_labelled_group(
    energies: dict[str, tuple[float, ...]],
    *,
    sources: tuple[str, ...] = ("no_op", "force", "sample"),
):
    rows = [candidate(index, source=source) for index, source in enumerate(sources)]
    value = group(0, rows)
    for index, row in enumerate(rows):
        row["basin_value"] = {
            "schema": MODULE.CANDIDATE_LABEL_SCHEMA,
            "E0": {
                "known": True,
                "energy_eV_per_atom": energies["E0"][index],
                "error": None,
            },
            "trajectory": {
                "horizons": {
                    str(horizon): {
                        "known": True,
                        "energy_eV_per_atom": energies[f"K{horizon}"][index],
                        "error": None,
                    }
                    for horizon in MODULE.HORIZONS
                },
                "error": None,
            },
            "cache_identity": {
                "key": f"candidate-{index}",
                "terminal": {
                    "field": "terminal_token_ids",
                    "identity": f"tokens-{index}",
                },
                "chgnet_package_version": "fake",
                "chgnet_model": "fake-0.3",
                "horizons": [3, 5, 10, 20],
                "fmax_eV_per_A": 0.1,
                "relax_cell": True,
            },
        }
    value["basin_headroom"] = MODULE.group_headroom(value)
    value["schema"] = MODULE.OUTPUT_SCHEMA
    return value


class BasinPreflightValueTest(unittest.TestCase):
    def test_one_trajectory_supplies_all_horizons_and_reuses_early_endpoint(self):
        rows = [candidate(0, source="no_op"), candidate(1, source="force")]
        predictor = FakePredictor({0: -1.0, 1: -1.1})
        # Frame 0, steps 1..5, then CHGNet's duplicate endpoint frame.
        relaxer = FakeRelaxer(
            {
                0: [-1.0, -1.01, -1.02, -1.03, -1.04, -1.05, -1.05],
                1: [-1.1, -1.11, -1.12, -1.13, -1.14, -1.15, -1.15],
            },
            steps_taken=5,
        )
        labelled = MODULE.label_groups(
            [group(0, rows)],
            predictor=predictor,
            relaxer=relaxer,
            structure_loader=loader,
        )
        self.assertEqual(len(relaxer.calls), 2)
        self.assertEqual(Counter(call["steps"] for call in relaxer.calls), {20: 2})
        self.assertTrue(all(call["relax_cell"] is True for call in relaxer.calls))
        self.assertTrue(all(call["fmax"] == 0.1 for call in relaxer.calls))
        first = labelled[0]["candidates"][0]
        self.assertEqual(first["K3_energy_eV_per_atom"], -1.03)
        self.assertEqual(first["K5_energy_eV_per_atom"], -1.05)
        self.assertEqual(first["K10_energy_eV_per_atom"], -1.05)
        self.assertEqual(first["K20_energy_eV_per_atom"], -1.05)
        self.assertEqual(first["terminal_single_point_energy_eV_per_atom"], -1.0)
        self.assertEqual(first["terminal_relax_k10_energy_eV_per_atom"], -1.05)
        horizons = first["basin_value"]["trajectory"]["horizons"]
        self.assertFalse(horizons["3"]["endpoint_reused"])
        self.assertTrue(horizons["10"]["endpoint_reused"])
        self.assertEqual(horizons["20"]["frame_index"], 5)
        self.assertEqual(predictor.calls, [("efsm", 16, 2)])

    def test_invalid_and_k1_groups_are_retained_without_relaxation(self):
        row = candidate(0, source="no_op", legal=False)
        relaxer = FakeRelaxer({}, steps_taken=0)
        labelled = MODULE.label_groups(
            [group(0, [row])],
            predictor=FakePredictor({}),
            relaxer=relaxer,
            structure_loader=loader,
        )
        self.assertEqual(len(labelled), 1)
        self.assertEqual(labelled[0]["K"], 1)
        self.assertEqual(len(labelled[0]["candidates"]), 1)
        self.assertEqual(labelled[0]["candidates"][0]["payload"], "keep-0")
        self.assertFalse(labelled[0]["candidates"][0]["E0_known"])
        self.assertIn(
            "terminal_illegal",
            labelled[0]["candidates"][0]["basin_value"]["E0"]["error"],
        )
        self.assertEqual(relaxer.calls, [])

    def test_headroom_sign_thresholds_and_winning_source(self):
        energies = {
            metric: (-1.0, -1.03, -0.9)
            for metric in ("E0", "K3", "K5", "K10", "K20")
        }
        labelled = make_labelled_group(energies)
        summary = MODULE._aggregate_headroom([labelled])
        k3 = labelled["basin_headroom"]["by_metric"]["K3"]
        self.assertAlmostEqual(k3["best_minus_no_op_eV_per_atom"], -0.03)
        self.assertAlmostEqual(k3["headroom_meV_per_atom"], 30.0)
        self.assertEqual(k3["winning_action_sources"], ["force"])
        self.assertEqual(
            summary["K3"]["groups_above_headroom_threshold_meV"],
            {"5": 1, "10": 1, "20": 1, "50": 0},
        )

    def test_ties_and_kendall_tau_b_are_explicit(self):
        direct = MODULE.kendall_tau_b([0.0, 1.0, 1.0], [0.0, 1.0, 2.0])
        self.assertEqual(direct["left_only_ties"], 1)
        self.assertIsNotNone(direct["tau_b"])
        self.assertGreater(direct["tau_b"], 0.0)
        self.assertLess(direct["tau_b"], 1.0)

        energies = {
            "E0": (0.0, 1.0, 1.0),
            "K3": (0.0, 1.0, 1.0),
            "K5": (0.0, 1.0, 2.0),
            "K10": (0.0, 1.0, 2.0),
            "K20": (0.0, 1.0, 2.0),
        }
        report = MODULE._aggregate_kendall([make_labelled_group(energies)])
        self.assertEqual(report["K3_vs_K5"]["groups_with_defined_tau_b"], 1)
        self.assertEqual(report["K3_vs_K5"]["left_only_ties"], 1)

    def test_exact_input_validation_and_sharding(self):
        groups = [
            group(index, [candidate(0, source="no_op")])
            for index in range(MODULE.EXPECTED_GROUPS)
        ]
        MODULE.validate_action_groups(groups)
        shards = [
            MODULE.select_groups_for_shard(groups, shard_rank=rank, shard_count=2)
            for rank in range(2)
        ]
        self.assertEqual([len(value) for value in shards], [64, 64])
        self.assertTrue(
            {row["sample_idx"] for row in shards[0]}.isdisjoint(
                {row["sample_idx"] for row in shards[1]}
            )
        )
        with self.assertRaisesRegex(ValueError, "exactly 128"):
            MODULE.validate_action_groups(groups[:-1])
        groups[1]["sample_idx"] = 2
        with self.assertRaisesRegex(ValueError, "ordered and contiguous"):
            MODULE.validate_action_groups(groups)

    def test_accepts_builder_nested_cursor_terminal_arrays_and_failure_name(self):
        groups = []
        for index in range(MODULE.EXPECTED_GROUPS):
            valid = candidate(0, source="no_op")
            valid["terminal_arrays"] = valid.pop("terminal_structure")
            valid["terminal_body_token_ids"] = valid.pop("terminal_token_ids")
            value = group(index, [valid])
            value["state"] = {"cursor": value.pop("cursor")}
            groups.append(value)
        MODULE.validate_action_groups(groups)
        loaded = MODULE.label_groups(
            groups[:1],
            predictor=FakePredictor({0: -1.0}),
            relaxer=FakeRelaxer(
                {0: [-1.0] * 22},
                steps_taken=20,
            ),
            structure_loader=loader,
        )
        identity = loaded[0]["candidates"][0]["basin_value"]["cache_identity"]
        self.assertEqual(identity["terminal"]["field"], "terminal_body_token_ids")

        invalid = candidate(0, source="no_op", legal=False)
        invalid["terminal_failure"] = invalid.pop("failure")
        bad_groups = []
        for index in range(MODULE.EXPECTED_GROUPS):
            value = group(index, [dict(invalid)])
            value["state"] = {"cursor": value.pop("cursor")}
            bad_groups.append(value)
        MODULE.validate_action_groups(bad_groups)

    def test_merge_preserves_all_groups_and_recomputes_final_report(self):
        energies = {
            metric: (-1.0,) for metric in ("E0", "K3", "K5", "K10", "K20")
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for rank in range(2):
                rows = []
                for index in range(rank, MODULE.EXPECTED_GROUPS, 2):
                    value = make_labelled_group(energies, sources=("no_op",))
                    value["sample_idx"] = index
                    value["state_type"] = "cell" if index % 2 == 0 else "xyz"
                    rows.append(value)
                with (root / f"labelled_groups_rank{rank}.jsonl").open(
                    "w", encoding="utf-8"
                ) as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
                report = MODULE.summarize_groups(
                    rows,
                    scope="shard",
                    shard_rank=rank,
                    shard_count=2,
                )
                (root / f"report_rank{rank}.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
            final = MODULE.merge_shards(root, shard_count=2)
            self.assertEqual(final["groups"], 128)
            self.assertEqual(final["K_histogram"], {"1": 128})
            self.assertTrue((root / "labelled_groups.jsonl").is_file())
            self.assertTrue((root / "PRELIGHT_VALUE_FINAL.json").is_file())
            self.assertTrue((root / "_SUCCESS").is_file())

    def test_wrapper_uses_two_a800s_eight_cpus_and_two_rank_processes(self):
        wrapper = (
            ROOT / "slurm" / "212_label_spad_basin_preflight_values.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=8", wrapper)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2", wrapper)
        self.assertIn("OMP_NUM_THREADS=4", wrapper)
        self.assertIn("--shard-rank 0 --shard-count 2", wrapper)
        self.assertIn("--shard-rank 1 --shard-count 2", wrapper)
        self.assertGreaterEqual(wrapper.count("CUDA_VISIBLE_DEVICES=\"${gpu["), 2)
        self.assertIn("wait \"${pid0}\"", wrapper)
        self.assertIn("wait \"${pid1}\"", wrapper)
        self.assertNotIn("--array", wrapper)


if __name__ == "__main__":
    unittest.main()
