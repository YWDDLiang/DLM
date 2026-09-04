from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "scripts" / "train_spad_basin_posterior.py"
SPEC = importlib.util.spec_from_file_location("train_spad_basin_posterior", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def candidate(source, tokens, legal=True, **values):
    return {
        "source": source,
        "action_token_ids": list(tokens),
        "terminal_legal": legal,
        **values,
    }


def group(sample_idx, candidates, *, state_type="xyz"):
    if state_type == "xyz":
        active = [8, 9, 10]
    else:
        active = [1, 2, 3, 4, 5, 6]
    return {
        "schema": "spad_basin_preflight_action_group_v1_labelled",
        "sample_idx": sample_idx,
        "state": {
            "sample_idx": sample_idx,
            "prompt": "Generate a crystal",
            "state_body": "placeholder",
            "N": 1,
            "plan_state": {"N": 1, "elements": ["Li"]},
            "species_program": ["Li"],
            "state_type": state_type,
            "active_generation_positions": active,
            "context_masked_generation_positions": [],
        },
        "candidates": candidates,
    }


class BasinPosteriorPureTest(unittest.TestCase):
    def test_source_prompt_left_padding_is_replayed(self):
        ids, attention = MODULE.left_pad_prompt_ids(
            [7, 8, 9], target_length=5, pad_token_id=0
        )
        self.assertEqual(ids, [0, 0, 7, 8, 9])
        self.assertEqual(attention, [0, 0, 1, 1, 1])

    def test_k1_is_retained_as_exact_zero_contract(self):
        values = MODULE.normalize_candidates(
            [candidate("no_op", [10, 11, 12], terminal_relax_k10_energy_eV_per_atom=-1.0)],
            transaction_width=3,
            value_field=MODULE.DEFAULT_VALUE_FIELD,
        )
        self.assertEqual(len(values), 1)
        self.assertEqual(MODULE.zero_posterior_reason(values), "k1_retained")

    def test_less_than_two_known_values_is_zero(self):
        values = MODULE.normalize_candidates(
            [
                candidate("physics_downhill", [13, 14, 15]),
                candidate(
                    "no_op",
                    [10, 11, 12],
                    terminal_relax_k10_energy_eV_per_atom=-1.0,
                ),
            ],
            transaction_width=3,
            value_field=MODULE.DEFAULT_VALUE_FIELD,
        )
        self.assertEqual(values[0]["source"], "no_op")
        self.assertEqual(
            MODULE.zero_posterior_reason(values),
            "fewer_than_two_known_legal_values",
        )

    def test_no_op_is_first_and_dynamic_k_is_preserved(self):
        for size in (2, 3, 4):
            rows = [
                candidate(
                    f"proposal_{index}",
                    [20 + index, 30 + index, 40 + index],
                    terminal_relax_k10_energy_eV_per_atom=-1.0 - index,
                )
                for index in range(size - 1)
            ]
            rows.insert(
                1,
                candidate(
                    "no_op",
                    [10, 11, 12],
                    terminal_relax_k10_energy_eV_per_atom=-0.5,
                ),
            )
            normalized = MODULE.normalize_candidates(
                rows,
                transaction_width=3,
                value_field=MODULE.DEFAULT_VALUE_FIELD,
            )
            self.assertEqual(len(normalized), size)
            self.assertEqual(normalized[0]["source"], "no_op")

    def test_context_masks_are_part_of_deployed_scoring_contract(self):
        batch = {
            "prompt_length": 5,
            "gen_length": 19,
            "generation_positions": (8, 9, 10),
            "context_masked_generation_positions": (12, 13, 14),
            "mask_id": 126336,
            "allowed_token_ids_by_generation_pos": [[1]] * 19,
            "lightweight_decoding_constraints": {
                "duplicate_coordinate_mask": True,
                "lattice_volume_mask": True,
                "pbc_min_distance_mask": True,
                "pbc_min_distance_A": 0.5,
                "pbc_image_radius": 2,
            },
        }
        contract = MODULE.deployed_scoring_contract(batch)
        self.assertEqual(
            contract["context_masked_generation_positions"], (12, 13, 14)
        )
        self.assertEqual(contract["temperature"], 0.7)
        self.assertIsNone(contract["atom_count_grammar"])
        self.assertEqual(
            contract["lightweight_decoding_constraints"]["pbc_image_radius"], 2
        )

    def test_four_pass_schedule_covers_each_group_exactly_four_times(self):
        schedule = MODULE.deterministic_posterior_schedule()
        self.assertEqual(len(schedule), 256)
        self.assertTrue(all(len(step) == 2 for step in schedule))
        counts = Counter(index for step in schedule for index in step)
        self.assertEqual(set(counts), set(range(128)))
        self.assertEqual(set(counts.values()), {4})
        self.assertEqual(schedule, MODULE.deterministic_posterior_schedule())

    def test_optimizer_updates_strictly_alternate(self):
        objectives = [
            MODULE.optimizer_objective(update)
            for update in range(1, MODULE.TOTAL_UPDATES + 1)
        ]
        self.assertEqual(len(objectives), 512)
        self.assertEqual(objectives[::2], ["clean_ce"] * 256)
        self.assertEqual(objectives[1::2], ["transaction_posterior"] * 256)

    def test_gradient_scale_uses_nearest_allowed_power(self):
        report = MODULE.select_posterior_gradient_scale(
            [10.0] * 5,
            [1.8] * 5,
        )
        self.assertEqual(report["selected_posterior_gradient_multiplier"], 4.0)
        self.assertTrue(report["frozen_for_all_posterior_updates"])
        self.assertFalse(report["per_batch_rescaling"])

    def test_dataset_requires_exact_ordered_128_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.jsonl"
            rows = []
            for index in range(128):
                rows.append(
                    group(
                        index,
                        [
                            candidate(
                                "no_op",
                                [10, 11, 12],
                                terminal_relax_k10_energy_eV_per_atom=-1.0,
                            )
                        ],
                    )
                )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            dataset = MODULE.BasinPosteriorGroupDataset(
                path, value_field=MODULE.DEFAULT_VALUE_FIELD
            )
            self.assertEqual(len(dataset), 128)
            self.assertEqual(dataset.summary()["candidate_k_histogram"], {1: 128})


class BasinPosteriorWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = (ROOT / "slurm" / "213_train_spad_basin_posterior_pilots.sbatch").read_text(
            encoding="utf-8"
        )

    def test_wrapper_uses_one_four_gpu_allocation(self):
        self.assertIn("#SBATCH --cpus-per-task=16", self.wrapper)
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4", self.wrapper)
        self.assertIn("--nproc_per_node=2", self.wrapper)
        self.assertIn("OPENBLAS_NUM_THREADS=4", self.wrapper)

    def test_wrapper_launches_two_fixed_routes_without_selection(self):
        self.assertIn("terminal_single_point_energy_eV_per_atom", self.wrapper)
        self.assertIn("terminal_relax_k10_energy_eV_per_atom", self.wrapper)
        self.assertIn("99017", self.wrapper)
        self.assertIn("99018", self.wrapper)
        self.assertIn("E0_MASTER_PORT", self.wrapper)
        self.assertIn("K10_MASTER_PORT", self.wrapper)
        self.assertIn("automatic_result_choice", self.wrapper)
        self.assertIn("automatic_result_choice\": False", self.wrapper)
        self.assertNotIn("train_full_mp20_transaction_value.py", self.wrapper)
        self.assertNotIn("train_potential_closure.py", self.wrapper)

    def test_wrapper_requires_explicit_preflight_authorization(self):
        self.assertIn("SPAD_BASIN_LABEL_RUN", self.wrapper)
        self.assertIn("PRELIGHT_TRAINING_AUTHORIZED", self.wrapper)
        self.assertIn("--authorization-marker", self.wrapper)


if __name__ == "__main__":
    unittest.main()
