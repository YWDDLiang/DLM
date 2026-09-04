from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "scripts" / "train_full_mp20_transaction_value.py"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
MODULE = None
if TORCH_AVAILABLE:
    SPEC = importlib.util.spec_from_file_location(
        "train_full_mp20_transaction_value", SCRIPT
    )
    MODULE = importlib.util.module_from_spec(SPEC)
    assert SPEC and SPEC.loader
    sys.modules[SPEC.name] = MODULE
    SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is unavailable")
class FullMP20TransactionValueTrainerTest(unittest.TestCase):
    class TinyTokenizer:
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"input_ids": [ord(value) % 127 + 1 for value in text]}

    def row(self, source_idx=0, *, failed=False):
        source_answer = "abcdefghi"
        active = [1, 2, 3]
        noop = [ord(source_answer[value]) % 127 + 1 for value in active]
        changed = list(noop)
        changed[-1] += 1
        return {
            "source_idx": source_idx,
            "source_weight": 1.0,
            "state": {
                "prompt": "p",
                "source_answer": source_answer,
                "active_positions": active,
                "deployment_stage": "anchor_first",
                "species_program": ["Li", "O"],
            },
            "candidates": [
                {
                    "action_token_ids": noop,
                    "legality": True,
                    "terminal_single_point_energy_eV_per_atom": -1.0,
                    "terminal_basin_energy_eV_per_atom": -2.0,
                },
                {
                    "action_token_ids": changed,
                    "legality": True,
                    "terminal_single_point_energy_eV_per_atom": -1.5,
                    "terminal_basin_energy_eV_per_atom": -3.5,
                },
            ],
            "failed": failed,
        }

    def dataset(self, rows):
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "groups.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        dataset = MODULE.FullMP20TransactionValueDataset(
            path,
            self.TinyTokenizer(),
            fields=MODULE.FieldConfig(),
            expected_rows=len(rows),
            require_llama_program=True,
        )
        return temporary, dataset

    def test_exact_1696_plus_1696_update_plan(self):
        self.assertEqual(MODULE.EXPECTED_ROWS, 27136)
        self.assertEqual(MODULE.GLOBAL_BATCH_SIZE, 16)
        self.assertEqual(MODULE.POSTERIOR_UPDATES, 1696)
        self.assertEqual(MODULE.CLEAN_CE_UPDATES, 1696)
        self.assertEqual(MODULE.TOTAL_UPDATES, 3392)
        counts = Counter(
            MODULE.optimizer_objective(update)
            for update in range(1, MODULE.TOTAL_UPDATES + 1)
        )
        self.assertEqual(
            counts,
            Counter({"clean_ce": 1696, "transaction_posterior": 1696}),
        )

    def test_two_ranks_are_disjoint_and_cover_one_full_epoch(self):
        permutation = MODULE.frozen_source_permutation(27136, seed=99017)
        seen = [Counter(), Counter()]
        for batch_index in range(1696):
            left = MODULE.rank_batch_indices(permutation, batch_index, 0)
            right = MODULE.rank_batch_indices(permutation, batch_index, 1)
            self.assertFalse(set(left) & set(right))
            seen[0].update(left)
            seen[1].update(right)
        combined = seen[0] + seen[1]
        self.assertEqual(set(combined), set(range(27136)))
        self.assertEqual(set(combined.values()), {1})

    def test_route_reads_only_its_declared_value_field(self):
        temporary, dataset = self.dataset([self.row()])
        try:
            single = dataset.materialize(0, route="single_point_full")
            basin = dataset.materialize(0, route="basin_consistent_full")
            self.assertEqual(single["energies"].tolist(), [-1.0, -1.5])
            self.assertEqual(basin["energies"].tolist(), [-2.0, -3.5])
            self.assertEqual(
                single["value_field"],
                "terminal_single_point_energy_eV_per_atom",
            )
            self.assertEqual(
                basin["value_field"],
                "terminal_basin_energy_eV_per_atom",
            )
        finally:
            temporary.cleanup()

    def test_failed_group_is_retained_and_forces_zero_posterior(self):
        temporary, dataset = self.dataset([self.row(failed=True)])
        try:
            self.assertEqual(len(dataset), 1)
            materialized = dataset.materialize(
                0, route="basin_consistent_full"
            )
            self.assertTrue(materialized["force_zero_posterior"])
            self.assertEqual(
                dataset.summary()["failure_or_declared_uninformative_rows_retained"],
                1,
            )
        finally:
            temporary.cleanup()

    def test_schema_adapter_accepts_row_level_energy_arrays(self):
        row = self.row()
        single = []
        basin = []
        for candidate in row["candidates"]:
            single.append(candidate.pop("terminal_single_point_energy_eV_per_atom"))
            basin.append(candidate.pop("terminal_basin_energy_eV_per_atom"))
            candidate["valid_action"] = candidate.pop("legality")
        row["single"] = single
        row["basin"] = basin
        fields = MODULE.FieldConfig(
            candidate_legality="valid_action",
            single_point_energy="single",
            basin_energy="basin",
        )
        temporary = tempfile.TemporaryDirectory()
        try:
            path = Path(temporary.name) / "groups.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = MODULE.FullMP20TransactionValueDataset(
                path,
                self.TinyTokenizer(),
                fields=fields,
                expected_rows=1,
                require_llama_program=True,
            )
            self.assertEqual(
                dataset.materialize(0, route="single_point_full")[
                    "energies"
                ].tolist(),
                [-1.0, -1.5],
            )
        finally:
            temporary.cleanup()

    def test_manual_allreduce_averages_gradients(self):
        torch = MODULE.torch
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        parameter.grad = torch.tensor([2.0])

        def add_other_rank(tensor):
            tensor.add_(torch.tensor([4.0]))

        MODULE.average_gradients_(
            [parameter], world_size=2, all_reduce_fn=add_other_rank
        )
        self.assertEqual(parameter.grad.tolist(), [3.0])

    def test_only_step3392_and_single_route_slurm_contract(self):
        self.assertEqual(MODULE.checkpoint_steps(), (3392,))
        wrapper = (
            ROOT / "slurm" / "194_train_full_mp20_transaction_values.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("--gres=gpu:NVIDIAA800-SXM4-80GB:2", wrapper)
        self.assertIn("--cpus-per-task=8", wrapper)
        self.assertIn("--nproc_per_node=2", wrapper)
        self.assertIn("--route \"${ROUTE}\"", wrapper)
        self.assertIn("--require-llama-program", wrapper)
        self.assertNotIn(" &", wrapper)
        self.assertEqual(wrapper.count("torch.distributed.run"), 1)

    def test_source_weight_and_llama_program_are_mandatory(self):
        bad_weight = self.row()
        bad_weight["source_weight"] = 0.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(bad_weight) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_weight=1"):
                MODULE.FullMP20TransactionValueDataset(
                    path,
                    self.TinyTokenizer(),
                    fields=MODULE.FieldConfig(),
                    expected_rows=1,
                    require_llama_program=True,
                )
        no_program = self.row()
        del no_program["state"]["species_program"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(no_program) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Llama species program"):
                MODULE.FullMP20TransactionValueDataset(
                    path,
                    self.TinyTokenizer(),
                    fields=MODULE.FieldConfig(),
                    expected_rows=1,
                    require_llama_program=True,
                )


if __name__ == "__main__":
    unittest.main()
