import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer
from crystal_dlm.manifold_corruption import CorruptionConfig


SPEC = importlib.util.spec_from_file_location(
    "build_pmtr_preflight_data",
    ROOT / "scripts" / "build_pmtr_preflight_data.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import PMTR preflight builder")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source(source_idx=3):
    elements = ["O", "Na", "Cl"]
    counts = [2, 1, 1]
    answer, _ = arrays_to_dynamic_answer(
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
        "source_row_idx": source_idx,
        "source_split": "train",
        "prompt": "compact plan",
        "answer": answer,
        "plan_state": {"N": 4, "elements": elements, "counts": counts},
        "species_program": ["Cl", "O", "Na"],
        "species_program_source": "frozen_planner_llama_pointer",
    }


def certificate_row(source_row_idx=3, proposal_index=0):
    return {
        "source_row_idx": source_row_idx,
        "source_split": "train",
        "proposal_index": proposal_index,
        "post_quantization_valid": True,
        "delta_energy": 0.3,
        "coordinate_force_dot_clean_retraction": 0.7,
        "lattice_descent_dot_spd_retraction": 0.5,
    }


class PMTRPreflightBuilderTest(unittest.TestCase):
    config = CorruptionConfig(
        lattice_log_std=0.10,
        coordinate_cartesian_std_A=0.28,
        max_delta_energy=1.0,
    )

    def certified_selection(self, row):
        lookup = MODULE.CertificateLookup([certificate_row(row["source_row_idx"])])
        return MODULE.select_source_corruption(
            row, certificates=lookup, seed=17, config=self.config
        )

    def test_all_component_roles_follow_closure_states_and_keep_one_corruption(self):
        row = source()
        selection = self.certified_selection(row)
        self.assertFalse(selection.fallback)
        proposal = selection.proposal
        assert proposal is not None
        program = MODULE.program_from_element_order(
            row["plan_state"],
            row["species_program"],
            order_source=row["species_program_source"],
        )
        states = MODULE.closure_states(program)
        repair_order = [state["loss"][0] for state in states]
        clean_tokens = tuple(parse_dynamic_answer(row["answer"], strict=True)["tokens"])
        corrupt_tokens = proposal.tokens
        protected = {0, 7, 11, 15, 19}
        observed_cell = set()
        observed_coordinate = set()

        for index, state in enumerate(states):
            built = MODULE.build_repair_row(
                row,
                source_idx=row["source_row_idx"],
                program_order=row["species_program"],
                program_source=row["species_program_source"],
                selection=selection,
                seed=17,
                state_index=index,
            )
            source_tokens = tuple(
                parse_dynamic_answer(built["source_answer"], strict=True)["tokens"]
            )
            self.assertEqual(built["answer"], row["answer"])
            self.assertEqual(built["forced_mask_positions"], state["forced"])
            self.assertEqual(built["loss_positions"], state["loss"])
            self.assertEqual(len(source_tokens), 7 + 4 * 4)
            self.assertEqual(
                parse_dynamic_answer(built["source_answer"], strict=True)["species"],
                ["O", "O", "Na", "Cl"],
            )
            repaired = protected | set(repair_order[:index])
            for position in range(len(source_tokens)):
                expected = (
                    clean_tokens[position]
                    if position in repaired
                    else corrupt_tokens[position]
                )
                self.assertEqual(source_tokens[position], expected)
            if state["kind"] == "cell_sequential_component":
                observed_cell.add(state["metadata"]["cell_component"])
                self.assertEqual(built["repair_target"]["kind"], "cell")
                self.assertEqual(len(built["repair_target"]["lattice_tangent"]), 3)
                self.assertIsNone(built["repair_target"]["site_slot_index"])
            else:
                observed_coordinate.add(state["metadata"]["coordinate_component"])
                self.assertEqual(built["repair_target"]["kind"], "site")
                self.assertEqual(
                    built["repair_target"]["site_slot_index"],
                    state["metadata"]["site_slot_index"],
                )
                self.assertEqual(
                    len(built["repair_target"]["cartesian_site_delta_A"]), 3
                )

        self.assertEqual(observed_cell, {"a", "b", "c", "alpha", "beta", "gamma"})
        self.assertEqual(observed_coordinate, {"x", "y", "z"})

    def test_reverse_species_state_does_not_splice_a_clean_storage_suffix(self):
        row = source()
        selection = self.certified_selection(row)
        program = MODULE.program_from_element_order(
            row["plan_state"], row["species_program"], order_source="pointer"
        )
        states = MODULE.closure_states(program)
        # First O coordinate follows the completed Na block.  Storage order and
        # repair order disagree here, which catches a naive clean-suffix splice.
        state_index = 9
        built = MODULE.build_repair_row(
            row,
            source_idx=3,
            program_order=row["species_program"],
            program_source="pointer",
            selection=selection,
            seed=17,
            state_index=state_index,
        )
        actual = tuple(parse_dynamic_answer(built["source_answer"], strict=True)["tokens"])
        clean = tuple(parse_dynamic_answer(row["answer"], strict=True)["tokens"])
        corrupt = selection.proposal.tokens
        repair_order = [state["loss"][0] for state in states]
        repaired = {0, 7, 11, 15, 19, *repair_order[:state_index]}
        self.assertEqual(
            actual,
            tuple(clean[i] if i in repaired else corrupt[i] for i in range(len(actual))),
        )
        later_unrepaired = set(repair_order[state_index + 1 :])
        self.assertTrue(later_unrepaired)
        self.assertTrue(all(actual[i] == corrupt[i] for i in later_unrepaired))

    def test_fallback_is_clean_ce_with_exact_current_masks(self):
        row = source(8)
        selection = MODULE.select_source_corruption(
            row,
            certificates=MODULE.CertificateLookup(),
            seed=2,
            config=self.config,
        )
        built = MODULE.build_repair_row(
            row,
            source_idx=8,
            program_order=row["species_program"],
            program_source=row["species_program_source"],
            selection=selection,
            seed=2,
            state_index=4,
        )
        self.assertTrue(selection.fallback)
        self.assertEqual(built["pmtr"]["mode"], "clean_ce_fallback")
        self.assertIsNone(built["repair_target"])
        self.assertEqual(built["source_answer"], built["answer"])
        self.assertEqual(built["forced_mask_positions"], [5, 6])
        self.assertEqual(built["loss_positions"], [5])

    def test_cli_reads_jsonl_certificates_and_writes_compact_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            certificate_path = root / "certificates.jsonl"
            output = root / "output"
            input_path.write_text(json.dumps(source(12)) + "\n", encoding="utf-8")
            certificate_path.write_text(
                json.dumps(certificate_row(12)) + "\n", encoding="utf-8"
            )
            MODULE.main(
                [
                    "--input-jsonl",
                    str(input_path),
                    "--certificates-jsonl",
                    str(certificate_path),
                    "--output-dir",
                    str(output),
                    "--seed",
                    "17",
                    "--lattice-log-std",
                    "0.10",
                    "--coordinate-std-A",
                    "0.28",
                    "--max-delta-energy",
                    "1.0",
                ]
            )
            rows = list(MODULE.iter_jsonl(output / "data.jsonl"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["pmtr"]["mode"], "manifold_repair")
            self.assertEqual(manifest["rows"], 1)
            self.assertEqual(manifest["selection"], "first_certified_at_most_4_else_clean_ce")
            self.assertNotIn("sha", json.dumps(manifest).lower())


if __name__ == "__main__":
    unittest.main()
