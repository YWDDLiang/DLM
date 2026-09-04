import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_spad_basin_closure_sft_data",
    ROOT / "scripts/build_spad_basin_closure_sft_data.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import SPAD basin-closure SFT builder")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def answer_for(elements, counts):
    tokens = [
        f"<N_{sum(counts):03d}>",
        "<LA_040>",
        "<LB_041>",
        "<LC_042>",
        "<AA_090>",
        "<AB_091>",
        "<AG_092>",
    ]
    slot = 0
    for element, count in zip(elements, counts, strict=True):
        for _ in range(count):
            tokens.extend(
                [
                    f"<E_{element}>",
                    f"<X_{10 + slot:03d}>",
                    f"<Y_{20 + slot:03d}>",
                    f"<Z_{30 + slot:03d}>",
                ]
            )
            slot += 1
    return "".join(tokens)


def source(source_idx, split="train"):
    elements = ["O", "Na", "Cl"]
    counts = [2, 1, 1]
    return {
        "source_row_idx": source_idx,
        "source_split": split,
        "prompt": "plan",
        "answer": answer_for(elements, counts),
        "plan_state": {"N": 4, "elements": elements, "counts": counts},
    }


class SPADBasinClosureSFTBuilderTest(unittest.TestCase):
    pointer = (["Cl", "O", "Na"], "frozen_planner_llama_pointer")

    @staticmethod
    def trainer_recognizes_rollout_masks(row):
        """Mirror llada_sft.py --require-rollout-masks row recognition."""

        return (
            str(row.get("schema") or "") == "rollout_matched_transition_v1"
            and "forced_mask_positions" in row
            and "loss_positions" in row
        )

    def test_cell_states_match_sequential_runtime_masks(self):
        coordinate_set = {
            position
            for slot in range(4)
            for position in MODULE.coordinate_positions(slot)
        }
        for component in range(6):
            row = MODULE.build_closure_row(
                source(component),
                source_idx=component,
                pointer=self.pointer,
                seed=7,
                state_index=component,
            )
            self.assertEqual(row["forced_mask_positions"], list(range(1 + component, 7)))
            self.assertEqual(row["loss_positions"], [1 + component])
            self.assertFalse(set(row["forced_mask_positions"]) & coordinate_set)
            self.assertTrue(row["closure"]["all_coordinates_visible"])

    def test_reverse_program_block_and_progressive_component_states(self):
        # Runtime helper reverses both the program and each species' predictor
        # order, leaving the same-species anchor until last.
        first = MODULE.build_closure_row(
            source(10),
            source_idx=10,
            pointer=self.pointer,
            seed=7,
            state_index=6,
        )
        self.assertEqual(first["closure"]["reverse_block_order"], ["Na", "O", "Cl"])
        self.assertEqual(first["closure"]["species"], "Na")
        self.assertEqual(first["closure"]["block_slot_indices"], [2])
        self.assertEqual(first["forced_mask_positions"], [16, 17, 18])
        self.assertEqual(first["loss_positions"], [16])

        o_second_site_y = MODULE.build_closure_row(
            source(11),
            source_idx=11,
            pointer=self.pointer,
            seed=7,
            state_index=6 + 3 + 4,
        )
        self.assertEqual(o_second_site_y["closure"]["species"], "O")
        self.assertEqual(o_second_site_y["closure"]["block_slot_indices"], [1, 0])
        self.assertEqual(o_second_site_y["closure"]["site_slot_index"], 0)
        self.assertEqual(o_second_site_y["closure"]["coordinate_component"], "y")
        self.assertEqual(o_second_site_y["forced_mask_positions"], [9, 10])
        self.assertEqual(o_second_site_y["loss_positions"], [9])
        self.assertTrue(o_second_site_y["closure"]["lattice_visible"])
        self.assertTrue(o_second_site_y["closure"]["other_species_visible"])
        self.assertTrue(
            o_second_site_y["closure"]["suffix_outside_active_block_visible"]
        )

    def test_block_slots_are_the_runtime_helper_single_source_of_truth(self):
        plan = source(12)["plan_state"]
        program = MODULE.program_from_element_order(
            plan,
            self.pointer[0],
            order_source=self.pointer[1],
        )
        runtime_blocks = MODULE.reverse_species_block_revision_slots(program)
        states = MODULE.closure_states(program)
        observed = []
        for state in states[6:]:
            slots = tuple(state["metadata"]["block_slot_indices"])
            if not observed or slots != observed[-1]:
                observed.append(slots)
        self.assertEqual(tuple(observed), runtime_blocks)
        self.assertEqual(runtime_blocks, ((2,), (1, 0), (3,)))

    def test_exact_body_protected_positions_and_loss_subset(self):
        protected = {0, 7, 11, 15, 19}
        for state_index in range(18):
            row = MODULE.build_closure_row(
                source(state_index),
                source_idx=state_index,
                pointer=self.pointer,
                seed=19,
                state_index=state_index,
            )
            self.assertEqual(row["answer"], row["source_answer"])
            self.assertEqual(row["schema"], "rollout_matched_transition_v1")
            self.assertEqual(
                row["closure_schema"], "spad_basin_closure_sft_v1"
            )
            self.assertEqual(len(MODULE.parse_dynamic_answer(row["answer"], strict=True)["tokens"]), 23)
            self.assertFalse(protected & set(row["forced_mask_positions"]))
            self.assertTrue(set(row["loss_positions"]) <= set(row["forced_mask_positions"]))
            self.assertFalse(row["outcomes_read"])

    def test_deterministic_assignment_and_declared_fallback(self):
        first = MODULE.build_closure_row(
            source(44), source_idx=44, pointer=None, seed=101
        )
        second = MODULE.build_closure_row(
            source(44), source_idx=44, pointer=None, seed=101
        )
        self.assertEqual(first, second)
        self.assertEqual(first["species_program"], ["O", "Na", "Cl"])
        self.assertEqual(first["species_program_source"], MODULE.FALLBACK_SOURCE)
        self.assertFalse(first["closure"]["pointer_semantics_available"])

    def test_formal_require_rollout_masks_recognizes_closure_rows(self):
        rows = [
            MODULE.build_closure_row(
                source(state_index),
                source_idx=state_index,
                pointer=self.pointer,
                seed=17,
                state_index=state_index,
            )
            for state_index in range(18)
        ]
        recognized = sum(self.trainer_recognizes_rollout_masks(row) for row in rows)
        self.assertEqual(recognized, len(rows))
        self.assertTrue(
            all(row["closure_schema"] == MODULE.CLOSURE_SCHEMA for row in rows)
        )

    def test_full_train_val_split_counts_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = root / "teacher"
            pointer = root / "pointer"
            output = root / "output"
            teacher.mkdir()
            pointer.mkdir()
            output.mkdir()
            expected = {"train": 7, "val": 5}
            for split, count in expected.items():
                with (teacher / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                    for source_idx in range(count):
                        handle.write(json.dumps(source(source_idx, split)) + "\n")
                with (pointer / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                    for source_idx in range(count - 1):
                        handle.write(
                            json.dumps(
                                {
                                    "source_row_idx": source_idx,
                                    "species_program": ["Cl", "O", "Na"],
                                    "species_program_source": "frozen_planner_llama_pointer",
                                }
                            )
                            + "\n"
                        )
                report = MODULE.build_split(
                    source_path=teacher / f"{split}.jsonl",
                    pointer_path=pointer / f"{split}.jsonl",
                    output_path=output / f"{split}.jsonl",
                    seed=31,
                )
                rows = list(MODULE.iter_jsonl(output / f"{split}.jsonl"))
                self.assertEqual(report["rows"], count)
                self.assertEqual(len(rows), count)
                self.assertEqual(
                    [row["source_row_idx"] for row in rows], list(range(count))
                )
                self.assertEqual(
                    rows[-1]["species_program_source"], MODULE.FALLBACK_SOURCE
                )


if __name__ == "__main__":
    unittest.main()
