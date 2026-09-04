import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_spad_basin_preflight_states.py"
SPEC = importlib.util.spec_from_file_location("materialize_spad_basin_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import basin preflight state materializer")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


OFFSETS = {
    "N": 10_000,
    "LA": 20_000,
    "LB": 21_000,
    "LC": 22_000,
    "AA": 23_000,
    "AB": 24_000,
    "AG": 25_000,
    "X": 30_000,
    "Y": 31_000,
    "Z": 32_000,
}
NUMERIC = re.compile(r"^<(N|LA|LB|LC|AA|AB|AG|X|Y|Z)_(\d{3})>$")


def token_id(token):
    match = NUMERIC.fullmatch(token)
    if match is None:
        raise ValueError(f"fixture has no action ID for {token}")
    return OFFSETS[match.group(1)] + int(match.group(2))


def coordinate_positions(slot):
    return [8 + 4 * slot, 9 + 4 * slot, 10 + 4 * slot]


def shifted_coordinate(token, amount):
    match = NUMERIC.fullmatch(token)
    if match is None or match.group(1) not in {"X", "Y", "Z"}:
        raise ValueError(token)
    return f"<{match.group(1)}_{(int(match.group(2)) + amount) % 101:03d}>"


def plan_shape(index):
    if index < 24:
        if index % 2:
            return ["Li", "O"], [7, 6]
        return ["Li", "P", "O"], [6, 4, 3]
    if index < 48:
        if index % 2:
            return ["Na", "Cl"], [6, 2]
        return ["Li", "B", "O"], [6, 1, 1]
    choice = index % 4
    if choice == 0:
        return ["Li", "O"], [2, 2]
    if choice == 1:
        return ["Na", "Al", "O"], [3, 2, 3]
    if choice == 2:
        return ["Mg", "O"], [5, 5]
    return ["Li", "P", "O"], [4, 3, 5]


def make_fixture(index):
    elements, counts = plan_shape(index)
    n = sum(counts)
    species = [element for element, count in zip(elements, counts) for _ in range(count)]
    predictor = [
        f"<N_{n:03d}>",
        f"<LA_{40 + index % 7:03d}>",
        f"<LB_{42 + index % 7:03d}>",
        f"<LC_{44 + index % 7:03d}>",
        "<AA_090>",
        "<AB_091>",
        "<AG_089>",
    ]
    for slot, element in enumerate(species):
        base = (index * 7 + slot * 11) % 101
        predictor.extend(
            [
                f"<E_{element}>",
                f"<X_{base:03d}>",
                f"<Y_{(base + 17) % 101:03d}>",
                f"<Z_{(base + 31) % 101:03d}>",
            ]
        )
    cell_positions = list(range(1, 7))
    cell_previous = [token_id(predictor[position]) for position in cell_positions]
    cell_new_tokens = []
    for token in predictor[1:7]:
        match = NUMERIC.fullmatch(token)
        cell_new_tokens.append(
            f"<{match.group(1)}_{int(match.group(2)) + 1:03d}>"
        )
    cell_new = [token_id(token) for token in cell_new_tokens]
    current = list(predictor)
    current[1:7] = cell_new_tokens
    cell_log = {
        "generation_positions": cell_positions,
        "previous_token_ids": cell_previous,
        "proposed_token_ids": cell_new,
        "new_token_ids": cell_new,
        "changed_components": 6,
        "all_sites_visible": True,
        "geometry_supported_before_restore": True,
        "restored_complete_noop": False,
    }

    slots_by_element = {
        element: [slot for slot, value in enumerate(species) if value == element]
        for element in elements
    }
    blocks = []
    for block_index, element in enumerate(reversed(elements)):
        slots = slots_by_element[element]
        block_positions = [
            position for slot in slots for position in coordinate_positions(slot)
        ]
        block_previous_tokens = [current[position] for position in block_positions]
        site_logs = []
        proposed_tokens = []
        for site_order_index, slot in enumerate(slots):
            positions = coordinate_positions(slot)
            old_tokens = [current[position] for position in positions]
            local_restore = index == 0 and block_index == 0 and site_order_index == 1
            if local_restore:
                new_tokens = old_tokens
            else:
                new_tokens = [
                    shifted_coordinate(token, component + 1)
                    for component, token in enumerate(old_tokens)
                ]
            proposed_tokens.extend(new_tokens)
            site_logs.append(
                {
                    "block_index": block_index,
                    "site_order_index": site_order_index,
                    "slot_index": slot,
                    "generation_positions": positions,
                    "previous_token_ids": [token_id(token) for token in old_tokens],
                    "new_token_ids": [token_id(token) for token in new_tokens],
                    "changed_components": sum(
                        left != right for left, right in zip(old_tokens, new_tokens)
                    ),
                    "restored_site_no_legal_z": local_restore,
                    "suffix_visible": True,
                }
            )
        restored_complete = index == 0 and block_index == 0
        final_block_tokens = (
            block_previous_tokens if restored_complete else proposed_tokens
        )
        for position, token in zip(block_positions, final_block_tokens):
            current[position] = token
        blocks.append(
            {
                "block_index": block_index,
                "slot_indices": slots,
                "generation_positions": block_positions,
                "previous_token_ids": [token_id(token) for token in block_previous_tokens],
                "proposed_token_ids": [token_id(token) for token in proposed_tokens],
                "new_token_ids": [token_id(token) for token in final_block_tokens],
                "changed_components": sum(
                    left != right
                    for left, right in zip(block_previous_tokens, final_block_tokens)
                ),
                "all_block_sites_masked_initially": True,
                "suffix_visible": True,
                "non_active_tokens_unchanged": True,
                "geometry_supported_before_restore": not restored_complete,
                "restored_complete_block": restored_complete,
                "restored_site_count": sum(
                    site["restored_site_no_legal_z"] for site in site_logs
                ),
                "site_revisions": site_logs,
            }
        )
    valid_endpoint = index != 127
    blocks[-1]["final_geometry_supported"] = valid_endpoint
    plan_state = {"N": n, "elements": elements, "counts": counts}
    plan = {
        "sample_idx": index,
        "preflight_idx": index,
        "source_row_idx": 1000 + index,
        "mp20_train_source_row_idx": 1000 + index,
        "source_split": "train",
        "outcomes_read": False,
        "prompt": f"prompt-{index}",
        "teacher_answer": "".join(predictor),
        "plan_state": plan_state,
        "species_program": list(reversed(elements)),
        # Deliberately useless: the materializer must ignore it.
        "preflight_state_type": "cell",
    }
    rollout = {
        "sample_idx": index,
        "text": "".join(current),
        "plan_state": plan_state,
        "conditioning_prompt": f"prompt-{index}",
        "parsed": valid_endpoint,
        "prompt_record": {
            "mp20_train_source_row_idx": 1000 + index,
            "source_split": "train",
        },
        "spad_basin_closure": True,
        "spad_basin_closure_cell_revision_log": cell_log,
        "spad_basin_closure_species_block_revision_log": blocks,
        "spad_basin_closure_metadata": {
            "cell_sampling_seed": 1_000_000 + index,
            "species_block_sampling_seed": 2_000_000 + index,
            "final_geometry_supported": valid_endpoint,
            "reverse_species_block_slots": [block["slot_indices"] for block in blocks],
        },
    }
    return plan, rollout, predictor


def fixtures():
    values = [make_fixture(index) for index in range(128)]
    return (
        [value[0] for value in values],
        [value[1] for value in values],
        {index: value[2] for index, value in enumerate(values)},
    )


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class MaterializeBasinPreflightStatesTest(unittest.TestCase):
    def test_4104_state_and_cursor_assignments_are_exactly_balanced(self):
        records = []
        for index in range(4104):
            elements, counts = plan_shape(index)
            records.append(
                {
                    "sample_idx": index,
                    "plan": {
                        "source_row_idx": index,
                        "plan_state": {
                            "N": sum(counts),
                            "elements": elements,
                            "counts": counts,
                        },
                    },
                }
            )
        state_types = MODULE.assign_state_types(records, expected_groups=4104)
        self.assertEqual(
            Counter(state_types.values()), Counter({"cell": 2052, "xyz": 2052})
        )
        xyz_records = [
            record
            for record in records
            if state_types[record["sample_idx"]] == "xyz"
        ]
        cursors = MODULE.assign_cursor_buckets(
            xyz_records, expected_groups=4104
        )
        self.assertEqual(
            Counter(cursors.values()),
            Counter({name: 513 for name in MODULE.CURSOR_BUCKETS}),
        )

    def test_128_rows_are_balanced_without_replacement(self):
        plans, rollouts, _ = fixtures()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plans_path = root / "plans.jsonl"
            rollouts_path = root / "rollouts.jsonl"
            output = root / "states"
            write_jsonl(plans_path, plans)
            write_jsonl(rollouts_path, rollouts)
            manifest = MODULE.run(
                argparse.Namespace(
                    plans_jsonl=plans_path,
                    rollouts_jsonl=rollouts_path,
                    output_dir=output,
                    mask_token="<MASK>",
                    tokenizer_vocab_json=None,
                )
            )
            states = list(MODULE.iter_jsonl(output / "states.jsonl"))

            self.assertEqual(len(states), 128)
            self.assertEqual(
                Counter(state["state_type"] for state in states),
                Counter({"cell": 64, "xyz": 64}),
            )
            self.assertEqual(
                Counter(
                    state["cursor_bucket"]
                    for state in states
                    if state["state_type"] == "xyz"
                ),
                Counter({name: 16 for name in MODULE.CURSOR_BUCKETS}),
            )
            self.assertGreaterEqual(manifest["high_N_rows"]["cell"], 12)
            self.assertGreaterEqual(manifest["high_N_rows"]["xyz"], 12)
            self.assertGreaterEqual(manifest["high_multiplicity_rows"]["cell"], 16)
            self.assertGreaterEqual(manifest["high_multiplicity_rows"]["xyz"], 16)
            self.assertEqual(manifest["invalid_final_endpoint_count"], 1)
            self.assertEqual(manifest["reference_log_replay_mismatches"], 0)
            self.assertFalse(manifest["outcomes_read"])
            self.assertFalse(manifest["selection"])
            self.assertFalse(manifest["replacement"])
            self.assertEqual(
                {state["mp20_train_source_row_idx"] for state in states},
                {1000 + index for index in range(128)},
            )
            self.assertTrue((output / "_SUCCESS").is_file())

            for coverage in manifest["strata"].values():
                self.assertLessEqual(abs(coverage["cell"] - coverage["xyz"]), 1)
            for state in states:
                self.assertEqual(state["schema"], MODULE.STATE_SCHEMA)
                self.assertTrue(state["reference_log_replay_matches_final"])
                self.assertEqual(
                    state["continuation_seeds"]["species_blocks"],
                    2_000_000 + state["sample_idx"],
                )
                self.assertNotIn("<MASK>", state["provisional_complete_body_tokens"])
                masked = {
                    index
                    for index, token in enumerate(state["state_body_tokens"])
                    if token == "<MASK>"
                }
                expected_masked = set(state["active_generation_positions"]) | set(
                    state["context_masked_generation_positions"]
                )
                self.assertEqual(masked, expected_masked)
                if state["state_type"] == "cell":
                    self.assertEqual(state["active_generation_positions"], list(range(1, 7)))
                    self.assertEqual(state["context_masked_generation_positions"], [])
                    self.assertIsNone(state["block_entry_snapshot"])
                else:
                    self.assertEqual(len(state["active_generation_positions"]), 3)
                    self.assertIsNotNone(state["block_entry_snapshot"])
                    self.assertEqual(
                        state["cursor"]["flattened_cursor_depth"] + 1,
                        state["cursor"]["flattened_cursor_ordinal"],
                    )

    def test_multiblock_complete_restore_reconstructs_and_replays(self):
        plans, rollouts, predictors = fixtures()
        resolver = MODULE.TokenResolver()
        records = MODULE.prepare_records(plans, rollouts, resolver)
        record = records[0]
        self.assertEqual(record["predictor"], predictors[0])
        self.assertGreater(len(record["blocks"]), 1)
        self.assertGreater(len(record["blocks"][0]["site_revisions"]), 1)
        self.assertTrue(record["blocks"][0]["restored_complete_block"])
        self.assertNotEqual(
            record["blocks"][0]["proposed_token_ids"],
            record["blocks"][0]["new_token_ids"],
        )

        state = MODULE.materialize_xyz_state(record, resolver, "<MASK>", "early")
        self.assertEqual(state["cursor"]["block_index"], 0)
        self.assertTrue(state["block_entry_snapshot"]["restored_complete_block"])
        self.assertTrue(state["reference_action"]["whole_block_restored_after_action"])
        self.assertTrue(state["reference_log_replay_matches_final"])
        self.assertEqual(state["cursor"]["site_order_index"], 1)
        self.assertNotEqual(
            state["provisional_complete_body_tokens"],
            state["block_entry_snapshot"]["body_tokens"],
        )
        prior_site = record["blocks"][0]["site_revisions"][0]
        for position, token_id_value in zip(
            prior_site["generation_positions"], prior_site["new_token_ids"]
        ):
            self.assertEqual(
                state["provisional_complete_body_tokens"][position],
                resolver.resolve(token_id_value, position=position),
            )

    def test_existing_output_directory_is_rejected(self):
        plans, rollouts, _ = fixtures()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plans_path = root / "plans.jsonl"
            rollouts_path = root / "rollouts.jsonl"
            output = root / "states"
            output.mkdir()
            write_jsonl(plans_path, plans)
            write_jsonl(rollouts_path, rollouts)
            with self.assertRaises(FileExistsError):
                MODULE.run(
                    argparse.Namespace(
                        plans_jsonl=plans_path,
                        rollouts_jsonl=rollouts_path,
                        output_dir=output,
                        mask_token="<MASK>",
                        tokenizer_vocab_json=None,
                    )
                )


if __name__ == "__main__":
    unittest.main()
