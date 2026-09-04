import importlib.util
from pathlib import Path
import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_full_mp20_terminal_actions",
    ROOT / "src" / "scripts" / "build_full_mp20_terminal_actions.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import terminal-action builder")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
LABEL_SPEC = importlib.util.spec_from_file_location(
    "label_full_mp20_terminal_values",
    ROOT / "scripts" / "label_full_mp20_terminal_values.py",
)
if LABEL_SPEC is None or LABEL_SPEC.loader is None:
    raise RuntimeError("cannot import shared terminal-value labeler")
LABEL_MODULE = importlib.util.module_from_spec(LABEL_SPEC)
LABEL_SPEC.loader.exec_module(LABEL_MODULE)


def answer(species, *, offset=0.0):
    text, _ = arrays_to_dynamic_answer(
        lengths=[4.0, 4.5, 5.0],
        angles=[90.0, 90.0, 90.0],
        species=species,
        frac_coords=[
            [
                (0.10 + offset + 0.20 * index) % 1.0,
                (0.20 + 0.17 * index) % 1.0,
                (0.30 + 0.13 * index) % 1.0,
            ]
            for index in range(len(species))
        ],
    )
    return text


def source(stage_index=0):
    names = ["cell", "anchor_second", "anchor_first"]
    base = 17
    row = {
        "schema": MODULE.SOURCE_SCHEMA,
        "source_row_idx": 0,
        "sample_idx": 0,
        "source_weight": 1.0,
        "plan_state": {
            "N": 4,
            "elements": ["Na", "O", "Cl"],
            "counts": [2, 1, 1],
        },
        "prompt": "fixture prompt",
        "species_program": ["Cl", "Na", "O"],
        "deployment_stage_index": stage_index,
        "deployment_stage": names[stage_index],
        "common_random_seed_base": base,
        "common_random_seed": base + stage_index * MODULE.STAGE_SEED_STRIDE,
        "remaining_reference_stages": [],
        "outcomes_read": False,
    }
    for index in range(stage_index + 1, 3):
        row["remaining_reference_stages"].append(
            {
                "stage": names[index],
                "common_random_seed": base + index * MODULE.STAGE_SEED_STRIDE,
            }
        )
    return row


class DeploymentChainTest(unittest.TestCase):
    def test_generated_body_order_not_teacher_order_resolves_llama_anchors(self):
        # Teacher ledger could have been Na,O,Na,Cl.  Generated body is reordered.
        generated = ["O", "Na", "Cl", "Na"]
        stages, anchors = MODULE.resolved_deployment_stages(source(0), generated)
        self.assertEqual(anchors["program_slots"], [2, 1, 0])
        self.assertEqual(anchors["anchor_first_slot"], 2)  # Llama chose Cl first.
        self.assertEqual(anchors["anchor_second_slot"], 1)  # Then Na.
        self.assertEqual(stages[1]["active_positions"], [12, 13, 14])
        self.assertEqual(stages[2]["active_positions"], [16, 17, 18])

    def test_cell_candidate_executes_anchor_second_then_anchor_first(self):
        stages, _ = MODULE.resolved_deployment_stages(
            source(0), ["O", "Na", "Cl", "Na"]
        )
        calls = []

        def executor(current, stage):
            calls.append((stage["stage"], stage["common_random_seed"]))
            return current + f"|{stage['stage']}", {"ok": True}

        terminal, _ = MODULE.execute_stage_chain("state", stages[1:], executor)
        self.assertEqual(
            calls,
            [
                ("anchor_second", 17 + MODULE.STAGE_SEED_STRIDE),
                ("anchor_first", 17 + 2 * MODULE.STAGE_SEED_STRIDE),
            ],
        )
        self.assertEqual(terminal, "state|anchor_second|anchor_first")

    def test_anchor_second_candidate_executes_only_anchor_first(self):
        stages, _ = MODULE.resolved_deployment_stages(
            source(1), ["O", "Na", "Cl", "Na"]
        )
        self.assertEqual(MODULE.continuation_stage_names(stages, 1), ["anchor_first"])

    def test_anchor_first_candidate_has_no_continuation(self):
        stages, _ = MODULE.resolved_deployment_stages(
            source(2), ["O", "Na", "Cl", "Na"]
        )
        self.assertEqual(MODULE.continuation_stage_names(stages, 2), [])

    def test_all_candidates_receive_identical_future_seed_per_stage(self):
        stages, _ = MODULE.resolved_deployment_stages(
            source(0), ["O", "Na", "Cl", "Na"]
        )
        candidates = [{"candidate_idx": index} for index in range(4)]
        requests = MODULE.continuation_requests(candidates, stages, 0)
        by_stage = {}
        for request in requests:
            by_stage.setdefault(request["stage"], set()).add(
                request["common_random_seed"]
            )
        self.assertEqual(by_stage["anchor_second"], {17 + MODULE.STAGE_SEED_STRIDE})
        self.assertEqual(by_stage["anchor_first"], {17 + 2 * MODULE.STAGE_SEED_STRIDE})

    def test_anchor_state_is_built_after_cell_not_independently_from_body(self):
        ledger = source(2)
        body_answer = answer(["O", "Na", "Cl", "Na"])
        context = MODULE._prepare_context(
            ledger,
            {
                "source_row_idx": 0,
                "sample_idx": 0,
                "plan_state": ledger["plan_state"],
                "text": body_answer,
                "parsed": True,
            },
        )

        class FakeExecutor:
            def __init__(self):
                self.inputs = []

            def execute_batch(self, requests):
                output = []
                for request in requests:
                    self.inputs.append((request["stage"]["stage"], request["answer"]))
                    tokens = list(MODULE.tokenize_answer_text(request["answer"]))
                    position = int(request["stage"]["active_positions"][0])
                    tokens[position] = (
                        "<LA_041>"
                        if request["stage"]["stage"] == "cell"
                        else "<X_055>"
                    )
                    output.append(("".join(tokens), {"ok": True}))
                return output

        executor = FakeExecutor()
        MODULE._run_prefixes([context], executor=executor, batch_size=8)
        self.assertIsNone(context["failure"])
        self.assertEqual([name for name, _ in executor.inputs], ["cell", "anchor_second"])
        # The anchor-second request sees the lattice committed by cell closure.
        self.assertIn("<LA_041>", executor.inputs[1][1])
        self.assertIn("<X_055>", context["state_answer"])


class CandidateRetentionTest(unittest.TestCase):
    def attempt(self, name, tokens, *, legal=True):
        return {
            "candidate_source": name,
            "active_action_tokens": list(tokens),
            "active_legal": legal,
            "active_failure": None if legal else "fixture_failure",
        }

    def test_failed_actions_and_variable_k_are_preserved_without_replacement(self):
        retained, audit = MODULE.retain_shared_candidates(
            [
                self.attempt("noop", ["a"]),
                self.attempt("reference_dlm", ["a"]),
                self.attempt("physics_downhill", ["b", "c"], legal=False),
                self.attempt("physics_reverse", ["d"]),
            ]
        )
        self.assertEqual(
            [item["candidate_source"] for item in retained],
            ["noop", "physics_reverse"],
        )
        self.assertEqual(
            [item["retention_status"] for item in audit],
            ["retained", "duplicate", "invalid", "retained"],
        )
        self.assertEqual(len(retained), 2)

    def test_failed_source_keeps_fixed_denominator_placeholder(self):
        record = MODULE.source_failure_record(source(0), "missing body")
        self.assertEqual(record["source_row_idx"], 0)
        self.assertEqual(record["candidate_count"], 0)
        self.assertEqual(record["status"], "failed_source")
        self.assertEqual(record["group_idx"], 0)
        self.assertEqual(record["state"]["deployment_stage"], "cell")
        self.assertFalse(record["replacement"])

    def test_physics_quantization_audit_uses_defined_directions_only(self):
        attempts = [
            self.attempt("noop", ["<X_010>"]),
            {
                **self.attempt("physics_downhill", ["<X_011>"]),
                "proposal": {"reason": "accepted"},
            },
            {
                **self.attempt("physics_reverse", ["<X_010>"]),
                "proposal": {"reason": "accepted"},
            },
        ]
        report = MODULE.physics_quantization_audit(attempts, threshold=0.5)
        self.assertEqual(report["direction_defined_proposals"], 2)
        self.assertEqual(report["quantized_changed_at_least_one_token"], 1)
        self.assertEqual(report["quantized_noop_duplicates"], 1)
        self.assertEqual(report["quantized_token_change_rate"], 0.5)
        self.assertTrue(report["threshold_passed"])

    def test_zero_direction_is_disclosed_not_used_to_game_change_rate(self):
        attempts = [
            self.attempt("noop", ["<X_010>"]),
            {
                **self.attempt("physics_downhill", ["<X_010>"], legal=False),
                "proposal": {"reason": "zero_force_has_no_direction"},
            },
            {
                **self.attempt("physics_reverse", ["<X_011>"]),
                "proposal": {"reason": "accepted"},
            },
        ]
        report = MODULE.physics_quantization_audit(attempts)
        self.assertEqual(report["undefined_direction_proposals"], 1)
        self.assertEqual(report["direction_defined_proposals"], 1)
        self.assertEqual(report["quantized_token_change_rate"], 1.0)

    def test_signed_support_is_fixed_without_result_driven_resampling(self):
        attempts = [
            self.attempt("noop", ["a"]),
            self.attempt("reference_dlm", ["b"]),
            self.attempt("physics_downhill", ["c"]),
            self.attempt("physics_reverse", ["d"]),
        ]
        retained, audit = MODULE.retain_shared_candidates(attempts)
        self.assertEqual(
            [item["candidate_source"] for item in audit],
            ["noop", "reference_dlm", "physics_downhill", "physics_reverse"],
        )
        self.assertEqual(len(retained), 4)

    def test_stagewise_gate_catches_cell_failure_hidden_by_site_pool(self):
        def summary(changed):
            return {
                "physics_proposal_slots": 2,
                "direction_defined_proposals": 2,
                "undefined_direction_proposals": 0,
                "quantized_changed_at_least_one_token": changed,
                "quantized_noop_duplicates": 2 - changed,
                "signed_pair_token_collision": 0,
                "accepted_legal_physics": changed,
                "realized_direction_known": changed,
                "realized_direction_unknown": 2 - changed,
                "realized_direction_consistent": changed,
                "realized_direction_expected_cosine_sum": float(changed),
                "realized_direction_expected_cosine_min": (
                    None if changed == 0 else 1.0
                ),
                "realized_direction_expected_cosine_max": (
                    None if changed == 0 else 1.0
                ),
            }

        rows = [
            {
                "deployment_stage": "cell",
                "physics_quantization_audit": summary(0),
            }
        ]
        for stage in ("anchor_second", "anchor_first"):
            rows.extend(
                {
                    "deployment_stage": stage,
                    "physics_quantization_audit": summary(2),
                }
                for _ in range(10)
            )
        pooled = MODULE.merge_physics_quantization_audits(
            [row["physics_quantization_audit"] for row in rows]
        )
        by_stage = MODULE.physics_quantization_by_stage(rows)
        self.assertTrue(pooled["threshold_passed"])
        self.assertFalse(by_stage["cell"]["threshold_passed"])
        self.assertTrue(by_stage["anchor_second"]["threshold_passed"])
        self.assertTrue(by_stage["anchor_first"]["threshold_passed"])
        self.assertFalse(MODULE.every_stage_passes_physics_token_change(by_stage))

    def test_hard_cap_above_five_percent_fails_locality_without_resampling(self):
        def audit(cap_hits):
            attempts = [self.attempt("noop", ["noop"])]
            for index in range(20):
                step = 0.40 if index < cap_hits else 0.10
                attempts.append(
                    {
                        **self.attempt(
                            "physics_downhill" if index % 2 == 0 else "physics_reverse",
                            [f"action-{index}"],
                        ),
                        "proposal": {
                            "reason": "accepted",
                            "kind": "site_xyz",
                            "step": step,
                        },
                    }
                )
            return MODULE.physics_quantization_audit(attempts)

        exactly_five_percent = audit(1)
        above_five_percent = audit(2)
        self.assertEqual(exactly_five_percent["hard_cap_hit_rate"], 0.05)
        self.assertTrue(exactly_five_percent["locality_preflight_passed"])
        self.assertEqual(above_five_percent["hard_cap_hit_rate"], 0.10)
        self.assertTrue(above_five_percent["locality_preflight_failed"])
        self.assertFalse(above_five_percent["retry_or_resample"])
        self.assertEqual(
            above_five_percent["selected_step_histogram"]["site_xyz:0.4A"],
            2,
        )
        stage_reports = {
            "cell": above_five_percent,
            "anchor_second": exactly_five_percent,
            "anchor_first": exactly_five_percent,
        }
        self.assertFalse(
            MODULE.every_stage_passes_physics_locality(stage_reports)
        )

    def test_one_shared_serialization_is_byte_identical_for_future_A_and_B(self):
        record = {
            "schema": MODULE.OUTPUT_SCHEMA,
            "source_row_idx": 9,
            "candidates": [
                {
                    "candidate_idx": 0,
                    "terminal_answer": "terminal",
                    "terminal_cif": "data_fixture",
                    "terminal_legal": True,
                }
            ],
            "outcomes_read": False,
        }
        a_bytes = MODULE.json_line_bytes(record)
        b_bytes = MODULE.json_line_bytes(dict(reversed(list(record.items()))))
        self.assertEqual(a_bytes, b_bytes)

    def test_success_schema_is_consumable_by_shared_A_B_value_labeler(self):
        LABEL_MODULE.validate_groups(
            [
                {
                    "group_idx": 0,
                    "source": {"source_split": "train", "source_row_idx": 0},
                    "stage": "cell",
                    "state": {"source_answer": "state"},
                    "shared_terminal_pool": True,
                    "candidates": [
                        {
                            "valid_terminal": True,
                            "terminal_answer": "terminal tokens",
                            "terminal_cif": None,
                            "action_token_ids": [1, 2, 3, 4, 5, 6],
                            "legality": True,
                        }
                    ],
                }
            ]
        )


class TransactionAndShardTest(unittest.TestCase):
    def test_actual_vocab_and_one_bin_scan_are_frozen_and_energy_free(self):
        contract = MODULE.actual_token_quantization_contract()
        self.assertEqual(contract["coordinate_token_ids_per_axis"], 101)
        self.assertEqual(contract["coordinate_periodic_intervals"], 100)
        self.assertEqual(contract["coordinate_effective_canonical_values"], 100)
        self.assertEqual(contract["coordinate_fractional_step"], 0.01)
        self.assertEqual(contract["length_step_A"], 0.1)
        self.assertEqual(contract["angle_step_degree"], 1.0)
        self.assertFalse(contract["energy_or_outcome_used_to_choose_step"])
        self.assertEqual(
            list(MODULE.FORCE_ONE_BIN_SCAN_STEPS_A),
            sorted(set(MODULE.FORCE_ONE_BIN_SCAN_STEPS_A)),
        )
        self.assertEqual(
            list(MODULE.STRAIN_ONE_BIN_SCAN_STEPS),
            sorted(set(MODULE.STRAIN_ONE_BIN_SCAN_STEPS)),
        )

    def test_site_realized_direction_uses_strict_periodic_MIC(self):
        state = {
            "lengths": [10.0, 10.0, 10.0],
            "angles": [90.0, 90.0, 90.0],
            "species": ["Li", "O"],
            "frac_coords": [[0.99, 0.20, 0.30], [0.50, 0.50, 0.50]],
        }
        positive = {**state, "frac_coords": [[0.01, 0.20, 0.30], [0.50, 0.50, 0.50]]}
        negative = {**state, "frac_coords": [[0.97, 0.20, 0.30], [0.50, 0.50, 0.50]]}
        plus = MODULE.realized_site_action_direction(
            state,
            positive,
            site_index=0,
            cartesian_force=[1.0, 0.0, 0.0],
            expected_sign=1,
        )
        minus = MODULE.realized_site_action_direction(
            state,
            negative,
            site_index=0,
            cartesian_force=[1.0, 0.0, 0.0],
            expected_sign=-1,
        )
        self.assertAlmostEqual(plus["realized_delta_cart_A"][0], 0.2)
        self.assertAlmostEqual(plus["cosine_with_original_direction"], 1.0)
        self.assertAlmostEqual(minus["cosine_with_original_direction"], -1.0)
        self.assertAlmostEqual(minus["cosine_with_expected_direction"], 1.0)
        self.assertFalse(plus["energy_read"])

    def test_cell_realized_direction_uses_quantized_symmetric_strain(self):
        state = {
            "lengths": [10.0, 10.0, 10.0],
            "angles": [90.0, 90.0, 90.0],
            "species": ["Li", "O"],
            "frac_coords": [[0.10, 0.20, 0.30], [0.50, 0.50, 0.50]],
        }
        contracted = {**state, "lengths": [9.9, 9.9, 9.9]}
        expanded = {**state, "lengths": [10.1, 10.1, 10.1]}
        downhill = MODULE.realized_cell_action_direction(
            state,
            contracted,
            stress=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            expected_sign=-1,
        )
        reverse = MODULE.realized_cell_action_direction(
            state,
            expanded,
            stress=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            expected_sign=1,
        )
        self.assertAlmostEqual(
            downhill["cosine_with_expected_direction"], 1.0
        )
        self.assertAlmostEqual(reverse["cosine_with_expected_direction"], 1.0)
        self.assertTrue(downhill["direction_consistent"])
        self.assertFalse(downhill["energy_read"])

    def test_complete_site_transaction_only_changes_xyz(self):
        plan = {"N": 2, "elements": ["Li", "O"], "counts": [1, 1]}
        before = answer(["Li", "O"])
        tokens = list(MODULE.tokenize_answer_text(before))
        tokens[8:11] = ["<X_012>", "<Y_022>", "<Z_032>"]
        after = "".join(tokens)
        old, new = MODULE.validate_transaction_transition(
            before,
            after,
            plan_state=plan,
            stage={"active_positions": [8, 9, 10]},
        )
        self.assertEqual(len(old), 3)
        self.assertEqual(new, ["<X_012>", "<Y_022>", "<Z_032>"])

    def test_four_contiguous_shards_cover_formal_denominator_once(self):
        shards = [MODULE.contiguous_shard(27_136, 4, rank) for rank in range(4)]
        self.assertEqual(shards[0][0], 0)
        self.assertEqual(shards[-1][1], 27_136)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(shards, shards[1:])))
        self.assertEqual(sum(stop - start for start, stop in shards), 27_136)


if __name__ == "__main__":
    unittest.main()
