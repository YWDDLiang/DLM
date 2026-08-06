import copy
import unittest

from crystal_dlm.wqcodiff.state import OrbitState, StratifiedState
from diagnostics.freeze_fixed_topology_composition_panel import (
    PanelFreezeError,
    freeze_panel,
    proposal_features,
    select_matched_controls,
)


def make_state(attempt_id: str, *, atomic_numbers: tuple[int, ...], sg: int) -> dict:
    state = StratifiedState(
        space_group=sg,
        lattice_system="synthetic",
        lattice_chart=(1.0, 1.1, 1.2),
        orbits=tuple(
            OrbitState(
                orbit_id=f"o{index}",
                wyckoff_type=index + 1,
                species=atomic_number,
                multiplicity=index + 1,
                primitive_multiplicity=index + 1,
                chart_dimension=index % 2,
                free_coordinate=() if index % 2 == 0 else (0.25,),
            )
            for index, atomic_number in enumerate(atomic_numbers)
        ),
        attempt_id=attempt_id,
    )
    return state.to_dict(canonical_storage=False)


def source_row(
    ordinal: int,
    *,
    mechanism: str,
    atomic_numbers: tuple[int, ...],
    sg: int,
) -> tuple[dict, dict]:
    attempt_id = f"a-{ordinal:024d}"
    generation = {
        "schema": "wq_parent_csp_probe_attempt_v1",
        "attempt_id": attempt_id,
        "ordinal": ordinal,
        "status": "succeeded",
        "retry_or_replacement_used": False,
        "proposal_state": make_state(
            attempt_id,
            atomic_numbers=atomic_numbers,
            sg=sg,
        ),
    }
    taxonomy = {
        "attempt_id": attempt_id,
        "ordinal": ordinal,
        "validity_mechanism": mechanism,
        "category": "strict_stable",
        "e_above_hull": 0.0,
        "density": 8.0,
        "struct_valid": True,
        "valid": mechanism == "valid",
    }
    return generation, taxonomy


class PanelFreezeTests(unittest.TestCase):
    def synthetic_inputs(self) -> tuple[list[dict], list[dict]]:
        specs = (
            ("no_charge_neutral_assignment", (3, 8), 1),
            ("no_charge_neutral_assignment", (12, 8, 8), 75),
            ("pauling_rejection", (26, 28), 195),
            ("valid", (11, 17), 2),
            ("valid", (14, 8, 8), 80),
            ("valid", (29, 30), 200),
            ("valid", (13, 8, 8), 90),
        )
        generation: list[dict] = []
        taxonomy: list[dict] = []
        for offset, (mechanism, atomic_numbers, sg) in enumerate(specs):
            pair = source_row(
                256 + offset,
                mechanism=mechanism,
                atomic_numbers=atomic_numbers,
                sg=sg,
            )
            generation.append(pair[0])
            taxonomy.append(pair[1])
        return generation, taxonomy

    def test_freeze_panel_exact_groups_and_determinism(self) -> None:
        generation, taxonomy = self.synthetic_inputs()
        kwargs = {
            "generation_rows": generation,
            "taxonomy_rows": taxonomy,
            "expected_attempts": 7,
            "expected_start_ordinal": 256,
            "expected_group_counts": {
                "no_neutral": 2,
                "pauling_only": 1,
                "valid_control": 2,
            },
        }
        panel_a, evidence_a = freeze_panel(**kwargs)
        panel_b, evidence_b = freeze_panel(**kwargs)
        self.assertEqual(panel_a, panel_b)
        self.assertEqual(evidence_a, evidence_b)
        self.assertEqual(
            [row["panel_group"] for row in panel_a],
            [
                "no_neutral",
                "no_neutral",
                "pauling_only",
                "valid_control",
                "valid_control",
            ],
        )
        self.assertEqual(evidence_a["scientific_generation_attempts_created"], 0)
        self.assertEqual(evidence_a["mlip_calls"], 0)

    def test_matching_ignores_post_outcome_mutations(self) -> None:
        generation, taxonomy = self.synthetic_inputs()
        expected = {
            "no_neutral": 2,
            "pauling_only": 1,
            "valid_control": 2,
        }
        panel_a, _ = freeze_panel(
            generation_rows=generation,
            taxonomy_rows=taxonomy,
            expected_attempts=7,
            expected_start_ordinal=256,
            expected_group_counts=expected,
        )
        mutated = copy.deepcopy(taxonomy)
        for index, row in enumerate(mutated):
            row["category"] = f"mutated-{index}"
            row["e_above_hull"] = 1000.0 - index
            row["density"] = -100.0 + index
            row["struct_valid"] = not bool(row["struct_valid"])
            row["valid"] = not bool(row["valid"])
        panel_b, _ = freeze_panel(
            generation_rows=generation,
            taxonomy_rows=mutated,
            expected_attempts=7,
            expected_start_ordinal=256,
            expected_group_counts=expected,
        )
        self.assertEqual(panel_a, panel_b)

    def test_irrelevant_taxonomy_mechanisms_are_excluded_not_rejected(self) -> None:
        generation, taxonomy = self.synthetic_inputs()
        irrelevant_generation, irrelevant_taxonomy = source_row(
            263,
            mechanism="post_validity_fingerprint_failure",
            atomic_numbers=(6, 8),
            sg=15,
        )
        generation.append(irrelevant_generation)
        taxonomy.append(irrelevant_taxonomy)
        panel, evidence = freeze_panel(
            generation_rows=generation,
            taxonomy_rows=taxonomy,
            expected_attempts=8,
            expected_start_ordinal=256,
            expected_group_counts={
                "no_neutral": 2,
                "pauling_only": 1,
                "valid_control": 2,
            },
        )
        self.assertEqual(len(panel), 5)
        self.assertNotIn(
            irrelevant_generation["attempt_id"],
            {row["attempt_id"] for row in panel},
        )
        self.assertEqual(
            evidence["ignored_taxonomy_mechanism_counts"],
            {"post_validity_fingerprint_failure": 1},
        )
        self.assertEqual(evidence["selection_pool_counts"]["ignored_other"], 1)

    def test_matching_covers_targets_with_unique_controls(self) -> None:
        generation, taxonomy = self.synthetic_inputs()
        prepared = []
        for generation_row in generation:
            state = StratifiedState.from_dict(generation_row["proposal_state"])
            prepared.append(
                {
                    "attempt_id": generation_row["attempt_id"],
                    "features": proposal_features(state),
                }
            )
        selected, evidence = select_matched_controls(
            targets=prepared[:2],
            candidates=prepared[3:],
            all_features=[row["features"] for row in prepared],
            count=2,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({row["attempt_id"] for row in selected}), 2)
        self.assertEqual(len(evidence["target_matches"]), 2)
        self.assertEqual(
            sorted(
                target
                for controls in evidence["selected_control_coverage"].values()
                for target in controls
            ),
            sorted(row["attempt_id"] for row in prepared[:2]),
        )

    def test_rejects_retry_source_and_identity_mismatch(self) -> None:
        generation, taxonomy = self.synthetic_inputs()
        generation[0]["retry_or_replacement_used"] = True
        with self.assertRaises(PanelFreezeError):
            freeze_panel(
                generation_rows=generation,
                taxonomy_rows=taxonomy,
                expected_attempts=7,
                expected_start_ordinal=256,
                expected_group_counts={
                    "no_neutral": 2,
                    "pauling_only": 1,
                    "valid_control": 2,
                },
            )

        generation, taxonomy = self.synthetic_inputs()
        generation[0]["proposal_state"]["attempt_id"] = "a-wrong"
        with self.assertRaises(PanelFreezeError):
            freeze_panel(
                generation_rows=generation,
                taxonomy_rows=taxonomy,
                expected_attempts=7,
                expected_start_ordinal=256,
                expected_group_counts={
                    "no_neutral": 2,
                    "pauling_only": 1,
                    "valid_control": 2,
                },
            )


if __name__ == "__main__":
    unittest.main()
