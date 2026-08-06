import unittest

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer
from crystal_dlm.r5_plan_state import (
    build_body_prompt,
    build_atomfields_plan_prompt,
    build_atomseq_plan_prompt,
    build_atomslots_plan_prompt,
    build_compact_plan_prompt,
    build_compact_plan_repair_prompt,
    build_countfields_plan_prompt,
    build_countvalence_plan_prompt,
    normalize_compact_plan_for_repair_target,
    parse_atomfields_plan_state,
    parse_atomseq_plan_state,
    parse_atomslots_plan_state,
    parse_compact_plan_state,
    parse_countfields_plan_state,
    parse_countvalence_plan_state,
    parse_plan_state_json,
    plan_state_from_arrays,
    plan_state_to_atomfields,
    plan_state_to_atomseq,
    plan_state_to_atomslots,
    plan_state_to_compact,
    plan_state_to_countfields,
    plan_state_to_countvalencefields,
    plan_state_to_json,
    validate_plan_state,
)


class R5PlanStateTests(unittest.TestCase):
    def make_arrays(self):
        answer, _ = arrays_to_dynamic_answer(
            lengths=[3.1, 3.1, 5.2],
            angles=[90.0, 90.0, 120.0],
            species=["Li", "Li", "O"],
            frac_coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        )
        return parse_dynamic_answer(answer, strict=True)

    def test_plan_state_round_trip_json(self):
        plan = plan_state_from_arrays(
            self.make_arrays(),
            metadata={"material_id": "toy-Li2O", "spacegroup.number": 194},
        )
        self.assertEqual(plan["N"], 3)
        self.assertEqual(plan["elements"], ["Li", "O"])
        self.assertEqual(plan["counts"], [2, 1])
        self.assertEqual(plan["formula"], "Li2O")
        self.assertEqual(plan["reduced_formula"], "Li2O")
        self.assertEqual(plan["anion_framework"], "oxide")
        self.assertTrue(validate_plan_state(plan).valid)

        encoded = plan_state_to_json(plan)
        decoded = parse_plan_state_json("prefix " + encoded + " suffix")
        self.assertEqual(decoded["formula"], "Li2O")
        prompt = build_body_prompt(plan)
        self.assertIn("plan_state:", prompt)
        self.assertIn("dynamic_crystal_body:", prompt)

    def test_compact_plan_canonicalizes_formula(self):
        plan = plan_state_from_arrays(
            self.make_arrays(),
            metadata={"material_id": "toy-Li2O", "spacegroup.number": 194},
        )
        compact = plan_state_to_compact(plan)
        decoded = parse_compact_plan_state(compact)
        self.assertEqual(decoded["elements"], ["Li", "O"])
        self.assertEqual(decoded["counts"], [2, 1])
        self.assertEqual(decoded["N"], 3)
        self.assertEqual(decoded["formula"], "Li2O")
        self.assertTrue(validate_plan_state(decoded).valid)
        self.assertIn("compact MP-20 crystal plan", build_compact_plan_prompt())

    def test_compact_plan_aggregates_duplicates(self):
        decoded = parse_compact_plan_state(
            "N=9;E=O:1,Na:1,K:1,Rb:1,Cs:1,Cs:1,Cs:1,Cs:1,Cs:1;LS=triclinic;SG=sg_003_015;VP=volpa_015_019"
        )
        self.assertEqual(decoded["elements"], ["O", "Na", "K", "Rb", "Cs"])
        self.assertEqual(decoded["counts"], [1, 1, 1, 1, 5])
        self.assertEqual(decoded["N"], 9)
        self.assertEqual(decoded["formula"], "ONaKRbCs5")
        self.assertTrue(validate_plan_state(decoded).valid)

    def test_compact_plan_requires_generated_n_to_match_counts(self):
        decoded = parse_compact_plan_state("N=2;E=Li:2,O:1;LS=hexagonal;SG=sg_168_194;VP=volpa_010_014")
        validation = validate_plan_state(decoded)
        self.assertFalse(validation.valid)
        self.assertFalse(validation.to_dict()["valid_generated_N"])
        self.assertFalse(validation.to_dict()["valid_N"])

    def test_atomseq_plan_derives_counts_without_generated_n(self):
        plan = plan_state_from_arrays(
            self.make_arrays(),
            metadata={"material_id": "toy-Li2O", "spacegroup.number": 194},
        )
        atomseq = plan_state_to_atomseq(plan)
        decoded = parse_atomseq_plan_state(atomseq)
        self.assertEqual(decoded["N"], 3)
        self.assertEqual(decoded["elements"], ["Li", "O"])
        self.assertEqual(decoded["counts"], [2, 1])
        self.assertEqual(decoded["formula"], "Li2O")
        self.assertTrue(validate_plan_state(decoded).valid)
        self.assertIn("atom-sequence MP-20", build_atomseq_plan_prompt())

    def test_atomseq_plan_rejects_more_than_twenty_atoms(self):
        line = "A=" + ",".join(["O"] * 21) + ";LS=triclinic;SG=sg_001_002;VP=volpa_010_014"
        with self.assertRaises(ValueError):
            parse_atomseq_plan_state(line)

    def test_atomslots_plan_derives_counts_from_nonempty_slots(self):
        plan = plan_state_from_arrays(
            self.make_arrays(),
            metadata={"material_id": "toy-Li2O", "spacegroup.number": 194},
        )
        atomslots = plan_state_to_atomslots(plan)
        decoded = parse_atomslots_plan_state(atomslots)
        self.assertIn("S=Li,Li,O,_", atomslots)
        self.assertEqual(decoded["N"], 3)
        self.assertEqual(decoded["elements"], ["Li", "O"])
        self.assertEqual(decoded["counts"], [2, 1])
        self.assertEqual(decoded["formula"], "Li2O")
        self.assertTrue(validate_plan_state(decoded).valid)
        self.assertIn("atom-slots MP-20", build_atomslots_plan_prompt())

    def test_atomslots_plan_rejects_more_than_twenty_slots(self):
        line = "S=" + ",".join(["O"] * 21) + ";LS=triclinic;SG=sg_001_002;VP=volpa_010_014"
        with self.assertRaises(ValueError):
            parse_atomslots_plan_state(line)

    def test_atomfields_plan_derives_counts_from_fixed_fields(self):
        plan = plan_state_from_arrays(
            self.make_arrays(),
            metadata={"material_id": "toy-Li2O", "spacegroup.number": 194},
        )
        atomfields = plan_state_to_atomfields(plan)
        decoded = parse_atomfields_plan_state(atomfields)
        self.assertIn("S01=Li;S02=Li;S03=O;S04=_", atomfields)
        self.assertEqual(decoded["N"], 3)
        self.assertEqual(decoded["elements"], ["Li", "O"])
        self.assertEqual(decoded["counts"], [2, 1])
        self.assertEqual(decoded["formula"], "Li2O")
        self.assertTrue(validate_plan_state(decoded).valid)
        self.assertIn("atom-fields MP-20", build_atomfields_plan_prompt())

    def test_atomfields_plan_requires_all_twenty_fields(self):
        line = "S01=O;S02=O;LS=triclinic;SG=sg_001_002;VP=volpa_010_014"
        with self.assertRaises(ValueError):
            parse_atomfields_plan_state(line)

    def test_atomfields_plan_rejects_out_of_range_slot_field(self):
        line = plan_state_to_atomfields(
            plan_state_from_arrays(self.make_arrays(), metadata={"material_id": "toy-Li2O", "spacegroup.number": 194})
        )
        with self.assertRaises(ValueError):
            parse_atomfields_plan_state(line + ";S21=O")

    def test_countfields_plan_derives_counts_from_fixed_pairs(self):
        plan = plan_state_from_arrays(
            self.make_arrays(),
            metadata={"material_id": "toy-Li2O", "spacegroup.number": 194},
        )
        countfields = plan_state_to_countfields(plan)
        decoded = parse_countfields_plan_state(countfields)
        self.assertIn("P01=Z003:C002;P02=Z008:C001;P03=Z000:C000", countfields)
        self.assertIn("LS=L_HEX", countfields)
        self.assertEqual(decoded["N"], 3)
        self.assertEqual(decoded["elements"], ["Li", "O"])
        self.assertEqual(decoded["counts"], [2, 1])
        self.assertEqual(decoded["formula"], "Li2O")
        self.assertEqual(decoded["lattice_system"], "hexagonal")
        self.assertTrue(validate_plan_state(decoded).valid)
        self.assertIn("count-fields MP-20", build_countfields_plan_prompt())

    def test_countfields_plan_rejects_extra_pair_field(self):
        line = plan_state_to_countfields(
            plan_state_from_arrays(self.make_arrays(), metadata={"material_id": "toy-Li2O", "spacegroup.number": 194})
        )
        with self.assertRaises(ValueError):
            parse_countfields_plan_state(line + ";P08=Z008:C001")

    def test_countfields_plan_rejects_count_sum_above_twenty(self):
        line = (
            "P01=Z008:C021;P02=Z000:C000;P03=Z000:C000;P04=Z000:C000;"
            "P05=Z000:C000;P06=Z000:C000;P07=Z000:C000;LS=L_TRI;SG=G001002;VP=V010014"
        )
        with self.assertRaises(ValueError):
            parse_countfields_plan_state(line)

    def test_countvalence_plan_encodes_charge_labels_without_filtering(self):
        plan = plan_state_from_arrays(
            self.make_arrays(),
            metadata={"material_id": "toy-Li2O", "spacegroup.number": 194},
        )
        plan["charge_bucket"] = "neutral_plausible"
        plan["validator"] = {
            "valid": True,
            "reason": "charge_neutral_pauling_valid",
            "oxidation_states": (1, -2),
        }
        countvalence = plan_state_to_countvalencefields(plan)
        decoded = parse_countvalence_plan_state(countvalence)
        self.assertIn("P01=Z003:C002:QP01;P02=Z008:C001:QM02", countvalence)
        self.assertIn("CB=B_NEU", countvalence)
        self.assertIn("LS=L_HEX", countvalence)
        self.assertEqual(decoded["N"], 3)
        self.assertEqual(decoded["elements"], ["Li", "O"])
        self.assertEqual(decoded["counts"], [2, 1])
        self.assertEqual(decoded["formula"], "Li2O")
        self.assertEqual(decoded["generated_charge_sum"], 0)
        self.assertEqual(decoded["generated_charge_bucket"], "neutral_plausible")
        self.assertTrue(validate_plan_state(decoded).valid)
        self.assertIn("chemistry-labeled MP-20", build_countvalence_plan_prompt())

    def test_countvalence_plan_allows_unknown_oxidation_supervision(self):
        line = (
            "P01=Z003:C001:QU00;P02=Z029:C001:QU00;P03=Z041:C001:QU00;"
            "P04=Z000:C000:QZ00;P05=Z000:C000:QZ00;P06=Z000:C000:QZ00;"
            "P07=Z000:C000:QZ00;CB=B_MET;LS=L_TRI;SG=G001002;VP=V010014"
        )
        decoded = parse_countvalence_plan_state(line)
        self.assertEqual(decoded["generated_charge_sum"], None)
        self.assertFalse(decoded["generated_charge_sum_known"])
        self.assertEqual(decoded["generated_charge_bucket"], "all_metal")
        self.assertIn(decoded["charge_bucket"], {"all_metal", "validator_unavailable"})

    def test_countvalence_plan_rejects_count_sum_above_twenty(self):
        line = (
            "P01=Z008:C021:QM02;P02=Z000:C000:QZ00;P03=Z000:C000:QZ00;"
            "P04=Z000:C000:QZ00;P05=Z000:C000:QZ00;P06=Z000:C000:QZ00;"
            "P07=Z000:C000:QZ00;CB=B_CHF;LS=L_TRI;SG=G001002;VP=V010014"
        )
        with self.assertRaises(ValueError):
            parse_countvalence_plan_state(line)

    def test_compact_repair_prompt_mentions_visible_plan_and_labels(self):
        prompt = build_compact_plan_repair_prompt(
            visible_plan="N=20;E=Li:2,O:20;LS=triclinic;SG=sg_001_002;VP=volpa_010_014",
            violation_labels=["atom_count_out_of_range", "generated_N_count_mismatch"],
        )
        self.assertIn("original_compact_plan:", prompt)
        self.assertIn("atom_count_out_of_range", prompt)
        self.assertIn("corrected_compact_plan:", prompt)

    def test_compact_repair_target_normalizes_visible_count_errors(self):
        target = normalize_compact_plan_for_repair_target(
            "N=56;E=O:12,Na:2,Al:12,Sr:2,La:4;LS=tetragonal;SG=sg_0016_074;VP=volpa_015_019;extra"
        )
        decoded = parse_compact_plan_state(target)
        self.assertEqual(decoded["N"], sum(decoded["counts"]))
        self.assertLessEqual(decoded["N"], 20)
        self.assertEqual(decoded["generated_N"], decoded["N"])
        self.assertEqual(decoded["spacegroup_bucket"], "sg_016_074")
        self.assertTrue(validate_plan_state(decoded).valid)

    def test_compact_repair_target_uses_sum_for_generated_n_mismatch(self):
        target = normalize_compact_plan_for_repair_target(
            "N=18;E=O:10,Ca:2,Ba:4;LS=trigonal;SG=sg_195_230;VP=volpa_015_019<|endoftext|>"
        )
        self.assertTrue(target.startswith("N=16;E="))
        decoded = parse_compact_plan_state(target)
        self.assertTrue(validate_plan_state(decoded).valid)


if __name__ == "__main__":
    unittest.main()
