import hashlib
import unittest

from crystal_dlm.h1a2_factorial_contract import build_factorial_ordinal_record
from real_ledger_preflight import convert_parsed_row


PLAN = (
    "formula: Li2O\n"
    "anion: oxide\n"
    "charge: neutral_plausible\n"
    "lattice: hexagonal\n"
    "spacegroup: sg_168_194\n"
    "volume: volpa_005_009\n"
    "end: plan"
)


class RealLedgerPreflightTests(unittest.TestCase):
    def test_raw_and_canonical_hashes_are_independent_and_body_compiles(self):
        raw = PLAN.replace(
            "spacegroup: sg_168_194\n",
            "spacegroup: sg_168_194\nprimate: hexagonal\n",
        )
        ordinal = build_factorial_ordinal_record(17029, sample_idx=30)
        contract = {
            "planner_arm": "P0",
            "include_sample_id": False,
            "prompt_text": "frozen H1 prompt without an identifier",
            "prompt_sha256": "1" * 64,
            "input_ids_sha256": "2" * 64,
            "tokenizer_identity_sha256": "3" * 64,
        }
        record = convert_parsed_row(
            planner_arm="P0",
            factorial_arm="M00",
            source={
                "sample_idx": 30,
                "planner_sampling_seed": ordinal["planner_sampling_seed"],
                "parsed": True,
                "raw_plan_text": raw,
                "plan_text": PLAN,
            },
            sample_idx=30,
            planner_base_seed=17029,
            input_contract=contract,
        )
        self.assertTrue(record["body_compilation_reached"])
        self.assertFalse(record["raw_plan_contract_conforming"])
        self.assertEqual(
            record["raw_plan_text_sha256"],
            hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            record["plan_text_sha256"],
            hashlib.sha256(PLAN.encode("utf-8")).hexdigest(),
        )
        self.assertNotEqual(
            record["raw_plan_text_sha256"],
            record["plan_text_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
