from pathlib import Path
import importlib.util
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_c3fd_h1a2_fusion_prospective.py"
SPEC = importlib.util.spec_from_file_location("freeze_fusion_prospective", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def source(element, index):
    return {
        "sample_idx": index,
        "plan_state": {
            "N": 3,
            "elements": [element, "O"],
            "counts": [2, 1],
            "formula": f"{element}2O",
            "anion_framework": "oxide",
            "charge_bucket": "certified_neutral",
            "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230",
            "volume_per_atom_bin": "volpa_015_019",
        },
    }


class FreezeC3FDH1A2FusionProspectiveTest(unittest.TestCase):
    def test_matched_full_and_v2_views(self):
        rows = [source("Li", 0), source("Na", 1), source("K", 2)]
        blocked = {MODULE.RICH.exact_identity(rows[0]["plan_state"])}
        ledger, full_rows, v2_rows, report = MODULE.freeze(
            rows,
            blocked_exact=blocked,
            count=2,
        )
        self.assertEqual([row["sample_idx"] for row in ledger], [0, 1])
        self.assertEqual(
            [row["exact_composition_identity"] for row in full_rows],
            [row["exact_composition_identity"] for row in v2_rows],
        )
        self.assertTrue(all(row["prompt"].endswith("\n") for row in full_rows))
        self.assertTrue(all(row["prompt"].endswith("dynamic_crystal_body:") for row in v2_rows))
        self.assertTrue(all(row["plan_state"]["prototype_key"] for row in full_rows))
        self.assertEqual(report["exclusions"]["blocked_exact"], 1)


if __name__ == "__main__":
    unittest.main()
