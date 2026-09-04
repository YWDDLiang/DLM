import importlib.util
from pathlib import Path
import sys
import unittest


ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("select_tau",ROOT/"scripts/select_spad_refinement_tau.py")
MODULE=importlib.util.module_from_spec(SPEC); assert SPEC and SPEC.loader
sys.modules[SPEC.name]=MODULE; SPEC.loader.exec_module(MODULE)


def report(values):
    return {"schema":"spad_low_noise_tau_calibration_final_v1","cells":[
        {"tau":tau,"rates":{"strict_sun":strict,"meta_sun":meta}}
        for tau,(strict,meta) in values.items()
    ]}


class TauSelectionTest(unittest.TestCase):
    def test_maximizes_weaker_target_fraction(self):
        value=MODULE.select(report({400:(.08,.45),600:(.09,.40),800:(.07,.49)}))
        self.assertEqual(value["selected_tau"],400)

    def test_tie_prefers_smaller_tau(self):
        value=MODULE.select(report({400:(.08,.40),600:(.08,.40),800:(.08,.40)}))
        self.assertEqual(value["selected_tau"],400)


if __name__=="__main__": unittest.main()
