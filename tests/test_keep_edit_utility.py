import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('focus_utility',ROOT/'operations/r03_c3fd_main_20260907/compile_keep_edit_utility.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class SignedUtilityTests(unittest.TestCase):
    def score(self,hull,novel=None,**kwargs):
        return dict(terminal_verified=True,terminal_status='verified',e_above_hull_eV_atom=hull,
            strict_stable=hull<=0,meta_stable=hull<=.1,novel=novel,**kwargs)

    def test_novel_stable_has_both_rewards_and_nonnovel_has_none(self):
        self.assertEqual(module.utility(self.score(-.01,True)),2)
        self.assertEqual(module.utility(self.score(-.01,False)),0)
        self.assertEqual(module.utility(self.score(.05,True)),1)

    def test_unused_or_historical_U_is_never_a_target(self):
        self.assertEqual(module.utility(self.score(-.01,True,strict_sun=False,unique_representative=False)),2)
        self.assertEqual(module.utility(self.score(-.01,False,strict_sun=True,unique_representative=True)),0)

    def test_unstable_does_not_need_missing_novelty(self):
        self.assertEqual(module.utility(self.score(.5,None)),0)
        self.assertIsNone(module.utility(self.score(.05,None)))

    def test_unknown_hull_and_nonconvergence_are_not_physical_negatives(self):
        self.assertIsNone(module.utility(dict(terminal_verified=True,terminal_status='verified',e_above_hull_eV_atom=None)))
        self.assertIsNone(module.utility(dict(terminal_verified=False,terminal_status='not_converged',e_above_hull_eV_atom=.2)))
        self.assertEqual(module.utility(dict(terminal_verified=False,terminal_status='invalid_terminal')),0)


if __name__=='__main__':unittest.main()
