import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('focus_utility',ROOT/'operations/r03_c3fd_main_20260907/compile_keep_edit_utility.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from crystal_dlm.utility_acceptance import accept_utility


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

    def test_raw_utility_margin_and_original_guard(self):
        self.assertTrue(accept_utility(.05,.05,proposal_generated=True))
        self.assertFalse(accept_utility(.049,.05,proposal_generated=True))
        self.assertFalse(accept_utility(2.,0.,proposal_generated=False))
        self.assertFalse(accept_utility(2.,0.,proposal_generated=True,known_sun_guard=True))
        self.assertFalse(accept_utility(None,0.,proposal_generated=True))
        with self.assertRaises(ValueError):accept_utility(float('nan'),0.,proposal_generated=True)

    def test_consensus_requires_both_scores_and_has_no_missing_reference_bypass(self):
        conditions=dict(proposal_generated=True,reference_required=True)
        self.assertTrue(accept_utility(.05,.05,reference_logit=0.,**conditions))
        self.assertFalse(accept_utility(2.,.05,reference_logit=-.001,**conditions))
        self.assertFalse(accept_utility(.049,.05,reference_logit=10.,**conditions))
        self.assertFalse(accept_utility(2.,.05,reference_logit=None,**conditions))
        with self.assertRaises(ValueError):accept_utility(2.,.05,reference_logit=float('nan'),**conditions)


if __name__=='__main__':unittest.main()
