"""Policy decisions and predicate attribution cannot use physics for selection."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'operations/r03_c3fd_main_20260907'))
from editor_trial_analysis import paired, policy_decision


def score(hull, *, novel=True, unique=True):
    return dict(terminal_verified=True, terminal_status='verified', e_above_hull_eV_atom=hull,
                novel=novel, unique_representative=unique,
                strict_sun=hull <= 0 and novel and unique, meta_sun=hull <= .1 and novel and unique)


class TrialAnalysisTests(unittest.TestCase):
    def test_learned_keep_and_probability_gates(self):
        trace = dict(proposal_generated=True, learned_mode=0, known_sun=False, learned_accept_probability=.7)
        self.assertFalse(policy_decision(trace, threshold=.5, respect_keep=True))
        self.assertTrue(policy_decision(trace, threshold=.65, respect_keep=False))
        self.assertFalse(policy_decision(trace, threshold=.8, respect_keep=False))
        self.assertFalse(policy_decision(dict(trace, known_sun=True), threshold=.5, respect_keep=False))
        self.assertFalse(policy_decision(dict(trace, proposal_generated=False), threshold=.5, respect_keep=False))

    def test_no_physics_fields_can_affect_decision(self):
        trace = dict(proposal_generated=True, learned_mode=1, known_sun=False, learned_accept_probability=.6)
        a = policy_decision(trace, threshold=.5, respect_keep=True)
        b = policy_decision(dict(trace, oracle_energy=999, verified=False, strict_stable=False), threshold=.5, respect_keep=True)
        self.assertTrue(a); self.assertEqual(a, b)

    def test_Stable_loss_N_gain_and_rejected_Stable_promotion_are_separate(self):
        before = [score(.05), score(-.02), score(-.02, novel=False)]
        after = [score(.05), score(.04), score(-.02)]
        proposal = [score(-.03), score(.04), score(-.02)]
        traces = [dict(proposal_generated=True, applied=bool(i), learned_mode=2, learned_accept_probability=.6) for i in range(3)]
        report = paired(before, after, [0, 1, 2], traces=traces, proposals=proposal)
        self.assertEqual(report['transitions']['Stable'], dict(gains=[], losses=[1]))
        self.assertEqual(report['transitions']['SUN'], dict(gains=[2], losses=[1]))
        self.assertEqual(report['predicate_change_accounting']['SUN_gain:N'], 1)
        self.assertEqual(report['predicate_change_accounting']['SUN_loss:Stable'], 1)
        self.assertEqual(report['proposal_decisions']['Stable_promotion_rejected'], 1)
        self.assertEqual(report['proposal_decisions']['Stable_damage_accepted'], 1)
        self.assertEqual(report['counts']['MSUN'], 3)

    def test_source_filter_excludes_unselected_outcomes(self):
        before = [score(.2), score(.2)]
        a = paired(before, [score(-.1), score(.2)], [1])
        b = paired(before, [score(999), score(.2)], [1])
        self.assertEqual(a, b)


if __name__ == '__main__': unittest.main()
