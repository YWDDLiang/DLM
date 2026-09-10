"""Protect SUN priority, missing-label semantics and independent final admission."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from crystal_dlm.sun_ranker import endpoint_targets,select_candidate


def score(hull,**kwargs):
    return dict(terminal_status='verified',terminal_verified=True,e_above_hull_eV_atom=hull,
        strict_stable=hull<=0,meta_stable=hull<=.1,novel=True,**kwargs)


class SUNRankerTests(unittest.TestCase):
    def test_stable_promotion_is_distinct_from_meta_promotion(self):
        self.assertEqual(endpoint_targets(score(-.01))[0],(1.,1.))
        self.assertEqual(endpoint_targets(score(.05))[0],(0.,1.))
        self.assertEqual(endpoint_targets(score(.3))[0],(0.,0.))

    def test_missing_reference_and_nonconvergence_are_different(self):
        row=score(.01);row.update(terminal_status='not_converged',terminal_verified=False)
        self.assertEqual(endpoint_targets(row),((0.,0.),'not_converged_operational_zero'))
        row['e_above_hull_eV_atom']=None
        self.assertIsNone(endpoint_targets(row)[0])
        row.update(terminal_status='worker_error',e_above_hull_eV_atom=.01)
        self.assertIsNone(endpoint_targets(row)[0])

    def test_MS_does_not_outvote_a_different_SUN_score_band(self):
        candidates=[dict(stream='SUN',valid=True,sun_gain=.08,ms_gain=.02),
            dict(stream='MS',valid=True,sun_gain=.03,ms_gain=10.)]
        self.assertEqual(select_candidate(candidates,sun_threshold=.02,ms_floor=0.),'SUN')
        self.assertIsNone(select_candidate(candidates,sun_threshold=.1,ms_floor=0.))
        self.assertIsNone(select_candidate(candidates,sun_threshold=0.,ms_floor=0.,known_sun=True))

    def test_geometry_validity_and_fixed_ties(self):
        candidates=[dict(stream='first',valid=True,sun_gain=.05,ms_gain=.02),
            dict(stream='second',valid=True,sun_gain=.05,ms_gain=.02),
            dict(stream='invalid',valid=False,sun_gain=1.,ms_gain=1.)]
        self.assertEqual(select_candidate(candidates,sun_threshold=.05,ms_floor=0.),'first')


if __name__=='__main__':unittest.main()
