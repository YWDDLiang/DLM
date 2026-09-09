import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from crystal_dlm.ranked_feedback import (ranked_preference,align_fixed_slots,
    target_action,validate_action_target,joint_stop_status)


def score(hull,sun=False,**kw):
    return dict(e_above_hull_eV_atom=hull,strict_sun=sun,terminal_status='verified',terminal_verified=True,**kw)


class RankedFeedback(unittest.TestCase):
    def test_SUN_cannot_be_traded_for_lower_energy(self):
        p=ranked_preference(score(-.01,True),score(-.5,False))
        self.assertEqual((p['chosen'],p['objective_level']),('before','SUN'))

    def test_fallback_hierarchy_and_relative_improvement(self):
        self.assertEqual(ranked_preference(score(.05),score(-.01))['objective_level'],'strict_stable')
        self.assertEqual(ranked_preference(score(.2),score(.05))['objective_level'],'meta_stable')
        self.assertEqual(ranked_preference(score(.3),score(.2))['objective_level'],'ordinary_improvement')
        self.assertIsNone(ranked_preference(score(.3),score(.295))['chosen'])

    def test_unknown_cannot_be_fabricated_as_failure(self):
        unknown=score(.3);unknown['terminal_verified']=False;unknown['terminal_status']='worker_timeout'
        self.assertIsNone(ranked_preference(unknown,score(-.01,True))['chosen'])
        unknown['terminal_status']='invalid_terminal'
        self.assertEqual(ranked_preference(unknown,score(-.01,True))['chosen'],'after')

    def test_reordering_preserves_blocks_and_rejects_composition_change(self):
        base=[100,1,2,3,4,5,6,20,1,2,3,21,4,5,6]
        target=base[:7]+base[11:15]+base[7:11]
        aligned,permutation=align_fixed_slots(target,base)
        self.assertEqual(aligned,base);self.assertEqual(permutation,[1,0])
        target[7]=99
        with self.assertRaises(ValueError):align_fixed_slots(target,base)

    def test_conditioned_edit_scope_is_enforced(self):
        base=[100,1,2,3,4,5,6,20,1,2,3,21,4,5,6]
        target=base.copy();target[8]=9
        action=target_action(base,target)
        self.assertEqual(action['mode'],1)
        validate_action_target(base,target,action['positions'])
        target[1]=10
        with self.assertRaises(ValueError):validate_action_target(base,target,action['positions'])

    def test_stress_prevents_early_stop_even_when_forces_are_small(self):
        self.assertFalse(joint_stop_status([[.01,0,0]],[[1,0,0],[0,0,0],[0,0,0]])['physical_converged'])
        self.assertTrue(joint_stop_status([[.01,0,0]],[[.4,0,0],[0,0,0],[0,0,0]])['physical_converged'])


if __name__=='__main__':unittest.main()
