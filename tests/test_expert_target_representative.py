import copy
import random
from types import SimpleNamespace
import unittest

from crystal_dlm.expert_target_representative import training_representative
from crystal_dlm.expert_edit import make_edit_view
from tests.test_expert_edit_training import Tokenizer


class TargetRepresentativeTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.v = self.tokenizer.get_vocab()
        self.old = ['<N_003>', '<LA_040>', '<LB_040>', '<LC_040>', '<AA_090>', '<AB_090>', '<AG_090>',
                    '<E_Na>', '<X_000>', '<Y_000>', '<Z_000>',
                    '<E_Na>', '<X_050>', '<Y_050>', '<Z_050>',
                    '<E_Cl>', '<X_020>', '<Y_020>', '<Z_020>']
        target = self.old.copy()
        for site, xyz in enumerate([(73, 87, 61), (23, 37, 11), (43, 57, 31)]):
            for axis, value in enumerate(xyz):
                target[8+4*site+axis] = f'<{"XYZ"[axis]}_{value:03d}>'
        self.record = {'record_id': 's', 'ancestor_id': 'a', 'task': 'S', 'num_atoms': 3,
            'old_body': [self.v[t] for t in self.old], 'target_body': [self.v[t] for t in target],
            'content_supervision': True, 'accept_label': True, 'target_physics_id': 'original-physics',
            'old_geometry': {'valid': True}, 'target_geometry': {'valid': True},
            'old_reliable': True, 'target_reliable': True, 'gain_eV_atom': .2,
            'action': {'mode': 'full_cell', 'sites': [0, 1, 2],
                       'positions': list(range(1,7)) + [8+4*i+a for i in range(3) for a in range(3)]}}

    def test_equivalent_teacher_is_reversible_and_keeps_original_physics_binding(self):
        original = copy.deepcopy(self.record)
        result = training_representative(self.record, self.tokenizer)
        self.assertEqual(self.record, original)
        self.assertEqual(result['target_body'], original['target_body'])
        self.assertEqual(result['target_physics_id'], 'original-physics')
        self.assertEqual(result['training_target_body'], original['old_body'])
        self.assertTrue(result['training_representative_zero_edit'])
        certificate = result['training_target_certificate']
        self.assertTrue(certificate['inverse_recovers_original_tokens'])
        self.assertFalse(certificate['physical_label_modified'])
        self.assertFalse(certificate['independently_physics_evaluated'])

    def test_content_uses_one_complete_representative_but_judge_keeps_original_endpoint(self):
        result = training_representative(self.record, self.tokenizer)
        # Change a lattice field so this becomes an actual nonzero whole-cell edit.
        result['target_body'] = result['target_body'].copy()
        result['target_body'][1] = self.v['<LA_050>']
        result = training_representative(result, self.tokenizer)
        rng = SimpleNamespace(randrange=lambda n: 8, random=lambda: 0.)
        view = make_edit_view(result, 'content', rng, [0], content_target_mode='next_token', m2t_probability=1.)
        positions = result['action']['positions']
        for position in positions[:8]:
            self.assertEqual(view['input_body'][position], result['training_target_body'][position])
        self.assertEqual(view['targets'][positions[8]], result['training_target_body'][positions[8]])
        judge = make_edit_view(result, 'judge', random.Random(2), [0])
        self.assertEqual(judge['input_body'], result['target_body'])

    def test_local_scope_and_non_S_records_are_untouched(self):
        for field, value in [('task', 'G'), ('content_supervision', False)]:
            row = dict(self.record, **{field: value})
            self.assertIs(training_representative(row, self.tokenizer), row)
        row = copy.deepcopy(self.record)
        row['action']['mode'] = 'local_xyz'
        self.assertIs(training_representative(row, self.tokenizer), row)


if __name__ == '__main__':
    unittest.main()
