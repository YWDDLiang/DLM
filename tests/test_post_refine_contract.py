import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crystal_dlm.post_refine_contract import Quality, make_gate, make_pair, preference, paired_change


def score(index, *, sun=False, stable=False, meta=False, h=.2):
    return {'trajectory_id': f't{index}', 'sample_idx': index, 'strict_sun': sun,
            'strict_stable': stable, 'meta_sun': sun or meta,
            'meta_stable': stable or meta, 'e_above_hull_eV_atom': h,
            'terminal_verified': False}


class PostRefineContracts(unittest.TestCase):
    def test_only_strict_sun_bypasses_diffusion_including_unverified_sun(self):
        rows = [{'trajectory_id': f't{i}', 'sample_idx': i + 100, 'evaluation_ordinal': i,
                 'success': True} for i in range(3)]
        values = [score(0, sun=True, stable=True, h=-.01),
                  score(1, meta=True, h=.05), score(2, stable=True, h=-.2)]
        for i, value in enumerate(values):
            value['sample_idx'] = i + 100
        gate = make_gate(rows, values, input_sha256='a', scored_input_sha256='a', score_identity={})
        self.assertEqual([d['run_diffusion'] for d in gate['decisions']], [False, True, True])
        self.assertEqual(gate['decisions'][0]['original_ordinal'], 100)
        self.assertEqual(gate['decisions'][1]['reason'], 'MSUN_still_requires_diffusion')

    def test_partial_or_foreign_cohort_cannot_gate(self):
        rows = [{'trajectory_id': 't0', 'sample_idx': 0, 'success': True}]
        with self.assertRaises(ValueError):
            make_gate(rows, [], input_sha256='a', scored_input_sha256='a', score_identity={})
        with self.assertRaises(ValueError):
            make_gate(rows, [score(0)], input_sha256='a', scored_input_sha256='b', score_identity={})

    def test_raw_is_preferred_when_refining_loses_sun(self):
        raw = Quality.from_score(score(0, sun=True, stable=True, h=-.01))
        refined = Quality.from_score(score(0, stable=True, h=-.8))
        self.assertEqual(preference(raw, refined)['chosen'], 'before')
        self.assertEqual(preference(refined, raw)['chosen'], 'after')

    def test_same_tier_reverse_energy_preference_and_tie(self):
        raw = Quality.from_score(score(0, stable=True, h=-.3))
        worse = Quality.from_score(score(0, stable=True, h=-.1))
        self.assertEqual(preference(raw, worse)['chosen'], 'before')
        self.assertIsNone(preference(raw, raw)['chosen'])

    def test_identical_responses_and_evaluation_pairs_do_not_train(self):
        pair = make_pair(source_id='s', conditioning={'plan': 'p'}, before_tokens=[1, 2],
                         after_tokens=[1, 2], before_score=score(0),
                         after_score=score(0, sun=True, stable=True, h=-.1),
                         before_identity={'cohort': 'a'}, after_identity={'cohort': 'b'},
                         round_index=1, purpose='evaluation')
        self.assertFalse(pair['training_use_allowed'])
        self.assertFalse(pair['pair_usable'])
        self.assertEqual(pair['preference']['reason'], 'identical_token_responses')

    def test_unknown_or_inconsistent_is_not_a_preference(self):
        unknown = Quality.from_score(score(0, h=None))
        good = Quality.from_score(score(0, sun=True, stable=True, h=-.1))
        self.assertIsNone(preference(unknown, good)['chosen'])
        bad_row = score(0, stable=True, h=-.5)
        bad_row['preference_physics_consistent'] = False
        self.assertIsNone(preference(Quality.from_score(bad_row), good)['chosen'])

    def test_stage_report_counts_both_gains_and_losses_on_original_ids(self):
        old = [score(100, sun=True, stable=True, h=-.1), score(106)]
        new = [score(106, sun=True, stable=True, h=-.1), score(100, stable=True, h=-.2)]
        result = paired_change(old, new)
        self.assertEqual((result['SUN_gains'], result['SUN_losses']), (1, 1))
        self.assertEqual(result['SUN_loss_sources'], [100])
        self.assertEqual((result['MSUN_gains'], result['MSUN_losses']), (1, 1))


if __name__ == '__main__':
    unittest.main()
