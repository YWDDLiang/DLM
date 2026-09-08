import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('sun_headroom_analysis', Path(__file__).resolve().parents[1]/'operations/r03_c3fd_main_20260907/analyze_sun_headroom.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture():
    variants = ['keep','local1','local4','all_xyz','full_cell']
    cases, arms = [], {}
    for i in range(8):
        candidates = {v:{'body':body,'mode':mode,'sites':sites} for v,body,mode,sites in [
            ('keep',[0,0],'none',[]),('local1',[1,0],'local_xyz',[0]),
            ('local4',[0,1],'all_xyz',[0,1]),('all_xyz',[1,1],'all_xyz',[0,1]),
            ('full_cell',[2,2],'full_cell',[0,1]) ]}
        cases.append({'ancestor_id':str(i),'composition_key':str(i),'old_body':[0,0], 'candidates':candidates,
                      'num_atoms':2,'old_geometry':{'valid':i>=4}})
    for seed in range(2):
        for v in variants:
            rows = []
            for i in range(8):
                sun = (i<6 and v=='local1') or (i==6 and seed==0 and v=='local4') or (i==7 and v=='keep')
                rows.append({'group_id':str(i),'strict_sun':sun,'meta_sun':sun,'reconstructed':True,
                             'terminal_verified':False,'terminal_status':'not_converged',
                             'terminal_energy_eV_atom':-1. if sun else 1.})
            arms[f'ref{seed}_{v}'] = {'rows':rows}
    return {'study':{'cases':cases,'variants':variants,'sources':8},'arms':arms,'execution_integrity_verified':True}


class HeadroomAnalysisTests(unittest.TestCase):
    def test_counts_sources_and_requires_held_noise_transfer(self):
        report = analysis.analyze(fixture())
        self.assertEqual(len(report['repeatable_wins']), 6)
        self.assertEqual(report['cross_noise']['0_to_1']['net_SUN'], 6)
        self.assertEqual(report['cross_noise']['1_to_0']['net_SUN'], 6)
        self.assertTrue(report['gate']['local_content_pilot_supported'])
        self.assertEqual(report['preservation']['sources_with_KEEP_SUN_both_seeds'], 1)

    def test_selection_does_not_peek_at_evaluation_noise(self):
        capsule = fixture()
        case, variants = capsule['study']['cases'][6], capsule['study']['variants']
        panels = {s:{v:{r['group_id']:r for r in capsule['arms'][f'ref{s}_{v}']['rows']} for v in variants} for s in range(2)}
        self.assertEqual(analysis.select_from_one_seed(case, variants, panels, 0), 'local4')
        panels[1]['full_cell']['6']['strict_sun'] = True
        self.assertEqual(analysis.select_from_one_seed(case, variants, panels, 0), 'local4')

    def test_existing_SUN_prefers_KEEP_on_ties(self):
        capsule = fixture()
        variants, case = capsule['study']['variants'], capsule['study']['cases'][7]
        panels = {0:{v:{r['group_id']:r for r in capsule['arms'][f'ref0_{v}']['rows']} for v in variants}}
        panels[0]['local1']['7']['strict_sun'] = True
        self.assertEqual(analysis.select_from_one_seed(case, variants, panels, 0), 'keep')

    def test_unresolved_SUN_is_not_a_preference(self):
        capsule = fixture()
        variants, case = capsule['study']['variants'], capsule['study']['cases'][0]
        panels = {0:{v:{r['group_id']:r for r in capsule['arms'][f'ref0_{v}']['rows']} for v in variants}}
        panels[0]['local1']['0']['strict_sun'] = None
        with self.assertRaisesRegex(ValueError, 'unresolved SUN'):
            analysis.select_from_one_seed(case, variants, panels, 0)

    def test_identical_body_outcome_mismatch_blocks_promotion(self):
        capsule = fixture()
        capsule['study']['cases'][0]['candidates']['local1']['body'] = [0,0]
        report = analysis.analyze(capsule)
        self.assertEqual(len(report['identical_body_inconsistencies']), 2)
        self.assertFalse(report['gate']['global_content_pilot_supported'])

    def test_identical_non_KEEP_bodies_with_conflicting_outcomes_are_rejected(self):
        capsule = fixture()
        capsule['study']['cases'][0]['candidates']['all_xyz']['body'] = [1,0]
        report = analysis.analyze(capsule)
        self.assertEqual(len(report['identical_body_inconsistencies']), 2)
        self.assertFalse(report['gate']['global_content_pilot_supported'])

    def test_duplicate_rows_and_truncated_bodies_cannot_pass(self):
        capsule = fixture()
        capsule['arms']['ref0_keep']['rows'].append(dict(capsule['arms']['ref0_keep']['rows'][0]))
        with self.assertRaisesRegex(ValueError, 'source rows'):
            analysis.analyze(capsule)
        capsule = fixture()
        capsule['study']['cases'][0]['candidates']['local1']['body'] = [1]
        with self.assertRaisesRegex(ValueError, 'body length'):
            analysis.analyze(capsule)

    def test_held_noise_reversal_counts_destroyed_SUN(self):
        capsule = fixture()
        capsule['arms']['ref1_keep']['rows'][6]['strict_sun'] = True
        report = analysis.analyze(capsule)
        direction = report['cross_noise']['0_to_1']
        self.assertEqual(direction['gains'], 6)
        self.assertEqual(direction['losses'], 1)
        self.assertEqual(direction['net_SUN'], 5)


if __name__ == '__main__': unittest.main()
