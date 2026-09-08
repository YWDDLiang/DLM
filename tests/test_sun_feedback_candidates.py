import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from scripts.generate_sun_feedback_candidates import candidate_bodies, keyed_seed, native_teacher_graph, validate_refinement_receipt
from crystal_dlm.sun_feedback_contract import sha256


class CandidateScopeTests(unittest.TestCase):
    def test_zero_motion_in_non_niggli_cell_preserves_original_slots_and_basis(self):
        from crystal_dlm.fixed_slot import build_special_tokens
        from crystal_dlm.expert_edit_data import quantize_arrays
        vocabulary = {token:index for index,token in enumerate(build_special_tokens())}
        arrays = {'lengths':[5.,3.,4.], 'angles':[90.,90.,90.], 'species':['Na','Na','O'],
                  'frac_coords':[[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]]}
        old, decoded, _ = quantize_arrays(arrays, vocabulary)
        graph = native_teacher_graph(decoded, 0)
        zero = dict(arrays, lengths=graph['length'].tolist(), angles=graph['angle'].tolist(), frac_coords=graph['x_coord'].tolist())
        target, coords, _ = quantize_arrays(zero, vocabulary)
        self.assertEqual(target, old)
        self.assertEqual(graph['length'].tolist(), [5.,3.,4.])
        result = candidate_bodies(old, dict.fromkeys(['first','middle','last'], target),
            {'old':decoded['frac_coords'], **dict.fromkeys(['first','middle','last'], coords['frac_coords'])})
        self.assertTrue(all(value['body']==old for value in result.values()))
        with self.assertRaises(ValueError):
            native_teacher_graph(dict(arrays, angles=[10.,10.,170.]), 0)

    def test_refinement_export_requires_same_study_protocol_and_noise_schedule(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            job = {'sample_idx':0,'case_idx':0,'variant':'keep','technical_repeat':0,'refiner_seed':100}
            spec = {'cases':[{'ancestor_id':'parent','source_row_idx':4}], 'schedule':[job]}
            study = root/'STUDY.json'
            study.write_text(json.dumps(spec))
            values = [{'probe_job_index':0, 'probe_case':0, 'probe_variant':'keep', 'technical_repeat':0,
                       'refiner_noise_seed_index':0, 'refiner_seed':100, 'group_id':'parent','source_row_idx':4,
                       'source_split':'train','purpose':'training_feedback','endpoint':'tau800'}]
            inputs = root/'refined_inputs.jsonl'
            inputs.write_text(json.dumps(values[0])+'\n')
            report = {'input_study_sha256':sha256(study),'training_use_allowed':True,
                'repeat_semantics':'independent_refiner_noise_seeds','deterministic_algorithms_enabled':True,
                'deterministic_algorithms_requested':True,'cublas_workspace_config':':4096:8',
                'diff_steps':800,'timesteps':1000,'forward_noise_added':False,'planned':1,
                'outputs_sha256':{'refined_inputs.jsonl':sha256(inputs)}}
            receipt = root/'REFINE_FINAL.json'
            receipt.write_text(json.dumps(report))
            validate_refinement_receipt(spec, study, root, values)
            values[0]['refiner_seed'] = 999
            inputs.write_text(json.dumps(values[0])+'\n')
            report['outputs_sha256']['refined_inputs.jsonl'] = sha256(inputs)
            receipt.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'seed differs'):
                validate_refinement_receipt(spec, study, root, values)
            report['input_study_sha256'] = 'wrong-study'
            receipt.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'receipt'):
                validate_refinement_receipt(spec, study, root, values)

    def fixture(self):
        old = [100,1,2,3,4,5,6] + [value for site in range(5) for value in [200+site,10,11,12]]
        target = old.copy()
        for p in range(1,7): target[p] += 20
        for site in range(5):
            for axis in range(3): target[8+4*site+axis] += 30
        coords = {'old': [[0,0,0]]*5, 'first': [[.99,0,0],[.4,0,0],[.1,0,0],[.2,0,0],[.3,0,0]],
                  'middle': [[.99,0,0],[.4,0,0],[.1,0,0],[.2,0,0],[.3,0,0]], 'last': [[0,0,0]]*5}
        return old, target, coords

    def test_local_and_coordinate_scopes_preserve_every_inactive_token(self):
        old, target, coords = self.fixture()
        before = old.copy()
        result = candidate_bodies(old, dict.fromkeys(['first','middle','last'], target), coords)
        self.assertEqual(result['local1']['sites'], [1])
        self.assertEqual(result['local4']['sites'], [1,2,3,4])
        for variant in ['local1','local4','all_xyz','full_cell']:
            changed = {i for i,(a,b) in enumerate(zip(old,result[variant]['body'])) if a!=b}
            self.assertEqual(changed, set(result[variant]['positions']))
            self.assertEqual(result[variant]['body'][0], old[0])
            self.assertEqual(result[variant]['body'][7::4], old[7::4])
        self.assertEqual(result['all_xyz']['body'][1:7], old[1:7])
        self.assertEqual(result['keep']['body'], old)
        self.assertEqual(old, before)

    def test_unencodable_teacher_remains_failed_instead_of_becoming_keep(self):
        old, target, coords = self.fixture()
        result = candidate_bodies(old, {'first':None,'middle':target,'last':None}, coords)
        self.assertIsNone(result['local1']['body'])
        self.assertIsNone(result['full_cell']['body'])
        self.assertEqual(result['keep']['body'], old)

    def test_teacher_species_change_is_rejected(self):
        old, target, coords = self.fixture()
        target[7] += 1
        with self.assertRaisesRegex(ValueError, 'immutable atom slots'):
            candidate_bodies(old, dict.fromkeys(['first','middle','last'], target), coords)

    def test_distinct_noise_repeats_are_reproducible_and_valid_torch_seeds(self):
        seeds = [keyed_seed(2026090841, 'source', 'F800', rep) for rep in range(2)]
        self.assertNotEqual(*seeds)
        self.assertEqual(seeds[0], keyed_seed(2026090841, 'source', 'F800', 0))
        self.assertTrue(all(0 <= value < 2**63 for value in seeds))


if __name__ == '__main__': unittest.main()
