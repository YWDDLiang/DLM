"""Physical evidence and executable-edit contracts, without an MLIP fixture."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crystal_dlm.expert_edit_data import (
    align_complete_target, bound_labels, certify_geometry, composition_key,
    derive_pair_supervision, endpoint_fingerprint, full_action, quantize_arrays,
    canonical_body, COMMON_RELAXATION_PROTOCOL, TERMINAL_VERIFICATION_PROTOCOL,
    prepare_collection, compile_collection, lattice_from_parameters, refined_arrays,
)
from crystal_dlm.fixed_slot import build_special_tokens, SYMBOL_TO_Z


def geometry(species=('Na', 'Cl')):
    return {'lengths': [4., 4., 4.], 'angles': [90., 90., 90.], 'species': list(species),
            'frac_coords': [[0., 0., 0.], [.5, .5, .5]][:len(species)]}


def terminal():
    return {'lattice': {'a': 4., 'b': 4., 'c': 4., 'alpha': 90., 'beta': 90., 'gamma': 90.},
            'sites': [{'species': [{'element': element, 'occu': 1}], 'abc': coord}
                      for element, coord in zip(geometry()['species'], geometry()['frac_coords'])]}


def label(energy, *, status='verified', verified=True):
    return {'verified': verified, 'status': status, 'terminal_energy': energy, 'final_structure': terminal()}


class ExpertEvidenceContracts(unittest.TestCase):
    def setUp(self):
        self.vocabulary = {token: i for i, token in enumerate(build_special_tokens())}
        self.old, _, _ = quantize_arrays(geometry(), self.vocabulary)
        target = geometry()
        target['lengths'] = [4.2, 4.2, 4.2]
        self.target, _, _ = quantize_arrays(target, self.vocabulary)
        self.pair = {'old_body': self.old, 'target_body': self.target, 'teacher_available': True,
                     'old_geometry': {'valid': True, 'certified': True},
                     'target_geometry': {'valid': True, 'certified': True}}

    def test_contact_certificate_finds_image_beyond_legacy_shell(self):
        skew = {'lengths': [1., 3.1, 5.], 'angles': [90., 90., 2.],
                'species': ['H'], 'frac_coords': [[0., 0., 0.]]}
        result = certify_geometry(skew)
        self.assertTrue(result['certified'])
        self.assertFalse(result['valid'])
        self.assertAlmostEqual(result['witness_distance_A'], .146050051, places=7)

    def test_periodic_boundary_contact_and_certificate_cap(self):
        value = geometry()
        value['frac_coords'] = [[.99, 0., 0.], [.01, 0., 0.]]
        self.assertFalse(certify_geometry(value)['valid'])
        capped = certify_geometry(geometry(), max_pair_images=1)
        self.assertIsNone(capped['valid'])
        self.assertFalse(capped['certified'])

    def test_quantization_clipping_and_fixed_composition_are_not_silent(self):
        value = geometry()
        value['lengths'][0] = 1e6
        with self.assertRaisesRegex(ValueError, 'clipped'):
            quantize_arrays(value, self.vocabulary)
        modified = list(self.target)
        modified[7] = modified[11]
        with self.assertRaisesRegex(ValueError, 'ordered species'):
            full_action(self.old, modified, 2)

    def test_reduced_composition_does_not_depend_on_supercell_or_order(self):
        self.assertEqual(composition_key(['Na', 'Cl']), composition_key(['Cl', 'Na', 'Na', 'Cl']))
        self.assertNotEqual(composition_key(['Na', 'Cl']), composition_key(['Na', 'Na', 'Cl']))

    def test_complete_target_alignment_preserves_physical_sites(self):
        value = geometry(('Cl', 'Na'))
        result, order = align_complete_target(geometry(), value)
        self.assertEqual(order, [1, 0])
        self.assertEqual(result['species'], ['Na', 'Cl'])
        self.assertEqual(result['frac_coords'], [value['frac_coords'][1], value['frac_coords'][0]])

    def test_stability_uses_both_actual_verified_terminals(self):
        result = derive_pair_supervision(self.pair, label(-1.), label(-1.1))
        self.assertEqual(result['S_outcome'], 'positive')
        self.assertAlmostEqual(result['gain_eV_atom'], .1)
        result = derive_pair_supervision(self.pair, label(-1., status='not_converged', verified=False), label(-1.1))
        self.assertIsNone(result['S_outcome'])
        self.assertEqual(result['G_reason'], 'terminal_reliability_recovery')

    def test_geometry_valid_old_can_need_terminal_geometry_recovery(self):
        result = derive_pair_supervision(self.pair, label(-1., status='invalid_terminal', verified=False), label(-1.1))
        self.assertEqual(result['G_reason'], 'relaxation_geometry_recovery')
        self.assertIsNone(result['S_outcome'])

    def test_certified_bad_terminal_overrides_a_legacy_verified_flag(self):
        old = label(-1.)
        old['final_structure']['lattice'].update(a=1., b=3.1, c=5., gamma=2.)
        result = derive_pair_supervision(self.pair, old, label(-1.1))
        self.assertIs(result['old_terminal_geometry'], False)
        self.assertIs(result['old_reliable'], False)
        self.assertEqual(result['G_reason'], 'relaxation_geometry_recovery')
        self.assertIsNone(result['S_outcome'])

    def test_missing_teacher_and_scale_uncertainty_are_not_negative_or_keep(self):
        result = derive_pair_supervision(dict(self.pair, teacher_available=False), label(-1.), None)
        self.assertIsNone(result['G_reason'])
        self.assertIsNone(result['S_outcome'])
        result = derive_pair_supervision(self.pair, dict(label(-1.), scale_uncertain=True), label(-2.))
        self.assertIsNone(result['S_outcome'])

    def test_identity_cannot_claim_gain_from_repeated_relaxation_noise(self):
        result = derive_pair_supervision(dict(self.pair, target_body=self.old), label(-1.), label(-1.1))
        self.assertEqual(result['S_outcome'], 'identity')
        self.assertEqual(result['gain_eV_atom'], 0.)

    def test_periodic_aliases_share_canonical_identity_before_physics(self):
        aliased = list(self.old)
        aliased[8] = self.vocabulary['<X_100>']
        self.assertEqual(canonical_body(aliased, self.vocabulary), self.old)

    def test_invalid_old_geometry_does_not_provide_a_stability_positive(self):
        pair = deepcopy(self.pair)
        pair['old_geometry']['valid'] = False
        result = derive_pair_supervision(pair, label(-1.), label(-2.))
        self.assertEqual(result['G_reason'], 'raw_geometry_recovery')
        self.assertIsNone(result['S_outcome'])

    def test_bound_labels_reject_same_id_on_different_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record = {'trajectory_id': 'x', 'group_id': 'g', 'source_split': 'train', 'endpoint': 'native',
                      'success': True, 'structure': terminal(), 'body': 'unused'}
            inputs = root / 'inputs.jsonl'
            inputs.write_text(json.dumps(record) + '\n')
            identities = {'model': 'fixture', 'model_checkpoint_sha256': 'pinned'}
            row = dict(record, endpoint_cache_key=endpoint_fingerprint(record), versions=identities, status='verified')
            (root / 'labels.jsonl').write_text(json.dumps(row) + '\n')
            report = {'purpose': 'expert_edit', 'statuses': {'verified': 1}, 'requested': 1, 'completed': 1,
                      'protocol': COMMON_RELAXATION_PROTOCOL, 'verification_protocol': TERMINAL_VERIFICATION_PROTOCOL,
                      'input_sha256': hashlib.sha256(inputs.read_bytes()).hexdigest(),
                      'runtime_identities': [identities]}
            (root / 'LABEL_FINAL.json').write_text(json.dumps(report))
            (root / '_SUCCESS').touch()
            self.assertEqual(set(bound_labels(inputs, root)[0]), {'x'})
            row['endpoint_cache_key'] = '0' * 64
            (root / 'labels.jsonl').write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'different input geometry'):
                bound_labels(inputs, root)
            report['protocol'] = dict(COMMON_RELAXATION_PROTOCOL, max_steps=100)
            (root / 'LABEL_FINAL.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'frozen common relaxation'):
                bound_labels(inputs, root)

    def test_prepare_bind_compile_end_to_end_and_pending_tamper(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            component = root / 'collection/shard0'
            for name in ('planner', 'body', 'teacher_refine'):
                (component / name).mkdir(parents=True)
            (component / '_SUCCESS').touch()
            (component / 'planner/run_config.json').write_text('{"purpose":"train"}')
            holder = root / 'heldout.jsonl'
            holder.write_text(json.dumps({'body_eligible': True, 'plan_state':
                {'N': 2, 'elements': ['Mg', 'O'], 'counts': [1, 1]}}) + '\n')
            plans, parents, atoms, coords = [], [], [], []
            for index, species in enumerate((('Na', 'Cl'), ('K', 'Cl'), ('Li', 'F'))):
                value = geometry(species)
                ids, _, _ = quantize_arrays(value, self.vocabulary)
                if index == 0:
                    ids[8] = self.vocabulary['<X_100>']
                plan = {'sample_idx': index, 'body_eligible': True, 'source_split': 'train',
                        'body_prompt': 'actual frozen rich condition ' + str(index),
                        'plan_state': {'N': 2, 'elements': list(species), 'counts': [1, 1]}}
                plans.append(plan)
                parents.append({'sample_idx': index, 'planner_record': plan, 'body_prompt': plan['body_prompt'],
                                'body_prompt_sha256': hashlib.sha256(plan['body_prompt'].encode()).hexdigest(),
                                'raw_body_token_ids': ids})
                atoms.extend(SYMBOL_TO_Z[x] for x in species)
                coords.extend(value['frac_coords'])
            for path, rows in ((component / 'planner/plans_for_dlm.jsonl', plans),
                               (component / 'body/raw_generations.jsonl', parents)):
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            refined = component / 'teacher.pt'
            payload = {'num_atoms': torch.tensor([[2, 2, 2]]), 'sample_indices': torch.arange(3),
                       'atom_types': torch.tensor([atoms]), 'frac_coords': torch.tensor([coords]),
                       'lengths': torch.tensor([[[4.2] * 3, [4.4] * 3, [1e6] * 3]]),
                       'angles': torch.tensor([[[90.] * 3] * 3])}
            torch.save(payload, refined)
            (component / 'teacher_refine/refinement_metrics.json').write_text(json.dumps(
                {'output_file': str(refined), 'num_evals': 1, 'diff_steps': 800}))
            manifest = {'purpose': 'train', 'run_root': str(root), 'source_identity': {'commit': 'f' * 40},
                        'components': [{'id': 'shard0', 'output_dir': 'collection/shard0', 'requests': 3,
                                        'sample_index_offset': 0}],
                        'evaluation': {'panel_files': {'cohort/cohort.jsonl':
                            {'sha256': hashlib.sha256(holder.read_bytes()).hexdigest()}}}}
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            def endpoint(record_id, group, source_idx, split, endpoint, arrays=None, body=None):
                structure = {'lattice': dict(zip(('a', 'b', 'c', 'alpha', 'beta', 'gamma'),
                                                 arrays['lengths'] + arrays['angles'])),
                             'sites': [{'species': [{'element': element, 'occu': 1}], 'abc': coordinate}
                                       for element, coordinate in zip(arrays['species'], arrays['frac_coords'])]}
                return {'trajectory_id': record_id, 'group_id': group, 'source_row_idx': source_idx,
                        'source_split': split, 'endpoint': endpoint, 'purpose': 'expert_edit',
                        'success': True, 'structure': structure, 'body': body}
            tokenizer = type('Tokenizer', (), {'get_vocab': lambda _: self.vocabulary})()
            prepared = root / 'prepared'
            with patch('crystal_dlm.expert_edit_data.physics_input', side_effect=endpoint):
                report = prepare_collection(manifest_path, holder, tokenizer, prepared)
            self.assertEqual(report['admitted_old'], 3)
            self.assertEqual(report['quantized_teacher_targets'], 2)
            pending = [json.loads(line) for line in (prepared / 'pairs_pending.jsonl').read_text().splitlines()]
            self.assertEqual(len(pending), 3)
            self.assertEqual(pending[0]['old_body'][8], self.vocabulary['<X_000>'])
            self.assertEqual(pending[0]['original_source_body'][8], self.vocabulary['<X_100>'])
            input_rows = [json.loads(line) for line in (prepared / 'all_inputs.jsonl').read_text().splitlines()]
            labels_dir = root / 'labels'
            labels_dir.mkdir()
            versions = {'model': 'fixture', 'model_checkpoint_sha256': 'pinned'}
            labels = [dict(row, status='verified', verified=True, versions=versions,
                           terminal_energy=-1.2 if row['endpoint'] == 'expert_quantized' else -1.,
                           final_structure=row['structure'], endpoint_cache_key=endpoint_fingerprint(row))
                      for row in input_rows]
            (labels_dir / 'labels.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in labels))
            (labels_dir / 'LABEL_FINAL.json').write_text(json.dumps({
                'purpose': 'expert_edit', 'requested': 5, 'completed': 5, 'statuses': {'verified': 5},
                'protocol': COMMON_RELAXATION_PROTOCOL, 'verification_protocol': TERMINAL_VERIFICATION_PROTOCOL,
                'runtime_identities': [versions],
                'input_sha256': hashlib.sha256((prepared / 'all_inputs.jsonl').read_bytes()).hexdigest()}))
            (labels_dir / '_SUCCESS').touch()
            result = compile_collection(prepared, labels_dir, root / 'compiled')
            self.assertEqual(result['comparable_sources'], 2)
            self.assertEqual(result['counts']['S:positive'], 2)
            compiled = [json.loads(line) for split in ('train', 'dev')
                        for line in (root / f'compiled/{split}.jsonl').read_text().splitlines()]
            self.assertTrue(all('target_body' not in row for row in compiled if row.get('state_only')))
            reverse = [row for row in compiled if row['label_reason'] == 'reverse_verified_degradation']
            self.assertTrue(all(row['old_physics_id'].startswith('expert-target:') for row in reverse))
            # A recovered training run may exclude the entire unresolved source;
            # the default contract must still reject that same engineering run.
            failed_group = labels[0]['group_id']
            labels[0].update(status='worker_error', verified=False, versions=None)
            (labels_dir / 'labels.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in labels))
            summary = json.loads((labels_dir / 'LABEL_FINAL.json').read_text())
            summary['statuses'] = {'verified': 4, 'worker_error': 1}
            (labels_dir / 'LABEL_FINAL.json').write_text(json.dumps(summary))
            (labels_dir / '_SUCCESS').unlink()
            (labels_dir / '_ENGINEERING_FAILED').touch()
            with self.assertRaisesRegex(ValueError, 'engineering failures'):
                compile_collection(prepared, labels_dir, root / 'not_recovered')
            recovered = compile_collection(prepared, labels_dir, root / 'recovered', exclude_worker_errors=True)
            self.assertEqual(recovered['excluded_engineering_ancestors'], [failed_group])
            remaining = [json.loads(line) for split in ('train','dev')
                         for line in (root / f'recovered/{split}.jsonl').read_text().splitlines()]
            self.assertTrue(all(row['ancestor_id'] != failed_group for row in remaining))
            (prepared / 'pairs_pending.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'pairs or physics inputs changed'):
                compile_collection(prepared, labels_dir, root / 'wrong')
            payload['num_atoms'] = payload['num_atoms'].repeat(2, 1)
            torch.save(payload, refined)
            with self.assertRaisesRegex(ValueError, 'exactly one evaluation'):
                refined_arrays(refined)


if __name__ == '__main__':
    unittest.main()
