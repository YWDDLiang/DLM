"""Independent checks of the MP20 geometry-only auxiliary contract."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from crystal_dlm.expert_edit_data import (
    build_geometry_auxiliary,
    certify_geometry,
    composition_key,
    decode_body,
    geometry_auxiliary_examples,
    quantize_arrays,
)
from crystal_dlm.fixed_slot import build_special_tokens


class Tokenizer:
    def __init__(self):
        self.vocabulary = {token: index for index, token in enumerate(build_special_tokens())}

    def get_vocab(self):
        return self.vocabulary


def clean_geometry(species):
    return {
        'lengths': [10., 10., 10.], 'angles': [90., 90., 90.],
        'species': list(species),
        'frac_coords': [[(i % 5) / 5., ((i // 5) % 2) / 2., (i // 10) / 2.]
                        for i in range(len(species))],
    }


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GeometryAuxiliaryAudit(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.vocabulary = self.tokenizer.get_vocab()
        self.inverse = {index: token for token, index in self.vocabulary.items()}

    def test_each_emitted_scope_is_an_executable_certified_recovery(self):
        observed_modes, local_sizes = set(), set()
        for n in (1, 2, 3, 8, 20):
            clean = clean_geometry(['Na'] * n)
            for seed in range(24):
                examples = geometry_auxiliary_examples(clean, self.vocabulary, random.Random(seed))
                self.assertEqual(len(examples), 1 if n == 1 else 4)
                for example in examples:
                    old, target, action = example['old_body'], example['target_body'], example['action']
                    old_arrays, target_arrays = decode_body(old, self.inverse), decode_body(target, self.inverse)
                    self.assertIs(certify_geometry(old_arrays)['valid'], False)
                    self.assertIs(certify_geometry(target_arrays)['valid'], True)
                    self.assertEqual(Counter(old_arrays['species']), Counter(target_arrays['species']))
                    protected = [0] + [7 + 4 * site for site in range(n)]
                    self.assertTrue(all(old[pos] == target[pos] for pos in protected))
                    changed = [pos for pos, values in enumerate(zip(old, target)) if values[0] != values[1]]
                    self.assertEqual(action['changed_positions'], changed)
                    self.assertEqual(action['changed_sites'], sorted({(pos - 8) // 4 for pos in changed if pos >= 8}))
                    self.assertTrue(set(changed).issubset(action['positions']))
                    reconstructed = old.copy()
                    for pos in action['positions']:
                        reconstructed[pos] = target[pos]
                    self.assertEqual(reconstructed, target)
                    observed_modes.add(action['mode'])
                    if action['mode'] == 'local_xyz':
                        local_sizes.add(len(action['sites']))
                        self.assertEqual(action['positions'], [8 + 4 * site + axis
                                         for site in action['sites'] for axis in range(3)])
        self.assertEqual(observed_modes, {'local_xyz', 'all_xyz', 'full_cell'})
        self.assertEqual(local_sizes, {1, 2, 4, 8})

    def test_bad_clean_target_never_becomes_auxiliary_teacher(self):
        bad = clean_geometry(['Na', 'Cl'])
        bad['frac_coords'][1] = bad['frac_coords'][0].copy()
        self.assertEqual(geometry_auxiliary_examples(bad, self.vocabulary, random.Random(1)), [])

    def test_builder_discards_oracle_fields_and_excludes_supercell_compositions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, heldout, output = root / 'source.jsonl', root / 'heldout.jsonl', root / 'output'
            rows = []
            for index, species in enumerate((['Na', 'Cl'], ['Na', 'Na', 'Cl', 'Cl'], ['K', 'Br'], ['Li', 'F'])):
                _, arrays, _ = quantize_arrays(clean_geometry(species), self.vocabulary)
                rows.append({
                    'source_row_idx': index, 'source_split': 'train',
                    'closure': {'source_answer_is_clean_teacher': True},
                    'source_answer': arrays['answer'],
                    'answer': 'THIS_ALTERNATE_ANSWER_MUST_NOT_BE_READ',
                    'prompt': 'ORACLE_PROMPT_SENTINEL',
                    'plan_state': {'spacegroup_bucket': 'ORACLE_SG_SENTINEL',
                                   'volume_per_atom_bin': 'ORACLE_VPA_SENTINEL'},
                })
            write_rows(source, rows)
            write_rows(heldout, [{'body_eligible': True, 'plan_state':
                                 {'N': 2, 'elements': ['Na', 'Cl'], 'counts': [1, 1]}}])
            report = build_geometry_auxiliary(source, file_hash(source), heldout, file_hash(heldout),
                                              self.tokenizer, output, source_limit=2, seed=14)
            emitted = [json.loads(line) for split in ('train', 'dev')
                       for line in (output / (split + '.jsonl')).read_text().splitlines()]
            self.assertEqual(report['admitted_sources'], 2)
            self.assertEqual({row['source_row_idx'] for row in emitted}, {2, 3})
            excluded = composition_key(['Na', 'Cl'])
            for row in emitted:
                self.assertNotEqual(row['composition_key'], excluded)
                self.assertEqual(row['task'], 'G')
                self.assertIsNone(row['gain_eV_atom'])
                self.assertIsNone(row['old_reliable'])
                self.assertIsNone(row['target_reliable'])
                self.assertFalse(row['GT_soft_hints_used'])
                self.assertNotIn('ORACLE_', row['prompt'])
                self.assertNotIn('THIS_ALTERNATE', json.dumps(row))
                prompt_plan = json.loads(row['prompt'].split('plan_state: ', 1)[1].split('\n', 1)[0])
                for key, value in (('spacegroup_bucket', 'sg_unknown'),
                                   ('volume_per_atom_bin', 'volpa_unknown'),
                                   ('lattice_system', 'unknown'), ('prototype_key', 'unknown')):
                    self.assertEqual(prompt_plan[key], value)
                self.assertEqual(composition_key(prompt_plan), row['composition_key'])
                expected_split = 'dev' if int(hashlib.sha256(row['composition_key'].encode()).hexdigest()[:8], 16) % 8 == 0 else 'train'
                self.assertEqual(row['source_split'], expected_split)
            clean_states = [row for row in emitted if row.get('state_only')]
            self.assertEqual(len(clean_states), 2)
            self.assertTrue(all(not row['content_supervision'] and row['accept_label'] is None
                                and 'target_body' not in row for row in clean_states))


if __name__ == '__main__':
    unittest.main()
