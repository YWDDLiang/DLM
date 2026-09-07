import hashlib
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

from crystal_dlm.expert_edit_data import (build_geometry_auxiliary, geometry_auxiliary_examples,
                                        quantize_arrays, composition_key)
from crystal_dlm.fixed_slot import build_special_tokens


class Tokenizer:
    def get_vocab(self):
        return {token:i for i,token in enumerate(build_special_tokens())}


class GeometryAuxiliaryTests(unittest.TestCase):
    def test_local_corruption_changes_only_its_declared_quantized_scope(self):
        tokenizer = Tokenizer()
        clean = {'lengths':[6.]*3, 'angles':[90.]*3, 'species':['Na']*12,
                 'frac_coords': [[i/3,j/2,k/2] for i in range(3) for j in range(2) for k in range(2)]}
        for seed in range(5):
            rows = geometry_auxiliary_examples(clean, tokenizer.get_vocab(), random.Random(seed))
            self.assertEqual({row['bucket'] for row in rows}, {'single_site','cooperative_sites','all_xyz','lattice_coupled'})
            for row in rows:
                self.assertFalse(row['old_geometry']['valid'])
                self.assertTrue(row['target_geometry']['valid'])
                changes = {i for i,(old,target) in enumerate(zip(row['old_body'],row['target_body'])) if old != target}
                self.assertTrue(changes)
                self.assertTrue(changes.issubset(row['action']['positions']))
                self.assertEqual(changes, set(row['action']['changed_positions']))
                if row['action']['mode'] == 'local_xyz':
                    self.assertEqual(row['old_body'][:7],row['target_body'][:7])
                    self.assertIn(len(row['action']['sites']), (1,2,4,8))

    def test_clean_training_source_excludes_main_compositions_and_gt_soft_hints(self):
        tokenizer = Tokenizer()
        vocab = tokenizer.get_vocab()
        inverse = {v:k for k,v in vocab.items()}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for index,species in enumerate((['Na','Cl'], ['Li'], ['Mg','O'])):
                arrays = {'lengths':[4.]*3,'angles':[90.]*3,'species':species,
                          'frac_coords':[[0.,0.,0.]] if len(species)==1 else [[0.,0.,0.],[.5,.5,.5]]}
                body, _, _ = quantize_arrays(arrays, vocab)
                sources.append({'source_row_idx':index,'source_split':'train','source_answer':''.join(inverse[x] for x in body),
                                'closure':{'source_answer_is_clean_teacher':True},
                                'plan_state':{'spacegroup_bucket':'ORACLE_SG','volume_per_atom_bin':'ORACLE_VOLUME'},
                                'prompt':'ORACLE_PROMPT_MUST_NOT_BE_USED'})
            path, heldout = root/'train.jsonl',root/'heldout.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in sources))
            heldout.write_text(json.dumps({'plan_state':{'N':2,'elements':['Na','Cl'],'counts':[1,1]}})+'\n')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            report = build_geometry_auxiliary(path,digest(path),heldout,digest(heldout),tokenizer,root/'out',source_limit=2)
            self.assertEqual(report['admitted_sources'],2)
            rows = [json.loads(line) for split in ('train','dev') for line in (root/f'out/{split}.jsonl').read_text().splitlines()]
            for row in rows:
                self.assertNotEqual(row['composition_key'],composition_key(['Na','Cl']))
                self.assertNotIn('ORACLE',row['prompt'])
                self.assertEqual(row['task'],'G')
                self.assertIsNone(row['old_reliable'])
                self.assertIsNone(row['target_reliable'])
                self.assertIsNone(row['gain_eV_atom'])
                if row.get('state_only'):
                    self.assertNotIn('target_body',row)


if __name__ == '__main__':
    unittest.main()
