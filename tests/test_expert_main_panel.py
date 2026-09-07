"""Frozen-panel identities and exact per-request refinement RNG regression."""
import copy
import hashlib
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.run_r03_integrated_body import (EDITOR_PANEL_SCHEMA, editor_panel_tasks,
    B0_ADAPTER_SHA256, base_record, write_endpoint, validate_editor_panel_holdout, skipped_editor_record)
from scripts.refine_dlm_with_crysllmgen import ProposalDataset, frozen_seeded_batches


def loader(dataset, **kwargs):
    return DataLoader(dataset, collate_fn=lambda rows: rows[0], **kwargs)


def graph(index, seed):
    return {'sample_idx':index,'refiner_noise_seed':seed,'n_atom':1,'edge_indices':np.empty((0,2),dtype=int),
            'length':[4.]*3,'angle':[90.]*3,'x_coord':[[0.]*3],'a_type':[11],'to_jimages':np.empty((0,3))}


class PanelTests(unittest.TestCase):
    def fixture(self):
        bodies, plans, seeds = [], [], []
        for i in range(3):
            plans.append({'cohort_ordinal':i,'planner_attempt_id':f'original:{i}', 'body_eligible':i != 0,
                          'plan_state':{'N':1,'elements':['Na'],'counts':[1]},'body_prompt':'original rich condition',
                          'body_prompt_sha256':'p','planner_sampling_seed':17})
            bodies.append({'ordinal':i,'sample_idx':i,'attempt_id':f'original:{i}', 'body_noise_seed':100+i,
                           'body_generation_complete':i != 0,'body_plan_match':i != 0,
                           'body_graph_complete':i==2,'raw_body_token_ids':[1]*11 if i else None,
                           'reason':'original graph failure' if i==1 else 'planner failure' if i==0 else None})
            seeds.append({'sample_idx':i,'body_noise_seed':100+i,'refiner_noise_seed':10000+i})
        return bodies, plans, seeds

    def test_complete_graph_failures_remain_editable_and_failed_plans_remain_in_ledger(self):
        tasks = editor_panel_tasks(*self.fixture(),seed=23)
        self.assertEqual([row['eligible'] for row in tasks],[False,True,True])
        self.assertEqual([row['sample_idx'] for row in tasks],[0,1,2])
        self.assertEqual([row['refiner_noise_seed'] for row in tasks],[10000,10001,10002])
        self.assertEqual(tasks[1]['body_prompt'],'original rich condition')
        self.assertEqual(tasks,editor_panel_tasks(*self.fixture(),seed=23))
        changed = editor_panel_tasks(*self.fixture(),seed=24)
        self.assertNotEqual(tasks[1]['editor_seed'],changed[1]['editor_seed'])
        self.assertEqual(tasks[1]['body_noise_seed'],changed[1]['body_noise_seed'])

    def test_shifted_order_and_missing_complete_tokens_are_engineering_errors(self):
        bodies, plans, seeds = self.fixture()
        with self.assertRaises(ValueError):
            editor_panel_tasks(list(reversed(bodies)),plans,seeds,seed=1)
        bodies[1]['raw_body_token_ids']=[]
        with self.assertRaises(ValueError):
            editor_panel_tasks(bodies,plans,seeds,seed=1)

    def test_original_body_failure_does_not_become_a_planner_failure(self):
        bodies,plans,seeds=self.fixture()
        bodies[1].update(schema='h1_body_safeaxis256_attempt_v1',body_eligible=True,
                         body_generation_complete=False,body_plan_match=False,text='partial token body',
                         earliest_failure_stage='body_sampling',message='original failure detail')
        tasks=editor_panel_tasks(bodies,plans,seeds,seed=1)
        row=skipped_editor_record(tasks[1],bodies[1])
        self.assertTrue(row['body_eligible']);self.assertFalse(row['editor_eligible'])
        self.assertEqual(row['attempt_status'],'body_failure')
        self.assertEqual(row['earliest_failure_stage'],'body_sampling')
        self.assertEqual(row['text'],'partial token body')

    def test_endpoint_writer_restores_original_order_before_uniqueness(self):
        tasks = editor_panel_tasks(*self.fixture(),seed=1)
        records = {i:dict(base_record(tasks[i]),schema=EDITOR_PANEL_SCHEMA,editor_attempted=False) for i in [2,0,1]}
        with TemporaryDirectory() as directory:
            path=Path(directory)
            report=write_endpoint(path,tasks,records,{},runtime=None,elapsed=0.)
            rows=[json.loads(line) for line in (path/'body_attempts.jsonl').read_text().splitlines()]
            self.assertEqual([row['attempt_id'] for row in rows],['original:0','original:1','original:2'])
            self.assertEqual(report['schema'],EDITOR_PANEL_SCHEMA)
            self.assertEqual(report['denominator'],3)

    def test_main_composition_is_excluded_from_training_and_development(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'_CHECKPOINT_SUCCESS').touch()
            data=root/'data';data.mkdir();(data/'DATA_FINAL.json').write_text('{}')
            path=data/'dev.jsonl'
            path.write_text(json.dumps({'source_split':'dev','composition_key':'Na:1'})+'\n')
            pin={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'report_sha256':hashlib.sha256((data/'DATA_FINAL.json').read_bytes()).hexdigest()}
            files={}
            for name in ('adapter_model.safetensors','adapter_config.json','expert_edit_modules.pt',
                         'expert_edit_config.json','periodic_state.pt','periodic_state_config.json',
                         'EXPERT_EDITOR.json','roundtrip_probe.pt'):
                (root/name).write_text('fixture');files[name]=hashlib.sha256(b'fixture').hexdigest()
            (root/'CHECKPOINT_FINAL.json').write_text(json.dumps({'model_files_sha256':files,
                'contract':{'train_files':[],'dev_files':[pin],
                'initialization':{'kind':'original_B0','adapter_sha256':B0_ADAPTER_SHA256}}}))
            cohort=[{'body_eligible':True,'plan_state':{'N':1,'elements':['Na'],'counts':[1]}}]
            with self.assertRaisesRegex(ValueError,'model selection'):
                validate_editor_panel_holdout(root,cohort)
            cohort[0]['plan_state']['elements']=['Cl']
            self.assertTrue(validate_editor_panel_holdout(root,cohort)['evaluation_compositions_absent_from_train_and_dev'])

    def test_refinement_rng_matches_frozen_seed_before_loader_and_survives_sharding(self):
        graphs=[graph(2,8127182874598280710),graph(7,8127182874598280711)]
        expected={}
        for item in graphs:
            seed=item['refiner_noise_seed']
            random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
            # This is the recorded original frozen runner order.
            ds=ProposalDataset([item],SimpleNamespace)
            next(iter(loader(ds,batch_size=1,shuffle=False)))
            expected[item['sample_idx']]=(torch.rand(8),np.random.rand(3),random.random())
        observed={}
        for item in reversed(graphs):
            for batch in frozen_seeded_batches([item],SimpleNamespace,loader,'refiner_noise_seed'):
                observed[int(batch.sample_idx.item())]=(torch.rand(8),np.random.rand(3),random.random())
        for key,(actual_t,actual_n,actual_r) in observed.items():
            want_t,want_n,want_r=expected[key]
            self.assertTrue(torch.equal(actual_t,want_t));np.testing.assert_array_equal(actual_n,want_n)
            self.assertEqual(actual_r,want_r)
        torch.manual_seed(graphs[0]['refiner_noise_seed'])
        self.assertFalse(torch.equal(torch.rand(8),expected[2][0]),'reseed-after-loader would silently change noise')


if __name__=='__main__': unittest.main()
