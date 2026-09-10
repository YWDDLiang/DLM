import sys
from pathlib import Path
from unittest.mock import patch
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from crystal_dlm import construction_recovery as recovery
from crystal_dlm import r03_geometry_bridge as bridge
from test_r03_geometry_bridge import TinyTokenizer, canvas, groups, native_constraints, lattice_logits


class CascadeTests(unittest.TestCase):
    def test_relaxed_drafts_keep_labels_without_zero_support_content_targets(self):
        example={'chosen_tokens':[1],'rejected_tokens':[0]}
        keep,bad=recovery.restrict_training_content(example,'G',lambda tokens:tokens==[1])
        self.assertTrue(keep);self.assertEqual(bad,['rejected_tokens'])
        self.assertEqual(example['healthy_anchor_tokens'],[1]);self.assertIsNone(example['chosen_tokens'])
        example={'chosen_tokens':[0],'rejected_tokens':[0],'mode_target':0,'accept_target':0}
        keep,bad=recovery.restrict_training_content(example,'E',lambda tokens:False)
        self.assertTrue(keep);self.assertEqual(example['mode_target'],0);self.assertIsNone(example['chosen_tokens'])

    def setUp(self):
        self.tokenizer = TinyTokenizer()
        self.body = canvas(self.tokenizer, 3, length=10, coords=[(0,0,0),(0,0,50),(0,0,None)])[0,2:].tolist()
        self.task = {'plan_state':{'N':3},'body_noise_seed':17,'body_prompt':'same prompt'}

    def test_stages_expand_then_relax_only_final_Z(self):
        calls = []
        def construct(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) < 5:
                raise bridge.GeometryNoLegalSupport({'reason':'pbc_no_legal_completion','failure_positions':[18]},torch.tensor([self.body]),0)
            result = self.body.copy(); result[18] = self.tokenizer.vocab['<Z_000>']
            return torch.tensor([result]), {}
        with patch.object(recovery, 'neighbor_sites', return_value=[1,2]):
            result, report = recovery.construct_cascade(None,None,self.task,None,construct=construct,
                constraints={},repair_constraints={},geometry_api=bridge,
                complete_geometry=lambda _: {'supported':False,'reason':'native_pair_below_0.5A'})
        self.assertEqual(len(calls),5)
        self.assertEqual(calls[1]['initial_body'][16:19],[126336]*3)
        self.assertEqual(calls[1]['initial_body'][:16],self.body[:16])
        self.assertTrue(all(calls[2]['initial_body'][i]==126336 for i in [12,13,14,16,17,18]))
        self.assertTrue(all(calls[3]['initial_body'][i]==126336 for i in range(1,7)))
        for call in calls[1:]:
            self.assertEqual([call['initial_body'][i] for i in [0,7,11,15]], [self.body[i] for i in [0,7,11,15]])
        self.assertEqual(calls[4]['initial_body'][8:10],self.body[8:10])
        self.assertEqual([c['relax_final_z'] for c in calls],[False]*4+[True])
        self.assertEqual(len({c['noise_seed_override'] for c in calls}),5)
        self.assertFalse(report['complete_geometry']['supported'])
        self.assertTrue(report['construction_recovery']['final_Z_relaxed'])

    def test_neighbor_search_handles_partial_periodic_coordinates(self):
        constraints = bridge.build_repair_constraints(self.tokenizer)
        self.assertEqual(recovery.neighbor_sites(self.body,3,[2],constraints),[0,1,2])

    def test_lattice_failure_reopens_gamma_before_retrying_coordinates(self):
        calls = []
        def construct(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise bridge.GeometryNoLegalSupport(
                    {'reason':'nondegenerate_lattice_has_no_legal_continuation'}, torch.tensor([self.body]), 0)
            return torch.tensor([self.body]), {}
        _, report = recovery.construct_cascade(None, None, self.task, None, construct=construct,
            constraints={}, repair_constraints={}, geometry_api=bridge,
            complete_geometry=lambda _: {'supported':True}, adaptive_lattice=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]['initial_body'][6], 126336)
        self.assertEqual(calls[1]['initial_body'][:6], self.body[:6])
        self.assertTrue(calls[1]['lattice_gamma_last'])
        self.assertEqual([calls[1]['initial_body'][i] for i in [0,7,11,15]],
                         [self.body[i] for i in [0,7,11,15]])
        self.assertEqual(report['construction_recovery']['episodes'][1]['stage'], 'conditional_gamma')

    def test_final_Z_escape_changes_DLM_mask_but_keeps_alias_schema(self):
        x = torch.tensor([[1,1,*self.body]])
        kwargs = dict(current_tokens=x,prompt_length=2,gen_length=len(self.body),semantic_group=4,step_in_group=0)
        strict = bridge.ConstructionGeometryMonitor(enabled=True,tokenizer=self.tokenizer,
            generation_position_groups=groups(3),native_constraints=native_constraints(self.tokenizer))
        with self.assertRaises(bridge.GeometryNoLegalSupport):
            strict.apply_logits(lattice_logits(self.tokenizer,x,[18]),**kwargs)
        relaxed = bridge.ConstructionGeometryMonitor(enabled=True,tokenizer=self.tokenizer,
            generation_position_groups=groups(3),native_constraints=native_constraints(self.tokenizer),relax_final_z=True)
        logits=lattice_logits(self.tokenizer,x,[18]);relaxed.apply_logits(logits,**kwargs)
        self.assertGreater(float(logits[0,20,self.tokenizer.vocab['<Z_000>']]),torch.finfo(logits.dtype).min)
        self.assertEqual(float(logits[0,20,self.tokenizer.vocab['<Z_100>']]),torch.finfo(logits.dtype).min)
        self.assertTrue(relaxed.report()['events'][-1]['Z_distance_support_relaxed'])
