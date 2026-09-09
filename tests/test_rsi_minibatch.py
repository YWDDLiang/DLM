"""Coverage, prompt conditioning, gradients, and per-request batching contracts."""
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from test_r03_physics_transfer import TinyTokenizer,make_body
from crystal_dlm.r03_physics_transfer import build_repair_constraints
from crystal_dlm.rsi_minibatch import epoch_indices,training_view,conditional_batch,propose_ranked_batch
from crystal_dlm.rsi_preference import propose_ranked_editor


class Policy(torch.nn.Module):
    def __init__(self,width):
        super().__init__();self.vector=torch.nn.Parameter(torch.linspace(-.2,.2,width));self.calls=0
    def forward(self,ids,attention_mask=None,edit_context=None):
        self.calls+=1;batch,length=ids.shape
        logits=self.vector[None,None,:].expand(batch,length,-1)
        # Each request's prompt contributes to its logits, independently of padding.
        tilt=(ids[:,0]%3).float()[:,None,None]*torch.linspace(0,.1,len(self.vector))[None,None,:]
        return SimpleNamespace(logits=logits+tilt,
            mode_logits=torch.tensor([[0.,0.,4.,0.]]).expand(batch,-1),
            site_logits=torch.zeros(batch,20),count_logits=torch.zeros(batch,4),
            quality_logits=torch.tensor([[0.,0.,0.,-5.]]).expand(batch,-1))


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.tok=TinyTokenizer();cls.support=build_repair_constraints(cls.tok)
        cls.width=max(cls.tok.vocab.values())+1
    def test_rare_records_no_longer_oversampled(self):
        counts=Counter(i for _,batch in epoch_indices(185,batch_size=8,world=4,epochs=4,seed=7) for i in batch)
        self.assertEqual(len(counts),185);self.assertEqual(sum(counts.values()),768)
        self.assertGreaterEqual(min(counts.values()),4);self.assertLessEqual(max(counts.values()),8)
        self.assertLessEqual(sum(counts[i] for i in (0,1,2)),24)
    def test_batch_matches_single_and_gradients(self):
        a,_=make_body(self.tok);b,_=make_body(self.tok,length=50)
        examples=[{'num_sites':2,'prompt':'first'}, {'num_sites':2,'prompt':'second'}]
        views=[training_view(examples[0],a,0,'G'),training_view(examples[1],b,8,'G')]
        model=Policy(self.width);actual,_=conditional_batch(model,self.tok,views,'G',self.support)
        self.assertEqual(model.calls,1)
        singles=torch.cat([conditional_batch(model,self.tok,[view],'G',self.support)[0] for view in views])
        torch.testing.assert_close(actual,singles)
        ga=torch.autograd.grad(actual.sum(),model.vector,retain_graph=True)[0]
        gb=torch.autograd.grad(singles.sum(),model.vector)[0]
        torch.testing.assert_close(ga,gb)
    def test_prompt_reaches_model(self):
        body,_=make_body(self.tok)
        a={'num_sites':2,'prompt':'unrecognized'}
        b={'num_sites':2,'prompt':'<LA_050>'}
        model=Policy(self.width)
        scores,_=conditional_batch(model,self.tok,[training_view(a,body,0,'G'),training_view(b,body,0,'G')],'G',self.support)
        self.assertNotEqual(float(scores[0]),float(scores[1]))
    def test_registered_semantic_groups_cover_native_numeric_slots(self):
        body,_=make_body(self.tok)
        groups=[[1,2,3,4,5,6],[8],[12],[9],[13],[10],[14]]
        example={'num_sites':2,'prompt':'p','plan_state':{'N':2},'generation_groups':groups}
        view=training_view(example,body,6,'G',mask_seed=19)
        self.assertEqual(set(view['order'][:6]),set(groups[0]))
        self.assertEqual(view['position'],8)
        self.assertEqual(view['current'][1:7],body[1:7])
        with self.assertRaisesRegex(ValueError,'numeric support'):
            training_view({**example,'generation_groups':[[1,2,3]]},body,0,'G')
    def test_editor_batch_preserves_independent_sampling(self):
        body,_=make_body(self.tok);model=Policy(self.width)
        requests=[dict(prompt='p',body=body,n=2,seed=s,known_sun=False) for s in (13,17)]
        singles=[propose_ranked_editor(model,self.tok,support=self.support,**r) for r in requests]
        batched=propose_ranked_batch(model,self.tok,requests,support=self.support,batch_size=2)
        for a,b in zip(singles,batched):
            for key in ('proposal_tokens','final_tokens','action','forward_calls','sampled_attempt_tokens'):
                self.assertEqual(a[key],b[key])
            self.assertEqual([(x['position'],x['token_id']) for x in a['sampling_trace']],
                             [(x['position'],x['token_id']) for x in b['sampling_trace']])


if __name__=='__main__':unittest.main()
