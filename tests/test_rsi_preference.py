"""Behavioral fixtures for legal gradients and retention; no physical claims."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from test_r03_physics_transfer import TinyTokenizer,make_body
from crystal_dlm.r03_physics_transfer import build_repair_constraints
from crystal_dlm.rsi_preference import legal_vector,conditional_logp,propose_editor
from crystal_dlm.fixed_slot import MASK_TOKEN_ID


class ScalarPolicy(torch.nn.Module):
    def __init__(self,width):
        super().__init__();self.vector=torch.nn.Parameter(torch.zeros(width))
    def forward(self,ids,attention_mask=None):
        return SimpleNamespace(logits=self.vector[None,None,:].expand(1,ids.shape[1],-1))


class PreferenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.tokenizer=TinyTokenizer()
        cls.support=build_repair_constraints(cls.tokenizer)
        cls.width=max(cls.tokenizer.vocab.values())+1

    def test_bad_cell_gamma_and_zero_length_have_zero_support(self):
        body,_=make_body(self.tokenizer)
        body[1]=MASK_TOKEN_ID
        vector,report=legal_vector(torch.zeros(self.width),body,2,1,self.support)
        self.assertTrue(report['available'])
        self.assertEqual(float(vector[self.tokenizer.vocab['<LA_000>']]),torch.finfo(vector.dtype).min)
        body,_=make_body(self.tokenizer)
        body[4]=self.tokenizer.vocab['<AA_001>']
        body[5]=self.tokenizer.vocab['<AB_179>']
        body[6]=MASK_TOKEN_ID
        vector,report=legal_vector(torch.zeros(self.width),body,2,6,self.support)
        self.assertEqual(float(vector[self.tokenizer.vocab['<AG_001>']]),torch.finfo(vector.dtype).min)

    def test_alias_gradient_and_periodic_contact_veto(self):
        body,_=make_body(self.tokenizer,coords=((0,0,0),(0,0,50)))
        body[-1]=MASK_TOKEN_ID
        raw=torch.zeros(self.width,requires_grad=True)
        vector,report=legal_vector(raw,body,2,len(body)-1,self.support)
        self.assertTrue(report['available'])
        self.assertEqual(float(vector[self.tokenizer.vocab['<Z_000>']]),torch.finfo(vector.dtype).min)
        body,_=make_body(self.tokenizer,coords=((0,0,0),(50,50,50)))
        body[-1]=MASK_TOKEN_ID
        vector,_=legal_vector(raw,body,2,len(body)-1,self.support)
        vector.log_softmax(-1)[self.tokenizer.vocab['<Z_000>']].backward()
        self.assertGreater(float(raw.grad[self.tokenizer.vocab['<Z_000>']]),0)
        self.assertGreater(float(raw.grad[self.tokenizer.vocab['<Z_100>']]),0)

    def test_preference_gradient_moves_winner_and_reference_is_zero_at_start(self):
        policy=ScalarPolicy(self.width)
        win,_=make_body(self.tokenizer,length=40);lose,_=make_body(self.tokenizer,length=50)
        example={'num_sites':2,'prompt':'original prompt'}
        with torch.no_grad():
            rw=conditional_logp(policy,self.tokenizer,example,win,0,'G',self.support)
            rl=conditional_logp(policy,self.tokenizer,example,lose,0,'G',self.support)
        w=conditional_logp(policy,self.tokenizer,example,win,0,'G',self.support)
        l=conditional_logp(policy,self.tokenizer,example,lose,0,'G',self.support)
        margin=(w-rw)-(l-rl)
        self.assertEqual(float(margin),0.)
        loss=-torch.nn.functional.logsigmoid(.1*margin)
        loss.backward()
        self.assertLess(float(policy.vector.grad[win[1]]),0.)
        self.assertGreater(float(policy.vector.grad[lose[1]]),0.)

    def test_known_sun_keep_is_exact_and_uses_one_call(self):
        policy=ScalarPolicy(self.width);body,_=make_body(self.tokenizer)
        # A small learned EDIT preference is outweighed by the declared SUN prior.
        output=SimpleNamespace(mode_logits=torch.tensor([[0.,0.,0.,1.]]))
        with patch('crystal_dlm.rsi_preference.forward_view',return_value=(output,2)):
            result=propose_editor(policy,self.tokenizer,prompt='p',body=body,n=2,support=self.support,
                                  seed=1,known_sun=True)
        self.assertEqual(result['final_tokens'],body)
        self.assertEqual(result['proposal_tokens'],body)
        self.assertFalse(result['proposal_generated'])
        self.assertEqual(result['forward_calls'],1)


if __name__=='__main__': unittest.main()
