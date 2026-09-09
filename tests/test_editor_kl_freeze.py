"""Exercise real optimizer state across a content KL latch on CPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from crystal_dlm.rsi_minibatch import train_bounded
from scripts.run_post_refine_cycle import write_json


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.content=torch.nn.Parameter(torch.tensor(0.))
        self.quality_head=torch.nn.Linear(1,1,bias=False)
        torch.nn.init.constant_(self.quality_head.weight,.5)


def view(row,target,cut,branch,mask_seed=0):
    return {'target':target}


def conditional(model,tokenizer,views,branch,support):
    q=torch.stack((model.content,-model.content)).log_softmax(0)
    return torch.stack([q[v['target']] for v in views]),[q for _ in views]


def head_loss(model,tokenizer,rows,device):
    # Content influences the decision features before the latch. It must be
    # frozen despite past Adam momentum and nonzero AdamW weight decay.
    value=model.quality_head((1+model.content).reshape(1,1))
    return (value-1.5).square().mean(),len(rows)


def exercise(directory,branch):
    torch.set_num_threads(1)
    model=Toy();selected=list(model.named_parameters())
    reference={n:p.detach().clone() for n,p in selected}
    optimizer=torch.optim.AdamW([p for _,p in selected],lr=.35,weight_decay=.1)
    snapshots=[];original_step=optimizer.step
    def step(*args,**kwargs):
        value=original_step(*args,**kwargs)
        snapshots.append({n:p.detach().clone() for n,p in selected})
        return value
    optimizer.step=step
    rows=[{'pair_id':str(i),'num_sites':1,'action_positions':[8],
           'chosen_tokens':0,'rejected_tokens':1,'proposal_tokens':[0],
           'accept_target':1,'mode_target':None} for i in range(4)]
    spec={'branch':branch,'batch_size':1,'seed':7,'epochs':2,'mask_cuts':1,'beta':.1,
          'anchor_weight':.2,'reference_kl_weight':1.,'max_reference_kl':.02,
          'max_training_seconds':60,'continue_heads_after_content_KL':True}
    with patch('crystal_dlm.rsi_minibatch.training_view',side_effect=view), \
         patch('crystal_dlm.rsi_minibatch.conditional_batch',side_effect=conditional), \
         patch('crystal_dlm.rsi_minibatch.editor_head_loss',side_effect=head_loss):
        result=train_bounded(model,None,rows,spec,selected,reference,optimizer,None,directory,write_json)
    return result,snapshots


class ContentFreeze(unittest.TestCase):
    def test_editor_content_stays_bit_exact_while_heads_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result,snapshots=exercise(root,'E')
            exposure=json.loads((root/'EXPOSURE_rank0.json').read_text())
            self.assertGreater(len(snapshots),2)
            self.assertTrue(exposure['head_continuation_after_KL'])
            self.assertEqual(exposure['content_optimizer_steps'],1)
            self.assertEqual(exposure['head_optimizer_steps'],8)
            self.assertGreater(exposure['KL_trigger']['maximum_rank_KL'],.02)
            for state in snapshots[1:]:
                self.assertTrue(torch.equal(state['content'],snapshots[0]['content']))
            self.assertFalse(torch.equal(snapshots[-1]['quality_head.weight'],snapshots[0]['quality_head.weight']))
            self.assertEqual(sum(exposure['content_updated_pair_visits'].values()),1)
            self.assertEqual(sum(exposure['head_updated_pair_visits'].values()),8)
            self.assertEqual(result[1],1)

    def test_generator_still_stops_all_updates_at_kl_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result,snapshots=exercise(root,'G')
            exposure=json.loads((root/'EXPOSURE_rank0.json').read_text())
            self.assertEqual(len(snapshots),1)
            self.assertFalse(exposure['head_continuation_after_KL'])
            self.assertEqual(exposure['head_optimizer_steps'],0)
            self.assertEqual(sum(exposure['pair_visits'].values()),2)
            self.assertEqual(sum(exposure['content_updated_pair_visits'].values()),1)
            self.assertEqual(result[0],1)


if __name__=='__main__': unittest.main()
