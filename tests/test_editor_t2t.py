"""Actual gradients, complete-pass coverage, held-out groups, and KL interruption."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from test_r03_physics_transfer import TinyTokenizer, make_body
from crystal_dlm.editor_t2t import dense_view, train_t2t
from crystal_dlm.rsi_minibatch import decision_head_parameter
from crystal_dlm.fixed_slot import MASK_TOKEN_ID


class DensePolicy(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.content = torch.nn.Parameter(torch.zeros(width))
        self.mode_head = torch.nn.Linear(1, 4)
        self.quality_head = torch.nn.Linear(1, 4)

    def forward(self, ids, attention_mask=None, edit_context=None, detach_head_features=False):
        hidden = (self.content.sum() + 1.).expand(len(ids), 1)
        if detach_head_features: hidden = hidden.detach()
        return SimpleNamespace(logits=self.content[None, None].expand(len(ids), ids.shape[1], -1),
            mode_logits=self.mode_head(hidden), quality_logits=self.quality_head(hidden),
            site_logits=hidden.expand(len(ids), 20), count_logits=hidden.expand(len(ids), 4))


class DenseTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        tok = TinyTokenizer(); left, _ = make_body(tok)
        target = left.copy(); target[8] = tok.vocab['<X_010>']
        rows = [dict(pair_id=str(i), source_split='train', prompt='p', current_tokens=left,
            content_target_tokens=target, content_positions=[8, 9, 10], num_sites=2,
            mode_target=0, accept_target=1, proposal_tokens=target, action_positions=[8, 9, 10],
            Stable_promotion=True) for i in range(4)]
        return tok, rows

    def train(self, *, heads_only=False, threshold=10., rate=.01, epochs=4):
        tok, rows = self.fixture(); torch.manual_seed(31)
        model = DensePolicy(max(tok.vocab.values())+1)
        selected = list(model.named_parameters())
        reference = {n: p.detach().clone() for n, p in selected}
        optimizer = torch.optim.AdamW([
            dict(params=[p for n, p in selected if not decision_head_parameter(n)], lr=rate),
            dict(params=[p for n, p in selected if decision_head_parameter(n)], lr=.001)], weight_decay=.1)
        spec = dict(branch='E', batch_size=1, epochs=epochs, seed=17, max_training_seconds=60,
            max_reference_kl=threshold, reference_kl_weight=1., editor_heads_only=heads_only)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write = lambda path, value: path.write_text(json.dumps(value))
            result = train_t2t(model, tok, rows, spec, selected, reference, optimizer, None, root, write)
            report = json.loads((root/'EXPOSURE_rank0.json').read_text())
        return model, reference, optimizer, report, result

    def test_all_targets_get_actual_complete_pass_updates(self):
        model, reference, _, report, _ = self.train(epochs=8)
        self.assertEqual(report['content_optimizer_steps'], 8)
        self.assertEqual(report['head_optimizer_steps'], 32)
        self.assertEqual(report['content_updated_pair_visits'], {str(i): 8 for i in range(4)})
        self.assertEqual(report['content_updated_token_visits'], {str(i): 24 for i in range(4)})
        self.assertFalse(torch.equal(model.content, reference['content']))

    def test_kl_discards_partial_pass_and_continues_heads(self):
        _, _, _, report, _ = self.train(threshold=1e-10, rate=.2)
        self.assertEqual(report['content_optimizer_steps'], 1)
        self.assertEqual(report['head_optimizer_steps'], 16)
        self.assertEqual(report['content_updated_pair_visits'], {str(i): 1 for i in range(4)})
        self.assertTrue(report['KL_trigger']['discarded_partial_pass'])

    def test_heads_control_content_is_bit_exact_with_adamw(self):
        model, reference, optimizer, report, _ = self.train(heads_only=True)
        self.assertTrue(torch.equal(model.content, reference['content']))
        self.assertIsNone(model.content.grad)
        self.assertNotIn(model.content, optimizer.state)
        self.assertEqual(report['content_updated_pair_visits'], {})
        self.assertTrue(any(not torch.equal(p, reference[n]) for n, p in model.named_parameters() if decision_head_parameter(n)))

    def test_dense_t2t_and_masked_views_keep_old_structure_visible(self):
        _, rows = self.fixture(); row = rows[0]
        for masked in (False, True):
            view = dense_view(row, [1], masked=masked)
            self.assertEqual(view['old_body'], row['current_tokens'])
            self.assertEqual(sum(t != -100 for t in view['targets']), 3)
            self.assertEqual(view['input_body'][8], MASK_TOKEN_ID if masked else row['current_tokens'][8])

    def test_real_editor_detached_head_does_not_train_content(self):
        from test_expert_edit_model import _case
        case = _case(inspection=True)
        out = case.model(case.current, attention_mask=case.attention, edit_context=case.context, detach_head_features=True)
        (out.mode_logits.sum() + out.quality_logits.sum()).backward()
        self.assertTrue(any(p.grad is not None for n, p in case.model.named_parameters() if decision_head_parameter(n)))
        self.assertTrue(all(p.grad is None for n, p in case.model.named_parameters() if not decision_head_parameter(n)))

    def test_same_composition_cannot_cross_trial_split(self):
        path = Path(__file__).resolve().parents[1]/'operations/r03_c3fd_main_20260907/prepare_editor_t2t_trial.py'
        spec = importlib.util.spec_from_file_location('trial_prepare_test', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        plans = [dict(ancestor_id=str(i), plan_state=dict(reduced_formula=str(i//2))) for i in range(100)]
        split = module.group_split(plans)
        self.assertEqual(split, module.group_split(plans))
        self.assertEqual(Counter(r['split'] for r in split), dict(train=60, dev=20, final=20))
        for i in range(0, 100, 2): self.assertEqual(split[i]['split'], split[i+1]['split'])


if __name__ == '__main__': unittest.main()
