"""Actual training-view and typed objective checks, including alias gradients."""
import random
import unittest
from types import SimpleNamespace

import torch

from crystal_dlm.expert_edit import EditOutput, ExpertEditObjective, make_edit_view, materialize_edit_batch
from crystal_dlm.expert_edit_data import quantize_arrays, full_action
from crystal_dlm.fixed_slot import build_special_tokens, MASK_TOKEN_ID


class Tokenizer:
    pad_token_id = 0
    def __init__(self):
        self.vocabulary = {word: i for i, word in enumerate(build_special_tokens())}
    def get_vocab(self):
        return self.vocabulary


class ViewAndLossTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        arrays = {'lengths': [4.] * 3, 'angles': [90.] * 3, 'species': ['Na', 'Cl'],
                  'frac_coords': [[0., 0., 0.], [.5, .5, .5]]}
        old, _, _ = quantize_arrays(arrays, self.tokenizer.get_vocab())
        arrays['lengths'] = [4.2] * 3
        target, _, _ = quantize_arrays(arrays, self.tokenizer.get_vocab())
        self.record = {'record_id': 'example', 'ancestor_id': 'source', 'task': 'G', 'num_atoms': 2,
                       'old_body': old, 'target_body': target, 'action': full_action(old, target, 2),
                       'old_geometry': {'valid': False}, 'target_geometry': {'valid': True},
                       'old_reliable': False, 'target_reliable': True, 'accept_label': True,
                       'content_supervision': True}

    def test_state_view_cannot_read_a_stored_teacher_target(self):
        row = dict(self.record, state_only=True)
        view = make_edit_view(row, 'judge', random.Random(3), [0])
        self.assertEqual(view['kind'], 'inspect')
        self.assertEqual(view['input_body'], self.record['old_body'])
        self.assertEqual(view['active'], [])
        self.assertTrue(all(value == -100 for value in view['targets']))

    def test_masked_and_visible_error_views_use_corrected_targets(self):
        for chance, masked in ((0., True), (.99, False)):
            rng = SimpleNamespace(randrange=lambda n: 0, random=lambda: chance)
            view = make_edit_view(self.record, 'content', rng, [0])
            self.assertEqual(view['targets'][1], self.record['target_body'][1])
            self.assertNotEqual(view['targets'][1], self.record['old_body'][1])
            self.assertEqual(view['input_body'][1], MASK_TOKEN_ID if masked else self.record['old_body'][1])
            self.assertEqual(view['input_body'][7], self.record['old_body'][7])

    def test_inspection_budget_does_not_reveal_teacher_scope_or_keep(self):
        state = make_edit_view(dict(self.record, state_only=True), 'inspect', random.Random(0), [0])
        positive = make_edit_view(self.record, 'inspect', random.Random(0), [0])
        local = dict(self.record, action={'mode': 'local_xyz', 'positions': [8,9,10], 'sites': [0]})
        small = make_edit_view(local, 'inspect', random.Random(0), [0])
        self.assertEqual(state['remaining'], positive['remaining'])
        self.assertEqual(small['remaining'], positive['remaining'])
        self.assertEqual(state['active'], small['active'])

    def test_next_token_supervision_matches_each_actual_scalar_decision(self):
        active = self.record['action']['positions']
        weights = {position: 0. for position in active}
        for cut in range(len(active)):
            rng = SimpleNamespace(randrange=lambda n: cut, random=lambda: 0.)
            legacy = make_edit_view(self.record, 'content', rng, [0], m2t_probability=1.)
            next_view = make_edit_view(self.record, 'content', rng, [0],
                                       content_target_mode='next_token', m2t_probability=1.)
            self.assertEqual(legacy['input_body'], next_view['input_body'])
            self.assertEqual(legacy['active'], next_view['active'])
            positions = [i for i, target in enumerate(next_view['targets']) if target != -100]
            self.assertEqual(positions, [active[cut]])
            self.assertTrue(all(next_view['input_body'][p] == MASK_TOKEN_ID for p in active[cut:]))
            weights[positions[0]] += 1 / len(active)
        self.assertEqual(len(set(weights.values())), 1)

    def test_next_token_loss_does_not_train_unconsumed_suffix_logits(self):
        rng = SimpleNamespace(randrange=lambda n: 0, random=lambda: 0.)
        view = make_edit_view(self.record, 'content', rng, [0], content_target_mode='next_token', m2t_probability=1.)
        batch = materialize_edit_batch([view], self.tokenizer, 'cpu')
        length = batch['input_ids'].shape[1]
        logits = torch.zeros(1, length, len(self.tokenizer.get_vocab()), requires_grad=True)
        output = EditOutput(logits, torch.zeros(1,4,requires_grad=True), torch.zeros(1,20,requires_grad=True),
                            torch.zeros(1,4,requires_grad=True), torch.zeros(1,4,requires_grad=True))
        loss, _ = ExpertEditObjective(self.tokenizer, 'cpu')(output,batch)
        loss.backward()
        self.assertGreater(float(logits.grad[0,2].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[0,3:].abs().sum()), 0.)

    def test_typed_alias_loss_backpropagates_without_inplace_corruption(self):
        rng = SimpleNamespace(randrange=lambda n: 0, random=lambda: 0.)
        views = [make_edit_view(self.record, 'content', rng, [0]),
                 make_edit_view(dict(self.record, task='S'), 'content', rng, [0])]
        batch = materialize_edit_batch(views, self.tokenizer, torch.device('cpu'))
        b, length = batch['input_ids'].shape
        logits = torch.zeros(b, length, len(self.tokenizer.get_vocab()), requires_grad=True)
        output = EditOutput(logits, torch.zeros(b, 4, requires_grad=True),
                            torch.zeros(b, 20, requires_grad=True), torch.zeros(b, 4, requires_grad=True),
                            torch.zeros(b, 4, requires_grad=True))
        loss, metrics = ExpertEditObjective(self.tokenizer, 'cpu')(output, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(metrics['G_content_views'], 1)
        self.assertEqual(metrics['S_content_views'], 1)
        zero, alias = self.tokenizer.vocabulary['<X_000>'], self.tokenizer.vocabulary['<X_100>']
        self.assertLess(float(logits.grad[0, 9, zero]), 0.)
        self.assertLess(float(logits.grad[0, 9, alias]), 0.)

    def test_unlabelled_heads_still_have_finite_connected_zero_gradients(self):
        view = make_edit_view(dict(self.record, state_only=True, old_reliable=None,
                                   old_geometry={'valid': None}), 'inspect', random.Random(0), [0])
        batch = materialize_edit_batch([view], self.tokenizer, torch.device('cpu'))
        length = batch['input_ids'].shape[1]
        tensors = [torch.zeros(1, length, len(self.tokenizer.get_vocab()), requires_grad=True),
                   torch.zeros(1, 4, requires_grad=True), torch.zeros(1, 20, requires_grad=True),
                   torch.zeros(1, 4, requires_grad=True), torch.zeros(1, 4, requires_grad=True)]
        loss, _ = ExpertEditObjective(self.tokenizer, 'cpu')(EditOutput(*tensors), batch)
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertTrue(all(value.grad is not None and torch.isfinite(value.grad).all() for value in tensors))


if __name__ == '__main__':
    unittest.main()
