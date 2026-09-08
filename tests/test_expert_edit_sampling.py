"""Executable inference contracts, including conditional reveal and gate controls."""
import unittest

import torch
from torch import nn

from crystal_dlm.expert_edit import EditOutput, edit_structure, edit_structures
from crystal_dlm.expert_edit_data import quantize_arrays
from crystal_dlm.fixed_slot import build_special_tokens, MASK_TOKEN_ID


class Tokenizer:
    pad_token_id = 0
    def __init__(self):
        self.vocab = {text: i for i, text in enumerate(build_special_tokens())}
    def get_vocab(self):
        return self.vocab
    def __call__(self, text, **kwargs):
        return {'input_ids': [0]}


class ConditionalEditor(nn.Module):
    def __init__(self, tokenizer, target, accept):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.tokenizer, self.target, self.accept = tokenizer, target, accept
        self.forward_calls, self.canvases = 0, []
    def forward(self, input_ids, attention_mask=None, edit_context=None):
        self.forward_calls += 1
        self.canvases.append(input_ids[0, 1:].tolist())
        logits = torch.full((*input_ids.shape, len(self.tokenizer.vocab)), -100.)
        for pos, token in enumerate(self.target):
            logits[0, pos+1, token] = 100.
        return EditOutput(logits, torch.tensor([[-10., -10., -10., 10.]]), torch.zeros(1,20),
                          torch.zeros(1,4), torch.tensor([[10.,10.,10.,10. if self.accept else -10.]]))


class EditorSamplingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        arrays = {'lengths': [4.]*3, 'angles': [90.]*3, 'species': ['Na', 'Cl'],
                  'frac_coords': [[0.,0.,0.], [.5,.5,.5]]}
        self.old, _, _ = quantize_arrays(arrays, self.tokenizer.vocab)
        arrays['lengths'][0] = 4.2
        self.target, _, _ = quantize_arrays(arrays, self.tokenizer.vocab)
    def run_sample(self, model, **kwargs):
        return edit_structure(model, self.tokenizer, prompt='fixture', body=self.old, num_sites=2,
                              allowed_modes={'G':[0,3], 'S':[3]}, tasks=('G',), **kwargs)
    def test_scalar_reveal_reads_previous_draw_and_counts_every_forward(self):
        model = ConditionalEditor(self.tokenizer, self.target, True)
        result = self.run_sample(model)
        self.assertEqual(result['body'], self.target)
        self.assertEqual(result['forward_calls'], 14)
        self.assertEqual(model.forward_calls, 14)
        self.assertEqual(model.canvases[1][1], MASK_TOKEN_ID)
        self.assertEqual(model.canvases[2][1], self.target[1])
        self.assertEqual(model.canvases[2][2], MASK_TOKEN_ID)
        self.assertEqual(result['changed_numeric_tokens'], 1)
    def test_gate_controls_keep_the_identical_proposal(self):
        rejected = self.run_sample(ConditionalEditor(self.tokenizer, self.target, False), seed=8)
        all_accept = self.run_sample(ConditionalEditor(self.tokenizer, self.target, False), seed=8, accept_all=True)
        self.assertEqual(rejected['trace'][0]['proposal_body'], all_accept['trace'][0]['proposal_body'])
        self.assertEqual(rejected['body'], self.old)
        self.assertEqual(all_accept['body'], self.target)

    def test_visible_old_canvas_preserves_future_inputs_and_consumes_new_prefix(self):
        model = ConditionalEditor(self.tokenizer, self.target, True)
        result = self.run_sample(model, proposal_input='old_values')
        self.assertEqual(model.canvases[1], self.old)
        self.assertEqual(model.canvases[2][1], self.target[1])
        self.assertEqual(model.canvases[2][2:], self.old[2:])
        self.assertNotIn(MASK_TOKEN_ID, model.canvases[2])
        self.assertEqual(result['body'], self.target)
        self.assertEqual(result['forward_calls'], 14)
        rejected = self.run_sample(ConditionalEditor(self.tokenizer, self.target, False),
                                   proposal_input='old_values')
        self.assertEqual(rejected['trace'][0]['proposal_body'], self.target)
        self.assertEqual(rejected['body'], self.old)
        self.assertEqual(rejected['proposal_input'], 'old_values')
    def test_insufficient_budget_never_commits_a_partial_structure(self):
        model = ConditionalEditor(self.tokenizer, self.target, True)
        result = self.run_sample(model, max_calls=6)
        self.assertEqual(result['body'], self.old)
        self.assertEqual(result['forward_calls'], 1)
    def test_untrained_action_is_not_a_valid_forced_diagnostic(self):
        with self.assertRaisesRegex(ValueError, 'absent from training'):
            self.run_sample(ConditionalEditor(self.tokenizer, self.target, True), force_mode='local_xyz')
    def test_alias_only_change_does_not_receive_edit_credit(self):
        self.old[8] = self.tokenizer.vocab['<X_100>']
        target = self.old.copy()
        target[8] = self.tokenizer.vocab['<X_000>']
        result = self.run_sample(ConditionalEditor(self.tokenizer, target, True))
        self.assertEqual(result['body'], self.old)
        self.assertEqual(result['changed_numeric_tokens'], 0)

    def test_batched_structures_preserve_per_request_rng_and_conditional_paths(self):
        tokenizer = self.tokenizer
        class BatchedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.forward_calls = 0
            def forward(self, input_ids, attention_mask=None, edit_context=None):
                self.forward_calls += 1
                b,length = input_ids.shape
                logits = torch.full((b,length,len(tokenizer.vocab)), -100.)
                for row in range(b):
                    prefix = int(edit_context.prompt_lengths[row])
                    n = int(edit_context.num_sites[row])
                    body = edit_context.old_token_ids[row,prefix:prefix+7+4*n].tolist()
                    body[1] = tokenizer.vocab['<LA_042>']
                    for pos, token in enumerate(body):
                        logits[row,prefix+pos,token] = 100.
                    # A real random draw per site tests independent RNG streams.
                    logits[row,prefix+8,tokenizer.vocab['<X_050>']] = 100.
                return EditOutput(logits, torch.tensor([[-10.,-10.,-10.,10.]]).repeat(b,1),
                                  torch.zeros(b,20), torch.zeros(b,4), torch.full((b,4),10.))
        small, _, _ = quantize_arrays({'lengths':[4.]*3,'angles':[90.]*3,'species':['Na'],
                                      'frac_coords':[[0.,0.,0.]]}, tokenizer.vocab)
        requests = [{'prompt':'fixture','body':body,'num_sites':n,'tasks':tasks,'seed':seed}
                    for body,n,tasks,seed in ((self.old,2,('G',),2),(small,1,('S',),3),
                                              (self.old,2,('G','S'),4))]
        expected = [edit_structure(BatchedModel(), tokenizer, **request,
                                    allowed_modes={'G':[0,3],'S':[3]}) for request in requests]
        model = BatchedModel()
        actual = edit_structures(model, tokenizer, requests, allowed_modes={'G':[0,3],'S':[3]}, batch_size=2)
        self.assertEqual(actual['results'], expected)
        self.assertEqual(actual['forward_rows'], sum(row['forward_calls'] for row in expected))
        self.assertEqual(actual['forward_batches'], model.forward_calls)
        self.assertLess(actual['forward_batches'], actual['forward_rows'])


if __name__ == '__main__':
    unittest.main()
