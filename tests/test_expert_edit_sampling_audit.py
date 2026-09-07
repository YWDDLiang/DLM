"""Adversarial request scheduling and snapshot isolation for batched editing."""
import unittest

import torch
from torch import nn

from crystal_dlm.expert_edit import EditOutput, edit_structure, edit_structures
from crystal_dlm.expert_edit_data import numeric_positions, quantize_arrays
from crystal_dlm.fixed_slot import MASK_TOKEN_ID, build_special_tokens


class AuditTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.vocab = {token: index for index, token in enumerate(build_special_tokens())}

    def get_vocab(self):
        return self.vocab

    def __call__(self, text, **kwargs):
        tag, length = map(int, text.split("/"))
        return {"input_ids": [tag] * length}


class StatefulBatchEditor(nn.Module):
    """Policies differ by request tag, never by the request's batch slot."""

    def __init__(self, tokenizer):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.tokenizer = tokenizer
        self.forward_calls = 0
        self.observed = {}

    def forward(self, input_ids, attention_mask=None, edit_context=None):
        self.forward_calls += 1
        batch, length = input_ids.shape
        logits = torch.full((batch, length, len(self.tokenizer.vocab)), -100.0)
        modes = torch.full((batch, 4), -10.0)
        sites = torch.arange(20, dtype=torch.float32)[None].repeat(batch, 1)
        counts = torch.tensor([[10.0, -10.0, -10.0, -10.0]]).repeat(batch, 1)
        quality = torch.full((batch, 4), 10.0)
        for row in range(batch):
            tag = int(input_ids[row, 0])
            prefix = int(edit_context.prompt_lengths[row])
            count = int(edit_context.num_sites[row])
            stop = prefix + 7 + 4 * count
            old = edit_context.old_token_ids[row, prefix:stop].tolist()
            current = input_ids[row, prefix:stop].tolist()
            active = edit_context.active_token_mask[row, prefix:stop].tolist()
            task = int(edit_context.task_ids[row])
            stage = "inspect" if not any(active) else "fill" if MASK_TOKEN_ID in current else "judge"
            self.observed.setdefault(tag, []).append({
                "old": old, "current": current, "active": active,
                "task": task, "stage": stage,
                "remaining": float(edit_context.remaining_steps[row]),
                "reveal": float(edit_context.reveal_fraction[row]),
            })
            modes[row, 0 if tag == 0 else 2 if tag == 2 else 3] = 10.0
            if tag == 1 and task == 1 and stage == "inspect":
                quality[row, 0] = -10.0  # S admission rejects this request only.
            if tag == 2 and task == 0 and stage == "judge":
                quality[row, 3] = -10.0  # Reject G, then let S inspect the old state.
            target = old.copy()
            target[1] = self.tokenizer.vocab["<LA_042>" if task == 0 else "<LA_044>"]
            for site in range(count):
                target[8 + 4 * site] = self.tokenizer.vocab["<X_065>"]
            for position, token in enumerate(target):
                logits[row, prefix + position, token] = 100.0
            # Consume a genuine request-local random draw during each XYZ refill.
            for site in range(count):
                logits[row, prefix + 8 + 4 * site, self.tokenizer.vocab["<X_075>"]] = 100.0
        return EditOutput(logits, modes, sites, counts, quality)


class BatchedSamplerAuditTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AuditTokenizer()
        self.allowed = {"G": [0, 2, 3], "S": [0, 2, 3]}

    def request(self, tag, count, tasks, seed, prefix=1):
        body, _, _ = quantize_arrays({
            "lengths": [4.0] * 3, "angles": [90.0] * 3,
            "species": ["Na" if i % 2 == 0 else "Cl" for i in range(count)],
            "frac_coords": [[(i + 1) / (count + 1), 0.2, 0.3] for i in range(count)],
        }, self.tokenizer.vocab)
        return {"prompt": f"{tag}/{prefix}", "body": body, "num_sites": count,
                "tasks": tasks, "seed": seed}

    def test_interleaved_stages_release_slots_and_preserve_snapshots_and_request_rng(self):
        requests = [
            self.request(0, 1, ("G",), 12),
            self.request(1, 2, ("G", "S"), 13, prefix=4),
            self.request(2, 1, ("G", "S"), 14, prefix=2),
            self.request(3, 3, ("G",), 15, prefix=3),
            self.request(4, 2, (), 16),
        ]
        expected = [
            edit_structure(StatefulBatchEditor(self.tokenizer), self.tokenizer,
                           **request, allowed_modes=self.allowed)
            for request in requests
        ]
        model = StatefulBatchEditor(self.tokenizer)
        sampled = edit_structures(model, self.tokenizer, requests,
                                  allowed_modes=self.allowed, batch_size=2)
        self.assertEqual(sampled["results"], expected)
        reversed_sample = edit_structures(
            StatefulBatchEditor(self.tokenizer), self.tokenizer, list(reversed(requests)),
            allowed_modes=self.allowed, batch_size=3,
        )
        self.assertEqual(list(reversed(reversed_sample["results"])), expected)
        self.assertEqual(sampled["forward_batches"], model.forward_calls)
        self.assertEqual(sampled["forward_rows"], sum(map(len, model.observed.values())))
        self.assertEqual(sampled["forward_rows"], sum(row["forward_calls"] for row in expected))
        self.assertNotIn(4, model.observed)
        self.assertEqual(expected[-1]["forward_calls"], 0)
        for request, result in zip(requests, sampled["results"]):
            self.assertEqual(sum(event["calls"] for event in result["trace"]), result["forward_calls"])
            fixed = set(range(len(request["body"]))) - set(numeric_positions(request["num_sites"]))
            self.assertTrue(all(result["body"][position] == request["body"][position] for position in fixed))
            self.assertNotIn(MASK_TOKEN_ID, result["body"])
        rejected_g = expected[2]["trace"][0]
        accepted_s = expected[2]["trace"][1]
        self.assertFalse(rejected_g["accepted"])
        self.assertTrue(accepted_s["accepted"])
        self.assertNotEqual(rejected_g["proposal_body"], rejected_g["old_body"])
        self.assertEqual(accepted_s["old_body"], rejected_g["old_body"])
        s_inspection = next(row for row in model.observed[2] if row["task"] == 1 and row["stage"] == "inspect")
        self.assertEqual(s_inspection["current"], rejected_g["old_body"])
        self.assertEqual(s_inspection["old"], rejected_g["old_body"])
        self.assertEqual(expected[1]["trace"][1]["reason"], "learned_S_admission_reject")
        self.assertEqual(expected[1]["body"], expected[1]["trace"][0]["proposal_body"])
        for rows in model.observed.values():
            for row in rows:
                if row["stage"] == "inspect":
                    self.assertFalse(any(row["active"]))
                    self.assertEqual(row["old"], row["current"])
                else:
                    self.assertTrue(all(
                        before == current for before, current, active
                        in zip(row["old"], row["current"], row["active"]) if not active
                    ))
                    self.assertEqual(MASK_TOKEN_ID in row["current"], row["stage"] == "fill")

    def test_exact_budget_completes_and_one_call_less_never_exposes_partial_geometry(self):
        request = self.request(3, 1, ("G",), 71, prefix=3)
        exact_model = StatefulBatchEditor(self.tokenizer)
        exact = edit_structure(exact_model, self.tokenizer, **request,
                               allowed_modes=self.allowed, max_calls=11)
        self.assertEqual(exact["forward_calls"], 11)  # 1 inspect + 9 scalar fills + 1 judge.
        self.assertEqual(exact_model.forward_calls, 11)
        self.assertTrue(exact["trace"][0]["accepted"])
        short_model = StatefulBatchEditor(self.tokenizer)
        short = edit_structure(short_model, self.tokenizer, **request,
                               allowed_modes=self.allowed, max_calls=10)
        self.assertEqual(short["forward_calls"], 1)
        self.assertEqual(short["body"], request["body"])
        self.assertEqual(short["trace"][0]["reason"], "insufficient_complete_proposal_budget")
        self.assertTrue(all(row["stage"] == "inspect" for row in short_model.observed[3]))


if __name__ == "__main__":
    unittest.main()
