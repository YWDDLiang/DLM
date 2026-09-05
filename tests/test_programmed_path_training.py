import math
import unittest
from types import SimpleNamespace

import torch

from crystal_dlm.programmed_path_training import sample_path_decisions, minibatch_path_loss, PathLogProbability, join_terminal_labels
from crystal_dlm.programmed_path_data import trace_terminal_body
from crystal_dlm.programmed_path_runtime import process_path_logits
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints
from test_state_programmed_runtime import TinyTokenizer, body, constraints


def example_path(counts):
    events = []
    for phase, count in zip(("construct", "cooperative", "closure"), counts):
        events.append({"op": "begin", "phase": phase, "kind": "construct", "positions": []})
        events.extend({"op": "draw", "position": 1, "token": i + 1, "log_probability": -1.} for i in range(count))
        events.append({"op": "end"})
    return {"trajectory_id": "p:0", "trace": {"initial_body": [0, 0], "mask_id": 9, "temperature": .7, "events": events}}


class PathTrainingTest(unittest.TestCase):
    def test_label_join_preserves_occurrences_and_excludes_heldout_or_missing_rows(self):
        paths, labels = [], []
        for group in range(2):
            for candidate in range(4):
                path = example_path((9, 18, 27))
                path.update(trajectory_id=f"{group}:{candidate}", checkpoint="warmup", collection_round=0,
                            group_id=str(group), source_row_idx=group, source_split="train", success=True,
                            candidate_index=candidate, final_body_token_ids=trace_terminal_body(path["trace"]))
                paths.append(path)
                labels.append({"trajectory_id": path["trajectory_id"], "group_id": str(group),
                               "source_row_idx": group, "source_split": "train", "verified": candidate < 2,
                               "raw_energy": 1. if candidate < 2 else None,
                               "terminal_energy": 0. if candidate < 2 else None})
        groups = join_terminal_labels(paths, labels, expected_conditions=2)
        self.assertEqual([len(g["candidates"]) for g in groups], [4, 4])
        with self.assertRaises(ValueError):
            join_terminal_labels(paths, labels[:-1], expected_conditions=2)
        with self.assertRaises(ValueError):
            join_terminal_labels(paths, labels + labels[:1], expected_conditions=2)
        with self.assertRaises(ValueError):
            join_terminal_labels([dict(paths[0], source_split="test"), *paths[1:]], labels, expected_conditions=2)

    def test_stratified_ht_recovers_phase_totals_and_empty_redistribution(self):
        for counts in ((9, 18, 27), (9, 0, 21), (1, 1, 14), (1, 0, 1)):
            samples = sample_path_decisions(example_path(counts), seed=1, pass_index=0)
            self.assertEqual(len(samples), min(6, sum(counts)))
            self.assertAlmostEqual(sum(1 / s["inclusion_probability"] for s in samples), sum(counts))
            self.assertEqual({s["phase"] for s in samples}, {p for p, n in zip(("construct", "cooperative", "closure"), counts) if n})

    def test_different_passes_change_decisions_reproducibly(self):
        path = example_path((9, 18, 27))
        a = sample_path_decisions(path, seed=2, pass_index=0)
        b = sample_path_decisions(path, seed=2, pass_index=1)
        self.assertEqual(a, sample_path_decisions(path, seed=2, pass_index=0))
        self.assertNotEqual([s["decision_index"] for s in a], [s["decision_index"] for s in b])

    def test_minibatch_scale_estimates_condition_objective_not_scalar_mean(self):
        examples = [{"weight": .25, "inclusion_probability": .2}, {"weight": .75, "inclusion_probability": .5},
                    {"weight": 1., "inclusion_probability": .25}, {"weight": 0., "inclusion_probability": 0.}]
        logp = torch.tensor([-1., -2., -3., -torch.inf], requires_grad=True)
        complete = minibatch_path_loss(logp, examples, dataset_size=4, validated_groups=2)
        first = minibatch_path_loss(logp[:2], examples[:2], dataset_size=4, validated_groups=2)
        last = minibatch_path_loss(logp[2:], examples[2:], dataset_size=4, validated_groups=2)
        self.assertAlmostEqual(float(complete.detach()), (1.25 + 3 + 12) / 2)
        self.assertTrue(torch.allclose((first + last) / 2, complete))
        complete.backward()
        self.assertEqual(float(logp.grad[-1]), 0.)

    def test_runtime_probability_and_alias_gradient_match_full_legal_support(self):
        tok = TinyTokenizer()
        c = constraints(tok)
        source = body(tok)
        x = torch.tensor([[0] + source])
        x[:, 9:12] = tok.mask_id
        raw = torch.randn(1, x.shape[1], len(tok.vocab), requires_grad=True)
        allowed = torch.zeros(len(source), len(tok.vocab), dtype=torch.bool)
        for pos, ids in enumerate(exact_dynamic_schema_constraints(tok, 2)):
            allowed[pos, ids] = True
        result, bad = process_path_logits(raw, x, prompt_length=1, gen_length=len(source), allowed=allowed,
            grammar=None, constraints=c, positions={0: 8}, mask_id=tok.mask_id)
        self.assertFalse(bad)
        canonical, alias = tok.vocab["<X_000>"], tok.vocab["<X_100>"]
        batch = {"input_ids": x, "geometry_context": SimpleNamespace(prompt_lengths=torch.tensor([1])),
                 "examples": [{"num_atoms": 2, "position": 8, "target_token": canonical, "temperature": .7}]}
        actual = PathLogProbability(tok, c)(raw, batch)[0]
        reference = torch.log_softmax(result[0, 9].double() / .7, -1)[canonical]
        self.assertLess(abs(float(actual.detach()) - float(reference.detach())), 2e-6)
        (-actual).backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertNotEqual(float(raw.grad[0, 9, canonical]), 0.)
        self.assertNotEqual(float(raw.grad[0, 9, alias]), 0.)


if __name__ == "__main__":
    unittest.main()
