import math
import unittest
from types import SimpleNamespace

import torch

from crystal_dlm.programmed_path_training import sample_path_decisions, minibatch_path_loss, PathLogProbability, join_terminal_labels, shape_matched_batches, training_decision_budget
from crystal_dlm.programmed_path_data import trace_terminal_body, training_candidates_per_condition
from crystal_dlm.programmed_path_runtime import process_path_logits, process_scalar_path_logits
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints
from test_state_programmed_runtime import TinyTokenizer, TinyBase, body, constraints, program
from crystal_dlm.periodic_state_conditioning import PeriodicStateConfig
from crystal_dlm.state_conditioned_model import StateConditionedDLM, context_from_programs, set_state_lora_trainable


def example_path(counts):
    events = []
    for phase, count in zip(("construct", "cooperative", "closure"), counts):
        events.append({"op": "begin", "phase": phase, "kind": "construct", "positions": []})
        events.extend({"op": "draw", "position": 1, "token": i + 1, "log_probability": -1.} for i in range(count))
        events.append({"op": "end"})
    return {"trajectory_id": "p:0", "trace": {"initial_body": [0, 0], "mask_id": 9, "temperature": .7, "events": events}}


class PathTrainingTest(unittest.TestCase):
    def test_dense_refresh_sampling_preserves_ht_and_the_registered_budget(self):
        self.assertEqual(training_decision_budget(0, 972), 6)
        for paths, expected in ((2000, 24), (4096, 12), (8192, 6)):
            budget = training_decision_budget(1, paths)
            self.assertEqual(budget, expected)
            self.assertLessEqual(2 * paths * budget, 98304)
        for budget in (6, 12, 17, 24):
            for counts in ((9, 18, 27), (2, 0, 3), (1, 1, 28)):
                states = sample_path_decisions(example_path(counts), seed=19, pass_index=2, budget=budget)
                self.assertEqual(len(states), min(budget, sum(counts)))
                self.assertAlmostEqual(sum(1 / s["inclusion_probability"] for s in states), sum(counts))
        with self.assertRaises(ValueError):
            training_decision_budget(2, 100)

    def test_k8_refresh_keeps_all_occurrences_and_rejects_incomplete_k4_pool(self):
        self.assertEqual(training_candidates_per_condition(0), 4)
        self.assertEqual(training_candidates_per_condition(1), 8)
        with self.assertRaises(ValueError):
            training_candidates_per_condition(2)
        paths, labels = [], []
        for candidate in range(8):
            path = example_path((3, 3, 3))
            path.update(trajectory_id=f"group:1:{candidate}", checkpoint="formal-round0", collection_round=1,
                        group_id="group", source_row_idx=7, source_split="train", success=True,
                        candidate_index=candidate, final_body_token_ids=trace_terminal_body(path["trace"]))
            paths.append(path)
            labels.append({"trajectory_id": path["trajectory_id"], "group_id": "group", "source_row_idx": 7,
                           "source_split": "train", "verified": candidate < 3})
        groups = join_terminal_labels(paths, labels, expected_conditions=1, candidates=8)
        self.assertEqual([c["candidate_index"] for c in groups[0]["candidates"]], list(range(8)))
        self.assertEqual(sum(c["verified"] for c in groups[0]["candidates"]), 3)
        with self.assertRaises(ValueError):
            join_terminal_labels(paths[:4], labels[:4], expected_conditions=1, candidates=8)

    def test_shape_buckets_cover_once_and_add_only_zero_mass_rows(self):
        rows = [{"id": i, "prompt_token_ids": [0] * (3 + i % 3), "input_body": [0] * 15,
                 "weight": 1., "inclusion_probability": .5} for i in range(19)]
        batches = shape_matched_batches(rows, global_batch=16, seed=23, pass_index=0)
        self.assertEqual(len(batches), 3)
        for batch in batches:
            self.assertEqual(len(batch), 16)
            self.assertEqual(len({len(e["prompt_token_ids"]) + len(e["input_body"]) for e in batch}), 1)
        retained = [e["id"] for batch in batches for e in batch if e["weight"] > 0]
        self.assertEqual(sorted(retained), list(range(19)))

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

    def test_scalar_projection_matches_dense_forward_and_gradient(self):
        tok = TinyTokenizer()
        c = constraints(tok)
        source = body(tok)
        allowed = torch.zeros(len(source), len(tok.vocab), dtype=torch.bool)
        for pos, ids in enumerate(exact_dynamic_schema_constraints(tok, 2)):
            allowed[pos, ids] = True
        for dtype in (torch.float32, torch.bfloat16):
            for position in (1, 6, 8, 9, 10):
                x = torch.tensor([[0] + source])
                x[0, position + 1] = tok.mask_id
                dense_input = torch.randn(1, x.shape[1], len(tok.vocab), dtype=dtype, requires_grad=True)
                scalar_input = dense_input.detach().clone().requires_grad_(True)
                dense, bad = process_path_logits(dense_input, x, prompt_length=1, gen_length=len(source),
                    allowed=allowed, grammar=None, constraints=c, positions={0: position}, mask_id=tok.mask_id)
                scalar, scalar_bad = process_scalar_path_logits(scalar_input, x, prompt_length=1,
                    gen_length=len(source), allowed=allowed, constraints=c, position=position, mask_id=tok.mask_id)
                self.assertEqual(bad, scalar_bad)
                self.assertTrue(torch.equal(dense[0, position + 1], scalar))
                target = source[position]
                dense_loss = -torch.log_softmax(dense[0, position + 1].float() / .7, -1)[target]
                scalar_loss = -torch.log_softmax(scalar.float() / .7, -1)[target]
                dense_loss.backward()
                scalar_loss.backward()
                self.assertTrue(torch.equal(dense_input.grad, scalar_input.grad))

    def test_active_projection_keeps_lora_and_conditioner_gradients(self):
        tok = TinyTokenizer()
        base = TinyBase(len(tok.vocab), recompute=True)
        model = StateConditionedDLM(base, tok, PeriodicStateConfig(16, width=12, radial_basis_count=4))
        set_state_lora_trainable(model)
        old = torch.tensor([[0] + body(tok)])
        x = old.clone()
        x[:, 9:12] = tok.mask_id
        context = context_from_programs(old, prompt_length=1, num_sites=2, programs=[program()], active_positions={0: [8, 9, 10]})
        logits = model(x, attention_mask=torch.ones_like(x), geometry_context=context).logits
        batch = {"input_ids": x, "geometry_context": context,
                 "examples": [{"num_atoms": 2, "position": 8, "target_token": tok.vocab["<X_000>"], "temperature": .7}]}
        loss = -PathLogProbability(tok, constraints(tok))(logits, batch).sum()
        loss.backward()
        self.assertGreater(float(model.state_conditioner.cell_projection.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(base.lora_B["default"].weight.grad.abs().sum()), 0.)
        self.assertIsNone(base.embedding.weight.grad)
        self.assertIsNone(base.output.weight.grad)


if __name__ == "__main__":
    unittest.main()
