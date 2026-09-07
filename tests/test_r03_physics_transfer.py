"""Tiny CPU contracts; these fixtures are not materials or physical labels."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from crystal_dlm.fixed_slot import MASK_TOKEN_ID, build_special_tokens
from crystal_dlm.r03_physics_transfer import (
    REPAIR_SUPPORT_PROTOCOL, TERMINAL_PROTOCOL, TERMINAL_VERIFICATION_PROTOCOL,
    TransferContractError, adapt_round_rows, build_repair_constraints,
    build_repair_prompt, canonicalize_body_aliases, geometry_support_report,
    map_native_body, repair_scalar_example, sample_transfer_example,
    select_and_normalize_records, supported_reference_kl, supported_scalar_logits,
    weighted_scalar_ce,
)
from scripts.train_r03_physics_transfer import set_repair_lora_trainable


class TinyTokenizer:
    def __init__(self, offset=2):
        self.vocab = {token: index + offset for index, token in enumerate(build_special_tokens())}
        self.reverse = {index: token for token, index in self.vocab.items()}
        self.pad_token_id = 0

    def get_vocab(self):
        return self.vocab

    def convert_ids_to_tokens(self, values):
        return [self.reverse.get(int(v), "<unknown>") for v in values]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab[text]]} if text in self.vocab else {"input_ids": [1, 1]}


def make_body(tokenizer, *, species=("O", "Li"), coords=((0, 0, 0), (50, 50, 50)), length=40):
    tokens = [f"<N_{len(species):03d}>", *[f"<{a}_{length:03d}>" for a in ("LA", "LB", "LC")],
              *[f"<{a}_090>" for a in ("AA", "AB", "AG")]]
    for symbol, xyz in zip(species, coords):
        tokens.append(f"<E_{symbol}>")
        tokens.extend(f"<{a}_{v:03d}>" for a, v in zip("XYZ", xyz))
    return [tokenizer.vocab[t] for t in tokens], "".join(tokens)


def fixture_pool(tokenizer, *, role="k8"):
    round_index, count = (1, 8) if role == "k8" else (0, 4)
    ids, body = make_body(tokenizer)
    plan = {"N": 2, "elements": ["O", "Li"], "counts": [1, 1], "lattice_system": "cubic"}
    reference = f"fixture_{role}_collection_policy"
    paths, labels = [], []
    for i in range(count):
        tid = f"group0:{round_index}:{i}"
        initial = ids.copy()
        trace = {"initial_body": initial, "mask_id": MASK_TOKEN_ID, "events": [], "success": True}
        path = {"trajectory_id": tid, "group_id": "group0", "source_row_idx": 17, "source_split": "train",
                "checkpoint": reference, "collection_round": round_index, "candidate_index": i,
                "success": True, "body": body, "final_body_token_ids": ids.copy(), "trace": trace,
                "plan_state": copy.deepcopy(plan), "prompt": "original recorded compact prompt\n",
                "species_program": ["O", "Li"], "species_program_source": "fixture_pointer"}
        verified = i < 2
        label = {"trajectory_id": tid, "group_id": "group0", "source_row_idx": 17, "source_split": "train",
                 "verified": verified, "status": "verified" if verified else "unknown",
                 "raw_energy": -1.0 if verified else None, "terminal_energy": -1.2 if verified else None,
                 "optimizer_converged": verified, "terminal_consistency": {"status": "consistent"} if verified else None,
                 "endpoint_cache_key": hashlib.sha256(body.encode()).hexdigest()}
        paths.append(path)
        labels.append(label)
    teacher = {"summary": {"trainable_teacher": True, "diagnostic_only": False, "solver_status": "optimal",
                           "primal_residual": 0.0},
               "provenance": {"checkpoint": reference, "collection_round": round_index,
                              "candidates_per_condition": count, "verification_protocol": TERMINAL_VERIFICATION_PROTOCOL,
                              "terminal_protocol": TERMINAL_PROTOCOL},
               "groups": [{"group_id": "group0", "candidates": [dict(label, weight=(0.75, 0.25)[i] if i < 2 else 0.0)
                                                                      for i, label in enumerate(labels)]}]}
    return teacher, paths, labels


class RepairSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = TinyTokenizer()
        cls.constraints = build_repair_constraints(cls.tokenizer)
        cls.width = max(cls.tokenizer.vocab.values()) + 1
        cls.plan = {"N": 2, "elements": ["O", "Li"], "counts": [1, 1]}

    def test_shared_prompt_ignores_rich_and_energy_fields(self):
        extended = dict(self.plan, volume_per_atom_bin="invented", charge="invalid", e_hull=-999)
        expected = '{"N":2,"counts":[1,1],"elements":["Li","O"]}\ndynamic_crystal_body:\n'
        self.assertEqual(build_repair_prompt(extended), expected)
        self.assertEqual(build_repair_prompt(self.plan), expected)
        with self.assertRaises(TransferContractError):
            build_repair_prompt(dict(self.plan, counts=[1, 2]))

    def test_xyz_view_keeps_cell_and_other_site(self):
        ids, _ = make_body(self.tokenizer)
        for axis in range(3):
            example = repair_scalar_example(ids, 1, axis, plan_state=self.plan, constraints=self.constraints)
            self.assertEqual(example["input_body"][:12], ids[:12])
            self.assertEqual(example["input_body"][12:12 + axis], ids[12:12 + axis])
            self.assertEqual(example["input_body"][12 + axis:], [MASK_TOKEN_ID] * (3 - axis))
            self.assertEqual(example["target_token"], ids[12 + axis])

    def test_alias_logaddexp_has_gradients_on_both_physical_aliases(self):
        ids, _ = make_body(self.tokenizer)
        view = repair_scalar_example(ids, 0, 0, plan_state=self.plan, constraints=self.constraints)
        raw = torch.zeros(self.width, requires_grad=True)
        canonical, alias = self.constraints["coordinate_alias_token_ids"]["X"]
        with torch.no_grad():
            raw[canonical], raw[alias] = math_log(2), math_log(3)
        vector, report = supported_scalar_logits(raw, view["input_body"], 0, 2, 8, constraints=self.constraints)
        self.assertTrue(report["available"])
        self.assertAlmostEqual(float(vector[canonical].detach()), math_log(5), places=6)
        self.assertEqual(float(vector[alias].detach()), torch.finfo(vector.dtype).min)
        vector[canonical].backward()
        self.assertAlmostEqual(float(raw.grad[canonical]), 0.4, places=6)
        self.assertAlmostEqual(float(raw.grad[alias]), 0.6, places=6)

    def test_vector_and_full_canvas_support_are_identical(self):
        ids, _ = make_body(self.tokenizer)
        view = repair_scalar_example(ids, 0, 2, plan_state=self.plan, constraints=self.constraints)
        raw = torch.linspace(-1, 1, self.width)
        full = torch.zeros(1, 4 + len(ids), self.width)
        full[0, 4 + view["position"]] = raw
        a, ar = supported_scalar_logits(raw, view["input_body"], 0, 2, view["position"], constraints=self.constraints)
        b, br = supported_scalar_logits(full, view["input_body"], 4, 2, view["position"], constraints=self.constraints)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(ar, br)

    def test_no_legal_z_is_explicit_and_never_trainable(self):
        ids, _ = make_body(self.tokenizer, species=("O", "Li", "Na"),
                           coords=((0, 0, 0), (0, 0, 50), (0, 0, 25)), length=10)
        plan = {"N": 3, "elements": ["O", "Li", "Na"], "counts": [1, 1, 1]}
        view = repair_scalar_example(ids, 2, 2, plan_state=plan, constraints=self.constraints)
        vector, report = supported_scalar_logits(torch.zeros(self.width), view["input_body"], 0, 3,
                                                  view["position"], constraints=self.constraints)
        self.assertFalse(report["available"])
        self.assertTrue(report["no_legal_completion"])
        self.assertEqual(report["reason"], "no_legal_Z_completion")
        with self.assertRaises(TransferContractError):
            weighted_scalar_ce(vector, view["target_token"])

    def test_native_support_and_periodic_alias_identity(self):
        ids, _ = make_body(self.tokenizer, coords=((100, 0, 0), (50, 50, 50)))
        canonical = canonicalize_body_aliases(ids, constraints=self.constraints)
        self.assertNotEqual(ids[8], canonical[8])
        self.assertEqual(ids[:8], canonical[:8])
        self.assertTrue(geometry_support_report(ids, constraints=self.constraints)["supported"])
        self.assertTrue(geometry_support_report(canonical, constraints=self.constraints)["supported"])
        overlap, _ = make_body(self.tokenizer, coords=((100, 0, 0), (0, 0, 0)))
        self.assertFalse(geometry_support_report(overlap, constraints=self.constraints)["supported"])

    def test_support_protocol_cannot_be_weakened(self):
        wrong = dict(self.constraints, pbc_min_distance_A=0.1)
        with self.assertRaises(TransferContractError):
            geometry_support_report(make_body(self.tokenizer)[0], constraints=wrong)

    def test_weighted_ce_and_reference_kl_are_exact_on_legal_support(self):
        minimum = torch.finfo(torch.float32).min
        vector = torch.tensor([1.0, 2.0, minimum], requires_grad=True)
        loss = weighted_scalar_ce(vector, 0, weight=0.25)
        expected = 0.25 * (torch.logsumexp(vector[:2] / 0.7, 0) - vector[0] / 0.7)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertAlmostEqual(float(supported_reference_kl(vector, vector.detach()).detach()), 0.0, places=6)
        with self.assertRaises(TransferContractError):
            weighted_scalar_ce(vector, 2)


class TransferProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.source, self.b0 = TinyTokenizer(20), TinyTokenizer(2)
        self.constraints = build_repair_constraints(self.b0)
        self.teacher, self.paths, self.labels = fixture_pool(self.source)

    def adapt(self, role="k8", teacher=None, paths=None, labels=None):
        return adapt_round_rows(teacher or self.teacher, paths or self.paths, labels or self.labels,
                                role=role, source_tokenizer=self.source, b0_tokenizer=self.b0,
                                constraints=self.constraints, expected_conditions=1)

    def test_physics_weights_transfer_but_old_token_ids_do_not(self):
        rows, rejected, report = self.adapt()
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rejected), 6)
        self.assertTrue(report["source_reference_verified"])
        self.assertNotEqual(rows[0]["source_body_token_ids"], rows[0]["body_token_ids"])
        self.assertEqual(rows[0]["original_teacher_weight"], 0.75)
        self.assertEqual(rows[0]["source_body_tokens"], self.b0.convert_ids_to_tokens(rows[0]["body_token_ids"]))

    def test_evaluation_feedback_is_rejected_even_when_weight_zero(self):
        self.labels[-1]["source_split"] = "evaluation"
        self.teacher["groups"][0]["candidates"][-1]["source_split"] = "evaluation"
        with self.assertRaises(TransferContractError):
            self.adapt()

    def test_original_reference_check_is_not_disabled_for_b0_transfer(self):
        self.paths[0]["checkpoint"] = "B0_instead_of_original_collection_policy"
        with self.assertRaises(TransferContractError):
            self.adapt()

    def test_missing_occurrence_or_changed_teacher_label_is_rejected(self):
        with self.assertRaises(TransferContractError):
            self.adapt(paths=self.paths[:-1])
        self.teacher["groups"][0]["candidates"][0]["terminal_energy"] = -9
        with self.assertRaises(TransferContractError):
            self.adapt()

    def test_changed_native_cache_key_is_incompatible_not_relabelled(self):
        self.labels[0]["endpoint_cache_key"] = "0" * 64
        self.teacher["groups"][0]["candidates"][0]["endpoint_cache_key"] = "0" * 64
        rows, rejected, _ = self.adapt()
        self.assertEqual(len(rows), 1)
        self.assertTrue(any("cache key" in row["reason"] for row in rejected))

    def test_positive_unverified_weight_is_a_contract_error(self):
        self.labels[0]["optimizer_converged"] = False
        self.teacher["groups"][0]["candidates"][0]["optimizer_converged"] = False
        with self.assertRaises(TransferContractError):
            self.adapt()

    def test_unnormalized_old_teacher_mass_is_not_silently_repaired(self):
        self.teacher["groups"][0]["candidates"][0]["weight"] = 1.0
        with self.assertRaises(TransferContractError):
            self.adapt()

    def test_uniform_reference_uncertainty_group_keeps_its_flags_and_weights(self):
        self.teacher["summary"].update(uniform_reference_group_ids=["group0"],
                                       fixed_reference_groups=[{"group_id": "group0", "reason": "original uncertainty"}],
                                       credibility_treatment="explicit_uncertainty_reference_constraint")
        for row in self.teacher["groups"][0]["candidates"][:2]:
            row["weight"] = 0.5
        # An accepted uncertainty constraint must not be overridden by a new
        # absolute-energy cutoff or a re-solved ranking in this adapter.
        self.labels[0].update(raw_energy=-499.0, terminal_energy=-500.0)
        self.teacher["groups"][0]["candidates"][0].update(raw_energy=-499.0, terminal_energy=-500.0)
        rows, _, report = self.adapt()
        normalized, _, _ = select_and_normalize_records(rows)
        self.assertEqual([r["original_teacher_weight"] for r in normalized], [0.5, 0.5])
        self.assertTrue(all(r["uniform_reference_group"] for r in normalized))
        self.assertEqual(normalized[0]["original_uniform_reference_record"]["reason"], "original uncertainty")
        self.assertEqual(normalized[0]["raw_energy"], -499.0)
        self.assertTrue(report["uniform_reference_weights_preserved"])
        self.teacher["groups"][0]["candidates"][0]["weight"] = 0.75
        self.teacher["groups"][0]["candidates"][1]["weight"] = 0.25
        with self.assertRaises(TransferContractError):
            self.adapt()

    def test_native_structure_and_body_must_identify_same_geometry(self):
        changed_structure = {"lattice": {"matrix": [[5, 0, 0], [0, 5, 0], [0, 0, 5]]},
                             "sites": [{"species": [{"element": "O", "occu": 1}], "abc": [0, 0, 0]},
                                       {"species": [{"element": "Li", "occu": 1}], "abc": [0.5, 0.5, 0.5]}]}
        self.paths[0]["structure"] = changed_structure
        self.labels[0]["endpoint_cache_key"] = hashlib.sha256(json.dumps(changed_structure, sort_keys=True).encode()).hexdigest()
        with self.assertRaises(TransferContractError):
            map_native_body(self.paths[0], self.labels[0], self.source, self.b0, constraints=self.constraints)

    def test_k8_priority_preserves_distinct_old_condition_groups(self):
        k8, _, _ = self.adapt()
        teacher4, paths4, labels4 = fixture_pool(self.source, role="k4")
        k4, _, _ = self.adapt("k4", teacher4, paths4, labels4)
        combined, omitted, report = select_and_normalize_records(k8, k4)
        self.assertEqual(len(combined), 2)
        self.assertEqual(len(omitted), 2)
        self.assertEqual(report["exact_compositions"], 1)
        different = copy.deepcopy(k4)
        for row in different:
            row["condition_signature"] = "a_different_recorded_scientific_condition"
        combined, _, report = select_and_normalize_records(k8, different)
        self.assertEqual(report["source_conditions"], 2)
        self.assertAlmostEqual(sum(row["dataset_weight"] for row in combined), 1.0)
        self.assertAlmostEqual(sum(row["dataset_weight"] for row in combined if row["source_role"] == "k4"), 0.5)
        self.assertFalse(report["old_dual_objective_guarantee_transferred"])
        example = sample_transfer_example(combined, tokenizer=self.b0, constraints=self.constraints, rng=random.Random(4))
        self.assertGreater(example["weight"], 0)


class FrozenParameterTests(unittest.TestCase):
    def test_optimizer_updates_only_existing_lora_not_tables_or_base(self):
        class TinyB0Copy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(3, 2)
                self.head = torch.nn.Linear(2, 3, bias=False)
                self.base = torch.nn.Linear(2, 2, bias=False)
                self.attention = torch.nn.Module()
                self.attention.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 1, bias=False)})
                self.attention.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 2, bias=False)})

            def get_input_embeddings(self):
                return self.embedding

            def get_output_embeddings(self):
                return self.head

        model = TinyB0Copy()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        selected = set_repair_lora_trainable(model)
        self.assertEqual(len(selected), 2)
        optimizer = torch.optim.AdamW([p for _, p in selected], lr=1e-6, weight_decay=0)
        loss = sum(p.square().sum() for _, p in selected)
        loss.backward()
        optimizer.step()
        names = {name for name, _ in selected}
        for name, value in model.named_parameters():
            self.assertEqual(value.requires_grad, name in names)
            if name not in names:
                self.assertTrue(torch.equal(value, before[name]))
        self.assertTrue(any(not torch.equal(p, before[n]) for n, p in selected))


def math_log(value):
    import math
    return math.log(value)


if __name__ == "__main__":
    unittest.main()
