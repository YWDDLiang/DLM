import copy
import unittest

from crystal_dlm.r03_control_pointer import (
    FormulaFeatureSpec,
    compile_revision_program,
    formula_replay_text,
    plan_composition,
)
from scripts.train_r03_control_pointer import prepare_rows, validate_approved_feature_spec
from scripts.export_r03_control_programs import attach_program


SPEC = FormulaFeatureSpec("h1_rich_plan_v1", False)


def teacher(index, elements, counts, split="train"):
    return {
        "source_split": split, "source_row_idx": index,
        "plan_state": {"N": sum(counts), "elements": elements, "counts": counts,
                       "volume": "teacher_only", "spacegroup": "teacher_only"},
        "soft_targets": {"unavailable_at_export": 123},
    }


def cache(index, atomic, counts, order, split="train"):
    return {"schema": "spad_species_pointer_row_v1", "source_split": split,
            "source_row_idx": index, "canonical_atomic_numbers": atomic,
            "canonical_element_counts": counts, "contact_tree_order_indices": order,
            "outcomes_read": False}


class FormulaReplayContractTest(unittest.TestCase):
    def test_canonical_replay_never_mutates_original_plan_or_reads_rich_fields(self):
        plan = {"N": 3, "elements": ["Zr", "O"], "counts": [1, 2],
                "formula": "ZrO2", "volume": "secret_teacher_volume"}
        before = copy.deepcopy(plan)
        text = formula_replay_text(None, plan, spec=SPEC, exact_prompt_text="NATIVE_PROMPT\n\n")
        self.assertEqual(text, "NATIVE_PROMPT\n\nformula: O2Zr")
        self.assertFalse(text.endswith("\n"))
        self.assertNotIn("secret_teacher", text)
        self.assertEqual(plan, before)

    def test_approved_cli_rejects_wrong_prompt_and_sample_id(self):
        validate_approved_feature_spec(SPEC)
        with self.assertRaisesRegex(ValueError, "frozen R03"):
            validate_approved_feature_spec(FormulaFeatureSpec("h1_rich_plan_formula_prefill_v1", False))
        with self.assertRaisesRegex(ValueError, "frozen R03"):
            validate_approved_feature_spec(FormulaFeatureSpec("h1_rich_plan_v1", True))

    def test_count_conservation_is_checked(self):
        with self.assertRaisesRegex(ValueError, "conservation"):
            plan_composition({"N": 4, "elements": ["Zr", "O"], "counts": [1, 2]})


class ContactSourceAlignmentTest(unittest.TestCase):
    def test_full_teacher_rows_retained_beyond_typed_cache(self):
        originals = [teacher(2, ["Zr", "O"], [1, 2]), teacher(19, ["Na", "Cl"], [1, 1])]
        cached = [cache(2, [8, 40], [2, 1], [1, 0])]
        rows, report = prepare_rows(
            originals, cached, split="train",
            missing_contacts={19: ((("Na", 1), ("Cl", 1)), ("Cl", "Na"))},
        )
        self.assertEqual([row["source_row_idx"] for row in rows], [2, 19])
        self.assertEqual(report["filtered_rows"], 0)
        self.assertEqual(report["existing_contact_labels"], 1)
        self.assertEqual(report["missing_contact_labels_computed"], 1)
        self.assertEqual(rows[0]["contact_tree_order_symbols"], ["Zr", "O"])
        self.assertEqual(rows[1]["contact_tree_order_indices"], [1, 0])
        self.assertNotIn("soft_targets", rows[0])
        self.assertEqual(set(rows[0]["plan_state"]), {"N", "elements", "counts"})

    def test_missing_contact_does_not_silently_filter_training_row(self):
        with self.assertRaisesRegex(ValueError, "missing contact"):
            prepare_rows([teacher(19, ["Na", "Cl"], [1, 1])], [], split="train")

    def test_mismatched_split_or_source_composition_rejected(self):
        with self.assertRaisesRegex(ValueError, "split"):
            prepare_rows([teacher(2, ["O"], [2], split="val")], [], split="train")
        with self.assertRaisesRegex(ValueError, "composition mismatch"):
            prepare_rows([teacher(2, ["O"], [2])], [cache(2, [8], [3], [0])], split="train")

    def test_duplicate_source_ids_rejected(self):
        row = teacher(2, ["O"], [2])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_rows([row, row], [], split="train")


class RevisionExportContractTest(unittest.TestCase):
    def test_exports_metadata_only_and_reverses_first_two_anchors(self):
        row = {"sample_idx": 7, "plan_state": {"N": 5, "elements": ["Na", "Cl", "O"],
                 "counts": [1, 1, 3], "formula": "NaClO3", "rich": {"charge": "kept"}},
               "prompt": "original body prompt", "raw_model_text": "original Planner tokens"}
        before = copy.deepcopy(row)
        # Canonical candidates are O, Na, Cl; choose Na, Cl, O.
        output = attach_program(row, [1, 2, 0], pointer_sha256="checkpoint")
        self.assertEqual(row, before)
        self.assertTrue(all(output[key] == value for key, value in before.items()))
        self.assertEqual(output["species_program"], ["Na", "Cl", "O"])
        control = output["r03_control"]
        self.assertEqual(control["revision_species"], ["Cl", "Na"])
        self.assertEqual(control["revision_slots"], [1, 0])
        self.assertEqual(control["slot_mapping"], "original_plan_element_order_counts")
        self.assertEqual(control["scope"], "post_construction_only")
        self.assertFalse(control["sampled_generation_modified"])

    def test_single_species_gets_one_anchor_and_invalid_permutation_fails(self):
        plan = {"N": 2, "elements": ["Si"], "counts": [2]}
        program = compile_revision_program(plan, ["Si"])
        self.assertEqual(program["revision_slots"], [0])
        with self.assertRaisesRegex(ValueError, "permutation"):
            attach_program({"plan_state": plan}, [0, 0], pointer_sha256="x")

    def test_existing_program_fields_are_never_overwritten(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            attach_program({"plan_state": {"N": 1, "elements": ["Si"], "counts": [1]},
                            "species_program": ["Si"]}, [0], pointer_sha256="x")


try:
    import torch
    from types import SimpleNamespace
    from crystal_dlm.r03_control_pointer import (
        R03ControlPointer, R03ControlPointerConfig, formula_hidden_batch, pointer_batch, species_pointer_loss,
    )
    from scripts.train_r03_control_pointer import evaluate_pointer, extract_features, train_one_pass
    from scripts.export_r03_control_programs import export_rows
except ModuleNotFoundError:
    torch = None


if torch is not None:
    class FixtureTokenizer:
        """CPU fixture, not a tokenizer/model downloaded from a real Planner."""

        def __init__(self):
            self.texts = []

        def __call__(self, texts, *, padding, add_special_tokens, return_tensors):
            self.texts.extend(texts)
            self_encoded = [[ord(char) % 31 + 1 for char in text] for text in texts]
            width = max(map(len, self_encoded))
            ids = torch.zeros(len(texts), width, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for index, sequence in enumerate(self_encoded):
                ids[index, -len(sequence):] = torch.tensor(sequence)
                mask[index, -len(sequence):] = 1
            return {"input_ids": ids, "attention_mask": mask}


    class FixtureDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

        def forward(self, *, input_ids, attention_mask, position_ids, **kwargs):
            hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 8) * self.marker
            self.position_ids = position_ids
            return SimpleNamespace(last_hidden_state=hidden)


    class FixtureP0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FixtureDecoder()


@unittest.skipIf(torch is None, "torch unavailable")
class PointerCPUFixtureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.head = R03ControlPointer(R03ControlPointerConfig(llama_hidden_size=8, pointer_size=8))
        self.rows, _ = prepare_rows(
            [teacher(2, ["O", "Zr"], [2, 1]), teacher(19, ["Si"], [2])],
            [cache(2, [8, 40], [2, 1], [1, 0]), cache(19, [14], [2], [0])], split="train",
        )

    def test_no_soft_parameters_and_finite_permutation_training(self):
        self.assertFalse(any("soft_embeddings" in key for key in self.head.state_dict()))
        batch = pointer_batch(self.rows)
        hidden = torch.randn(2, 8)
        logits = self.head.permutation_logits(hidden, batch["atomic_numbers"], batch["counts"],
                                             batch["valid_mask"], teacher_order=batch["teacher_order"])
        loss = species_pointer_loss(logits, batch["teacher_order"], batch["valid_mask"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.head.parameters() if p.grad is not None))
        order = self.head.decode(hidden, batch["atomic_numbers"], batch["counts"], batch["valid_mask"])
        self.assertEqual(sorted(order[0, :2].tolist()), [0, 1])
        self.assertEqual(order[1, 0].item(), 0)

    def test_duplicate_candidates_and_out_of_range_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self.head.decode(torch.zeros(1, 8), torch.tensor([[8, 8]]), torch.tensor([[1, 1]]),
                             torch.tensor([[True, True]]))
        batch = pointer_batch(self.rows)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.head.permutation_logits(torch.zeros(2, 8), batch["atomic_numbers"], batch["counts"],
                                         batch["valid_mask"], teacher_order=torch.tensor([[2, 0], [0, 0]]))

    def test_one_pass_visits_every_source_and_updates_only_head(self):
        frozen_features = torch.randn(2, 8)
        original = frozen_features.clone()
        before = {key: value.clone() for key, value in self.head.state_dict().items()}
        result = train_one_pass(self.head, frozen_features, self.rows, batch_size=2, lr=1e-3, seed=86017, device="cpu")
        self.assertEqual(result["rows_seen"], 2)
        self.assertEqual(result["updates"], 1)
        self.assertTrue(torch.equal(frozen_features, original))
        self.assertTrue(any(not torch.equal(before[key], value) for key, value in self.head.state_dict().items()))
        metrics = evaluate_pointer(self.head, frozen_features, self.rows, batch_size=2, device="cpu")
        self.assertEqual(metrics["rows"], 2)
        self.assertGreaterEqual(metrics["root_accuracy"], 0)

    def test_replay_features_use_last_attended_token_and_frozen_fixture(self):
        model = FixtureP0()
        tokenizer = FixtureTokenizer()
        hidden = formula_hidden_batch(model, tokenizer, self.rows, spec=SPEC, exact_prompt_text="NATIVE\n")
        expected = torch.tensor([ord("r") % 31 + 1, ord("2") % 31 + 1]).float()
        self.assertTrue(torch.equal(hidden[:, 0], expected))
        self.assertFalse(hidden.requires_grad)
        self.assertTrue(all("teacher_only" not in text for text in tokenizer.texts))
        model.model.marker.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "completely frozen"):
            formula_hidden_batch(model, tokenizer, self.rows, spec=SPEC, exact_prompt_text="NATIVE\n")

    def test_identical_replays_share_compute_but_preserve_every_source_row(self):
        duplicate = {**copy.deepcopy(self.rows[0]), "source_row_idx": 100,
                     "contact_tree_order_indices": [0, 1]}
        rows = [self.rows[0], self.rows[1], duplicate]
        tokenizer = FixtureTokenizer()
        features = extract_features(FixtureP0(), tokenizer, rows, spec=SPEC,
                                    batch_size=2, exact_prompt_text="NATIVE\n")
        self.assertEqual(features.shape, (3, 8))
        self.assertEqual(len(tokenizer.texts), 2)
        self.assertTrue(torch.equal(features[0], features[2]))
        self.assertEqual(rows[2]["source_row_idx"], 100)
        self.assertNotEqual(rows[0]["contact_tree_order_indices"], rows[2]["contact_tree_order_indices"])

    def test_export_preserves_upstream_failure_and_original_row_order(self):
        rows = [self.rows[0], {"sample_idx": 55, "parsed": False, "message": "original failure"},
                {**self.rows[0], "attempt_status": "decode_failure"}, self.rows[1]]
        output, report = export_rows(
            rows, model=FixtureP0(), tokenizer=FixtureTokenizer(), pointer=self.head,
            feature_spec=SPEC, pointer_sha256="fixture", batch_size=2, exact_prompt_text="NATIVE\n",
        )
        self.assertEqual(len(output), len(rows))
        self.assertEqual(output[1]["message"], "original failure")
        self.assertEqual(output[1]["r03_control"]["status"], "upstream_failure")
        self.assertEqual(output[2]["r03_control"]["status"], "upstream_failure")
        self.assertEqual(report["successful_programs"], 2)
        for original, exported in zip(rows, output):
            self.assertTrue(all(exported[key] == value for key, value in original.items()))


if __name__ == "__main__":
    unittest.main()
