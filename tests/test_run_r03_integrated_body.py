"""CPU call-contract fixtures; no original B0 model or GPU is loaded."""
import copy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from scripts.run_r03_integrated_body import (
    FrozenRuntime, construct_batch, frozen_imports, make_batches,
    native_revision_slots, prepare_tasks, repair_batch, summarize, upstream_failure,
)


OLD_ROOT = (Path(__file__).resolve().parents[2] / "diffsion_language_model_meets_diffusion" /
            "workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1")


def sample_plan():
    return {"N": 3, "elements": ["Ti", "O"], "counts": [1, 2], "formula": "TiO2", "reduced_formula": "TiO2",
            "anion_framework": "oxide", "charge_bucket": "neutral_plausible",
            "lattice_system": "cubic", "spacegroup_bucket": "sg_195_230",
            "volume_per_atom_bin": "volpa_015_019", "prototype_key": "fixture"}


def load_schedule_fixture():
    runtime = FrozenRuntime(OLD_ROOT)
    with frozen_imports(runtime):
        schedule = importlib.import_module("safe_axis_schedule")
        state = importlib.import_module("crystal_dlm.r5_plan_state")
        lengths = importlib.import_module("crystal_dlm.r5_dynamic_length")
        noise = importlib.import_module("paired_noise")
        runtime.module = SimpleNamespace(
            h1a2_safe_axis_generation_schedule=schedule.h1a2_safe_axis_generation_schedule,
            require_safe_axis_schedule=schedule.require_safe_axis_schedule,
            build_body_prompt=state.build_body_prompt,
            exact_body_token_count=lengths.exact_body_token_count,
            exact_dynamic_schema_constraints=lambda tokenizer, n: [[0, 1]] * (7 + 4 * n),
            count_prefill_for_batch=lambda tokenizer, n, size: {0: [n] * size},
            element_prefill_for_batch=lambda tokenizer, plans: {
                7 + 4 * slot: [100 + slot] * len(plans) for slot in range(plans[0]["N"])
            },
            merge_prefill_maps=lambda *maps: {key: value for item in maps for key, value in item.items()},
            MASK_TOKEN_ID=126336,
        )
    return runtime


try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None or not OLD_ROOT.exists(), "CPU torch and archived schedule fixture required")
class FrozenR03CallContractTest(unittest.TestCase):
    def setUp(self):
        self.runtime = load_schedule_fixture()

    def test_archived_imports_are_isolated_and_restored(self):
        import crystal_dlm.r5_plan_state as current
        original = sys.modules["crystal_dlm.r5_plan_state"]
        with frozen_imports(self.runtime):
            historical = importlib.import_module("crystal_dlm.r5_plan_state")
            self.assertIsNot(historical, original)
            self.assertIn(str(OLD_ROOT), historical.__file__)
        self.assertIs(sys.modules["crystal_dlm.r5_plan_state"], current)

    def test_plan_and_all_failures_preserved_with_exact_global_axis_schedule(self):
        plan = sample_plan()
        original = copy.deepcopy(plan)
        rows = [{"sample_idx": 8, "plan_state": plan, "attempt_status": "complete"},
                {"sample_idx": 9, "attempt_status": "planner_parse_failure", "plan_state": plan}]
        tasks = prepare_tasks(rows, self.runtime, seed=901)
        self.assertEqual(tasks[0]["plan_state"], original)
        self.assertFalse(tasks[1]["eligible"])
        self.assertEqual(tasks[1]["sample_idx"], 9)
        groups = tasks[0]["schedule"]
        self.assertEqual(groups, self.runtime.module.h1a2_safe_axis_generation_schedule(plan))
        position_to_group = {position: index for index, group in enumerate(groups) for position in group}
        self.assertLess(max(position_to_group[8 + 4 * site] for site in range(3)),
                        min(position_to_group[9 + 4 * site] for site in range(3)))
        self.assertLess(max(position_to_group[9 + 4 * site] for site in range(3)),
                        min(position_to_group[10 + 4 * site] for site in range(3)))
        self.assertEqual(len(tasks), len(rows))
        self.assertEqual(plan, original)

    def test_duplicate_request_or_wrong_body_prompt_cannot_silently_run(self):
        row = {"sample_idx": 8, "plan_state": sample_plan()}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_tasks([row, row], self.runtime, seed=1)
        tasks = prepare_tasks([{**row, "body_prompt": "a different prompt"}], self.runtime, seed=1)
        self.assertFalse(tasks[0]["eligible"])
        self.assertIn("disagrees", tasks[0]["reason"])

    def test_batches_keep_each_original_seed_and_never_repeat_a_request(self):
        rows = [{"sample_idx": 100 + index, "plan_state": sample_plan(), "body_noise_seed": 123 + index}
                for index in range(10)]
        tasks = prepare_tasks(rows, self.runtime, seed=901)
        batches = make_batches(tasks, batch_size=8)
        self.assertEqual([len(batch) for batch in batches], [8, 2])
        self.assertEqual([task["body_noise_seed"] for batch in batches for task in batch], list(range(123, 133)))

    def test_frozen_sampler_receives_original_hyperparameters_prefill_and_schedule(self):
        tasks = prepare_tasks([{"sample_idx": 8, "plan_state": sample_plan(), "body_noise_seed": 887}], self.runtime, seed=901)
        seen = {}

        def fake_generate(model, prompt, **kwargs):
            seen.update(kwargs)
            suffix = torch.ones((prompt.shape[0], kwargs["gen_length"]), dtype=torch.long)
            for position, values in kwargs["prefill_token_ids_by_generation_pos"].items():
                suffix[:, position] = torch.tensor(values)
            return torch.cat([prompt, suffix], dim=1)

        self.runtime.module.generate_paired_exact_plan = fake_generate
        model = torch.nn.Linear(1, 1)

        def tokenizer(prompts, **kwargs):
            self.assertFalse(kwargs["add_special_tokens"])
            return {"input_ids": torch.tensor([[17, 18]]), "attention_mask": torch.tensor([[1, 1]])}

        suffix, trace = construct_batch(model, tokenizer, tasks, self.runtime, constraints={"native": True})
        self.assertEqual(suffix.shape, (1, 19))
        self.assertEqual(seen["temperature"], 0.7)
        self.assertEqual(seen["cfg_scale"], 0.0)
        self.assertEqual(seen["remasking"], "low_confidence")
        self.assertEqual(seen["base_seeds"], [887])
        self.assertEqual(seen["generation_position_groups"], tasks[0]["schedule"])
        self.assertEqual(seen["lightweight_decoding_constraints"], {"native": True})
        self.assertEqual(trace["prefill"]["0"], [3])


class DenominatorTest(unittest.TestCase):
    def test_failed_planner_row_is_counted_in_metrics_denominator(self):
        records = [{"body_eligible": False, "reason": "planner_failure"},
                   {"body_eligible": True, "body_generation_complete": True,
                    "body_plan_match": True, "body_graph_complete": True}]
        metrics = summarize(records, elapsed=1)
        self.assertEqual(metrics["requested_samples"], 2)
        self.assertEqual(metrics["planner_failures"], 1)
        self.assertEqual(metrics["graph_acceptance_rate"], 0.5)

    def test_explicit_failed_status_takes_precedence_over_parseable_plan(self):
        self.assertIsNotNone(upstream_failure({"attempt_status": "failed", "plan_state": sample_plan()}))


if torch is not None:
    class RepairTokenizer:
        def __init__(self):
            from crystal_dlm.fixed_slot import build_special_tokens
            self.vocab = {token: index + 2 for index, token in enumerate(build_special_tokens())}
            self.pad_token_id = 0

        def get_vocab(self):
            return self.vocab

        def __call__(self, texts, **kwargs):
            if isinstance(texts, str):
                return {"input_ids": [self.vocab[texts]] if texts in self.vocab else [1, 1]}
            return {"input_ids": torch.ones(len(texts), 2, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), 2, dtype=torch.long)}


    class RepairFixtureModel(torch.nn.Module):
        def __init__(self, tokenizer):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.width = max(tokenizer.vocab.values()) + 1
            self.targets = [tokenizer.vocab[f"<{axis}_{value:03d}>"] for axis, value in zip("XYZ", (20, 30, 40))]
            self.calls = 0
            self.contexts = []

        def forward(self, tokens, *, attention_mask):
            self.calls += 1
            self.contexts.append(tokens.detach().clone())
            logits = torch.zeros((*tokens.shape, self.width), dtype=torch.float32)
            logits[:, :, self.targets] = 12.0
            return SimpleNamespace(logits=logits)


    def repair_fixture(tokenizer):
        tokens = ["<N_003>", "<LA_040>", "<LB_040>", "<LC_040>", "<AA_090>", "<AB_090>", "<AG_090>"]
        for symbol, coordinates in zip(("O", "Li", "Cl"), ((0, 0, 0), (50, 50, 50), (100, 75, 75))):
            tokens.append(f"<E_{symbol}>")
            tokens.extend(f"<{axis}_{value:03d}>" for axis, value in zip("XYZ", coordinates))
        body = [tokenizer.vocab[token] for token in tokens]
        task = {"body_noise_seed": 991, "plan_state": {"N": 3, "elements": ["O", "Li", "Cl"], "counts": [1, 1, 1]},
                "source_row": {"species_program": ["Li", "O", "Cl"], "r03_control": {
                    "status": "ok", "scope": "post_construction_only", "sweeps": 1,
                    "revision_species": ["O", "Li"], "revision_slots": [0, 1],
                    "slot_mapping": "original_plan_element_order_counts"}}}
        return task, body


@unittest.skipIf(torch is None or not OLD_ROOT.exists(), "CPU torch and archived noise fixture required")
class RepairTransactionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        from crystal_dlm import r03_physics_transfer
        self.physics = r03_physics_transfer
        self.tokenizer = RepairTokenizer()
        self.constraints = self.physics.build_repair_constraints(self.tokenizer)
        self.task, self.body = repair_fixture(self.tokenizer)
        self.noise = load_schedule_fixture().modules["paired_noise"]

    def test_anchor_slots_follow_actual_unsorted_original_plan(self):
        self.assertEqual(native_revision_slots(self.task, self.body, self.tokenizer), [0, 1])
        wrong = self.body.copy()
        wrong[7], wrong[11] = wrong[11], wrong[7]
        with self.assertRaisesRegex(ValueError, "prefill"):
            native_revision_slots(self.task, wrong, self.tokenizer)

    def test_only_active_xyz_commits_and_nonactive_periodic_alias_is_unchanged(self):
        model = RepairFixtureModel(self.tokenizer)
        original = self.body.copy()
        repaired, reports = repair_batch(model, self.tokenizer, [self.task], [self.body],
                                         constraints=self.constraints, noise_api=self.noise, physics_api=self.physics)
        self.assertEqual(self.body, original)
        self.assertEqual(model.calls, 6)
        self.assertEqual(reports[0]["attempted_anchors"], 2)
        self.assertTrue(reports[0]["geometry_after"]["supported"])
        active = {8, 9, 10, 12, 13, 14}
        self.assertTrue(all(repaired[0][position] == value for position, value in enumerate(original) if position not in active))
        self.assertEqual(repaired[0][16], self.tokenizer.vocab["<X_100>"])
        self.assertIn(16, reports[0]["transactions"][0]["context_alias_positions"])
        self.assertTrue(any(int(context[0, 2 + 16]) == self.tokenizer.vocab["<X_000>"] for context in model.contexts))

    def test_empty_support_rolls_back_whole_xyz_without_resampling(self):
        original_support = self.physics.supported_scalar_logits

        def reject_y(raw, body, prompt_length, n, position, **kwargs):
            vector, report = original_support(raw, body, prompt_length, n, position, **kwargs)
            if (position - 8) % 4 == 1:
                return torch.full_like(vector, torch.finfo(vector.dtype).min), {
                    "available": False, "no_legal_completion": True, "reason": "fixture_no_completion", "legal_count": 0}
            return vector, report

        proxy = SimpleNamespace(**{name: getattr(self.physics, name) for name in (
            "REPAIR_SUPPORT_PROTOCOL", "geometry_support_report", "canonicalize_body_aliases", "repair_scalar_example")},
            supported_scalar_logits=reject_y)
        model = RepairFixtureModel(self.tokenizer)
        repaired, reports = repair_batch(model, self.tokenizer, [self.task], [self.body],
                                         constraints=self.constraints, noise_api=self.noise, physics_api=proxy)
        self.assertEqual(repaired[0], self.body)
        self.assertEqual(model.calls, 4)
        self.assertEqual(reports[0]["rolled_back_anchors"], 2)
        self.assertEqual(reports[0]["committed_anchors"], 0)

    def test_missing_controller_is_disclosed_without_model_calls(self):
        task = copy.deepcopy(self.task)
        task["source_row"]["r03_control"]["status"] = "controller_failure"
        model = RepairFixtureModel(self.tokenizer)
        repaired, reports = repair_batch(model, self.tokenizer, [task], [self.body],
                                         constraints=self.constraints, noise_api=self.noise, physics_api=self.physics)
        self.assertEqual(model.calls, 0)
        self.assertEqual(repaired[0], self.body)
        self.assertEqual(reports[0]["status"], "controller_failure")


if __name__ == "__main__":
    unittest.main()
