import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("PyTorch is required") from exc


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sample_c3fd_llama_typed_planner",
    ROOT / "src" / "scripts" / "sample_c3fd_llama_typed_planner.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import fused typed sampler")
MODULE = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from crystal_dlm.ccfd import FormulaToken
from crystal_dlm.composition_pair_prior import ValenceNode
from crystal_dlm.c3fd_native_plan import serialize_native_plan
from crystal_dlm.c3fd_llama_typed_planner import TypedResidualLogits


class FakeReachability:
    def __init__(self):
        self.proposal_calls = []

    def can_complete(self, state, *, family, target_arity, max_species):
        self.proposal_calls.append((state.target_atoms, family, target_arity, max_species))
        return state.target_atoms == 2 and family == "halide" and target_arity == 2

    def legal_species_counts(self, state, *, family, target_arity, max_species):
        if len(state.tokens) == 0:
            return (FormulaToken.from_symbol("Na", 1, 1),)
        if len(state.tokens) == 1:
            return (FormulaToken.from_symbol("Cl", -1, 1),)
        return ()


class FakeRuntime:
    def __init__(self, *, fail_actions=False):
        self.interaction = SimpleNamespace(strata=[(0, 1, 2), (0, 2, 2)])
        self.family_values = ["halide"]
        self.nodes = (ValenceNode(11, 1), ValenceNode(17, -1))
        self.node_to_id = {node: index for index, node in enumerate(self.nodes)}
        self.max_count = 2
        self.max_species = 7
        self.eos_action_index = 4
        self.stability_goal_id = 0
        self.reachability = FakeReachability()
        self.soft_values = {
            "lattice_system": ["cubic", "triclinic", "<UNKNOWN>"],
            "spacegroup_bucket": ["sg_195_230", "sg_001_002", "<UNKNOWN>"],
            "volume_per_atom_bin": ["volpa_010_014", "volpa_015_019", "<UNKNOWN>"],
        }
        self.sequence_lengths = []
        self.action_calls = 0
        self.fail_actions = fail_actions

    def proposal_logits(self):
        return torch.tensor([20.0, 0.0])

    def action_logits(self, state, **_kwargs):
        self.action_calls += 1
        if self.fail_actions:
            raise RuntimeError("synthetic action failure")
        return torch.zeros(5), {"step": self.action_calls}

    def residual_logits(self, sequence):
        self.sequence_lengths.append(sequence.length)
        return TypedResidualLogits(
            proposal=torch.zeros(1, 2),
            actions=torch.zeros(1, sequence.length, 5),
            soft_fields={
                "lattice_system": torch.tensor([[8.0, 0.0, 0.0]]),
                "spacegroup_bucket": torch.tensor([[8.0, 0.0, 0.0]]),
                "volume_per_atom_bin": torch.tensor([[8.0, 0.0, 0.0]]),
            },
        )

    def soft_logits(self, _context):
        return {
            "lattice_system": torch.tensor([8.0, 0.0, -2.0]),
            "spacegroup_bucket": torch.tensor([8.0, 0.0, -2.0]),
            "volume_per_atom_bin": torch.tensor([8.0, 0.0, -2.0]),
        }

    def terminal_certificate(self, state):
        return {
            "benchmark_compatible": bool(
                state.remaining_atoms == 0 and state.net_charge == 0 and len(state.tokens) == 2
            ),
            "certificate_class": "benchmark_compatible",
        }


class FakeLlama:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        embeds = kwargs["inputs_embeds"]
        return SimpleNamespace(hidden_states=(embeds + 1.0,))


class TypedSamplerTest(unittest.TestCase):
    def sample(self, runtime=None, sample_idx=4):
        return MODULE.sample_single_trajectory(
            runtime or FakeRuntime(),
            sample_idx=sample_idx,
            seed=21,
            temperature=0.9,
            top_p=0.95,
            top_k=0,
        )

    def test_proposal_is_pre_masked_by_family_N_arity_can_complete(self):
        runtime = FakeRuntime()
        record = self.sample(runtime)
        proposal = record["audit"][0]
        self.assertEqual(proposal["selected_index"], 1)
        self.assertEqual(proposal["pre_masked_strata"], 1)
        self.assertEqual(runtime.reachability.proposal_calls[0][:3], (1, "halide", 2))
        self.assertEqual(runtime.reachability.proposal_calls[1][:3], (2, "halide", 2))

    def test_exact_eos_hard_validity_and_stable_native_serializer(self):
        record = self.sample()
        self.assertTrue(record["comp_valid"])
        self.assertEqual(record["semantic_trace"][-1], {"action": "EOS"})
        self.assertEqual(record["plan_state"]["N"], 2)
        self.assertEqual(record["plan_state"]["elements"], ["Na", "Cl"])
        self.assertEqual(record["plan_state"]["counts"], [1, 1])
        self.assertEqual(record["plan_text"], serialize_native_plan(record["plan_state"]))
        self.assertEqual(
            set(record["species_program"]), set(record["plan_state"]["elements"])
        )
        self.assertEqual(record["species_program_source"], "canonical_compatibility")
        action_events = [row for row in record["audit"] if row["step"] == "action"]
        self.assertEqual(action_events[-1]["action"], "EOS")
        self.assertEqual(len(action_events), 3)

    def test_one_trajectory_deterministic_seed_and_no_hidden_sampling_attempts(self):
        first_runtime = FakeRuntime()
        second_runtime = FakeRuntime()
        first = self.sample(first_runtime, sample_idx=17)
        second = self.sample(second_runtime, sample_idx=17)
        self.assertEqual(first["plan_text"], second["plan_text"])
        self.assertEqual(first["audit"], second["audit"])
        self.assertEqual(first["trajectory_attempts"], 1)
        self.assertEqual(first_runtime.action_calls, 3)
        self.assertEqual(first_runtime.sequence_lengths, [1, 2, 3, 4, 4])

    def test_audit_contains_fused_KL_and_selected_base_rank(self):
        record = self.sample()
        for event in record["audit"]:
            self.assertIn("kl_fused_vs_c3fd", event)
            self.assertIn("selected_action_base_c3fd_rank", event)
            self.assertGreaterEqual(event["selected_action_base_c3fd_rank"], 1)
            self.assertGreaterEqual(event["kl_fused_vs_c3fd"], -1e-7)

    def test_failures_preserve_requested_sample_idx_without_second_attempt(self):
        runtime = FakeRuntime(fail_actions=True)
        records, plans, metrics = MODULE.sample_requests(
            runtime,
            [{"sample_idx": 8}, {"sample_idx": 9}],
            seed=21,
            temperature=0.9,
            top_p=0.95,
            top_k=0,
        )
        self.assertEqual([row["sample_idx"] for row in records], [8, 9])
        self.assertEqual([row["trajectory_attempts"] for row in records], [1, 1])
        self.assertEqual(plans, [])
        self.assertEqual(runtime.action_calls, 2)
        self.assertEqual(metrics["requested_samples"], 2)
        self.assertEqual(metrics["all_request_benchmark_comp_valid"], 0)

    def test_llama_recomputes_full_sequence_without_KV_cache(self):
        llama = FakeLlama()
        embeds = torch.zeros(1, 4, 3)
        hidden = MODULE.recompute_llama_hidden(llama, embeds)
        self.assertEqual(tuple(hidden.shape), (1, 4, 3))
        self.assertEqual(len(llama.calls), 1)
        call = llama.calls[0]
        self.assertIs(call["use_cache"], False)
        self.assertNotIn("past_key_values", call)
        self.assertTrue(call["output_hidden_states"])

    def test_source_ledger_requires_exact_unique_requested_ordinals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            path.write_text('{"sample_idx":5}\n{"ordinal":7}\n', encoding="utf-8")
            self.assertEqual(
                MODULE.load_requested_rows(path, requested=2),
                [
                    {"sample_idx": 5, "source_position": 0},
                    {"sample_idx": 7, "source_position": 1},
                ],
            )
            path.write_text('{"sample_idx":5}\n{"ordinal":5}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.load_requested_rows(path, requested=2)

    def test_contract_has_no_fallback_or_unbounded_sampling_loop(self):
        source = (ROOT / "src/scripts/sample_c3fd_llama_typed_planner.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("while True", source)
        self.assertNotIn("for attempt", source)
        self.assertNotIn("past_key_values=", source)
        self.assertIn('"retry": False', source)
        self.assertIn('"best_of_n": False', source)

    def test_parser_can_freeze_a_nondefault_requested_count(self):
        args = MODULE.build_parser().parse_args(
            [
                "--c3fd-checkpoint", "c3fd.pt",
                "--data-dir", "data",
                "--llama-model", "llama",
                "--fused-planner-final", "fused",
                "--source-ledger", "ledger.jsonl",
                "--output-dir", "out",
                "--expected-c3fd-sha256", "a",
                "--expected-vocabulary-sha256", "b",
                "--expected-source-ledger-sha256", "c",
                "--expected-typed-config-sha256", "d",
                "--expected-typed-state-sha256", "e",
                "--expected-adapter-config-sha256", "f",
                "--expected-adapter-model-sha256", "g",
                "--requested", "1200",
                "--expected-requested", "1200",
                "--seed", "23",
                "--expected-seed", "23",
            ]
        )
        self.assertEqual(args.requested, 1200)
        self.assertEqual(args.expected_requested, 1200)


if __name__ == "__main__":
    unittest.main()
