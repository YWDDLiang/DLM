from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("PyTorch is required") from exc


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.c3fd_calibration import StratumInteraction
from crystal_dlm.c3fd_llama_typed_planner import (
    C3FDLlamaTypedPlannerConfig,
    C3FDLlamaTypedResidualPlanner,
)
from scripts.train_c3fd_llama_typed_planner import (
    FrozenC3FDBundle,
    SEED,
    assert_c3fd_frozen,
    audit_step0_equality,
    collate_typed_rows,
    cosine_with_warmup_lambda,
    forward_fused_batch,
    freeze_c3fd,
    validate_training_contract,
)


class FakeHead:
    num_joint_actions = 5

    @staticmethod
    def joint_action_scores(species_logits, count_logits, **_kwargs):
        real = (
            species_logits[..., :2].unsqueeze(-1)
            + count_logits.unsqueeze(-2)
        ).reshape(*species_logits.shape[:-1], 4)
        return torch.cat((real, species_logits[..., 2:3]), dim=-1)


class FakeC3FD(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.25))
        self.config = SimpleNamespace(
            num_species=2,
            max_count=2,
            max_sequence_length=5,
        )
        self.head = FakeHead()

    def forward(self, context, *, previous_species_indices, **_kwargs):
        batch, width = previous_species_indices.shape
        device = context.device
        zero = self.anchor.detach() * 0.0
        species = torch.zeros(batch, width, 3, device=device) + zero
        count = torch.zeros(batch, width, 2, device=device) + zero
        # Make teacher ranks deterministic but non-trivial.
        species[..., 0] = 0.4
        count[..., 0] = 0.2
        rich = {
            "lattice_system": torch.zeros(batch, width, 2, device=device) + zero,
            "spacegroup_bucket": torch.zeros(batch, width, 2, device=device) + zero,
            "volume_per_atom_bin": torch.zeros(batch, width, 2, device=device) + zero,
        }
        return SimpleNamespace(
            family_logits=torch.zeros(batch, 2, device=device) + zero,
            n_logits=torch.zeros(batch, 20, device=device) + zero,
            arity_logits=torch.zeros(batch, 7, device=device) + zero,
            species_logits=species,
            count_logits=count,
            rich_logits=rich,
        )


class FakeLlama(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.lora_adapter = nn.Linear(hidden, hidden, bias=False)
        nn.init.eye_(self.lora_adapter.weight)

    def forward(self, *, inputs_embeds, output_hidden_states, use_cache, **_kwargs):
        if not output_hidden_states or use_cache:
            raise AssertionError("typed trainer changed Llama execution contract")
        hidden = self.lora_adapter(inputs_embeds)
        return SimpleNamespace(hidden_states=(hidden,))


def vocabulary():
    return {
        "species": [
            {"id": 0, "atomic_number": 8, "oxidation_state": -2},
            {"id": 1, "atomic_number": 14, "oxidation_state": 4},
        ],
        "soft_vocabulary": {
            "anion_framework": ["oxide_like", "alloy_or_intermetallic"],
            "lattice_system": ["cubic", "monoclinic"],
            "spacegroup_bucket": ["sg_a", "sg_b"],
            "volume_per_atom_bin": ["small", "large"],
        },
    }


def bundle():
    model = FakeC3FD()
    freeze_c3fd(model)
    interaction = StratumInteraction(
        strata=((0, 3, 1), (1, 4, 1)),
        log_corrections=(0.0, 0.0),
        counts=(1, 1),
        alpha=1.0,
    )
    calibration = {
        name: {"temperature": 1.0}
        for name in ("family", "n", "arity", "species", "count")
    }
    return FrozenC3FDBundle(
        model=model,
        context=torch.zeros(1, 4),
        interaction=interaction,
        calibration=calibration,
        vocabulary=vocabulary(),
        proposal_legal_mask=torch.tensor([True, True]),
        stratum_to_index={(0, 3, 1): 0, (1, 4, 1): 1},
        checkpoint_sha256="a" * 64,
        vocabulary_sha256="b" * 64,
    )


def row(*, stratum=0, weight=1.0):
    family, n_value = ((0, 3) if stratum == 0 else (1, 4))
    species_id = 0 if stratum == 0 else 1
    count = n_value
    # Fake max_count is two; use N=2-compatible count targets while retaining
    # distinct proposal strata in the fixture.
    count = 2
    action = species_id * 2 + count - 1
    return {
        "schema": "c3fd_llama_fused_typed_dataset_v1",
        "stability_condition": "meta_or_better" if stratum == 0 else "higher",
        "sample_weight": weight,
        "proposal_target": {
            "family_id": family,
            "N": n_value,
            "arity": 1,
        },
        "species_ids": [species_id],
        "count_targets": [count],
        "ledger_steps": [
            {"remaining_atoms": n_value, "net_charge": 0, "remaining_species": 1, "branch": "unset"},
            {"remaining_atoms": n_value, "net_charge": 0, "remaining_species": 1, "branch": "unset"},
            {"remaining_atoms": 0, "net_charge": 0, "remaining_species": 0, "branch": "ionic"},
        ],
        "legal_action_indices": [
            sorted({action, 0 if action != 0 else 1}),
            [4],
        ],
        "soft_targets": {
            "lattice_system": {"label": 0},
            "spacegroup_bucket": {"label": 1},
            "volume_per_atom_bin": {"label": 0},
        },
    }


def residual_module():
    return C3FDLlamaTypedResidualPlanner(
        C3FDLlamaTypedPlannerConfig(
            llama_hidden_size=8,
            typed_embedding_size=4,
            num_stability_goals=2,
            num_proposal_states=3,
            num_proposal_strata=2,
            num_species=2,
            max_count=2,
            ledger_feature_size=6,
            num_lattice_systems=2,
            num_spacegroup_buckets=2,
            num_volume_per_atom_bins=2,
            max_sequence_length=5,
        )
    )


class PositionAndMaskTest(unittest.TestCase):
    def test_proposal_action_and_terminal_positions_are_exact(self):
        batch = collate_typed_rows([row(stratum=0), row(stratum=1)], bundle=bundle())
        self.assertEqual(batch["proposal_state_ids"].tolist(), [[0, 1, 1], [0, 2, 2]])
        self.assertEqual(batch["previous_species_indices"].tolist(), [[-1, -1, 0], [-1, -1, 1]])
        self.assertEqual(batch["previous_count_values"].tolist(), [[0, 0, 2], [0, 0, 2]])
        self.assertEqual(batch["previous_n_values"].tolist(), [[0, 3, 0], [0, 4, 0]])
        self.assertEqual(batch["soft_position_indices"].tolist(), [2, 2])
        self.assertEqual(batch["action_targets"].tolist(), [[1, 4], [3, 4]])
        self.assertTrue(torch.equal(batch["ledger_features"][:, 0], torch.zeros(2, 6)))

    def test_illegal_teacher_proposal_and_action_fail_closed(self):
        frozen = bundle()
        frozen = FrozenC3FDBundle(**{**frozen.__dict__, "proposal_legal_mask": torch.tensor([False, True])})
        with self.assertRaises(ValueError):
            collate_typed_rows([row(stratum=0)], bundle=frozen)
        bad = row(stratum=0)
        bad["legal_action_indices"][0] = [0]
        with self.assertRaises(ValueError):
            collate_typed_rows([bad], bundle=bundle())


class ExecutionTest(unittest.TestCase):
    def test_step0_distribution_is_exactly_frozen_c3fd(self):
        frozen = bundle()
        batch = collate_typed_rows([row(stratum=0), row(stratum=1)], bundle=frozen)
        result = audit_step0_equality(
            llama=FakeLlama(),
            residual=residual_module(),
            bundle=frozen,
            batch=batch,
            atol=1e-7,
        )
        self.assertLessEqual(result["max_abs_log_probability_delta"], 1e-7)

    def test_row_balanced_loss_backpropagates_but_c3fd_stays_frozen(self):
        frozen = bundle()
        llama = FakeLlama()
        residual = residual_module()
        batch = collate_typed_rows(
            [row(stratum=0, weight=1.0), row(stratum=1, weight=3.0)],
            bundle=frozen,
        )
        loss, diagnostics = forward_fused_batch(
            llama=llama,
            residual=residual,
            bundle=frozen,
            batch=batch,
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreaterEqual(diagnostics["fused_vs_base_kl"], -1e-7)
        self.assertGreaterEqual(diagnostics["teacher_base_rank"], 1.0)
        loss.backward()
        self.assertGreater(float(residual.action_head.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(frozen.model.anchor.grad)
        assert_c3fd_frozen(frozen.model)

    def test_frozen_c3fd_rejects_trainable_or_gradient_state(self):
        model = FakeC3FD()
        with self.assertRaises(RuntimeError):
            assert_c3fd_frozen(model)
        freeze_c3fd(model)
        assert_c3fd_frozen(model)


class FixedContractTest(unittest.TestCase):
    def test_one_epoch_single_seed_effective_batch_and_final_only(self):
        args = SimpleNamespace(
            seed=SEED,
            epochs=1,
            batch_size=2,
            grad_accum=8,
            lr=2e-5,
            warmup_steps=100,
        )
        validate_training_contract(args)
        for name, value in (("seed", 1), ("epochs", 2), ("grad_accum", 4)):
            changed = SimpleNamespace(**vars(args))
            setattr(changed, name, value)
            with self.assertRaises(ValueError):
                validate_training_contract(changed)
        source = (ROOT / "src/scripts/train_c3fd_llama_typed_planner.py").read_text(encoding="utf-8")
        self.assertIn('"eligible_checkpoint": "final_only"', source)
        self.assertIn('"checkpoint_selection": "none"', source)
        self.assertNotIn("best_checkpoint", source)

    def test_cosine_schedule_has_warmup_and_reaches_zero(self):
        self.assertAlmostEqual(cosine_with_warmup_lambda(0, total_steps=200, warmup_steps=100), 0.01)
        self.assertAlmostEqual(cosine_with_warmup_lambda(99, total_steps=200, warmup_steps=100), 1.0)
        self.assertAlmostEqual(cosine_with_warmup_lambda(200, total_steps=200, warmup_steps=100), 0.0)


if __name__ == "__main__":
    unittest.main()
