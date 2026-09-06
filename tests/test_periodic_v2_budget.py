import torch
from torch.utils.data import DistributedSampler

from scripts.train_periodic_dlm_v2 import budget


def test_six_rank_budget_retains_every_source_view_twice_and_zero_weight_padding():
    plan = budget()
    assert plan["effective_states"] == 108544
    assert plan["updates"] == 4524 and plan["padding_per_epoch"] == 16
    coverage = torch.zeros(plan["real_states_per_epoch"], dtype=torch.long)
    sequence = range(plan["padded_states_per_epoch"])
    for epoch in range(2):
        padding = 0
        for rank in range(6):
            sampler = DistributedSampler(sequence, num_replicas=6, rank=rank, seed=27)
            sampler.set_epoch(epoch)
            assert len(sampler) % 4 == 0  # micro2 times accumulation2
            for index in sampler:
                if index < len(coverage):
                    coverage[index] += 1
                else:
                    padding += 1
        assert padding == 16
    assert bool((coverage == 2).all())


def test_padding_scale_preserves_original_state_risk_including_ineligible_zero_states():
    plan = budget()
    real = plan["real_states_per_epoch"]
    losses = torch.arange(real, dtype=torch.float64).remainder(11)
    losses[::3] = 0  # e.g. retained ineligible prefixes
    padded = torch.cat((losses, torch.zeros(plan["padding_per_epoch"])))
    step_means = padded.reshape(-1, 6, 2, 2).mean(dim=(1, 2, 3))
    torch.testing.assert_close(step_means.mean() * len(padded) / real, losses.mean(), atol=1e-12, rtol=0)
