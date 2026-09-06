from types import SimpleNamespace

from crystal_dlm.sampling_layout import sampling_batches


def test_six_workers_preserve_two_worker_reference_batches_and_every_request():
    # Ragged shape buckets make ordinal-modulo-six produce different batches.
    items = [(i, 0, {"program": SimpleNamespace(num_atoms=2 + i % 4),
                     "prompt_token_ids": [0] * (5 + i % 5)}) for i in range(256)]
    def layout(world, logical=None):
        return [tuple((item[0], item[1]) for item in batch)
                for rank in range(world)
                for batch in sampling_batches(items, batch_size=4, rank=rank,
                                               world_size=world, layout_world_size=logical)]
    reference = layout(2)
    six = layout(6, 2)
    assert sorted(six) == sorted(reference)
    assert sorted(x for batch in six for x in batch) == [(i, 0) for i in range(256)]
    assert sorted(layout(6)) != sorted(reference)


def test_multiple_candidates_stay_in_the_same_predeclared_batch():
    compiled = {"program": SimpleNamespace(num_atoms=2), "prompt_token_ids": [1, 2]}
    items = [(i, j, compiled) for i in range(9) for j in range(4)]
    batches = [batch for rank in range(6) for batch in sampling_batches(
        items, batch_size=4, rank=rank, world_size=6, layout_world_size=2)]
    assert len(batches) == 9
    assert all(len({x[0] for x in batch}) == 1 for batch in batches)
