"""Keep sampling batches fixed while redistributing them to more GPU workers.

The fixed development reference bucketed requests by ordinal modulo two and
then by (atom count, prompt length). Reproduce that layout before assigning
whole batches to six workers: physical action seeds and batch membership do
not depend on the number of execution workers.
"""
from collections import defaultdict


def sampling_batches(items, *, batch_size, rank, world_size, layout_world_size=None):
    if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid sampling allocation")
    if layout_world_size is not None and layout_world_size < 1:
        raise ValueError("sampling layout must have a positive logical world size")
    buckets = defaultdict(list)
    for item in items:
        ordinal, _, compiled = item
        key = (compiled["program"].num_atoms, len(compiled["prompt_token_ids"]))
        if layout_world_size is None:
            if ordinal % world_size != rank:
                continue
        else:
            key = (ordinal % layout_world_size, *key)
        buckets[key].append(item)
    serial = 0
    for key in sorted(buckets):
        bucket = buckets[key]
        for offset in range(0, len(bucket), batch_size):
            batch = bucket[offset:offset + batch_size]
            if layout_world_size is None or serial % world_size == rank:
                yield batch
            serial += 1
