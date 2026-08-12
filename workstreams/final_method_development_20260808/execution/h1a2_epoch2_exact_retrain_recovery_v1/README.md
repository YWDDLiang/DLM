# H1-A2 epoch-2 exact retraining recovery

This immutable source bundle reproduces the historical H1-A2 epoch-2 planner
training stage only. It starts from the byte-frozen epoch-1 LoRA adapter and
runs the historical trainer once for exactly one additional epoch (3,392
optimizer updates).

Scientific identity is fail-closed on:

- the historical trainer and every imported planner helper;
- base-model metadata/index hashes plus all four shard names and exact byte
  sizes (full shard hashing was intentionally avoided after a no-throughput
  read on the shared filesystem; the byte-exact historical adapter and plan
  replay independently anchor the base-model identity);
- epoch-1 adapter/tokenizer hashes;
- train/validation/test row counts and hashes;
- Python and package versions;
- optimizer, scheduler, LoRA, seed, precision, batching, and evaluation
  defaults used by the June run.

The scheduler allocation uses one A800 and at most eight CPUs. The historical
launcher allocated two A800s, but the trainer was a plain single-process
Python program and used only CUDA device 0; GPU count is therefore a resource
change, not a training-method change.

The job does not sample plans, run a DLM, refine structures, evaluate S.U.N.,
or access Materials Project. Those stages are frozen only after the newly
trained adapter has been audited.
