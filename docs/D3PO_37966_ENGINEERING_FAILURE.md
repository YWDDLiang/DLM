# D3PO job 37966 engineering failure

Date: 2026-08-30

## Terminal status

- Slurm job: `37966`, `h1a2-d3po-min`, one A800/eight CPUs.
- Failure occurred while loading seed `81017`, before the step-0 equality
  canary, optimizer construction, or any parameter update.
- `_FAILED.json` records `ValueError: LLaDAModelLM does not support gradient
  checkpointing.` No `_SUCCESS`, adapter, checkpoint, validation metric, or
  scientific sample exists.

## Root cause

The dedicated trainer treated the generic Transformers
`gradient_checkpointing_enable()` API as mandatory. The installed LLaDA model
class explicitly advertises that it does not support that API. This is an
engineering compatibility error, not a data, objective, seed, or scientific
failure.

The actual frozen sequences have maximum tokenized length `152`; the run uses
one BF16 8B backbone, two small rank-8 adapters, and optimizer state only for
the policy LoRA. An 80GB A800 does not require activation checkpointing for the
frozen microbatch of two sequences.

## Authorized minimal recovery

Enable gradient checkpointing only when the model class reports support;
otherwise record it as disabled. Do not change the base/pair hashes, policy or
reference initialization, loss, beta, learning rate, LoRA configuration,
gradient accumulation, 348 updates, training seeds, checkpoint policy, GPU
count, or CPU count. Preserve job37966 and this report permanently, then allow
one recovery submission under the same scientific contract.
