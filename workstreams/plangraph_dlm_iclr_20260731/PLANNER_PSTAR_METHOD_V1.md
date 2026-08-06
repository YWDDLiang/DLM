# H1-A2 Look-Ahead Consistent Planner V1

Status: `method_frozen_local_implementation_complete_remote_smoke_pending`

Frozen: 2026-08-01

## Purpose

Improve the model-generated H1-A2 Planner while preserving the exact
fully-de-novo interface. The method targets dependencies among already-visible
seven-line fields during training; it never repairs or constrains sampled
Plans at inference.

Method name: `h1a2_lookahead_consistent_planner_v1`

## Frozen identity and inference

- initialization: frozen H1-A2 epoch-2 adapter SHA-256
  `65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a`;
- base: frozen Meta-Llama-3-8B identity used by H1-A2;
- prompt, tokenizer, chat template, parser, and output schema: identical to
  P0;
- exact visible output:

```text
formula: ...
anion: ...
charge: ...
lattice: ...
spacegroup: ...
volume: ...
end: plan
```

- sampler: temperature 0.9, top-p 0.95, top-k 50, maximum 96 new tokens;
- no sample ID, retry, replacement, repair, filter, rerank, beam/candidate
  search, or model-visible PlanGraph.

All visible Plan values remain model generated.

## Arms

### P0

Frozen H1-A2 epoch-2. No training.

### P-control

Same initialization, training rows, microbatch order, optimizer, update
budget, validation panel, and field-balanced target-only loss as P*. It omits
only the look-ahead auxiliary loss.

### P*

Uses the P-control field-balanced target-only loss plus training-only
look-ahead classification heads.

P-control is a continuation/mechanism control, not a factorial arm and not a
fallback new method.

## Data

- source: original H1-A2 train split only;
- no generated Plans or crystals;
- no ValidReplay inclusion/exclusion rule;
- no chemistry-negative or joint-negative sequences;
- no S.U.N., energy, hull, MP API, CHGNet, or MLIP label;
- common train stream: exactly 3,200 microbatches, deterministically selected
  from the original train split by an external row-SHA ledger;
- selection must be proportional across atom count, arity, anion, charge,
  lattice, space-group, and volume categories;
- P-control and P* consume the byte-identical ordered stream;
- validation panel: 256 frozen original validation rows, deterministically
  stratified and shared by P0/P-control/P*.

The stream builder and exact balancing algorithm must be frozen and unit
tested before any model is loaded.

Every selected row contributes one equally weighted microbatch. Any historical
`sample_weight` field is retained only as source provenance and is not consumed
by the P-control/P* objective.

## Field-balanced target loss

Only answer tokens are supervised. Prompt tokens always have label `-100`.
Answer-token spans are derived from the exact fast-tokenizer offset mapping;
failure to map every supervised token to exactly one field is fatal.

Field groups and weights:

| Group | Lines | Weight |
|---|---|---:|
| formula | formula | 0.35 |
| chemistry | anion, charge | 0.25 |
| geometry | lattice, spacegroup, volume | 0.35 |
| terminator | end | 0.05 |

Within a group, token NLL is averaged before applying the group weight.
Weights sum to 1.0. A missing or empty field span is a fatal data error.

Define the common field-balanced loss as `L_field`.

## Look-ahead auxiliary loss

Training-only heads read causal hidden states at two registered boundaries:

1. the last token of the formula line predicts the target categories
   `anion`, `charge`, `lattice`, `spacegroup`, and `volume`;
2. the last token of the lattice line predicts target `spacegroup` and
   `volume`.

Each category vocabulary is the lexicographically sorted set observed in the
frozen original training split. Unknown validation labels are fatal. Each head
is a single affine projection from the final-layer hidden state. The seven
cross-entropies are averaged to form `L_lookahead`.

P* loss:

```text
L_Pstar = 0.80 * L_field + 0.20 * L_lookahead
```

P-control loss:

```text
L_control = L_field
```

Auxiliary heads are initialized with seed 17, saved in checkpoint audit
inventories, and discarded for inference. Only the LoRA adapter changes
inference behavior. The heads cannot score or filter sampled Plans.

Local implementation:

- `crystal_dlm/h1a2_planner_objective.py`;
- `crystal_dlm/h1a2_planner_batch.py`;
- `scripts/llama_h1a2_lookahead_sft.py`.

Implementation status does not satisfy the execution gate: the differentiable
Torch objective, real tokenizer offsets, and full training loop still require
the registered one-A800 32/32 numerical smoke.

## Training contract

| Parameter | Frozen value |
|---|---:|
| Precision | BF16 |
| Planner GPUs per arm | 1xA800 |
| CPU limit | 8 |
| Per-device batch | 1 |
| Gradient accumulation | 8 |
| Optimizer updates | 400 |
| Learning rate | 2e-6 |
| Weight decay | 0 |
| Scheduler | cosine |
| Warmup | 25 updates |
| Gradient clip | 1.0 |
| Maximum sequence length | 768 |
| Validation cadence | 50 updates |
| Seed | 17 |

No learning-rate fallback is authorized after scientific training begins. A
32-row engineering smoke may stop the launch for implementation or numerical
failure, but may not choose a method based on sampled Plan quality.

## Checkpoint selection

P0 is evaluated once on the same 256-row panel.

- P-control: among checkpoints with target NLL no worse than +1% relative to
  P0, select the lowest `L_field`; ties choose the earlier checkpoint.
- P*: apply the same P0 NLL noninferiority gate, then select the lowest
  `L_field`; ties choose the earlier checkpoint.
- `L_lookahead` and per-head accuracies are reported as mechanism diagnostics
  but do not override worse target likelihood.
- no autoregressive sample, crystal generation, S.U.N., energy, or hull result
  participates in checkpoint selection.

## Single Plan-only screen

After checkpoint selection, sample P0/P-control/P* once on a common
512-ordinal ledger. P* is eligible only if:

- parse/completion drop versus P0 is at most 0.5 percentage points;
- composition validity is at least 2 points above P0;
- composition validity strictly exceeds P-control;
- target NLL remains within +1% of P0;
- unique-formula rate is at least 95% of P0;
- mean atom count differs from P0 by at most 0.5;
- no registered marginal TVD worsens by more than 0.02;
- all-metal and single-element shortcut rates do not inflate.

The selected checkpoint and method are never changed after seeing this
screen. Failure is a scientific stop for the Planner axis.

## Exclusions

- previous generated-JSON PlanGraph checkpoints;
- ValidReplay and JointChem checkpoints;
- synthetic invalid-sequence ranking;
- generated-output replay or self-training;
- inference-time chemistry constraints, replacement, or repair;
- any refiner or body-DLM change during Planner selection.
