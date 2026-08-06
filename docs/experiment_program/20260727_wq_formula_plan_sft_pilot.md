# WQ formula-plan / chemistry-aware SFT pilot

Date: 2026-07-27  
Run: `20260720_0401-crysllmgen-wq-final-v3`  
Status: authorized for one short continuation-SFT and one paired-64 proposal gate

## Why this is the next experiment

The charge-aware STOP pilot did not produce enough benefit. On the frozen
paired-64 panel, composition validity moved only from 61/64 to 62/64 and
charge-neutrality failures moved from 3 to 2, despite 37 STOP deferrals. This
shows that delaying termination cannot reliably repair a composition whose
species and counts were chosen incrementally.

The next intervention therefore moves chemistry before geometry. The model
first emits a complete primitive-cell formula plan, then emits a Wyckoff record
whose orbit multiplicities must consume that plan exactly. This keeps the ICLR
main claim centered on a language-model representation of crystallographic
structure. It introduces no MLIP guidance, no training-time CHGNet, no S.U.N.
feedback, no external API, and no post-generation repair.

## Frozen representation

The formula-plan grammar is:

```text
F=E=<species-code>,N=<primitive-count>;...;END
```

Species codes use the existing 1--89 MP20 vocabulary. Entries are strictly
increasing, unique, positive-count, and sum to at most 20 primitive atoms.

Planning is chemistry aware:

- `END` is available when the completed composition passes the same reduced
  SMACT/Pauling composition-validity rule used by CrysLLMGen metrics.
- If the plan is invalid and MP20 support remains, the decoder must add another
  species/count entry.
- If the 20-atom or 89-species support is exhausted, `END` is allowed so the
  attempt remains terminal and invalidity remains in the denominator.

The Wyckoff-body decoder is plan exact:

- a Wyckoff type is legal only when its primitive multiplicity fits at least one
  remaining planned species count;
- the species token is restricted to species with enough remaining count;
- each completed orbit subtracts its primitive multiplicity from that species;
- `STOP` is legal only after every planned count is exactly zero.

There is one plan call and one body call through the same adapter. There is no
retry, replacement, best-of-N, reranking, or repair.

## Training data and optimization

The only training source is the frozen MP20-train mixed WQ SFT JSONL:

```text
runs/20260720_0401-crysllmgen-wq-final-v3/
  outputs/sft_wq_mixed_seed11_ctx512_v3/coarse_sft.jsonl
```

Its 27,135 proposal examples are checked using the same reduced SMACT/Pauling
composition-validity definition used by the target metric. Invalid source
compositions are excluded from this chemistry-focused continuation only; the
count is reported rather than assumed. Each retained proposal is transformed
into:

1. one unconditional formula-plan example;
2. one formula-conditioned Wyckoff-body example;
3. one replayed legacy direct-edit example for every other retained structure.

The resulting task ratio is 2:2:1. No held-out generation, CrysLLMGen metric,
CHGNet result, S.U.N. result, Materials Project query, or prior pilot failure is
used as a training label.

Training is a short continuation from the already selected epoch-3 WQ adapter:

- 1 A800, 8 CPU, offline `diff_meets_diff`;
- 200 optimizer updates;
- microbatch 8, accumulation 8, global batch 64;
- maximum sequence length 640;
- LR `2e-5`, 5% warmup, constant-with-warmup;
- SDPA, BF16, gradient checkpointing off;
- one final adapter only.

This is a learnability/effect pilot, not another formal three-epoch run.

## Paired-64 effect gate

The baseline is reused byte-for-byte from job 28302; it is not regenerated.
Only 64 formula-plan attempts are generated for ordinals 1024--1087. Pair IDs
and the plan-stage sampling seed are inherited from the frozen baseline. The
body-stage seed is a deterministic stage-specific derivation.

Promotion requires all of:

- exact 64 baseline and 64 formula-plan denominators;
- at least 63/64 successful formula-plan generations;
- exact plan/body match for every successful generation;
- at least 63/64 composition-valid formula-plan outputs;
- at least +2 composition-valid samples versus the 61/64 baseline;
- no increase in charge-neutrality failures;
- at most one baseline-valid to formula-invalid paired regression;
- unique-formula/all-attempt rate at least 0.90;
- absolute mean primitive-atom-count shift at most 4.

Passing authorizes preparation—not automatic submission—of a larger
three-by-256 direct-metrics panel. Failing stops this arm for diagnosis; it does
not trigger a retry or another training run.

## Minimal record policy

Only four scientific records are retained:

1. this frozen design/contract and installed patch identity;
2. one exclusive Slurm claim and submission record;
3. one training report plus final adapter hashes;
4. one paired-64 terminal report and attempts hash.

Routine polling does not create audit sidecars. Unrelated queue jobs are only
recorded and never modified.
