# Efficiency-first Potential-Closure sprint plan

Status: **approved for preflight execution on 2026-09-04**. Phase 0, interface
implementation and train-only data construction may start immediately. Formal
DLM training starts only after the frozen launch conditions below are met.

## Approach

Test the only new mechanism that directly targets stability: train the existing
SPAD DLM to prefer lower-energy **complete lattice/XYZ transactions at fixed
composition**. Reuse C3FD, the trained Llama/pointer, SPAD, job39556 native
bodies and the tau800 fallback. Run the remote Phase 0 audit concurrently with
local interface implementation and target a native answer in 4.5--7.5 hours
before spending time on full MP20, Pointer DPO or low-tau refinement.

## Scope

### In

- C3FD legal composition support and current typed Planner;
- current Llama scaffold pointer and programs;
- exact `7+4N`, SPAD/MIC and suffix-visible revision;
- trained cell/site closure states;
- 2,048 K4 groups with raw CHGNet potential labels;
- closure-only and potential-closed training cells;
- one-stream native stability evaluation;
- tau800 fallback only after a native positive direction.

### Out of the first critical path

- Pointer DPO;
- full `27,136 x K4` labels;
- second training seed;
- tau200;
- new geometry adapter, residual head or vocabulary;
- model494-derived training labels;
- inference-time CHGNet, reranking or best-of-N;
- expensive nonessential Direct components.

## Frozen model responsibilities

| Component | Responsibility |
|---|---|
| C3FD | base composition distribution and chemical reachable support |
| Llama typed residual | learned preference inside the C3FD legal support; LS/SG/VPA |
| Llama pointer | equilibrium scaffold order used by SPAD |
| DLM/SPAD | all fine lattice/XYZ tokens and non-causal closure |
| CHGNet | offline raw potential label and diagnostic only |
| model494 tau800 | unchanged system fallback after native bodies are frozen |

## Formal training launch conditions

Preflight is not formal training. Submit the two 348-update training cells only
after all three decisions are available.

### A. Representation is usable

- all 9,047 MP20 validation rows encode and reconstruct through exact `7+4N`;
- quantization reduces Direct validity by at most 1 percentage point;
- median quantized-minus-continuous CHGNet energy is at most 15 meV/atom;
- force/stress distributions and all available cached stability flips are
  reported, with no unexplained nonfinite tail.

If the median energy penalty exceeds 15 meV/atom or Direct loses more than one
point, stop before training and revise representation precision. The force and
stress tails diagnose severity but do not create a hidden replacement rule.

### B. Closure actions are learnable

- exactly 2,048 groups exist, with 512 in each predeclared state stratum;
- every group has a legal no-op preserving exact composition;
- at least 1,800 groups have two or more distinct legal energy-known actions;
- raw CHGNet energy coverage is at least 98% of legal candidates;
- median within-group legal energy spread is at least 10 meV/atom;
- duplicate actions reconstruct identically and receive identical energies.

### C. Training and deployment states agree

- cell actions change only the complete six-token lattice block;
- site actions change only one complete XYZ block;
- visibility, mask, program order and no-op semantics match inference;
- three/six-token action scores use mean log probability per active token;
- clean CE and posterior losses are finite under the fixed 0.5/0.5 mixture;
- the initial policy is exactly the retained BS checkpoint.

Meeting A--C authorizes formal pilot training; it does not certify a positive
scientific result.

## Post-training continuation conditions

Run stream18 and frozen tau800 only when potential-closed versus closure-only on
stream17 shows:

- paired raw CHGNet mean improvement with a composition-bootstrap interval
  below zero;
- no adverse median force or stress direction;
- raw Direct loss no greater than 1 percentage point;
- NU loss no greater than 2 percentage points; and
- at least one of Strict or Meta S.U.N. improves on the fixed denominator.

Expand to full MP20 and two independent full-data seeds only if both streams
retain the raw-energy direction and pooled S.U.N. is positive. These are
predeclared continuation decisions; all unfavorable pilot outcomes remain
reported.

## Action items

- [ ] **Run one compact Phase 0 package.** In parallel, measure MP20-val
  continuous-to-token quantization effects, decompose existing BS raw
  stress/coordination errors, and audit train-only program/VPA/pair coverage.

- [ ] **Implement only three interfaces.** Add a six-token cell-closure action,
  generalize K4 scoring to three/six tokens with length normalization, and add
  the fixed 0.5/0.5 clean-CE/potential training schedule.

- [ ] **Freeze 2,048 train-only closure groups.** Use 512 groups in each of
  MP20-restoration-cell, MP20-restoration-site, on-policy-cell and
  on-policy-site. Reuse job39556 programs and BS predictor bodies for the
  on-policy half; never reuse its model494 terminal ranks.

- [ ] **Generate only missing K4 actions.** Each restoration group contains
  no-op, clean teacher and two DLM actions; each on-policy group contains no-op
  and three DLM actions. Every action is a complete lattice or XYZ block.

- [ ] **Label 8,192 raw candidates.** Reconstruct each full crystal, remove
  definitively invalid actions from support, and compute CHGNet energy/force/
  stress through the verified explicit `cuda:0` device path.

- [ ] **Train two concurrent cells.** From the same BS checkpoint and seed,
  train closure+clean-CE and potential-closed for 348 updates, one A800 each.
  The base DLM remains unchanged.

- [ ] **Run one fixed-stream native comparison.** Compare base, closure-only
  and potential-closed with identical Plan/program/request-site randomness.
  Compute raw energy/force/stress and cached S.U.N. first; run fast validity in
  parallel.

- [ ] **Continue only after native evidence.** If potential closure improves
  stability relative to closure-only without a material Direct/NU loss, run
  stream18 and frozen tau800. Only then expand full MP20 and two full-data
  seeds. Pointer DPO and tau200 remain later conditional work.

## Resource schedule and ETA

| Stage | Resources | Wall time | Cumulative |
|---|---:|---:|---:|
| Phase 0 package | 1 A800 + 8 CPU; CPU analyses parallel | 0.5--1.5 h | runs alongside code |
| Code and tests | CPU/local + remote smoke tests | 2--3 h | **2--3 h combined** |
| K4 generation | 4 A800 + 32 CPU | 0.3--0.8 h | 2.3--3.8 h |
| Raw CHGNet labels | 1 A800 + 8 CPU | 0.1--0.3 h | 2.4--4.1 h |
| Two training cells | 2 A800 + 16 CPU | 0.5--1.0 h | 2.9--5.1 h |
| Stream17 native generation/eval | 2--4 A800 + 16--32 CPU | 1.5--2.5 h | **4.5--7.5 h** |
| Conditional stream18 + tau800 | up to 6 A800 + 48 CPU | 2--3 h | **6.5--10.5 h** |

Maximum active use remains six A800 and two jobs. CHGNet uses batch 8--16 and
does not serialize structures one at a time unless a batch fails.

## Decision evidence

The first result answers one question:

> Does complete-transaction, same-composition potential learning improve the
> native DLM beyond closure training alone?

Use paired composition-level evidence:

- raw CHGNet energy interval and lower-energy fraction;
- force and stress direction;
- fixed-denominator raw Strict/Meta S.U.N.;
- Direct and N/U/NU on the same attempts;
- number and type of lattice/XYZ transactions actually changed.

An unfavorable result is retained. It stops full-data expansion; it does not
trigger another method, tau sweep or checkpoint selection.

## Expected result

The highest-confidence expectation is preservation of composition and most of
SPAD's structural validity. The intended new signal is lower raw energy and
stress because value is assigned to complete reconstructed cells/sites rather
than minimum-distance proxies or post-model494 outcomes. Raw Meta should move
before Strict; crossing both final `10%/50%` targets may require the full-MP20
two-seed expansion after this mechanism study.

## Paper use

If positive, the main story is one chain:

```text
C3FD chemical support
-> Llama scaffold program
-> non-causal SPAD crystal transactions
-> same-composition potential-closed DLM
-> optional frozen continuous refinement
```

The first two stages are preserved assets. Complete-transaction potential
closure is the sole new method in this sprint.
