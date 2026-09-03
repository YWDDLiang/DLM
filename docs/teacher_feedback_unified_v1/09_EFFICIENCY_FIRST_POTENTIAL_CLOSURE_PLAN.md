# Efficiency-first Potential-Closure sprint plan

Status: **approved for preflight execution on 2026-09-04**. Phase 0, interface
implementation and train-only data construction may start immediately. Formal
DLM training starts only after the frozen launch conditions below are met.

Execution status: Phase 0 is running as Slurm job `39596` (one A800, eight
CPUs). Formal potential-closure training has not started.

## Approach

Test the only new mechanism that directly targets stability: train the existing
SPAD DLM to prefer lower-energy **complete lattice/XYZ transactions at fixed
composition**. Reuse C3FD, the trained Llama/pointer, SPAD, job39556 native
bodies and the tau800 fallback. Run the remote Phase 0 audit concurrently with
local interface implementation and target a native answer in 6--11 hours
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

Preflight is not formal training. Submit the two 2,048-update training cells only
after all three decisions are available.

### A. Representation is usable

- all 9,047 MP20 validation rows encode and reconstruct through exact `7+4N`
  with exact composition and parse preservation;
- fast structural validity on a fixed stratified 512-row subset loses at most
  1 percentage point; use cached full Direct when available, but do not wait
  for expensive uncached Direct submetrics before the stability preflight;
- on the same fixed stratified 512-row subset, median quantized-minus-continuous
  absolute median CHGNet energy shift is at most 15 meV/atom with paired
  E/F/stress coverage at least 98%;
- force/stress distributions and all available cached stability flips are
  reported, with no unexplained nonfinite tail.

If the absolute median energy shift exceeds 15 meV/atom or fast structural validity
loses more than one point, stop before training and revise representation
precision. On the subset
with compatible cached stability labels, define

\[
R_{\rm retain}=
\frac{\#(\text{continuous stable and quantized stable})}
{\#(\text{continuous stable})}.
\]

Report retention at both the Strict (`0`) and Meta (`0.1 eV/atom`) thresholds.
Because adding a proxy energy delta to the exact-zero DFT threshold is
especially noise-sensitive, the formal blocker is Meta retention: if it is
below `0.60`, stop the pure-token trainer and revisit representation precision.
Strict retention remains a mandatory diagnostic. The force and stress tails
diagnose severity but do not create a hidden replacement rule.

### B. Closure actions are learnable

- exactly 2,048 groups exist, with 512 in each predeclared state stratum;
- every group has a legal no-op preserving exact composition;
- after deduplication, at least 1,024 groups have two or more distinct legal
  energy-known actions, with at least 256/512 informative groups in every
  predeclared stratum;
- raw CHGNet energy coverage is at least 98% of legal candidates;
- every informative group has legal energy spread at least 1 meV/atom;
- duplicate actions reconstruct identically and receive identical energies.

Candidate generation uses the frozen BS temperature and at most eight
request-keyed proposal attempts. Keep the first distinct legal actions in
proposal order, never according to energy. Identical actions are merged; a
missing slot remains absent rather than being filled by dynamic temperature or
additional search.

### C. Training and deployment states agree

- cell actions change only the complete six-token lattice block;
- site actions change only one complete XYZ block;
- visibility, mask, program order and no-op semantics match inference;
- three/six-token action scores use the sum of conditional token log
  probabilities, and each group loss is divided by its transaction length;
- clean CE, cell-posterior and site-posterior losses/gradients are independently
  finite and nonzero on five fixed dry batches;
- posterior/CE trainable-gradient median norm ratios lie in `[1e-2, 1e2]` and
  CE-posterior median cosine similarities exceed `-0.5`;
- all finite target posteriors satisfy the fixed `0.05 nat` KL budget;
- the initial policy is exactly the retained BS checkpoint.

Gradient probes never change a loss weight, LR, KL budget or epoch count. A
failed probe diagnoses an implementation/data error and blocks training.

Meeting A--C authorizes formal pilot training; it does not certify a positive
scientific result.

## Post-training continuation conditions

Run stream18 and frozen tau800 only when potential-closed versus closure-only on
stream17 shows:

- composition validity remains at least 95%;
- paired raw CHGNet median improvement is below zero and the composition-
  bootstrap 95% interval for the mean lies below zero;
- raw Direct loss no greater than 1 percentage point;
- Meta S.U.N. paired wins are not fewer than losses; and
- neither Strict nor Meta shows a significant adverse one-sided exact McNemar
  result.

Always report paired energy deltas, Direct/Strict/Meta wins and losses, and
exact McNemar tests. Do not combine a fixed `wins-losses` cutoff with a p-value:
significance depends on the total number of discordant pairs.

Expand to full MP20 and two independent full-data seeds only if both streams
retain the raw-energy direction, the pooled composition-clustered interval is
below zero, pooled Meta wins exceed losses, and Direct retains its 1-point
non-inferiority. These are predeclared continuation decisions; all unfavorable
pilot outcomes remain reported.

## Action items

- [ ] **Run one compact Phase 0 package.** In parallel, measure MP20-val
  continuous-to-token quantization effects, decompose existing BS raw
  stress/coordination errors, and audit train-only program/VPA/pair coverage.

- [ ] **Implement only three interfaces.** Add a six-token cell-closure action,
  generalize the at-most-K4 scorer to three/six tokens with group loss divided
  by transaction length, and add objective-separated interleaved updates.

- [ ] **Freeze 2,048 train-only closure groups.** Use 512 groups in each of
  MP20-restoration-cell, MP20-restoration-site, on-policy-cell and
  on-policy-site. Reuse job39556 programs and BS predictor bodies for the
  on-policy half; never reuse its model494 terminal ranks.

- [ ] **Generate only missing K4 actions.** Each restoration group contains
  no-op, clean teacher and two DLM actions; each on-policy group contains no-op
  and three DLM actions. Use fixed temperature and at most eight proposals,
  retain the first distinct legal actions in request order, and never alter
  temperature or select by energy. Every action is a complete lattice or XYZ
  block.

- [ ] **Label 8,192 raw candidates.** Reconstruct each full crystal, remove
  definitively invalid actions from support, and compute CHGNet energy/force/
  stress through the verified explicit `cuda:0` device path.

- [ ] **Probe gradients without updating weights.** On five fixed batches,
  separately backpropagate clean CE, cell posterior and site posterior through
  the BS trainable LoRA, record norms/cosines, clear gradients, and apply the
  frozen launch conditions without automatic reweighting.

- [ ] **Train two concurrent cells.** From the same BS checkpoint and seed,
  train closure+clean-CE and potential-closed, one A800 each. Use one fixed
  four-step cycle: clean MP20 CE, cell transaction, clean MP20 CE, site
  transaction. Repeat 512 cycles for 2,048 optimizer updates. Cell/site
  posterior each consume three group epochs; on-policy states receive posterior
  only. Use 100 warmup updates, LR `5e-6`, and save only update 2048. The base
  DLM remains unchanged.

- [ ] **Run one fixed-stream native comparison.** Compare base, closure-only
  and potential-closed with identical Plan/program/request-site randomness.
  Compute raw energy/force/stress and cached S.U.N. first; run fast validity in
  parallel. Inference executes one trained cell closure followed by one XYZ
  closure for each of the first two distinct-species anchors in the Llama
  program, in reverse order. A unary composition revisits one anchor; no other
  site and no second cell closure are used.

- [ ] **Continue only after native evidence.** If potential closure improves
  stability relative to closure-only without a material Direct/NU loss, run
  stream18 and frozen tau800. Only then expand full MP20 and two full-data
  seeds. Pointer DPO and tau200 remain later conditional work.

## Resource schedule and ETA

| Stage | Resources | Wall time | Cumulative |
|---|---:|---:|---:|
| Phase 0 package | 1 A800 + 8 CPU; CPU analyses parallel | 0.5--1.5 h | runs alongside code |
| Code and tests | CPU/local + remote smoke tests | 2--3 h | **2--3 h combined** |
| Fixed-8 proposal generation | 4 A800 + 32 CPU | 0.6--1.2 h | 2.6--4.2 h |
| Raw CHGNet labels | 1 A800 + 8 CPU | 0.1--0.3 h | 2.7--4.5 h |
| Five-batch gradient probe | 1 A800 + 8 CPU | 0.2--0.4 h | 2.9--4.9 h |
| Two 2,048-update training cells | 2 A800 + 16 CPU | 1.5--2.5 h | 4.4--7.4 h |
| Stream17 native generation/eval | 2--4 A800 + 16--32 CPU | 1.5--2.5 h | **6--11 h** |
| Conditional stream18 + tau800 | up to 6 A800 + 48 CPU | 2--3 h | **8--14 h** |

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
- paired Direct/Strict/Meta wins, losses and exact McNemar tests;
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
