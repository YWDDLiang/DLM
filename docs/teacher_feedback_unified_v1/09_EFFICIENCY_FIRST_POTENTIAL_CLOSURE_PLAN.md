# Efficiency-first Potential-Closure sprint plan

Status: **approved for preflight execution on 2026-09-04**. Phase 0, interface
implementation and train-only data construction may start immediately. Formal
DLM training starts only after the frozen launch conditions below are met.

Execution status: Phase 0 job `39596` completed in 37 seconds and authorized
the action-pool preflight. Formal potential-closure training job `39603` is
running.

Phase 0 results: `9,047/9,047` exact codec round trips; fixed-512 CHGNet pair
coverage `100%`; quantized-minus-continuous median energy `+2.846 meV/atom`
(`q05=-0.430`, `q95=+19.031 meV/atom`); frozen fast Direct `458 -> 458/512`;
Meta proxy retention `505/512 = 98.63%`. Strict exact-zero proxy retention is
`25/207` and remains diagnostic only because a few-meV proxy error flips an
exact-zero threshold.

Matched state freeze job `39597` completed: 512 outcome-blind MP20-train
sources yield exactly 512 groups in each of MP20-clean cell/site and on-policy
cell/site strata. Anchor ranks are balanced `259/253`; actual body species
order is used when mapping the Llama program to a site.

Candidate transaction job `39598` was reduced from 32 CPU/500G to 16 CPU/300G,
then failed before model/data/science because an intervening SSH reconnect had
cleared its submitted ROOT variables. Parameter-identical recovery job `39599`
completed in 16 minutes on four A800/16 CPU: `2004/2048` groups retain K>=2,
the variable-K histogram is `K1/K2/K3/K4=44/68/1041/895`, all four strata
remain 512, and 2,837 duplicate draws were merged with zero invalid draws.

Dependent label job `39600` failed before loading CHGNet because the wrapper
used Bash's reserved `GROUPS` variable. Commit `5c85531` renamed it; recovery
job `39601` completed in 95 seconds on one A800/8 CPU. All `6,883/6,883`
retained candidates have raw CHGNet E/F/stress; `2,003/2,048` groups are
informative, with strata `512/499/504/488`, and the median within-group energy
spread is `2.624 eV/atom`. Every action-pool condition passes. Five-batch
gradient probe job `39602` completed in 143 seconds and passed every condition:
cell/CE and site/CE gradient-norm median ratios are `0.109/0.160`, median
cosines are `0.0058/0.0572`, and every gradient is finite/nonzero with KL at
most `0.05 nat`. Formal dual-cell training job `39603` began on two A800/eight
CPUs with closure-only control and potential-closed sharing initialization,
seed, optimizer and the 2,048-update budget. The potential cell stopped at
update 580 because its float64-normalized K<=4 posterior was down-cast to
float32 before an unnecessarily strict sum check; this can move a valid sum by
one float32 ULP. The control cell continues unchanged. Commit `0798a05` keeps
the tiny categorical loss in float64, and its 12-test remote regression suite
passes. The control cell completed all 2,048 updates with the expected
`1024/512/512` objective counts and its sole step-2048 checkpoint; the parent
job then exited failed because the original potential process was already
ineligible. Parameter-identical potential-only recovery job `39604` completed
all 2,048 updates on one A800/four CPUs with the expected `1024/512/512`
objective counts, clean/on-policy transaction exposures and sole step-2048
checkpoint; no scientific setting or data changed. Native stream17 generation
job `39605` completed in 11 minutes 30 seconds on two A800/eight CPUs; both
arms are `256/256` parsed and Plan-matched. Batched raw evaluation job `39606`
completed in 100 seconds: every arm is composition-valid `256/256`; BS,
control and potential structural validity are each `255/256`. Potential minus
control raw CHGNet energy has mean `-0.236 eV/atom`, median approximately zero,
and `133/118/5` lower/higher/equal pairs; its composition-bootstrap mean 95%
interval is `[-0.599, +0.131]`, so the predeclared significance continuation
condition is not met despite a favorable point direction. At the user's
explicit request to obtain both raw and tau800 results, fixed tau800 diagnostic
job `39607` is running on four A800/16 CPUs; this does not promote or select the
method.

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
- 2,048 variable-K groups (K=2..4 when informative) with raw CHGNet potential
  labels;
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
- parsed structural validity is reported on the same fixed denominator;
- Meta S.U.N. paired wins are not fewer than losses; and
- neither Strict nor Meta shows a significant adverse one-sided exact McNemar
  result.

Always report paired energy deltas and Strict/Meta wins and losses, together
with composition and parsed structural validity. Skip Direct on this critical
path. Do not combine a fixed `wins-losses` cutoff with a p-value:
significance depends on the total number of discordant pairs.

Expand to full MP20 and two independent full-data seeds only if both streams
retain the raw-energy direction, the pooled composition-clustered interval is
below zero, pooled Meta wins exceed losses, and both validity rates remain
intact. These are predeclared continuation decisions; all unfavorable
pilot outcomes remain reported.

## Action items

- [x] **Run one compact Phase 0 package.** In parallel, measure MP20-val
  continuous-to-token quantization effects, decompose existing BS raw
  stress/coordination errors, and audit train-only program/VPA/pair coverage.

- [ ] **Implement only three interfaces.** Add a six-token cell-closure action,
  generalize the at-most-K4 scorer to three/six tokens with group loss divided
  by transaction length, and add objective-separated interleaved updates.

- [x] **Freeze 2,048 train-only closure groups.** Use 512 groups in each of
  MP20-restoration-cell, MP20-restoration-site, on-policy-cell and
  on-policy-site. Reuse job39556 programs and BS predictor bodies for the
  on-policy half; never reuse its model494 terminal ranks.

- [x] **Generate the fixed variable-K actions.** Each restoration group contains
  no-op, clean teacher and two DLM actions; each on-policy group contains no-op
  and three DLM actions. Use fixed temperature and at most eight proposals,
  retain the first distinct legal actions in request order, and never alter
  temperature or select by energy. Every action is a complete lattice or XYZ
  block.

- [x] **Label all 6,883 retained raw candidates.** Reconstruct each full crystal, remove
  definitively invalid actions from support, and compute CHGNet energy/force/
  stress through the verified explicit `cuda:0` device path.

- [x] **Probe gradients without updating weights.** On five fixed batches,
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
  Compute raw energy/force/stress, composition validity, parsed structural
  validity and cached S.U.N.; do not run Direct. Inference executes one trained
  cell closure followed by one XYZ
  closure for each of the first two distinct-species anchors in the Llama
  program, in reverse order. A unary composition revisits one anchor; no other
  site and no second cell closure are used.

- [ ] **Continue only after native evidence.** If potential closure improves
  stability relative to closure-only while retaining both validity rates, run
  stream18 and frozen tau800. Only then expand full MP20 and two full-data
  seeds. Pointer DPO and tau200 remain later conditional work.

## Resource schedule and ETA

| Stage | Resources | Wall time | Cumulative |
|---|---:|---:|---:|
| Phase 0 package | completed: 1 A800 + 8 CPU | 37 s | complete |
| Code and tests | CPU/local + remote smoke tests | 2--3 h | **2--3 h combined** |
| Fixed-8 proposal generation | completed: 4 A800 + 16 CPU | 16 min | complete |
| Raw CHGNet labels | completed: 1 A800 + 8 CPU | 95 s | complete |
| Five-batch gradient probe | completed: 1 A800 + 8 CPU | 143 s | complete |
| Control + potential recovery to 2,048 | completed | 2 h 19 min recovery wall time | complete |
| Stream17 native generation/eval (no Direct) | completed | generation 11.5 min + eval 100 s | complete |
| Tau800 diagnostic (no Direct) | active: 4 A800 + 16 CPU, two workers/GPU | about 15--45 min | running |

All future work is capped at four A800 and exactly four requested CPUs per GPU,
with at most two jobs. CHGNet uses batch 8--16 and does not serialize
structures one at a time unless a batch fails.

GPU occupancy is treated as an execution target rather than a new scientific
variable. DLM generation keeps batch size 8. CHGNet keeps batch size 16. The
conditional tau800 run uses four A800s and two independent, disjoint
sample-index-seeded refinement workers per GPU (four workers per arm across two
GPUs); this preserves each sample exactly while targeting at least 70% device
utilization. No sampling temperature, seed, Plan, checkpoint or denominator is
changed to improve utilization.

## Decision evidence

The first result answers one question:

> Does complete-transaction, same-composition potential learning improve the
> native DLM beyond closure training alone?

Use paired composition-level evidence:

- raw CHGNet energy interval and lower-energy fraction;
- force and stress direction;
- fixed-denominator raw Strict/Meta S.U.N.;
- paired Strict/Meta wins and losses;
- composition validity and parsed structural validity on the same attempts;
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
