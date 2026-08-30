# DLM stability 36-hour option checklist

Date: 2026-08-30  
Deadline: 2026-08-31 23:30 Asia/Shanghai  
Resources: at most six A800 GPUs total, 4--8 CPUs per active GPU, at most two
concurrent jobs; CPU-only immutable data work may use up to 48 cores.

## Approach

Run one confirmatory DLM method, not a post-result tournament. The recommended
main path is the smallest intervention directly supported by D3PO's replicated
post-refiner signal: full-sequence same-composition listwise alignment, with raw
validity and reference anchors. Keep one low-risk training fallback ready before
the prospective cohort is evaluated. Structural planning and self-intent remain
train/validation feasibility studies and cannot consume the final test in this
sprint.

## Scope

- **In:** C3FD composition+N, dynamic `7+4N`, exact full-axis, BASE step696,
  listwise train/chemsys-validation assets, two training seeds, model494 tau800,
  one prospective 256-composition evaluation.
- **Out:** AR, external rich Plan as the main method, direct Planner CIF,
  best-of-N/rerank/replacement, composition tilt, tau/temperature/checkpoint
  sweeps, RL/GRPO/SMC, full E(3)-equivariant backbone rewrite.

## Option summary

| ID | Option | Role | New model components | Data reuse | Estimated pre-eval resource | Expected value | Main risk | Decision |
|---|---|---|---|---|---|---|---|---|
| A | Minimal Listwise-Anchored DLM | **Primary** | none beyond existing LoRA/listwise loss | 886 train + 166 val groups, 3614 candidates | <=48 CPU builder; two A800 training seeds, about 1--2 wall hours | Directly targets same-composition energy while preserving raw validity | masked-sequence score remains a likelihood surrogate | Execute |
| B | Energy-weighted winner SFT | **Predeclared fallback** | none | one raw-valid lowest-refined-energy body per group + original BASE SFT | two A800 seeds, about 1 wall hour | Lowest engineering risk; directly imitates favorable complete CIFs | discards non-winner continuous information; may only give small Meta shift | Use only if A trainer/calibration is not ready within four hours |
| C | C3FD-style constrained Structural Planner | Parallel feasibility / next-cycle option | small typed heads + compatibility compiler | MP20 structure labels; current Planner backbone | CPU label audit; at most one A800 only after main path is safe | Can make coarse Plan internally executable and reduce DLM search space | composition-to-structure is multimodal; legal Plan is not stable Plan | Train/validation only in this sprint unless A is terminal and time remains |
| D | DLM self-intent VPA/CN | Diagnostic / deferred | two 8-way heads/embeddings | completed 27,136/9,047 label asset | zero-GPU predictability audit; two A800 if ever promoted | Compact internal plan keeps DLM central | VPA/CN may be too coarse and prediction may collapse | No GPU this sprint without clear incremental oracle and predictability evidence |
| E | DLM-to-refiner bridge correction | Conditional attribution | no generator update | reuse the exact raw bodies from A | one matched refine-only job, up to six A800 | Tests whether raw gain is erased by clean-body-as-`x_tau` mismatch | cannot rescue a generator with no raw signal | Run only if A has replicated raw left shift that tau800 erases/reverses |
| F | Periodic/E(3) graph adapter | Longer-term architecture | periodic equivariant adapter | MP20 + relaxation trajectories | exceeds sprint | Best principled route to spatial invariance | implementation/training risk cannot fit 36 hours | Document only; do not start |

## Option A — Minimal Listwise-Anchored DLM

### Frozen method

For every exact-composition group, retain all physically distinct `K=2..8`
candidates. Compute robust centered weights from post-model494 CHGNet energy.
Raw CIFs are parsed under the frozen Direct structural rule: minimum periodic
distance at least 0.5 Å and volume at least 0.1. Raw-invalid candidates are
lexicographically worst. The model receives one shared typed mask/noise per
group.

```text
L_A = L_centered_listwise
    + lambda_q * mean(S_theta^2)
    + 0.20 * L_best_raw_valid_denoising
    + lambda_base * L_BASE_masked_CE
```

No crystal-system, SG, VPA, CN, distance, or symmetry head is added. The DLM
learns by changing the likelihood of complete CIF bodies.

### Resources

- immutable listwise-safety builder: CPU-only, <=48 cores;
- calibration: fixed 64 train groups, no GPU grid;
- training: seeds 82017/82018, one A800 each, eight CPUs each, exactly 348
  updates, only step348;
- generation/refinement: six A800, 48 CPUs, six cells once;
- offline raw/refined evaluation: six A800, 48 CPUs once;
- official MP: CPU/API once.

### Success interpretation

- both seed-level raw energy directions favorable or at least non-adverse with
  Direct/NU retained;
- all four refined cells favorable and official hull concordant for a strong
  claim;
- CI crossing zero is replicated suggestive, not failure deletion;
- post-refiner-only gain is pipeline-aware but raw DLM remains unsolved.

## Option B — Energy-weighted winner SFT

For each composition, select exactly one candidate using a frozen order:

1. raw structurally valid before invalid;
2. lowest post-model494 CHGNet energy;
3. answer SHA tie break.

Train with fixed `0.20 winner CE + 0.80 original/base CE`, two seeds, 348
updates. No reward model, no sequence-ratio calibration, no checkpoint search.
It may replace A only before any prospective outcome is generated. A and B may
not both consume the same final cohort as competing claims.

## Option C — Constrained Structural Planner

Reuse the successful C3FD design pattern, not the failed text-rich Planner:

- typed probabilistic heads for crystal system, metric shape, volume interval,
  site multiplicity, and species-site counts;
- a constraint compiler for crystal-system/lattice consistency, allowed SG set,
  exact N/multiplicity completion, and exact species counts;
- output a calibrated distribution over underdetermined Plans, never exact
  coordinates or a direct CIF.

This can guarantee Plan consistency but not which polymorph is lowest energy.
It remains train/validation feasibility work and cannot delay A/B.

## Option D — Self-intent

The completed VPA/CN asset has 100% train/validation coverage and non-collapsed
entropy. That proves labels exist, not that composition predicts them or that
they improve energy. Run only CPU predictability/association analyses now. A
future GPU promotion requires both incremental oracle value and calibrated
self-prediction beyond majority baselines.

## Option E — Refiner bridge

The current pipeline places a clean DLM proposal directly at an intermediate
tau state although model494 training used forward corruption. Re-run only the
same A bodies under one valid forward-noise tau800 bridge if, and only if, A
shows replicated raw improvement that disappears after refinement. This is an
attribution result, not another generator arm.

## Atomic execution checklist

### Evidence and data lock

- [x] Audit R5C, H1-A2, R03, rich Planner, C3FD, CTV, SGTC, D3PO, and model494.
- [x] Freeze C3FD-v2.5, dynamic `7+4N`, full-axis, temperature 0.7, BASE
  step696, and model494 tau800.
- [x] Build and archive VPA_Q8/CN_ENV8 train/validation labels as diagnostic
  assets only.
- [ ] Build immutable `listwise_safety_v2` from the frozen 886/166 groups.
- [ ] Verify every candidate's answer SHA and source ordinal against frozen
  source provenance.
- [ ] Record raw CIF parse, minimum periodic distance, volume, structural
  validity, missingness, and the deterministic best-valid index.
- [ ] Freeze output JSONL/manifest/code/input/output SHA and prove the CLI has no
  main/sealed/prospective outcome argument.

### Method A implementation

- [ ] Connect the existing `listwise_alignment` loss to the LLaDA LoRA training
  loop without adding model heads.
- [ ] Enforce one shared typed geometry mask/noise for all candidates in a
  composition group.
- [ ] Add the quadratic reference bound, 0.20 best-valid anchor, and fixed BASE
  CE anchor mix.
- [ ] Add finite-loss, group-weight-one, no-cross-composition, and reference-
  equality tests.
- [ ] Run local and remote unit tests, pycompile, and a step0 adapter-equality
  canary.

### Calibration and fallback lock

- [ ] Calibrate reward temperature, quadratic coefficient, and BASE CE mix once
  on exactly 64 train groups by gradient-scale diagnostics.
- [ ] Freeze the resulting constants in MD/JSON before any prospective model
  generation.
- [ ] If Method A is not train-ready within four hours, freeze Method B and stop
  A implementation rather than expanding architecture.
- [ ] Ensure only A or B is authorized for the prospective cohort.

### Prospective cohort

- [ ] Freeze a new unused C3FD sampling ledger of exactly 256 Plans before GPU
  training.
- [ ] Prove exact-composition disjointness from MP20 train and D3PO main/sealed.
- [ ] Save Planner checkpoint/sampling seed, source ordinals, Plan JSONL SHA,
  N/arity/family/element distributions, and no-outcome-read marker.
- [ ] Amend the final training contract with the cohort SHA; GPU remains blocked
  until this step passes.

### Training

- [ ] Submit exactly one two-seed training job using at most two A800 GPUs and
  eight CPUs per GPU.
- [ ] Require step0 reference equality before optimizer step1.
- [ ] Run seeds 82017/82018 for exactly 348 updates; do not early-stop or select
  a seed/checkpoint.
- [ ] Retain only each seed's step348 policy and immutable training manifest.
- [ ] Report observed GPU-hours separately from requested scheduler ceiling.

### Evaluation

- [ ] Submit one six-cell BASE/policyA/policyB x streams17/18 generation/refiner
  job using all six A800 GPUs and 48 CPUs.
- [ ] Preserve every requested attempt; no retries, replacements, or survivor
  filtering.
- [ ] Submit one twelve-cell raw/refined Direct+full-CHGNet evaluation job using
  all six A800 GPUs and 48 CPUs.
- [ ] Report raw before refined: body, Direct C/S/J, N/U/NU, energy ECDF and
  paired composition deltas.
- [ ] Freeze one complete relative official-input manifest and execute at most
  one fresh MP query.
- [ ] Finalize two training seeds separately, streams averaged within
  composition, composition-cluster bootstrap, S.U.N., CI, and McNemar.

### Conditional and terminal actions

- [ ] Run Option E only if the preregistered interface-erasure pattern appears.
- [ ] Keep Options C/D CPU-only unless Method A/B is terminal and resources/time
  remain; do not let them consume the final cohort.
- [ ] Assign exactly one terminal class: Strong, Replicated Suggestive,
  Interface-only, Negative/Unstable, or Engineering.
- [ ] Archive positive and negative runs separately with contracts, hashes,
  logs, exit state, MD/JSON/CSV, and root cause.
- [ ] Update BUILD_STATUS/PAPER_STORY, run tests, commit, push, and write the
  single `36H_FINAL_REPORT`.

## Resource schedule

| Phase | Jobs | GPUs | CPUs | Expected wall time | Parallel work |
|---|---:|---:|---:|---:|---|
| safety data + trainer implementation | 1 CPU job + local work | 0 | <=48 | 2--4 h | C/D audits only if they do not block |
| two-seed training | 1 | 2 | 16 | 1--2 h | CPU reporting/tests |
| generation + model494 | 1 | 6 | 48 | about 1--2 h based on prior 256 runs | none |
| raw/refined offline evaluation | 1 | 6 | 48 | about 2 h | report scaffolding |
| official/finalization | 1 CPU/API process | 0 | <=8 | 1--3 h | archive/docs |

Unused GPUs remain unused until a frozen, tested scientific stage needs them;
the six-GPU allowance is a ceiling, not a reason to launch multiple methods.

## Discussion decision

Recommended portfolio: execute A; keep B ready as the sole pre-outcome fallback;
perform C/D only as non-blocking feasibility work; reserve E for a specific
observed interface-erasure pattern; defer F. Any change to which of A or B owns
the prospective cohort must occur before its first outcome and be recorded in
the contract.

