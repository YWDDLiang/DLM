# CrysLLMGen-WQ Closed-Loop Revision: Revised ICLR Experiment Plan

Date: 2026-07-20

## 1. Paper target and claim

The active paper remains restricted to de novo crystal generation.  Its main
claim is no longer that a discrete diffusion language model is superior.  The
candidate claim is:

> Starting from CrysLLMGen's Llama-to-diffusion pipeline, replacing its
> atom-wise one-way interface with a symmetry-native Wyckoff quotient and
> geometry-triggered, dimension-changing feedback produces better and more
> stable de novo crystals.

The intended comparison is a direct closed-loop extension of the official
CrysLLMGen one-way handoff.  Llama and CSPDiffusion are not claimed as new.  The
contributions are the symmetry quotient interface, changing Wyckoff strata,
target-stratum bridges, and geometry-to-topology revision.

Working method name: `C-WQ-GEOREV`.  Do not use `co-diffusion` to imply that
the AR branch is a diffusion process.  CrysLLMGen is the engineering and
experimental parent, not merely a literature baseline.

## 2. Model-asset decision

The primary checkpoint is now fixed to the user-supplied instruction model:

- `/public/home/jiaosz/hengzhang/models/LLM-Research/Meta-Llama-3-8B-Instruct/`;
- four real safetensor shards, 16,060,556,376 bytes in total;
- `LlamaForCausalLM`, BF16, 32 layers, hidden size 4096, context limit 8192;
- official tokenizer chat template present, with `<|begin_of_text|>` and
  `<|eot_id|>` boundaries;
- configuration SHA256
  `61f3de03a16ca8046b05dc777bce72717022bc8522152eee61a69072272ef54b` and
  weight-index SHA256
  `146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b`.

The path was verified read-only from the existing server session on
2026-07-20: all indexed shards exist and each is multi-GB rather than a Git-LFS
pointer.  The complete per-file preflight identity is recorded in
`configs/experiments/wyckoff_codiffusion/model_asset_preflight_meta_llama3_8b_instruct_v1.json`.
Its current SHA256 is
`6785a879c804338a9a38b8303f2ecbfc783392f04f21028e84be000b6fcee5f0`.
A compute-node permission check and Slurm offline forward remain Gate A
requirements before any training job.

`/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/` is retained only as a
registered base-model ablation/fallback asset.  It is not the primary route and
cannot silently replace the instruction checkpoint.  The similarly named
`/public/home/jiaosz/ywliang/models/Llama-3.1-8B-Instruct/` remains unusable:
its four apparent weight shards are only 135-byte Git-LFS pointers.

The earlier H1 result therefore remains a historical interface diagnostic, not
a verdict on this checkpoint: base-model SFT loss fell normally to 1.1245, but
unconstrained free-form chat parsing reached only 66%.  The revised protocol
uses the official instruction serialization while making the assistant answer
a finite-state-constrained crystal grammar, eliminating free-form output.

Every model file, tokenizer, config, license, byte count, and SHA256 must be
frozen before the first GPU job.  GPU jobs remain strictly offline.

The inherited MP20 continuous checkpoint is also present and registered:

- path:
  `/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt`;
- size: 147,645,242 bytes;
- SHA256:
  `573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e`;
- preflight record:
  `configs/experiments/wyckoff_codiffusion/model_asset_preflight_crysllmgen_mp20_v1.json`.

It is the same checkpoint historically used for R5-C 800-step CrysLLMGen
refinement.  Compute-node loading and state-dict shape mapping remain Gate A;
its existence is no longer an unresolved blocker.

Two existing MP20 LoRA adapters were also found under the upstream execution
tree.  Both point to the registered instruction backbone, but both are rank-8
q/v-only adapters: `8b-mp-llama3-v1` was trained for five epochs/1,215 steps,
and `llama3-mp-20` for one epoch/848 steps.  Gate A selects and hashes the exact
adapter used for `C-ATOM-OFFICIAL`.  Neither can silently stand in for
`C-ATOM-MATCHED`, because the latter is an equal-capacity representation
control against the rank-16 seven-projection WQ adapter.

## 3. CrysLLMGen engineering base

The active implementation is derived from the official local repository
`reference/crysllmgen/` at commit
`94bb287751cd20a882c7c1df7ca736633d78e5e1` under its MIT license.  The
immutable reference remains untouched.  Derived code is placed under
`crystal_dlm/wqcodiff/crysllmgen/`; its exact 26-file source snapshot is under
the `upstream/` child and is bound by `UPSTREAM_MANIFEST.json`.  Active derived
modules are added beside that child, while the existing Wyckoff state, chart,
bridge, attempt, audit, and evaluation modules are reused as extensions.

The fork preserves these CrysLLMGen components:

- `llm_finetune.py` and the LoRA training/data flow;
- `crysllmgen_sample.py` and the proposal-to-refinement orchestration;
- the six-layer `models_ddpm/cspnet.py` atom backbone;
- `CSPDiffusion`, its beta/sigma tables, and predictor-corrector sampler;
- MP20 preprocessing plus official `compute_metrics.py` definitions.

The complete source mapping and parity contract are frozen in
`docs/experiment_program/CRYSLLMGEN_FORK_MAP.md`.  Before new features are
enabled, a disabled-extension mode must reproduce the upstream atom pipeline on
256 hash-fixed proposals.  The previous from-scratch `WQCoDenoiser` is no
longer the primary paper backbone.

CrysLLMGen's generate-until-parseable loop is deliberately not preserved: one
requested sample is one terminal attempt.  An invalid Llama proposal is an
attempt failure and is never replaced.  This accounting change is applied to
the upstream control and every proposed method.

## 4. CrysLLMGen-compatible Wyckoff grammar

### 4.1 Initial coarse crystal proposal

There is no global `MASK` initial state.  As in CrysLLMGen, Llama proposes both
discrete content and a coarse continuous structure.  The atom-wise lattice,
species, and coordinate text is replaced by:

```text
BOS
SG=<1..230>;
CELL=(Q=<crystal-system-valid 1..6D lattice chart>);
ORB=(O=<orbit_id>,W=<space-group-valid Wyckoff id>,E=<element>,U=<0..3D free coordinate>);
...
STOP
```

The exact byte format and tokenizer trie are frozen in protocol v4.  Existing
Llama tokens encode the compact ASCII form; no vocabulary resize is placed on
the critical path.

Generation rules are enforced before probability normalization:

- `SG` is the first semantic field and is fixed after emission;
- the lattice chart dimension and constraints are fixed by the emitted SG;
- at least one orbit must be emitted before `STOP`;
- each Wyckoff id must be legal in the emitted space group;
- every `U` has exactly the chart dimension of its Wyckoff orbit;
- accumulated multiplicity must remain in the MP20 range 1--20 atoms;
- no padding token is a semantic orbit or NULL canvas;
- invalid next tokens have exactly zero probability;
- a bridge or parser failure is terminal and is never resampled.

The orbit multiset is unordered.  Every training presentation receives a
SeedDeriver-controlled random orbit permutation.  Canonical order is used only
for storage, hashing, and evaluation.  A canonical-order ablation is run on
validation data only.

The parsed proposal becomes a `StratifiedState`, is expanded through the
PyXtal/Wyckoff affine maps, and enters the inherited CrysLLMGen refinement at a
registered intermediate timestep.  It does not discard Llama's coarse
geometry and restart continuous diffusion from random noise.

### 4.2 Direct edit grammar

Committed topology is revised through explicit commands, not by restarting
from all-MASK:

```text
NO_OP
BIRTH W=<id> E=<element>
DEATH O=<orbit_id>
CHANGE_TYPE O=<orbit_id> W=<id>
CHANGE_SPECIES O=<orbit_id> E=<element>
```

`CHANGE_TYPE` has death-plus-birth semantics.  Llama sees the current serialized
topology plus quantized geometry evidence for the affected orbits.  It emits at
most one command per invocation.  The target-stratum bridge creates any newly
introduced free-coordinate chart before continuous denoising continues.

`MASK` may remain in an auxiliary corruption-recovery task, but it is absent
from de novo inference semantics and absent from paper diagrams of the initial
state.

## 5. CrysLLMGen-derived coupled architecture

### 5.1 Llama coarse proposer and editor

- Backbone: frozen `Meta-Llama-3-8B-Instruct` BF16 weights at the registered
  user-supplied path; the base model is not substituted during the cycle.
- Adaptation: LoRA rank 16, alpha 32, dropout 0.05 on q/k/v/o and
  gate/up/down projections.
- Maximum sequence length: 256 tokens.
- Prompt serialization: the checkpoint's official chat template with one
  frozen minimal system instruction and a structured user payload.  No ad-hoc
  `System/User/Assistant` plaintext template is allowed.
- Training target: loss only on the compact structured assistant answer;
  finite-state decoding and forced `STOP` remain mandatory, so instruction
  tuning never creates a free-form generation surface.
- One shared adapter is trained on a mixture of initial-generation and edit
  commands.  All WQ no-edit/edit controls use this identical checkpoint.
- PyTorch SDPA is the default; no new FlashAttention dependency is required.
- The active training and sampling entry points are ports of CrysLLMGen's
  `llm_finetune.py` and `crysllmgen_sample.py`; hard-coded hub login, implicit
  network resolution, and unrestricted generate-until-valid behavior are
  removed.

### 5.2 Wyckoff wrapper around CrysLLMGen CSPDiffusion

The continuous refiner starts from the inherited CrysLLMGen implementation:

- six-layer CSPNet, hidden size 512, fully connected periodic edges, sinusoidal
  distance features, and lattice inner-product conditioning;
- the original cosine beta schedule, wrapped-coordinate sigma schedule, and
  two-call predictor-corrector reverse step;
- the original Llama-proposal-to-`diff_steps` injection interface.

For a WQ state, the affine Wyckoff map expands orbit free coordinates to the
atom graph consumed by CSPNet.  Its atom-coordinate score is projected through
the regularized Wyckoff Jacobian pseudoinverse back to the free-coordinate
tangent.  The lattice output is projected to the SG-compatible lattice chart.
Small orbit-pooling, evidence, trigger, and bridge heads may be added, but the
shared CSP layers and sampler remain checkpoint-compatible with CrysLLMGen.

If `torch_scatter` or PyG cannot be used in the registered environment, their
fully connected edge and scatter operations are ported to native PyTorch only
after tensor-level parity below `1e-6`; the identical port is used in every
matched method.  This is an engineering substitution, not a contribution.

The six geometry signals remain collision deficit, coordination anomaly,
lattice strain, symmetry residual, score norm, and basin uncertainty.  Llama
receives preregistered 16-bin versions of these signals.  The continuous gate
uses the unquantized values.

### 5.3 Runtime loop and call accounting

1. Llama generates one coarse WQ crystal proposal.
2. The proposal is parsed once, expanded to atoms, and injected into the
   inherited CrysLLMGen refinement at the frozen `tau`/`diff_steps`.
3. CSPDiffusion performs the registered predictor-corrector schedule while the
   small geometry gate is evaluated after each reverse step.
4. Llama is called only when the frozen gate triggers an edit.
5. The edit is applied once; a target-stratum bridge initializes new geometry,
   the state is re-expanded, and the same CrysLLMGen sampler resumes.

`C-ATOM-OFFICIAL` reproduces the upstream MP20 setting of 800 reverse steps,
which entails 1,600 CSPNet forwards.  Primary matched comparisons use a frozen
respaced schedule with 64 total CSPNet forwards; all methods share the same
timesteps and random draws.  The direct CrysLLMGen injection rule is the main
matched rule.  A correctly forward-noised injection is reported only as a
pre-registered sensitivity analysis and cannot replace a failed main result.

The 8B model is never called at every diffusion step.  Per field there are at
most two edits, each reverse step has at most one topology event, and total edit
commands are capped at half the initial field count.  Report Llama prompt
tokens, generated tokens, forwards, FLOPs, wall time, and GPUh separately from
continuous-backbone calls.

## 6. Training contract

The frozen training inventory is
`configs/experiments/wyckoff_codiffusion/training_evaluation_inventory_v1.json`.
There are three main trainable families: matched atom LoRA, shared WQ
proposal/edit LoRA, and the shared WQ CSP refiner.  Each has seeds 11/23/47,
for nine main runs.  One seed-11 canonical-order WQ LoRA is trained only for
the permutation-presentation ablation.  The maximum scheduled total is
therefore **10 training runs**, not ten unrelated architectures.  Proposal and
edit curricula are sequential stages of the same WQ adapter; warm-up/joint/
revision phases are stages of the same WQ refiner; all inference controls reuse
those checkpoints.

### 6.1 Llama coarse-proposal task

- Train split only: 27,135 selected P1 structures.
- Three fixed epochs for screening; one new random orbit permutation per epoch.
- Effective batch: 64 sequences.
- AdamW, LoRA LR `1e-4`, weight decay `1e-2`, 3% warmup, cosine decay.
- BF16, gradient checkpointing, gradient clip 1.0.
- Final epoch checkpoint is used; no MLIP or test-set checkpoint selection.
- Screening seed 11; final seeds 11/23/47.

Targets contain the complete coarse WQ proposal: SG, lattice chart, orbit
types/species, and free coordinates.  Atom and WQ serializers use the same
training structures, prompt budget, model, and number of optimizer tokens.
`C-ATOM-MATCHED` is trained with the same rank, target modules, optimizer-token
budget, and seeds as the WQ adapter.  The rank-8 upstream adapter remains
exclusive to `C-ATOM-OFFICIAL`.

### 6.2 Mixed edit curriculum

Continue the same LoRA adapter while retaining 50% initial-generation examples
to prevent forgetting.  The other 50% are minimal edit commands generated from
train structures under:

- deletion, false insertion, wrong Wyckoff, wrong species, and joint errors;
- corruption levels 30/50/70/90%;
- clean, noisy, shuffled, and absent geometry evidence;
- direct event targets with no best-of or alternative valid target selection.

The edit stage is fixed to 20,000 optimizer updates at effective batch 64.  A
fixed hash chooses the corruption/operator/evidence mix for every attempt.

### 6.3 CrysLLMGen-initialized continuous training

- `C-ATOM-OFFICIAL` and `C-ATOM-MATCHED` load the same registered
  CrysLLMGen diffusion checkpoint and do not use SUN to select or retrain it;
- the WQ refiner imports all compatible CSPNet weights from that checkpoint and
  records every mapped, newly initialized, and rejected parameter;
- 20,000 WQ chart/projection/bridge warm-up updates with inherited CSP layers
  frozen, followed by 40,000 shared joint-geometry updates;
- 40,000 event, revision, and coupled-geometry updates, for 100,000 total WQ
  optimizer updates;
- effective batch 128 structures;
- AdamW LR `2e-4`, weight decay `1e-2`, 5% warmup, cosine decay;
- EMA 0.999, gradient clip 1.0;
- fixed endpoint checkpoint.

If the previously used CrysLLMGen MP20 diffusion checkpoint is absent from the
authorized server model/run locations, the cycle stops and asks the user.  It
is not silently replaced by a newly selected or downloaded checkpoint.

Before full training, the component-gradient audit must show finite gradients
for every supervised head and report each component's share.  A dominant
geometry branch cannot be silently accepted or tuned using SUN.

## 7. Main experiment matrix

### 7.1 Primary and upstream-parity methods

| ID | Representation and refinement | Topology feedback | Purpose |
|---|---|---|---|
| `C-ATOM-OFFICIAL` | official atom text + official 800-step CrysLLMGen | none | upstream reproduction |
| `C-ATOM-MATCHED` | attempt-compliant atom text + respaced CrysLLMGen | none | matched engineering/compute control |
| `C-WQ-HANDOFF` | WQ coarse proposal + WQ-wrapped CrysLLMGen | topology frozen | representation and one-way baseline |
| `C-WQ-CONFEDIT` | same WQ checkpoints | confidence-only edits | non-geometric edit control |
| `C-WQ-GEOREV` | same WQ checkpoints | geometry-triggered direct edits | final candidate |

`C-WQ-HANDOFF` tests the frozen injection fractions
`tau/T={0.25,0.5,0.75,1.0}` using a topology-conditioned continuous prior.
The best `tau` is selected on validation and frozen before generation metrics.

The old WQ-D3PM and WQ-DLM results are diagnostic appendix material.  They are
not retrained in this cycle.  The historical R5-C DLM is reported separately
under its legacy protocol.  Public CrysLLMGen numbers are protocol-different;
the locally reproduced `C-ATOM-OFFICIAL` and matched `C-ATOM-MATCHED` are the
actual executable baselines.

### 7.2 Mechanism controls

All controls reuse the same attempt id, initial Llama sample, initial continuous
noise, and number of Llama edit invocations:

- no edit;
- birth/death only;
- full edit set;
- random orbit with matched edit count;
- shuffled geometry bins;
- confidence-only trigger;
- extra Llama calls whose outputs are ignored;
- fixed topology with the same continuous-call budget.

An oracle-trigger/oracle-payload diagnostic is run only on corrupted validation
structures.  It estimates the upper bound of the bridge and continuous process;
it is never reported as a de novo method or included in SUN headlines.  If the
oracle cannot improve geometry/recovery, training a better revision policy is
not a rational use of the remaining budget.

The primary causal comparison is `C-WQ-GEOREV` versus `C-WQ-HANDOFF`.  The
representation comparison is `C-WQ-HANDOFF` versus `C-ATOM-MATCHED`.  The
engineering-parity comparison is `C-ATOM-MATCHED` versus
`C-ATOM-OFFICIAL`.  Matched methods use identical Llama backbone, LoRA rank,
training token budget, inherited CSP checkpoint, respaced timesteps, random
draws, and attempt denominator.

### 7.3 Llama/CrysLLMGen-specific diagnostics

- constrained versus unconstrained parsing on a fixed 256 validation sample;
- random-order versus canonical-order SFT;
- frozen Llama versus LoRA on topology validation metrics;
- initial-generation NLL, token count, and completion length;
- edit exact match, event accuracy, orbit-pointer accuracy, and payload accuracy;
- legality before and after grammar masking;
- repeated-orbit and early-STOP failure audits;
- diversity and mode collapse by SG, family, atom count, and orbit count.

These diagnostics cannot select a checkpoint using SUN.

## 8. Stage gates

### Gate A: assets, upstream parity, and grammar

- all real model shards exceed expected multi-GB sizes and match frozen hashes;
- offline load and one BF16 forward pass succeed through Slurm;
- the CrysLLMGen source commit, MIT notice, diffusion checkpoint, and parameter
  mapping are frozen;
- disabled-extension `C-ATOM-MATCHED` reproduces upstream parser, beta/sigma
  tables, one-step CSP tensors, and deterministic sampler outputs within
  `1e-6` on 256 hash-fixed proposals;
- one million synthetic grammar transitions contain zero illegal next states;
- de novo inference contains no initial `MASK` and no semantic padding;
- 256 constrained samples have 100% parse and topology-legality rates;
- no output-dependent retry or replacement.

Failure stops the Llama route.  The base checkpoint may be run only as an
explicitly labelled ablation after registration; it never replaces a failed or
inaccessible instruction checkpoint in a headline method.

### Gate B: CrysLLMGen-compatible proposal and WQ handoff

- validation answer NLL improves over the frozen backbone;
- all three sampling seeds pass parser/legality checks;
- SG, atom-count, orbit-count, and element-count distributions show no severe
  collapse relative to train/validation;
- WQ representation is not worse than atom serialization on matched topology
  validity and validation recovery;
- fixed 256 de novo atom and WQ samples pass their respective parser/validity
  smoke without output-dependent retries;
- `C-ATOM-OFFICIAL` reaches a credible reproduction of the published/previous
  local CrysLLMGen validity and coverage panel, with protocol differences
  explicitly reported;
- `C-WQ-HANDOFF` successfully expands, refines, and round-trips at least 99% of
  parser-valid WQ proposals.

Failure removes the large-Llama main route; do not use continuous diffusion to
hide a failed topology prior.

### Gate C: geometry revision over the CrysLLMGen-derived handoff

On `3 x 1000` development attempts:

- MatterSim `MLIP-SUN@0.1` at least +2 pp over frozen `C-WQ-HANDOFF`;
- all three sampling seeds have the same direction;
- Novel&Unique loss no more than 2 pp;
- revision precision exceeds right-to-wrong edit rate;
- shuffled/random/extra-call controls do not reproduce the benefit;
- realized compute no more than 2x the no-edit champion.

Failure removes the flagship revision claim.  MP-Doob, DLM, and relaxation
guidance cannot rescue it.

### Final oral gate

- MatterSim `MLIP-SUN@0.1` at least +5 pp, 95% CI lower bound at least +2 pp;
- MatterSim+MACE unanimous SUN at least +3 pp;
- three training seeds and raw/common-refiner/relaxed stages agree in direction;
- Novel&Unique loss no more than 2 pp;
- positive major-family, matcher-sensitivity, and orbit-changed/unchanged
  subsets;
- quality/compute Pareto satisfies the registered 2x rule.

## 9. Generation and evaluation

The evaluation inventory contains **10 evaluation families** rather than ten
independent sample-selection opportunities: upstream/formal parity; Llama
proposal/edit quality; CrysLLMGen validity/coverage; MLIP-SUN/relaxation;
novelty/uniqueness; distribution/diversity; symmetry/Wyckoff fidelity;
revision causality; efficiency/failure accounting; and statistical
robustness.  It registers 5 primary method configurations, 4 additional
inference-only causal controls, and 4 validation-only diagnostics.  None of
those inference controls trains another model.

- Development: 256 smoke, then `3 x 1000` attempts per surviving method.
- Final: 10,000 attempts per method split `3334/3333/3333` across train seeds.
- Multi-MLIP subset: 6,000 hash-fixed attempt ids.
- Primary: MatterSim 5M.
- Secondary: MACE-MP-0b3 medium.
- Guide/third evaluator: CHGNet 0.3.0.
- Run the inherited CrysLLMGen `compute_metrics.py` definitions for atom and
  expanded-WQ outputs with `CRYSLLMGEN_METRICS_NUM_CPUS=1`; do not alter the
  metric formulas.
- Use the exact frozen R5-C MatterGen evaluation script after converting every
  attempt, including failures, to its extxyz/denominator contract.
- Report only `MLIP-SUN@0.0/@0.1`; no new DFT.
- Keep evaluator-specific energy caches and hulls separate.
- Report CrysLLMGen validity, coverage, property distribution, novelty,
  uniqueness, symmetry, topology revisions, and compute.
- Use 10,000 hierarchical bootstrap replicates over training seed, sampling
  seed, and duplicate cluster; Holm correction for confirmatory secondary
  comparisons.

The Day-7 corruption-recovery attempts are never treated as de novo generation
and never enter SUN.

## 10. Revised Day 4--28 schedule

### Days 4--6: CrysLLMGen fork, parity, and asset gates, <=70 GPUh

- audit exact GPUh already spent before allocating the remainder;
- vendor the source-only CrysLLMGen snapshot and preserve its MIT notice;
- implement disabled-extension `C-ATOM-MATCHED` and pass the 256-proposal
  upstream parity gate before changing representation;
- freeze the complete Meta-Llama-3-8B-Instruct shard hashes, tokenizer, chat
  template, byte counts, shared-path permissions, and license record;
- implement serialization, parser, legality trie, direct edit grammar, and
  attempt records;
- run 100-step LoRA and offline loading smoke;
- pass Gate A and freeze protocol v4/registry v2.

### Days 7--11: WQ proposal and inherited refiner, <=320 cumulative new GPUh

- train seed-11 matched atom and WQ Llama adapters from the CrysLLMGen SFT
  port, plus the seed-11 canonical-order diagnostic adapter;
- import the CrysLLMGen CSP checkpoint, train WQ projection/bridge heads, and
  then the shared WQ continuous refiner;
- train the mixed edit curriculum;
- run component-gradient and 256-sample topology gates;
- stop if Gate B fails.

Recommended four-GPU allocation during this phase:

- GPU 0: WQ Llama LoRA;
- GPU 1: atom Llama LoRA;
- GPU 2: CrysLLMGen-initialized WQ geometry/event model;
- GPU 3: evaluator-specific reference/cache preparation.

### Days 12--16: CrysLLMGen/WQ matched screening, <=670 cumulative new GPUh

- run five primary routes with train seed 11;
- 256 smoke followed by `3 x 1000` development attempts;
- freeze `C-WQ-HANDOFF`, its injection `tau`, and its respaced schedule;
- perform Gate C and freeze or delete `C-WQ-GEOREV`.

### Days 17--21: mechanism evidence, <=970 cumulative new GPUh

- birth/death-only, full edit, random, shuffled, confidence, and extra-call
  controls;
- matcher/family/orbit-change diagnostics;
- freeze final method and all configs by Day 21;
- no parameter or threshold changes after the freeze.

### Days 22--28: final evidence, reserve at least 800 GPUh

- train missing seeds 23/47 for the three registered main model families; the
  three WQ routes still reuse the same paired WQ LoRA/refiner checkpoints;
- finish 10k attempts and fixed 6k multi-MLIP subset;
- run R5-C-compatible CrysLLMGen and MLIP-SUN evaluation;
- complete statistics, failure audit, hashes, tables, and figures.

The global 2050 GPUh cap is not reset.  Previously consumed jobs count against
it.  Before restart, compute the exact remaining budget; retain an 80 GPUh
failure buffer and never spend the Week-4 reserve on new variants.

## 11. Failure policy

- Missing or compute-node-inaccessible instruction checkpoint: stop and ask
  the user; never download or substitute on a compute node.
- Missing CrysLLMGen diffusion checkpoint or failure of upstream parity: stop
  and ask; do not relabel a from-scratch refiner as CrysLLMGen-derived.
- The instruction Llama still collapses after constrained answer SFT: drop
  Llama as the main route rather than weakening validity gates.
- WQ Llama does not beat atom serialization: remove the representation claim.
- Geometry revision fails +2 pp development gate: remove the flagship
  mechanism and report the one-way Llama-to-diffusion system only.
- Final gain is positive but below oral thresholds: submit only at the quality
  supported by the evidence; do not label it oral-level.
- Evaluator hull/reference or attempt accounting fails: affected SUN results
  are ineligible for the paper.

## 12. Required implementation changes from protocol v3

- create immutable protocol v4 and registry v2; never edit v3 artifacts;
- vendor and hash the CrysLLMGen MIT source snapshot at commit `94bb287...`;
- add an exact upstream lane and a disabled-extension parity mode;
- port local-model loading, official chat serialization, and answer-only labels
  into the CrysLLMGen Llama training/sampling entry points;
- replace its generate-until-valid loop with one terminal attempt per request;
- replace initial `MASK` topology sampling with Llama BOS/STOP generation;
- remove DLM/D3PM from active training plans;
- add a Llama model identity and LoRA identity to every attempt manifest;
- add token/FLOP/call accounting for initial and edit invocations;
- replace semantic `true_remask` wording with direct topology revision;
- add finite-state constrained decoding and its normalization audit;
- add WQ lattice/orbit/free-coordinate serialization and random-order
  provenance alongside the inherited atom serializer;
- wrap the inherited CSPNet with Wyckoff expansion, tangent projection,
  SG-compatible lattice charts, geometry evidence, and target-stratum bridges;
- add deterministic JSONL-to-extxyz and CrysLLMGen input adapters;
- retain the historical DLM only under `legacy_dlm_r5c/`.
