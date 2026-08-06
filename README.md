# Current ICLR Focus: Shared Plan + Diffusion Language Model

As of 2026-07-28, the active research focus is a fully de novo shared-Plan
pipeline built on the strongest restored R5-C assets:

```text
goal -> H1-A2 Planner -> shared Plan
                       -> exact-length R5-C body DLM -> draft
                       -> Plan-conditioned CrysLLMGen diffusion -> crystal
```

The immediate task is not another Planner/body training run.  It is an
algebraic exact-null repair of the strongest later shared-Plan refiner, followed
by a frozen matched/null/shuffled mechanism gate and then one H1-A2
fully-de-novo paired panel.

The Wyckoff-quotient program is paused and preserved in the non-destructive
checkpoint at
[`archive/20260728_wq_pause_checkpoint/`](archive/20260728_wq_pause_checkpoint/README.md).

Active entry points:

- [ICLR Plan + DLM relaunch](docs/experiment_program/20260728_iclr_plan_dlm_relaunch.md);
- [complete post-R5-C results review](workstreams/r5c_reactivation_20260728/ALL_POST_R5C_RESULTS_REVIEW.md);
- [mutable reactivation workstream](workstreams/r5c_reactivation_20260728/README.md);
- [frozen provenance snapshot](legacy_dlm_r5c/README.md).

The scientific restart uses H1-A2 epoch 2 as the Planner, the frozen R5-C
exact-length DLM as body, and the original CrysLLMGen diffusion parent.  The
later S2 shared-Plan adapter is an initialization/diagnostic only until its
null path is algebraically identical to the parent.  Historical outputs are
never overwritten; every A800 request remains limited to at most eight CPUs.

# Paused Program: CrysLLMGen-Derived Stratified Wyckoff Revision

This workspace targets one ICLR oral-quality crystal-generation paper:

> A closed-loop Wyckoff-quotient extension of CrysLLMGen's Llama-to-diffusion
> crystal generator.

The discrete topology specifies the space group and an unordered multiset of active
Wyckoff/species orbits. It indexes a continuous stratum containing the
symmetry-compatible lattice metric and the free coordinates of those orbits.
An autoregressive CrysLLMGen proposer emits the space group first, followed by
the lattice chart and a dynamic sequence of Wyckoff/species/free-coordinate
orbit tuples. The space group is fixed after emission. The proposal is expanded
and refined by the inherited CrysLLMGen CSPDiffusion path, while geometry
evidence can trigger direct orbit birth, death, type change, or species change.
Newly introduced continuous charts are initialized by a target-stratum bridge
before refinement resumes.

The Day-7 diagnostic did not promote DLM, so the active method does not start
from a global `MASK` topology state and does not claim that DLM is superior to
AR or D3PM. `MASK` may be used only as a local training-corruption or infilling
symbol. The preserved, previously valid R5-C DLM program is isolated under
[`legacy_dlm_r5c/`](legacy_dlm_r5c/README.md).

The official CrysLLMGen repository at commit `94bb287...` is the code and
experimental parent, not merely an external baseline. Its exact atom-wise path
is preserved as an upstream reproduction. The flagship must beat a matched
one-way WQ handoff by using refinement geometry to correct already proposed
Wyckoff topology.

The active cycle is hard-capped at four A800 GPUs for 28 days: 2050 usable
GPU-hours, with 800 reserved for the frozen champion/final comparison in Week 4.

## Active Documents

- [Four-week execution runbook](docs/experiment_program/FOUR_WEEK_RUNBOOK.md)
- [Decision log](docs/experiment_program/DECISIONS.md)
- [AR topology decision](docs/experiment_program/AR_TOPOLOGY_DECISION.md)
- [Revised CrysLLMGen-WQ experiment plan](docs/experiment_program/LLAMA_AR_REVISED_EXPERIMENT_PLAN.md)
- [CrysLLMGen fork and modification map](docs/experiment_program/CRYSLLMGEN_FORK_MAP.md)
- [Llama-3-8B-Instruct asset preflight](configs/experiments/wyckoff_codiffusion/model_asset_preflight_meta_llama3_8b_instruct_v1.json)
- [CrysLLMGen MP20 checkpoint preflight](configs/experiments/wyckoff_codiffusion/model_asset_preflight_crysllmgen_mp20_v1.json)
- [Training and evaluation inventory](configs/experiments/wyckoff_codiffusion/training_evaluation_inventory_v1.json)
- [Frozen Day-7 protocol v3](configs/experiments/wyckoff_codiffusion/protocol_v3.yaml)
- [Frozen Day-7 registry v1](configs/experiments/wyckoff_codiffusion/experiment_registry_v1.yaml)

The primary AR asset is the verified shared
`Meta-Llama-3-8B-Instruct` checkpoint at
`/public/home/jiaosz/hengzhang/models/LLM-Research/Meta-Llama-3-8B-Instruct/`.
Protocol v4 and registry v2 will be created only after its full shard-hash,
grammar, and Slurm offline-forward gates pass. Protocol v3 remains immutable
because the completed Day-7 artifacts are bound to its hash.

Superseded plans, protocols, failed runs, data, references, and reports stay
locally preserved but are excluded from the active source bundle and every
server execution path.

## Claim Boundary

The project does not claim the first crystal DLM, the first joint
discrete--continuous crystal diffusion, the first bidirectional atom/structure
model, or the first variable-cardinality generator. The candidate novelty is
the combination of:

1. topology-indexed continuous Wyckoff strata;
2. orbit-level dimension-changing birth/death transitions;
3. constraint- and geometry-adaptive direct revision of committed topology.

All stability headlines are explicitly `MLIP-SUN@0.0` or `MLIP-SUN@0.1`
because no new DFT is planned. Raw, common-refiner, and relaxed stages are
reported separately.

## Active Evidence

Historical run directories remain under runs but are excluded from active source sync:

- the R5-C conditional oracle diagnostic;
- the legacy A100 evaluator-sensitivity baseline.

They are not headline-eligible. All paper results must be rerun under the new
attempt-level protocol with stable IDs, fixed seeds, no output-dependent
selection, frozen evaluators, and one final refinement per submitted attempt.

## Registered Work Order

1. Audit the remaining GPU budget and freeze the complete Llama asset.
2. Vendor the CrysLLMGen source-only snapshot, preserve its MIT notice, and
   pass disabled-extension atom-pipeline parity.
3. Implement and formally audit the atom/WQ proposal grammars, inherited
   CSPDiffusion wrapper, and direct edit process; then freeze protocol v4.
4. Train the registered 3 x 3-seed main runs plus one seed-11 presentation
   ablation, pass the proposal/handoff gate, and only then begin SUN screening.
5. Screen the five matched routes, pass or stop geometry revision, and
   freeze the final method by Day 21.
6. In Week 4, run only frozen champion/final three-seed, 10k-attempt,
   multi-MLIP, symmetry, novelty, compute, statistics, and failure audits.
