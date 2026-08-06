# R5-C Reactivation Plan

Date: 2026-07-28

> **Status update:** Stage 0–2 remain the frozen R5-C restoration/reference
> plan.  Stage 3 onward is superseded by
> `20260728_iclr_plan_dlm_relaunch.md`, after the later Shared-Plan,
> PlanBridge, and PlanV2 results were incorporated.  The active ICLR route is
> H1-A2 epoch-2 Planner + frozen R5-C exact-length DLM + a repaired
> algebraic-null Plan-conditioned CrysLLMGen refiner.  No exposure-matched body
> continuation is authorized before the direct shared-Plan mechanism gate.

## Decision

The active research focus returns from the Wyckoff-quotient program to the
earlier R5-C exact-length diffusion-language-model line.  The WQ work is
preserved as a paused branch.  It is not deleted, silently merged into R5-C, or
used to rewrite historical R5-C results.

The restored R5-C program has two layers:

1. the canonical conditional R5-C body/refinement result;
2. the de novo planner/executor experiments performed after that reproduction,
   through H1-A2/H1-A3, free-geometry, H1-G1, H2-P1, and the completed H1-A4
   joint-basin experiment.

## Frozen Scientific Anchor

R5-C is an exact-length conditional body generator:

```text
plan_state
  -> exact-length R5-C body (7 + 4N tokens)
  -> CrysLLMGen refinement
  -> CrysLLMGen metrics
  -> original A100-script CHGNet S.U.N.
```

The canonical full-1000 result is conditional because the prompt supplies the
plan/composition.  It is an upper reference for the body/refinement interface,
not a complete de novo planner claim.

Historical canonical values:

| Metric | CrysLLMGen baseline | R5-C |
| --- | ---: | ---: |
| composition validity | 89.2 | 90.7 |
| structure validity | 99.9 | 99.8 |
| joint validity | 89.1 | 90.5 |
| coverage recall | 94.1079 | 96.6947 |
| density Wasserstein distance | 0.2462 | 0.1105 |
| number-of-elements Wasserstein distance | 0.1018 | 0.0161 |
| A100/CHGNet strict lower bound | 9.00% | 10.30% |
| A100/CHGNet strict adjusted | 9.31% | 10.61% |
| A100/CHGNet meta-like lower bound | 46.10% | 72.20% |
| A100/CHGNet meta-like adjusted | 47.67% | 74.38% |

These values remain historical until a reactivated run passes the environment
and protocol preflight.  New evaluations must run entirely in
`diff_meets_diff` and use the preserved original A100 `eval_sun.py` and
`eval_sun_resumable.py` scripts with CHGNet.  MatterSim is not introduced into
training or headline evaluation.

## Verified Restore Anchors

Local frozen snapshot:

```text
legacy_dlm_r5c/
```

Key SHA256 anchors:

```text
crystal_dlm/r5_plan_body.py
  3478ddf657873ea055e5816c423ce36be5ecf0cd1a73c6ee1e5514648047be83
scripts/sample_llada_r5c_plan_body.py
  5239a9a9ef9c078911a03ddb4791a217433ad62cf524caba99d0b6cc12c913b0
evidence/reports/20260531_r5c_full_pipeline_reproducibility_and_technical_evolution.md
  a5cc3ff4459eada0cabed19199a2b3a7464887ba87af6a68d8490883ae5e61ad
```

Local frozen tests on 2026-07-28:

```text
168 total
139 passed
27 skipped optional/dependency tests
2 errors limited to intentionally absent archived doping datasets
0 R5-C core-code failures
```

A800 assets verified present:

```text
runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final
runs/20260531_0040-r5c-full1000-sun
runs/20260531_2200-a100-eval-sun-mpapi-cache-final2
data/dlm_sft/mp_20_r5_exact_length/test.jsonl
reference/a100_eval_sun/eval_sun.py
reference/a100_eval_sun/eval_sun_resumable.py
/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/
```

Environment verified:

```text
/public/home/jiaosz/miniconda3/envs/diff_meets_diff
Python 3.10.18
torch 2.4.0+cu121
```

## What The Follow-Up Experiments Established

The post-reproduction work should be restored as evidence, not repeated
blindly:

- H1-A2 was the best usable de novo planner/body branch of that phase:
  strict adjusted 9.71%, meta-like adjusted 48.94%.
- H1-A3 did not promote.  More planner SFT shifted atom count upward and
  degraded the joint distribution.
- the default free-geometry executor was the best robust executor ablation:
  strict adjusted 9.60%, meta-like adjusted 50.39%, but it still did not close
  the R5-C conditional meta-like gap.
- H1-G1 retained high structural validity and coverage but stayed around
  48-50% meta-like adjusted.
- H2-P1 plain-text proposal collapsed before refinement because graph yield was
  only about 4%.
- H1-A4 completed both formal epochs with exit status 0.  Epoch 1 reached
  comp-valid 84.6%, struct-valid 100.0%, strict adjusted 8.55%, and meta-like
  adjusted 44.92%.  Epoch 2 reached comp-valid 86.4%, struct-valid 99.9%,
  strict adjusted 8.35%, and meta-like adjusted 47.32%.
- H1-A4 epoch 2 passed the hybrid target gate and both epochs passed the
  acceptable planner gate, but neither approached the conditional R5-C anchor
  (10.61% strict adjusted, 74.38% meta-like adjusted).  H1-A4 is therefore a
  completed negative result, not unfinished work to rerun.

The dominant bottleneck was low-hull-basin entry, especially for oxides and
chalcogenides, rather than parsing, graph construction, or CrysLLMGen
structural validity.  H1-A4 resolves the earlier attribution fork in favor of
plan/executor basin compatibility: improving planner syntax and marginal
calibration alone did not recover the conditional R5-C stability.

## Execution Order

### Stage 0 — Reactivation preflight

- Work only from a copy of `legacy_dlm_r5c/`; keep the frozen snapshot
  unchanged.
- Verify source anchors, model/checkpoint paths, test JSONL, evaluator scripts,
  CHGNet imports, and cached MP/hull inputs.
- Enforce the permanent resource rule: one A800 may request at most eight CPU
  cores.
- Use new run IDs and output directories.  Never resume into or overwrite a
  historical run.

### Stage 1 — R5-C anchor confirmation

Run one fixed-seed 256-attempt conditional confirmation without training:

```text
frozen plan_state
  -> frozen R5-C checkpoint
  -> frozen CrysLLMGen refiner
  -> CrysLLMGen direct metrics
  -> original A100-script CHGNet strict@0.0 and meta@0.1
```

This is the first GPU experiment after reactivation.  It uses one A800 and at
most eight CPUs.  Promotion requires matching the qualitative historical
behavior: near-perfect parse/plan match, high graph yield, composition validity
near 90%, structure validity near 100%, and no unexplained collapse in the
original A100/CHGNet evaluator.

### Stage 2 — Formal 1000 confirmation

Only after Stage 1 passes, run the frozen full-1000 conditional pipeline.  This
re-establishes the exact comparison row under the current
`diff_meets_diff` environment.  It does not train a new model.

### Stage 3 — Begin the post-H1-A4 continuation

**Superseded:** do not execute the body-continuation sequence below.  It is
retained as historical reasoning only.  Follow
`20260728_iclr_plan_dlm_relaunch.md`.

Do not rerun H1-A4, H1-A3 epoch 3, free-geometry CE reweighting, or the failed
plain-text H2-P1 branch.  The next branch tests whether generated-plan exposure
is the missing interface between the planner and the frozen R5-C body:

1. use the already completed H1-A4/H1-G1 outputs for an offline, family-stratified
   compatibility analysis; this consumes no new GPU generation;
2. build a versioned continuation set from valid MP-20 plan/body pairs, with
   fixed rebalancing toward the oxide, chalcogenide, and charge-sensitive bins
   that were under-served by the de novo branches;
3. train one small exposure-matched R5-C body continuation while keeping the
   planner and CrysLLMGen refiner frozen;
4. evaluate one fixed, unscreened 256-attempt de novo panel with the same
   planner seeds, then run CrysLLMGen metrics and original A100/CHGNet S.U.N.;
5. expand to 1000 only if graph yield, composition validity, diversity, and
   strict/meta S.U.N. jointly improve over the retained H1-A2/free-geometry
   controls.

The continuation uses chemistry/structure supervision, not MatterSim or another
MLIP in the training loss.  Existing CHGNet/e_hull labels are diagnostic and
checkpoint-level evaluation evidence only.  No candidate-pool selection,
retry, repair, inference-time MLIP guidance, or per-sample reranking is allowed.

### Stage 4 — Paper-scale decision

The R5-C line becomes paper-ready only if a fully de novo branch approaches the
conditional body reference without sacrificing graph yield or diversity.
Until then:

- R5-C conditional is a strong mechanism/upper-reference result;
- H1-A2/free-geometry/H1-G1 remain diagnostic baselines;
- no de novo headline claim is made from the conditional full-1000 row.

## Minimal Record Policy

To avoid repeating the WQ audit overhead, each new experiment records only:

- immutable config and source SHA;
- exact command, environment, Slurm resources, seeds, and exit status;
- generated denominator and no-retry/no-rerank assertion;
- CrysLLMGen report and A100/CHGNet S.U.N. report;
- one terminal summary.

No per-poll audit sidecars are created.
