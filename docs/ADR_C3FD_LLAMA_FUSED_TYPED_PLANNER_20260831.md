# ADR: stability-conditioned C3FD–Llama fused typed Planner

Date: 2026-08-31

Status: **accepted; speed-first single-run implementation**

## Context

The user wants Llama to do more than predict three soft structural fields. The
Planner should let Llama learn which compositions are favored by MP20 stability
data while C3FD keeps the generated action sequence chemically feasible. The
downstream interface remains Compact V2 and the DLM remains the paper center.

Resources and time rule out a matrix of small experiments. The implementation
therefore has one training run, one CPU contract check, and one single-stream
body+Direct screen before the geometry-aware DLM work.

## Decision

Build one fused autoregressive **typed** Planner:

```text
target stability = meta-or-better
                   +
serialized partial C3FD conservation state
                   v
            Llama-3-8B + fresh LoRA
                   |
                   | log p_Llama(typed action)
                   v
frozen C3FD calibrated logits + legal-action mask
                   |
                   | log p = log p_C3FD + log p_Llama
                   v
       one sampled N/family/arity/species/count action
                   |
                   `---- repeat until one legal terminal composition
                                      |
                                      v
                      Llama scores the Compact-V2 soft fields
                                      |
                                      v
                         canonical C3FD_NATIVE_PLAN_V2
                                      |
                                      v
                       existing Compact-V2 masked DLM
```

At each step C3FD enumerates only actions that remain reachable under atom
count, charge, valence-family, arity, and benchmark-compatibility constraints.
Llama scores those same actions from the serialized partial state. The two
calibrated log-probabilities are added with fixed unit coefficients and sampled
once. There is no candidate pool, post-hoc filter, retry, replacement, rerank,
or best-of-N.

This makes Llama causally relevant to `N`, family, arity, every element/count
choice, and the three Compact-V2 structural fields. C3FD is an internal
scientific decoding constraint rather than an external validity checker.

## Stability supervision

Join the immutable C3FD semantic MP20-train rows to their original MP20-train
metadata by `source_row_idx`. Convert train-only `e_above_hull` into the fixed
conditioning tiers already used by the project:

- `strict`: `e_above_hull <= 0`;
- `meta`: `0 < e_above_hull <= 0.1`;
- `higher`: `e_above_hull > 0.1`.

Train Llama by teacher-forced likelihood of the one observed typed composition
and Compact-V2 fields conditioned on its actual tier. At inference request
`meta_or_better`. This is target-conditioned generation, not test-outcome
training: no development, prospective, CHGNet, model494, or official-query row
enters the Planner data. Do not add a second preference/RL stage or tune tier
weights.

## One-run contract

- fresh Llama LoRA plus the minimal action scorer; frozen C3FD checkpoints and
  hard masks;
- MP20 train only for optimization; standard validation is monitoring only;
- one fixed seed `85017`, one epoch, final checkpoint only;
- product-of-experts coefficient exactly `1 + 1`; no grid;
- one CPU round-trip/mask/teacher-action check;
- one fixed stream17 body+Direct run using existing Compact-V2 DLM seed82017;
- requested-denominator composition validity must be at least 95%; all failed
  rows remain failures;
- no model494, CHGNet, or new official query before the body+Direct result.

If comp-valid passes, the fused Planner is retained so the paper can preserve
the C3FD-constrained Llama story. Raw Direct is reported but is not used to
launch alternative Planner variants. The next optimization target is G1 on the
Compact-V2 DLM; G2 remains a later optional parallel candidate.

## Downstream DLM

Reuse the two-epoch Compact-V2 DLM checkpoint from job38703 for the first fused
Planner screen. Do not spend resources reproducing an identical base. G1 adds
periodic geometry training to this Compact-V2 representation. G2 keeps the
previous residual-relation-adapter idea and uses the same one-seed/one-stream
economy if it is later launched.

## Claim boundary

The Planner contribution is: a pretrained Llama composition prior fused inside
a conservation-constrained C3FD decoder preserves high chemical validity while
conditioning a masked crystal DLM. It is not a claim that Llama alone predicts
true hull energy, nor that hard constraints improve thermodynamic stability.

