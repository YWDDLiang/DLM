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
Llama scores those same actions from typed embeddings of the partial action and
ledger state. Its residual output heads are zero-initialized. Consequently the
initial Llama `log_softmax` is uniform on each legal set and the normalized
fused distribution is exactly the original C3FD distribution at step zero. The
two log-probabilities are added with fixed unit coefficients and sampled once.
There is no candidate pool, post-hoc filter, retry, replacement, rerank, or
best-of-N.

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

The frozen V4 build subsequently established that every eligible MP20
train/validation row is already `meta_or_better` under the dataset's native
filter. The actual run therefore learns a positive **MP20 near-stable support
prior**, not a contrastive high-versus-low hull classifier. The unused `higher`
ID remains in the schema for explicit accounting but has zero training rows.
The paper must not claim that the Planner predicts hull energy from composition.

## One-run contract

- fresh Llama LoRA plus the minimal action scorer; frozen C3FD checkpoints and
  hard masks;
- MP20 train only for optimization; standard validation is monitoring only;
- one fixed seed `85017`, one epoch, final checkpoint only;
- product-of-experts coefficient exactly `1 + 1`; no grid;
- row-balanced loss exactly `(proposal CE + mean action CE + mean soft-field
  CE) / 3`, so high-arity compositions do not receive larger total weight;
- record fused-versus-C3FD action KL and selected-action C3FD rank during the
  same sampling run; these are diagnostics, not extra experiment cells;
- one CPU round-trip/mask/teacher-action check;
- one fixed stream17 body+Direct run using existing Compact-V2 DLM seed82017;
- requested-denominator composition validity must be at least 95%; all failed
  rows remain failures;
- no model494, CHGNet, or new official query before the body+Direct result.

Formal typed data job39046 completed in 33 seconds with train `24,558` and
validation `8,158` rows. Missing typed witnesses (`2,578/888`) and one invalid
validation teacher sequence remain explicit in the manifest. Jobs39028,
39030, and39035 are preserved engineering negatives (join-key correction and
removal of unnecessary per-row training-mask compilation).

This is not the complete MP20 split. The Compact-V2 DLM used the full MP20
standard `27,136/9,047` train/validation rows, while this fused Planner
inherited the older certificate-authorized `ctv_minimal_spec_v4` subset. Keep
the current run as development evidence. Any paper-final Planner retraining
must use all `27,136/9,047` rows and may not delete rows merely because a typed
C3FD witness/certificate is unavailable; those rows require an explicit
non-filtering supervision path. Per user decision, do not rerun the active
sprint solely to repair this scope.

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
