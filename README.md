# DLM: H1 Planner + exact-length DLM + CrysLLMGen

This repository contains the code, experiment contracts, frozen ledgers, and
reproduction records for the H1 crystal-generation program. The current
research question is narrow:

> Can a small, chemistry-aligned change to the H1 Planner improve strict
> raw-attempt `comp_valid` without changing the downstream DLM, refiner, or
> evaluator?

The full generation pipeline is:

```text
goal
  -> H1 Planner (P0 / C0 / C1)
  -> canonical Plan
  -> exact-length R5-C body DLM (B0, 7 + 4N tokens)
  -> CrysLLMGen model_494 refiner (800 reverse steps)
  -> Direct metrics
  -> common-snapshot S.U.N. metrics
```

## Current status: 2026-08-06

The project has a useful successful module result, but no claim that the full
CR-Plan route improves crystal generation:

| Result | What was tested | Outcome |
|---|---|---|
| **R03 safe-axis** | Changed only the body-DLM reveal order while keeping Planner, body weights, tokenizer, Plan, refiner, seeds, and evaluator fixed | Body completion `246 -> 248/256`; pooled raw joint validity `+5/1024`; completed-snapshot strict S.U.N. lower bound `99 -> 117/1024`. Meta S.U.N. fell `523 -> 496/1024`, so this is a successful DLM/schedule module, not a broad stability win. |
| **CR-Plan paired-32** | Added formula-prefix reachability to the frozen Planner; old missing-oxidation-state semantics were still in force | Engineering gates passed; composition `17/32 -> 18/32`, primary `9/32 -> 10/32`, terminal charge failures `0`. The candidate was very slow (`112.813/215.895 s` median/p95 vs P0 `2.835/3.022 s`), so this was not a scientific prefix-gain result. |
| **Exact-tokenizer V2 audit** | Audited the optimized trie/DP support against the complete frozen tokenizer vocabulary | `128,256` decoded fragments matched the scalar oracle; maximum audit states `76,267`; trie/scalar support time `10.3488/121.9064 s` (`~11.78x`). No model, GPU, generation, or downstream evaluation was used. |
| **E1 physical probe** | Tested the optimized support implementation on real Planner traces, separately from the V4 logical-state gate | Full-prefix median/off ratio `1.468x`; `14/14` charge-applicable full attempts had real preterminal support differences; actual-trace support and scalar parity were exact. This only justified a separately preregistered route amendment. |
| **Four-arm Plan-only 512 amendment** | `off`, `grammar-only`, `terminal-only`, and `full-prefix` with the corrected fail-closed policy | Engineering/scientific stop. Full vs terminal raw `comp_valid` was `+6/512`, but nonshortcut/primary was only `+1/512` and shortcuts increased by `+5`; terminal-only also had seven generation errors. No paired-64, paired-256, Body, Direct, or S.U.N. run followed. |

The frozen conclusion is therefore: the exact-length DLM and safe-axis schedule
are usable modules; the dominant remaining bottleneck is Planner chemistry and
composition. The failed CR-Plan route is retained as evidence and is not
repaired by changing denominators, selecting seeds, or tuning on S.U.N.

## What is being improved now: minimal no-charge ion-auxiliary SFT

The current authorized candidate keeps the existing formula-first H1 Plan and
changes only the causal part that is most likely hurting composition:

| Arm | Inference representation | Training purpose |
|---|---|---|
| `P0` | Frozen historical seven-line Plan, including `charge:` | Baseline |
| `C0` | Same Plan with the generated `charge:` line removed | Measures the minimal schema/continued-SFT effect; auxiliaries use neutral atom/count supervision |
| `C1` | The same six-line no-charge Plan as C0 | Adds explicit oxidation-state/ion witnesses only in same-head training auxiliaries |

At inference, C0/C1 generate only:

```text
formula: Li2O
anion: oxide
lattice: cubic
spacegroup: sg_195_230
volume: volpa_016_020
end: plan
```

There is no generated `charge:` or `ions:` field. The frozen evaluator still
derives charge taxonomy from the generated formula. It does not repair,
replace, filter, retry, rerank, or select an oxidation witness after sampling.
An invalid formula remains a raw failed attempt.

The 3,200-record C0/C1 task mixture is frozen before any generated result is
read:

- 30% stable primary no-charge Plans;
- 5% repeated atom/ion sequence-to-formula supervision;
- 5% matched element/count versus element/oxidation infill;
- 40% full-MP20 conditional anchors, with the formula in the input rather
  than an unconditional answer target;
- 20% conditional P0 KL/logit anchors over the non-formula fields.

Thus invalid MP-20 formulas are not silently declared physically wrong, but
they are also not repeatedly taught as unconditional positive answers. Formula
tokens receive weight `2.0`; other answer tokens receive weight `1.0`.

The DLM is deliberately unchanged: the downstream contract remains frozen B0,
the exact `7+4N` body length, `model_494`, exact 800 reverse steps, the Direct
evaluator, and the common S.U.N. snapshot. RL is not part of this first test.

### Why this change

The old target is causally backward: the model first emits a formula and only
then emits `charge: charge_fail`, so the charge label cannot guide the earlier
element/count decisions. A compact formula also gives each count only sparse
token supervision. The ion/atom auxiliary exposes repeated chemistry tokens
to the same LM head without changing the deployed Plan format.

MP-20 itself is not being discarded. Under the frozen H1 heuristic taxonomy,
the local 27,136-row training split contains 7,079 nonshortcut primary rows,
9,302 all-metal shortcuts, 226 unary shortcuts, 9,806 charge-neutrality
failures, 440 Pauling failures, and 283 oxidation-state-missing rows. This is
an evaluator-alignment problem, not evidence that the structures are
physically invalid.

SMACT `4.0.0` is frozen as a secondary witness/audit contract. The
paper-comparable legacy evaluator remains the primary metric, so upgrading
SMACT does not rewrite historical headline numbers or change the denominator.

## Metric interpretation

The paper's published CrysLLMGen `93.55%` is a **strict raw-attempt
composition-validity** number. It is not a S.U.N. survivor-denominator
number. The current local reference values are not checkpoint/recipe-parity
reproductions:

| Reference | Strict raw `comp_valid` |
|---|---:|
| Published CrysLLMGen | `93.55%` |
| Local CrysLLMGen asset | `89.2%` |
| H1-A2 epoch-2 Planner | `87.8%` |
| Frozen P-control discovery screen | `456/512 = 89.0625%` |

The no-charge experiment will report, on the same common evaluator and raw
all-attempt denominator:

- Planner parse/completion, `comp_valid`, primary/nonshortcut and shortcut
  taxonomy, failure reasons, unique formulas, top-1 frequency, element/arity
  coverage, and mean atom count;
- Body generation/completion, composition validity, `struct_valid`, and raw
  joint validity;
- CrysLLMGen metrics: `comp_valid`, `struct_valid`, `valid`,
  `wdist_density`, `wdist_num_elems`, `cov_recall`, and `cov_precision`;
- raw and completed/survivor secondary tables, novelty, novel-unique,
  uniqueness, and paired discordance;
- strict/meta S.U.N. on a common union snapshot, with coverage and unknown
  accounting.

No new no-charge SFT `comp_valid`, `struct_valid`, joint, or S.U.N. result is
claimed yet: local source repair and audit passed, but the real train-data
build, tokenizer/model smoke, and A800 training remain pending restoration of
the maintained remote tmux path.

## Frozen evaluation ladder

1. Build the exact C0/C1 train and validation ledgers and pass the tokenizer,
   witness, mask, P0-KL, and finite-gradient preflights.
2. Run the paired Planner-only raw-64 gate. Require positive raw and primary
   gains, no shortcut inflation, no meaningful parse/completion loss, and no
   new failure class.
3. Only a passing raw-64 enters the independent paired raw-256 Planner gate.
4. Only a passing Planner candidate enters frozen B0/D1 + `model_494` Body,
   Direct, and then common-snapshot S.U.N. evaluation.
5. RL is a last-resort, separately frozen Planner-only fallback; S.U.N. and
   downstream outputs cannot be used to select a checkpoint or tune the SFT.

All scientific gates use raw attempts. There is no retry, replacement,
repair, filtering, reranking, survivor-only rescue, or S.U.N.-based tuning.

## Reproduction entry points

- [Current no-charge ion-auxiliary SFT annex](workstreams/plangraph_dlm_iclr_20260731/analysis/H1_NOCHARGE_ION_AUX_SFT_EXECUTION_ANNEX_V1.md)
- [Root-cause analysis and SFT rationale](workstreams/plangraph_dlm_iclr_20260731/analysis/H1_COMP_VALID_ROOT_CAUSE_AND_SFT_PLAN_V2.md)
- [CR-Plan policy, audit, and terminal evidence](workstreams/plangraph_dlm_iclr_20260731/analysis/H1_CRPLAN_MISSING_POLICY_AND_SUPPORT_OPTIMIZATION_REVIEW_V1.md)
- [Single TODO/result index](workstreams/plangraph_dlm_iclr_20260731/EXPERIMENT_TODO_INDEX_V3.md)
- [R03 safe-axis reproducibility report](workstreams/plangraph_dlm_iclr_20260731/H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md)
- [DLM fixed-panel audit record](workstreams/plangraph_dlm_iclr_20260731/DLM_FIXED_PANEL_AUDIT_V1.json)
- [Planner and Plan implementations](crystal_dlm/h1_llm_planner.py)
- [No-charge ion-auxiliary implementation](crystal_dlm/h1_nocharge_ion_aux.py)
- [Exact-length body implementation](crystal_dlm/r5_plan_body.py)
- [CrysLLMGen integration](crystal_dlm/wqcodiff/crysllmgen/)

The older Wyckoff-quotient program is preserved for provenance but is paused;
it is not the current H1 experimental headline. Likewise, `legacy_dlm_r5c/`
contains the restored historical R5-C program and is not silently mixed into
the new no-charge comparison.

## Repository boundaries

Source code, experiment contracts, ledgers, tests, terminal reports, and
reproduction documents are versioned here. Large checkpoints, model weights,
datasets, run directories, caches, archives, and secrets are excluded by
[`.gitignore`](.gitignore) and must be referenced by immutable SHA/path records
when a run depends on them.
