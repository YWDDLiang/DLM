# ADR: C3FD-conditioned Llama Compact-V2 Planner and DLM mainline

Date: 2026-08-31

Status: **accepted for one matched development screen**

## Context

The old-rich F/M prospective run preserved C3FD composition validity and
recovered raw/refined structural execution, but its refined Meta S.U.N. was
about 38%. The earlier Compact-V2 DLM reached about 55% refined Meta S.U.N. on
an MP20-overlapping development cohort. That number is not a prospective
comparison, but it identifies the best currently available DLM interface.

The paper must remain DLM-centered while connecting C3FD to Llama as a genuine
scientific conditioner rather than a post-hoc validity filter.

## Decision

Adopt one candidate mainline:

```text
frozen C3FD composition + semantic state
        |-- lock N/elements/counts/anion_framework
        `-- learned soft prefix
                    v
       Llama + LoRA jointly generates only
       lattice_system / spacegroup_bucket / volume_per_atom_bin
                    v
       canonical C3FD_NATIVE_PLAN_V2
                    v
       existing fresh Compact-V2 masked DLM
                    v
       raw structure -> fixed model494 only for system evaluation
```

The C3FD hard composition cannot be changed by Llama. The soft-prefix projector
reuses Route M's existing two-layer implementation, but the visible target is
the three-field Compact-V2 structural suffix rather than the historical
seven-line H1-A2 suffix. Llama's role is joint autoregressive modeling of the
correlated structural hints from the frozen C3FD state.

The existing Compact-V2 DLM checkpoints from job38703 already consume exactly
`C3FD_NATIVE_PLAN_V2`; do not retrain them merely to add Llama. Use fixed seed
82017 for the one-seed development screen by numerical preregistration, never
because of its outcome. G1 and optional G2 subsequently modify this Compact-V2
DLM baseline, not the frozen old-rich DLM.

## Minimal implementation

1. Reuse the C3FD feature packer, soft-prefix projector, Llama LoRA trainer,
   Compact-V2 serializer/parser, and job38703 DLM checkpoint.
2. Add only a Compact-V2 suffix target and sampler for the three soft fields.
3. Train one fixed Planner seed on MP20-train teacher Compact-V2 targets. The
   input is frozen C3FD state; energy, hull, DLM outcomes, and prospective
   outcomes are absent.
4. Enforce train/serve equality of keys, types, order, and serializer; preserve
   every failed Planner attempt without filling a field.
5. Compare direct C3FD-predicted V2 versus Llama-expanded V2 with the same
   existing DLM seed82017, stream17, composition ledger, noise, and denominator.
   Run body plus Direct first and reuse the already frozen official cache only
   if downstream development reporting is later authorized. No new MP query is
   needed or allowed.

## Why Llama is not decorative

Direct C3FD heads predict the three soft fields separately. The Llama arm sees
the frozen C3FD semantic state through learned prefix embeddings and models the
joint sequence of lattice system, space-group bucket, and volume bin. The
matched direct-C3FD ablation tests whether this joint learned conditioner adds
anything beyond the hard-valid composition proposal. If it does not improve or
at least preserve raw realization, the paper must not claim a Llama benefit.

## Trade-offs

- The Compact-V2 development high point contains MP20 train/validation overlap
  and is not an expected prospective score.
- Teacher-target Planner SFT still has train/inference value shift; the matched
  screen measures whether Llama reduces it.
- Reusing job38703 saves a redundant two-epoch DLM retrain but fixes the base
  checkpoint and seed before seeing the new Planner outcome.
- The old-rich F/M result remains a disclosed negative/ablation and is never
  deleted or relabeled.

## Promotion rule

Retain the Llama Compact-V2 Planner if requested-denominator composition
validity is at least 95%, body execution is within one percentage point of the
direct Compact-V2 control, and raw Direct is not lower. Otherwise retain direct
C3FD Compact-V2 as the Planner interface. This is a development decision, not
a seed-robust claim.

