# Force-Score periodic DLM checklist V1

## Objective

Improve raw structural validity and stability without changing the Planner,
composition distribution, `7+4N` representation or model494 tau800 endpoint.

The proposed transition is:

\[
L=L_{\mathrm{CE}}+L_{\mathrm{G2-valid}}+L_{\mathrm{ForceScore}},
\]

where CE preserves the exact crystal language, G2-valid defines the analytic
periodic feasible region, and Force-Score supervises the local energy-descending
transition carried by the existing G2 residual.

## Current-method failure audit

- Scope: current promoted G2 prospective/raw-refined cells, full-epoch A
  raw/refined cells, and current Plan1200 accounting only.
- Excluded: BASE, G0/G1, native canary, uncertainty-gated B and retired methods.
- Current result: 1,024 structural observations, 260 Direct structural failures,
  all 260 explained by periodic minimum distance below 0.5 Å; volume failures 0.
- Additional interface failures: 10 upstream parse/body failures in the four G2
  cells and 20 Plan1200 CIF construction failures.

## Phase A — 512-row teacher preflight (complete)

This is a mechanism test, not a training dataset or paper result.

- 64 independent MP20-train structures × 8 perturbations = 512 rows.
- Distribution: 256 collision, 128 coordinate jitter, 64 near-threshold, and 64
  lattice/wrap sentinels.
- No energy, hull, test or prospective outcome is read during selection.
- Measure CHGNet force/stress coverage, one-step energy change, periodic minimum
  distance, invalid→valid transitions, hard-token retention and sub-bin rate.
- Compare continuous force targets with the same targets quantized back through
  the exact dynamic `7+4N` representation.

Job 39212 completed all 512 rows. The continuous force step lowered CHGNet
energy on 76.8% of states, but only 54.3% remained lower after exact `7+4N`
quantization. More importantly, 59/64 near-threshold 0.60 Å structures crossed
from valid to invalid despite their energy decrease. Pure Force-Score therefore
does not support direct student or full-data training: it can exchange validity
for a lower local energy.

## Phase A2 — species-margin-projected teacher preflight (complete)

Reuse the exact same 512 rows, CHGNet teacher and tokenization. Do not enlarge
or resample the preflight set.

- use the existing species-aware PBC barrier and lattice-validity terms;
- gate force supervision off for severe/nonlocal collisions;
- project a locally credible force direction onto the first-order geometric
  feasible half-space;
- form wrapped XYZ soft labels only over adjacent token bins that remain
  geometrically valid after exact quantization;
- use barrier-only targets when no force-active valid token is available;
- report energy and validity jointly, never counting invalid lower-energy states
  as teacher successes.

Job 39224 restored Direct validity for all 256 initially invalid collision rows
and caused only one valid→invalid transition. It nevertheless failed as an
energy-aware teacher: the force remained active on only 28.2% of non-severe
rows, only 20.3% of projected force candidates lowered energy, and 120 rows
were Direct-valid but unresolved against the stricter species margin. Treating
the 0.60–1.40 Å species prior as a hard feasible set over-constrained the
teacher and raised collision energies by roughly 1.2–2.3 eV/atom.

## Phase A3 — hard-validity / soft-species separation

Reuse the same 512 rows again. The deterministic hard projection now represents
only parser/Direct feasibility: exact triclinic PBC minimum distance at least
0.50 Å, with the existing 0.001 Å numerical projection tolerance and an exact
post-quantization validity check.
The existing species-aware 0.60–1.40 Å relation remains a differentiable G2
training prior and is not used to reject or overwrite a force target.

- severe collisions receive hard-barrier-only targets;
- locally credible states receive force followed by the 0.55 Å projection;
- valid states fall back to identity when the quantized projected force does
  not lower energy;
- no inference-time projection, repair, reranking or energy calculation is
  introduced.

Proceed to a student only if teacher coverage is complete, near-threshold
valid→invalid is at most 1/64, all four collision severities recover Direct
validity on at least 90%, force remains active on at least 50% of non-severe
rows, and at least 70% of force-active quantized targets lower energy.

## Phase B — micro-student preflight

Only after Phase A3 succeeds:

- split by base structure: 48 structures / 384 rows train and 16 structures /
  128 rows holdout;
- train G2 residual only for 128–256 updates;
- freeze Planner, base DLM, Compact-V2 LoRA and q0;
- use wrapped soft XYZ targets plus a secondary direction loss;
- verify holdout collision, raw Direct, paired CHGNet and gradient compatibility.

This phase tests whether a valid teacher can actually be learned through the G2
residual. It is not the final model.

## Phase C — full MP20 training

Only after both preflights succeed:

- complete MP20 train: 27,136 independent structures;
- one deterministic perturbation/current state per structure, weighted to the
  audited current-method failure distribution;
- one training seed, effective batch 16, one epoch ≈ 1,696 updates;
- release only the epoch endpoint; no checkpoint or hyperparameter selection;
- full MP20 validation is evaluation-only;
- final screen reuses the fixed current 256 Plans and noise, raw first;
- run model494 tau800 and official S.U.N. only after a positive raw result.

## Fixed boundaries

- no Planner resampling or composition tilt;
- no inference-time CHGNet, force calculation, repair or candidate selection;
- parser/schema failures stay under CE and exact-token accounting;
- severe non-graphable geometry stays under G2-valid and remains in denominators;
- official hull is never a training target.
