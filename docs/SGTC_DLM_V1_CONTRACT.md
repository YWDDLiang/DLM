# SGTC-DLM-v1 contract

Date: 2026-08-28

Status: frozen after CTV-v1 failed its state-centered value gate and before
SGTC data construction or training.

SGTC means **Stable-Geometry Token Curriculum**. It is a supervised non-RL
fallback for the exact-cardinality special-token executor.

## Scientific question

Does geometry-only continuation on real strict-stable MP-20 structures improve
fixed-composition stable realization beyond geometry-only continuation on the
full C3FD-certified MP-20 training distribution?

## Matched arms

- `G0 all`: every C3FD-certified MP-20 row;
- `G1 strict`: rows whose source `e_above_hull<=1e-8 eV/atom`.

The selection field is offline data metadata. Training JSON removes hull,
formation-energy, energy and stability fields recursively. The visible prompt
remains the same minimal certified-composition JSON and contains no energy or
stability token.

Both arms resume independently from `ctv_minimal_base_36898/step-696` with the
same seed `81017`, effective batch 16, cosine LR `5e-6`, and exactly 348 updates.
Step 348 is selected without checkpoint or epoch search.

## Geometry-only corruption and loss

For exact `7+4N` dynamic bodies, candidate masked/supervised positions are only:

- relative positions 1--6: lattice lengths and angles;
- each site's three XYZ positions.

The N token and every element token remain visible and receive zero loss. This
matches candidate-to-structure inference, where C3FD fixes composition before
the DLM realizes geometry. Lattice and XYZ weights are all 1 inside this
geometry-only denominator; there is no hidden stability weighting.

No generated unstable structure is a CE target. No reranking, replacement,
RL, energy prompt, stable token, model494 change or tau search is allowed.

Training NLL does not select an arm. Both successful arms advance to the same
matched downstream screen; the eventual S.U.N. gate retains the registered
body/Direct, structural novelty/uniqueness and retention floors.
