# Difficulty-decomposed Planner pilot result

## Identity

- Training job: `34697`
- Sampling continuation: `34704`
- Training: H1-A2 epoch-2 continuation, 400 updates, two fixed seeds
- Sampling: 256 rich Plans per arm and seed
- Frozen downstream was not run because the Plan-only screen failed

## Training result

| Seed | CE control final val NLL | Weighted candidate final val NLL |
|---:|---:|---:|
| 17 | 0.282842 | 0.282482 |
| 18 | 0.284210 | 0.284231 |

The candidate preserved basic Plan language modeling but did not provide a
repeat-stable NLL gain.

## Plan-256 result

| Seed | Control parsed | Candidate parsed | Control all-metal | Candidate all-metal | Control oxide | Candidate oxide |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 255/256 | 254/256 | 82 | 71 | 71 | 82 |
| 18 | 256/256 | 255/256 | 76 | 74 | 56 | 70 |

Using the frozen H1-A2 historical family conversion rates only as a
pre-downstream projection:

| Seed | Strict mix control | Strict mix candidate | Meta mix control | Meta mix candidate |
|---:|---:|---:|---:|---:|
| 17 | 7.966% | 7.548% | 48.320% | 46.934% |
| 18 | 8.214% | 7.803% | 48.141% | 47.388% |

## Decision

The candidate fails the fast proposal screen: both seeds lose one parsed Plan
and both shift projected Strict/Meta yield downward, largely through increased
oxide share. It is not sent to the DLM/refiner and is not promoted to the
public paper claim.

This pilot used the original unnormalized factor combination. The repository
now contains the corrected within-stratum-normalized implementation. On
2026-08-26 the user authorized a scientifically distinct V2 run: reuse the
matched V1 controls, retrain only normalized candidates at seeds 17/18, and
then run real downstream evaluation rather than treating projected family mix
as the final outcome. V1 remains frozen negative evidence.
