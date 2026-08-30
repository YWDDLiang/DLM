# C3FD rich-field semantics and predictability audit v2

Date: 2026-08-30

## Provenance

- CPU Slurm job: `38319`, `COMPLETED`, exit `0:0`, elapsed `00:00:07`, 16 CPUs,
  zero GPUs.
- Validation rows: 9,047 from
  `data/c3fd_semantic_v21_step1b_20260828/val.jsonl`, SHA256
  `67526f40a32f74722b7e425f5888885918a60630c00911651be26258cec514b3`.
- Seed17 checkpoint SHA256:
  `87c1673f709c14488848196e3f466b0f4797cc3511108a07b6d498ca96d17d53`.
- Seed18 checkpoint SHA256:
  `d2e31996d034ba0f452ba72906539d6067bf3d7bf8969940c0f7961011b02b07`.
- Immutable JSON output SHA256:
  `704f26bcf6ba8903a7b5ac1bc7ddc277c03a782e8708601ff67aa85889486c9d`.
- Output path:
  `$ROOT/data/c3fd_rich_field_predictability_v2_20260830`.
- `outcomes_read=false`; no structure generation, energy, hull, S.U.N. or test
  outcome entered this audit.

## What the labels actually mean

The source implementation, `src/crystal_dlm/r5_plan_state.py`, derives
`lattice_system` from cell lengths and angles. It derives
`spacegroup_bucket` independently from `spacegroup.number.conv` metadata.
Therefore the first is a metric-cell class and the second is a symmetry class.
A primitive cell can have a rhombohedral/trigonal metric while belonging to a
cubic conventional space group. A hard one-to-one map is not a ground-truth
compatibility law for this representation.

The deployed C3FD sampler did something different: it sampled lattice and
volume rich logits, then set SG by a hard one-to-one lattice lookup. That made
its generated text internally uniform but discarded a nonredundant trained
prediction target.

## Two-checkpoint validation results

| Field | Seed17 accuracy | Seed18 accuracy | Majority | NLL seed17/18 | ECE seed17/18 |
|---|---:|---:|---:|---:|---:|
| anion framework head (unused) | 94.94% | 94.92% | 47.51% | 0.166/0.171 | 0.0129/0.0120 |
| charge head (unused) | 94.46% | 94.29% | 48.88% | 0.142/0.140 | 0.0106/0.0107 |
| metric lattice | 61.61% | 62.11% | 27.91% | 1.047/1.042 | 0.0117/0.0113 |
| SG bucket, separately supervised head | 60.61% | 60.66% | 22.33% | 1.073/1.067 | 0.0055/0.0154 |
| volume/atom bin | 69.90% | 69.61% | 28.34% | 0.781/0.786 | 0.0359/0.0318 |

Volume ordinal MAE is `0.404/0.407` bins. These are reproducible, balanced
improvements over majority prediction; none by itself proves a stability gain.

## SG nonredundancy

| Quantity | Seed17 | Seed18 |
|---|---:|---:|
| target one-to-one metric/SG map agreement | 42.60% | 42.60% |
| target `H(SG | metric lattice)` | 0.8813 nats | 0.8813 nats |
| separately supervised SG-head accuracy | 60.61% | 60.66% |
| current compiler SG accuracy | 26.78% | 27.50% |
| lattice+independent-SG joint accuracy | 48.51% | 48.77% |
| predicted joint TVD from target | 0.1437 | 0.1281 |

The earlier “100% lattice/SG compatibility” audit measured the compiler's own
one-to-one rule, not fidelity to the original rich labels. The old roughly 41%
metric/SG match rate is close to the real training distribution and should not
have been treated as automatically defective.

## Decision

1. Keep anion and charge hard-derived from the exact composition certificate.
2. Test three predicted soft structural fields: metric lattice, a separately
   supervised SG bucket, and volume/atom bin. Validation accuracy motivates the
   test but does not establish sampled-Plan or stability superiority.
3. Sample SG after the existing volume draw, then require a byte-level
   regression proving that adding it does not perturb the previously frozen
   composition/lattice/volume RNG sequence.
4. Do not hard-enforce a one-to-one metric-lattice/SG map.
5. In the matched counterfactual arm, permute the complete predicted
   `(lattice, SG, volume)` tuple jointly to preserve its empirical joint
   distribution.
6. Treat all three as soft context with dropout during later DLM training.
7. Interpret `R0-RCF` only as alignment of the three soft fields to the hard
   composition. It is not an SG-only effect or an overall-composition effect.
8. Use the matched development canary, not teacher-forced accuracy, to decide
   whether composition-aligned rich context helps raw/refined realization.
