# DLM Direct and stability decision — 2026-09-02

Status: frozen training decision after rollout capture and before any new model
weights are trained.

## Decision

Do **not** train the proposed rollout active-group CE policy. Its own
teacher-continuation realizability check fails in Y and Z.

The next admissible model is a fresh Compact-V2 DLM trained on a
**Plan-aligned canonical site order**, followed by the existing G2 periodic
relation adapter. This fixes a measured train/inference representation mismatch
without changing any crystal, composition, energy label, Plan distribution or
sampling parameter.

No stability claim follows from canonicalization alone. Raw Direct and paired
raw CHGNet are co-primary. If canonical DLM+G2 improves Direct but does not move
raw energy left, the pre-specified fallback is a stability-conditioned periodic
manifold-score executor trained only from MP20; ordinary CE continuation,
stable-row filtering and generated/refined teachers remain closed.

## 1. Why rollout CE was rejected before training

Job 39259 captured four real exact-axis states for 128 MP20-train Plans. It
changed no weights. Every source preserves committed model tokens, including
errors, and uses the paired original MP20 body only at still-masked positions.

| Continuation point | Direct | Delta vs final BASE | invalid→valid | valid→invalid |
|---|---:|---:|---:|---:|
| final BASE trajectory | 59/128 | — | — | — |
| lattice + MP20 remainder | 122/128 | +63 | 65 | 2 |
| X + MP20 remainder | 102/128 | +43 | 49 | 6 |
| Y + MP20 remainder | 54/128 | -5 | 9 | 14 |
| Z + MP20 remainder | 43/128 | -16 | 3 | 19 |

After canonicalizing the MP20 target's site records to the inference element
order, lattice remains `122/128`, X is `95/128`, Y is `57/128`, and Z remains
`43/128`. The later-stage failure is therefore not just a serialization issue:
once wrong lattice/X/Y values are committed, the original MP20 Y/Z coordinates
are not a valid conditional expert action.

This is the exact failure mode that a result-first pilot would have hidden.
Scheduled sampling addresses exposure bias when the reference action remains
meaningful under model states; here that prerequisite is violated. The method is
closed without training. See the original
[Scheduled Sampling paper](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html)
for the train/inference discrepancy it addresses; the incompatibility conclusion
is our inference from the crystal audit above.

## 2. Measured representation mismatch

Inference hard-prefills element slots by expanding the Plan's
`elements × counts` order. The existing MP20 teacher body retains the source
CIF's site order.

Full-data audit:

| Split | Rows | Exact teacher/inference element order | Rows affected | Mean mismatched slots |
|---|---:|---:|---:|---:|
| train | 27,136 | 4,069 (14.995%) | 23,067 | 6.519 |
| validation | 9,047 | 1,296 (14.325%) | 7,751 | 6.514 |

Every row has the correct species multiset. The mismatch is a physically
irrelevant site permutation, but it is not irrelevant to this model: dynamic
`7+4N` is a positional sequence and exact-axis decoding commits fixed slots.
The Transformer is globally visible but has no exact site-permutation
equivariance. G2 also builds species-conditioned edges from those slots.

Consequences:

1. training and inference use different element-slot distributions in about
   85% of examples;
2. coordinate tokens must model unnecessary site-order entropy;
3. at inference, G2 may combine canonical species slots with coordinate
   distributions learned under many unrelated site orders;
4. chemically implausible local environments can therefore hurt both Direct
   and raw energy even when composition is exact.

## 3. Canonical site-order method

For each MP20 structure:

1. keep the lattice exactly unchanged;
2. expand the Plan's element order to `N` slots;
3. stably match each requested element slot to one source site of that element;
4. move the complete `(species, fractional coordinate)` record together;
5. serialize the permuted records to the same dynamic `7+4N` body.

This is a pure site permutation. It preserves the physical structure, PBC pair
distances, composition, MP material identity, formation energy and
`e_above_hull`. It drops no rows and adds no target, filter or oracle field.

The method is intentionally minimal: no origin shift, coordinate sorting,
same-species permutation augmentation, new Planner field or decoding repair is
added in this run.

## 4. Why this is likely to help

The causal chain is direct:

```text
Plan element order == training body element order == inference prefill order
        -> stable slot semantics
        -> coordinates remain attached to the intended species slots
        -> G2 sees consistent species–distance relations
        -> fewer structural failures and more plausible local chemistry
```

It has a stronger prior than another loss experiment:

- it removes a measured 85% support mismatch;
- it is information-preserving and physically invariant;
- it changes no downstream evaluator or sampling distribution;
- exact chemistry cannot regress because N/elements remain hard-prefilled;
- G2 has already shown that a periodic residual can move raw Direct, while its
  weak raw-energy result is consistent with noisy species/slot semantics.

Calibrated expectation before training:

- probability of a meaningful raw Direct improvement: roughly 65–80%;
- probability of a paired raw CHGNet left shift: roughly 45–65%;
- probability of immediately reaching both final 10%/50% S.U.N. targets is
  lower, because model494 and hull novelty still contribute downstream.

These are engineering priors, not statistical claims. The method is admitted
because its necessary interface premise is proven, not because a favorable
checkpoint will be selected.

## 5. Frozen experiment, if data invariance passes

1. Build full canonical MP20 train/validation data: `27,136/9,047`, no subset.
2. Verify all rows preserve lattice, species-coordinate records, body length,
   composition and source provenance; train/validation ordering remains fixed.
3. Train one fresh LoRA from the shared pretrained crystal LLaDA, not from the
   old noncanonical Compact-V2/G2 checkpoint. Reuse the registered Compact-V2
   two-epoch optimizer schedule and one seed; no epoch, LR or checkpoint sweep.
4. Train one G2 adapter on the canonical DLM using the promoted G2-PBC-R
   configuration and one full epoch.
5. Evaluate OLD-G2, canonical DLM and canonical DLM+G2 on one frozen matched
   Plan/noise cell. Report body, composition and raw Direct for all arms.
6. Run paired raw CHGNet for canonical DLM+G2 whenever body/composition are
   preserved and Direct is not more than two attempts below OLD-G2. Stability
   is not skipped merely because Direct is flat.

Promotion requires both:

- Direct non-adverse (`>= OLD-G2 - 2/256`) with net invalid→valid reported;
- paired raw CHGNet known pairs `>=240`, median delta `<=-0.01 eV/atom`, and
  lower-energy fraction `>=0.55`.

A `+8/256` Direct gain and median energy `<=-0.02 eV/atom` are the meaningful
improvement targets, not mandatory deletion gates.

The promoted G2-PBC-R recipe is reused literally: rank-64 two-layer acyclic
`q0 -> soft geometry -> zero-initialized residual -> q1`, strict triclinic
125-image minimum-image geometry, normalized species-aware overlap margin
`clamp(0.55(r_i+r_j), 0.60 A, 1.40 A)`, and metric/pair-RDF/overlap/coordination
weights `0.1/0.1/0.2/0.05`. The failed detached uncertainty gate remains off.
Existing exact `7+4N`, hard chemistry, legal-family, duplicate-coordinate and
lattice-volume decoding constraints remain active. Canonical ordering is thus
the only new variable before G2; the effective geometric tricks are not lost.

## 6. Stability fallback fixed in advance

If canonical ordering improves Direct but not energy, do not add more CE epochs.
The only next candidate is a small **stability-conditioned periodic manifold
score executor**:

- positive structures and geometry targets: full MP20-train only;
- corruption: wrapped-Normal fractional coordinates plus symmetric log-metric
  lattice noise, calibrated from rollout error scales but never using generated
  structures as teachers;
- network: the already specified zero-initialized SPD lattice and PBC-torus
  coordinate executor after q0/G2;
- condition: MP20 `e_above_hull` embedded in the executor during training and
  set to `0 eV/atom` at inference;
- output: an active-stage adjacent-token logit residual, with exact chemistry
  and exact-axis commitment unchanged;
- no inference CHGNet, reranking, best-of-N or completed-sample repair.

This fallback has an explicit stability mechanism. Periodic/equivariant
denoising of lattice and fractional coordinates is supported by
[DiffCSP](https://arxiv.org/abs/2309.04475) and
[CDVAE](https://arxiv.org/abs/2110.06197). MatterGen trains on stable structures
with crystal-specific corruption and demonstrates property adapters, including
energy-above-hull conditioning; see the
[MatterGen paper](https://www.nature.com/articles/s41586-025-08628-5) and
[official implementation](https://github.com/microsoft/mattergen).
[CrystalFlow](https://arxiv.org/abs/2412.11693) independently supports
formation-energy-conditioned lattice/coordinate generation.

The proposed executor is not claimed equivalent to those full continuous
generators. The transferable principle is that stability needs a periodic
geometry denoising field and an explicit property condition; positive-only CE
on stable rows is insufficient, as our SGTC result already demonstrated.

## 7. Closed alternatives

- rollout active-group CE on noncanonical or canonical MP20 suffixes;
- more ordinary SFT epochs on the old target order;
- strict-stable row filtering/oversampling (SGTC terminal negative);
- generated or model494-refined structures as positive teachers;
- Force/BTRD projected residuals;
- post-refiner D3PO as a raw-stability mechanism;
- inference filters, reranking, best-of-N or replacement.
