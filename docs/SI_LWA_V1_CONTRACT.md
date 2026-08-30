# SI-LWA-v1 prospective implementation contract

Date: 2026-08-30  
Status: **preregistered; GPU blocked pending intent-data and prospective-test freeze**

## Objective and module boundary

SI-LWA-v1 tests whether a masked DLM can recover global structural intent and
learn same-composition continuous-energy ordering without moving composition
selection, adding inference-time search, or delegating structure to the
Planner.

- C3FD emits only exact composition and atom count `N`.
- The frozen `ctv_minimal_base_36898` step696 adapter is the initialization and
  reference policy.
- The DLM predicts two internal structural intents and the complete dynamic
  crystal body.
- Frozen model494 at tau800 performs identity-preserving continuous refinement.
- No external rich Plan, AR executor, reranking, replacement, best-of-N,
  composition tilt, checkpoint selection, or test-outcome tuning is allowed.

## Frozen source identities

| Role | Path | SHA-256 |
|---|---|---|
| MP20 train | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/reference/crysllmgen/data/mp_20/train.csv` | `9b8031cf4ea7bb62709c74735da7ec11d00e367c5eaa05658fad5b5e7a530dde` |
| MP20 validation | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/reference/crysllmgen/data/mp_20/val.csv` | `ae7b87064fb63cdd9b2e3a69b4695046edede96c96b652c5aae7fcf195aff398` |
| listwise manifest | `$ROOT/data/d3po_listwise_groups_v1_20260830/D3PO_LISTWISE_GROUP_MANIFEST.json` | `4a7b0f8b56e03e5d8dfdf0f515e7b37a7035efea5a6072c03635d8d1cad51fb9` |
| listwise train | `$ROOT/data/d3po_listwise_groups_v1_20260830/train.jsonl` | `5d8d4db03725a8439b1498539d6efcfdc5b41b9eac2f186172ca1b48e1f56b8b` |
| listwise validation | `$ROOT/data/d3po_listwise_groups_v1_20260830/validation.jsonl` | `e20c0ed78b4c8a61bf09de42e4bd5561cdca1f607e992bab5ab91e99160318fc` |
| base adapter config | `$BASE/adapter_config.json` | `8101ee2a917dd1b08d5ef5d90472207a01161a6bcd2b03c78f9e037e756e6300` |
| base adapter weights | `$BASE/adapter_model.safetensors` | `6ea3c2a633706968e4b3e3cf77e98e46399c23e1568333522283472634553ecb` |

The listwise asset contains 886 train groups, 166 chemical-system-held-out
validation groups, and 3614 physically distinct candidates with `K=2..8`.
Every composition has total weight one. Its manifest records that D3PO main and
sealed holdout outcomes were not read.

## New intent-label asset

Only MP20 train and validation CIFs are read. Test CIFs and all generated-test
outcomes are unsupported by the builder CLI.

### VPA_Q8

1. Parse the CIF into a periodic pymatgen `Structure`.
2. Compute `v = log(volume / number_of_sites)`.
3. Fit seven internal quantile boundaries at `1/8, ..., 7/8` using valid MP20
   train rows only.
4. Assign train and validation rows to classes `0..7` with those frozen edges.

### CN_ENV8

1. For each site, obtain CrystalNN neighbor weights and sum them into an
   effective coordination number.
2. Build a normalized eight-bin histogram over effective-CN intervals
   `[0,2), [2,4), ..., [12,14), [14,+inf)`.
3. Fit eight deterministic train-only representatives with fixed seed 82000;
   each representative must be an observed train histogram nearest its fitted
   cluster center.
4. Assign train and validation rows to the nearest frozen representative under
   squared Euclidean distance, breaking ties by lower class index.

Every parse/CrystalNN failure remains a labelled missing row in the audit and is
never silently repaired. The immutable output contains train/validation JSONL,
quantile edges, representative histograms/material IDs, class counts, entropy,
coverage, failure reasons, source identities, code hashes, output hashes, and
`_SUCCESS`.

## Candidate safety augmentation

The existing listwise rows already contain exact composition, prompt, body
answer, source identity, and post-model494 CHGNet energy. A new immutable join
adds only train-source evidence that can be traced without reading prospective
test outcomes:

- raw Direct validity when available;
- raw CHGNet energy when available;
- explicit missingness reason otherwise;
- within-composition refined rank;
- within-composition raw rank among raw-valid/known candidates;
- deterministic best-valid anchor.

Raw-invalid candidates are lexicographically worst. Absolute energies are never
compared across compositions. No missing raw value is imputed from a different
composition or from a test run.

## Representation and model

The original dynamic `7+4N` body positions and tokenizer vocabulary are
unchanged. Two field-specific masked trailer states are implemented outside the
full vocabulary softmax:

```text
original dynamic body | VPA_Q[0..7] | CN_ENV[0..7]
```

- each field has an eight-way head and an eight-row learned embedding table;
- at inference both fields start masked;
- the DLM predicts intent first, freezes the selected embeddings, then executes
  the unchanged lattice -> all-X -> all-Y -> all-Z body schedule;
- parser and model494 consume only the original dynamic body;
- intent dropout is exactly 0.20 during training;
- body-to-intent reconstruction is retained to prevent posterior collapse.

## Training objective

Mixed batches are deterministic and preserve each composition's total weight.

```text
L = L_body_masked_CE
  + lambda_intent * (L_VPA + L_CN)
  + lambda_recon  * L_body_to_intent
  + lambda_list   * L_same_composition_refined_listwise
  + lambda_raw    * L_raw_validity_and_rank
  + lambda_ref    * mean(S_theta^2)
  + 0.20          * L_best_valid_denoising
```

All candidates in one group share the same typed geometry mask/noise. Refined
rewards are robustly centered inside composition. Raw invalid candidates receive
the worst safety order before raw-energy ranking. Numeric lambdas are frozen
once by gradient-norm calibration on exactly 64 train groups; no grid is run.

## Feasibility before full training

Report intent coverage, class entropy, majority-normalized accuracy, VPA ordinal
error, CN accuracy, and body-to-intent reconstruction on train and held-out
chemical systems. Then use 32 train/validation compositions and two fixed
generation/refiner streams for three matched conditions:

1. minimal base;
2. oracle train-derived intent;
3. self-predicted intent.

If oracle intent improves neither raw Direct nor raw/refined continuous energy,
self-intent is a scientific NO-GO and the run switches to listwise-only. If
oracle works but self-prediction collapses to majority-level accuracy or near-zero
entropy, the bottleneck is composition-to-intent and the run also switches to
listwise-only. External rich Plan information is not substituted.

## Full training

- initialization/reference: frozen BASE step696;
- train seeds: `82017`, `82018`;
- LoRA rank/alpha/dropout: `8/32/0`;
- learning rate: `5e-6`;
- updates: exactly `348` per seed;
- only `step-348` is retained;
- no early stopping, seed selection, checkpoint selection, or validation-driven
  update count;
- at most two A800 GPUs concurrently, eight CPUs per GPU.

Step0 must be exactly reference-equal before either optimizer starts. Every run
stores loss components, gradient finiteness, intent entropy, manifests, adapter
hashes, `_SUCCESS` or `_FAILED`, and a root-cause note.

## Prospective evaluation lock

Before any SI-LWA GPU training, freeze one new outcome-blind C3FD 256-composition
cohort from a previously unused sampling ledger. Its JSONL, source ordinal
ledger, model/checkpoint identity, sampling seed, exact-composition exclusions,
distribution audit, and SHA-256 must be added to the final contract. Until that
amendment exists, this contract forbids GPU training.

The final evaluation is one attempt per Plan:

- BASE + policy82017 + policy82018;
- streams 17/18 with frozen DLM/refiner seeds;
- temperature 0.7, exact-axis, model494 tau800;
- one six-cell generation, one twelve-cell raw/refined offline evaluation, and
  at most one fresh official query.

The independent unit is 256 compositions. Streams are averaged inside
composition before bootstrap; they are process reproductions, not independent
experimental units.

## Interpretation and fallback

- **Strong:** both training seeds and all four cells improve refined energy;
  both seed-level raw directions do not reverse; Direct/NU remain safe;
  composition-bootstrap refined CI upper bound is below zero; official hull is
  concordant.
- **Replicated suggestive:** both seed aggregates and official/Meta directions
  agree but the interval crosses zero.
- **Interface-only:** raw improves but model494 erases or reverses it.
- **Negative/unstable:** seed directions disagree or no replicated raw/refined
  left shift exists.

Historical `10/50` and H1-A2/R03 values are references, not result-deletion
gates. If the builder, heads, and oracle feasibility do not reach a complete
PASS/NO-GO within six hours of implementation start, self-intent stops and the
same two-seed/348-update prospective contract runs listwise-only with raw safety,
reference bound, and best-valid anchor. Pure AR is excluded from this sprint.

