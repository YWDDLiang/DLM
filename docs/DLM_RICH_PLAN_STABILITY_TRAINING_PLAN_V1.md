# DLM Rich-Plan Stability Training Plan V1

Date: 2026-08-27

Status: Phase A terminal; total-2-epoch checkpoint selected, Phase B pair mining next.

## Approach

Freeze the existing H1 rich-Plan scientific contract and improve only the DLM
executor. First make the ordinary stable-data DLM sufficiently trained. Then
teach thermodynamic preference with same-exact-Plan stable/unstable body pairs,
where stability is training metadata rather than an input token. Finally repair
CFG so that it removes only soft rich-Plan context while preserving hard
composition and cardinality anchors.

## Clarifications and decisions

### Rich Plan

The canonical `h1_rich_plan_v1` contains six scientific fields and one terminal
line:

```text
formula: ...
anion: ...
charge: ...
lattice: ...
spacegroup: ...
volume: ...
end: plan
```

`N`, `elements` and `counts` are derived deterministically from `formula`; they
are execution anchors, not additional Planner-generated fields.

Historical L6 and L7 are experiment levels, not different schemas:

- L6 is the 256-attempt `freegeo256 weighted balanced` screen;
- L7 is the full-1000 `freegeo ablation_default` confirmation;
- both are compatible with the same six-field/seven-line rich-Plan contract.

This plan therefore uses one schema, L6-scale evaluation for development and
L7-scale evaluation for confirmation.

### JSON `plan_state`

The Planner emits seven-line rich text. A deterministic parser converts it into
an internal Python `plan_state`, and `build_body_prompt()` serializes that state
as JSON for the DLM. Teacher training rows and inference rows both reach the DLM
as JSON, so there is no literal train/inference format mismatch.

The narrower audit risk is provenance and redundancy:

- teacher `plan_state` is derived from a known MP-20 structure;
- inference `plan_state` is parsed from a model-generated rich Plan;
- the JSON repeats formula-derived `N/elements/counts` and includes internal
  bookkeeping fields, some of which may be `unknown` at inference.

V1 does **not** change serialization. Prompt compaction is deferred to a later
ablation so it cannot confound training-duration and stability-supervision
effects.

### Stability signal

- Do not add `target_stability`, `E0`, an energy value or a stable/unstable token
  to the DLM prompt.
- Do not make stability a seventh Planner field.
- Use stable/unstable and continuous energy only as sidecar training metadata.
- Do not train ordinary reconstruction CE on unstable bodies; they enter only
  the preference/rejection term.

## Scope

- In:
  - frozen six-field/seven-line rich Plan;
  - current JSON `plan_state` DLM serialization;
  - sufficient stable-data LoRA training;
  - origin-shift augmentation after implementation audit;
  - same-Plan energy preference training without RL;
  - hard-anchor-preserving rich-Plan CFG;
  - L6 256-attempt screen and L7 requested-1000 confirmation.
- Out:
  - Planner retraining or proposal-distribution optimization;
  - explicit stability prompt tokens;
  - inference reranking or best-of-K reporting;
  - policy-gradient/RL fine-tuning;
  - changing model494, Direct, novelty or official-hull contracts;
  - changing the public `105/488` result before a full pass.

## Frozen scientific question

> Given the same exact rich Plan, composition, N, DLM noise streams and frozen
> refiner, can sufficient stable-data training plus same-Plan thermodynamic
> preference improve Strict and Meta stable conversion without reducing body,
> Direct or novelty yield?

## Phase A — train the ordinary DLM sufficiently

### Starting point

- Original B0: 1.000 total epoch, 1,696 optimizer steps.
- Retained plain-control checkpoint: B0 plus 1,000 continuation steps,
  approximately 1.590 total epochs.
- One MP-20 exact-length epoch: 1,696 optimizer steps over 27,136 rows under the
  audited world-size/accumulation contract.

### Continuation schedule

Continue only the retained plain-control checkpoint:

| Target | Total steps from original base | Additional steps from retained checkpoint |
|---|---:|---:|
| existing | 2,696 | 0 |
| 2 total epochs | 3,392 | 696 |
| 3 total epochs | 5,088 | 2,392 |
| 4 total epochs | 6,784 | 4,088 |
| 5 total epochs | 8,480 | 5,784 |

Training contract:

- standard masked DLM CE only;
- LoRA rank 8/alpha 32 and the same model/tokenizer;
- start LR `1e-5`, cosine decay, 100-step warmup;
- gradient clipping 1.0;
- save at each integer total epoch;
- keep full rich Plan and current JSON serialization;
- enable random origin-shift augmentation only after verifying that it is
  applied to every train epoch and never to validation/test.

Checkpoint selection must use a frozen validation Plan cohort with downstream
body/refiner/CHGNet metrics. Validation CE alone is not a selection endpoint.
Continue from 3 to 5 epochs only while held-out stable conversion or energy
quantiles improve without novelty collapse.

## Phase B — build same-Plan thermodynamic pairs

### Cohort

- Use train-only P0 rich Plans disjoint by exact formula and Plan identity from
  L6 validation and L7 test.
- Begin with 256 Plans × 4 independent DLM/refiner streams to estimate usable
  pair yield.
- Expand to 512–1,000 Plans × 4–8 streams only if the pilot yields enough
  confident within-Plan pairs.
- Freeze D1 exact-plan decoding, temperature, support masks, sample-index RNG
  and model494.

### Labelling

For every successful body:

1. preserve the original DLM token sequence;
2. run frozen model494;
3. compute compatible CHGNet formation energy and hull diagnostics;
4. optionally require agreement with MatterSim for strong pairs;
5. retain unknown hull as missing, never unstable;
6. record Direct validity but do not train on novelty/uniqueness.

Within an exact composition the phase-diagram reference is constant, so
ordering structures by compatible formation energy is equivalent to ordering
them by `E_hull`.

Construct a pair only when:

- both bodies match the identical exact Plan;
- both reconstruct and have known compatible energies;
- the positive has lower energy than the negative by at least
  `0.06 eV/atom` initially;
- the energy ordering is not contradicted by a second MLIP when available.

Use the lowest-energy valid body as `y+` and the highest-energy valid body as
`y-`. A body may appear in at most one primary pair per Plan in V1 to prevent a
small number of Plans dominating training.

## Phase C — preference-train without a stability condition

For the same random mask/timestep on `y+` and `y-`, define the sequence score as
negative mean geometry-token denoising CE:

```text
s_theta(y | P, mask) = -mean_masked_geometry_CE(theta, y, P, mask)
```

Use the selected sufficient ordinary DLM as a frozen reference:

```text
delta_theta = s_theta(y+|P) - s_theta(y-|P)
delta_ref   = s_ref(y+|P)   - s_ref(y-|P)

L_pref = softplus(-beta * (delta_theta - delta_ref))
L_total = L_stable_MP20_CE + lambda * L_pref
```

Rules:

- use identical mask positions and corruption noise for each pair;
- score only free geometry positions, never prefilled N/elements/counts;
- keep ordinary CE on original stable MP-20 bodies and stable generated
  positives;
- never apply reconstruction CE to unstable negatives;
- calibrate `lambda` by gradient norm so preference gradients initially equal
  5–10% of CE gradient norm;
- test one conservative `beta` first; do not launch a grid unless the pilot
  shows positive held-out energy movement;
- log preference margin, reference margin, CE, pair count and energy-gap bins.

This is supervised pairwise preference learning, not RL, and stability labels
never appear in the prompt.

## Phase D — repair rich-Plan CFG

The current CFG implementation masks every prompt token in its baseline branch.
That removes formula, N and element counts, so its logit difference conflates
composition, cardinality and soft rich fields.

Replace it with two prompts that share hard anchors:

```text
full branch:
  formula/N/elements/counts + anion/charge/lattice/spacegroup/volume

hard branch:
  identical formula/N/elements/counts
  anion/charge/lattice/spacegroup/volume -> unknown
```

Then compute:

```text
logits = logits_hard + w * (logits_full - logits_hard)
```

- `w=1` reproduces ordinary full-rich conditional sampling;
- `w>1` amplifies only the five soft rich fields;
- no stability signal is used by CFG;
- use the same body-noise ledger for every `w`;
- rename the argument to `rich_plan_cfg_weight` to avoid the current ambiguous
  `cfg_scale + 1` convention.

Train the DLM with 10–20% soft-rich dropout rows so the hard branch is in
distribution. Formula, N, elements and counts must never be dropped.

CFG is a separate inference ablation. Preference-trained weights must first be
evaluated at `w=1`; CFG is retained only if it improves conversion rather than
merely increasing agreement with noisy soft Plan fields.

## Sequential experiment matrix

### Gate A — training sufficiency on L6-scale validation

| Arm | Weights | Preference | Rich CFG |
|---|---|---|---|
| A0 | existing total 1.59 epochs | off | `w=1` |
| A1 | total 3 epochs | off | `w=1` |
| A2 | total 5 epochs, only if A1 has not plateaued | off | `w=1` |

Freeze one sufficient-base checkpoint before building the preference candidate.
All checkpoint outcomes remain reported; do not retain only the best row.

### Gate B — thermodynamic preference on L6

| Arm | Weights | Preference | Rich CFG |
|---|---|---|---|
| B0 | frozen sufficient base | off | `w=1` |
| B1 | same initialization/budget | on | `w=1` |

Run two body/refiner seeds. Promote only if pooled Strict and Meta directions are
positive and both seeds are disclosed.

### Gate C — CFG ablation on the same B1 weights

| Arm | Rich CFG |
|---|---:|
| C0 | `w=1.0` |
| C1 | `w=1.25` |
| C2 | `w=1.5` only if C1 remains valid/noncollapsed |

No retraining and no best-of-K selection are allowed in this gate.

### Gate D — L7 requested-1000 confirmation

Compare only the frozen sufficient base with the single preselected candidate,
using identical first-1000 rich Plans, sample ordinals and refiner streams.
Preserve every failure in the 1,000-attempt denominator and run fresh official
MP hull evaluation once.

## Promotion criteria

### L6 development gate

- requested denominator: 256 per seed;
- parsed/body/Direct-joint/novelty each no worse than `-1 pp`;
- pooled Strict and Meta attempt directions both positive;
- no seed may show a material Meta collapse;
- aspirational absolute target: Strict at least `26/256`, Meta at least
  `128/256` on the promoted pooled-equivalent rate;
- stable counts, energy quantiles and stable-to-S.U.N. retention reported
  separately.

### L7 confirmation gate

- requested denominator: 1,000 per arm;
- Strict at least `100/1000` and Meta at least `500/1000`;
- Strict and Meta both improve relative to the matched sufficient base;
- body, Direct joint, novelty and both retention rates each no worse than
  `-1 pp`;
- official unknown remains excluded from known denominators and failed in the
  all-attempt lower-bound view only by missingness, never relabelled unstable;
- paired exact McNemar and energy-distribution shifts fully reported;
- public `105/488` remains unchanged until all gates pass.

## Stop rules

- Stop duration continuation if stable yield worsens across two consecutive
  checkpoints while novelty also contracts.
- Stop preference training if train margin rises but held-out same-Plan energy
  or stable conversion does not improve at two evaluations.
- Disable CFG if it only increases soft-field adherence while energy or S.U.N.
  worsens.
- Do not call a Strict-only or Meta-only gain a complete contribution.
- Do not launch Planner retraining to compensate for a DLM failure.

## Code changes

1. Add train-only same-Plan pair metadata and identity audits to a new data
   builder under `scripts/`.
2. Extend `src/scripts/llada_sft.py` collators with positive/negative sequences,
   shared masks and reference-score inputs.
3. Add the reference-corrected geometry-only preference loss behind a default-
   off CLI flag.
4. Add online/offline origin-shift augmentation tests for exact dynamic bodies.
5. Add a helper that produces full-rich and hard-anchor-only `plan_state`
   prompts without changing formula-derived fields.
6. Replace whole-prompt CFG in the experimental sampler with explicit paired
   prompt tensors; leave legacy behavior frozen for historical reproduction.
7. Add unit tests for Plan identity, shared corruption masks, no-negative-CE,
   hard-anchor preservation and `w=1` equivalence.
8. Add L6/L7 ledger assemblers and finalizers that preserve all requested
   ordinals.

## Action items

- [x] Freeze and fingerprint the L6 validation and L7 test Plan cohorts.
- [ ] Audit teacher/inference `plan_state` key/value distributions without
      changing prompt serialization.
- [ ] Implement and test exact-body origin-shift augmentation.
- [x] Continue the plain-control checkpoint through total 2 and 3 epochs.
- [x] Select the sufficient base using the validation Pareto contract (total 2 epochs).
- [ ] Generate train-only same-Plan multi-seed bodies and frozen-refiner labels.
- [ ] Build conservative energy pairs and publish pair-yield diagnostics.
- [ ] Implement and train the default-off preference candidate.
- [ ] Implement hard-anchor-preserving rich-Plan CFG and run its inference-only
      ablation.
- [ ] Run the two-seed L6 gate, then one frozen L7 requested-1000 confirmation.
- [ ] Write MD/JSON/CSV reports, test, commit and push only after the scientific
      state is terminal.

## Expected artifacts

- `DLM_TRAINING_SUFFICIENCY_FINAL.md/json/csv`
- `DLM_STABILITY_PAIR_DATA_MANIFEST.md/json`
- `DLM_STABILITY_PREFERENCE_L6_FINAL.md/json/csv`
- `DLM_RICH_CFG_L6_FINAL.md/json/csv`
- `DLM_STABILITY_L7_FINAL.md/json/csv`

Failed and duplicate model weights are removed only after the corresponding
metrics, manifests and terminal reports are safely copied back. Retain one
sufficient-base reference and one final candidate until the program is closed.
