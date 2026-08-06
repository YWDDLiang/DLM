# H1-A2 R03 Safe-Axis DLM Reproducibility and Attribution Report V1

Status: `frozen_report_only`

Report date: 2026-08-03

Scientific status:

- R03B, R03C, and R03D passed their preregistered body-generation safety
  gates.
- R03E completed cleanly but stopped scientifically because the candidate
  improved strict S.U.N. while reducing meta S.U.N.
- R03F stopped at its zero-unknown engineering gate.
- R03G is a complete lower-bound, report-only S.U.N. recomputation.
- R03H is a complete read-only attribution of the R03G endpoint changes.
- `formal_g3=false`, `automatic_promotion=false`,
  `automatic_training=false`, `checkpoint_reselection=false`, and
  `automatic_downstream=false`.

This report records the complete R03 experiment chain needed to audit or
reproduce the result. It does not authorize a rerun and does not change any
frozen terminal decision.

## 1. Executive scientific conclusion

The R03 evidence supports a qualified but useful conclusion:

> The exact-length body DLM is a successful module for producing parseable,
> structurally valid crystal proposals under the H1 contract. Its decoding
> schedule can be changed safely when the schedule preserves the coordinate
> mask's causal precondition. The remaining dominant direct-validity
> bottleneck is the Planner's proposed composition, not the DLM or the
> continuous refiner.

The evidence does **not** support the stronger claim that safe-axis decoding
improves the complete generator on every stability objective. Relative to the
frozen H1 schedule, safe-axis:

- removed the catastrophic duplicate-coordinate failure introduced by the
  original mixed-axis PlanGraph schedule;
- raised body completion from 246 to 248 of 256;
- raised pooled raw joint validity by 5 of 1024 continuous-refiner
  realizations;
- raised the completed-snapshot lower-bound strict S.U.N. count from 99 to
  117 of 1024;
- reduced the corresponding meta S.U.N. count from 523 to 496 of 1024.

R03H showed that this is a polarization of the finite-hull distribution:
safe-axis creates more exact-zero-hull outcomes but also more outcomes above
0.1 eV/atom. Therefore:

- **DLM mechanics:** successful;
- **safe-axis as a broad stability improvement:** not passed;
- **next model bottleneck:** Planner chemistry/composition;
- **next experimental rule:** change one Planner factor while keeping the
  successful DLM, exact-length contract, refiner, seeds, denominators, and
  evaluator frozen.

## 2. System boundary and causal decomposition

The full generator is:

```text
H1-A2 Planner P0
  -> canonical seven-line Plan
  -> exact-length R5-C body DLM B0
  -> proposal graph
  -> frozen CrysLLMGen model_494, exact 800 reverse steps
  -> Direct evaluator
  -> CHGNet/novelty/MP-hull S.U.N. evaluator
```

The R03 treatment changes only the ordering in which masked body-coordinate
positions are revealed:

```text
control:   P0 + B0 + D1
candidate: P0 + B0 + D2_SAFE_AXIS
```

It does not change the Planner output, body weights, body prompt, tokenizer,
count/element prefill, schema masks, composition, stateless body noise,
continuous-refiner checkpoint, per-ordinal refiner noise, Direct evaluator,
S.U.N. thresholds, or raw denominator.

This separation is essential when interpreting validity:

| Failure location | Assigned module | Meaning |
|---|---|---|
| Plan cannot be parsed | Planner | No body invocation |
| Planned composition is SMACT-invalid | Planner | Body is obeying an invalid chemical request |
| Body parse/graph construction fails | body DLM or schedule/mask interface | Discrete generation failure |
| Refined structure fails distance/volume test | continuous proposal/refiner path | Geometric failure after body success |
| Novel-unique structure crosses an `E_hull` threshold | end-to-end pipeline | Stability endpoint, not a pure DLM metric |

## 3. Frozen identities

### 3.1 Planner

| Field | Frozen value |
|---|---|
| Role | `P0`, H1-A2 epoch-2 Planner |
| Base | `/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/` |
| Adapter | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final` |
| Adapter SHA-256 | `65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a` |
| Schema | `h1_rich_plan_v1_seven_lines` |
| Plan provenance | model-proposed |

The first-256 frozen ledger contains two Planner parse failures, ordinals 86
and 211. They remain failed in both arms and are never sent to the body model.
No Plan is repaired, replaced, filtered, or regenerated.

### 3.2 Body DLM

| Field | Frozen value |
|---|---|
| Role | `B0`, R5-C exact-length body |
| Base | `/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct` |
| Adapter | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final` |
| Adapter file | `adapter_model.safetensors` |
| Adapter bytes | `6,391,016,776` |
| Adapter SHA-256 | `5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d` |
| Tokenizer vocabulary size | `128830` |
| Vocabulary SHA-256 | `3acc073da85047265769f2dccd93543fa9d7cbfa95021aef54ef282b13ce2f37` |
| `tokenizer.json` SHA-256 | `3a21588abca8e56155cc7b6cabb81df51992ccd2e89704aec770912f24e75509` |
| Tokenizer-config SHA-256 | `8e89acaa54a8fb8fc7d228165ac483f61b7fef7c4c9761214092511190f75de2` |
| Output-length contract | exactly `7 + 4N` answer tokens |
| Temperature | `0.7` |
| CFG scale | `0.0` |
| Remasking | `low_confidence` |
| Maximum body batch | `8` |
| Frozen constraints | schema logit mask, count prefill, element prefill, Plan composition, duplicate-coordinate mask, lattice-volume mask |

`N` is the atom count frozen by the accepted Plan. Both arms have the same
`N`, element sequence, body prompt, active token positions, and output length.

### 3.3 Attempt and noise ledger

| Field | Frozen value |
|---|---|
| Path | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1/data/attempt_ledger.jsonl` |
| SHA-256 | `24295854aac87f3eb9ad7cc293f2bf2d2eb1d8c292b7f05aeaad8348b6665c8f` |
| Scientific ordinals | `0..255` |
| Body-eligible ordinals | `254` |
| Body-ineligible ordinals | `86, 211` |
| Sampling policy | paired stateless per-ordinal seeds |

The four R03E repeats reuse the same scientific seed ledger. A repeat ID
identifies an independent CUDA process realization; it is not a new
scientific seed.

### 3.4 Continuous refiner

| Field | Frozen value |
|---|---|
| Model | CrysLLMGen `model_494` |
| Checkpoint | `/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt` |
| SHA-256 | `573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e` |
| Training timesteps | `1000` |
| Reverse steps used | exactly `800` |
| Evaluations per proposal | `1` |
| Effective batch | `1` |
| Noise source | frozen per-ordinal `refiner_noise_seed` |

Every body-success proposal must complete the exact 800-step refiner before
Direct or S.U.N. evaluation. An unrefined proposal is never scored as a
successful generated structure.

## 4. The schedule intervention

### 4.1 D1 control

The H1 control reveals positions in the following global axis order:

```text
prefilled count/elements
  -> lattice
  -> all X coordinates
  -> all Y coordinates
  -> all Z coordinates
```

This ordering is coupled to the frozen duplicate-coordinate mask. The mask
acts on a candidate Z token only after its X/Y pair is known, so D1 gives the
mask the information it needs.

### 4.2 Rejected mixed-axis D2

The original PlanGraph D2 grouped all X/Y/Z positions for an
element-multiplicity group and then revealed the most confident token in that
mixed group. A Z token could be committed before its X or Y values. The
duplicate mask does not mask X or Y, so later coordinate decisions could
complete an exact duplicate without encountering the guard.

Observed on the paired first-32 screen:

- D1: 31/32 body successes;
- mixed-axis D2: 14/32;
- 18 new `body:DuplicateCoordinateError` failures;
- 93 exact duplicate-coordinate pairs in D2 versus 0 in D1;
- unique-Z fraction fell from `0.65496` to `0.51105`;
- exact paired McNemar `p=1.52587890625e-05`.

This is a schedule/constraint-interface failure, not evidence of a changed
Planner, checkpoint, tokenizer, seed, or GPU.

### 4.3 D2_SAFE_AXIS candidate

The accepted safe-axis treatment preserves PlanGraph grouping inside each
axis while restoring the mask precondition:

```text
prefilled count/elements
  -> lattice
  -> PlanGraph-grouped X blocks
  -> PlanGraph-grouped Y blocks
  -> PlanGraph-grouped Z blocks
```

Required invariants for every body-eligible ordinal:

1. exact `7 + 4N` position coverage;
2. treatment applied;
3. no mixed-axis coordinate group;
4. `z_before_xy_count = 0`;
5. every active X/Y position precedes every active Z position;
6. the same count/element prefill and schema constraints as D1.

The implementation is in each R03B/C/D source directory as
`safe_axis_schedule.py`; the source manifest, not a working-tree name, is
the authoritative identity.

## 5. Experimental ladder and immutable evidence

The small-to-large ladder was deliberate. Expansion occurred only after the
previous stage passed with no new failure class.

| Stage | Comparison and scope | Execution | Source SHA-256 | Terminal/failure SHA-256 | Result |
|---|---|---|---|---|---|
| R03 | D1 vs rejected mixed-axis D2, paired 32, body only | job `29669` | `1b27f9f28ecf0d17f1a28aaf886718a457631f6035e23e883918d2499b16e365` | `898191bfe23b66ecf811eb8b223d1a7356181b273a8468c2439a926d213b09e3` | D1 31/32; D2 14/32; scientific stop |
| R03B | D1 vs safe-axis, paired 32, body only | job `29837` | `cd21ee664f5c227698e96187906944a7d29016bd270da645c56ed6501cb3c866` | `57d53922bd62e96aca2becee0b6f9c48d067111421c8e20a0b30dc1507b735c7` | 31/32 vs 31/32; gate pass |
| R03C | D1 vs safe-axis, paired 64, body only | job `29844` | `2c3485f8cf539ed5e33f3303a695947a2bd5fe72e83456d6f6f20cd5dca156bd` | `34eb0c1804c9b9dfbb43009eea56b85438e15c8b27dc81c70dbb34044852824a` | 61/64 vs 63/64; gate pass |
| R03D | D1 vs safe-axis, paired 256, body only | job `29862` | `6b0dd8298c9a423712a3965b00f8aeae9c06c824a37fbf1d94dbbf94aebcf15f` | `fa03cfa22d0765311bd55e350d1547b6180cede410d8c12054e45306279f002c` | 246/256 vs 248/256; gate pass |
| R03E | frozen R03D bodies; refine800, Direct, frozen-cache S.U.N.; four process repeats | `29912_[0-3]%2 -> 29916` | `7beaf38b7d378ecf8fb31627f195f5fe6095350b60d99882c0a11b39cbf211a4` | `7fa49a6feb372d4d5e5dc442a187657a39001b6190dd6a64af0bf7f53c293b02` | complete; scientific stop on meta |
| R03F | MP-hull completion only | A800 login-node CPU sidecar | `c26e141d130bd640eefe23b4fcce08fe464eea0f99c52125956b655f51d1a485` | failure `4380b0f270db959264004da959bc8c4afe78a7e862be79aa1667d31133349020` | all queries complete; zero-unknown gate failed closed |
| R03G | offline recomputation with only zero-unknown gate disabled | A800 login-node CPU sidecar | `06fa4bcd2fc4e340906c9e2396af2c6cbe7f7fb9178156d4112eba96201f4858` | `3cb705ea9b572f37dc46696a8aa59a8ae90a0ec90db4789d341052bab7d3bfff` | complete; lower-bound report only |
| R03H | read-only paired ordinal attribution | A800 login-node CPU sidecar | `770b4b1407db1386d3db6131b3bac43259db601ea6729c3878f36d50428fe151` | report `566dd3c59cbfaa04243c923f42ef2f726d50a1441fd1e7f5fd2796ff847b42be` | attribution complete |

Supporting R03D identities:

- generation report:
  `f86790015fd090f5243e8785aabf7b0c9ee8ddd8c85dbf02fb5dafd0acb4b880`;
- schedule-invariant report:
  `560155678cad0ce00737b0052301d3be46da0ace18d00f871d83764477b61d3d`;
- control body attempts:
  `7f486fd61dd4d73ebdf10a065e344a208a7dd274a499f40f9b8a9502cf6908c1`;
- candidate body attempts:
  `c030f5548e94f1bf4cdaaecf3614417acc998b9c320bb1c6a6436d740767364f`;
- control proposal graphs:
  `b7506859563b1282fd41cde19d740a5d7fb9f3bafd61ec3665c9718e13469e78`;
- candidate proposal graphs:
  `023fdd56dd786c788b8219f66d91eea2a9933991c522bbddfc04218ede8a8e8e`.

## 6. Pairing, denominators, and inference

### 6.1 Primary denominator

All primary results use the raw all-attempt denominator:

- R03B: 32 attempts per arm;
- R03C: 64 attempts per arm;
- R03D: 256 attempts per arm;
- R03E/G/H: 256 attempts per arm per repeat.

Planner failures, body failures, refinement failures, invalid structures,
non-novel samples, non-unique representatives, and unknown hulls remain in
the denominator and score false for downstream conjunctions. There is no
sample replacement.

### 6.2 R03E repeat design

The arm order was frozen before refined candidate metrics:

| Repeat | First arm | Second arm |
|---:|---|---|
| 0 | control | candidate |
| 1 | candidate | control |
| 2 | candidate | control |
| 3 | control | candidate |

Each repeat is one independent CUDA process on one A800, with both arms
packed inside that process. The repeat order balances first/second-arm
effects. The same scientific ordinals and seeds are reused in every repeat.

The pooled 1024 counts are descriptive. They are not treated as 1024
independent scientific samples. Registered inference includes:

- paired exact McNemar tests within the raw ordinal mapping;
- a deterministic hierarchical paired bootstrap that resamples repeat blocks
  and ordinals;
- per-repeat sign stability;
- 50,000 bootstrap replicates with the registered seed.

### 6.3 Prohibited operations

Every stage records:

```text
retry=false
replacement=false
repair=false
filter=false
rerank=false
```

No failed sample is repaired or replaced. No sample is selected because of a
Direct or S.U.N. outcome.

## 7. Evaluation definitions

### 7.1 Direct evaluator

The Direct evaluator is the frozen CrysLLMGen `GenEval/mp20` path:

- composition validity: SMACT chemical validity;
- structure validity: all pairwise distances at least 0.5 Å and volume at
  least 0.1;
- joint validity: composition-valid and structure-valid.

Two aggregations must not be mixed:

1. **raw all-attempt:** upstream failures remain false; this is primary for
   paired causal comparisons;
2. **H1-aligned successful/refined:** condition on structures that completed
   generation and refine800; this is used only to compare with historical
   H1/CrysLLMGen Direct tables.

Historical H1-A2 selected the first 1,000 accepted outputs from a larger Plan
pool. That survivor-prefix difference must be disclosed whenever its table is
shown beside raw all-attempt R03 values.

### 7.2 S.U.N.

For every raw attempt:

```text
S.U.N. = stable/metastable AND unique-representative AND novel
```

- strict: `E_hull <= 0.0 eV/atom`;
- meta: `E_hull <= 0.1 eV/atom`;
- novelty/uniqueness and CHGNet relaxations are frozen from R03E;
- R03G uses one common 227-chemical-system compatible-entry snapshot for all
  eight repeat × arm evaluations;
- residual unknown hulls remain false.

R03F queried 107 deduplicated missing chemical systems with zero transport
retries and wrote the common snapshot:

`56f91774c798854d253c0726773593c415456a8b5361f31802c44d8e1bbad917`.

It then failed closed because the registered zero-unknown condition was not
met. R03G changed only that engineering gate, used no network or API key, and
did not rerun generation, refinement, CHGNet, Direct, novelty, or uniqueness.

## 8. Results

### 8.1 Body-generation ladder

| Stage | Control success | Candidate success | Candidate minus control | Duplicate-coordinate failures |
|---|---:|---:|---:|---:|
| R03 mixed-axis, 32 | 31/32 | 14/32 | -17 | 0 vs 18 |
| R03B safe-axis, 32 | 31/32 | 31/32 | 0 | 0 vs 0 |
| R03C safe-axis, 64 | 61/64 | 63/64 | +2 | 0 vs 0 |
| R03D safe-axis, 256 | 246/256 | 248/256 | +2 | 0 vs 0 |

R03D paired completion:

| Pair state | Count |
|---|---:|
| both success | 243 |
| candidate only | 5 |
| control only | 3 |
| both fail | 5 |

The R03D point gain is `2/256 = 0.78125` percentage point, with exact
McNemar `p=0.7265625`. R03D establishes non-inferiority/safety and a small
point improvement, not a statistically established body-completion gain.

### 8.2 R03E pooled raw Direct results

Four repeats, 1,024 raw attempts per arm:

| Arm | Generation complete | Composition valid | Structure valid | Joint valid |
|---|---:|---:|---:|---:|
| D1 control | 984 | 848 | 982 | 846 |
| safe-axis candidate | 992 | 852 | 989 | 851 |
| Candidate minus control | +8 | +4 | +7 | +5 |

These raw counts deliberately include upstream failures.

### 8.3 H1-aligned Direct reaggregation

Conditioning only on successful, refine800-complete structures:

| Arm | Refined denominator | Composition valid | Structure valid | Joint valid |
|---|---:|---:|---:|---:|
| D1 control | 984 | 848/984 = 86.1789% | 982/984 = 99.7967% | 846/984 = 85.9756% |
| safe-axis candidate | 992 | 852/992 = 85.8871% | 989/992 = 99.6976% | 851/992 = 85.7863% |

Historical locally frozen references:

| System | Composition valid | Structure valid | Joint valid |
|---|---:|---:|---:|
| Historical CrysLLMGen, 1,000 | 89.2% | 99.9% | 89.1% |
| H1-A2 epoch 2, 1,000 | 87.8% | 99.9% | 87.7% |

Interpretation:

- structure validity remains essentially saturated for all systems;
- the R03 composition gap is much larger than the true post-refiner
  structure-failure count;
- safe-axis changes completion and the stability distribution, but it does
  not repair invalid Planner chemistry.

The H1-aligned values are a secondary audit. The primary R03 treatment effect
continues to use raw all-attempt counts.

### 8.4 Frozen-cache R03E S.U.N.

| Arm | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|
| D1 control | 50/1024 | 314/1024 |
| safe-axis candidate | 54/1024 | 286/1024 |
| Candidate minus control | +4/1024 | -28/1024 |

The preregistered R03E gate required a positive mean meta effect with at least
three of four repeats non-negative. That condition failed, so the frozen
decision is `safe_axis_refined_signal_stopped`.

### 8.5 R03G completed-snapshot lower-bound S.U.N.

R03G resolved 753 of 825 source unknowns. The remaining 72 are exactly nine
per arm and repeat and score false. Existing finite parity is 974/974, with
maximum absolute reproduction error
`2.842170943040401e-14 eV/atom`.

Per-repeat control to candidate counts:

| Repeat | Strict control | Strict candidate | Delta | Meta control | Meta candidate | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 27 | 28 | +1 | 133 | 122 | -11 |
| 1 | 22 | 31 | +9 | 130 | 123 | -7 |
| 2 | 26 | 29 | +3 | 134 | 125 | -9 |
| 3 | 24 | 29 | +5 | 126 | 126 | 0 |
| Pooled | 99 | 117 | +18 | 523 | 496 | -27 |

Pooled rates:

| Arm | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|
| D1 control | 99/1024 = 9.67% | 523/1024 = 51.07% |
| safe-axis candidate | 117/1024 = 11.43% | 496/1024 = 48.44% |
| Candidate minus control | +1.7578 points | -2.6367 points |

Inference:

| Endpoint | Exact McNemar | Hierarchical paired-bootstrap 95% CI | Sign stability |
|---|---:|---:|---:|
| strict | `p=0.0198343` | `[0.0000, +3.6133]` points | 4/4 positive |
| meta | `p=0.0464644` | `[-5.5664, +0.3906]` points | 3 negative, 1 zero |

Because 72 applicable hulls remain unknown, these are conservative
lower-bound, report-only values. They do not reopen the R03E gate.

## 9. Failure attribution

### 9.1 Raw structure invalidity is mostly upstream failure

Across four repeats:

| Arm | Raw structure-invalid count | Planner parse failures | Body failures | True post-refiner structural invalidity |
|---|---:|---:|---:|---:|
| D1 control | 42 | 8 | 32 | 2 |
| safe-axis candidate | 35 | 8 | 24 | 3 |

The true refined failures are rare and arise from a minimum distance below
0.5 Å; their volumes are normal. Ordinals 31 and 145 recur. This evidence
does not identify `model_494` as the main failure source.

### 9.2 Composition invalidity follows the frozen Plan

On shared-success ordinals, both schedules retain identical formulas and
composition-validity labels. In the safe-axis candidate, the 35
composition-invalid successful ordinals are already explained by Planner
chemistry:

| Planner self-tag | Count |
|---|---:|
| charge failure | 24 |
| Pauling failure | 4 |
| neutral/plausible tag but Direct-invalid | 7 |

Historical H1 evidence is consistent:

- 1,186 parsed Plans;
- 1,044 composition-valid;
- 142 invalid;
- 98 charge-neutrality failures;
- 37 Pauling failures;
- 7 without usable oxidation states.

Therefore the correct module diagnosis is:

- body DLM: high completion and exact-length compliance;
- continuous refiner: approximately 99.7–99.8% conditional structure
  validity;
- Planner: dominant composition/joint-validity bottleneck.

### 9.3 R03H stability attribution

Residual unknowns are not responsible for the treatment effect:

- unknown ordinals are
  `11,24,44,55,60,131,170,189,217`;
- all contain Yb;
- they are paired in both arms in every repeat;
- there are 36 both-unknown pairs and zero one-arm-unknown pairs;
- contribution to strict and meta deltas is exactly zero.

Strict `+18/1024` decomposes into:

| Source | Net contribution |
|---|---:|
| finite-hull threshold crossings | +16 |
| novel-unique eligibility | +2 |
| residual unknown | 0 |

All 117 candidate and all 99 control strict-positive values have exactly
`E_hull=0.0`.

Meta `-27/1024` decomposes into:

| Source | Net contribution |
|---|---:|
| finite 0.1-eV/atom crossings | -27 |
| novel-unique eligibility | 0 |
| residual unknown | 0 |

There are 60 candidate gains and 87 candidate losses across the meta
threshold. Of 147 finite meta-discordant pairs, 89 have both arms more than
0.01 eV/atom from the threshold, so the adverse signal is not merely
floating-point boundary jitter.

The marginal state shift is:

```text
strict       +18
meta-only    -45
above-meta   +36
ineligible    -9
unknown        0
```

This is stability polarization, not broad improvement.

## 10. Reproduction procedure

### 10.1 Preservation rules

1. Never write into an H1, R5-C, R03B–H source, or completed run root.
2. Never reuse an existing run identity.
3. Never edit a source directory after computing its `SOURCE_SHA256.txt`.
4. Verify every manifest with `sha256sum -c`.
5. Use a fresh, preregistered replay source and fresh run root.
6. Preserve ordinals 0..255, including failures 86 and 211.
7. Preserve all failed attempts; do not repair or replace them.
8. Use the exact tokenizer, body adapter, refiner, and 800-step configuration.
9. Validate all requested Slurm partitions with `sinfo` before the first
   `sbatch`.
10. Do not serialize an MP credential. If MP access is required, use a
    user-owned mode-0600 temporary file, read it once, unlink it immediately,
    and keep it out of command lines, logs, manifests, and reports.

The project checkout has no usable Git metadata. Reproducibility relies on
immutable paths and SHA verification rather than Git rollback.

### 10.2 Source verification

From the project root on the execution cluster:

```bash
ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
cd "$ROOT"

for source in \
  workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis32_v1 \
  workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis64_v1 \
  workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1 \
  workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1 \
  workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_mpcomplete_v1 \
  workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_no_completed_hull_v1 \
  workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_attribution_v1
do
  (
    cd "$source"
    sha256sum -c SOURCE_SHA256.txt
  )
done
```

This verifies the historical sources. It does not authorize running their
fixed-path `submit_once.sh` or `run_once.sh` again.

### 10.3 Focused tests

Run the tests from the source directory using the registered execution
environment:

```bash
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

"$PYTHON" -m unittest \
  workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis32_v1/test_protocol.py

"$PYTHON" -m unittest \
  workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis64_v1/test_protocol.py

"$PYTHON" -m unittest \
  workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1/test_protocol.py

"$PYTHON" -m unittest \
  workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1/test_protocol.py

"$PYTHON" -m unittest \
  workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_mpcomplete_v1/tests/test_completion.py

"$PYTHON" -m unittest \
  workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_no_completed_hull_v1/tests/test_completion.py

"$PYTHON" -m unittest \
  workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_attribution_v1/tests/test_attribution.py
```

The exact historical local/A800 test counts are:

- R03B: 7/7;
- R03C: 7/7;
- R03D: 8/8;
- R03G source: 12/12.

### 10.4 Safe replay packaging

The historical launchers contain fixed source and run paths and refuse an
existing run root. A legitimate replay must therefore:

1. copy the frozen execution source to a new replay identity;
2. change only the source/run identity fields needed to avoid collision;
3. record that packaging-only change in a new execution manifest;
4. regenerate `SOURCE_SHA256.txt`;
5. rerun the focused tests and real-ledger preflight;
6. verify that every scientific field remains identical;
7. submit only after a second operator has reviewed the manifest diff.

Do not simply edit and run the historical source in place.

### 10.5 Body-only ladder

The historical launch interface for R03B/C/D is:

```bash
bash execution_identity/submit_once.sh
```

Each `submit_once.sh`:

- refuses an existing submission record;
- computes the source-manifest SHA;
- creates only the fresh run/log path;
- passes the SHA into the Slurm job;
- records the returned job ID and source SHA;
- sets `automatic_downstream=false`.

For a replay, preserve the original ladder:

```text
paired 32
  -> only if gate passes, paired 64
  -> only if gate passes, paired 256
  -> only under a new authorization, refined repeats
```

Do not jump directly to 256 after changing an implementation.

### 10.6 R03E refined repeats

Historical topology:

```text
packed GPU array 29912_[0-3]%2
  -> afterok assembly 29916
```

Each array element runs both arms in its preregistered order. The source
launcher submits the array and then an `afterok` assembly.

One scheduler-only incident must be preserved in the record: the source
assembly script requested a nonexistent `cpu` partition. The GPU array was
not canceled or resubmitted. Only the assembly was attached as job 29916 with
the valid `normal` partition. Repair-log SHA:

`ec223e8d0b11e86f39bbf544a26439d1e84e5945a0058c8f68174abda97fc8d0`.

Before a replay:

```bash
sinfo -h -o '%P' | sed 's/*$//' | sort -u
```

Every partition named in every `sbatch` script must occur in this list.

### 10.7 R03F/G/H evaluation-only sidecars

These stages run CPU-only on the A800 login node through the user-maintained
nested connection. They must not use Slurm or a GPU:

```bash
export CUDA_VISIBLE_DEVICES=""
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY
```

Historical entry points:

```text
R03F: preflight.sh -> run_once.sh with a one-use private key file
R03G: preflight.sh -> run_once.sh, offline
R03H: preflight.sh -> run_once.sh, offline
```

R03F is not replayable without a separately authorized fresh identity and a
new runtime-only credential carrier. R03G and R03H must reuse only the
registered immutable inputs; neither may query the network.

## 11. Terminal gates to verify

### 11.1 Body-only gate

- job state `COMPLETED 0:0`;
- raw ordinals exact and complete;
- paired input mismatch zero;
- shared eligible-row batch partition true;
- exact-length coverage for every eligible ordinal;
- treatment count exact;
- mixed-axis groups zero;
- `z_before_xy_count=0`;
- no duplicate-coordinate regression;
- no new failure class;
- `automatic_downstream=false`.

### 11.2 Refined-repeat gate

- all four repeat jobs and assembly `COMPLETED 0:0`;
- balanced arm order exact;
- body inputs and proposal graphs match registered SHAs;
- generation rerun false;
- each body success receives exact refine800 at batch one;
- Direct and S.U.N. use the registered evaluator;
- all endpoints have 256 raw entries per arm/repeat;
- paired tests, hierarchical bootstrap, and sign stability present;
- no promotion/training/downstream.

### 11.3 Offline S.U.N. gate

- one shared snapshot for all eight arms;
- existing finite parity 100%;
- every source unknown conserved as resolved or still unknown;
- residual unknowns explicitly reported and scored false;
- strict/meta vectors length 256 and `strict <= meta`;
- generation/refine/CHGNet/Direct/novelty rerun false;
- network/API/Slurm/GPU false for R03G/H;
- source R03E/R03F evidence unchanged.

## 12. Known reproducibility limitation

The continuous refiner is not bitwise deterministic. A retained zero-change
control replay established:

- body attempts were byte-identical;
- all 246 proposal graphs were tensor-exact;
- metadata, shapes, dtypes, species, and site counts matched;
- final continuous coordinates did not match byte-for-byte.

The cause is CUDA `torch_scatter` floating-point reduction order. Small
differences are amplified along the 800-step diffusion trajectory.

Consequently:

- exact identity remains mandatory before continuous refinement;
- final coordinate byte equality is not a valid scientific gate;
- paired arms must be evaluated under a preregistered common repeat ledger;
- all repeats and effect-sign stability must be reported.

The governing amendment is
`H1_REFINER_REPRODUCIBILITY_AMENDMENT_V1.md`.

## 13. Artifact locator

### 13.1 Local protocol and analysis records

| Purpose | Path |
|---|---|
| immutable H1 fallback | `workstreams/plangraph_dlm_iclr_20260731/H1_FALLBACK_MANIFEST.md` |
| one-variable recovery plan | `workstreams/plangraph_dlm_iclr_20260731/H1_SINGLE_VARIABLE_RECOVERY_PLAN_V1.md` |
| rejected D2 diagnosis | `workstreams/plangraph_dlm_iclr_20260731/H1_R03_D2_SCHEDULE_DIAGNOSIS_V1.md` |
| R03E repeat protocol | `workstreams/plangraph_dlm_iclr_20260731/H1_R03E_REFINER_REPEAT_PROTOCOL_V1.md` |
| continuous-refiner amendment | `workstreams/plangraph_dlm_iclr_20260731/H1_REFINER_REPRODUCIBILITY_AMENDMENT_V1.md` |
| operational/result index | `workstreams/plangraph_dlm_iclr_20260731/EXPERIMENT_TODO_INDEX_V3.md` |

### 13.2 Frozen execution sources

```text
workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis32_v1
workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis64_v1
workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1
workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1
workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_mpcomplete_v1
workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_no_completed_hull_v1
workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_attribution_v1
```

### 13.3 Remote run roots

```text
/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_schedule32_v1
/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis32_v1
/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis64_v1
/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis256_v1
/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis_refined_repeats4_v1
/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260803_h1_body_safeaxis_refined_repeats4_mpcomplete_v1
/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260803_h1_body_safeaxis_refined_repeats4_no_completed_hull_v1
/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260803_h1_body_safeaxis_refined_repeats4_attribution_v1
```

All are read-only evidence roots.

## 14. Claim boundary for the next paper stage

Supported:

1. exact-length DLM generation is robust under the frozen H1 contract;
2. decoding order is a causal design variable with a concrete
   schedule/constraint interface;
3. safe-axis preserves the useful PlanGraph grouping while eliminating the
   original duplicate-coordinate collapse;
4. the body/refiner path achieves approximately 99.7–99.8% conditional
   structural validity;
5. safe-axis produces a repeat-stable strict S.U.N. gain in the registered
   lower-bound evaluation;
6. current composition/joint validity is primarily Planner-limited.

Not yet supported:

1. safe-axis improves broad meta stability;
2. the four CUDA process repeats are four independent scientific sample
   sets;
3. R03G is a zero-unknown completed-hull estimate;
4. the current result is formal G3 or an automatically promotable system;
5. an external SOTA claim under a common public evaluator and equal sample
   budget.

The next scientific experiment should therefore preserve the successful DLM
path and isolate exactly one Planner chemistry intervention. It should target
charge neutrality and Pauling compatibility before combining Planner and
body changes.

