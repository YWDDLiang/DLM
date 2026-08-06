# H1-A2 epoch-2 baseline dossier

Status: frozen reconstruction for the Plan + Diffusion Language Model ICLR line  
Prepared: 2026-07-29  
Scientific anchor: successful H1-A2 epoch-2, not the later step-500 continuation branch

## Executive decision

The innovation line should restart from the successful H1-A2 epoch-2 Planner adapter and keep the original R5-C body and CrysLLMGen parent refiner frozen for the first experiment.

The first change should target the Planner, because:

1. H1-A2 epoch-2 already demonstrated a real fully-de-novo Plan effect and reproduced the historical S.U.N scale.
2. Its largest direct defect is formula chemistry: 142 of 1,186 validly parsed plans fail the composition validator.
3. The current parser validates syntax and enumerated labels, but does not enforce that formula, anion, charge, lattice, space-group bucket, and volume bin form a jointly plausible tuple.
4. The later same-source step-500 conditioning continuation strengthened matched-vs-shuffled Plan identity but reduced meta S.U.N, so it is not a safe baseline for further continuation.

The selected first experiment is therefore a low-learning-rate, epoch-2-anchored Planner continuation with chemistry-valid positives and joint-plan contrastive negatives. It changes no inference-time retry, repair, filter, reranker, MLIP, or S.U.N selection rule.

## Evidence levels

This dossier deliberately distinguishes three evidence classes.

- **P — proven by a run artifact or original report:** exact values copied from the successful run, parity evaluation, frozen checkpoint, or a recorded SHA.
- **R — recovered frozen source:** exact code in the restored R5-C archive, with its SHA-256. It is the strongest available implementation evidence, but the June run did not record a source git commit proving byte-for-byte runtime identity.
- **I — inferred from source and launcher:** behavior follows from the recovered code and launch command but was not serialized as a dedicated field in the historical run manifest.

The missing June runtime git commit must remain marked missing. The recovered source must not be relabeled as a proven historical commit.

## Frozen scientific identity

| Item | Value | Evidence |
|---|---|---|
| Successful run | `runs/20260603_034533-h1a2-epoch2-3-fullmetrics` | P |
| Planner family | H1-A2 rich seven-line Planner | P/R |
| Base LLM | `/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/` | P |
| Epoch-1 warm start | `runs/20260602_182700-h1a2-rich-l3base-256/outputs/h1a2_llama_rich_sft/final` | P |
| Epoch-2 adapter SHA-256 | `65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a` | P |
| Generated plans SHA-256 | `444c909e73f53f47528d7a6a429320cb0dbe68f5974baf0ed54f8c3cb514e7dc` | P |
| Raw R5-C body SHA-256 | `a4f061c1cac023609d82a94c01b2139c04f618e15fb766e9b367d85509ff80fa` | P |
| Proposal graphs SHA-256 | `e9e0f05159592ab2c7d8cd77086c58d306f90fe83cf6d0ac5ac2128594950ffc` | P |
| Historical 1000-tensor SHA-256 | `e69598591aeb1ac0d6bf3335c532bc16379670f443b74d72f5fc1841e0744446` | P |
| Successful parity terminal report SHA-256 | `9f9f3dbd528de69f25d88b806c8e8eb8bc852454c512daeae45e62a8a3c6a62f` | P |
| Historical runtime source commit | not recorded | missing |

The restored baseline archives are:

- `r5c_frozen_baseline_20260728.tar.gz`, SHA-256 `ad1b7f5b9ee0df0c06396ef1d3865f7a5e7b2e4d3f4b46216445288e04be8325`
- `r5c_reactivation_bundle_20260728.tar.gz`, SHA-256 `63f699f670ab8c450e7e196ec824f7009ea5a4e9f6e7aee7f743d81f51d25d1b`

## Pipeline

```text
MP20 row
  -> seven-line Llama Planner target
  -> one-shot Planner sampling
  -> R5-C exact-plan LLaDA body
  -> CrysLLMGen parent refinement
  -> direct CrysLLMGen metrics
  -> original A100/CHGNet S.U.N protocol on A800
```

The seven-line Plan schema is:

```text
formula: <flat integer-count formula, total atoms 1..20>
anion: <anion class>
charge: <chemistry-status class>
lattice: <crystal-system class>
spacegroup: <space-group bucket>
volume: <five-wide volume-per-atom bin>
end: plan
```

The R5-C body derives element counts and exact output length from `formula`; a body with `N` atoms has `7 + 4N` generated body tokens.

## Exact data identity

Data directory:

`data/dlm_sft/mp_20_h1a2_rich_planner_noid_l3base`

| Split | Rows | SHA-256 |
|---|---:|---|
| train | 27,136 | `d431dfec1de8c3240dbc5648867be1b4b676fd85276e805a177b9944f3a1a157` |
| validation | 9,047 | `59327aa789ae5d2bbb66d8a8f0dc882d594bcc14623aa96ce95076ed1b6fc540` |
| test | 9,046 | `032845826acf1fcb9e7893fc91da05bc0cd9d2363c80ae459d5c446c3c6d5ea8` |

Each source row contributes one direct Plan target. The historical files contain no sample ID. Future work must add an external immutable row ledger instead of modifying the target text.

## Planner training hyperparameters

| Parameter | H1-A2 epoch-2 value | Evidence |
|---|---:|---|
| Base precision | BF16 | P/R |
| Gradient checkpointing | enabled | P/R |
| LoRA rank | 16 | P |
| LoRA alpha | 32 | P |
| LoRA dropout | 0.05 | P |
| LoRA bias | none | P |
| LoRA target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | P |
| Per-device batch | 1 | P |
| Gradient accumulation | 8 | P |
| Effective batch per process | 8 | I |
| Maximum sequence length | 768 | P |
| Optimizer | AdamW | R |
| Learning rate | `2e-5` | P |
| Weight decay | 0 | P |
| Scheduler | cosine | P/R |
| Warmup | 100 updates | P |
| Gradient clipping | 1.0 | R |
| Seed | 17 | P |
| Epoch-2 updates | 3,392 | P |
| Epoch-2 continuation | exactly one additional epoch from the epoch-1 adapter | P |
| Validation cadence | 500 updates | P |
| Maximum validation batches | 50 | P |
| Final validation loss | `0.31758024722337724` | P |
| Epoch-2 elapsed time | 7,316.21 seconds | P |

The epoch-2 launcher restarts the optimizer and scheduler from the epoch-1 adapter. “Epoch-2” therefore means two total data passes across the lineage, but a fresh one-epoch continuation run with its own cosine schedule.

The historical wrapper requested 2×A800, 16 CPU, 180 GB, and 30 hours. The Planner SFT itself used one Python process; sampling, body generation, and refinement used two ranks. This is part of the reconstruction, not permission to repeat it. Every new A800 job must use at most 8 CPU per A800; the proposed pilot uses one A800 and no more than 8 CPU.

## Planner sampling hyperparameters

| Parameter | Value |
|---|---:|
| Requested plans | 1,200 |
| Historical world size | 2 |
| Batch per process | 4 |
| Temperature | 0.9 |
| Top-p | 0.95 |
| Top-k | 50 |
| Maximum new tokens | 96 |
| Seed | 17 plus rank |
| Sampling | enabled |
| Retry / repair / rerank | none |
| Valid parsed plans | 1,186 / 1,200 = 98.833% |

Generation stops at and is truncated after `end: plan`.

Historical distribution diagnostics:

| Diagnostic | Generated | Teacher/reference |
|---|---:|---:|
| Mean atom count | 11.0363 | 10.4233 |
| `N >= 12` | 47.25% | 42.8324% |
| Ternary | 61.9167% | 58.1994% |
| Four or more elements | 20.9167% | not frozen here |
| Single-element | 0.5% | not frozen here |
| Textual all-metal label | 23.1667% | 29.8718% |

Total-variation distances were: atom count `0.08550`, arity `0.06004`, element presence `0.09783`, anion `0.08860`, charge `0.10491`, lattice `0.09781`, space-group bucket `0.08672`, and volume bin `0.07062`.

## R5-C body hyperparameters

| Parameter | Value | Evidence |
|---|---|---|
| Base model | LLaDA-8B-Instruct | P |
| R5-C checkpoint | `runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final` | P |
| Input plans | 1,186 valid Planner outputs | P |
| Historical world size | 2 | P |
| Batch per process | 8 | P |
| Temperature | 0.7 | P |
| CFG | 0 | P |
| Remasking | low confidence | P |
| Conditioning | full Plan state | P |
| Schedule | exact Plan schedule | P/R |
| Schema logit mask | enabled | P/R |
| Count-token prefill | enabled | P/R |
| Plan composition frozen | enabled | P/R |
| Duplicate-coordinate mask | enabled | P/R |
| Lattice-volume mask | enabled | P/R |
| Minimum lattice radians | `1e-4` | P/R |
| Retry / replacement / rerank | none | P |
| Accepted proposal graphs | approximately 1,171 / 1,186 = 98.735% | P |

Because the Plan composition is frozen into exact slots, accepted R5-C bodies match the sampled formula by construction. This transfers formula errors from the Planner into the final structure rather than correcting them.

## CrysLLMGen parent refinement

| Parameter | Value |
|---|---|
| Parent checkpoint | `/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt` |
| Refined inputs | first 1,000 accepted proposal graphs |
| Batch | 128 |
| Diffusion timesteps | 1,000 |
| Refinement steps | 800 |
| Evaluations per input | 1 |
| Run type | `train` in the historical CrysLLMGen interface |
| Historical world size | 2 |
| Refiner training | none |

The historical protocol generated 1,200 Plans, obtained 1,186 valid Plans, and evaluated the first 1,000 accepted graphs. This is recoverable history but is not the preferred new protocol, because it creates a survivor-prefix denominator. The new pilot must request exactly the registered denominator and retain every attempt.

## Reproduced baseline metrics

The 2026-07-29 exact historical parity evaluation recovered the intended original A100/CHGNet S.U.N scale.

| System | Composition | Structure | Joint | Strict S.U.N | Meta S.U.N |
|---|---:|---:|---:|---:|---:|
| Historical CrysLLMGen 1000 | 89.2% | 99.9% | 89.1% | 90 / 1000 = 9.0% | 461 / 1000 = 46.1% |
| H1-A2 epoch-2 1000 | 87.8% | 99.9% | 87.7% | 94 / 1000 = 9.4% | 474 / 1000 = 47.4% |

Coverage-adjusted values were:

- CrysLLMGen: strict 9.31%, meta 47.67%
- H1-A2 epoch-2: strict 9.71%, meta 48.94%

Coverage-adjusted numbers are reports only and must not become the training or checkpoint-selection objective.

The conditional gold-Plan R5-C reference reached approximately 10.61% adjusted strict and 74.38% adjusted meta. The large meta gap is evidence that improving formula validity alone cannot close the whole S.U.N gap.

The H1-A2 parity report also recorded 890 novel-and-unique structures, 862 hull-evaluated structures, and 28 hull-unknown structures.

## Composition failure decomposition

Among the 1,186 parsed H1-A2 epoch-2 Plans:

| Result | Count | Rate |
|---|---:|---:|
| Composition-valid | 1,044 | 88.027% |
| Composition-invalid | 142 | 11.973% |
| Charge-neutrality failure | 98 | 8.263% |
| Pauling failure or rejected ratio | 37 | 3.120% |
| Missing oxidation state | 7 | 0.590% |

Valid-reason categories were:

- charge-neutral and Pauling-valid: 685
- all-metal shortcut: 353
- single-element shortcut: 6

The all-metal shortcut is an evaluator-valid category, not a desirable optimization target. Any new method must prevent a superficial rise in composition validity caused by inflating all-metal generations.

The evaluated first 1,000 refined structures contained 878 composition-valid, 999 structure-valid, and 877 jointly valid structures.

## Why composition validity alone is insufficient

Within the frozen 1,000:

- strict conditional on composition-valid is approximately `94 / 878 = 10.7%`
- meta conditional on composition-valid is approximately `474 / 878 = 54.0%`

If a method recovered 20 additional composition-valid structures while preserving these conditional rates, the rough expected gain would be only:

- strict: about 2.1 additional structures, or 0.21 percentage points
- meta: about 10.8 additional structures, or 1.08 percentage points

This is an attribution estimate, not a statistical promise. It shows why the first method must improve both formula chemistry and joint Plan realizability.

## Confirmed implementation limitation

The restored parser:

1. enforces formula syntax and `1 <= N <= 20`;
2. normalizes the six rich Plan fields;
3. computes whether the declared anion matches the formula;
4. does not reject an anion mismatch;
5. treats `charge` as a generated categorical label rather than recomputing chemistry validity;
6. does not test joint compatibility among formula, anion/charge, lattice, space-group bucket, and volume bin.

Consequently, parse validity and low marginal TVDs can pass while the full tuple remains chemically invalid or geometrically off-manifold.

## Core recovered code map

| Component | Restored path | SHA-256 | Evidence |
|---|---|---|---|
| Planner parser/schema | `baseline/crystal_dlm/h1_llm_planner.py` | `d45ccc23fad4284fdeef53d7bbdc5e4044fb6b598092461663473ed8f5a4f8ad` | R |
| Formula composition validator | `baseline/crystal_dlm/composition_validity.py` | `ca1c94f583e0c97a172b5c9b7ba96505257fd74dedfc618b584c34486ac1f178` | R |
| Plan state | `baseline/crystal_dlm/r5_plan_state.py` | `6bc1e446afdbe405df601f609a2326c9f376a7f036e9d27dafcecd57965866fc` | R |
| Plan-to-body encoding | `baseline/crystal_dlm/r5_plan_body.py` | `3478ddf657873ea055e5816c423ce36be5ecf0cd1a73c6ee1e5514648047be83` | R |
| Exact dynamic length | `baseline/crystal_dlm/r5_dynamic_length.py` | `c022ddda92caac1c60b91b239c1bc155c734cad86b595282ce9724e239d002c6` | R |
| Planner data builder | `baseline/scripts/build_h1_llm_formula_sft_data.py` | `81b9851c4e50cc2821923ccb9addb3590e2115402754198cd79eec8e31f05e26` | R |
| Planner trainer | `baseline/scripts/llama_formula_sft.py` | `a45e8027e7732fe12057c934b5a2dfa9bc3151fc4f30a50a3b94d40b342f2dba` | R |
| Planner sampler | `baseline/scripts/sample_llama_h1_formula_plans.py` | `d38743f2f647d798800724b09537fbe492706805c00d7ee34c5ca8d74e39adc8` | R |
| Planner gate | `baseline/scripts/evaluate_h1_planner_gate.py` | `eea26b5bcc6d767d733babf289257f9cfb7253fd2395708897cc650162747a29` | R |
| R5-C sampler | `baseline/scripts/sample_llada_r5_exact_length.py` | `686bb2a2612f3dbf65272d6ec49dea9d5b2d5e80d1c9cd955673e4dadb8e6db6` | R |
| CrysLLMGen refiner bridge | `baseline/scripts/refine_dlm_with_crysllmgen.py` | `a0cb2c54c149aee0a4f36c147d93eca6799e3b17879f76c108721c934f087b2a` | R |
| Direct metrics | `baseline/scripts/run_crysllmgen_metrics.py` | `fd89897f61ddc672bc877f700ecd7ed3d5e1b0f3c2c3e67e335d740f3601fa9b` | R |
| Historical S.U.N launcher | `baseline/launchers/pre_wyckoff/a800/run_a100_eval_sun_dlm_only.sh` | `6544515e0ca21e29b33b2ad53942269b48ca7c21c5a8d5a37bd003e8c6f957f9` | R |

Paths in this table are relative to `workstreams/r5c_reactivation_20260728`.

## What not to repeat

- Do not continue the old step-500 same-source branch. It improved matched-vs-shuffled Plan identity but lost 3.5–4.3 percentage points of meta S.U.N relative to the earlier references.
- Do not add another unmodified full Planner epoch at learning rate `2e-5`; H1-A3 showed that extra epochs can increase atom-count drift and reduce S.U.N.
- Do not optimize the all-metal shortcut.
- Do not repeat the free-geometry CE reweighting arm that failed its promotion gate.
- Do not use CHGNet, MatterSim, any MLIP, Materials Project API results, S.U.N, or hull values in training, filtering, retry, repair, reranking, or checkpoint selection.
- Do not use the historical 1,200-to-first-1,000 survivor-prefix evaluation in new confirmatory work.

## Innovation roadmap

### Phase 1 — H1-A2C chemistry/joint Planner continuation

Start from the frozen epoch-2 adapter. Use same-split MP20 targets only.

The low-risk arm uses:

- chemistry-valid target replay;
- class balancing that does not preferentially increase all-metal cases;
- low learning rate and a short fixed update budget;
- an epoch-2 reference anchor to prevent atom-count, arity, element, and rich-field drift.

The innovation arm adds two sequence-level contrastive negatives:

1. **chemistry negative:** count/element corruption that preserves the atom-count and arity bucket but fails charge neutrality or the Pauling test;
2. **joint-Plan negative:** a lattice/space-group/volume or anion/charge tuple shuffled from a matched-bucket material while preserving marginal labels.

The positive is always the unmodified Plan from one MP20 row. The model is rewarded for assigning the positive full Plan a higher normalized log-likelihood than both negatives. No negative is emitted at inference.

### Phase 2 — Plan-aware diffusion residual

Keep the epoch-2 Planner and R5-C body frozen. Add a zero-initialized residual/FiLM path to the frozen CrysLLMGen `model_494` denoiser:

- condition on formula/counts plus lattice, space-group, and volume Plan fields;
- leave atom identities and atom order immutable;
- predict only residual coordinate-score and lattice-noise corrections;
- train only on MP20 teacher geometry, sampled diffusion noise, and reconstruction losses;
- balance ground-truth chemistry families without using hull or MLIP labels;
- require exact-null equality and matched-vs-shuffled Plan identity.

This directly targets the low-energy structural basin while preserving the stock parent output at initialization. It is distinct from resuming the failed step-500 Plan-to-DLM branch.

The minimal recovered insertion points are:

- `CSPDiffusion.forward` and both decoder calls in `CSPDiffusion.sample`;
- per-atom features immediately after `CSPNet.atom_latent_emb`;
- graph-level features immediately before `CSPNet.lattice_out`.

A wrapper that adds zero-initialized residual heads to the existing `pred_x` and `pred_l` is safer than rewriting all six CSP layers.

### Phase 3 — higher-risk structured Planner

If free-text formula sampling remains the dominant failure, replace the formula line with explicit element/count/oxidation slots and finite-state constrained decoding. Python then derives the printed formula. This is a model representation change, not post-hoc repair or reranking, but it requires new tokenizer/schema compatibility work and should not be the first experiment.

## Frozen evaluation order

1. Build immutable train/validation/test row ledgers.
2. Train the short Planner arms.
3. Select a checkpoint only with held-out Plan metrics and distribution-drift gates.
4. Run paired 256-attempt R5-C/refiner screening with exactly 256 registered attempts per arm.
5. Freeze the winning checkpoint.
6. Run original A100/CHGNet S.U.N as evaluation only.
7. Promote to paired 1,000 only after the screen passes.

The detailed Planner preregistration is in `H1A2C_JOINTCHEM_V1_EXPERIMENT.md`. The combined Planner/diffusion design is in `H1A2_EPOCH2_INNOVATION_ROADMAP.md`.
