# Rich-compatible DLM execution gap audit V1

Date: 2026-08-30
Audited baseline: `79ac47d` (`codex/h1a2-rich-planner-audit`)
Mode: zero-GPU, source/test audit only; no training, sampling, refinement, evaluation, Slurm submission, asset download, or Git commit was run.

## Verdict

**Current execution verdict: NO-GO.** The repository contains most low-level
rich-Plan, exact-composition generation, refinement, evaluation, and listwise
data/loss components, but it does not contain one asset-complete, same-ledger,
six-cell `M0/RCF/R0` execution path. The historical rich-compatible checkpoint
is referenced and historically hashed, but its weights are not in this
checkout. The listwise fallback is also not runnable: data builders and a pure
loss exist, while the listwise model scoring/collator, trainer, two-seed
training wrapper, and integration tests are missing.

Status meanings:

- `EXISTS`: concrete implementation is present in the repository.
- `PARTIAL`: reusable implementation exists, but the required execution path,
  asset, binding, or test is incomplete.
- `MISSING`: the required entry point or integration was not found, or the
  nominal generic wrapper is an explicit stub.

| Required item | Status | Short finding |
|---|---|---|
| Historical rich-compatible DLM identity | `PARTIAL` | B0 path/SHA exist in repository history; current checkout has placeholders only. |
| Seven-line rich Plan formatter/parser | `EXISTS` | Formatter and parser exist; direct parser round-trip test and canary routing test are missing. |
| Historical seven-line-to-DLM prompt path | `EXISTS` | Seven lines are parsed to `plan_state`, then serialized as the JSON body prompt. |
| Frozen 256-composition recovery ledger entry | `PARTIAL` | Immutable/identity primitives exist; exact shared `M0/RCF/R0` ledger and RCF permutation are missing. |
| Generation core | `EXISTS` | Exact-composition sampler supports frozen elements and per-sample seeds. |
| Recovery generation wrapper | `MISSING` | Generic wrapper is a stub; old runnable wrappers are bound to other arms/resources. |
| Refinement core | `EXISTS` | Generic model494 wrapper and deterministic seed support exist. |
| Recovery refinement wrapper | `PARTIAL` | Chained by old experiments, but not by an `M0/RCF/R0` same-ledger wrapper. |
| Evaluation core | `EXISTS` | Fixed-denominator raw/refined evaluator runtime exists. |
| Recovery evaluation wrapper | `PARTIAL` | Generic wrapper is a stub; runnable wrappers and assembler are experiment/schema-bound. |
| Listwise group/safety data | `EXISTS` | Immutable CPU builders, hashes, holdout guards, and tests exist; source assets are absent locally. |
| Listwise loss | `EXISTS` | Pure continuous-energy listwise loss exists. |
| Listwise trainer | `MISSING` | Neither SFT nor D3PO trainer consumes listwise groups/loss. |
| Listwise training wrapper/integration test | `MISSING` | Only a data-builder wrapper exists. |

## 1. Historical rich-compatible DLM checkpoint references

### Current checkout

Status: `PARTIAL`.

- `docs/ASSET_TRANSFER_LEDGER.md:11-12` names the LLaDA base and B0 DLM
  checkpoint. The checkpoint is
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final`,
  but its state is explicitly “已确认，待复制”.
- `checkpoints/dlm/README.md:1-4` is only a placeholder describing expected
  adapter/tokenizer files. The directory contains no model payload in this
  checkout.
- `scripts/download_checkpoints.sh:7-23` can download a caller-supplied
  `H1A2_DLM_URL` to `checkpoints/dlm/checkpoint.tar.gz`; it supplies no URL,
  extraction, or SHA binding.
- The sampler loader uses the checkpoint tokenizer when the path exists and
  loads either a PEFT adapter or a full checkpoint
  (`src/scripts/sample_llada_dynamic_crystals.py:84-126`). Thus a path string or
  tar placeholder is not an executable checkpoint.

### Git-reachable historical inventory

The repository retains stronger evidence on
`origin/codex/evidence-first-sun-msun` at commit `c87cfb6`:

- `c87cfb6:workstreams/final_method_development_20260808/DLM_CANDIDATE_REGISTRY.json:7-12`
  records B0 as `protected_control`, the same R5C checkpoint path, adapter SHA256
  `5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d`,
  and `6391016776` bytes.
- The same registry records B1 as historical inventory only/not promotable
  (`:15-20`) and B2 as a stopped/non-revivable candidate (`:26-31`). They are
  inventory references, not authorized recovery alternatives.
- Historical launcher
  `c87cfb6:workstreams/r5c_reactivation_20260728/baseline/launchers/pre_wyckoff/a800/run_h1_llm_plan_dlm_body.sh:7-8,30,256-265`
  binds B0 to `full_plan_state`, exact-composition sampling, and
  `--freeze-plan-composition`.
- Historical H1-A2 extension launcher
  `c87cfb6:workstreams/r5c_reactivation_20260728/baseline/launchers/pre_wyckoff/a800/run_h1a2_epoch_extension_fullmetrics.sh:7-10,185,205-214`
  uses the same B0 checkpoint after sampling `h1_rich_plan_v1` Plans.

Conclusion: the repository identifies and hashes B0 and demonstrates its
historical rich compatibility, but the live base/checkpoint/tokenizer files are
not present. B1/B2 must not be silently substituted.

## 2. Seven-line parser and actual DLM prompt contract

Status: parser `EXISTS`; canary integration `PARTIAL`.

- `src/crystal_dlm/r5_plan_body.py:354-365` renders exactly seven lines:
  formula, anion, charge, lattice, spacegroup, volume, and `end: plan`.
- `src/crystal_dlm/r5_plan_body.py:400-453` parses labelled text and enforces
  the complete rich schema/end marker when `h1_rich_plan_v1` is requested.
  Formula-derived composition/N is reconstructed at `:455-495`; rich fields are
  validated and copied at `:498-520`.
- `src/crystal_dlm/h1_llm_planner.py:228-254` shows the historical bridge:
  parse the seven-line Planner output, validate it, retain canonical text and
  `plan_state`, then build the DLM `prompt` with `build_body_prompt(plan)`.
- `src/crystal_dlm/r5_plan_state.py:1254-1260` shows that the historical DLM
  body prompt is JSON `plan_state`, not raw seven-line text.
- C3FD renders through the same formatter
  (`src/crystal_dlm/ccfd_v2.py:917-930`), and its sampler stores both
  `plan_text` and `plan_state` (`scripts/sample_c3fd_plans.py:417-461`). Only
  lattice and volume are sampled from rich logits; anion/charge are derived and
  spacegroup is compiled (`scripts/sample_c3fd_plans.py:417-440`).

The current paired-interface audit duplicates the seven-line renderer and emits
`minimal_prompt`, `rich_prompt`, and `plan_state`
(`scripts/audit_c3fd_rich_interface.py:88-99,134-148`). The body sampler reads
`plan_state`; it uses a row prompt verbatim only when the selected prompt field
exists, otherwise it rebuilds the JSON body prompt
(`src/scripts/sample_llada_r5_exact_length.py:58-81`). Its CLI defaults to
`--prompt-field prompt` (`:283-299`). Therefore:

- default consumption of paired-interface rows rebuilds the historically
  evidenced JSON `full_plan_state` prompt;
- explicitly selecting `rich_prompt` feeds raw seven-line text, for which this
  audit found no historical checkpoint-compatibility evidence;
- no integration test freezes which route `R0/RCF` must use.

Existing tests:

- `tests/test_ccfd_v2.py:181-222` asserts the exact seven rendered lines and
  composition/N preservation.
- `tests/test_audit_c3fd_rich_interface.py:29-46` covers prompt content,
  lattice/spacegroup mapping, and TVD helper behavior.
- `tests/test_stability_mechanism.py:39-48` round-trips the JSON hard-anchor
  body prompt, not seven-line `parse_composition_plan`.

There is no current direct unit test of `parse_composition_plan` rich
round-trip. A zero-GPU in-memory audit smoke did pass for a seven-line NaCl
Plan (`line_count=7`, `plan_format=h1_rich_plan_v1`, composition preserved),
but that is not a checked-in test.

## 3. Frozen composition ledger entry

Status: `PARTIAL`; the required recovery ledger entry is `MISSING`.

- `scripts/audit_c3fd_rich_interface.py:114-159` verifies exactly 1,000
  ordinals per Planner seed, composition certificates, rich fields, and paired
  minimal/rich identities.
- Its CLI writes exclusive JSONL files, source/output hashes, a manifest, and a
  success marker (`:169-217`). This is a useful immutable paired-interface
  audit, but it emits separate full seed17/seed18 files.
- It does not select one new 256-row cohort, recompute blocked identity/chemsys
  overlap, assign one shared ledger SHA to all arms, or create the
  marginal-preserving RCF permutation. Repository search found RCF only in the
  planning documents; `docs/DLM_RICH_PLANNER_DECISION_LOG_V2.md:68` also records
  the RCF/R0 wrapper as absent.
- `scripts/freeze_d3po_test_cohort.py:64-92,125-202` is the closest reusable
  freeze entry: reduced-identity exclusion, duplicate rejection, default count
  256, source ordering, hashes, manifest, and `_SUCCESS`.
  `tests/test_freeze_d3po_test_cohort.py:27-58` covers supercell leakage,
  duplicates, and exact-versus-reduced identity.
- `src/scripts/sample_llada_r5_exact_length.py:104-132,410-417` freezes element
  tokens during generation. This enforces a consumed ledger's composition; it
  does not create or outcome-blindly select that ledger.

## 4. Generation, refinement, and evaluation wrappers

### Generation

Status: core `EXISTS`; recovery wrapper `MISSING`.

- `src/scripts/sample_llada_r5_exact_length.py:283-325` exposes model,
  checkpoint, prompt JSONL/field, prompt style, seed, per-sample seed,
  composition freeze, and schedule arguments.
- It groups equal-N records, pre-fills N/elements, applies exact schema/schedule,
  and emits attempt-preserving raw records (`:392-485`).
- `slurm/40_generate_body.sbatch:16-19` is an explicit stub that exits 2.
- `slurm/38_dlm_condition_schedule_l6.sbatch:49-96` is runnable historical
  full-plan/hard-anchor generation+refinement, but requests eight GPUs
  (`:1-9`) and has no M0/RCF/R0 arms or one-ledger hash contract.
- `slurm/66_d3po_fixed256_generation.sbatch:19-45,82-125` supplies a useful
  six-cell/hash/seed/failure-marker orchestration skeleton, but it is bound to
  three D3PO checkpoints and the SGTC/minimal sampler, not rich recovery.
- No existing test directly imports or executes
  `sample_llada_r5_exact_length.py` end to end.

### Refinement

Status: core `EXISTS`; recovery wrapper `PARTIAL`.

- `src/scripts/refine_dlm_with_crysllmgen.py:183-218` provides a generic
  proposal/checkpoint/output/seed CLI and records that it performs no training.
- It verifies mutually exclusive seed modes, loads model494, and supports
  deterministic sample-index or graph-field seeds (`:224-278`), then writes
  sample indices with refined tensors and metrics (`:289-347`).
- `tests/test_ctv_refiner_seed.py:23-46` covers propagation of a common refiner
  seed, but was skipped in the audited environment because PyTorch is absent.
- The current diffusion checkpoint directory is also placeholder-only, so the
  core was not executable locally.

### Evaluation

Status: core `EXISTS`; recovery wrapper `PARTIAL`.

- `eval_runtime/run_full_reconstructed_eval.py:32-118` validates arm/repeat,
  fixed attempt ledger, frozen source hashes, caches, model assets, and the
  denominator before evaluation.
- It computes Direct/N/U/CHGNet attempt records, but relaxation explicitly uses
  `cuda` (`:120-150`); this full evaluator is not a zero-GPU runtime.
- `tests/test_eval_runtime_source_manifest.py:17-59` verifies relative source
  manifests and rejects unsafe/late-added files. It does not execute the full
  evaluator.
- `slurm/60_evaluate.sbatch:15-21` is an explicit generic stub.
- Runnable old wrappers exist, but are bound to old R03 assets and arm schemas:
  `slurm/39_dlm_condition_schedule_l6_eval.sbatch:13-27,59-118` requests eight
  GPUs, while the six-GPU pattern in
  `slurm/67_d3po_fixed256_eval.sbatch:49-127` is D3PO-specific.
- `scripts/assemble_grounding_repeat.py:64-75,87-120` only accepts
  `control/candidate` and hard-codes historical method/body-arm labels; it
  cannot preserve three distinct M0/RCF/R0 identities unchanged. No direct
  assembler/full-evaluator integration test exists.

## 5. Listwise loss, trainer, and wrapper

### Data builders — `EXISTS`

- `scripts/build_d3po_listwise_groups.py:1-7` restricts inputs to historical
  train-only sources and excludes main/sealed holdout CLI inputs.
- It creates one weight-one group per exact composition, requires `K>=2`, and
  enforces chemsys-disjoint train/validation splits (`:162-185,188-280`).
- It writes source/code/output hashes and explicit no-holdout flags
  (`:325-381`) and rechecks source identity before/after reading (`:400-445`).
- `tests/test_build_d3po_listwise_groups.py:48-169` covers group weighting,
  `K<2`, chemsys separation, hashes, holdout guards, and CLI exclusion; its
  physical StructureMatcher case is dependency-gated.
- `scripts/build_listwise_safety_v2.py:182-281` joins raw candidates by hashed
  identity and adds raw parse/composition/geometry safety. Atomic preparing/
  failed/final output and no-holdout flags are at `:293-356`.
- `tests/test_build_listwise_safety_v2.py:45-107` covers CLI guards and basic
  helpers; pymatgen-dependent safety tests are dependency-gated.
- `slurm/71_listwise_safety_v2.sbatch:13-67` is a hash-pinned CPU data-builder
  wrapper (`gpu_jobs_used=0`), not a training wrapper. Its expected data paths
  are not present in this checkout.

### Pure loss — `EXISTS`

- `src/crystal_dlm/listwise_alignment.py:1-7` explicitly declares that it has
  no model, tokenizer, trainer, loader, sampler, or job logic.
- Robust continuous rewards/advantages are implemented at `:85-127`; bounded
  linear, quadratic, and best-anchor terms at `:130-195`; the shared-noise
  wrapper at `:198-255`.
- `tests/test_listwise_alignment.py:26-199` covers rewards, centering, guards,
  components, gradients, shift behavior, and lowest-energy anchor, but all ten
  cases were skipped locally because PyTorch is absent.

### Trainer — `MISSING`

- The SFT trainer computes per-sample denoising CE
  (`src/scripts/llada_sft.py:1673-1730`). Its training loop combines only task
  loss, optional counterfactual grounding, and element alignment
  (`:2684-2719`); there is no listwise group collator, shared group corruption,
  reference-corrected score path, or call to `listwise_alignment`.
- `src/scripts/llada_d3po.py:1-7,35-51` is intentionally a dedicated pairwise
  D3PO trainer and imports `d3po_pair_loss`, not the listwise loss.

### Training wrapper and integration test — `MISSING`

- `slurm/65_d3po_train.sbatch:19-27,65-123` is a hash-pinned two-seed/348-step
  wrapper for pairwise D3PO data and `llada_d3po.py`; it is not listwise.
- No wrapper invokes listwise groups, `shared_noise_listwise_alignment_loss`, a
  listwise trainer, two listwise seeds, or a listwise terminal manifest.
- Existing tests stop at pure loss/data contracts. There is no model-forward,
  collator, optimizer, checkpoint, wrapper, or end-to-end listwise integration
  test.

## 6. Zero-GPU verification performed

`pytest` is not installed, so relevant checked-in `unittest` modules were run
with bytecode and test-cache creation disabled. Across 57 discovered cases:

- 31 passed;
- 26 skipped because the local Python 3.14 environment lacks PyTorch,
  transformers, PEFT, and/or pymatgen;
- 0 failed.

Passing coverage included the seven-line C3FD renderer, paired-interface
helpers, hard-anchor JSON prompt, frozen-cohort identity rules, evaluation
source manifests, listwise group builder, listwise safety non-pymatgen helpers,
and D3PO wrapper static contracts. The sampler/model/refiner/listwise tensor
paths were not dynamically validated because their dependencies are absent.
No result in this section proves checkpoint or full-pipeline executability.

## 7. Minimal critical path (execution closure only)

This is the shortest closure of the already frozen method; it does not add a
new scientific method.

### Rich recovery canary

1. Resolve the historical B0 base/checkpoint/tokenizer and model494 assets;
   verify B0 against the historical SHA and record all live SHA256 values.
2. Freeze one new blocked-identity/chemsys-audited 256-composition ledger with
   one ledger SHA; materialize M0, RCF, and R0 views without changing hard
   composition/N/ordinals. The exact entry point and RCF renderer are currently
   missing.
3. Add direct round-trip/integration coverage for
   seven-line render -> parser -> `plan_state` -> historical JSON body prompt ->
   sampler record, including explicit prompt-field behavior.
4. Bind the existing exact-composition sampler and refiner into one <=6-GPU,
   two-stream, three-arm wrapper with input hashes, immutable attempt directory,
   failure marker, and one ledger SHA. The D3PO six-cell wrapper is reusable
   only as an orchestration pattern.
5. Preserve all three arm identities through raw/refined assembly and the
   fixed-256 evaluator; add assembler/evaluator contract tests. Only after these
   five items pass is the recovery canary executable.

### Listwise fallback

1. Materialize and hash the already specified listwise group/safety assets.
2. Connect grouped shared corruption and model/reference scores to the existing
   pure listwise loss in a trainer; no such trainer currently exists.
3. Add the frozen two-seed/348-update training wrapper and model-to-checkpoint
   integration tests. Until then, listwise remains `PARTIAL`, not runnable.

## 8. Reusable components

| Component | Reusable scope | Boundary |
|---|---|---|
| Historical B0 registry | Checkpoint path/SHA/size identity | Weights/base/tokenizer absent locally. |
| `r5_plan_body` | Canonical seven-line render/parse | Direct checked-in round-trip test absent. |
| C3FD sampler output | Exact `plan_state` plus canonical rich text | Requires checkpoint/data assets. |
| C3FD paired-interface audit | Paired prompts, composition identity, hashes | Not a held-out 256 ledger; no RCF. |
| D3PO cohort freezer | Reduced-identity exclusion, deterministic count, manifest | D3PO schema; no rich arm views. |
| Exact-length LLaDA sampler | Frozen composition, exact schedule, per-sample seed | No recovery wrapper/direct test. |
| CrysLLMGen refiner wrapper | model494 loading, common seeds, indexed output | Asset/dependency absent; no recovery wrapper. |
| D3PO six-cell wrappers | Six-GPU orchestration, hashes, common streams, markers | D3PO checkpoints/prompts/labels are hard-bound. |
| Fixed-256 eval runtime | Attempt preservation, Direct/N/U/energy, source hashes | CUDA/external frozen assets; old arm schema. |
| Listwise builders/loss | Frozen groups, safety labels, pure objective | No trainer or training wrapper. |
| D3PO trainer utilities | Model/tokenizer loading, scheduler, checkpoint I/O | Pairwise semantics are not a listwise trainer. |

## Files changed by this audit

- `docs/RICH_DLM_EXECUTION_GAP_AUDIT_V1.md`
