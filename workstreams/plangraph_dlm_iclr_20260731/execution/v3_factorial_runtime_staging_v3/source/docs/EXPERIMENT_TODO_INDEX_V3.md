# PlanGraph-DLM V3 Experiment TODO Index

Status: `io_factorial_contract_complete_scientific_training_blocked`

Last updated: 2026-08-01

This is the single operational index for all future H1-A2-aligned
PlanGraph-DLM work. A task is marked complete only when its result and evidence
are linked here. Governing scientific details remain in
`EXPERIMENT_CHARTER_V3_H1A2_TWO_FACTOR.md` and
`EXPERIMENT_REGISTRY_V3_H1A2_TWO_FACTOR.json`.

## Status legend

- `[x] COMPLETE`: result exists and evidence is linked.
- `[-] IN PROGRESS`: current active work.
- `[ ] TODO`: not started or not yet proven.
- `[!] BLOCKED`: cannot safely continue without a missing input or decision.
- `[S] SCIENTIFIC STOP`: completed experiment failed a preregistered gate.

No historical H1 source, checkpoint, report, or run directory may be modified.
All new outputs use new V3 paths. Engineering fixes retain their failed
evidence; scientific gate failures are never repaired by changing data, seed,
denominator, threshold, or checkpoint-selection rules.

## Current state

| Item | Status | Result | Evidence |
|---|---|---|---|
| H1-A2 fallback identity | COMPLETE | P0, B0, and frozen refiner identities recorded; historical raw S.U.N. is 9.4% strict and 47.4% meta | `H1_FALLBACK_MANIFEST.md`; `EXPERIMENT_CHARTER_V3_H1A2_TWO_FACTOR.md` |
| Plan provenance correction | COMPLETE | Structure-derived R5-C Plans are body supervision only; fully-de-novo Plans must be sampled by P0/P* | `EXPERIMENT_CHARTER_V3_H1A2_TWO_FACTOR.md` sections 1 and 4 |
| Factorial design | COMPLETE | `M00=P0+B0`, `M10=P*+B0`, `M01=P0+B*`, `M11=P*+B*`; no shuffle | `EXPERIMENT_REGISTRY_V3_H1A2_TWO_FACTOR.json` |
| Local DLM integration | COMPLETE | B1/B2 data, corruption, preflight, and schedule tests: 21 passed, 2 skipped because local Torch is absent | `LOCAL_VALIDATION_V3_H1A2_TWO_FACTOR.json` |
| Historical DLM multi-GPU contract | COMPLETE | R5-C used 2xA800 DDP, per-device batch 1, accumulation 8, global batch 16 | `legacy_dlm_r5c/launchers/pre_wyckoff/a800/run_r5_crystal_state_dlm.sh`; registry section `distributed_execution` |
| P* scientific method | COMPLETE | Look-Ahead Consistent Planner v1 is frozen and its single-GPU trainer is implemented; numerical/A800 smoke remains a separate pending task | `PLANNER_PSTAR_METHOD_V1.md`; `scripts/llama_h1a2_lookahead_sft.py` |
| P* deterministic data builder | COMPLETE | Common P-control/P* train and validation streams can be selected deterministically with exact quotas and source-SHA fail-close; execution on the frozen H1-A2 source remains pending | `scripts/build_h1a2_lookahead_planner_data.py`; `tests/test_h1a2_lookahead_data.py` |
| Real H1-A2 Planner source | COMPLETE | train 27,136 / val 9,047; all rows use the exact seven-line schema, `h1_rich_plan_v1`, and weight 1.0; source SHAs frozen | Registry `frozen_identity.planner_source_data`; completed-result log below |
| Preflight/smoke authorization | COMPLETE | CPU materialization, real-tokenizer preflight, and bounded 32-row Planner/DLM engineering smoke are authorized; scientific training and downstream remain unauthorized | `AUTHORIZATION_V3_PREFLIGHT_SMOKE_20260801.json` |
| V3 remote jobs | COMPLETE TO CURRENT AUTHORIZATION | CPU preflight 29318, Planner smoke 29322, sidecar preflight 29331, both 2xA800 B1/B2 arms 29337, and repaired assembly 29345 completed `0:0`; failed assembly 29338 is preserved | Original and repair records plus the result ledger below |

## Immutable execution rules

- Planner inference keeps the exact H1-A2 seven-line text schema.
- The Planner model proposes every visible Plan value.
- No sample ID, retry, replacement, repair, filtering, reranking, or
  survivor-prefix denominator.
- Raw model Plan text, canonical Plan text, compiled body prompt, and their
  identities are retained per ordinal.
- Teacher Plans may train B1/B2 but may never replace sampled P0/P* Plans.
- The continuous CrysLLMGen refiner remains frozen.
- Raw all-attempt metrics are primary. Paper-compatible conditional metrics
  are secondary reports only.
- No S.U.N., MP, energy, hull, CHGNet, or generated-crystal result is used for
  training or checkpoint selection.
- G4 requires a new explicit authorization after G3; it is never automatic.

## Phase 0 — Baseline and protocol

- [x] `V3-000` Record immutable P0/B0/refiner identities.
  - Result: recorded in the V3 registry.
- [x] `V3-001` Preserve H1 as the submission fallback.
  - Result: fallback deadline and read-only policy recorded.
- [x] `V3-002` Remove shuffle from all future registered comparisons.
  - Result: shuffle is prohibited in the registry and charter.
- [x] `V3-003` Separate model-sampled Plans from structure-derived teacher
  Plans.
  - Result: fail-closed provenance firewall implemented locally.
- [x] `V3-004` Freeze the P* method, data stream, loss, budget, and selection
  rule.
  - Result: method, loss weights, 3,200 microbatches, 256-row validation panel,
    400 updates, LR, and selection rules frozen.
  - Evidence: `PLANNER_PSTAR_METHOD_V1.md`.

## Phase 1 — Input/output and deterministic execution

- [x] `V3-100` Register identical Planner prompt, tokenizer, chat template,
  parser, and seven-line schema for P0/P*.
- [x] `V3-101` Register additive body tokenization and exact `7 + 4*N`
  semantic length.
- [x] `V3-102` Persist and hash raw/canonical sampled Plan text and exact body
  prompt.
  - Result: three SHA-256 identities are emitted by the canonical Plan record.
  - Test: `tests.test_h1_llm_planner`.
- [x] `V3-103` Add rank-independent per-ordinal seed derivation for Planner,
  body, and refiner stages.
  - Result: one `h1a2_factorial_contract_v1` ordinal record now derives
    Planner, body, and refiner seeds only from the frozen base seed, ordinal,
    stage, and shared role; rank/world-size/batch order are absent.
- [-] `V3-104` Make distributed merge order strictly ascending by ordinal and
  reject duplicate/missing attempt identities.
  - Current result: implemented for Planner sampling and for a strict
    four-arm contract merger. Wiring the merger into the future body/refiner
    runner remains pending.
- [x] `V3-105` Add tests proving P0/P* inference input identity and
  M00/M01, M10/M11 body-prompt identity.
  - Result: P0/P* exact prompt bytes, token IDs, tokenizer/chat-template
    identity, no-sample-id input, exact seven-line model output, sampled-Plan
    provenance, within-Planner Plan/body-prompt hashes, shared noise, and
    duplicate/missing rejection all pass fail-closed tests.
  - Evidence: `IO_FACTORIAL_CONTRACT_VALIDATION_V1.json`; 27/27 targeted
    tests passed.
- [x] `V3-106` Run the real-tokenizer remote preflight at the frozen H1-A2
  Planner maximum length 768.
  - Result: job 29318 completed `0:0`; both full 3,200/256 and smoke 32/32
    streams passed the real fast tokenizer, field mapping, EOS, and
    formula-before-lattice causal-boundary checks.
  - Maximum observed sequence length: 383 tokens of the frozen 768 limit.

Phase-1 exit gate:

1. all identity tests pass;
2. one ordinal produces the same Plan/body/refiner seeds independent of rank;
3. no teacher-Plan fallback path exists at inference;
4. every mismatch fails closed.

## Phase 2P — Planner factor

Registered arms:

- P0: frozen H1-A2 epoch-2;
- P-control: identical P* stream/budget with field-balanced target-only loss;
- P*: field-balanced loss plus training-only look-ahead consistency.

Tasks:

- [x] `V3-P01` Implement exact seven-line field span mapping.
  - Result: fail-closed schema, offset mapping, and look-ahead vocabularies
    implemented.
  - Evidence: `crystal_dlm/h1a2_planner_objective.py`.
- [x] `V3-P02` Implement field-balanced target-only loss.
  - Result: CPU reference and lazy-Torch differentiable implementations are
    wired into the dedicated P0/P-control/P* runner. Torch numerical tests are
    registered but skipped locally because Torch is absent; `V3-P06` remains
    the execution proof.
- [x] `V3-P03` Implement training-only look-ahead heads and checkpoint audit
  inventory.
  - Result: seven deterministically seeded affine heads, boundary-state loss,
    per-head diagnostics, checkpoint state/config SHA inventory, and
    inference discard are implemented.
- [x] `V3-P04` Build and hash the common 3,200-microbatch P-control/P* stream
  on the execution cluster.
  - Result: 3,200 unique training rows materialized without replacement or
    filtering; data manifest SHA-256
    `4c2ba66923ccea16244cbc7ec138e69e07f50c4bca6ec7bfa1c631cd7f406c4d`.
  - Execution: job 29318.
- [x] `V3-P05` Build and hash the fixed 256-row validation panel.
  - Result: 256 unique validation rows materialized with the same frozen
    selection contract; full real-tokenizer preflight report SHA-256
    `6f6802ff9ad77dd2e50b24761b72bbe8c748bdf90a251363b4a2e0927e3d61e4`.
  - Execution: job 29318.
- [x] `V3-P06` Run CPU fixtures and one-A800 32-row deterministic smoke.
  - Result: CPU fixtures, the 32/32 real-tokenizer preflight, and both
    one-A800 smoke arms passed. Each arm completed exactly 32 microbatches
    and four optimizer updates.
  - P-control result: `COMPLETED 0:0`, engineering gate passed; target NLL
    `0.2682495 -> 0.2676392`, field loss
    `0.5183716 -> 0.5178833`, peak CUDA reserved memory 17,622,368,256
    bytes.
  - P* result: `COMPLETED 0:0`, engineering gate passed; target NLL
    `0.2682495 -> 0.2676697`, field loss
    `0.5183716 -> 0.5177612`, look-ahead loss
    `2.2856124 -> 2.1956991`, peak CUDA reserved memory 17,647,534,080
    bytes. Training-only heads were retained in the checkpoint inventory and
    marked discarded for inference.
- [ ] `V3-P07` Train P-control and P* for exactly 400 updates, validation every
  50 updates.
- [ ] `V3-P08` Select one checkpoint per trained arm without generation or
  S.U.N.
- [ ] `V3-P09` Run one common 512-ordinal P0/P-control/P* Plan-only screen.
- [ ] `V3-P10` Assemble Planner terminal decision and failure taxonomy.

Planner pass gate:

- P* parse/completion drop versus P0 is at most 0.5 percentage points;
- P* composition validity is at least 2 points above P0 and strictly above
  P-control;
- fixed-panel target NLL is no worse than +1% relative to P0;
- unique-formula rate is at least 95% of P0;
- mean atom count differs by at most 0.5;
- no registered marginal TVD worsens by more than 0.02;
- all-metal and single-element shortcut rates do not inflate.

If P* fails, record `[S]` and retain P0. Do not modify P* after seeing the
512-sample output.

## Phase 2D — Body-DLM factor

Registered arms:

- B0: frozen R5-C;
- B1: B0 plus `iid:current-order = 2:1`;
- B2: B0 plus `iid:compiled-PlanGraph = 2:1`.

Tasks:

- [x] `V3-D01` Implement B1/B2 corruption and inference schedules locally.
  - Result: targeted tests passed; Torch-only cases skipped locally.
- [x] `V3-D02` Verify historical two-GPU DDP contract.
  - Result: world size 2, batch/GPU 1, accumulation 8, global batch 16.
- [x] `V3-D03` Publish and hash the complete 45,229-row teacher-plan sidecar
  on the execution cluster.
  - Result: CPU-only job 29331 completed `0:0` in 7m33s; source archive SHA-256
    `df455438a3b00c6df3eaa54b99df43e95798dafa0d25355b00f985a367dcbde8`.
  - All 45,229 train/val/test rows were published with zero conversion
    failures. D1 and D2 each passed all 36,183 train+validation records with
    prompt/answer byte identity and the real frozen tokenizer.
  - Registered max length was 382; observed maxima were 333 train and 325
    validation. The job could not and did not submit training.
  - Evidence: terminal report SHA-256
    `75e9297a28b375cd594d9c55b77c313f13430cff1800b3c8c8287420cdd29ad8`.
- [x] `V3-D04` Run 2xA800 planned-corruption integration smoke on 32 rows.
  - Current execution: sequential B1/B2 array 29337 (`0-1%1`) completed both
    arms `0:0`; original CPU assembly 29338 produced a preserved false-negative
    terminal and exited `3:0`. CPU-only array-aware repair 29345 completed
    `0:0` under a new run root and published terminal SHA `5364d6b5…`.
  - Frozen contract: 32 train and 32 validation rows, 16 microbatches/rank,
    two optimizer updates, per-device batch 1, accumulation 8, global batch
    16, exact 16/16 no-padding validation shards, engineering-only, and no
    checkpoint serialization.
- [x] `V3-D05` Confirm unique validation denominators and rank-0-only
  checkpoint publication under DDP.
  - Current implementation: a deterministic no-padding validation sampler and
    exact rank-shard audit are locally tested. For world size two, 32 rows map
    16/16 and the full odd 9,047-row validation set maps 4,524/4,523 with no
    duplicate or missing index.
  - Runtime result: both arms used exact 16/16 shards, 32 unique validation
    rows, zero duplicate/missing rows, world size 2, and rank-0-only report
    publication; no checkpoint was serialized.
- [-] `V3-D06` Freeze learning rate and measured resource envelope.
  - Resource envelope frozen for future manifests: 2xA800, 8 CPUs, 64 GiB,
    two hours per arm, maximum concurrency one, global effective batch 16.
  - Learning rate remains open: engineering 5e-5 increased fixed-panel loss
    by 7.90% for B1 and 6.52% for B2 after two updates and is not accepted for
    scientific training.
- [ ] `V3-D07` Train B1/B2 for at most 400 updates, validation every 50.
- [ ] `V3-D08` Select B2 only by fixed NLL, completion, and paired dependency
  margin.
- [ ] `V3-D09` Assemble DLM terminal decision.

DLM pass gate:

- all losses/gradients are finite;
- B2 fixed-panel NLL is within +1% of B0;
- conditional completion drops by at most 1 point;
- B2 dependency margin is positive and strictly exceeds B1;
- checkpoint selection uses no generation or S.U.N.

If B2 fails, record `[S]` and retain B0. B1 is never promoted as the proposed
method.

## Phase 3 — Factorial engineering integration

- [ ] `V3-300` Freeze one immutable prompt/seed/ordinal ledger.
- [ ] `V3-301` Run M00/M10/M01/M11 for 32 engineering attempts each.
- [ ] `V3-302` Verify exact Plan SHA pairing within each Planner pair.
- [ ] `V3-303` Verify body/refiner noise pairing and ordered merge.
- [ ] `V3-304` Verify earliest-failure accounting and 32/32 raw denominators.
- [ ] `V3-305` Freeze G3 source, execution, authorization, and submission
  manifests.

This phase cannot select a scientific arm. An engineering failure may be
repaired with a new version and an explicit repair log.

## Phase 4 — G3 paired-256 factorial screen

- [ ] `V3-400` Run M00/M10/M01/M11 on the same 256 registered ordinals.
- [ ] `V3-401` Evaluate raw composition/structure/joint validity and
  completion.
- [ ] `V3-402` Evaluate unique, novel, strict S.U.N., and meta S.U.N. using the
  frozen evaluator snapshot.
- [ ] `V3-403` Compute paired bootstrap, McNemar tests, factorial main effects,
  and interaction.
- [ ] `V3-404` Assemble G3 terminal report and decision.

G3 pass gate:

- completion at least 97%;
- structure-validity drop at most 2 points versus M00;
- M10 retains the Planner gain and M01 has a positive DLM effect;
- M11 composition-valid target is at least +5 points versus M00;
- M11 joint-valid exceeds M00;
- strict S.U.N. gains at least 1 point or exceeds 10%;
- meta S.U.N. gains at least 3 points or exceeds 50%;
- M11 beats at least one single-factor arm;
- factorial interaction is not strongly negative;
- Phase-2 likelihood gates remain true.

## Phase 5 — G4 confirmatory study

- [ ] `V3-500` Request explicit user authorization after a passing G3.
- [ ] `V3-501` Freeze an independent confirmatory ordinal ledger.
- [ ] `V3-502` Run four arms at 1,000 attempts each if the registered resource
  budget permits.
- [ ] `V3-503` Produce raw and clearly labelled paper-compatible conditional
  metrics.
- [ ] `V3-504` Freeze figures, tables, confidence intervals, and paper result
  text.

No task in this phase may start automatically.

## Phase 6 — Minimal ablation and manuscript

- [ ] `V3-600` Planner ablation: P-control versus P*.
- [ ] `V3-601` DLM ablation: B1 versus B2.
- [ ] `V3-602` Optional fixed `iid:planned=1:2` diagnostic only after the main
  method passes; it cannot replace the registered 2:1 method.
- [ ] `V3-603` Freeze failure-attribution and distribution-drift appendix.
- [ ] `V3-604` Complete manuscript draft by 2026-09-12.
- [ ] `V3-605` Complete final verification by 2026-09-16.

The refiner is not modified in this workstream.

## Job and result ledger

| Stage | Run root | Job IDs | State | Result/report | Notes |
|---|---|---|---|---|---|
| V3 local preparation | local workspace | none | IN PROGRESS | 36 targeted Planner/data tests run: 34 passed and 2 registered Torch-only tests skipped locally; `LOCAL_VALIDATION_V3_H1A2_TWO_FACTOR.json` | No V3 Slurm submission |
| P* data/tokenizer preflight | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_pstar_data_preflight_v1` | 29318 | COMPLETE | `COMPLETED 0:0`; full data `4c2ba669…`; smoke data `25355701…`; full preflight `6f6802ff…`; smoke preflight `96da13af…` | 3,200/256 and 32/32 passed; max sequence 383/768; CPU-only |
| Planner smoke | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_pstar_smoke32_v1` | 29322_[0-1]%1 | COMPLETE | both arms `COMPLETED 0:0`, 32 microbatches / 4 updates, finite improving losses; P-control report `6ea059cd…`, P* report `12a9b87f…` | One A800 each, concurrency one; engineering only; no downstream triggered |
| DLM sidecar/tokenizer preflight | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_sidecar_preflight_v1` | 29331 | COMPLETE | `COMPLETED 0:0`; terminal `75e9297a…`; D1 `d49f32ad…`; D2 `2a275f2f…` | 45,229/45,229 sidecar rows; D1/D2 each 36,183/36,183; prompt/answer byte identity; max observed 333/325 under 382; CPU-only, no model/downstream |
| DLM two-GPU smoke | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_v1`; repair `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_assembly_repair_v2` | 29337_[0-1]%1; failed assembly 29338; repair 29345 | COMPLETE | B1/B2 and repair `COMPLETED 0:0`; corrected terminal `5364d6b5…`, gate true; failed terminal `894d4a70…` preserved | No GPU rerun; exact 2-rank/32-row DDP contract passed; LR 5e-5 not promoted |
| Planner 512 | pending | pending | TODO | pending | Raw all-attempt |
| DLM screen | pending | pending | TODO | pending | No generation selection |
| Four-arm 32 | pending | pending | TODO | pending | Engineering only |
| G3 paired-256 | pending | pending | TODO | pending | Scientific screen |
| G4 confirmatory | pending | pending | TODO | pending | Manual authorization required |

## Active next actions

1. Obtain separate authorization for one non-adaptive B1/B2 engineering
   recheck at the predeclared `5e-6`; do not run an LR sweep.
2. Wire the completed factorial contract into a fresh body/refiner runner and
   finish `V3-104` without modifying historical H1 code or assets.
3. Convert the prepared Planner and DLM scientific drafts into executable
   source/authorization/submission bundles only after the corresponding
   authorization and DLM LR gate.
4. Submit no 400-update training, Planner-512, crystal generation, S.U.N.,
   G3, or G4 job under the current authorization.

Prepared, non-executable drafts:
`execution/v3_scientific_training_drafts_v1/DRAFT_MANIFEST_INDEX.json`.

## Completed result log

### 2026-08-01 — Local deterministic Planner, objective, and data slice

- Command:
  `python3 -m unittest tests.test_h1a2_planner_batch tests.test_h1a2_lookahead_data tests.test_h1a2_planner_objective tests.test_ordinal_rng tests.test_h1_llm_planner`
- Result: 36 tests run: 34 passed and 2 Torch-only tests skipped because Torch
  is absent locally.
- Added exact raw Plan, canonical Plan, and body-prompt SHA-256 identities.
- Added `h1a2_ordinal_seed_v1`, independent of rank/world size/batch order.
- Added strict ordinal sorting plus duplicate/missing detection for Planner
  distributed merge.
- Added exact seven-line field parsing, offset ownership, label vocabularies,
  and the CPU reference field-balanced loss.
- Added deterministic largest-remainder joint-stratum selection for the common
  3,200-row train stream and fixed 256-row validation panel, with exact source
  SHA checking, content-hash ordering, no replacement, and shared
  P-control/P* rows.
- Added fail-closed additive prompt/answer tokenization, EOS parity, group IDs,
  causal boundary positions, and look-ahead label preparation.
- Added the dedicated frozen P0/P-control/P* trainer with exact official
  update/cadence checks, differentiable field/look-ahead losses, validation
  reports, and auditable checkpoint publication.
- Syntax compilation passed for the new/modified Planner, data, batching, and
  training modules.
- No model was loaded and no V3 remote job was submitted.

### 2026-08-01 — Frozen real H1-A2 source audit

- Source:
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/data/dlm_sft/mp_20_h1a2_rich_planner_noid_l3base`.
- Train: 27,136 rows, SHA-256
  `d431dfec1de8c3240dbc5648867be1b4b676fd85276e805a177b9944f3a1a157`.
- Validation: 9,047 rows, SHA-256
  `59327aa789ae5d2bbb66d8a8f0dc882d594bcc14623aa96ce95076ed1b6fc540`.
- Exact seven-line schema failures: 0 in both splits.
- All rows are `h1_llm_formula_plan`, prompt style `h1_rich_plan_v1`, and
  `sample_weight=1.0`.
- Historical epoch-2 train config independently records Meta-Llama-3-8B,
  max length 768, and this same data directory.
- This was a read-only audit; H1 assets were not changed.

### 2026-08-01 — Preflight source-transfer stop

- The frozen source archive passed local and outer-host SHA/byte checks.
- The outer-to-nested terminal EOF raced the pasted payload and logged the
  user-maintained nested SSH out.
- No Slurm job was submitted and the registered run root was not created.
- Work stopped immediately under the frozen connection rule. Recovery is
  indexed in `TRANSFER_INCIDENT_V3_20260801.md` and requires the user to
  restore `ssha800:1.0`.

### 2026-08-01 — H1-A2 P* data and real-tokenizer preflight

- Job 29318 completed `0:0` in 14m13s; stderr was empty.
- Materialized the frozen common P-control/P* streams: 3,200 train / 256
  validation and 32 train / 32 validation smoke rows, all with unique source
  identities.
- Full data manifest SHA-256:
  `4c2ba66923ccea16244cbc7ec138e69e07f50c4bca6ec7bfa1c631cd7f406c4d`.
- Smoke data manifest SHA-256:
  `2535570102079d055abaf99e22a927523d6f212ae49d7c03e28a52ca68606d9b`.
- Full and smoke real-tokenizer report SHA-256 values:
  `6f6802ff9ad77dd2e50b24761b72bbe8c748bdf90a251363b4a2e0927e3d61e4`
  and
  `96da13af8951ca5da78a93ca237bf4465adef837c38da6d54aae91b442eb1b9b`.
- The tokenizer is the frozen H1-A2 epoch-2 fast tokenizer. Full-train token
  lengths were 369–383 and full-validation lengths were 370–382, below the
  registered maximum 768. Field mapping, answer EOS parity, and
  formula-before-lattice causal boundaries all passed.
- `automatic_downstream=false` and
  `scientific_training_authorized=false` remained unchanged.

### 2026-08-01 — One-A800 P-control/P* numerical smoke

- Slurm array 29322 completed both elements `0:0`; P-control elapsed 5m08s
  and P* elapsed 3m12s including full model and adapter loading.
- Both arms used exactly one NVIDIA A800-SXM4-80GB, eight CPUs, batch one,
  accumulation eight, 32 microbatches, four optimizer updates, seed 17,
  no shuffle, and no generation/S.U.N selection.
- P-control target NLL improved `0.2682495 -> 0.2676392`; field loss improved
  `0.5183716 -> 0.5178833`; peak reserved CUDA memory was 17,622,368,256
  bytes.
- P* target NLL improved `0.2682495 -> 0.2676697`; field loss improved
  `0.5183716 -> 0.5177612`; look-ahead loss improved
  `2.2856124 -> 2.1956991`; peak reserved CUDA memory was 17,647,534,080
  bytes.
- P-control training/engineering report SHA-256 values:
  `3e9b07b12be11bd8ae739e5486b8f30a6729e282b4c9f321f371851e653e4142`
  and
  `6ea059cdd99c2491d0ec563a30c3a0745bd496cb09899bdfe79d50b5412744f6`.
- P* training/engineering report SHA-256 values:
  `727ac61f166858cd54933a03314a2a4836113480ea8a66c018681be1cb7e540a`
  and
  `12a9b87f88284969c64f6f34cfe752e09fd9b5ae5e0bd3a952aa378240f1f011`.
- The only stderr item was a PyTorch deprecation `FutureWarning`; the
  OOM/NaN/CUDA/NCCL/traceback/exception scan was empty.
- Both engineering reports kept `automatic_downstream=false` and
  `scientific_training_authorized=false`.

### 2026-08-01 — DLM sidecar and D1/D2 real-tokenizer preflight

- CPU job 29331 completed `0:0` in 7m33s; stderr was empty.
- Published all 45,229 frozen R5-C teacher-body records: 27,136 train, 9,047
  validation, and 9,046 test, with zero failed conversions and zero duplicate
  training pairs.
- D1 current-order and D2 compiled-PlanGraph preflight each passed all 36,183
  train+validation records. Prompt/answer byte identity, additive
  tokenization, semantic length, and dependency-group construction all passed.
- Registered maximum model length was 382. Observed maxima were 333 train and
  325 validation for both D1 and D2.
- Sidecar manifest SHA-256:
  `b38f6e414893b086fa03e5d049a4d3105a4ac153e6baf68faabe6b2a1ff4616b`.
- D1 and D2 report SHA-256 values:
  `d49f32adf347a498631967271d266d4bb8d3f8314d05b248e7f4d6cf68bf2a2f`
  and
  `2a275f2fd8881e4d2779fea193565b7d5e76d4f6834feaadcdd1665d4aad07ac`.
- Terminal report SHA-256:
  `75e9297a28b375cd594d9c55b77c313f13430cff1800b3c8c8287420cdd29ad8`.
- The source Plan remains explicitly labelled
  `structure_derived_teacher_plan_state` and is confined to body-DLM
  supervision. It is not a model-sampled inference Plan.
- The run used no GPU or model weights and kept
  `automatic_downstream=false`, `scientific_training_authorized=false`,
  crystal generation off, and S.U.N. evaluation off.

### 2026-08-01 — Exact DDP validation denominator implementation

- Added a deterministic rank-strided validation sampler with no padding.
- For world size two, the 32-row smoke panel is exactly 16/16 and the full
  odd 9,047-row validation split is exactly 4,524/4,523.
- The runtime report fails closed on any duplicate, missing, out-of-range, or
  rank-order-drifted index and records an ordered SHA-256 for every rank.
- Distributed validation loss now aggregates weighted loss mass and sample
  weight rather than averaging rank/batch means.
- Focused local command:
  `python3 -m unittest tests.test_distributed_data`.
- Result: 5/5 passed; runtime 2xA800 confirmation remains part of `V3-D05`.

### 2026-08-01 — B1/B2 two-A800 engineering smoke submitted

- Frozen source installation passed 9/9 targeted tests on A800, including the
  no-padding distributed sampler, weighted validation aggregation, and
  fail-closed terminal assembler.
- Array 29337 runs B1 then B2 with `0-1%1`; each element requests two A800s,
  8 CPUs, 64 GiB, and one hour. CPU assembly 29338 uses `afterany:29337`.
- Source archive SHA-256:
  `e8e2917b79eff360018c0bfe25352926b83f7d4064dc97a2dd34745266ae4525`.
- Execution manifest SHA-256:
  `10c6d802e437a2ccf4b6fa0c54aa52f48a75f404fe2d24749cbfff5082f1c8b5`.
- Authorization SHA-256:
  `7786f43727ffdcd8cbddcaf9818f73f616de82f779d7d3f45497f1a58e4a8a29`.
- Submission record SHA-256:
  `dae849f8c57698b05454c34dace0b3555a398614099f2bf2a1763a74a15aa36d`.
- Scientific training and automatic downstream remain unauthorized and false.

### 2026-08-01 — B1/B2 runtime result and assembly repair

- B1 and B2 both completed `0:0` using exactly two A800s, world size 2,
  global batch 16, 32 training sequences, two optimizer updates, and 32 unique
  validation rows with zero duplicate/missing rows.
- B1 fixed-panel loss was 2.2864404554 before training and 2.4671571362 after
  two updates; B2 was 2.2864404926 and 2.4354479681. These are engineering
  results only and do not authorize checkpoint use.
- Peak monitored memory was 47,036 MiB on rank/GPU 0 and at most 40,516 MiB
  on rank/GPU 1, within the 64-GiB host allocation and 80-GiB A800 limit.
- Assembly 29338 failed `3:0` because its parser keyed `sacct` by JobIDRaw:
  Slurm returned raw allocation IDs 29339 and 29337 while the stable array
  JobIDs were 29337_0 and 29337_1. Failed terminal SHA-256 is `894d4a70…`.
- A new CPU-only repair bundle matches by JobID, retains JobIDRaw as evidence,
  passes two parser tests plus a full local reconstruction, and is submitted
  as job 29345. It completed `0:0`; corrected terminal SHA-256 is
  `5364d6b589036f163c026e52e1964558525477d557faf7521bea90761ac1346a`
  with `engineering_gate_passed=true`. It changes no scientific protocol and
  reruns no GPU work.

### 2026-08-01 — H1-A2 I/O and factorial identity contract

- Added `crystal_dlm/h1a2_factorial_contract.py` without changing the
  historical H1 sampler or any H1 asset.
- P0 and P* now have a testable exact inference-input identity: same
  no-sample-id prompt bytes, token IDs, tokenizer identity, chat template,
  rich seven-line schema, and parser; only checkpoint identity may differ.
- Persisted inference Plans must be exact seven-line raw model outputs.
  Missing fields, added renderer text, teacher-plan provenance, SHA mismatch,
  retry/replacement/repair/filter/rerank flags, or ordinal/seed mismatch fail
  closed.
- M00/M01 consume byte-identical persisted P0 Plan/body prompts; M10/M11 do
  the same for P*. Planner/body/refiner seeds are stateless per ordinal.
- The four-arm merger sorts by ordinal and fixed arm order and rejects every
  duplicate, missing, out-of-range, or stray arm identity.
- Focused command:
  `python3 -m unittest tests.test_h1a2_factorial_contract tests.test_ordinal_rng tests.test_h1_llm_planner`.
- Result: 27/27 passed; validation record SHA-256
  `6ba1a1fe32081a28d23f4a5c5096da74962b10e014b7e0500e3d9bf02f4c576a`.
- Scientific training drafts were prepared with zero job IDs and
  `automatic_downstream=false`. The DLM draft rejects smoke LR `5e-5` and
  records one recommended, separately authorized `5e-6` engineering recheck;
  it does not freeze or submit that candidate.
- The self-contained staging archive was copied through the restored
  outer/nested path to
  `/runs/20260801_h1a2_v3_contract_drafts_staging_v2`: archive SHA matched on
  local/outer/A800, 5/5 key-source hashes matched, and the 7 new contract
  tests passed on A800. No model, GPU, or Slurm job was used.
- The earlier minimal staging v1 is retained as transfer/test evidence: its
  file hashes passed, but it omitted imported repository modules. The bounded
  packaging-only repair is recorded in
  `repair_log/V3_CONTRACT_STAGING_BUNDLE_V2_20260801.json`.
