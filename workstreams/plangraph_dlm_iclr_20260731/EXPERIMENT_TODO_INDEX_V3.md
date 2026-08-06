# PlanGraph-DLM V3 Experiment TODO Index

Status: `phase2t_parallel_planner_and_nconditioned_noplan_plans_registered`

Last updated: 2026-08-06

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
- `[E] ENGINEERING STOP`: a registered implementation/reproducibility gate
  failed; evidence is retained and scientific results are not inferred.
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
| P* scientific method | COMPLETE | Look-Ahead Consistent Planner v1 is frozen, implemented, and passed the registered one-A800 32-row P-control/P* smoke | `PLANNER_PSTAR_METHOD_V1.md`; job 29322; smoke reports in the result ledger |
| P* deterministic data builder | COMPLETE | Common P-control/P* streams were materialized from frozen H1-A2 data with exact 3,200/256 quotas, source-SHA fail-close, and identical row order | job 29318; P* data/tokenizer preflight row in the result ledger |
| Real H1-A2 Planner source | COMPLETE | train 27,136 / val 9,047; all rows use the exact seven-line schema, `h1_rich_plan_v1`, and weight 1.0; source SHAs frozen | Registry `frozen_identity.planner_source_data`; completed-result log below |
| Preflight/smoke authorization | COMPLETE | CPU materialization, real-tokenizer preflight, and bounded 32-row Planner/DLM engineering smoke were authorized and completed | `AUTHORIZATION_V3_PREFLIGHT_SMOKE_20260801.json` |
| Scientific-training authorization | COMPLETE | P-control/P* 400-update training, B1/B2 one-epoch training, and registered likelihood-only checkpoint selection were authorized and completed; formal G3/G4 remained outside that authorization | `AUTHORIZATION_V3_SCIENTIFIC_TRAINING_20260801.json`; jobs 29391–29394 |
| Planner/dependency screens | SCIENTIFIC STOP | Planner-512 retained P0 and the paired dependency screen retained B0; neither proposed factor passed its preregistered Phase-2 gate | jobs 29452–29457; terminal SHAs and metrics below |
| Post-stop refined S.U.N. diagnostic | COMPLETE / DIAGNOSTIC STOP | V7 supplied immutable generation, exact refine800, and direct metrics; evaluation-only V8 completed frozen-cache S.U.N. for all four arms. B2 sharply reduced generation/joint completion and neither P* nor B2 improved strict S.U.N.; Phase-2 scientific stops remain in force. | `execution/v3_poststop_sun256_diagnostic_v7`; `execution/v3_poststop_sun256_evaluation_repair_v8`; V8 terminal SHA `6d2e26d2…` |
| V3 remote jobs | COMPLETE | All four V8 array elements and assembly 29618 completed `0:0`; terminal status `complete`, decision `diagnostic_only_retain_phase2_scientific_stops` | jobs 29617/29619/29620/29625 and 29618; source manifest `6238e7f0…`; submission record `0b2bf596…` |
| H1 single-variable recovery | R03 SCIENTIFIC STOP / R03B+R03C+R03D GATE PASS / R03E SCIENTIFIC STOP / R03F ENGINEERING STOP / R03G REPORT-ONLY / R03H ATTRIBUTION COMPLETE | The exact H1 D1 control reproduced 31/32 completion, while original mixed-axis D2 fell to 14/32. Safe-axis restored 31/32 at R03B, improved 61→63 successes at R03C, and improved 246→248 of 256 at R03D, with zero duplicate-coordinate failures and no new failure class. In four frozen-refiner repeats, candidate minus control was +5/1024 joint, +4/1024 strict S.U.N., but -28/1024 meta S.U.N.; the frozen R03E gate therefore stopped the signal. R03F resolved all 107 deduplicated MP queries into one 227-system snapshot, then failed closed at the completed-hull zero-unknown gate. R03G reused that snapshot offline with only that gate disabled: 753/825 unknowns resolved, 72 remained and were scored false; strict changed to +18/1024 while meta remained adverse at -27/1024. R03H attributed strict +18 to +16 finite-hull crossings and +2 novel-unique eligibility, while meta -27 came entirely from finite-hull crossings; the 72 residual unknowns were exactly paired and contributed zero to both effects. These report-only results do not reopen the R03E stop. Full frozen reproduction dossier: `H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md`. | R03B terminal `57d53922…`; R03C terminal `34eb0c18…`; R03D terminal `fa03cfa2…`; R03E terminal `7fa49a6f…`; R03F failure `4380b0f2…`; R03G terminal `3cb705ea…`; R03H report `566dd3c5…` |
| Next-stage Planner/chemistry/DLM-RL feasibility | COMPLETE / HISTORICAL DECISION INPUT | Three independent studies were reconciled against the frozen H1/R03 evidence. Frozen P-control confirmation and formula charge-reachability were conditional-go Planner directions. Two pre/post-refiner RL LoRAs were investigated only as a diagnostic concept, never execution-authorized; the later full Body audit and ICLR portfolio supersede that option with one post-ICLR multi-fidelity policy and `0 GPUh` for this submission. No experiment or training was started. | `H1_PLANNER_CHEMISTRY_DLM_RL_FEASIBILITY_REPORT_V1.md`; report SHA `57663b4e…` |
| Body-DLM complete protocol and mask-aware RL design | COMPLETE / DECISION-ONLY | The complete B0/R5-C → D1 → frozen `model_494` refine800 pipeline was audited down to its 2,481 immutable crystal special tokens, legal supports, exact `7+4N` layout, and stochastic `6+3N` slots. A follow-up coverage audit found only 1,437/2,481 tokens represented in the local 9,046-row held-out exact corpus and 1,013/2,343 schema-defined stochastic-action tokens held-out-unseen, dominated by 953 length tokens. This is not yet a full-train result, so RL now has a mandatory Gate −1 full-corpus coverage/legal-mass audit. Historical TraceRL remains formal-experiment NO-GO. No training or experiment was started. | `H1_BODY_DLM_COMPLETE_PROTOCOL_AND_RL_DESIGN_V1.md`, SHA `ff473ac1…`; `H1_BODY_SPECIAL_TOKEN_COVERAGE_AUDIT_V1.md`, SHA `070bbce7…` |
| ICLR 2027 time-boxed improvement portfolio | COMPLETE / DECISION-ONLY | Three independent proposals, three cross-reviews, one independent red-team review, and a release consistency audit selected exactly one conditional main line: frozen `P0` versus `P0+CR-Plan`. `PILS-L` is Gate −1-only cold backup; an early 08-08 novelty or 08-10 engineering stop grants only backup eligibility, while actual replacement requires a release-ready execution annex and new authorization by 08-10. Mask-aware RL, two-model RL, compiler, reachability-mass SFT, and further safe-axis work are CUT for this submission; RL has `0 A800 GPUh`. Hard science cutlines are 08-15 paired-64, 08-22 paired-256, 08-31 independent `4×256`, and 09-05 science freeze. This portfolio authorizes no experiment. | Roadmap SHA `47e0b9fe…`; narrow re-audit `c70e5c0a…`; release manifest `roadmap_reviews/ROADMAP_RELEASE_MANIFEST_V1.json`, SHA `930bb766…` |
| CR-Plan R0 execution | V4 ENGINEERING TERMINAL / E1 PHYSICAL PASS / AMENDMENT-ONLY | The original CR-0 timeout and clean paired-32 pass remain preserved, while the old missing-state policy keeps paired-32 ineligible for a prefix-reachability claim. V4 retained exact parity but failed its immutable state gate (`415,689 > 100,000`). A later, separately authorized E1 sidecar asked only the distinct physical-cost question: 18/arm real-model attempts gave full/off median `1.468x`, exact trace/scalar parity, and preterminal support differences in `14/14` applicable attempts. E1 does not repair V4 or authorize 512; it only makes a new preregistered route amendment technically eligible. | Paired terminal `a1a34a66…`; V4 terminal `55df7801…`; E1 source `aab2c113…`; archive `c0766a6b…`; terminal `b7c8e2bf…`; `H1_CRPLAN_MISSING_POLICY_AND_SUPPORT_OPTIMIZATION_REVIEW_V1.md` |
| N-conditioned Planner-free DLM parallel plan | COMPLETE / DECISION-ONLY PARALLEL AMENDMENT | User direction opens an independent `H1-NPDLM` planning track beside CR-Plan. The first treatment keeps exact `7+4N`, freezes `N`, generates the `N` element tokens plus lattice/coordinates, and compares against a same-update rich-Plan continuation. It separates a paired `N_from_P0` Plan-information ablation from an operational train-prior `N` view, freezes pure SFT before chemistry/RL, and stages Gate −1 → 32 → 64 → 256 → independent `4×256`. CR-Plan remains the only currently allocated headline; NPDLM has `0 authorized GPUh` until a separate global portfolio/execution annex freezes compute, numeric gates, and paper allocation before R1. | `H1_NOPLAN_NCONDITIONED_PARALLEL_PLAN_V1.md`, SHA `14578ad8…`; release manifest `roadmap_reviews/NPDLM_PARALLEL_PLAN_RELEASE_MANIFEST_V1.json`, SHA `02badd21…` |
| Raw/canonical Plan identity repair | COMPLETE LOCALLY | Parser-accepted raw and canonical text now retain independent SHA-256 identities; 8/8 focused contract tests pass | `crystal_dlm/h1a2_factorial_contract.py`; `tests/test_h1a2_factorial_contract.py` |
| Factorial runtime | COMPLETE | Fresh P0/P* Planner, B0/B* body, frozen-refiner, and four-arm all-attempt runners are wired. Raw seven-line conformance is now advisory at inference; semantic identity and ordinal accounting remain strict. | `FACTORIAL_RUNTIME_VALIDATION_V1.json`; v4 diagnostic source and tests |
| DLM one-epoch amendment | COMPLETE | B1/B2 must each consume all 27,136 rows once: 1,696 updates at global batch 16 and user-directed historical LR 5e-5; adverse two-update smoke evidence is retained and the separate LR recheck is superseded | `PROTOCOL_AMENDMENT_V3_DLM_ONE_EPOCH_20260801.json`; `PROTOCOL_OVERRIDE_V3_DLM_LR5E5_AUTHORIZED_20260801.json` |
| DLM fixed 100-row panel | COMPLETE | Real 9,047-row validation ledger audited; rank 0 selects even ordinals 0–98 and rank 1 odd ordinals 1–99, giving global ordinals 0–99 exactly once | `DLM_FIXED_PANEL_AUDIT_V1.json`; panel ledger SHA `f4fad132…` |
| Checkpoint retention and failed-run cleanup | COMPLETE | Future arms retain selected best ∪ latest two after selection; 59 failed/no-promotion checkpoint payloads removed, freeing 118,287,679,488 bytes (110.164 GiB); H1A2/R5-C protected | `CHECKPOINT_CLEANUP_20260801.json`; `crystal_dlm/checkpoint_retention.py`; `scripts/prune_checkpoints.py` |

## Immutable execution rules

- The H1-A2 seven-line schema remains the Planner output target, but exact raw
  line conformance is advisory rather than a downstream blocking gate.
- Raw Planner text and every format deviation are preserved and labelled.
  When the historical H1-A2 parser has already accepted an ordinal, its frozen
  canonical `plan_text` continues to body generation; an originally
  `parsed=false` ordinal remains a Planner failure.
- The Planner model proposes every visible Plan value.
- No sample ID, retry, replacement, repair, filtering, reranking, or
  survivor-prefix denominator.
- Raw model Plan text, canonical Plan text, compiled body prompt, and their
  identities are retained per ordinal.
- Teacher Plans may train B1/B2 but may never replace sampled P0/P* Plans.
- The continuous CrysLLMGen refiner remains frozen. In every generated-crystal
  experiment, all successful body outputs must pass through the same
  `model_494` checkpoint for exactly 800 reverse steps before any direct
  metric or S.U.N. evaluation.
- Every future frozen runtime that invokes the A100 S.U.N. evaluator must be
  assembled by `scripts/a800/stage_crysllmgen_a100_sun_runtime.py`; hand-picked
  copies of the runner are prohibited.
- Before submission and again before any model/GPU/scientific work, that
  runtime must pass the isolated `python -I` import-origin preflight. Every
  required module must exist inside the frozen runtime root; shared-checkout
  fallback and delayed entrypoint imports are fail-closed.
- Raw all-attempt metrics are primary. Paper-compatible conditional metrics
  are secondary reports only.
- No S.U.N., MP, energy, hull, CHGNet, or generated-crystal result is used for
  training or checkpoint selection.
- Checkpoint pruning occurs only after checkpoint selection and independently
  within each training arm. The retained set is selected best union the latest
  two numeric-step checkpoints. A dry run is mandatory before `--apply`.
- Historical H1A2, canonical R5-C, and the formal CrysLLMGen anchor are exempt
  from cleanup and remain read-only.
- Every future scientific comparison begins from an exact H1 control and may
  change only one principal factor: Planner, body checkpoint, body decoding
  schedule, or evaluation coverage. A combined candidate is forbidden until
  its constituent factors pass independently.
- New decoding or model candidates first run on 32 or 64 frozen attempts.
  Expansion to 256 is blocked by any unexplained failure class, excess
  duplicate-coordinate failure, or more than two points of completion loss.
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
- [x] `V3-005` Enforce post-selection checkpoint retention and remove failed
  checkpoint payloads.
  - Result: a fail-closed dry-run-first utility retains selected best ∪ latest
    two within one training arm. Seven focused tests passed.
  - Cleanup: 59 checkpoint directories from 36 explicitly failed,
    diagnostic-only, or no-promotion runs were removed. Run directories,
    `final` artifacts, reports, and logs were retained.
  - Capacity: checkpoint storage fell from 184,092,106,752 to
    65,804,427,264 bytes, freeing exactly 118,287,679,488 bytes
    (110.164 GiB).
  - Protection: all H1A2 checkpoints, canonical R5-C, the formal CrysLLMGen
    anchor, and the external R5-C shared root were untouched.
  - Evidence: `CHECKPOINT_CLEANUP_20260801.json`;
    `crystal_dlm/checkpoint_retention.py`;
    `scripts/prune_checkpoints.py`.

## Phase 1 — Input/output and deterministic execution

- [x] `V3-100` Register identical Planner prompt, tokenizer, chat template,
  parser, and seven-line target schema for P0/P*.
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
- [x] `V3-104` Make distributed merge order strictly ascending by ordinal and
  reject duplicate/missing attempt identities.
  - Result: fresh Planner, body, and refiner runners each perform strict
    ordinal merge; the four-arm assembler rejects duplicate, missing, stray,
    or out-of-order identities and preserves all upstream failures.
  - Evidence: `FACTORIAL_RUNTIME_VALIDATION_V1.json`; local and A800 targeted
    tests both passed 50/50.
- [x] `V3-105` Add tests proving P0/P* inference input identity and
  M00/M01, M10/M11 body-prompt identity.
  - Result: P0/P* exact prompt bytes, token IDs, tokenizer/chat-template
    identity, no-sample-id input, sampled-Plan provenance, within-Planner
    Plan/body-prompt hashes, shared noise, and duplicate/missing rejection all
    pass fail-closed tests. Exact raw seven-line conformance is recorded but
    no longer blocks a historically parsed ordinal from downstream use.
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
4. every identity, ordinal, seed, provenance, or semantic-parse mismatch fails
   closed; raw-format-only deviations are retained as nonblocking warnings.

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
- [x] `V3-P07` Train P-control and P* for exactly 400 updates, validation every
  50 updates.
  - Result: jobs 29391/29392 completed `0:0`; both arms consumed all 3,200
    microbatches for exactly 400 updates and published eight scheduled
    checkpoints.
- [x] `V3-P08` Select one checkpoint per trained arm without generation or
  S.U.N.
  - Result: P-control step 400 and P* step 400 are frozen for Planner-512.
    Both are complete one-epoch endpoints. P* step 400 fixed-panel target NLL
    is `0.2923660`, or `0.99347×` the frozen P0 value, and therefore passes
    the registered +1% noninferiority gate. The earlier likelihood-only P*
    step-350 preference is superseded for the downstream screen to avoid
    unequal training exposure.
  - Evidence: Planner terminal SHA `ecaf779e…`;
    `PROTOCOL_OVERRIDE_V3_PLANNER_FULL_EPOCH_ENDPOINT_20260801.json`.
- [x] `V3-P09` Run one common 512-ordinal P0/P-control/P* Plan-only screen.
  - Result: array 29452 (`0-2%1`) and assembly 29454 all completed `0:0`.
    P0/P-control/P* composition validity was respectively
    `434/512 = 84.77%`, `456/512 = 89.06%`, and
    `442/512 = 86.33%`, on the same raw all-attempt ordinals.
- [S] `V3-P10` Assemble Planner terminal decision and failure taxonomy.
  - Decision: `scientific_stop_retain_P0`; `Pstar_selected=false`.
  - P* gained only `+1.5625` composition-valid points over P0, below the
    registered +2-point gate, did not beat P-control, and inflated the
    registered all-metal shortcut rate. Parse/completion, NLL
    noninferiority, uniqueness, mean-N, and registered marginal-drift gates
    passed.
  - Evidence: terminal SHA
    `09f66256f5d5f96d0a4b161770801adaea5da9e75758389ff45b10d0680f3c0c`.

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
- [x] `V3-D06` Freeze learning rate, full-epoch budget, and measured resource
  envelope.
  - Result: the user explicitly froze the historical LR at `5e-5`; there is
    no LR sweep or separate two-update acceptance gate. The two-update smoke
    degradation remains recorded as risk evidence.
  - One epoch is exactly 27,136 rows / global batch 16 = 1,696 optimizer
    updates. Validation is frozen at steps
    `0,212,424,636,848,1060,1272,1484,1696` on one common 100-row panel.
  - Resource envelope remains 2xA800, 8 CPUs, 64 GiB, two hours per arm,
    maximum concurrency one. Historical R5-C completed the same 1,696-update
    two-A800 epoch in 37m58s.
  - The exact common 100-row panel passed the real-ledger audit: report SHA
    `47c3a591…`, ledger SHA `f4fad132…`, zero duplicates/padding, and identical
    B1/B2 plus initial/intermediate/terminal membership.
  - Evidence: `PROTOCOL_AMENDMENT_V3_DLM_ONE_EPOCH_20260801.json`,
    `PROTOCOL_OVERRIDE_V3_DLM_LR5E5_AUTHORIZED_20260801.json`, and
    `DLM_FIXED_PANEL_AUDIT_V1.json`.
- [x] `V3-D07` Train B1/B2 for exactly one complete epoch: 1,696 updates,
  without metric-based early stopping.
  - Result: jobs 29393/29394 completed `0:0`; both arms consumed all 27,136
    rows once and validated at steps `0,212,...,1696`.
  - Fixed-panel target NLL changed from the common initialization
    `1.969803` to `1.460710` for B1 and `1.466090` for B2, reductions of
    25.84% and 25.57%, respectively. Both pass the registered likelihood
    noninferiority gate.
  - Evidence: DLM terminal SHA `ea22ced9…`;
    `DLM_LOSS_COMPARISON_V1.json`.
- [x] `V3-D08` Select B2 only by fixed NLL, completion, and paired dependency
  margin.
  - Result: array 29456 (`0-1%1`) and assembly 29457 completed `0:0`.
    B1/B2 both used terminal step1696 on the fixed validation ordinals 0–99.
- [S] `V3-D09` Assemble DLM terminal decision.
  - Decision: `scientific_stop_retain_B0`; `Bstar_selected=false`.
  - B1 margin was `0.259809`; B2 remained positive at `0.233067`, but
    `B2-B1 = -0.026741` with paired bootstrap 95% CI
    `[-0.055592, 0.001986]`. B2 therefore failed the strict
    dependency-margin improvement gate despite passing fixed-panel NLL
    noninferiority.
  - No generation, S.U.N., energy, hull, or shuffle arm was used for this
    decision.

DLM pass gate:

- all losses/gradients are finite;
- B2 fixed-panel NLL is within +1% of B0;
- conditional completion drops by at most 1 point;
- B2 dependency margin is positive and strictly exceeds B1;
- checkpoint selection uses no generation or S.U.N.;
- only the terminal step-1696 checkpoint is eligible.

If B2 fails, record `[S]` and retain B0. B1 is never promoted as the proposed
method.

## Phase 2R — H1 control restoration and single-variable recovery

- [E] `V3-R00` Reproduce the immutable H1 P0+B0+model_494 path.
  - Scope: one P0 arm, all original 256 attempts, original H1 attempt ledger,
    R5-C body batch 8, original paired body/refiner noise, refiner batch 64,
    and exactly 800 reverse steps.
  - Result: job 29646 ended `FAILED 1:0` on the preregistered byte gate.
    `body_attempts.jsonl` is byte-identical and all 246 proposal-graph tensors
    are exact, but all 246 successful continuous structures differ.
  - Diagnosis: both runs used node99 and identical discrete inputs. CUDA
    `torch_scatter` reduction non-determinism is amplified over 800 diffusion
    steps, so continuous byte equality is not a valid reproducibility gate.
  - Disposition: retain the stop; use the versioned continuous-refiner
    amendment rather than weakening this failed result post hoc.
  - Excluded: Planner sampling, training, direct metrics, S.U.N.,
    checkpoint selection, and automatic downstream.
- [x] `V3-R01` Separate raw model Plan identity from canonical parser identity.
  - Result: `persist_parser_accepted_model_sampled_plan` hashes the immutable
    raw bytes independently while compiling only the frozen parser-accepted
    canonical Plan.
  - Test: 8/8 focused factorial-contract tests pass, including a nonconforming
    advisory raw line that continues without hash mismatch.
- [x] `V3-R02` Run the corrected real-ledger Plan conversion preflight.
  - Gate: every original `parsed=true` ordinal reaches body compilation; no
    `raw sampled Plan SHA mismatch` remains.
  - Execution: CPU-only job 29658, source manifest
    `3893728fbe1ca6b2cb087fbdb1620a64eab606be8c08d6d96d6a665dabe9456a`;
    no model, GPU, generation, refinement, metric, or downstream action.
  - Result: `COMPLETED 0:0`; P0/P-star each retained all 256 attempts,
    compiled all 253 original `parsed=true` ordinals, preserved all three
    original Planner failures, and recorded zero conversion errors. Report
    SHA-256: `270443a0a30dbc43f9363d6b95fab4b3eaae714a37b6f2382918ea11636fe0c3`.
- [S] `V3-R03` Isolate body scheduling on 32 frozen H1 P0 Plans.
  - Arms: B0 weights with H1 exact-plan schedule versus B0 weights with one
    corrected PlanGraph schedule.
  - Pairing: both arms run in one Slurm job with the same loaded B0 model,
    same batch partition, same prompts, ordinals, and per-ordinal body-noise
    seeds. Historical H1 first-32 output is read-only calibration, not the
    paired control.
  - Execution: job 29669, source manifest
    `1b27f9f28ecf0d17f1a28aaf886718a457631f6035e23e883918d2499b16e365`;
    32/32 frozen rows are eligible and 32/32 have an active D2 treatment.
  - Result: `COMPLETED 0:0`; terminal status `complete`, gate false, decision
    `scientific_stop_retain_h1_exact_plan`. D1 retained `31/32` successes and
    the historical failure class; D2 achieved only `14/32`, a 17-count drop,
    with 18 new `body:DuplicateCoordinateError` failures. Paired input
    mismatches were zero and the shared batch partition was exact.
  - Evidence: terminal SHA-256
    `898191bfe23b66ecf811eb8b223d1a7356181b273a8468c2439a926d213b09e3`;
    generation report SHA-256
    `fdc9aa4939274a2a44173358f7735f53a7bd7648a161c31ec979bfbc50991835`.
  - No candidate checkpoint, Planner change, refinement change, direct metric,
    or S.U.N. is allowed in this stage.
- [x] `V3-R03-DIAG` Diagnose the frozen R03 D2 failures without rerunning.
  - Result: all 18 failures are complete, Plan-matched body outputs rejected
    at graph construction for exact coordinate collisions. D2 has 93
    duplicate pairs versus zero for D1: 51 within element groups and 42
    across groups. Thirteen failed samples contain cross-group collisions.
  - Axis signature: mean unique-X and unique-Y fractions are effectively
    unchanged, while unique-Z falls from `0.65496` under D1 to `0.51105`
    under D2.
  - Root cause: D2 mixes XYZ inside confidence-ranked element groups, but the
    frozen duplicate mask only constrains Z when X/Y are already known. H1 D1
    globally fills X and Y before Z; D2 does not.
  - Statistical evidence: paired exact McNemar
    `p=1.52587890625e-05`; D2 failure-rate Wilson 95% interval
    `[39.33%, 71.83%]`.
  - Evidence: `H1_R03_D2_SCHEDULE_DIAGNOSIS_V1.md`;
    `analysis/r03_schedule_diagnosis.py` SHA `085c0fb4…`.
- [x] `V3-R03B` Register and test one H1-preserving safe-axis PlanGraph
  schedule.
  - Candidate order: lattice; PlanGraph-grouped X blocks; PlanGraph-grouped Y
    blocks; PlanGraph-grouped Z blocks.
  - Pre-GPU invariant: every active X/Y precedes every active Z and
    `z_before_xy_count=0` in synthetic reveal traces.
  - First scientific test remains schedule-only paired 32:
    `P0+B0+D1` versus `P0+B0+D2-safe-axis`; no mask, model, data, seed,
    denominator, refinement, or evaluation change.
  - H1/R5-C exact-length is a hard invariant: both arms retain exactly
    `7+4N` output positions, the same count/element prefill, dynamic schema
    constraints, body prompt, checkpoint, and paired stateless noise.
  - Local and A800 tests: 7/7 passed. Real frozen first-32 ledger preflight:
    32/32 body-eligible, 32/32 treatment active, `N=4..20`, exact-length
    position coverage preserved, zero mixed-axis groups, zero
    `z_before_xy`, and all X/Y groups precede every Z group.
  - Sole Slurm job `29837` completed `0:0` in 8m40s; run root
    `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis32_v1`;
    source manifest SHA `cd21ee664f5c227698e96187906944a7d29016bd270da645c56ed6501cb3c866`;
    submission record SHA `d68c873b51cb50d299182663b29b46dfc71bdf32f5a7e746d03b4bc2ee5a3315`.
  - Result: gate passed with decision `safe_axis32_safe_to_expand`.
    D1 and D2-safe-axis were both 31/32, both had zero duplicate-coordinate
    failures and the same single `body:ValueError` failure class. Paired input
    mismatches were zero, shared partition identity passed, treatment applied
    to 32/32 attempts, and no new candidate failure class appeared.
  - Exact-length audit: all generated token counts matched `7+4N`; all 32
    invariant rows passed, with total mixed-axis groups `0`, total
    `z_before_xy=0`, and all X/Y groups before all Z groups. Terminal SHA-256
    `57d53922bd62e96aca2becee0b6f9c48d067111421c8e20a0b30dc1507b735c7`.
- [x] `V3-R03C` Confirm the R03B schedule-only result on paired first-64.
  - Scope remains one variable:
    `P0+B0+D1` versus `P0+B0+D2-safe-axis` in one Slurm job with one loaded
    model, shared batch partition, identical Plans/prompts/ordinals/body-noise
    seeds, and no checkpoint, mask, data, refinement, metric, or S.U.N.
    change.
  - Frozen denominator is raw ordinals `0..63`, all 64 body-eligible.
    Historical read-only H1 calibration is `61/64`; its three failures are
    ordinals 3, 36, and 40, all in the existing `body:ValueError` class.
  - H1/R5-C exact-length remains hard-frozen: both schedules cover exactly
    `7+4N` output positions with the same count/element prefill and schema
    constraints.
  - Local and A800 preflight passed 7/7 tests. Real first-64 audit passed
    64/64 exact-length coverage, 64/64 active treatments, and 64/64
    candidate invariants; mixed-axis groups and `z_before_xy` were both zero,
    and every X/Y position precedes every Z position.
  - Sole Slurm job `29844` completed `0:0` in 15m42s; run root
    `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis64_v1`;
    source manifest SHA
    `2c3485f8cf539ed5e33f3303a695947a2bd5fe72e83456d6f6f20cd5dca156bd`;
    submission record SHA
    `8efcf53b0e4a4de5c1ec5b2089ad00de5a7f682ebe7cfb8cdc48e878e4a73a8e`.
  - Result: gate passed with decision `safe_axis64_safe_to_expand_to_256`.
    D1 reproduced historical H1 at `61/64`; safe-axis achieved `63/64`.
    Both had zero duplicate-coordinate failures, safe-axis introduced no new
    failure class, and paired input mismatches were zero. The shared batch
    partition passed and treatment was active on all 64 attempts.
  - All 64 schedule invariant rows passed with exact `7+4N` coverage,
    mixed-axis groups `0`, `z_before_xy=0`, and every X/Y position preceding
    every Z position. Terminal SHA
    `34eb0c1804c9b9dfbb43009eea56b85438e15c8b27dc81c70dbb34044852824a`;
    generation report SHA
    `c1a15d70d7d3251767e36efd43da24097c9b52fcc6f288612ceef26c4575caac`;
    invariant report SHA
    `1bfa51952c7ffc37703f06a020f1fc81878002975fb4959c29281815792a2aeb`.
  - `automatic_downstream=false`; expansion to 256 and every subsequent
    stage still require a separately registered decision.
- [x] `V3-R03D` Confirm safe-axis as the only changed factor on paired all-256.
  - Scope remains `P0+B0+D1` versus `P0+B0+D2-safe-axis` in one Slurm job,
    one loaded model instance, one shared eligible-row batch partition, and
    identical Plans, prompts, ordinals, body-noise seeds, tokenizer, masks,
    prefill, temperature, and exact-length contract.
  - Raw all-attempt denominator is ordinals `0..255`. Exactly 254 frozen
    P0 rows are body-eligible; ordinals 86 and 211 remain the original
    `planner_parse_failed` attempts in both arms and are never sent to the
    body model.
  - Read-only historical H1 calibration is `246/256`, with eight
    `body:ValueError` failures and the two original Planner failures.
    Both arms must preserve exact `7+4N` generation on all 254 eligible rows.
  - Local and A800 preflight passed 8/8 focused tests. The real frozen ledger
    passed 254/254 safe-axis invariants and exact-length coverage, with
    treatment active on all 254 eligible rows.
  - Sole Slurm job `29862` completed `0:0` in 34m24s; run root
    `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis256_v1`;
    source manifest SHA
    `6b0dd8298c9a423712a3965b00f8aeae9c06c824a37fbf1d94dbbf94aebcf15f`;
    submission record SHA
    `3b7dd69e12c3e64bdc91578d88687eeea9202fcdd2fdcb0c27b36d3f06e6da57`.
  - Result: gate passed with decision `safe_axis256_confirmed`. D1 exactly
    reproduced the frozen H1 completion and failure taxonomy at `246/256`;
    safe-axis achieved `248/256`, with six `body:ValueError` failures and the
    same two original Planner failures. Duplicate-coordinate failures were
    zero in both arms and no candidate failure class was added.
  - Paired completion was 243 both-success, five candidate-only, three
    control-only, and five both-failed. The candidate gain is 2/256
    (`+0.78125` point), with exact McNemar `p=0.7265625`; this confirms safety
    and a small point improvement, not a statistically established gain.
  - Paired identity mismatch was zero, the shared eligible-row partition
    passed, treatment applied to all 254 eligible rows, and all exact
    `7+4N` and safe-axis invariants passed. Terminal SHA
    `fa03cfa22d0765311bd55e350d1547b6180cede410d8c12054e45306279f002c`;
    generation report SHA
    `f86790015fd090f5243e8785aabf7b0c9ee8ddd8c85dbf02fb5dafd0acb4b880`;
    invariant report SHA
    `560155678cad0ce00737b0052301d3be46da0ace18d00f871d83764477b61d3d`.
  - This stage does not run refinement, direct metrics, S.U.N., training,
    checkpoint selection, promotion, or any automatic downstream.
- [!] `V3-R03F` Complete missing MP hull references and recompute all R03E
  S.U.N. endpoints.
  - Terminal engineering stop / no retry. All 107 deduplicated chemical-system
    queries resolved with zero transport retries and one common 227-system
    snapshot was written, but the first-arm completed-hull hard gate failed at
    `r0_control`. No terminal report, `_SUCCESS`, or completed-cache S.U.N.
    vectors were published.
  - Authorized only after all four R03E repeats and assembly reach
    `COMPLETED 0:0`.
  - Treat the completed R03E generation, exact-800 refinement, direct
    metrics, relaxation energies, novelty/uniqueness mapping, and 256-attempt
    ordinals as immutable inputs. Do not rerun generation, refinement,
    CHGNet, direct metrics, novelty, filtering, repair, or sample selection.
  - Deduplicate the union of missing chemical systems across all eight
    `repeat × arm` evaluations and create one common Materials Project thermo
    snapshot. Query every subsystem required by
    `get_entries_in_chemsys` semantics, apply the registered compatible-entry
    processing, and then freeze the completed snapshot before recomputation.
  - Existing finite `E_hull` values must reproduce exactly from the completed
    snapshot. Recompute strict (`<=0.0 eV/atom`) and meta
    (`<=0.1 eV/atom`) S.U.N. for every original attempt in every arm/repeat,
    retaining the raw all-attempt denominator of 256.
  - The terminal completion gate requires zero missing `E_hull` among all
    applicable novel-unique relaxed structures. A residual unknown, source
    parity mismatch, API/authentication failure, or input-SHA mismatch stops
    the stage without changing the R03E report.
  - API credentials are runtime-only secrets carried in a user-owned
    mode-0600 temporary file, read once and unlinked immediately; they must
    never appear in source, environment dumps, logs, manifests, or reports.
  - Write only a new sidecar run. Report every repeat, paired McNemar,
    hierarchical paired bootstrap, effect-sign stability, and a direct
    frozen-cache-versus-completed-cache coverage delta. No training,
    checkpoint reselection, promotion, formal G3/G4, or further automatic
    downstream is authorized.
- [x] `V3-R03G` Recompute all R03E S.U.N. endpoints without the
  completed-hull zero-unknown gate.
  - Scope was evaluation-only and changed exactly one rule from R03F:
    `completed_hull_zero_unknown_gate_enabled=false`. The clean R03E
    generation/refine800/direct/novelty evidence and the failed R03F common
    227-system snapshot were reused read-only; network and API-key access were
    both false and no new MP query was issued.
  - Result: terminal status
    `complete_without_completed_hull_zero_unknown_gate`; all eight arms
    retained 256 raw ordinals. Of 825 source unknowns, 753 were resolved and
    72 remained, exactly nine in each arm. Every residual unknown was scored
    strict/meta false. All 974 existing finite values reproduced within
    `2.85e-14 eV/atom`, giving 100% registered parity.
  - Pooled candidate minus control was strict `117-99=+18/1024`
    (`+1.7578` points; McNemar `p=0.0198343`; hierarchical paired bootstrap
    95% CI `[0.0000,+3.6133]` points; 4/4 repeat deltas positive) and meta
    `496-523=-27/1024` (`-2.6367` points; McNemar `p=0.0464644`;
    hierarchical 95% CI `[-5.5664,+0.3906]` points; three repeat deltas
    negative and one zero).
  - Relative to frozen-cache R03E, the strict effect changed from `+4` to
    `+18` counts while the meta effect changed only from `-28` to `-27`.
    Because 72 hulls remain unknown and meta remains adverse, these are
    lower-bound report-only estimates. The R03E scientific stop, no-promotion
    decision, and all downstream blocks remain unchanged.
  - Evidence: source manifest
    `06fa4bcd2fc4e340906c9e2396af2c6cbe7f7fb9178156d4112eba96201f4858`;
    common snapshot
    `56f91774c798854d253c0726773593c415456a8b5361f31802c44d8e1bbad917`;
    terminal
    `3cb705ea9b572f37dc46696a8aa59a8ae90a0ec90db4789d341052bab7d3bfff`;
    decision
    `0f9b33f53dcf066b9c77f92fc6adcd799bc18b2df6bce431c89fb6fe453dfa6b`.
- [x] `V3-R03H` Attribute the frozen R03G strict gain and meta loss by paired
  repeat and ordinal.
  - This was a read-only report sidecar over all four repeats, both arms, and
    all 256 raw ordinals. It did not recompute S.U.N., change either frozen
    threshold, select samples, query MP, read an API key, use Slurm/GPU, or
    rerun generation, refinement, CHGNet, direct metrics, or novelty.
  - The 72 residual unknowns were exactly balanced: every repeat had the same
    nine unknown ordinals `11,24,44,55,60,131,170,189,217` in both arms.
    Their nine chemical systems had identical arm frequencies and all
    contained Yb. There were 36 both-unknown paired records and zero one-arm
    unknown records, so residual-unknown contribution to both treatment
    effects was exactly zero.
  - Strict candidate minus control `+18/1024` decomposed into `+16` from
    finite-hull threshold crossings and `+2` from novel-unique eligibility.
    Candidate-only/control-only strict discordances were `36/18`; all 117
    candidate and all 99 control strict-positive values were exactly
    `E_hull=0.0`, with no negative-value artifact.
  - Meta candidate minus control `-27/1024` decomposed entirely into finite
    `0.1 eV/atom` threshold crossings: `60` candidate gains versus `87`
    candidate losses. Novel-unique eligibility contributed `12-12=0`, and
    residual unknowns contributed zero.
  - The marginal frozen state shift was strict `+18`, meta-only `-45`,
    above-meta `+36`, ineligible `-9`, and residual-unknown `0`. Thus safe-axis
    polarized the finite novel-unique stability distribution: it produced
    more exact-zero outcomes but also more outcomes above `0.1 eV/atom`,
    rather than improving broad stability.
  - Of 147 finite meta-discordant pairs, 89 had both arms more than
    `0.01 eV/atom` from the meta threshold; the adverse meta result is not
    explained only by tiny boundary jitter. The recurrent 3/4-or-more meta
    losses were ordinals `4,98,165,223,233`; recurrent meta gains were
    `17,166,193`. Strict gains recurred at ordinal `156` in 4/4 repeats and
    `157` in 3/4.
  - Source manifest
    `770b4b1407db1386d3db6131b3bac43259db601ea6729c3878f36d50428fe151`;
    attribution report
    `566dd3c59cbfaa04243c923f42ef2f726d50a1441fd1e7f5fd2796ff847b42be`;
    decision
    `644e8b649bc39a8580510a0f1444c3106731ec423e627ca2c252fc8553e014be`.
    R03G terminal/decision/input SHAs remained unchanged. Formal G3,
    promotion, training, checkpoint reselection, and automatic downstream
    remain false.
- [!] `V3-R04` Cross body checkpoint and schedule only after R03 passes.
  - Blocked: original R03/D2 failed. Crossing any schedule with a body
    checkpoint remains forbidden unless the separately registered R03B
    schedule passes first.
- [ ] `V3-R05` Reopen a Planner candidate only after an independent Plan-only
  screen passes; keep B0 and the H1 body/refiner path fixed.
- [ ] `V3-R06` Combine independently passing Planner and body factors.

The governing protocol is `H1_SINGLE_VARIABLE_RECOVERY_PLAN_V1.md`.

## Phase 3 — Factorial engineering integration

Formal Phase 3 is blocked by the two Phase-2 scientific stops. The diagnostic
run below does not satisfy, replace, or reopen these gates.

- [ ] `V3-300` Freeze one immutable prompt/seed/ordinal ledger.
  - Runtime readiness: the strict ledger loader, seed derivation, and
    fail-closed four-arm identity checks are complete; the actual
    execution-specific 32-ordinal ledger is intentionally not frozen before
    the P* and B* checkpoints exist.
- [ ] `V3-301` Run M00/M10/M01/M11 for 32 engineering attempts each.
- [ ] `V3-302` Verify exact Plan SHA pairing within each Planner pair.
- [ ] `V3-303` Verify body/refiner noise pairing and ordered merge.
- [ ] `V3-304` Verify earliest-failure accounting and 32/32 raw denominators.
- [ ] `V3-305` Freeze G3 source, execution, authorization, and submission
  manifests.

This phase cannot select a scientific arm. An engineering failure may be
repaired with a new version and an explicit repair log.

## Phase 4 — G3 paired-256 factorial screen

Formal G3 remains blocked and unauthorized. Do not relabel the post-stop
diagnostic as G3 and do not select a checkpoint from its generated-crystal or
S.U.N. results.

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

## Post-stop four-arm refined S.U.N. diagnostic — not G3

- [x] `V3-DIAG-001` Freeze diagnostic-only authorization and decision firewall.
  - Result: M00=P0+B0, M10=P*+B0, M01=P0+B2, M11=P*+B2;
    256 raw all-attempt ordinals per arm; formal G3, checkpoint reselection,
    promotion, training feedback, and automatic downstream are false.
- [x] `V3-DIAG-002` Preserve the failed v1 submission and repair only its
  engineering implementation.
  - Result: prepare job 29495 failed before generation because the isolated
    runtime omitted `crystal_dlm/__init__.py`; dependent array 29496 was
    cancelled without running. No partial scientific output is reused.
  - Repair: v2 added the package marker, fixed the Planner terminal-field
    assertion, and recorded the exact failure in
    `repair_log/001_v1_prepare_runtime_package_and_slurm_packing.json`.
- [x] `V3-DIAG-003` Preserve the failed packed v2 execution.
  - Result: all four array elements stopped during preparation because P*
    ordinal 30 contained one additional model-emitted line. No body,
    refinement, direct metric, or S.U.N. result was produced or reused.
  - Decision: per explicit user authorization, exact seven-line raw
    conformance is now an advisory label. The raw text remains immutable and
    the already accepted frozen canonical Plan continues downstream.
- [x] `V3-DIAG-004` Validate the advisory-format repair before submission.
  - Result: v3 exposed isolated-runtime import ordering during the
    pre-submission test and was never submitted. v4 fixes only that import
    order. Focused A800 tests passed 8/8 and the real preparation integration
    produced 253 continuing plus 3 original Planner failures for both P0 and
    P*. P* ordinal 30 continues with
    `raw_plan_contract_conforming=false` and its raw text preserved.
- [x] `V3-DIAG-005` Preserve the failed v4 packaging execution.
  - Result: 29520_[0-3] and afterany assembly 29521 failed before body
    generation. `runtime/scripts/sample_llada_dynamic_crystals.py` was present
    but the bundled `scripts` directory lacked `__init__.py`, so an unrelated
    installed regular package shadowed it. Only per-arm Plan preparation
    completed; no body, refinement, direct metric, or S.U.N. output is reused.
  - Repair: v5 adds only the isolated `runtime/scripts/__init__.py` marker.
    The source manifest passed, 8/8 focused tests passed, and the actual body
    runner `--help` import completed in the A800 environment.
- [x] `V3-DIAG-006` Preserve the failed v5 source-inventory execution.
  - Result: 29531_[0-3] and afterany assembly 29532 failed before Plan or
    model loading. The pre-submission import checks created ordinary Python
    `__pycache__/*.pyc` files, and the exact source file-set validator treated
    those interpreter caches as unregistered source files.
  - No Plan/body/refinement/direct/S.U.N. `_SUCCESS` exists and no partial
    output is reused.
  - Repair: v6 ignores only Python bytecode cache paths in the source
    file-set comparison, keeps exact SHA checking for every registered source
    file, and runs with `PYTHONDONTWRITEBYTECODE=1`. Focused tests passed 9/9,
    the real body-runner import passed, and the clean source manifest passed
    again on A800.
- [x] `V3-DIAG-007` Preserve the failed v6 tokenizer-identity execution.
  - Result: 29540_[0-3] and afterany assembly 29541 failed after immutable
    Plan preparation but before body generation. The gate expected a
    vocabulary SHA produced by a standalone preflight reconstruction rather
    than the tokenizer serialized with the frozen B0/B2 checkpoints.
  - Audit: R5-C B0, B1, and B2 have byte-identical `tokenizer.json` and
    `tokenizer_config.json`, the same 128,830-entry vocabulary, and identical
    IDs for all 2,481 frozen data tokens; no token is missing.
  - Repair: v7 retains the exact vocabulary gate with the audited checkpoint
    vocabulary SHA and adds exact checkpoint tokenizer-file SHA checks. It
    does not change a tokenizer, token ID, model, input, or output.
- [x] `V3-DIAG-008` Run and preserve the terminal packed v7 four-arm array.
  - Execution: 29549 (`0-3%2`), one A800 / 8 CPUs / 96 GiB /
    36 hours per element.
  - Each element owns its immutable input ledger and runs, in order:
    preparation → body generation → frozen `model_494` diffusion refinement
    for 800 reverse steps → generation finalization → direct metrics →
    frozen-cache S.U.N. → arm validation.
  - Unrefined successful output is prohibited from evaluation.
  - Result: all four arms completed body generation, exact refine800, and
    direct metrics, then failed `1:0` only when importing the incomplete S.U.N.
    runtime. The failure is engineering-only; all v7 S.U.N. outputs are
    unusable, while the exact hashed generation/refine/direct evidence is
    immutable and eligible for an evaluation-only repair.
- [x] `V3-DIAG-008A` Close the future S.U.N. runtime-packaging test gap.
  - Root cause: v7 copied the top-level S.U.N. runner but omitted five modules
    in its transitive runtime dependency closure. The imports occurred inside
    `main()` after direct metrics, so import/compile preflight never exercised
    them.
  - Prevention: the canonical staging tool now copies the exact dependency
    closure, the runner imports those dependencies at module load, and an
    isolated origin audit verifies that every dependency resolves under the
    frozen root before expensive work.
  - Verification: 3/3 new positive/negative isolation tests and 12/12 existing
    S.U.N. adapter/pipeline tests passed locally. This change is for future
    bundles only and does not mutate or rescue v7.
- [x] `V3-DIAG-009` Preserve the failed v7 diagnostic terminal report.
  - Execution: 29550 with `afterany:29549`.
  - Result: terminal status `failed`, decision
    `diagnostic_execution_failed_no_reselection`, SHA
    `b249c982b4ede6b7fbfee20f9ab49a910e31f66a87f596f5b4b80a28ffae6e9f`;
    formal G3, promotion, and automatic downstream all remained false.
- [x] `V3-DIAG-010` Run the isolated evaluation-only v8 S.U.N. repair.
  - Frozen input: exact v7 M00/M10/M01/M11 `generation.jsonl`,
    `generation_report.json`, `_SUCCESS`, direct attempt metrics, and direct
    reports are pinned by SHA. No generation, diffusion refinement, or direct
    metric is rerun.
  - Gate: 15 focused local tests passed; the A800 real-input preflight verified
    all four 256-attempt mappings and every isolated runtime import origin.
  - Execution: packed S.U.N. array 29617 (`0-3%2`, one A800 / 8 CPUs /
    96 GiB / 36 hours, max concurrency 2) followed by afterany assembly 29618.
    Source manifest SHA is
    `6238e7f0df9259b47d4f7641c735d21e43ef1273ee29bcf2925a26a7405b8599`;
    submission record SHA is
    `0b2bf5965c2cb6876e5c5ad873c0c851c27721e91c9036784fa670cb25c845c3`.
  - Decision firewall: diagnostic only; MP API, checkpoint reselection,
    formal G3, promotion, training feedback, and automatic downstream remain
    false.
  - Result: all four array elements and assembly 29618 completed `0:0`.
    Generation/refinement and direct-metric rerun flags are false for every
    arm; all successful v7 inputs retain exact `model_494` refine800 evidence.
  - Raw all-attempt direct counts
    `(generation/composition/structure/joint)`:
    M00 `243/203/241/201`, M10 `242/204/242/204`,
    M01 `72/62/72/62`, and M11 `92/77/92/77`.
  - Raw all-attempt strict/meta S.U.N. counts:
    M00 `13/58` (`5.08%/22.66%`),
    M10 `8/59` (`3.13%/23.05%`),
    M01 `7/20` (`2.73%/7.81%`), and
    M11 `4/24` (`1.56%/9.38%`).
  - Interpretation: P* at B0 changed meta S.U.N. by only `+0.39` point and
    strict S.U.N. by `-1.95` points. At B2 it raised joint validity by
    `+5.86` points but did not improve S.U.N. significantly. B2 versus B0
    reduced joint validity by `-54.30` points at P0 and `-49.61` points at P*,
    so the body factor is the dominant failure.
  - Terminal: status `complete`, decision
    `diagnostic_only_retain_phase2_scientific_stops`, Phase-2 Planner/body
    gates false, formal G3/promotion/automatic downstream false; report SHA
    `6d2e26d263669e207c2ddbecfd3ee78bd66eb473527a466d958a6fcd906dabb6`.

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

## Parallel Plan NP — N-conditioned Planner-free Body-DLM

This is an independently registered planning track beside the CR-Plan track.
It is not part of the old V3 factorial and does not reopen B1/B2 or R03. Its
governing document is
`H1_NOPLAN_NCONDITIONED_PARALLEL_PLAN_V1.md`.

- [x] `NP-000` Record the scientific claim, total-treatment estimand, and
  portfolio amendment.
  - Result: external-Planner-free but explicitly `N`-conditioned; exact
    `7+4N`; no fully-unconditional claim.
- [x] `NP-001` Freeze the conceptual controls and evaluation views.
  - Result: same-update `R_cont` versus `N_only`; paired `N_from_P0`
    Plan-information ablation is separated from operational
    `N_train_prior`.
- [x] `NP-002` Record the leakage, corruption, support, and denominator
  firewalls.
  - Result: `N_only` immutable-visible `{N}`, `R_cont`
    immutable-visible `{N,E_i}`; shared semantic streams only on overlapping
    lattice/coordinate groups; inference hard masks are distinct from the
    frozen full-vocabulary SFT CE; composition is scored on raw all-attempts.
- [ ] `NP-010` Produce one global portfolio/execution annex before R0.
  - Required: exact source/model/data/tokenizer identities, new global and
    NPDLM GPU caps, resource priority, paper-allocation rule, numeric gates,
    primary genuine-chemistry endpoint, multiplicity, checkpoint rule, and
    fixed independent `4×256` confirmation design.
  - Current authorization: `0 GPUh`; documentation only.
- [ ] `NP-020` Gate −1 CPU/read-only implementation and adversarial leakage
  audit.
- [ ] `NP-030` R0 bounded 32-attempt engineering screen.
- [ ] `NP-040` R1 64-attempt mechanism screen.
- [ ] `NP-050` R2 256-attempt common-refiner/common-evaluator preliminary
  screen.
- [ ] `NP-060` R3 four truly independent paired-256 confirmation panels.
- [ ] `NP-070` Optional `C_only` explanation study.
  - Status: post-hoc exploratory under V1; it cannot confirm the initial
    two-arm claim without a new preregistration and independent ledger.
- [ ] `NP-080` Optional charge-reachability and element-stage RL follow-ons.
  - Blocked until pure controlled SFT passes its frozen no-collapse gate.
  - Chemistry masks and RL remain separate factors and require new
    authorizations.

No NP task beyond documentation starts automatically.

## Job and result ledger

| Stage | Run root | Job IDs | State | Result/report | Notes |
|---|---|---|---|---|---|
| V3 local preparation | local workspace | none | IN PROGRESS | 36 targeted Planner/data tests run: 34 passed and 2 registered Torch-only tests skipped locally; `LOCAL_VALIDATION_V3_H1A2_TWO_FACTOR.json` | No V3 Slurm submission |
| P* data/tokenizer preflight | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_pstar_data_preflight_v1` | 29318 | COMPLETE | `COMPLETED 0:0`; full data `4c2ba669…`; smoke data `25355701…`; full preflight `6f6802ff…`; smoke preflight `96da13af…` | 3,200/256 and 32/32 passed; max sequence 383/768; CPU-only |
| Planner smoke | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_pstar_smoke32_v1` | 29322_[0-1]%1 | COMPLETE | both arms `COMPLETED 0:0`, 32 microbatches / 4 updates, finite improving losses; P-control report `6ea059cd…`, P* report `12a9b87f…` | One A800 each, concurrency one; engineering only; no downstream triggered |
| DLM sidecar/tokenizer preflight | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_sidecar_preflight_v1` | 29331 | COMPLETE | `COMPLETED 0:0`; terminal `75e9297a…`; D1 `d49f32ad…`; D2 `2a275f2f…` | 45,229/45,229 sidecar rows; D1/D2 each 36,183/36,183; prompt/answer byte identity; max observed 333/325 under 382; CPU-only, no model/downstream |
| DLM two-GPU smoke | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_v1`; repair `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_assembly_repair_v2` | 29337_[0-1]%1; failed assembly 29338; repair 29345 | COMPLETE | B1/B2 and repair `COMPLETED 0:0`; corrected terminal `5364d6b5…`, gate true; failed terminal `894d4a70…` preserved | No GPU rerun; exact 2-rank/32-row DDP contract passed; LR 5e-5 not promoted |
| Factorial runtime staging | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_factorial_runtime_staging_v5` | none | COMPLETE | archive `62328f3a…`; local/outer/A800 SHA match; 11/11 key hashes; runner imports passed; local and A800 tests 50/50 | Validation only; no model load, GPU, Slurm, generation, S.U.N., or downstream |
| Planner 512 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_planner512_v1` | 29452_[0-2]%1 → 29454 | SCIENTIFIC STOP | all jobs `COMPLETED 0:0`; terminal `09f66256…`; P0/P-control/P* comp-valid `434/456/442` of 512; retain P0 | P* gain vs P0 +1.5625 points, below +2; P* below P-control and all-metal gate failed |
| Planner scientific epoch | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_pcontrol_pstar_scientific400_v1` | 29391_[0-1]%1 → 29392 | COMPLETE | all jobs `COMPLETED 0:0`; terminal `ecaf779e…`; P-control/P* step400 NLL `0.291393/0.292366` | Both downstream endpoints are the complete 400-update epoch |
| DLM one-epoch training | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_b1_b2_scientific1epoch_lr5e5_v1` | 29393_[0-1]%1 → 29394 | COMPLETE | all jobs `COMPLETED 0:0`; terminal `ea22ced9…`; B1/B2 terminal NLL `1.460710/1.466090` | Both endpoints are step1696; no LR sweep, early stopping, or generation selection |
| Paired dependency margin | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_dependency_margin_v1` | 29456_[0-1]%1 → 29457 | SCIENTIFIC STOP | all jobs `COMPLETED 0:0`; B1/B2 margins `0.259809/0.233067`; delta `-0.026741`; retain B0 | Fixed validation rows 0–99; likelihood only; no shuffle/generation/S.U.N. |
| Refined S.U.N. diagnostic v1 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v1` | 29495 → 29496 | ENGINEERING FAILED | prepare `FAILED 1:0` on missing isolated package marker; dependent array cancelled before running | Evidence retained; no partial output reused |
| Refined S.U.N. diagnostic v2 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v2` | 29505_[0-3]%2 → 29506 | ENGINEERING FAILED | strict raw-format check rejected P* ordinal 30 before body generation; no scientific output reused | Failure evidence and repair log retained |
| Refined S.U.N. diagnostic v3 | not submitted | none | PRE-SUBMISSION FAILED | isolated-runtime import-order test failed; no Slurm job or scientific output | Fixed in v4 without changing model/data/seed/denominator |
| Refined S.U.N. diagnostic v4 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v4` | 29520_[0-3]%2 → 29521 | ENGINEERING FAILED | bundled `runtime/scripts` namespace was shadowed before body generation; all elements and assembly failed `1:0` | Only Plan preparation succeeded; no scientific output reused |
| Refined S.U.N. diagnostic v5 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v5` | 29531_[0-3]%2 → 29532 | ENGINEERING FAILED | all elements and assembly `FAILED 1:0` before Plan/model loading because interpreter bytecode caches appeared as unregistered source files | No scientific output reused; exact source checking remains, with only bytecode cache paths excluded in v6 |
| Refined S.U.N. diagnostic v6 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v6` | 29540_[0-3]%2 → 29541 | ENGINEERING FAILED | all elements and assembly `FAILED 1:0`; Plan preparation completed, then the preflight-reconstructed tokenizer SHA mismatched the actual frozen checkpoint tokenizer | No body/refine/S.U.N. output reused; B0/B1/B2 tokenizer files and all data-token IDs audit identical |
| Refined S.U.N. diagnostic v7 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_v3_poststop_sun256_diagnostic_v7` | 29549_[0-3]%2 → 29550 | TERMINAL ENGINEERING FAILURE | all four arms completed generation, exact refine800, and direct metrics, then failed only at delayed S.U.N. imports; no v7 S.U.N. output used | Immutable v7 omitted the S.U.N. dependency closure; exact hashed generation/refine/direct evidence was reused read-only by v8 |
| S.U.N. evaluation repair v8 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1a2_v3_poststop_sun256_evaluation_repair_v8` | 29617_[0-3]%2 → 29618 | COMPLETE / DIAGNOSTIC STOP | all jobs `COMPLETED 0:0`; terminal `6d2e26d2…`; strict/meta M00 `13/58`, M10 `8/59`, M01 `7/20`, M11 `4/24` of 256 | Frozen-cache evaluation only; no generation/refine/direct rerun, MP API, reselection, promotion, or downstream |
| H1 exact P0 replay | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_exact_replay_p0_v1` | 29646 | RETAINED ENGINEERING STOP | `FAILED 1:0`; body SHA exact `d1970e14…`; 246/246 proposal graphs tensor-exact; generation byte mismatch 246/256 | Same node99/A800 and exact800; continuous CUDA scatter differences amplified; no direct metrics, S.U.N., training, or downstream |
| H1 real-ledger Plan conversion preflight | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_real_ledger_plan_preflight_v1` | 29658 | COMPLETE | `COMPLETED 0:0`; gate true; report `270443a0…`; P0/P-star `253/253` parser-accepted attempts compiled with zero conversion errors | CPU-only; no model, GPU, generation, refinement, metric, or downstream action |
| H1 body schedule32 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_schedule32_v1` | 29669 | SCIENTIFIC STOP | `COMPLETED 0:0`; terminal `898191bf…`; D1 `31/32`, D2 `14/32`; D2 added 18 duplicate-coordinate failures; retain D1 | Paired identity and shared batch partition passed; no refine, direct metric, S.U.N., training, or downstream |
| H1 body safe-axis32 R03B | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis32_v1` | 29837 | COMPLETE / GATE PASS | `COMPLETED 0:0`; D1 `31/32`, safe-axis `31/32`; duplicate-coordinate `0/0`; paired mismatch 0; terminal `57d53922…` | Exact-length and all 32 schedule invariants passed; no refine, direct metric, S.U.N., training, promotion, or downstream |
| H1 body safe-axis64 R03C | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis64_v1` | 29844 | COMPLETE / GATE PASS | `COMPLETED 0:0`; D1 `61/64`, safe-axis `63/64`; duplicate-coordinate `0/0`; paired mismatch 0; terminal `34eb0c18…` | Exact-length and all 64 schedule invariants passed; no refine, direct metric, S.U.N., training, promotion, or downstream |
| H1 body safe-axis256 R03D | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis256_v1` | 29862 | COMPLETE / GATE PASS | `COMPLETED 0:0`; D1 `246/256`, safe-axis `248/256`; duplicate-coordinate `0/0`; paired mismatch 0; exact McNemar `p=0.7266`; terminal `fa03cfa2…` | 254/254 exact-length and schedule invariants passed; original Planner failures 86/211 retained; no refine, direct metric, S.U.N., training, promotion, or downstream |
| H1 safe-axis refined repeats R03E | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis_refined_repeats4_v1` | 29912_[0-3]%2 → 29916 | COMPLETE / SCIENTIFIC STOP | all jobs `COMPLETED 0:0`; terminal `7fa49a6f…`; pooled candidate-control joint `+5/1024`, strict `+4/1024`, meta `-28/1024`; decision `safe_axis_refined_signal_stopped` | Four fixed repeats used the same `model_494` refine800 and frozen-cache S.U.N.; hierarchical meta 95% CI `[-4.88,-0.59]` points; no promotion/downstream |
| H1 R03E MP-complete S.U.N. R03F | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260803_h1_body_safeaxis_refined_repeats4_mpcomplete_v1` | A800 login-node evaluation-only sidecar | TERMINAL ENGINEERING STOP / NO RETRY | all 107/107 deduplicated MP chemsys queries resolved with zero transport retries and a common 227-system snapshot was written; execution then failed closed at `r0_control completed hull hard gate failed`, with no terminal report, `_SUCCESS`, or completed-cache S.U.N. vectors | Source `c26e141d…`; failure `4380b0f2…`; query progress `fa40676f…`; common snapshot `56f91774…`; no generation/refine/CHGNet/direct/novelty rerun or automatic downstream |
| H1 R03E no-completed-hull S.U.N. R03G | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260803_h1_body_safeaxis_refined_repeats4_no_completed_hull_v1` | A800 login-node offline evaluation-only sidecar | COMPLETE / REPORT-ONLY | terminal `3cb705ea…`; 753/825 unknowns resolved, 72 remain (9/arm) and score false; finite parity 974/974; candidate-control strict `+18/1024`, meta `-27/1024` | Source `06fa4bcd…`; decision `0f9b33f5…`; shared snapshot `56f91774…`; network/API/upstream rerun/Slurm/GPU/downstream all false |
| H1 R03G paired ordinal attribution R03H | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260803_h1_body_safeaxis_refined_repeats4_attribution_v1` | A800 login-node read-only report sidecar | COMPLETE / ATTRIBUTION-ONLY | report `566dd3c5…`; residual unknown effect `0/0`; strict `+18 = +16` finite crossings `+2` eligibility; meta `-27 = -27` finite crossings `+0` eligibility | Source `770b4b14…`; decision `644e8b64…`; no endpoint/threshold/sample/upstream/network/API/Slurm/GPU/downstream change |
| H1 CR-Plan R0 CR-0 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260804_h1_crplan_r0_paired32_v1` | 30325 | ENGINEERING TIMEOUT / PAIRED-32 NOT SUBMITTED | top-level `TIMEOUT`, batch `CANCELLED 0:15` at 40m12s; 26/26 unit tests passed in 0.397s; tokenizer/dependency audit produced no terminal report or `_SUCCESS`; observation `39b7aa27…` | Source `f94ecb8f…`; submission `432db4bf…`; max RSS 359292K, max disk read 273.06M, total CPU 1.44s; no retry, GPU, promotion, or downstream |
| H1 CR-Plan R0 paired-32 V5 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260804_h1_crplan_r0_paired32_script_package_repair_v5` | 30356 → 30358 | COMPLETE / ENGINEERING GATE PASS / PREFIX CLAIM HOLD | CR-0 and paired-32 `COMPLETED 0:0`; paired terminal `a1a34a66…`; control/candidate parse `31/32→32/32`, composition-valid `17/32→18/32`, primary `9/32→10/32`; one candidate-only paired flip, McNemar `p=1.0`; all registered paired-32 gates true | Source `7ff37f45…`; archive `e697118d…`; historical/seed/prompt/config mismatch 0 and terminal charge failure 0. Full-prefix attribution is held because eight table-missing elements make the conservative prefix contract degenerate; latency `112.813/215.895s` and max DP states `1,714,193` require reviewed policy plus pre-512 optimization |
| H1 CR-Plan exact-tokenizer policy preflight V2 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260804_h1_crplan_fourarm512_exact_tokenizer_preflight_repair_v2` | A800 login-node CPU-only audit | COMPLETE / GATE PASS | status `pass`, failures `[]`, `_SUCCESS`; all 128,256 decoded tokens have trie/scalar legal-ID, terminal-ID, rejection-count, cursor, and certificate parity; all missing-state and shortcut fixtures align with frozen Direct; maximum trie DP states `76,267` | V1's sole failure was an audit-only `Fe-Pm` all-metal fixture error; V2 preserves policy and adds the correct missing-precedence check. Source `72c66c4c…`; archive `596d7518…`; report `59173a6e…`; trie/scalar aggregate `10.3488/121.9064 s`; no GPU/model/network/generation/downstream |
| H1 CR-Plan optimized-support preflight V3 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260804_h1_crplan_fourarm512_exact_support_optimization_preflight_v3` | A800 login-node CPU-only setup | SEALED RUNTIME FAILURE / AUDIT NOT STARTED | source/archive identity passed, but the launcher selected base Python 3.9 and failed importing `dataclass(slots=True)` before exact audit execution | Source `9409b9ee…`; archive `a425fa47…`; failure `2629c492…`; no model/GPU/network/generation/downstream and no scientific-policy change |
| H1 CR-Plan optimized-support runtime repair V4 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260805_h1_crplan_fourarm512_exact_support_optimization_preflight_runtime_repair_v4` | A800 login-node CPU-only audit | ENGINEERING TERMINAL / STATE BUDGET FAIL | process status `pass` and all fixed/real optimized-scalar, bitset-set, probability, missing/shortcut, and endpoint parity gates passed; 169 real cursor signatures produced 507 mode rows. Fixed max states `72,390`, but real-cursor max `415,689 > 100,000`; `_FAILED` exists and `_SUCCESS` does not | Python3.10 runtime-only repair; local `39/39`, isolated/A800 `35/35`; source `1f6dcd99…`; archive `d4838032…`; audit `3a10b0da…`; terminal `55df7801…`; no performance probe, 512, model/GPU/network/generation/downstream |
| H1 CR-Plan E1 physical-performance probe | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260805_h1_crplan_e1_physical_performance_probe_v1` | 30435 | COMPLETE / EXPLORATORY PHYSICAL PASS / AMENDMENT-ONLY | `COMPLETED 0:0`; 18/arm plus 2 scalar references; median off/terminal/full `2.593/3.806/3.808 s`, full/off `1.468x`; p95 `88.805/10.930/10.911 s`; trace and scalar parity 100%; `14/14` applicable attempts affected preterminally; terminal `b7c8e2bf…` | Source `aab2c113…`; archive `c0766a6b…`; submission `15ac4585…`; max cumulative states terminal/full `1,567,682` are reported but do not reuse/repair V4; V4 terminal remains `55df7801…`; no 512 or downstream |
| H1 CR-Plan four-arm Plan-only 512 route amendment | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260805_h1_crplan_fourarm512_route_amendment_v1` | 30441/30442/30443/30448 → 30444 | TERMINAL ENGINEERING FAILURE + SCIENTIFIC STOP | all four GPU arms `COMPLETED 0:0` with exact 512 raw attempts; assembly wrote terminal `8cef9c2a…` then failed closed `2:0`; terminal-only had 7 generation errors. Full-terminal raw composition `+6/512` (6/0 flips, McNemar `p=0.03125`), but nonshortcut/primary only `+1/512` (`p=1.0`) versus required `+11`, while shortcuts increased `+5` | Source `f0241f91…`; archive `9aa3d32c…`; ledger `0bd468b9…`; full terminal charge failures 0, parse/completion `+7/+7`, diversity/coverage noninferior; mechanism activity recorded but does not rescue failed causal/safeguard gates. Decision `stop_crplan_retain_frozen_h1`; no paired-64/downstream |
| H1 comp_valid SFT-first route | local V2 root-cause/decision document; remote cleanup/audit pending | none | SUPERSEDED BY AUTHORIZED NO-CHARGE C0/C1 EXECUTION | Published CrysLLMGen `93.55%` is treated as strict raw-output comp validity, not a S.U.N survivor metric. Evidence points primarily to checkpoint/recipe non-parity plus H1's compact, causally backward SFT target; MP-20 is physically real but `38.801%` of train rows fail the frozen heuristic evaluator and `35.112%` are shortcuts | `analysis/H1_COMP_VALID_ROOT_CAUSE_AND_SFT_PLAN_V2.md` remains diagnosis; executable design is superseded by the no-charge ion-aux annex below |
| H1 no-charge matched ion-aux SFT C0/C1 | local source + `analysis/H1_NOCHARGE_ION_AUX_SFT_EXECUTION_ANNEX_V1.md`; remote immutable run pending | none | USER AUTHORIZED / SOURCE-REPAIR V2 ISOLATED PASS / REMOTE BLOCKED ON MAINTAINED CONNECTION | Keep formula-first six-line rich Plan but delete generated `charge:`; C0 and C1 share one exact 3,200-record ledger and differ only in 10% matched auxiliary semantics (neutral atom/count versus explicit oxidation witness). Formula/chemistry payload tokens receive 2x loss; every full-MP20 formula, including KL records, is input-only and never an unconditional answer target. P0/C0/C1 raw Planner ladder is 64→256; frozen B0/D1 and `model_494` remain unchanged and downstream is forbidden until Planner gates pass | Legacy SMACT3 snapshot contract `96a4a4af…`; full local val/test legacy primary `2264/2344`, stable SMACT4 uniform-primary `1406/1463`, mixed-only `95/99`, parity failures `0/0`. The first untransferred local archive lacked one test-only helper. Source-repair V2 passed file hashes, isolated `101/101`, preflight and nine shell checks: inventory `354308ae…`, manifest `6c79d202…`, archive `4c71caa6…`; Ledger64 `ea566afe…`, Ledger256 `a0aafedb…`. No WSL distro was running at the final check, so no remote transfer/job occurred. Real train build, tokenizer/dual-adapter smoke and two fixed 400-update A800 arms await restoration of the user-maintained tmux path. No RL, DLM training, reselection, repair/filter/retry or S.U.N.-based tuning |
| G3 paired-256 | blocked | pending | BLOCKED | Phase-2 Planner and DLM scientific gates failed | Do not conflate with the running post-stop diagnostic |
| G4 confirmatory | pending | pending | TODO | pending | Manual authorization required |

## Active next actions

1. Preserve CR-0 job 30325, V5 jobs 30356/30358, the V2/V3/V4 preflight chain,
   E1 job 30435, and the four-arm route-amendment jobs
   30441/30442/30443/30448/30444. Route-amendment terminal `8cef9c2a…`
   establishes the final CR-Plan stop: all four arms reached 512 raw attempts,
   but terminal-only had seven generation errors; full-terminal nonshortcut
   gain was only `+1/512` against the frozen `+11` gate and shortcuts increased
   by five. Retain frozen H1. Do not drop failed attempts, count shortcuts as
   primary gain, reinterpret the affected-rate accounting anomaly, change the
   policy/thresholds, or submit paired-64, paired-256, independent panels, or
   any other downstream repair.
2. Execute the user-authorized minimal-change route only through
   `analysis/H1_NOCHARGE_ION_AUX_SFT_EXECUTION_ANNEX_V1.md`. Protect P0,
   selected P-control, B0, `model_494`, local CrysLLMGen adapters and paper
   parity assets. Freeze the local source/data/tokenizer/SMACT4 audit, then run
   exactly two matched 400-update continuations: C0 neutral auxiliary and C1
   oxidation auxiliary. The generated Plan remains formula-first and deletes
   only `charge:`; the Body-DLM is not trained or changed. Use the fixed step
   400 endpoints and raw paired 64→256 Planner gates; only a passing Planner
   may enter frozen B0/D1 + `model_494` common Direct/CrysLLMGen/S.U.N.
   evaluation. Current remote work is blocked until the user-maintained tmux
   connection is visible and an already-installed exact SMACT4 runtime passes
   the dual-runtime preflight; never create/reconnect it or install a new
   environment automatically.
3. Preserve the clean R03E terminal and its scientific stop; never select or
   replace a repeat by outcome.
4. Preserve the failed R03F evidence and the successful report-only R03G
   sidecar. Do not relabel R03G as completed-hull evidence: its 72 residual
   unknowns are explicit false-scored lower bounds.
5. Preserve the completed R03H attribution. The 72 residual unknowns are
   exactly paired and contribute zero to both effects; do not tune thresholds,
   select ordinals, or relabel the diagnostic as a scientific pass.
6. If safe-axis is revisited, first join every frozen ordinal's schedule
   features to its paired R03H transition state. Use recurrent meta-loss and
   strict-gain ordinals only as descriptive mechanism checks, never as a
   selected evaluation subset.
7. Interpret R03E/R03G/R03H only with
   `H1_R03E_REFINER_REPEAT_PROTOCOL_V1.md`: report every repeat, hierarchical
   paired bootstrap, McNemar, effect-sign stability, and the explicit
   frozen-cache-versus-completed-cache coverage delta.
8. Do not expand original mixed-axis D2 or combine safe-axis with B1/B2;
   B2 remains scientifically stopped and unselected.
9. Keep P* and B2 unselected. Formal G3/G4, checkpoint reselection, promotion,
   training feedback, S.U.N.-based tuning, and automatic downstream remain
   blocked.
10. Scheduler prevention rule: before the first `sbatch`, validate every
   declared partition with `sinfo`; do not assume a generic `cpu` partition.
11. Treat `ICLR2027_TIMEBOXED_FULL_IMPROVEMENT_ROADMAP_V1.md` as the immutable
   pre-amendment portfolio evidence and
   `H1_NOPLAN_NCONDITIONED_PARALLEL_PLAN_V1.md` as the user-directed
   incremental amendment. The amendment supersedes only the single-track
   planning allocation; neither document authorizes execution.
12. The original conditional `P0+CR-Plan` route is terminally stopped. The
    new user-directed comp_valid amendment supersedes the former SFT/RL
    allocation only for cleanup, read-only audit and SFT-first design; P0
    remains the frozen scientific control until a new preregistered comparison
    passes. P-control is a candidate SFT endpoint, not a retrospectively
    promoted anchor.
13. The early CR-Plan engineering-stop condition for `PILS-L` backup
    eligibility is satisfied, but eligibility is not authorization. Do not
    start matched SFT, endpoint work, or GPU use unless a release-ready PILS-L
    execution annex and separate user authorization freeze its compute,
    checkpoints, gates, and paper-allocation rule. CPU/read-only coverage and
    B0 legal-mass Gate −1 remain preparatory only and must not silently become
    an experiment.
14. Body mask-aware RL, two-model RL, compiler, reward labeling and historical
    TraceRL remain cut. The prior reachability-mass-SFT cut is superseded only
    by `analysis/H1_COMP_VALID_SFT_FIRST_ROUTE_V1.md`; it currently authorizes
    design and preflight, not GPU training. Planner RL remains `0 A800 GPUh`
    and may be proposed only after the supervised ladder is falsified.
15. The old 08-15/08-22/08-31 cutlines are retained as pre-amendment portfolio
    evidence, not silently reused for the new SFT route. A new execution annex
    must freeze realistic dates, denominators and compute before any job. A
    missed gate never authorizes a smaller denominator, selected seed, relaxed
    noninferiority margin, or additional factor.
16. NPDLM is a separate pure-SFT planning track, not the cut RL or
    reachability-mass SFT work. Its current allocation is nevertheless
    `0 authorized GPUh`. Before R0, one global portfolio/execution annex must
    freeze its exact numeric gates, primary endpoint, multiplicity,
    checkpoint, resource/global caps, and paper-allocation rule.
17. The first NPDLM treatment keeps the exact interleaved `7+4N` sequence,
    freezes `N`, and generates elements, lattice, and coordinates. It may not
    simultaneously add CR-Plan, safe-axis, PILS-L, chemistry masks, a new
    tokenizer, RL, repair, filter, retry, replacement, or rerank.
18. NPDLM must keep its two interpretations separate: `N_from_P0` is a paired
    Plan-information ablation and not operational Planner-free;
    `N_train_prior` executes no Planner but is a separately labeled
    population comparison. No test target may supply its Plan or operational
    count.
19. CR-Plan and NPDLM use separate source/run/checkpoint/ledger/selection
    identities. Neither track may tune from, consume the ring-fenced budget
    of, combine with, or automatically replace the other. Without a pre-R1
    portfolio amendment, NPDLM remains exploratory/appendix even if its
    preliminary screen is favorable.

Active monitor: none.

Prepared, non-executable drafts:
`execution/v3_scientific_training_drafts_v1/DRAFT_MANIFEST_INDEX.json`.

## Completed result log

### 2026-08-04 — CR-Plan R0 freeze, backup, and CPU CR-0 timeout

- Before changing the Planner path, the last successful H1/R03 implementation,
  execution records, and parameter identities were archived as
  `source_bundles/20260804_h1_success_anchor_pre_crplan_v1.tar.gz`, SHA-256
  `16e18b2ea9a8a781a9f8f2e8919cbb5b179748035c1c884362bfe9fb9348fb80`.
  Large frozen model assets were not duplicated; their absolute paths and
  cryptographic identities are recorded inside the archive.
- The formula-only CR-Plan intervention and paired-32 execution package were
  frozen as `source_bundles/20260804_h1_crplan_r0_paired32_v1.tar.gz`,
  archive SHA-256
  `29f9d4f1ac2de3ffb18c352b33c81e397913233339115541a2d0b45c812ce77f`
  and source-manifest SHA-256
  `f94ecb8fa4c1b38a39cc4882f74dd2077e145b79333d3081dfc1faab8722285d`.
  Local tests passed 26/26, an independent review found no CR-0 blocker, and
  both archives matched across the local→5090→A800 SCP chain.
- Only CPU CR-0 was submitted. Job 30325 passed the same 26/26 unit tests in
  0.397 seconds, but the frozen 40-minute allocation expired while the remote
  tokenizer/dependency audit was still loading from shared storage. Slurm
  recorded top-level `TIMEOUT`, batch `CANCELLED 0:15`, max RSS 359292K,
  max disk read 273.06M, and only 1.44 seconds total CPU.
- No `terminal_report.json` or `_SUCCESS` was produced. The immutable
  observation sidecar has SHA-256
  `39b7aa279f29851ccbfbf704035bc624c6af87abe1e103bd26320727ee07067b`.
  This is an engineering runtime-budget failure, not a chemical-validity or
  scientific CR-Plan result. The paired-32 job was never submitted; there was
  no automatic retry, GPU use, promotion, or downstream action.

### 2026-08-04 — CR-Plan exact-support optimization engineering terminal

- The clean paired-32 V5 result remains an engineering result only:
  control/candidate parse `31→32`, composition-valid `17→18`, primary
  `9→10`, and one candidate-only paired flip. Its old allow-missing policy
  makes it ineligible for a prefix-reachability claim.
- The Direct-aligned fail-closed endpoint policy and exact-tokenizer V2 audit
  passed. The accepted optimized implementation retained the complete legal
  support and online telemetry while adding a combined trie, decision-only
  speculative terminal checks, and exact Boolean charge bitsets.
- V3 was sealed before audit because its launcher selected base Python 3.9.
  The failure report is `2629c492…`; it loaded no model and used no GPU,
  network, generation, or downstream action.
- A single runtime-only V4 repair selected the existing Python 3.10.18
  environment. Scientific code and the audit script were byte-identical to
  V3. Local tests passed `39/39`; isolated and A800 tests passed `35/35`.
- V4 evaluated the exact 128,256-token vocabulary and all `169` unique
  formula-value cursors from paired-32 (`507` mode rows). Optimized/scalar
  legal IDs, terminal IDs, rejection counts, certificates, bitset/set
  reachability, mask→top-k→top-p probabilities, missing/shortcut precedence,
  and terminal/full Direct semantics all matched.
- The release gate nevertheless failed: fixed-fixture semantic DP states
  peaked at `72,390`, while real-cursor states peaked at `415,689`, above the
  preregistered `100,000` limit. The terminal status is
  `engineering_terminal_state_budget_exceeded`; `_FAILED` exists and
  `_SUCCESS` does not.
- V4 identities: source `1f6dcd99…`, archive `d4838032…`, audit report
  `3a10b0da…`, terminal `55df7801…`. No performance probe, four-arm 512,
  paired-64, paired-256, independent panel, model load, GPU job, network call,
  generation, training, reselection, promotion, or downstream action occurred.
- CR-Plan is closed for the frozen ICLR route. This stop cannot be repaired by
  redefining states, selecting easier cursors/seeds, moving work off-clock,
  changing thresholds, shrinking denominators, or introducing another
  implementation family after observing the audit.

### 2026-08-05 — CR-Plan E1 exploratory physical-feasibility terminal

- User-authorized E1 asked a new physical-cost question without modifying V4
  or reusing its `100,000`-state gate. Job 30435 completed `0:0` with terminal
  SHA `b7c8e2bf…`, `_SUCCESS`, and no `_FAILED`.
- The balanced primary panel has 18 attempts in each of off, terminal-only,
  and full-prefix. Median latency is `2.593/3.806/3.808 s`; p95 is
  `88.805/10.930/10.911 s`. Full/off median is `1.468x`, just inside the
  `1.5x` gate. The p95 ratio passes mechanically but is dominated by the
  `88.805 s` off cold-path outlier and must be reported untrimmed.
- Actual-trace support parity passed over 105 unique formula cursors, 36
  diagnostic traces, and 264 sampled legal checks. Both scalar references
  matched token IDs, text, parse, certificates, prompt/input/seed, and
  diagnostics exactly.
- All `14/14` charge-applicable full-prefix attempts had a real preterminal
  support difference. Maximum cumulative semantic states were `1,567,682` for
  terminal/full and are disclosed without changing the V4 conclusion.
- Source `aab2c113…`, archive `c0766a6b…`, submission `15ac4585…`; original V4
  terminal SHA remains `55df7801…`. No Body/refiner/Direct/S.U.N., network,
  training, reselection, promotion, 512, or downstream action occurred.
- Kill-or-go: **GO only for a new preregistered route amendment**. E1 is an
  engineering feasibility result, not a scientific result or automatic 512
  authorization. Until an explicit amendment is frozen, the paper default
  remains the successful frozen H1.

### 2026-08-04 — N-conditioned Planner-free DLM parallel plan

- The user opened a second planning track to proceed beside the existing
  Planner/CR-Plan line. The old roadmap remains immutable historical
  evidence; the new decision supersedes only its single-track planning
  allocation.
- `H1-NPDLM` keeps the frozen interleaved dynamic-v1 answer and exact
  `7+4N` length. `N` remains externally visible/frozen, while all `N` element
  tokens, six lattice/angle tokens, and all coordinates are generated. This
  is external-Planner-free and N-conditioned, not fully unconditional.
- Three independent reviews covered causal controls, chemistry/leakage, and
  special-token/RL mechanics. Their blockers were incorporated: the final
  plan distinguishes serialization from reveal order, same updates from
  equal compute, target suffix labels from conditioning leakage, full-vocab
  SFT CE from inference hard masks, and committed positions from the full RL
  action trace.
- The plan separates `N_from_P0`, a paired Plan-information ablation, from
  `N_train_prior`, an operational no-Planner view. It registers dedicated
  immutable-visible masks, arm-independent semantic RNG streams, raw
  all-attempt composition reconstruction, genuine-chemistry/collapse
  taxonomy, and a fixed Gate −1 → 32 → 64 → 256 → independent `4×256`
  ladder.
- Pure controlled SFT must be evaluated before any chemistry mask or RL.
  CR-Plan remains the only currently allocated headline; NPDLM has
  `0 authorized GPUh` until a global portfolio/execution annex freezes
  resources, exact gates, checkpoint/multiplicity, and paper allocation
  before R1.
- Plan SHA-256:
  `14578ad8f9de7cdcb9c228e5c8b962f1d3f7afa217db0a6d718f554383f25942`.
  Release-manifest SHA-256:
  `02badd2181549cbcc8553fde3fea2fe94bf68c5026db5d61716ee387f6508b93`.
  No training, generation, refinement, query, checkpoint selection,
  promotion, or automatic downstream action was started.

### 2026-08-04 — ICLR 2027 time-boxed improvement portfolio

- Three independent proposals covered Planner chemistry/P-control,
  special-token Body representation, and mask-aware multi-fidelity RL.
  Each proposal was reviewed by a different owner, followed by an independent
  red-team attack and a separate release-consistency audit.
- Every cross-review and the red team selected the same portfolio:
  `P0+CR-Plan` is the only conditional MAIN; `PILS-L` is Gate −1 cold backup
  only; mask-aware RL, two permanent RL models, chemistry compiler,
  reachability-mass SFT, and safe-axis continuation are CUT for this ICLR
  execution queue. RL receives exactly `0 A800 GPUh`.
- The CR-Plan novelty gate is now mechanical: a new four-arm
  original/grammar-only/terminal-only/full-prefix Plan-only panel uses 512 raw
  attempts per arm, predeclared yield/affected-rate/diversity/latency gates,
  fixed mixed-valence/applicability semantics, and no Body/refiner/S.U.N.
  feedback.
- PILS-L never starts automatically. An 08-08 novelty or 08-10 engineering
  stop grants only backup eligibility; actual replacement requires a
  release-ready execution annex and new authorization by 08-10. Otherwise the
  paper falls back to frozen H1.
- Hard dates are 08-15 paired-64, 08-22 paired-256, 08-31 four genuinely
  independent paired-256 panels, and 09-05 science freeze. The CR-Plan target
  and absolute caps are 96/136 A800 GPUh. Preliminary evidence cannot put
  CR-Plan in the title or abstract.
- Historical common Direct/S.U.N. is the scientific hard gate. A missing extra
  robustness evaluator only downgrades broad-stability/SOTA claims; a reverse
  result must be disclosed and kills those claims without rewriting the
  common-evaluator raw comp/joint comparison.
- Final roadmap SHA-256:
  `47e0b9fe1ce1a891edf0df9744e8e8d68cd9fa6225f2339bc49e99608cbf2bc7`.
  The final narrow re-audit found zero blockers, SHA-256
  `c70e5c0a2e2931473a636d5899164f632b4fe4de767a799a8705625037c4300c`.
  External release manifest SHA-256:
  `930bb76612bb5fe1ec295e41a2d438eea1ed9d6635f11f5e915efd6971f52d16`.
- No experiment, training, query, checkpoint selection, promotion, or
  downstream action was started or authorized by this planning portfolio.

### 2026-08-04 — Body-DLM complete protocol and mask-aware RL design

- The current B0/R5-C exact-length Body-DLM was audited end to end rather
  than treated as a generic text diffusion model. Its frozen tokenizer has
  128,830 tokens and adds 2,481 unique crystal special tokens: count,
  lattice/angle, space-group, element, coordinate, and padding/empty
  families. Token IDs and the tokenizer SHA identities are part of the
  scientific contract.
- The active sequence is exactly `7+4N`: one atom-count token, six
  lattice/angle tokens, then `E/X/Y/Z` per atom. Count and element slots are
  Plan-prefilled, so the stochastic Body-DLM action count is `6+3N`, not the
  full sequence and never the full vocabulary.
- The old `scripts/llada_trace_rl.py` is formal-experiment NO-GO. Its
  full-vocabulary token likelihood, reconstructed traces, and omission of
  reveal-position probability cannot represent the actual masked diffusion
  policy.
- The preferred conditional-go design keeps the frozen B0/D1 science
  anchor, adds a fresh RL-only LoRA, and replaces only the policy/trajectory
  layer with D1-grouped `K=1` Plackett–Luce remasking. Each transition stores
  legal-support-renormalized token probabilities and reveal-position
  probability.
- One multifidelity policy is preferred over permanently maintaining two
  models: pre-refiner warm-up, randomized pre/post-refiner mixed training,
  then a short post-refiner-direct finish. Pre-refiner and post-refiner
  rewards remain separate report columns, and meta S.U.N. is a hard
  noninferiority gate.
- The design and staged `R0 → 32 → 64 → 256 → 4×256` gates are recorded in
  `H1_BODY_DLM_COMPLETE_PROTOCOL_AND_RL_DESIGN_V1.md`, SHA-256
  `ff473ac144b060a461d5d060a57f0ad124183faa25563152164753e064fcf376`.
  No source protocol was amended, no model was trained, and no experiment
  was submitted.
- A follow-up read-only token audit found `1,437/2,481` special tokens and
  `1,330/2,343` stochastic-action tokens represented in the frozen local
  9,046-row held-out exact corpus. The `1,013` held-out-unseen action tokens
  include `953` axis-specific length tokens. This is explicitly not claimed
  as full-train coverage because the complete 27,136-row train JSONL is not
  local.
- The coverage audit is recorded in
  `H1_BODY_SPECIAL_TOKEN_COVERAGE_AUDIT_V1.md`, SHA-256
  `070bbce7432b25543e73f4e6c73975eb6cd86ba86a2f042cb496309479352725`.
  It inserts a mandatory Gate −1 before RL and keeps token/support changes
  separate from policy optimization.

### 2026-08-03 — Three-direction Planner/chemistry/DLM-RL feasibility study

- Three independent investigations covered: reopening the frozen P-control
  Planner, bringing chemical constraints into Planner decoding, and training
  two Body-DLM RL adapters with rewards measured before versus after the
  frozen continuous refiner.
- The reconciled decision is recorded in
  `H1_PLANNER_CHEMISTRY_DLM_RL_FEASIBILITY_REPORT_V1.md`, SHA-256
  `57663b4e0fa7c11a16dfb1faddf5764b741afe75100f819a3aaede8930c0b2ca`.
- P-control is a conditional-go confirmation candidate, not a promoted
  baseline. Its old `456/512` composition result is exploratory and requires
  a shortcut/drift audit followed by a fresh 1,024-ordinal confirmation.
- The preferred chemistry mechanism is a formula-line
  charge-reachability product automaton, followed by a separately tested
  formula-derived chemistry-field compiler. Last-count masking and any
  post-hoc repair/filter/retry/rerank are no-go.
- Two pre/post-refiner DLM-RL LoRAs are useful only as a small causal
  diagnostic. A no-training Gate 0 must first prove exact token/position
  action likelihoods and that the pre-refiner reward predicts the
  post-refiner endpoint. Meta S.U.N. is a noninferiority constraint, not an
  optional weighted reward term.
- No source protocol was amended, no model was trained, and no local or
  remote experiment was submitted by this study. Any DLM-RL execution
  requires a new workstream because the current V3 protocol prohibits
  S.U.N., energy, hull, and generated-crystal results in training or
  checkpoint selection.

### 2026-08-03 — R03E terminal and R03F MP-completion fail-close

- R03E completed cleanly and remains immutable. Its terminal SHA is
  `7fa49a6feb372d4d5e5dc442a187657a39001b6190dd6a64af0bf7f53c293b02`;
  pooled candidate minus control over 1024 attempts was `+5` joint-valid,
  `+4` strict S.U.N., and `-28` meta S.U.N. The registered decision remains
  `safe_axis_refined_signal_stopped`.
- R03F used source manifest SHA
  `c26e141d130bd640eefe23b4fcce08fe464eea0f99c52125956b655f51d1a485`.
  It resolved all 107/107 deduplicated MP queries with zero transport retries
  and wrote a shared 227-system snapshot with SHA
  `56f91774c798854d253c0726773593c415456a8b5361f31802c44d8e1bbad917`.
- Evaluation then failed closed at
  `r0_control completed hull hard gate failed`. Failure-report SHA is
  `4380b0f270db959264004da959bc8c4afe78a7e862be79aa1667d31133349020`.
  The run produced no terminal report, `_SUCCESS`, or completed-cache S.U.N.
  vectors, so no completed-cache scientific effect is inferred.
- No generation, refinement, CHGNet/relaxation, direct metric, novelty, or
  sample operation was rerun; there was no retry/replacement/repair/filter/
  rerank, promotion, training feedback, or automatic downstream.
- Most valuable next step, only under separate authorization: a read-only
  audit of the first-arm hard-gate inputs to distinguish residual hull
  unknowns from source-count/parity mismatch before designing any successor.

### 2026-08-03 — R03G no-completed-hull offline terminal

- R03G changed only the R03F zero-unknown completion gate. It reused the
  common 227-system snapshot SHA
  `56f91774c798854d253c0726773593c415456a8b5361f31802c44d8e1bbad917`
  offline with zero new queries, no API-key read, and no network, Slurm, GPU,
  generation, refinement, CHGNet, direct-metric, or novelty rerun.
- All eight arms retained 256 raw ordinals. The 825 original unknowns were
  conserved as 753 resolved plus 72 still unknown; each arm retained exactly
  nine unknowns, and each residual unknown scored strict/meta false. Existing
  finite parity was 974/974; the largest numerical reproduction difference
  was `2.842170943040401e-14 eV/atom`.
- Per-repeat control→candidate strict/meta counts were:
  R0 `27/133 → 28/122`, R1 `22/130 → 31/123`,
  R2 `26/134 → 29/125`, and R3 `24/126 → 29/126`.
- Pooled candidate minus control was strict `+18/1024`
  (`p=0.0198343`, hierarchical 95% CI `[0.0000,+3.6133]` points,
  4/4 positive repeats) and meta `-27/1024`
  (`p=0.0464644`, hierarchical 95% CI `[-5.5664,+0.3906]` points,
  three negative repeats and one zero).
- Frozen-cache R03E effects were strict `+4/1024` and meta `-28/1024`;
  therefore hull coverage strengthened the strict point signal but did not
  repair the adverse meta result. Because 72 hulls remain unknown, R03G is a
  lower-bound report-only result and does not reopen the R03E gate.
- Source manifest SHA is
  `06fa4bcd2fc4e340906c9e2396af2c6cbe7f7fb9178156d4112eba96201f4858`;
  terminal SHA is
  `3cb705ea9b572f37dc46696a8aa59a8ae90a0ec90db4789d341052bab7d3bfff`;
  decision SHA is
  `0f9b33f53dcf066b9c77f92fc6adcd799bc18b2df6bce431c89fb6fe453dfa6b`.
  Formal G3, promotion, training feedback, checkpoint reselection, and
  automatic downstream all remain false.
- Most valuable next step: a read-only paired ordinal attribution of strict
  gains versus meta losses before proposing one further H1-preserving
  single-variable change.

### 2026-08-03 — R03H paired ordinal attribution terminal

- R03H completed as a new read-only report sidecar over four repeats, two
  arms, and every raw ordinal `0..255`. It verified and reused the frozen R03G
  endpoint vectors and R03E relaxation compositions; it did not recompute an
  endpoint, change a threshold, select a sample, query MP, or rerun any
  upstream operation.
- The 72 residual unknowns were exactly paired. Every repeat and arm had the
  same nine unknown ordinals `11,24,44,55,60,131,170,189,217`; there were 36
  both-unknown paired records and no one-arm unknown records. The nine
  chemical systems had identical arm frequencies and shared Yb, so residual
  unknowns contribute exactly zero to both the strict and meta effects.
- Strict `+18/1024` was a genuine exact-zero-hull signal: all 117 candidate
  and all 99 control strict values were exactly `0.0 eV/atom`. Its net
  decomposition was `+16` finite-hull crossings, `+2` novel-unique
  eligibility, and `0` residual unknown.
- Meta `-27/1024` was entirely a finite-hull redistribution: 60 candidate
  gains versus 87 candidate losses across `0.1 eV/atom`, while eligibility
  contributed `12-12=0` and residual unknowns contributed zero. The marginal
  state shift was strict `+18`, meta-only `-45`, above-meta `+36`,
  ineligible `-9`, unknown `0`.
- This is polarization, not broad stability improvement. Of the 147 finite
  meta discordances, 89 had both arms more than `0.01 eV/atom` from the
  threshold, so the loss is not merely numerical boundary jitter. The paired
  finite `E_hull` median delta was zero and mean delta was
  `-0.0020530 eV/atom`; candidate was lower/equal/higher in
  `357/82/400` pairs.
- Recurrent strict gains were ordinal 156 in 4/4 repeats and 157 in 3/4.
  Recurrent 3/4 meta losses were ordinals `4,98,165,223,233`; recurrent 3/4
  meta gains were `17,166,193`. These are descriptive mechanism anchors, not
  authorized sample-selection criteria.
- Source manifest SHA is
  `770b4b1407db1386d3db6131b3bac43259db601ea6729c3878f36d50428fe151`;
  attribution-report SHA is
  `566dd3c59cbfaa04243c923f42ef2f726d50a1441fd1e7f5fd2796ff847b42be`;
  decision SHA is
  `644e8b649bc39a8580510a0f1444c3106731ec423e627ca2c252fc8553e014be`.
  The frozen R03G terminal, decision, and input-contract SHAs remained
  unchanged.
- Scientific decision: the strict signal is real and repeat-stable, but the
  broader meta endpoint remains adverse. R03E stays stopped; formal G3,
  promotion, training, checkpoint reselection, and automatic downstream
  remain false.
- Most valuable next step: join all frozen ordinals to their safe-axis
  schedule features and paired transition states, then preregister exactly
  one H1 schedule change aimed at reducing the recurrent meta-loss mechanism.
  Do not tune S.U.N. thresholds or evaluate only selected ordinals.

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
- At this initial validation point, persisted inference Plans were required
  to be exact seven-line raw model outputs. The later diagnostic amendment
  supersedes raw-format-only blocking: added renderer/model text is labelled
  and preserved, while an already parsed canonical Plan continues. Missing
  semantics, teacher-plan provenance, SHA mismatch,
  retry/replacement/repair/filter/rerank flags, or ordinal/seed mismatch still
  fail closed.
- M00/M01 consume byte-identical persisted P0 Plan/body prompts; M10/M11 do
  the same for P*. Planner/body/refiner seeds are stateless per ordinal.
- The four-arm merger sorts by ordinal and fixed arm order and rejects every
  duplicate, missing, out-of-range, or stray arm identity.
- Focused command:
  `python3 -m unittest tests.test_h1a2_factorial_contract tests.test_ordinal_rng tests.test_h1_llm_planner`.
- Result: 27/27 passed; validation record SHA-256
  `6ba1a1fe32081a28d23f4a5c5096da74962b10e014b7e0500e3d9bf02f4c576a`.
- Scientific training drafts were prepared with zero job IDs and
  `automatic_downstream=false`. At that time, the DLM draft rejected smoke LR
  `5e-5` in favor of `5e-6`; this historical decision was later superseded by
  the explicit user override restoring `5e-5`. No scientific job had yet been
  submitted at this historical checkpoint.
- The self-contained staging archive was copied through the restored
  outer/nested path to
  `/runs/20260801_h1a2_v3_contract_drafts_staging_v2`: archive SHA matched on
  local/outer/A800, 5/5 key-source hashes matched, and the 7 new contract
  tests passed on A800. No model, GPU, or Slurm job was used.
- The earlier minimal staging v1 is retained as transfer/test evidence: its
  file hashes passed, but it omitted imported repository modules. The bounded
  packaging-only repair is recorded in
  `repair_log/V3_CONTRACT_STAGING_BUNDLE_V2_20260801.json`.

### 2026-08-01 — Initial H1-A2 factorial runtime wired and validated

- Added fresh, non-historical runners for model-sampled Planner output,
  B0/B* body generation, the frozen 800-step refiner, and strict four-arm
  all-attempt assembly. The seven-line schema remains the target; its
  raw-format-only blocking behavior was later relaxed without relaxing
  semantic parsing or attempt accounting.
- B* compiles its PlanGraph only from the persisted sampled P0/P* Plan. B0
  uses the historical current-order policy. No structure-derived teacher Plan
  is accepted at inference.
- The body runner enforces tokenizer-vocabulary SHA identity, additive
  prompt/answer tokenization, exact `7 + 4*N` body length, schema/count/
  composition agreement, and one stateless body seed per ordinal.
- Planner, PlanGraph, body, and refiner failures remain in the denominator;
  rank merges and the final four-arm merge fail closed on missing, duplicate,
  stray, or reordered identities.
- Local targeted suite: 50/50 passed. A800 validation: archive SHA matched
  local/outer/A800, 11/11 key hashes passed, all four runners compiled and
  imported, and 50/50 targeted tests passed.
- Final validation archive:
  `/runs/20260801_h1a2_v3_factorial_runtime_staging_v5`, SHA-256
  `62328f3a31b30a8f31fd4f1b3ba4777cfd86f7d881a3577f2d2522814607c04c`.
- Two packaging-only failures are preserved: v3 exposed an installed
  `scripts` package shadowing conflict; v4 then exposed two omitted existing
  test dependencies. Neither changed runtime science, model, data, seed,
  denominator, or thresholds.
- Evidence: `FACTORIAL_RUNTIME_VALIDATION_V1.json`,
  `repair_log/V3_FACTORIAL_RUNTIME_STAGING_IMPORT_REPAIR_V4_20260801.json`,
  and
  `repair_log/V4_FACTORIAL_RUNTIME_TEST_DEPENDENCY_REPAIR_V5_20260801.json`.
- No model weights were loaded; no GPU or Slurm job was used; scientific
  training, crystal generation, S.U.N., and automatic downstream remain
  unauthorized.

### 2026-08-01 — DLM complete-one-epoch protocol and LR override frozen

- Planner and DLM epoch semantics are now explicit. The Planner frozen stream
  is 3,200 microbatches / accumulation 8 = 400 updates, already exactly one
  epoch. The DLM frozen stream is 27,136 rows / global batch 16 = exactly
  1,696 updates.
- B1 and B2 must each complete all 1,696 updates. Metric-based early stopping
  is forbidden; only genuine nonfinite, CUDA/NCCL/OOM, or identity-contract
  failures may terminate an arm early.
- The initial amendment proposed `5e-6`; the user subsequently overrode that
  field and froze the historical LR `5e-5`, without a sweep or a separate
  two-update gate. The adverse two-update smoke remains visible.
  Validation runs at
  `0,212,424,636,848,1060,1272,1484,1696` on one common 100-row panel.
- Intermediate validation is monitoring only. The terminal step-1696
  checkpoint is the sole checkpoint eligible for B1/B2 scientific selection.
- The frozen envelope remains 2xA800, 8 CPUs, 64 GiB, two hours per arm, with
  sequential arms. Historical R5-C completed the same 1,696-update two-A800
  epoch in 37m58s.
- Evidence: `PROTOCOL_AMENDMENT_V3_DLM_ONE_EPOCH_20260801.json`,
  `PROTOCOL_OVERRIDE_V3_DLM_LR5E5_AUTHORIZED_20260801.json`, and
  `AUTHORIZATION_V3_SCIENTIFIC_TRAINING_20260801.json`. Scientific training
  is authorized; automatic downstream remains unauthorized.

### 2026-08-01 — Real-ledger DLM 100-row panel audit

- Added `scripts/audit_h1a2_dlm_fixed_panel.py`; its two focused local tests
  passed.
- The audit ran CPU-only against the published
  `h1a2_r5c_plangraph_sidecar_v2` validation ledger with 9,047 rows. No model,
  GPU, or Slurm job was used.
- Rank 0 selects 50 even ordinals from 0 through 98; rank 1 selects 50 odd
  ordinals from 1 through 99. The global panel is therefore exactly ordinals
  0–99 once each, with no padding, duplicate, or missing row.
- Audit report SHA-256:
  `47c3a5916ef72b701903ed9a319688eb449d671eecdeec4a50029569123659b5`.
  Panel ledger SHA-256:
  `f4fad13216eefe51b073c80df5b6d91507db12dbfb7f95603b01ed9858ab03cd`.
- Model-visible panel identity is
  `8401854824585d35f3385a87c814f68463f677309ae41a5796b9448c73b51dde`.
  The panel is frozen identically for B1/B2 and for step 0, all intermediate
  checks, and terminal step 1696.
- Remote evidence root:
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_fixed_panel_audit_v1`.
  Scientific training and automatic downstream remain unauthorized.

### 2026-08-01 — Scientific training authorized; historical DLM LR restored

- The user authorized the registered P-control/P* 400-update Planner epoch,
  B1/B2 1,696-update body-DLM epoch, and likelihood-only checkpoint selection.
- The DLM LR is explicitly frozen at the historical `5e-5`. The B1/B2
  two-update smoke increases of 7.90% and 6.52% remain adverse risk evidence;
  they are not discarded or relabeled.
- Both DLM arms still consume all 27,136 training rows once, validate on the
  fixed common 100-row panel at steps
  `0,212,424,636,848,1060,1272,1484,1696`, and expose only the terminal
  step-1696 checkpoint for selection.
- Planner-512, crystal generation, S.U.N., G3/G4, automatic promotion, and
  all other downstream action remain unauthorized.
- Evidence: `PROTOCOL_OVERRIDE_V3_DLM_LR5E5_AUTHORIZED_20260801.json` and
  `AUTHORIZATION_V3_SCIENTIFIC_TRAINING_20260801.json`.

### 2026-08-01 — Planner and DLM scientific training DAGs submitted

- Planner array 29391 runs P-control then P* (`0-1%1`), each for the complete
  3,200-microbatch / 400-update epoch on one A800. Likelihood-only assembly is
  job 29392 with `afterany:29391`.
- DLM array 29393 runs B1 then B2 (`0-1%1`), each for the complete
  27,136-row / 1,696-update epoch on two A800s at LR `5e-5`. Likelihood-only
  assembly is job 29394 with `afterany:29393`.
- Initial scheduler state was PENDING for both arrays and dependency-pending
  for both assembly jobs.
- Exact source archives are `b156e3ff…` (Planner) and `14de6308…` (DLM).
  Submission-record SHA-256 values are `35b3ff5f…` and `2b9c9b85…`.
- A remote over-broad preflight attempted one historical CPU dataset-builder
  test whose builder module was intentionally absent from the training
  archive. The actual DLM runtime imports passed and its narrowed A800 suite
  passed 21/21; no source, data, model, seed, denominator, or protocol changed.
  Evidence:
  `repair_log/V3_SCIENTIFIC_REMOTE_PREFLIGHT_SCOPE_NOTE_20260801.json`.
- Automatic downstream and promotion remain false.

### 2026-08-01 — Checkpoint retention policy and failed-run cleanup

- Added a fail-closed post-selection retention utility. Within one training
  arm it keeps the selected best checkpoint plus the latest two numeric-step
  checkpoints; if best overlaps the latest two, only two directories remain.
  The CLI is dry-run by default and requires explicit `--apply`.
- Focused command:
  `python3 -m unittest tests.test_checkpoint_retention -v`.
  Result: 7/7 passed.
- The cleanup candidate set was frozen to 36 explicitly failed,
  diagnostic-only, or no-promotion historical runs. The active queue contained
  only H1A2 assembly 29394; candidate/protected intersection was empty.
- Deleted 59 checkpoint payload directories and no run directory, `final`
  artifact, report, or log. Exact released capacity was 118,287,679,488 bytes
  (110.164 GiB).
- Checkpoint storage changed from 113 directories / 184,092,106,752 bytes to
  54 directories / 65,804,427,264 bytes.

- After cleanup, the complete main `runs` root used 201,922,678,784 bytes
  (188.055 GiB), while the complete main project used 216,485,027,840 bytes
  (201.617 GiB). The protected external R5-C root used 35,830,538,240 bytes
  (33.370 GiB), for 234.987 GiB across those two roots.
- `/public` used 65,703,967,744,000 bytes (59.757 TiB) with
  6,292,577,427,456 bytes (5.723 TiB) available, 92% used.
- H1A2, canonical R5-C, the formal CrysLLMGen anchor, and the external R5-C
  shared root were protected. Their checkpoint path/size snapshot was
  identical before and after cleanup.
- Remote exact deletion manifest SHA-256:
  `7ad74966416923b1a41d241704522e6caace8f79ca1056d3b80a678e453ec925`.
  Local summary: `CHECKPOINT_CLEANUP_20260801.json`.

### 2026-08-01 — Full-epoch endpoints and authorized screens submitted

- Planner likelihood-only training completed `0:0`. P-control step400 has
  fixed-panel target NLL `0.291393`; P* step400 has `0.292366`. P* step400 is
  `0.99347×` frozen P0 NLL and therefore remains inside the +1%
  noninferiority gate.
- The downstream Planner comparison freezes P-control and P* at step400,
  rather than mixing the complete P-control epoch with P* step350 (7/8
  epoch). This endpoint choice was made before any Planner-512 output.
- Body-DLM B1/B2 completed the common 1,696-update epoch. Fixed-panel NLL
  fell from `1.969803` to `1.460710` for B1 and `1.466090` for B2. The
  registered loss remains unchanged; the paired dependency margin is the
  discriminating next test.
- Remote source archive SHA is `ace6e787…`. Both 216-file source inventories
  passed; Planner tests passed 3/3 and dependency tests passed 5/5. The real
  validation preflight retained rows 0–99, selected 100 deterministic donors,
  and built 391 matched/counterfactual dependency pairs.
- The first pure-Python test invocation omitted `PYTHONPATH`; it failed before
  model/GPU/Slurm activity and was repeated successfully with only the
  environment fixed. Repair log SHA: `73d73cf5…`.
- Planner array 29452 and assembly 29454 were submitted under manifest
  `df0e127b…`; submission SHA `5ebddeba…`.
- The first Planner assembly submission named the absent `cpu` partition and
  created no job. Historical jobs 29392/29394 and `sinfo` confirmed `short`;
  assembly 29454 and dependency assembly 29457 use the command-line
  `--partition=short` override without changing the immutable source bundle.
  Repair log SHA: `840574a0…`. Array 29452 was not retried.
- Dependency array 29456 and assembly 29457 were submitted under manifest
  `14e15c0c…`; submission SHA `b29220c8…`.
- Monitoring automation:
  `h1-a2-v3-planner512-dependency-margin-monitor`.
- Crystal generation, S.U.N., conditional body completion, promotion, G3/G4,
  and automatic downstream remain false.

### 2026-08-01 — Phase-2 scientific stops

- Planner jobs 29452/29454 completed `0:0`. P0, P-control, and P* obtained
  composition-valid counts `434`, `456`, and `442` of the same 512 raw
  all-attempt ordinals.
- P* improved over P0 by only `8/512 = 1.5625` percentage points, missed the
  registered +2-point gain, failed to beat P-control, and failed the
  all-metal shortcut gate. The frozen decision is
  `scientific_stop_retain_P0`; terminal SHA
  `09f66256f5d5f96d0a4b161770801adaea5da9e75758389ff45b10d0680f3c0c`.
- Dependency jobs 29456/29457 completed `0:0`. B1/B2 mean paired margins were
  `0.2598086` and `0.2330673`; B2-B1 was `-0.0267413`, with 95% bootstrap CI
  `[-0.0555922, 0.0019861]`.
- B2 therefore failed the strict improvement gate even though its own margin
  was positive and its fixed-panel NLL was noninferior. The frozen decision is
  `scientific_stop_retain_B0`.
- Formal factorial engineering, G3, checkpoint promotion, and automatic
  downstream remain blocked.

### 2026-08-01 — Packed post-stop refine800 + S.U.N. diagnostic v7 submitted

- The user separately authorized a diagnostic-only 256-attempt four-arm
  generated-crystal/S.U.N. run and explicitly required diffusion refinement
  for every arm.
- v1 prepare job 29495 failed before generation with
  `ModuleNotFoundError: crystal_dlm.h1a2_factorial_contract`; its isolated
  runtime contained the module but omitted `crystal_dlm/__init__.py`, so an
  older shared regular package shadowed it. Slurm cancelled dependent array
  29496. No generated or refined attempt existed and no partial result is
  reused.
- v2 added the isolated package marker, corrected the Planner terminal-field
  assertion, and packed the work to stay within submission QOS. Its four
  elements nevertheless stopped during preparation because P* ordinal 30 had
  an additional raw model line. It produced no body, refinement, or S.U.N.
  output.
- The user then removed exact seven-line raw conformance as a blocking gate.
  v4 preserves and labels every raw deviation, uses the already accepted
  frozen canonical Plan, and continues the ordinal. The real preparation
  integration yields 253 continuing and 3 original Planner failures for both
  P0 and P*. P* ordinal 30 is retained with an advisory warning rather than
  repaired or replaced.
- v3 was not submitted because its pre-submission test exposed isolated
  runtime import ordering. v4 fixes only that engineering issue. Focused A800
  tests passed 8/8. Its source manifest SHA is
  `ab77239c17a72c4982459e38b7777c5a91e704d2f4b9ffebeb58cc512d2d03f5`;
  its runtime manifest SHA is
  `232279e2b7da41ab2d1d72732735dea03cfe211c0c8069fedd8ac4c8f6ed9746`;
  and its transfer archive SHA is
  `e734084dbfcb2266a44ef33a3f7ce16436f14be15c268264fb705bb84de767c9`.
- All v4 elements 29520_[0-3] then failed before body generation because the
  frozen `runtime/scripts` directory lacked a regular-package marker and was
  shadowed by an installed package. Assembly 29521 recorded the failure. Only
  Plan preparation `_SUCCESS` files exist; no generated/refined/S.U.N. output
  is reused. The common arm stderr SHA is `e3d01979…`.
- v5 adds only `runtime/scripts/__init__.py`. Its full source manifest, 8/8
  focused tests, actual body-runner import, and shell syntax checks passed on
  A800. Source manifest SHA is
  `e8176a67db5568f881ee60673b26f330474aa238247d16f9c7aaeb8259dd217a`;
  runtime manifest SHA is
  `0c83c9aa648dccf16532b40f1e74d31df96ca2e2ec2b21893504772730992e7c`;
  transfer archive SHA is
  `cfa26e929145799f225172e4297b982036bf1c89956e042e59438d7afaea37b1`.
- All v5 elements 29531_[0-3] and assembly 29532 failed before Plan/model
  loading because ordinary `__pycache__/*.pyc` artifacts created by the
  import tests appeared as extra source files. No stage `_SUCCESS` exists and
  no output is reused. The common arm stderr SHA is `18abef57…`.
- v6 excludes only Python bytecode cache paths from the source file-set
  comparison; every registered source file remains exact-SHA checked and the
  jobs export `PYTHONDONTWRITEBYTECODE=1`. Focused tests passed 9/9, the real
  body-runner import passed, and shell/source checks passed on A800. Source
  manifest SHA is
  `f23b5a01be28d23f3c71f6bdd41347259d1ed52293b7072bb49e1fa5bd32f0cf`;
  runtime manifest SHA remains
  `0c83c9aa648dccf16532b40f1e74d31df96ca2e2ec2b21893504772730992e7c`;
  transfer archive SHA is
  `20d966eb26a0409c3dcd6334313c35daeff62505d4512e273ff69749bbeb85fe`.
- All v6 elements 29540_[0-3] and assembly 29541 failed `1:0` after Plan
  preparation but before body generation. The configured expected vocabulary
  SHA was from a standalone preflight tokenizer reconstruction, while the
  runner correctly loaded the tokenizer serialized with each frozen
  checkpoint. No body/refinement/direct/S.U.N. output exists or is reused.
- The R5-C B0, B1, and B2 checkpoint tokenizers were audited directly:
  all three have byte-identical `tokenizer.json` SHA `3a21588a…` and
  `tokenizer_config.json` SHA `8e89acaa…`, identical 128,830-entry vocabulary
  SHA `3acc073d…`, identical IDs for all 2,481 frozen data tokens, and zero
  missing tokens.
- v7 keeps the exact vocabulary gate, points it at that audited frozen
  checkpoint identity, and additionally verifies both tokenizer files before
  loading the body model. No tokenizer, token ID, checkpoint, data, seed,
  denominator, or evaluation policy changed. Focused A800 tests passed 10/10
  and the checkpoint-tokenizer integration audit passed. Source manifest SHA
  is `5d05e23da6ba4e0e49f4646a5db8181c27f3dd47d67148cac9baaa842fa6a42f`;
  runtime manifest SHA remains `0c83c9aa…`; transfer archive SHA is
  `afa1276e40aaf2fe9639350b4e035cb220b090795cd638cd66600f80a55da59f`.
- Array 29549 (`0-3%2`) runs each arm end to end on one A800:
  arm-local preparation → body generation → frozen CrysLLMGen `model_494`
  refine800 → direct metrics → frozen-cache S.U.N. Assembly 29550 is the only
  second Slurm submission and depends `afterany` on the entire array.
- The report is diagnostic only: no MP API, no retry/replacement/repair/filter/
  rerank, no formal G3, no checkpoint reselection, no promotion, no training
  feedback, and no automatic downstream.
- The existing lightweight monitor was updated in place to the v7 jobs and
  remains read-only.

### 2026-08-02 — v7 delayed S.U.N. import failure and future fail-fast gate

- M00 and M10 completed Plan/body generation, frozen `model_494` refine800,
  and direct CrysLLMGen metrics. M00 produced 243 refined successes and 13
  recorded failures; M10 produced 242 refined successes and 14 recorded
  failures, each over the immutable 256 all-attempt denominator.
- Both arms then exited `1:0` at the first S.U.N. runner import with
  `ModuleNotFoundError: No module named
  'crystal_dlm.wqcodiff.contracts'`. No frozen-cache S.U.N. output was
  produced. M01/M11 were still running in the unchanged array at the
  2026-08-02 check.
- The frozen v7 runtime contains the top-level runner but not its complete
  adapter dependency closure:
  `crystal_dlm/wqcodiff/contracts.py`,
  `crystal_dlm/wqcodiff/crysllmgen/__init__.py`,
  `crystal_dlm/wqcodiff/crysllmgen/a100_sun.py`, and
  `crystal_dlm/wqcodiff/crysllmgen/epoch_training.py`, together with the
  required isolated package marker.
- The prior 10/10 preflight was insufficient because it compiled/imported the
  packed pipeline and body runner but did not execute the S.U.N. runner's
  imports hidden inside `main()`. The error was therefore delayed until after
  the expensive direct-metric stage.
- Running the new `--verify-only` gate against the untouched local v7 runtime
  rejects it immediately and names all five missing runtime files, before any
  Slurm, model, GPU, generation, or refinement work.
- Future execution bundles must use
  `scripts/a800/stage_crysllmgen_a100_sun_runtime.py` SHA
  `890c743f31e3f860fd4a10f0380e75c2c149cf4b2593ec16c48e71160d4c226f`.
  The staged runner imports its adapter closure at module load and exposes an
  isolated origin preflight; runner SHA is
  `e952795c32efe1be328165d99389ae3934a6698d29027569376bbb984cf788de`.
- Regression test
  `tests/test_a100_sun_runtime_closure.py` SHA
  `b6d09fa3b24b8ef1d0d967ae92b73a3948dc8abf4e6ef3a02318bba93676f312`
  passed 3/3 cases: complete isolated runtime succeeds, a missing delayed
  dependency fails before scientific work, and an incomplete package cannot
  leak in from the shared checkout. Twelve existing S.U.N. adapter/pipeline
  tests also passed.
- v7 remains immutable and is not patched or resubmitted by this diagnosis.
  This closes the specific missing/escaped runtime-dependency class before
  future GPU work; it does not change any model, data, seed, denominator,
  refinement, or S.U.N. scientific protocol.

### 2026-08-02 — evaluation-only v8 S.U.N. repair terminal

- Array elements 29617/29619/29620/29625 and assembly 29618 all completed
  `0:0`. The terminal report is complete with SHA
  `6d2e26d263669e207c2ddbecfd3ee78bd66eb473527a466d958a6fcd906dabb6`.
- V8 reused the exact hashed v7 generation/refine800/direct evidence.
  `generation_or_refinement_rerun=false`,
  `direct_metrics_rerun=false`, frozen-cache-only evaluation is true, and
  `mp_api_enabled=false`.
- Per-arm raw all-attempt counts are:

  | Arm | Generation | Composition | Structure | Joint | Unique | Novel unique | Strict S.U.N. | Meta S.U.N. |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | M00=P0+B0 | 243 | 203 | 241 | 201 | 243 | 221 | 13 (5.08%) | 58 (22.66%) |
  | M10=P*+B0 | 242 | 204 | 242 | 204 | 240 | 206 | 8 (3.13%) | 59 (23.05%) |
  | M01=P0+B2 | 72 | 62 | 72 | 62 | 72 | 61 | 7 (2.73%) | 20 (7.81%) |
  | M11=P*+B2 | 92 | 77 | 92 | 77 | 90 | 67 | 4 (1.56%) | 24 (9.38%) |

- The Planner effect is small at B0: joint validity `+1.17` points, meta
  S.U.N. `+0.39` point, and strict S.U.N. `-1.95` points. At B2, P* improves
  joint validity by `+5.86` points (exact McNemar `p=0.0444`) but meta
  S.U.N. changes only `+1.56` points (`p=0.541`) and strict S.U.N. changes
  `-1.17` points.
- B2 is the dominant failure: versus B0 it reduces joint validity by
  `-54.30` points under P0 and `-49.61` points under P*. M11 versus M00 is
  `-48.44` joint-valid points, `-3.52` strict-S.U.N. points, and `-13.28`
  meta-S.U.N. points.
- Terminal decision:
  `diagnostic_only_retain_phase2_scientific_stops`. Both Phase-2 gates,
  formal G3, checkpoint reselection, promotion, and automatic downstream
  remain false.
