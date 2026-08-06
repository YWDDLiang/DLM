# PlanGraph-DLM ICLR Workstream

Status: `v3_h1a2_authorized_screens_running`

> **2026-08-01 execution update:** execution is governed by
> `EXPERIMENT_CHARTER_V3_H1A2_TWO_FACTOR.md` and
> `EXPERIMENT_REGISTRY_V3_H1A2_TWO_FACTOR.json`. V1 and V2 are retained only
> as historical design/negative evidence. Authorized V3 CPU preflights and
> bounded Planner/body engineering smokes are complete; strict factorial
> runtime validation and the exact DLM 100-row panel audit are also complete.
> The registered P-control/P* and B1/B2 scientific training DAGs completed as
> jobs 29391→29392 and 29393→29394. The separately authorized Planner-512 and
> paired direct-dependency margin screens are running as 29452→29454 and
> 29456→29457. Both trained Planner
> arms use the complete step-400 epoch; both DLM arms use terminal step 1696.
> Crystal generation, S.U.N., G3/G4, and automatic promotion remain blocked.
>
> V3 corrects a provenance error in V2: an R5-C `plan_state` reconstructed
> from a target structure is teacher supervision for the body DLM, not a
> model-proposed Planner result. The fully-de-novo reference is the frozen
> H1-A2 epoch-2 Planner sampling a seven-line Plan, followed by the frozen
> R5-C body and frozen refiner.

Created: 2026-07-31

Target: complete a submission-ready ICLR manuscript by 2026-09-17 while
preserving the complete H1 series as the immutable fallback.

## Working thesis

Crystal-language tokens do not have independent semantics. Composition,
symmetry, lattice, site identity, and coordinates form an explicit dependency
graph. V3 tests two independent learned improvements:

1. a model-sampled, H1-A2-compatible Planner improvement over frozen
   H1-A2 epoch 2; and
2. a dependency-aligned body-DLM improvement initialized from frozen R5-C.

The confirmatory study is `P0/P* × B0/B*`. Within each Planner pair, the body
arms receive the same realized sampled Plan and paired noise. Structure-derived
teacher Plans may train or diagnose the body DLM, but they are never counted as
Planner evidence. There is no shuffle arm.

The continuous diffusion refiner stays frozen. This keeps the paper centered
on the two requested contributions: the Planner model and the discrete DLM.

## Isolation boundary

- This directory is the only writable home for the new experiment design.
- H1 source, checkpoints, reports, frozen bundles, and run directories are
  read-only baselines.
- New runs must use a new run root and must never resume into an H1 directory.
- `AUTHORIZATION_V3_PREFLIGHT_SMOKE_20260801.json` authorizes only prerequisite
  CPU materialization, real-tokenizer preflight, and bounded 32-row
  engineering smoke.
- `AUTHORIZATION_V3_SCIENTIFIC_TRAINING_20260801.json` separately authorizes
  the registered 400-update Planner training, one-epoch B1/B2 training, and
  likelihood-only checkpoint selection. Automatic downstream action and
  promotion remain unauthorized.

## Documents

- `H1_FALLBACK_MANIFEST.md`: immutable H1 assets and fallback policy.
- `EXPERIMENT_TODO_INDEX_V3.md`: single live task/result/job index for all V3
  work.
- `PLANNER_PSTAR_METHOD_V1.md`: frozen H1-A2-compatible P* method contract.
- `AUTHORIZATION_V3_PREFLIGHT_SMOKE_20260801.json`: bounded prerequisite/smoke
  authorization; it does not authorize scientific training.
- `AUTHORIZATION_V3_SCIENTIFIC_TRAINING_20260801.json`: current scientific
  training authorization boundary.
- `AUTHORIZATION_V3_PLANNER512_DEPENDENCY_SCREEN_20260801.json`: authorization
  for the raw Planner-512 and likelihood-only dependency-margin screens.
- `PROTOCOL_OVERRIDE_V3_PLANNER_FULL_EPOCH_ENDPOINT_20260801.json`: freezes
  step400 for both trained Planner arms to remove unequal-exposure bias.
- `PROTOCOL_OVERRIDE_V3_DLM_LR5E5_AUTHORIZED_20260801.json`: user-directed
  DLM learning-rate override and retained smoke-risk evidence.
- `EXPERIMENT_CHARTER_V3_H1A2_TWO_FACTOR.md`: governing corrected two-factor
  design.
- `EXPERIMENT_REGISTRY_V3_H1A2_TWO_FACTOR.json`: machine-readable V3 identity,
  provenance firewall, arms, budgets, and stop gates.
- `IMPLEMENTATION_TASKS_V3_H1A2_TWO_FACTOR.md`: remaining pre-submission work.
- `EXPERIMENT_CHARTER_V2_H1A2.md`: superseded one-factor design evidence.
- `EXPERIMENT_REGISTRY_V2_H1A2.json`: superseded V2 identity, arms,
  budgets, and stop gates.
- `IMPLEMENTATION_TASKS_V2_H1A2.md`: superseded V2 backlog.
- `EXPERIMENT_CHARTER_V1.md`: preregistered scientific design, metrics, and
  decision gates for the now-stopped generated-JSON experiment.
- `IMPLEMENTATION_TASKS_V1.md`: bounded implementation and execution backlog.
- `EXPERIMENT_REGISTRY_V1.json`: machine-readable status and arm registry.
- `TRAINING_INTEGRATION_REPORT_V1.md`: local D0/D1/D2 integration evidence and
  remote frozen-data/tokenizer preflight evidence.
- `ENGINEERING_PILOT_32_MANIFEST_V1.json`: frozen four-arm A800 runtime-pilot
  review point; it does not authorize submission.

## Paper fallback rule

The new method must earn its place. If no candidate clears the confirmation
gate by 2026-08-31, the submission returns to the verified H1-A2 epoch-2 line
and its existing H1 studies. The new work may then appear only as a carefully
scoped negative or diagnostic result; it must not displace a reproducible H1
baseline with a weaker unfinished system.

## Current V3 implementation checkpoint

The single live operational record is `EXPERIMENT_TODO_INDEX_V3.md`. It marks
completed work and attaches its result before the next task advances.

The earlier V1 remote dataset, 768-token preflight, engineering pilot, Planner
checkpoints, and G1 outputs are historical negative evidence only. They are
not initialization, data, or validation evidence for V3. The previous
generated-JSON Planner, ValidReplay, and JointChem checkpoints are not eligible
as P*.

The Planner P* method is frozen as the model-generated, H1-A2-compatible
Look-Ahead Consistent Planner v1. Its common 3,200/256 stream, real-tokenizer
preflight, and one-A800 32-row P-control/P* smoke are complete. The B1/B2
teacher sidecar and real-tokenizer preflight are complete, and both revised
2xA800 32-row body smokes completed successfully. Their engineering learning
rate `5e-5` worsened fixed-panel loss in the two-update smoke. That adverse
evidence is retained, but the user explicitly restored the historical
scientific LR `5e-5`; there is no LR sweep. B1/B2 must each complete the full
27,136-row epoch: 1,696 optimizer updates at global batch 16.
Validation occurs at step 0 and every 212 updates; only the terminal
step-1696 checkpoint is eligible. The exact protocol is frozen in
`PROTOCOL_AMENDMENT_V3_DLM_ONE_EPOCH_20260801.json` plus the user override
`PROTOCOL_OVERRIDE_V3_DLM_LR5E5_AUTHORIZED_20260801.json`. The real-ledger
panel audit passed with global validation ordinals 0–99 exactly once and is
recorded in `DLM_FIXED_PANEL_AUDIT_V1.json`.

Input/output alignment is fail-closed: the sampled seven-line Plan is persisted
and hashed, the two body arms for that Planner receive identical prompt bytes,
and R5-C training requires additive prompt/answer tokenization with exactly
`7 + 4*N` body tokens. Historical R5-C body-DLM training already used
2xA800 NCCL DDP (`torchrun`, batch `1` per GPU, accumulation `8`, global batch
`16`), and V3 B1/B2 retain that contract. Planner training remains
single-GPU.

Fresh factorial runners now cover P0/P* model sampling, B0/B* body generation,
the frozen 800-step refiner, and strict four-arm all-attempt assembly.
Model-sampled Plan provenance, tokenizer-vocabulary identity, additive
tokenization, exact `7 + 4*N` body length, stateless ordinal seeds, ordered
rank merge, paired Plan/prompt identity, and earliest-failure preservation
are fail-closed. Local and A800 targeted suites both passed 50/50. This was
validation-only: no model was loaded and no GPU or Slurm job was used.

Planner 400-update training and DLM one-epoch training completed under fresh
immutable run roots. P-control and P* both use the complete step-400 epoch for
the authorized Planner-512 screen; B1 and B2 both use terminal step1696 for
the authorized paired direct-dependency margin screen. The common DLM
fixed-panel NLL fell by 25.84% for B1 and 25.57% for B2 relative to
initialization, so the loss is not changed after observing training.
Crystal generation, S.U.N. evaluation, checkpoint promotion, G3/G4, and
automatic downstream action remain unauthorized.
