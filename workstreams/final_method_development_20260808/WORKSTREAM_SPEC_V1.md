# Crystal-DLM Evidence-First workstream specification v1

Status: frozen before reading any C0/C1, SFT-v2, SFT-v2-C, B3, or downstream
scientific result.

Frozen on: 2026-08-08 (Asia/Shanghai)

Branch: `codex/evidence-first-sun-msun`

## Objective

Improve both raw strict S.U.N. and raw Meta S.U.N. point estimates relative to
the protected `P0+B0+safe-axis+model_494` baseline. Promotion additionally
requires the independent Meta S.U.N. difference 95% interval lower bound to be
greater than -2 percentage points and joint-valid S.U.N. not to decline in
point estimate.

Raw-attempt metrics are primary. Chemistry-gated and joint-valid S.U.N. are
co-reported without replacing the raw denominator. The public CrysLLMGen
93.55% composition-validity value is a raw-attempt reference, not a gate for
this workstream.

## Protected identities

| Asset | SHA-256 |
|---|---|
| Planner P0 adapter | `65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a` |
| Body B0 adapter | `5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d` |
| Refiner `model_494` | `573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e` |

Protected assets are immutable and may never be deleted by checkpoint cleanup.

## Global safety and scientific gates

- `struct_valid`: no more than 1 percentage-point decline.
- COV-P, COV-R, novel, unique, and element coverage: no more than 2 percentage-
  point decline each.
- Planner composition validity must pass the candidate-stage literal positive
  gain gate.
- No new failure class, shortcut collapse, or evaluator coverage gap.
- Unknown, relaxation failure, and hull failure are false in primary S.U.N.
  denominators; coverage-adjusted values are secondary only.
- No retries, replacements, repairs, filtering, reranking, best-of-n, seed
  selection, denominator shrinkage, or result-conditioned sample expansion.

## Planner portfolio

All four authorized Planner candidates execute to their registered terminal:
C0, C1, SFT-v2, and SFT-v2-C. This `always_execute` user override supersedes
the earlier rule that could skip SFT-v2 after a 93.55% C1 result.

C0/C1 retain their pre-existing immutable package, fixed checkpoint 400,
400-update training contract, raw64/raw256 gates, and B0/D1 downstream.

SFT-v2 and SFT-v2-C start from protected P0 and generate the minimal six-line
no-charge Plan:

```text
formula: ...
anion: ...
lattice: ...
spacegroup: ...
volume: ...
end: plan
```

Their common data multiset is:

- one formula-input-only structural anchor for every one of the 27,136 frozen
  MP20 training rows (`ALL`);
- one unconditional six-line Plan for every row in `POS`, where `POS` is the
  intersection of legacy SMACT 3.1 nonshortcut primary validity and exact
  SMACT 4 stable uniform-primary witness validity;
- one deterministic chemistry auxiliary record per `POS` row, cycling through
  repeated-atoms to formula, repeated-ions to charge sum plus formula, masked
  oxidation, and formula to elements/counts/N.

Evaluator-invalid formulas may occur only in formula-conditioned anchors.
They must never be unconditional formula-generation targets.

Both candidates use LR 2e-6, batch size 1, gradient accumulation 8, weight
decay 0, cosine schedule, warmup `max(25, round(0.03 * total_updates))`, one
complete frozen ledger epoch, and `ceil(record_count / 8)` optimizer updates.
The final partial accumulation is normalized by its actual microbatch count;
records are neither dropped nor repeated. Formula, oxidation-code,
charge-sum, and element/count answer tokens receive weight 2; all other answer
tokens receive weight 1; every sample weight is 1. Only the fixed final
checkpoint may be promoted.

SFT-v2 uses a frozen hash-shuffle. SFT-v2-C uses the identical record multiset,
seed role, optimizer, and update count, changing only order: the first 10% of
records alternates direct-POS and auxiliary-POS records without replacement;
the remainder uses deterministic deficit-round-robin over anchors and unused
POS records. Every record occurs exactly once.

Raw64 relative-to-P0 gates for both candidates:

- composition-valid and nonshortcut-primary each improve by at least 1/64;
- parse and completion lose at most one attempt each;
- shortcut count does not increase;
- unique-formula rate and element coverage lose at most 2pp;
- absolute mean-N drift is at most 0.5;
- anchor NLL degradation is at most 1%;
- no new failure class.

Raw256 keeps literal positive gains of at least 1/256 for composition-valid
and nonshortcut-primary and adds arity, top-1 frequency, coarse-field marginal,
and SMACT4 audits. McNemar and 10,000-replicate paired bootstrap are reported,
not substituted for the literal gain gate. At most two non-P0 Planner
candidates enter protected downstream256.

## DLM portfolio

First inventory B0/B1/B2 and audit IID, D1, synthetic safe-axis, and actual
B0+safe-axis rollout states for NLL, calibration, commit/group width, and state
distribution.

B3 is mandatory: B0 initialization, identical R5-C rows and prompt/answer
bytes, `dynamic_v1`, IID:safe-planned 2:1, zero mixed-axis groups,
`z_before_xy=0`, the historical two-GPU optimizer and one-epoch scale, and
fixed safe-axis inference. S.U.N. cannot select a checkpoint.

B3 expands only if synthetic and actual-rollout NLL point estimates are both
below B0, body64 completion loses at most one attempt, and duplicate/new
failures do not increase. Calibration is reported but is not a significance
gate. A passing B3 permits B3-R 1:1 and 1:2 with fixed updates, data order, and
optimizer. B4 is allowed only if synthetic improvement fails to transfer to
actual rollout state. B5 is allowed only after a strict decoder derivation and
standard-loss degeneration parity. B6, on-policy training, and RL are outside
scope.

## Integration and final evaluation

Both portfolios must reach terminal or saturation before final model freeze.
Integration256 is capped at protected P0+B0+safe-axis, top-two Planner with
B0+safe-axis, P0 with top-two DLM, top-Planner plus top-DLM, and at most one
decision-changing interaction cell.

Final evaluation contains the protected baseline and at most two fully passing
new methods. Each method receives two independent ledgers:

- fixed raw1000: exactly 1,000 attempts, all failures retained;
- released-sampler-compatible accepted1000: record every raw call and reject
  only Planner parse failures or Body proposals with no processable graph until
  exactly 1,000 proposals are accepted; after acceptance, every Direct/S.U.N.
  outcome remains in the accepted denominator.

Generation and the Novel-intersection-Unique union are frozen before a common
MP reference cache is prefetched with an ephemeral `MP_API_KEY`. The key must
not appear in commands, files, logs, manifests, or source. Formal Slurm GPU
evaluation is offline. Missing credentials yield
`HOLD_EVALUATOR_INCOMPLETE`, not guessed or partial S.U.N.

Each final arm reports Planner, Body, distribution, novelty/uniqueness, pure
stability, raw/chemistry-gated/joint-valid strict and meta S.U.N., hull
coverage/unknown/failure taxonomy, Plan adherence, draft-to-refiner retention,
latency, forward calls, VRAM, and effective safe-axis width. Proportions include
integer counts, denominators, and Wilson 95% intervals. Paired stages use exact
McNemar and 10,000 paired bootstrap replicates. Final independent arms use an
independent difference bootstrap.

## Forbidden scope

No Planner RL, DLM RL, TraceRL/B6, refiner training, old stopped-route revival,
formal ablation campaign, or new external-baseline reproduction is authorized.
Scientific failure is a valid terminal and cannot be rescued by changing a
seed, denominator, threshold, checkpoint, or factor after observing results.

## Execution and cleanup

Every run uses a new immutable source/run root, source manifest, archive,
ledger, authorization record, submission record, and terminal. Local files are
the code source of truth. Remote paths must not already exist. Existing running
jobs are never modified or cancelled.

After the explicitly authorized one-time local construction of the portable
SMACT4 runtime bundle, local execution of project programs or environment
installation is prohibited. Tests, data jobs, training, generation, and
evaluation execute on A800 only through the two pre-existing 5090 tmux
sessions. Local work is limited to source editing, evidence registration,
transfer preparation, and Git operations.

After each terminal, preserve fixed endpoints, promoted checkpoints, manifests,
hashes, logs, results, and a deletion manifest. Other intermediate payloads may
be removed only after the deletion list and hashes are frozen. Protected
checkpoints are never deleted.
