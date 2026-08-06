# H1 comp_valid root-cause audit and SFT-first plan V2

Status: `decision_only_remote_cleanup_pending`

Date: 2026-08-05

This document supersedes the diagnosis and training design in
`H1_COMP_VALID_SFT_FIRST_ROUTE_V1.md`.  It does not authorize training,
sampling, checkpoint deletion, or downstream execution.

## 1. Metric correction

The published CrysLLMGen MP-20 value `93.55%` is treated as the strict
raw-output composition-validity result reported in the paper's de-novo table.
It must not be discounted by a S.U.N survivor denominator, coverage
adjustment, or an assumed `2--5%` correction.

The released sampler's parse/atom filtering behavior is a separate
implementation-contract issue.  It is relevant to exact reproduction, but it
is not a license to rescale the published `93.55%`.

Consequently, the gaps are real:

- published CrysLLMGen: `93.55%`;
- locally reproduced CrysLLMGen asset: `89.2%`;
- H1-A2 epoch-2: `87.8%`;
- frozen P-control discovery screen: `456/512 = 89.0625%`.

## 2. Root-cause conclusion

The primary problem is the H1 supervised objective and representation, not a
corrupted MP-20 corpus.  MP-20 nevertheless creates an important
evaluator-alignment problem.

### 2.1 The public `93.55%` model has not been reproduced locally

The paper and public training command specify LLaMA-2 7B, one epoch, AdamW,
and learning rate `1e-4`.  The locally registered "official" CrysLLMGen atom
asset is instead:

- Meta-Llama-3-8B-Instruct;
- rank-8 `q_proj/v_proj` LoRA;
- five epochs / 1,215 steps;
- terminal training loss `0.494306`;
- local direct composition validity `89.2%`.

H1 is different again: Meta-Llama-3-8B base, rank-16 LoRA over seven
projection/MLP modules, two total data passes, and learning rate `2e-5`.

Therefore `93.55`, `89.2`, and `87.8` are not currently checkpoint/recipe
parity measurements.  The first mandatory audit is model/data/tokenizer/
evaluator parity, not a speculative new optimizer sweep.

### 2.2 MP-20 is physically real but is not a clean SMACT-positive corpus

Under the frozen H1/CrysLLMGen-style composition taxonomy, the exact local
MP-20 training split has 27,136 rows:

| Stratum | Count | Rate |
|---|---:|---:|
| nonshortcut charge/Pauling-valid primary | 7,079 | 26.087% |
| all-metal shortcut | 9,302 | 34.279% |
| single-element shortcut | 226 | 0.833% |
| charge-neutrality failure | 9,806 | 36.136% |
| Pauling failure | 440 | 1.621% |
| oxidation-state missing | 283 | 1.043% |

Thus `38.801%` of real MP-20 train rows fail this heuristic evaluator and
another `35.112%` pass only by shortcut.  Similar rates occur in validation
and test.

This does **not** mean that MP-20 structures are physically invalid.  It means
that ordinary maximum-likelihood SFT on all MP-20 rows is misaligned with a
SMACT-based generation metric.  The same MP-20 benchmark still supports a
published `93.55%`, so the corpus itself cannot explain the H1 ceiling.

### 2.3 The current seven-line target contains a causal contradiction

An actual H1 teacher target can be:

```text
formula: Ga4Te4
anion: chalcogenide
charge: charge_fail
...
```

The prompt asks for a chemically plausible formula, but SFT assigns normal
likelihood to an evaluator-invalid formula and explicitly teaches the
`charge_fail` continuation.  Because `charge:` follows `formula:`, the
charge label cannot causally guide the already generated element counts.

The compact `Ga4Te4` representation also gives each element/count decision
only one or a few token losses.  CrysLLMGen instead serializes one atom symbol
per site, so composition receives repeated token supervision.  Its public
training code also uses an element-infill task roughly one third of the time,
using the same language-model head.  H1 has neither repeated atom supervision
nor a same-head chemical infill objective.

### 2.4 Existing experiments rule out naive valid-only replay

The earlier ValidReplay candidate already used `80%` chemistry-valid targets
and `20%` anchor replay.  Its corrected decision was
`stop_no_plan_candidate`; paired-256 direct composition was `208/256` versus
P0 `211/256`, and downstream Body/CIF failures increased.

P* also showed that an auxiliary look-ahead head can create negative transfer
and shortcut inflation.  Therefore the next SFT must change the causal
representation and use the normal LM head.  Repeating "filter valid rows and
continue SFT" or adding another auxiliary classifier is not justified.

## 3. Stage A: parity and teacher-forced diagnosis before training

No generated result is used to tune this stage.

### A1. Public-versus-local CrysLLMGen identity table

Record and hash:

- exact LLaMA-2 7B paper checkpoint/adapter availability;
- local LLaMA-3 CrysLLMGen adapter identity;
- MP-20 split row and SHA identities;
- tokenizer versions and tokenization of all element symbols and counts 1--20;
- prompt, LoRA modules/rank, epoch count, learning rate, maximum length;
- generation temperature/top-p/top-k/maximum length;
- SMACT/pymatgen versions, oxidation-state tables, shortcut semantics and
  Pauling behavior.

Run the same frozen formula ledger through both the released evaluator and
the H1 common evaluator.  Any evaluator disagreement is reported directly;
neither output is relabelled.

If the exact paper LLaMA-2 adapter is unavailable, say
`PAPER_CHECKPOINT_NOT_REPRODUCED`.  Do not call the local LLaMA-3 adapter a
reproduction of the `93.55%` endpoint.

### A2. Frozen representation probe

On disjoint MP-20 validation rows, compare three teacher-forced views with the
same base checkpoint and update budget:

1. current compact formula target;
2. canonical repeated atom-symbol sequence;
3. repeated element-plus-oxidation-witness sequence.

Measure:

- formula/atom token NLL;
- exact formula reconstruction from teacher-forced prefixes;
- next-token primary-support mass;
- charge-witness consistency;
- element/count tokenizer fragmentation;
- gradient norm and cosine against the original Plan loss.

This is a diagnostic.  No autoregressive `comp_valid` is read for checkpoint
selection.

## 4. Stage B: one primary SFT candidate

### B1. Causal representation

The candidate generates a canonical ionic atom sequence before geometry
labels.  A representative target is:

```text
chemistry: primary
ions: Li+1 Li+1 O-2
lattice: ...
spacegroup: ...
volume: ...
end: plan
```

Rules:

- each emitted ion is a generated token sequence, not a prefilled answer;
- the formula and atom count are deterministically compiled from the ion
  sequence as part of the declared representation;
- compilation performs no element replacement, count repair, filtering,
  retry, reranking, or charge correction;
- a malformed or nonneutral sequence remains a raw failed attempt;
- uniform and frozen mixed-valence witnesses are allowed;
- unary, true all-metal, and oxidation-missing rows have explicit separate
  modes and never count as primary gain.

This ordering gives the model repeated atom supervision and exposes a
charge-bearing prefix before the final composition is complete.

### B2. Same-head multitask SFT

Use task tokens and the ordinary causal-LM head:

- `60%` primary ionic-sequence generation;
- `20%` primary element/ion infill, mirroring the useful part of the public
  CrysLLMGen recipe;
- `20%` full-MP20 original-schema anchor replay.

Only the 7,079 train rows with a frozen primary witness enter the first two
tasks.  All MP-20 rows remain eligible for the anchor task, so heuristic
false negatives are not declared physically wrong or silently erased.

The mixture, row ledger, witness choice, canonical ordering, tokenizer,
optimizer and update budget must be frozen before any generated sample is
read.  Begin from P0 and include a matched old-schema field-balanced control.

### B3. Training and selection

Initial bounded proposal:

- P0 initialization;
- one A800 per arm, at most 8 CPU;
- BF16, batch 1, gradient accumulation 8;
- LoRA rank/modules unchanged from P0;
- learning rate `2e-6`;
- 400 updates, 25-update warmup, cosine decay;
- validation every 50 updates;
- one fixed seed and immutable row order.

Checkpoint eligibility uses only held-out NLL, exact reconstruction,
primary-support mass and schema audits.  It cannot read generated
composition validity, S.U.N., energy, hull, Body or refiner results.

Stop before full training if:

- any witness/compiler/evaluator parity check fails;
- the teacher token is outside legal support;
- median auxiliary/original gradient cosine is below `-0.25`;
- original-schema validation NLL degrades by more than `1%`;
- primary validation formula diversity collapses by more than `2 pp`.

## 5. Raw-attempt scientific ladder

Every denominator is fixed raw attempts.  No retry, replacement, repair,
filter, rerank, fallback, or survivor denominator is allowed.

### B4. Plan-only 64 engineering/science screen

Candidate versus matched control:

- raw composition gain at least `+3/64`;
- nonshortcut/primary gain at least `+3/64`;
- charge failures decrease by at least 25%;
- shortcut count does not increase;
- parse/completion lose at most one attempt;
- unique-formula rate and fixed-alphabet element coverage each lose at most
  `2 pp`;
- no new failure class.

### B5. Paired Plan-only 256

- raw composition gain at least `+8/256`;
- nonshortcut/primary gain at least `+8/256`;
- paired 95% bootstrap interval for raw gain has lower endpoint above zero;
- exact McNemar and all discordant ordinals reported;
- shortcut, top-1 frequency, mean atom count, arity, family, uniqueness and
  element coverage pass frozen noninferiority bounds.

### B6. Strict raw 1,000 confirmation

Only a passing 256 enters a new independent 1,000-attempt ledger.  Report on
the same common evaluator:

- H1 P0;
- the selected SFT candidate;
- locally reproduced CrysLLMGen `89.2%`;
- exact paper checkpoint if and only if its identity is available.

The practical target is at least `92%` raw comp_valid with a positive paired
gain and no shortcut/diversity inflation.  `93.55%` remains the public
reference, not a post-hoc pass threshold.  Only after this Plan gate may the
frozen Body/refiner be evaluated for raw joint validity and S.U.N.

## 6. Stage C: one supervised preference fallback

If Stage B improves teacher-forced support and raw composition but misses the
256 gate, permit one immutable on-policy preference dataset disjoint from all
evaluation ledgers.

- pair primary-valid and invalid samples within matched atom-count, arity and
  broad family strata;
- all-metal and unary outputs are never positive preferences;
- cap formula multiplicity and deduplicate reduced formulas;
- perform one DPO/ORPO-style continuation from the selected SFT checkpoint;
- retain a fixed original-data KL/NLL anchor;
- no iterative generate-select-train loop.

This stage is not allowed if Stage B shows no support-mass movement, because
the positive actions would not be reachable enough for preference training.

## 7. Stage D: Planner-only RL, last resort

RL is considered only after the single Stage C opportunity fails.

Use the best SFT checkpoint and a frozen formula-only reward:

- primary nonshortcut valid: `+1`;
- unary/all-metal shortcut: `0`;
- charge, Pauling or oxidation-missing failure: `-1`;
- parse/incomplete/forbidden output: `-1`;
- KL penalty to the SFT policy.

Do not reward Body, refiner, S.U.N., novelty or energy in this comp_valid
stage.  Freeze the number of rollouts, group size, KL coefficient, update
count and stop gates before execution.  Stop immediately on shortcut
inflation, top-1 collapse, element-coverage loss, or a new failure class.

Body-DLM RL cannot fix this problem because the formula is already frozen
before Body generation.

## 8. Checkpoint cleanup boundary

Remote deletion remains blocked until the user-maintained nested A800 SSH pane
is verifiably connected.  Once connected, first write a dry-run reference
graph and tombstone manifest.

Protect:

- H1 P0;
- selected P-control step 400;
- B0 and `model_494`;
- the local CrysLLMGen adapters and any exact paper LLaMA-2 adapter;
- all source manifests, terminal reports, ledgers, raw outputs and evaluator
  reports.

Payload-only deletion candidates:

- all P* checkpoints;
- P-control intermediates except step 400;
- failed ValidReplay and JointChem checkpoint payloads;
- B2 and no-promotion smoke payloads;
- B1 only after proving no remaining scientific-reference edge.

No deletion is inferred from a directory name.  Every removed payload requires
an exact path, bytes, identity/reference audit and reason in the tombstone
manifest.

## 9. Decision

Do not start RL now.  Do not repeat ValidReplay.  First establish whether the
public CrysLLMGen checkpoint/recipe was ever reproduced, then test a
CrysLLMGen-inspired repeated ionic-sequence plus same-head infill SFT under
strict raw-attempt gates.

The defensible diagnosis is:

> MP-20 is not broken; H1 currently combines an evaluator-misaligned corpus
> with a compact, causally backward SFT target.  The most valuable next
> intervention is a chemistry-first autoregressive representation trained by
> ordinary SFT, not a stronger inference mask or immediate RL.
