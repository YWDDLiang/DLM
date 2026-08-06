# H1 no-charge ion-auxiliary SFT execution annex V1

Status: `LOCAL_EXECUTION_PACKAGE_AND_DUAL_EVALUATOR_AUDIT_PASS_REMOTE_TRAIN_DATA_TOKENIZER_GPU_PENDING`

Date: 2026-08-06

User direction: minimally modify the current H1 rich Planner by deleting the
generated `charge:` field, use explicit oxidation-state/ion supervision only
as same-head training auxiliaries, keep the Body-DLM and frozen refiner
unchanged, and carry a passing Planner through common Direct/CrysLLMGen and
S.U.N. evaluation.

This annex supersedes only the executable design in
`H1_COMP_VALID_ROOT_CAUSE_AND_SFT_PLAN_V2.md`. Historical CR-Plan, P*, B2,
safe-axis, cleanup, and evaluator evidence remains immutable.

## 1. Scientific question

Does deleting the causally backward self-reported charge label and adding
training-only ion arithmetic improve raw, all-attempt composition validity of
the existing formula-first H1 Planner without materially moving away from the
MP-20 distribution or changing the downstream DLM?

The primary contrast is `C1 - C0`:

- `P0`: frozen historical seven-line H1 Planner;
- `C0`: P0 continued with a six-line no-charge Plan and neutral atom/count
  auxiliaries;
- `C1`: the identical continuation, except that the matched auxiliaries carry
  explicit oxidation-state witnesses.

`P0` measures the combined effect of deleting `charge` plus continued SFT.
`C0` isolates that minimal schema/continuation effect. `C1-C0` isolates the
explicit ion-supervision effect.

## 2. Frozen inference representation

`C0` and `C1` generate exactly six lines:

```text
formula: Li2O
anion: oxide
lattice: cubic
spacegroup: sg_195_230
volume: volpa_016_020
end: plan
```

No `charge:` or `ions:` field is generated at inference. Formula parsing still
derives element counts and `N`; the evaluator derives charge taxonomy after
generation. It may not repair a formula, replace an element/count, select an
oxidation witness on behalf of the generator, retry, filter, rerank, or shrink
the raw denominator.

The compiled plan_state and Body prompt contain the same formula, anion,
lattice, space-group bucket, volume bin, element counts, and `N` fields used by
the frozen H1 Body. Only the removed self-report is absent.

## 3. Data and evaluator contracts

### 3.1 Source

- exact MP-20 train: 27,136 rows;
- exact MP-20 validation: 9,047 rows;
- train/validation/test material and reduced-formula leakage are reported;
- no generated evaluation output enters training or checkpoint selection.

### 3.2 Dual evaluator

The paper-comparable primary composition metric remains the frozen local
CrysLLMGen-compatible legacy evaluator. SMACT `4.0.0` is frozen as a secondary audit
and witness source with explicit ICSD24 filter, Pauling, alloy, metallicity,
and mixed-valence settings. Package version alone is not an evaluator
identity.

Exact SMACT identity:

- release: `4.0.0` (not a floating `>=4` dependency);
- official wheel SHA256:
  `e3eb968da92d47a8ef9a4af42af5589a6de61cccbca9d329937e1f4e402f0551`;
- release source commit: `c2b4d3ce6fa3a8c39fd21ab252934253dc66e131`;
- ICSD24: `include_zero=false`, `consensus=3`, `commonality=medium`;
- Pauling enabled, alloys enabled, metallicity shortcut disabled, mixed
  valence enabled;
- filtered map: 93 elements, 149 states, SHA256
  `9789ef90afc08faed4eb082bb762c22a4a7675dfc0ae9a21a6951d2f583efad6`;
- complete contract SHA256:
  `ad070f3ad8d025ec9d7918dc1aa84e3f8faaf4ee0a124fbe8602208d5609ca19`.

The de-novo positive pool is the stable intersection:

```text
legacy nonshortcut charge-neutral/Pauling-valid
AND
SMACT4 valid with a deterministic allowed witness
```

Unary, true all-metal, oxidation-missing, mixed-valence-only, and evaluator
flip strata remain explicit. They do not count as primary gain. SMACT4-only or
legacy-only MP-20 rows remain eligible for the full-MP20 conditional/KL
anchors, not unconditional positive formula targets.

### 3.3 Immutable task ledger

Each arm consumes exactly 3,200 task records, corresponding to 400 optimizer
updates at global batch 8. Source row, task role, infill cursor, witness,
ordering, and seed are stateless and identical across C0/C1.

| Fraction | Count | C0 | C1 |
|---|---:|---|---|
| no-charge primary Plan | 30% | stable-primary six-line Plan | identical |
| sequence-to-formula | 5% | repeated atom sequence -> formula | repeated ion sequence -> formula |
| infill | 5% | element/count infill | matched element/oxidation infill |
| full-MP20 conditional anchor | 40% | formula in input; nonformula fields in target | identical |
| conditional P0 KL/logit anchor | 20% | formula in input; frozen P0 distribution over nonformula answer only | identical |

No evaluator-invalid MP-20 formula is an unconditional answer target, including
the KL records. Both conditional record types put formula only in the prompt.
The SFT anchor supervises anion, lattice, space group, volume, and end marker;
the KL anchor regularizes those same nonformula answer positions against frozen
P0 and carries no generated `charge:` position.

Within the stable-primary pool, capped raking matches full MP-20 atom-count,
arity, broad element/anion family, lattice, space-group, and volume-bin
marginals wherever support exists. Missing support is reported, not fabricated
by uncontrolled repetition. Per-source-row exposure and exact/reduced formula
multiplicity are capped and reported.

## 4. Oxidation witness and auxiliary representation

Witnesses are generated offline and frozen before training. SMACT's search is
performed on reduced integer stoichiometry; the deterministic witness is then
expanded by the same gcd to the exact unreduced 1--20-atom MP-20 sequence.
Uniform witnesses
are primary. Mixed-valence witnesses are retained as a separate diagnostic
stratum and do not silently enter the legacy-primary positive set.

Machine-readable ions use existing signed codes:

```text
I=Li:QP01,Li:QP01,O:QM02
```

`QP##`, `QM##`, and `QZ00` mean positive, negative, and zero oxidation state.
The witness audit requires supported elements, 1--20 atoms, exact charge sum
zero, allowed states, deterministic ordering, endpoint agreement, and exact
ion/atom sequence -> formula round-trip.

BERTOS may be recorded as an agreement diagnostic. It cannot replace the
frozen evaluator, change a witness after generation, filter a sample, or act
as an inference-time model.

## 5. Training contract

Both arms start from the same immutable P0 adapter and use:

- same base model and tokenizer;
- same rank/modules/alpha/dropout as P0;
- BF16, batch 1, gradient accumulation 8;
- 400 updates, seed 26080617;
- LR `2e-6`, 25-step warmup, cosine decay, weight decay 0;
- maximum length chosen only from the pre-generation tokenizer audit;
- checkpoints at 100/200/300/400, with step 400 the fixed endpoint;
- no generated comp_valid, Body, Direct, energy, hull, novelty, or S.U.N.
  metric used for selection.

Answer supervision is token-level. Formula tokens in the primary Plan task
have weight 2.0; other answer tokens have weight 1.0. C0/C1 use identical
weight-mask shapes. P0 KL is computed on a formula-input-only conditional
prompt over the nonformula answer, using a frozen P0 adapter and a shared base
model; fixed coefficient `beta=0.05`.

Training stops before scientific sampling on NaN/Inf, OOM, task/row mismatch,
witness mismatch, answer-mask mismatch, source identity mismatch, or more than
1% degradation of the frozen full-MP20 anchor validation NLL.

## 6. Preflight gates

Before any `sbatch`:

1. all declared partitions are verified with `sinfo`;
2. local unit tests and isolated archive tests pass;
3. source, archive, data ledger, tokenizer, P0, SMACT4, legacy evaluator, and
   constraint-contract SHA identities are frozen;
4. all 3,200 records per arm and the validation records parse exactly;
5. C0/C1 source row/task/cursor/weight-mask identities are 100%;
6. all C1 ion witnesses round-trip and sum to zero;
7. prompt/answer token lengths fit without answer truncation;
8. the shared-base dual-adapter P0 KL path has exact initialization parity;
9. one minimal A800 forward/backward smoke has finite CE/KL/gradients and no
   downstream execution.

The maintained connection remains local `wq-starteam:1.0` -> outer host ->
user-maintained `ssha800:1.0`. No agent-created reconnect, replacement tmux,
or direct local-to-A800 SSH is allowed.

### 6.1 Local execution evidence (2026-08-06)

- six-line parser/Body/sampler/CR-Plan identity integration and auxiliary
  tests plus immutable execution-protocol tests: `101/101` pass;
- the initially convenient same-process label path was rejected: a frozen
  build now requires a read-only snapshot produced under exact SMACT `3.1.0`
  and refuses to recompute legacy labels under SMACT `4.0.0`;
- exact legacy snapshot contract SHA256:
  `96a4a4af2f58d25e80afdfadc0725d0160a2fb38198bccfdcc850522c6e802a7`;
- full local MP20 validation snapshot: `9,047` rows, legacy primary `2,264`;
  the SMACT4 stable uniform-primary intersection is `1,406`, mixed-valence
  diagnostic `95`, charge/Pauling failure `763`, and official/witness parity
  failures `0`;
- full local MP20 test snapshot: `9,046` rows, legacy primary `2,344`; the
  SMACT4 stable uniform-primary intersection is `1,463`, mixed-valence
  diagnostic `99`, charge/Pauling failure `781`, projection overflow `1`, and
  official/witness parity failures `0`;
- exact SMACT `4.0.0` runs in a separate isolated Python 3.12 target and never
  writes the paper-comparable legacy metric;
- task mix, ion/atom round-trip, zero charge, formula-input-only anchors, and
  no-charge answers all passed;
- the local immutable execution candidate is
  `execution/h1_nocharge_ion_aux_sft_v1`: initial submission ends at raw64,
  while raw256 has a distinct submission lock and requires a passing raw64
  terminal. Ledger64 SHA256 is `ea566afe7b0d39f480428fcced6da5a68e52391bc5279875b56992514bd7b2a3`;
  Ledger256 SHA256 is `a0aafedbf80968419c4f57758901d8da60e598c7110194c3579d7b6571883004`;
- the first local source freeze was not transferred: its inventory was valid,
  but isolated tests stopped at import because one historical read-only test
  helper was absent. The preserved V1 identities are source inventory
  `6737dab2…`, manifest `62e9e461…`, archive `0899e618…`;
- source-repair V2 added only that test helper. Its isolated archive passed
  exact per-file hashing, `101/101` tests, deferred-remote preflight, and all
  nine shell/Slurm syntax checks. Frozen V2 identities: source inventory
  `354308ae2fba693fdc5f7c93537266bd9812d0dcef3745dbf756111e0056c6fd`,
  source manifest
  `6c79d2027643df6e96e3568b8b3d6f67fe319b2136b9c8f8676b3236ae48c0d6`,
  archive
  `4c71caa6596163a77c2453948d6a386b5a084db66edf49b92e31a37e8a1bc619`;
- local repository has exact `val.csv` and `test.csv` but no `train.csv`, and
  the local runtime has no frozen Llama tokenizer/model. Therefore the real
  3,200/640 ledger, tokenizer audit, dual-adapter smoke, and GPU jobs remain
  remote preflight work; validation/test evidence cannot substitute for a
  train-split build. Remote submission additionally requires discovery of an
  already available Python runtime matching the exact SMACT4 contract; no
  environment installation is implicit in this authorization.

As of the final local check on 2026-08-06, Windows had no running
`vmmemWSL/wslhost`; the maintained `wq-starteam:1.0 -> ssha800:1.0` path was
therefore inaccessible. No WSL distribution, tmux, SSH connection, remote
environment, file transfer, or Slurm job was created by the agent.

## 7. Planner-only raw ladder

All ledgers are independent from training and from each other. Generation is
one sample per ordinal with no retry/replacement/repair/filter/rerank/fallback.
P0 uses its frozen seven-line prompt; C0/C1 share the exact six-line prompt.
Temperature/top-p/top-k/maximum length and stateless ordinal roles are common.

### 7.1 Paired 64

Primary gate `C1-C0`:

- raw legacy comp_valid gain at least `+3/64`;
- nonshortcut primary gain at least `+3/64`;
- parse and completion each lose at most one;
- unary and all-metal shortcuts do not increase;
- unique-formula rate and fixed-alphabet element coverage each lose at most
  2 percentage points;
- absolute mean-N drift at most 0.5;
- no new failure class or identity mismatch.

P0 is fully reported but is not used to rescue a failed C1-C0 gate.

### 7.2 Paired 256

Only a passing 64 proceeds. Primary gate `C1-C0`:

- raw legacy comp_valid gain at least `+8/256`;
- nonshortcut primary gain at least `+8/256`;
- gain versus P0 is nonnegative;
- paired bootstrap 95% interval, discordance table, and exact McNemar are
  reported;
- completion/parse, shortcut, mean N, arity, top-1 frequency, formula
  uniqueness, element coverage, family, and SMACT4 secondary safeguards pass;
- no new failure class.

A failed Planner gate is the scientific terminal for this SFT candidate. It is
not repaired by a smaller denominator, new seed, checkpoint reselection,
threshold change, RL, or downstream selection.

## 8. Frozen downstream evaluation

Only a passing C1 Planner enters downstream. DLM weights remain unchanged:

- frozen B0/D1 exact `7+4N` Body contract;
- same frozen Body checkpoint;
- same `model_494`, exact 800 reverse steps, batch 1;
- same refiner settings and Direct evaluator;
- no DLM training, safe-axis, B2, RL, checkpoint selection, or altered mask.

P0/C0/C1 use the same downstream ordinal ledger and Body/refiner/evaluator
contract. Every Planner or Body failure remains in the raw denominator.

The paired-256 downstream report includes:

- Planner parse, completion, legacy/SMACT4 comp_valid, primary and shortcut
  taxonomy;
- Body generation completion, composition validity, structure validity, and
  joint validity;
- CrysLLMGen metrics: `comp_valid`, `struct_valid`, `valid`,
  `wdist_density`, `wdist_num_elems`, `cov_recall`, `cov_precision`;
- raw all-attempt and completed/survivor-denominator secondary tables;
- unique, novel, novel-unique, top-1 frequency, element/arity/family coverage,
  atom-count and density distributions;
- paired discordance, McNemar, bootstrap intervals, and failure transitions;
- common-evaluator comparisons with frozen H1-A2 and the available local
  CrysLLMGen checkpoint, without claiming exact reproduction of the paper
  checkpoint unless identity parity exists.

## 9. S.U.N. contract

S.U.N. is evaluation-only and cannot tune the Planner checkpoint, task mix,
loss weights, DLM, or evaluator.

- one common treatment/control union snapshot;
- strict and meta S.U.N. rates/counts on raw attempts;
- completed/survivor secondary rates;
- coverage, finite/unknown counts, residual unknown handling, novelty and
  uniqueness eligibility;
- exact common-snapshot identity for P0/C0/C1;
- credentials, if required, only through a user-provided runtime 0600 secret
  carrier that is unlinked after reading and never written to source, command
  line, log, manifest, or TODO.

If common snapshot coverage is incomplete, the terminal is
`HOLD_EVALUATOR_INCOMPLETE`; missing evaluations cannot be scored selectively
or used to tune the model.

Downstream success requires C1-C0 raw joint gain at least `+5/256`, structure
validity loss no worse than 1 percentage point, strict S.U.N. delta nonnegative,
meta S.U.N. delta nonnegative, complete common-snapshot accounting, and no new
failure class. A negative meta delta is a scientific stop.

## 10. Compute and mutation boundary

- local/CPU audit: no model loading unless explicitly required by tokenizer
  or isolated forward tests;
- training: at most two one-A800 400-update arms;
- Planner screens: P0/C0/C1 at 64, then 256 only after pass;
- downstream: paired 256 only after Planner pass;
- no automatic independent 1,000, RL, DLM training, checkpoint reselection,
  formal promotion, or unrelated experiment;
- only new immutable source/run roots and new jobs are touched;
- no running job may be cancelled or modified;
- historical run roots, source manifests, ledgers, terminal reports, and
  protected checkpoints remain unchanged.

## 11. Terminal deliverable

The final report must state one of:

- `SCIENTIFIC_PASS_ELIGIBLE_FOR_SEPARATE_CONFIRMATION`;
- `SCIENTIFIC_STOP_RETAIN_FROZEN_H1`;
- `ENGINEERING_STOP`;
- `HOLD_EVALUATOR_INCOMPLETE`.

It includes every metric in Sections 8--9, source/job/SHA identities,
all-attempt denominators, failure attribution, MP-20 distribution drift,
defensible claims, limitations, and the single highest-value next action.
