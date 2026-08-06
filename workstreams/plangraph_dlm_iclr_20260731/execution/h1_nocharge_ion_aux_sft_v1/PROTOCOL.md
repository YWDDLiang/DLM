# H1 no-charge matched ion-auxiliary SFT V1

Status: local execution candidate.  Submission is forbidden until the
maintained remote connection, exact dual evaluator runtimes, immutable
source/archive identity, MP20 train split, tokenizer audit, and `sinfo` checks
all pass.

## Question and contrast

Keep the formula-first H1 Planner and frozen Body-DLM, delete only the
generated `charge:` line, and test whether training-only oxidation arithmetic
raises strict raw composition validity.  The causal contrast is C1-C0:

- P0: historical seven-line rich Planner;
- C0: six-line no-charge continuation with neutral atom/count auxiliaries;
- C1: the identical continuation with explicit oxidation-state auxiliaries.

C0 and C1 share source rows, task roles, cursors, ordering, optimizer geometry,
seed, prompts outside the matched auxiliary payload, formula weights, and
checkpoint endpoint.  Only auxiliary chemistry semantics differ.

## Data and evaluator firewall

The legacy paper-comparable metric and upgraded label audit may never run in
the same evaluator process:

1. SMACT 3.1.0 plus the frozen `composition_validity.py` exports a read-only
   MP20 metadata/legacy-label snapshot.
2. Exact SMACT 4.0.0 reads that snapshot and appends deterministic ICSD24
   oxidation witnesses.  It may not recompute or overwrite legacy labels.

The positive de-novo pool is legacy nonshortcut primary intersected with
SMACT4 uniform-primary.  Mixed-valence-only, unary, all-metal, oxidation
missing, evaluator flips, and failures remain explicit strata.

Each arm has exactly 3,200 train and 640 validation records.  The train mix is
30% direct no-charge Plan, 5% sequence-to-formula, 5% matched infill, 40%
full-MP20 conditional anchor, and 20% conditional P0 KL anchor.  Every full
MP20 formula is input-only.  Invalid formulas are never unconditional targets.

## Training

- common Meta-Llama-3-8B and P0 LoRA initialization;
- BF16, batch 1, gradient accumulation 8, exactly 400 updates;
- seed 26080617, LR 2e-6, warmup 25, cosine schedule, weight decay 0;
- formula/chemistry payload answer tokens weight 2, others weight 1;
- conditional nonformula P0 forward KL beta 0.05;
- checkpoints 100/200/300/400; checkpoint 400 is fixed;
- no generated metric or downstream result selects a checkpoint;
- full-MP20 conditional-anchor validation NLL may degrade at most 1% relative
  to the frozen P0 adapter.

NaN/Inf, OOM, missing gradients, source/ledger/tokenizer/adapter mismatch, or
anchor-NLL failure stops before scientific sampling.

## Planner ladder

Independent stateless raw ledgers are used for 64 and 256.  P0/C0/C1 use one
sample per ordinal, temperature 0.9, top-p 0.95, top-k 50, maximum 96 new
tokens, N 1..20, and no retry/replacement/repair/filter/rerank/fallback.

The 64 C1-C0 gate requires legacy comp_valid and legacy nonshortcut-primary
gain each >=3/64, parse/completion loss <=1, no shortcut inflation, <=2pp
unique/element-coverage loss, |mean-N drift| <=0.5, nonnegative SMACT4
secondary deltas, and no new generation failure class.

The 256 gate requires both primary gains >=8/256, C1 legacy metrics not below
P0, parse/completion loss <=2 attempts, the same safeguards, and <=2pp top-1
formula-frequency increase.  Paired discordance, exact McNemar, and a frozen
10,000-draw paired bootstrap interval are always reported.

## Downstream

Only a passing C1 checkpoint may enter a separately frozen downstream run.
Body/refiner stay unchanged: B0/D1 exact 7+4N, model_494, exact 800 reverse
steps, batch 1, and the common Direct/CrysLLMGen evaluator.  Final S.U.N. uses
one frozen treatment/control union snapshot with complete coverage/unknown
accounting.  Missing credentials or evaluator coverage yields HOLD, never an
imputed score.

No automatic RL, Body training, checkpoint reselection, CR-Plan combination,
repair, filtering, denominator change, or threshold change is authorized.

## Execution DAG

The initial one-shot DAG is deliberately capped at raw64:

```text
data (normal; SMACT3 snapshot -> SMACT4 witnesses -> tokenizer audit)
  -> smoke[2] (gpu; C0/C1 finite forward+backward, no optimizer step)
  -> train[2] (gpu; C0/C1 fixed 400 updates)
  -> planner64[3] (gpu; P0/C0/C1 independent raw attempts)
  -> assemble64 (normal; separate SMACT4 audit + legacy primary gate)
```

`planner256` has a separate submit lock and fresh `sinfo` snapshot.  It
requires an immutable raw64 terminal with status `planner_gate_pass`; it is
not part of the initial DAG.  No downstream job is present in either submit
script.

The two evaluator commands receive distinct absolute Python executables.
Preflight rejects a shared executable, a non-3.1 legacy runtime, a non-4.0
secondary runtime, or a changed SMACT4 contract.  It never installs or edits a
remote environment.
