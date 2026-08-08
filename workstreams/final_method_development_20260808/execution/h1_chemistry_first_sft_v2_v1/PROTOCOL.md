# H1 chemistry-first SFT-v2 immutable execution protocol

Identity: `h1_chemistry_first_sft_v2_smact_split_v2`.

This package executes the two preregistered Planner candidates `sft_v2` and
`sft_v2_c`. Both start from the protected P0 adapter and consume exactly the
same record multiset once. They differ only in frozen order: hash shuffle
versus the registered chemistry-first curriculum. The generated Plan remains
the existing six-line no-charge representation. Oxidation witnesses are
training-only and cannot appear as an unconditional invalid formula target.

The data path is split across a hard evaluator firewall. A800 exports a
read-only MP20 snapshot under its existing SMACT 3.1 runtime and does not
install, import, or execute SMACT 4.0. The exact SMACT 4.0.0 ICSD24 witness
ledger is produced only on the local machine. Its manifest binds every source
row to the frozen source-inventory SHA, parent snapshot SHA, split, ordinal,
material identity, formula, legacy verdict, exact witness and contract SHA.
A800 consumes that immutable ledger with JSON/hash checks only. Missing,
duplicate, extra, stale or
mismatched rows are engineering failures. The resulting POS intersection is
therefore exact without running SMACT 4.0 on A800.

Training uses batch 1, accumulation 8, LR 2e-6, zero weight decay, cosine
schedule, the derived warmup, and exactly one complete ledger epoch. The last
partial accumulation group is divided by its actual microbatch count. Only the
derived fixed endpoint is saved. Full validation-anchor NLL is measured before
and after training; degradation above 1% is a scientific stop for that
candidate, while the other candidate continues.

Raw64 compares each candidate with one common P0 realization on the same
stateless ordinal ledger. Generation stops before assembly. The complete raw
files are transferred to the local machine, audited with exact SMACT 4.0, and
returned as a sealed per-arm bundle. A800 verifies each report against the
current raw-file SHA, source inventory, science ledger, exact denominator and
contract SHA before its SMACT 3.1 gate assembly. Raw256 repeats the same split
and may include only candidates whose raw64 terminal passes every gate. Every
raw attempt remains in the denominator. There is no retry, replacement,
repair, filter, rerank, best-of-n, Body generation, refiner, Direct structure
evaluation, S.U.N., checkpoint reselection, downstream submission, or RL in
this package.

All remote paths must be new. The source inventory, archive, P0 adapter,
runtime contracts, MP20 counts, ledgers, partitions, and submission records are
verified before the first `sbatch`. Scientific failures return completed
terminal evidence; engineering failures fail closed and require a new
immutable repair version.

The local-machine exception is restricted to the exact SMACT 4.0 witness and
raw secondary-audit producers. All model code, source tests, tokenizer checks,
data assembly, training, generation and primary SMACT 3.1 evaluation run on
A800 through the existing `ssha800` or `ssha800_2` tmux sessions. No exact
SMACT 4.0 executable path exists in any A800 submission, and the retired
portable-runtime builders are excluded from this source snapshot.

Submission is deliberately split at every cross-machine boundary. In the
current V6 repair, the sealed V3 snapshot/data are reused byte-for-byte and the
old `submit_snapshot_once.sh` / `submit_once.sh` entrypoints are disabled.
`submit_identity_probe_once.sh` authorizes only the real-P0 identity probe;
`submit_identity_repair_smoke_once.sh` can submit the two-candidate smoke only
after that probe passes. `submit_training64_once.sh` submits only fixed-endpoint
training and raw64 generation. `submit_assemble64_once.sh` is allowed only
after the sealed local raw64 audit returns. Raw256 generation and assembly use
two further separately locked submissions. Thus no job crosses a local SMACT4
boundary automatically, and no scientific job is queued before independently
validated A800 probe and smoke evidence has passed.

## V4 smoke identity repair

The V3 engineering smoke stopped before its first model forward because PEFT
0.16 loaded the trainable candidate P0 adapter in FP32 but the frozen reference
copy in the BF16 base-model dtype. All 448 tensor keys matched, no unrelated
parameter was trainable, and no optimizer step or scientific generation ran.
V4 changes only the adapter construction path: both copies are loaded from the
same protected FP32 P0 payload through the same PEFT autocast path, and every
reference parameter is then frozen before forward or optimizer construction.
The sealed V3 data and legacy snapshot are reused byte-for-byte under a
dedicated reuse manifest; data, task order, prompts, optimizer, seeds, ledgers,
models, evaluators, and gates are unchanged. V4 must pass a fresh dual-arm GPU
smoke before `submit_training64_once.sh` is eligible.

## V5 source-gate path repair

V4 source freezing completed, but a static check before its source gate found
that `audit_source_on_a800.sh` still targeted the existing immutable V3
isolated-extraction directory. The gate was not run and no marker or job was
created. V5 changes only active transfer, staging, freeze, run, log and
isolated-extraction paths. Isolated extraction is now scoped under the V5 run
root. The V4 adapter-load repair and every model, data, task, optimizer, seed,
ledger, evaluator and gate byte remain unchanged. Existing V3/V4 evidence is
never deleted or reused. V5 must pass the same reduced A800 source gate and a
fresh dual-arm GPU smoke before training is eligible; SMACT4 is not executed
on A800.

## V6 exact protected-P0 identity-copy repair

V5 reached the two-candidate A800 smoke and failed before the first model
forward or optimizer step. Candidate and reference exposed the same 448 LoRA
keys and no unrelated trainables, but every value differed by the PEFT 0.16
second-adapter BF16-load rounding signature. The frozen V5 failure report and
both independent reviews bind this diagnosis. The exact gate is not relaxed.

V6 first attests the trainable candidate against the protected on-disk FP32 P0
weight and config SHA. It then loads an independently stored reference adapter,
copies candidate values into that storage with in-place tensor copy, freezes the
reference, and requires source/candidate/reference byte identity, FP32 finite
values, non-overlapping storage, candidate-only trainables, and the expected
active adapter after device and checkpointing setup. Assignment or storage
aliasing is forbidden.

Before smoke, one new real-P0 A800 probe loads the frozen 8B model and protected
adapter but performs no forward, optimizer construction, training, generation,
or SMACT4 work. Its immutable report is independently validated. Each fresh
smoke then additionally requires exact candidate/reference logits on one fixed
validation record, finite candidate-only backward gradients, unchanged adapter
value hashes, and no optimizer step. Smoke markers alone cannot authorize
training: the science submission re-parses all probe and smoke reports and
records their admission SHAs. Data, record order, prompts, optimizer, model,
seeds, ledgers, evaluator contracts, and scientific gates are unchanged.
