# H1 chemistry-first SFT-v2 immutable execution protocol

Identity: `h1_chemistry_first_sft_v2_v1`.

This package executes the two preregistered Planner candidates `sft_v2` and
`sft_v2_c`. Both start from the protected P0 adapter and consume exactly the
same record multiset once. They differ only in frozen order: hash shuffle
versus the registered chemistry-first curriculum. The generated Plan remains
the existing six-line no-charge representation. Oxidation witnesses are
training-only and cannot appear as an unconditional invalid formula target.

The data stage first exports a read-only legacy SMACT 3.1 snapshot, then uses
the exact SMACT 4.0.0 ICSD24 contract to identify POS and build both ledgers.
The build fails closed on source counts, evaluator identity, witness parity,
split leakage, multiset/order identity, weight masks, or tokenizer truncation.
Exact SMACT 4.0 runs through a portable CPython 3.12.13 bundle containing 54
SHA-locked Linux wheels. The bundle is installed with `--no-index --no-deps`
inside a new run-local directory, with user-site imports disabled, and is
published atomically only after `pip check`, exact contract, and Transformers
tokenizer probes pass. It never mutates a shared Conda environment.

Training uses batch 1, accumulation 8, LR 2e-6, zero weight decay, cosine
schedule, the derived warmup, and exactly one complete ledger epoch. The last
partial accumulation group is divided by its actual microbatch count. Only the
derived fixed endpoint is saved. Full validation-anchor NLL is measured before
and after training; degradation above 1% is a scientific stop for that
candidate, while the other candidate continues.

Raw64 compares each candidate with one common P0 realization on the same
stateless ordinal ledger. Raw256 is a separate submission and may include only
candidates whose raw64 terminal passes every gate. Every raw attempt remains in
the denominator. There is no retry, replacement, repair, filter, rerank,
best-of-n, Body generation, refiner, Direct structure evaluation, S.U.N.,
checkpoint reselection, downstream submission, or RL in this package.

All remote paths must be new. The source inventory, archive, P0 adapter,
runtime contracts, MP20 counts, ledgers, partitions, and submission records are
verified before the first `sbatch`. Scientific failures return completed
terminal evidence; engineering failures fail closed and require a new
immutable repair version.

The one-time local construction and offline probe of the portable runtime
bundle is the explicit exception authorized on 2026-08-08. After that freeze,
all project programs, tests, data construction, training, generation, and
evaluation run only on A800 through the existing `ssha800` or `ssha800_2`
tmux sessions. The local machine remains the source-editing, evidence, and Git
control plane only.
