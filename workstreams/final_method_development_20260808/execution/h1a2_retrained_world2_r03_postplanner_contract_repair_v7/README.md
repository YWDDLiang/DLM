# H1-A2 retrained world2 post-planner contract repair V7

V5 job `31900` completed all five exact historical-world2 planner cohorts and
planner assembly, then failed before the first body model load. All four fresh
children raised the same `KeyError: 'adapter_file'`: V5's config omitted six
runtime identity fields required by the frozen V4 body runner. No V5 body,
refinement, or evaluation cell reached success.

This immutable continuation reuses the V5 planner outputs byte-for-byte and
does not contain planner launch or planner assembly scripts. It adds the
missing frozen body identities:

- adapter filename, byte count, and SHA-256;
- tokenizer vocabulary identity and tokenizer file SHA-256 values;
- tokenizer size.

Preparation verifies V5's terminal Slurm state, exact failure signature,
source manifest, planner terminal/deep-distribution/topology hashes, and all
five cohort hashes. V6's preparation was interrupted before submission at the
user's request because its added audit reread the 6.39 GB adapter. V7 preserves
that abort evidence and does not rehash either large model artifact. It checks
the current adapter/model494 paths and byte sizes against their previously
registered SHA-256 identities, while retaining the lightweight source,
cohort, tokenizer, and scientific-wrapper checks.

Exactly one non-array Slurm job requests at most four A800 GPUs and 32 CPUs.
It runs, in waves:

1. four fresh R03 D2-safe-axis + model494/refine800 cells;
2. the H1-A2 B0/D1 control alongside the single topology body realization;
3. four independent topology model494/refine800 processes;
4. generation assembly and freezing of nine post-model494 official-S inputs.

Pre-refine structures are intermediates and are not evaluated. No planner
resampling, retry, replacement, scientific repair, filter, rerank, training,
or RL is permitted. Together with V5, this uses two Slurm IDs and reserves one
final Slurm ID for the official clean S.U.N. evaluation.
