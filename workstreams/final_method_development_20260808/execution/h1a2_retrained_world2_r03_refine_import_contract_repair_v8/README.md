# H1-A2 retrained R03 refine-import continuation V8

V5 job `31900` froze the five historical-world2 planner cohorts. V7 job
`31965` then completed all four fresh R03 body256 outputs, with succeeded
counts `[246, 242, 250, 245]`, before every child failed at the first
model494 import. No V7 refinement or post-model494 output was created.

The engineering failure was deterministic: `body_refine.sbatch` retained the
R03D body-only `PYTHONPATH` when it invoked the frozen V4 `refine1000.py`, so
Python could not import `scripts.refine_dlm_with_crysllmgen` from the R03E
runtime. V8 fixes only that import boundary. It copies the four complete V7
body directories byte-for-byte and sets the refiner path to
`SOURCE:R03E_RUNTIME:R03E:R03D:PROJECT` before model494 starts. Fresh R03 body
generation is not rerun.

One non-array Slurm job requests four A800 GPUs and 32 CPUs. It runs:

1. four model494/refine800 continuations over the frozen V7 R03 bodies;
2. the not-yet-run H1-A2 B0/D1 control beside one topology body realization;
3. four independent topology model494/refine800 processes;
4. generation assembly for nine post-model494 official-S inputs.

Preparation verifies jobs `31900` and `31965`, all V5 planner/cohort evidence,
the four V7 body reports and file sets, and the exact V7 import failure. It
does not reread either large model artifact for hashing; registered identity,
path, and byte size remain the large-artifact contract.

Pre-refine structures are not evaluated. Planner resampling, retry,
replacement, scientific repair, filtering, reranking, training, and RL are
forbidden.
