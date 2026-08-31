# Fixed-256 evaluation adapter

These files are a local copy of the frozen H1-A2 V3 evaluation runtime. The
only protocol change is permitting an active denominator of 256 for the four
pre-registered Candidate-A repeats. Direct validity, reconstruction, novelty,
uniqueness, CHGNet relaxation, and all metric thresholds remain unchanged.

The original V3 runtime only accepted denominators 1,000 and 1,200, so job
34719 stopped before scientific evaluation. This adapter allows the intended
fixed-256 screen without padding or silently changing its denominator.

The query adapter additionally accepts a named environment variable so the MP
credential can remain in process memory; it is removed from the child process
environment before any request and is never written to the run directory.

## Future raw F/M reuse

The prospective evaluator can now split each stream's raw F/M union by the
lossless canonical `Structure.as_dict()` SHA-256 and relax each exact unique
structure once. Results are mapped back to every original attempt, so the
fixed denominator, row order, failures, unknown energies, novelty, uniqueness,
and output label schemas are unchanged. The per-cell audit records ordered
input/result identities and pair hit/miss counts.

If the raw cells contain `R_F + R_M` reconstructed occurrences and `U` exact
unique structures, CHGNet work falls by `(R_F + R_M - U) / (R_F + R_M)` while
the two F/M workers remain parallel. This is a computational reuse boundary,
not a scientific equivalence rule: near structures are misses, StructureMatcher
still serves only the frozen N/U calculation, and model494-refined cells never
enter the reuse path because their continuous CUDA refinement is nondeterministic.

After scientific completion, the evaluator records multiprocessing cleanup and
the Slurm wrapper grants each completed process group five minutes to exit. A
group is diagnosed and terminated only after its `_SUCCESS` or `report.json`
artifact exists; processes still doing science have no wall-clock kill timeout.
