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
