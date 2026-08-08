# Result: Planner SFT-v2-C

Status: `ENGINEERING_SMOKE_V5_RUNNING_JOB_31064_1`

V3 smoke task `31036_1` failed `1:0` before the first forward with the same
PEFT candidate/reference precision asymmetry as SFT-v2. Curriculum data passed
and no optimizer step or scientific generation ran. The V4 same-load-path
repair source froze, but its gate was not run after a stale immutable V3
isolated-extraction path was found. Two independent reviews approved a V5
path-only repair. Scientific results remain pending.
V5 source/data-reuse gates passed and fresh smoke task `31064_1` is running;
training and raw generation are not submitted.
