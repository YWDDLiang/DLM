# Result: Planner SFT-v2-C

Status: `ENGINEERING_HOLD_AFTER_V5_SMOKE_FAILURE_JOB_31064_1`

V3 smoke task `31036_1` failed `1:0` before the first forward with the same
PEFT candidate/reference precision asymmetry as SFT-v2. Curriculum data passed
and no optimizer step or scientific generation ran. The V4 same-load-path
repair source froze, but its gate was not run after a stale immutable V3
isolated-extraction path was found. Two independent reviews approved a V5
path-only repair. Scientific results remain pending.
V5 source/data-reuse gates passed, but smoke task `31064_1` failed `1:0` after
`00:15:02` at the same pre-forward identity gate. Its identity report is
byte-identical to V3 (SHA `ac04b540...`) with all 448 tensor values different
and maximum absolute difference `6.103515625e-05`. Independent propose/red-team
review and a focused runtime probe are mandatory before any new repair.
Training and raw generation are not submitted.
