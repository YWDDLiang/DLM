# Result: Planner SFT-v2

Status: `ENGINEERING_SOURCE_GATE_PATH_REPAIR_V5_PREPARING`

V3 smoke task `31036_0` failed `1:0` before the first forward because PEFT
loaded the candidate/reference copies of the protected P0 adapter at different
precision. Data passed and no optimizer step or scientific generation ran.
The V4 same-load-path repair source froze, but its source gate was not run
because a static check found a stale immutable V3 isolated-extraction path.
Two independent reviews approved a V5 path-only repair. Scientific results
remain pending.
