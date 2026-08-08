# Result: Planner SFT-v2

Status: `ENGINEERING_SMOKE_REPAIR_V4_PREPARING`

V3 smoke task `31036_0` failed `1:0` before the first forward because PEFT
loaded the candidate/reference copies of the protected P0 adapter at different
precision. Data passed and no optimizer step or scientific generation ran.
The V4 same-load-path repair is frozen; scientific results remain pending.
