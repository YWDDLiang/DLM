# H1 P0 Plan1200 × R03/B3 V2 pre-prepare failure

- Observed at: `2026-08-10T17:27:59Z`
- Frozen source commit: `957b3a19c28c8f4e9d241684947c36bffdab4626`
- Transfer archive SHA-256: `2880df59f27b5f3f5b3f04dca0bf1bdd8a741bf261c395e3c0e121feb255b154`
- Intended run root: `20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_import_repair_v2`
- Terminal classification: pre-prepare engineering failure, fail-closed

The transferred archive passed byte identity, file-set, Python-cache, private-key,
repository-package import, P0 adapter identity, and SMACT 3.1 preflights.  The V2
prepare entry was then addressed by direct execution while its Git archive mode
was `0644`.  The kernel rejected the entry before executing any script line:

```text
bash: .../h1_p0_plan1200_r03_b3_prepost_repeats3_v2/prepare_planner_once.sh: Permission denied
```

No V2 run root was created, no preparation marker exists, and no Slurm job was
submitted.  V2 is sealed and is not retried.  V3 changes only the executable
entry contract and run identity; planner/body/refiner configurations, three
planner seeds, paired seed namespace, evaluation denominators, and statistical
protocol remain frozen.
