# H1-A2 retrained planner -> R03 -> clean S.U.N. recovery contract repair V5

V5 is an immutable scheduler repair after V4 requested 64 CPUs and was
cancelled while still pending under the user's new 32-CPU requirement. V5
preserves V4's four-A800 waves and scientific contract, changing only the
parent CPU allocation and its audited runtime bound.

V4 was itself an immutable scheduler repair after the submitted V3 requested eight
A800s and was cancelled while still pending under the user's new four-A800
maximum. V4 preserves V3's scientific contract and changes only allocation
width and execution waves.

V3 had already repaired the V2 interface error proving that the frozen
historical sampler (`d38743...`) does not expose the later `--seed-mode` or
`--formula-constraint-mode` CLI flags. The historical behavior is intrinsic:
the sampler seeds rank `r` with `base_seed + r`, batches four examples per
rank, and has no formula-constraint implementation. V3 therefore invokes the
old sampler with its actual historical interface and audits that newer
provenance fields are absent from raw records. This changes no sampling
semantics.

All five raw1200 planner cohorts, planner assembly, four fresh R03 runs, the
topology-matched R03 panel, the H1-A2 B0/D1 control, four independent
refiners, and terminal assembly are packed inside one four-A800 Slurm
allocation with 32 CPUs. Planner cohorts run in waves `2 + 2 + 1`; generation runs as four
fresh R03 cells, then the two-GPU H1-A2 control alongside the one-GPU topology
body, then four topology refiners. At no point can more than four A800s be
visible, and no wave requests more than 32 CPU threads. No pre-refine metric is computed.

Scientific contract:

- Four independent planner cohorts. Each first generates exactly 1,200 raw
  attempts, matching historical H1-A2, and only then freezes 256 attempts.
- Every cohort uses the historical H1-A2 topology: two A800 ranks, batch size
  four per rank, stateful seed `base_seed + rank`, temperature 0.9, top-p 0.95,
  top-k 50, max-new-tokens 96, `h1_rich_plan_v1`, and no sample id.
- The first 256 records of the rank-concatenated 1,200-record file are frozen
  without parse-success filtering. Under the historical world-size-two merge,
  these are rank-0 sample indices `0, 2, ..., 510`; directly requesting 256
  would instead mix 128 records from each rank and is therefore forbidden.
- Before R03 starts, all four cohorts receive a deep distribution audit against
  the byte-frozen historical seed-17/world2 cohort.  The audit records complete
  element-presence, stoichiometric-element, chemsys, formula, atom-count,
  element-count, charge, anion, lattice, space-group, volume and family
  distributions, plus TV/Jensen-Shannon/Hellinger distances and the largest
  per-attempt element/chemsys shifts.  Its pooled 1,024 view is descriptive.
- R03 safe-axis body generation feeds model494 refine800. Only the resulting
  `post_model494` structures receive Direct, CrysLLMGen, and clean-official
  S.U.N. evaluation. Pre-refine structures are intermediate inputs and are not
  scored. No retry, replacement, repair, filtering, or reranking is used.
- The same frozen body/refiner seed ledger is reused across all four cohorts.
- Before submission, the refiner contract audit rehashes `model_494` and proves
  that the historical-best and current wrappers use the same frozen
  CrysLLMGen kernel: 1,000 scheduler timesteps, exactly 800 reverse updates
  (`range(800, 0, -1)`), one evaluation, batch size one, and the same 256-row
  refiner seed vector.
- The historical best four repeats reused one frozen body/proposal-graph panel
  and the same seed vector in four independent A800 processes.  They measured
  CUDA process-realization variability, whereas these four cohorts additionally
  measure planner-cohort variability.
- A strict topology panel uses the seed-17/world2 planner cohort, one R03
  body/proposal realization, and four independent one-A800 refine800 processes
  with identical body, proposal graphs, and seed vector.
- A controlled H1-A2 DLM comparator reuses that seed-17 planner cohort with the
  historical B0/R5-C checkpoint and D1 exact-plan schedule, then applies the
  same model494 refine800 and post-only evaluation.
- Official MP querying is a later fail-closed stage after generation assembly;
  credentials are never part of this source bundle or a Slurm environment.

Fresh cohort seeds are frozen as 52021, 62023, 72031, and 82037.  They are
distinct from every cohort already evaluated in the old-checkpoint replay.
