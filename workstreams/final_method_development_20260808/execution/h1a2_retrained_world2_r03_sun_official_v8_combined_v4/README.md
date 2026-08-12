# Retrained H1-A2 post-refine official S.U.N. recovery from V8

This immutable bundle consumes the completed V8 generation/refinement run and
evaluates only its nine `post_model494` cells. No pre-refine metric is run.

It consumes nine `post_model494` / refine800 cells from
`20260812_h1a2_retrained_world2_r03_refine_import_contract_repair_v8`:

- four independent fresh-planner R03 cohorts;
- four independent one-A800 refinement processes sharing the same seed17
  planner cohort, R03 body/proposal graphs, and refiner seed vector;
- one controlled historical H1-A2 B0/D1 DLM generation using that same seed17
  planner cohort, followed by model494 refine800.

Pre-refine structures are intermediate inputs only. They are not passed to
Direct, CrysLLMGen, S.U.N., meta-S.U.N., or any table in this evaluator.

The GPU preliminary phase freezes reconstruction, novelty, uniqueness, and
CHGNet-relaxed energy. Its legacy-cache hull labels are diagnostic only. The
final phase byte-verifies the proven official evaluator and reuses its
`load_resolved`, `load_unresolved`, `make_phase_diagram`, `exact_hull`,
`evaluate_cell`, and `exact_mcnemar` definitions unchanged. Cell layout,
post-only summaries, and report rendering are the only adapter changes.

Official references use
`MPRester.get_entries_in_chemsys(..., compatible_only=True,
thermo_types=[GGA_GGA+U])`. Historical/August polluted cache rows are never
used. Unsupported systems are explicit `hull_unknown`; fixed-all and
skip-unknown denominators are both reported.

The terminal report includes every frozen CrysLLMGen count, all post-refine
Direct fields, separate strict S.U.N. and meta-S.U.N. tables, hull coverage,
50,000-draw panel-level hierarchical rate bootstraps, the four-process
historical topology recovery comparison, and per-process exact McNemar tests
between the shared H1-A2 B0/D1 control and R03 D2 topology outputs. Because the
same H1-A2 control vector is reused four times, no pooled independent-block
claim is made for that comparison.

Before Slurm submission, the login node audits the V8 all-reconstructed
chemical-system inventory against the clean official cache and queries only
genuinely missing systems. Missing official references remain explicit
`hull_unknown`. The credential is read from a mode-0600 one-time carrier and
destroyed before the first MP request.

The sole Slurm job uses at most four A800 GPUs and 32 CPUs. It runs the nine
frozen U/N + CHGNet cells in waves `0-3`, `4-7`, and `8`, adopts the precompleted
official cache, and produces the final official report in the same allocation.
No array, generation, refinement, training, retry, repair, filtering, reranking,
or RL job is submitted.
