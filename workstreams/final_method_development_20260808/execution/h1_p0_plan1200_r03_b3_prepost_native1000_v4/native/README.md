# P0 Plan1200 R03/B3 CrysLLMGen-native post-refine 1000 supplement

This package supplements, but does not replace, the preregistered all-attempt
V3 panel in
`20260811_h1_p0_plan1200_r03_b3_prepost_native1000_cohort_contract_repair_v4`.

The upstream `reference/crysllmgen/crysllmgen_sample.py` increments its
`num_samples` counter only after the LLM output parses through `process_one`.
Consequently an upstream run with `--num_samples 1000` sends exactly 1,000
successfully reconstructed LLM candidates to the diffusion refiner; failed LLM
draws occur before that denominator.  The V3 panel instead freezes 1,000 body
attempts and retains body failures in its denominator.  Both views are useful,
but they answer different questions.

This supplement reproduces the upstream sampling contract without rerunning
work that V3 has already completed:

1. For each repeat, freeze all parse-success P0 plans from the already terminal
   raw-1,200 planner draw in planner ordinal order.  The first 1,000 rows are
   byte-checked against the V3 cohort; remaining rows are the frozen reserve.
2. Reuse the V3 R03/B3 body and model_494 results for the first 1,000 plan
   candidates.  Generate each reserve plan once, with the same D2 safe-axis
   decoder and a deterministic extension of the frozen seed namespace.
3. Within each arm and repeat, select the first 1,000 body-success candidates
   by frozen candidate order.  This outcome-conditioned completion is exactly
   the semantic difference from the all-attempt panel and is reported
   explicitly; no individual plan is retried or replaced with another draw.
4. Reuse model_494 outputs for selected candidates from the V3 prefix.  Refine
   only selected reserve candidates, once each, with 800 diffusion steps.  A
   refiner failure is an engineering failure: it is not replaced or retried.
5. Require exactly 1,000 refined structures for every arm/repeat, then run the
   complete CrysLLMGen and detailed S.U.N. evaluators.  CrysLLMGen and the
   conservative S.U.N. view use all 1,000 refined structures; the exact legacy
   S.U.N. reconstructed denominator is retained as the paper-compatible
   headline and is shown alongside all-1,000 rates.

The R03 and B3 supplement arrays are separate GPU jobs.  The original V3
all-attempt results remain the inferentially cleaner paired comparison.  This
native supplement is the direct upstream CrysLLMGen-denominator comparison;
because arm-specific body failures can lead to different selected plan sets,
cross-arm pooled values are descriptive unless candidate identity is shared.

There is no training, RL, same-plan retry, stochastic replacement, repair,
filtering, or reranking.  GPU work uses only the `gpu` partition and requires
one A800 with SMACT 3.1.
