# Immutable V4: Plan1200 R03/B3 all-attempt + CrysLLMGen-native 1000

This package is a minimal, fail-closed repair of the frozen V3 execution.
It imports the three byte-identical P0 planner cohorts, their complete native
candidate pools, and both completed MP caches by explicit SHA-256 contract.
It does not regenerate plans or query MP.

The only runtime repair is the input-row contract: the producer's actual
parse-success evidence is `parsed_plan` plus `plan_state`; the V3 cohort and
native-reserve consumers incorrectly required a non-existent top-level
`parsed` field. Each consumer now shares its producer-schema validator between
preflight and runtime, with dependency-light self-tests for both row types.

Two separately labelled views are produced:

1. `V4_ALL_ATTEMPT_1000`: the V3 scientific protocol, with exactly 1,000 body
   attempts per arm/repeat and both pre/post-model494 evaluations.
2. `V4_CRYSLLMGEN_NATIVE_SUCCESS1000`: the upstream CrysLLMGen-style first
   1,000 body successes per arm/repeat, all passed through model494 before the
   post-refine evaluation.

No same-plan retry, replacement, repair, filter, rerank, training, checkpoint
selection, promotion, or RL is permitted. GPU jobs use `gpu`, never
`gpu_long`; runtime requires one A800 and SMACT 3.1.
