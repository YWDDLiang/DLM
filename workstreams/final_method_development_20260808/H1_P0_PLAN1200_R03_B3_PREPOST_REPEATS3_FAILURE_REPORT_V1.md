# P0 plan1200 × R03/B3 pre/post evaluation: engineering-failure report V1

Updated: 2026-08-11 (Asia/Shanghai)

## Outcome

The requested three-batch experiment did not reach scientific sampling. The
three-task P0 planner array failed at the same frozen-source import boundary,
and its dependent assembler sealed a fail-closed terminal report. No R03 or
B3 body job was submitted, so there are no pre-refine or post-refine
CrysLLMGen/S.U.N. metrics from this run.

This is an engineering failure, not a scientific result for P0, R03, B3, the
refiner, CrysLLMGen, or S.U.N.

## Frozen execution identity

| field | value |
|---|---|
| run root | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260810_h1_p0_plan1200_r03_b3_prepost_repeats3_v1` |
| planner array | `31561`, `gpu`, `0-2%3` |
| planner internal jobs | `31561`, `31563`, `31564` |
| planner assembly | `31562`, `normal`, `afterany:31561` |
| repeat seeds | `17029`, `27183`, `31415` |
| intended raw attempts | `1200` per repeat |
| intended frozen cohort | first `1000` parse successes per repeat |
| planner source manifest | `f945a8a847dfecb84c47d789a0dd71181fad7ea6a6cd92a020194863fe922887` |
| source commits | `fbe94e6`, `be42548` |

## Terminal evidence

All three planner tasks passed the resource/runtime gates and reported one
`NVIDIA A800-SXM4-80GB`, PyTorch `2.4.0+cu121`, and SMACT `3.1.0`. Each then
entered `planner_sample1200` and failed identically before emitting a plan:

```text
ModuleNotFoundError: No module named 'scripts.sample_llada_dynamic_crystals'
```

The failing import is line 63 of
`scripts/sample_llama_h1_formula_plans.py`. The referenced module file is in
the frozen source inventory, but the archive did not package the local
`scripts` directory as a regular Python package. The runtime therefore could
not resolve the intended local module. This is a source-packaging/importability
defect.

| repeat | Slurm state | exit | runtime gate | plan rows | frozen cohort |
|---:|---|---:|---|---:|---:|
| 0 | `FAILED` | `1:0` | passed | 0 | 0 |
| 1 | `FAILED` | `1:0` | passed | 0 | 0 |
| 2 | `FAILED` | `1:0` | passed | 0 | 0 |

The planner tasks each ran for `00:14:17`. Assembly job `31562` produced
`status=failed`, an empty repeat list, and
`three_independent_plan_batches=false`; it exited fail-closed. Its terminal
record also confirms `automatic_body_submission=false`,
`automatic_training=false`, and `automatic_rl=false`.

## Downstream status

| requested stage | status |
|---|---|
| three independent P0 plan batches | not produced |
| first-1000 frozen cohorts | not produced |
| separate R03 GPU jobs | not submitted |
| separate B3 GPU jobs | not submitted |
| pre-refine CrysLLMGen metrics | unavailable |
| pre-refine complete S.U.N. metrics | unavailable |
| model_494 refine800 | not started |
| post-refine CrysLLMGen metrics | unavailable |
| post-refine complete S.U.N. metrics | unavailable |
| paired three-repeat statistics | unavailable |

There was no retry, replacement, repair, filter, rerank, training, or RL.
The locally prepared body package was not transferred or submitted.

## Preserved evidence

The minimal evidence bundle contains the three planner stdout/stderr pairs,
the planner assembly logs, Slurm accounting snapshot, submission record,
per-repeat exit markers, source identity records, and terminal report.

| item | value |
|---|---|
| local path | `execution/h1_p0_plan1200_r03_b3_prepost_repeats3_v1/evidence/planner_failure_evidence_31561_31562.tar.gz` |
| bytes | `3377` |
| SHA256 | `ecadf983ac9f1676d637c7c239299729f39cf0e667c0823962c981452efa2da3` |
| transfer verification | A800, relay, and local hashes identical |

## Required decision

The frozen V1 route is terminal and must not be repaired or resubmitted in
place. Continuing requires explicit authorization for a new immutable V2.
The minimal repair is to package `scripts` as an unambiguous local Python
package and add an isolated import preflight before SBatch. The scientific
contract—three independent P0 batches, shared within-repeat plans, separate
R03/B3 jobs, 1000-attempt denominators, complete pre/post metrics, no retry,
and no RL—would remain unchanged.
