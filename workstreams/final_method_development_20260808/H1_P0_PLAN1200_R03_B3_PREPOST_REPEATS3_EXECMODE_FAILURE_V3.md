# P0 Plan1200 × R03/B3 pre/post evaluation: terminal failure V3

Updated: 2026-08-11 (Asia/Shanghai)

## Outcome

The requested three-repeat experiment is terminal as an engineering failure.
All three P0 planner batches completed and produced sufficient parse-success
plans, but every R03 and B3 body task stopped at ordinal 0 before model
generation. The separately requested CrysLLMGen-native supplement—accumulate
1,000 body-success candidates and pass all 1,000 through diffusion
refine—therefore also stopped before reserve generation.

No pre-refine or post-refine CrysLLMGen/S.U.N. metric exists for V3. This is
not scientific evidence for or against P0, R03, B3, model_494, CrysLLMGen, or
S.U.N.

## Frozen protocols

V3 froze two complementary denominator contracts:

1. **All-attempt primary:** each arm × repeat must execute exactly 1,000
   terminal body/DLM attempts on ordinals 0..999. Failed generation attempts
   remain in the denominator. The same 1,000 ordinals are evaluated before
   and after model_494.
2. **CrysLLMGen-native supplement:** traverse each frozen planner candidate
   pool once, retain the first 1,000 successful `process_one` body outputs in
   planner order, and pass all selected 1,000 through model_494. There is no
   retry, replacement, repair, filter, or rerank of a failed candidate.

The native supplement follows the audited upstream behavior in
`reference/crysllmgen/crysllmgen_sample.py`: the collection index advances by
`len(data_dicts)`, and `SampleDataset(collected_data)` refines those collected
successes. The audited upstream file SHA256 is
`6c641e2d873184d0301d824c256cf27b07c1e59cf6c547d38162ee49866fc0d9`.

## Frozen execution identity

| field | value |
|---|---|
| run root | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3` |
| V3 source commit | `6b9053ee876244b3bd0d211f25f7f09d6f912671` |
| V3 transfer archive | `85e10e9f629b86b31d668fa94023290b97c436861b1e2df1ab4c556930b05b1c` |
| body source manifest | `080db87fc12319b02000e121b306b7d26eed194b26afe2921276a1647ccc7ed8` |
| planner array / assembly | `31565` / `31566` |
| main R03 / B3 arrays | `31569` / `31570` |
| main assembly | `31571` |
| native source commit | `f1dfad5c7c83f11e9ddb6ccad30c6cd42f7a6802` |
| native source manifest | `d0b1b37381062e19b365381e64ef740d56c67969414531f1716afb2bb553d999` |
| native R03 / B3 arrays | `31576` / `31577` |
| native assembly | `31578` |

V1 and V2 remain separately sealed and were not modified or rerun.

## Inputs and infrastructure that did complete

Planner array `31565` and assembler `31566` completed `0:0`. The three raw
1,200 batches produced 1,189, 1,193, and 1,194 parse-success plans. For each
repeat, the first 1,000 parse successes were frozen; the same repeat cohort
was shared by R03 and B3, while the three repeat cohorts remained distinct.

The first-1,000 union MP cache completed with 2,464 populated rows and SHA256
`bf0dc8edd2f38f121286adbc97ff3f889b3c520135b29492a56ba562c86eaade`.
The native candidate-pool extension completed with 2,865 populated rows and
SHA256
`bc622ae450c54d480a74b9b376f292c700d8e70db85073a447497521246b2d37`.
It resolved 401 additional chemical systems with six bounded workers, at
most 10 HTTP requests/s, and zero transport retries. The one-time credential
carrier was destroyed, no credential was serialized, and no MP query ran in
Slurm.

These completed infrastructure stages do not create a scientific result
because no body sample was generated.

## Terminal Slurm state

| route | jobs | state | exit | boundary |
|---|---|---|---|---|
| main R03, repeats 0..2 | `31569`, `31572`, `31573` | `FAILED` | `1:0` | body input contract, ordinal 0 |
| main B3, repeats 0..2 | `31570`, `31574`, `31575` | `FAILED` | `1:0` | body input contract, ordinal 0 |
| main assembly | `31571` | `FAILED` | `3:0` | fail-closed on missing engineering successes |
| native R03, repeats 0..2 | `31576`, `31579`, `31580` | `FAILED` | `1:0` | required main `_SUCCESS` absent |
| native B3, repeats 0..2 | `31577`, `31581`, `31582` | `FAILED` | `1:0` | required main `_SUCCESS` absent |
| native assembly | `31578` | `FAILED` | `3:0` | fail-closed on changed terminal evidence |

The native tasks never entered reserve generation or model_494 refinement.

## Exact root cause

The frozen cohort rows contain `parsed_plan` and `plan_state`, but no
top-level `parsed` key. A read-only audit of repeat 0, ordinal 0 found:

| invariant | observed |
|---|---|
| `parsed` key present | `false` |
| `parsed` value | `null` |
| `raw_rich_seven_line_forwarded` | `false` |
| `canonical_charge_bucket_visible` | `true` |
| body prompt contract | `historical_r5c_plan_state_json_exact_length` |

`body_source/run_body_safeaxis1000.py` nevertheless requires
`row.get("parsed") is True` and raises
`ValueError: frozen cohort contract changed at ordinal 0` otherwise. All six
main tasks failed at that same check.

The preflight checked repeat identity, prompt forwarding, charge visibility,
prompt contract, and charge presence, but did not check the top-level
`parsed` field. It therefore reported `status=pass` for a schema that the
runtime rejected. The defect is the mismatch between the cohort producer,
the body consumer, and preflight coverage; it is not a model or evaluator
failure.

## Scientific availability

| requested stage | R03 | B3 |
|---|---|---|
| 3 × 1,000 all-attempt body generation | unavailable | unavailable |
| pre-refine CrysLLMGen metrics | unavailable | unavailable |
| pre-refine complete S.U.N. metrics | unavailable | unavailable |
| paired model_494 refine for all attempts | not started | not started |
| post-refine CrysLLMGen metrics | unavailable | unavailable |
| post-refine complete S.U.N. metrics | unavailable | unavailable |
| native first-1,000 body successes | not selected | not selected |
| native full-1,000 diffusion refine | not started | not started |
| three-repeat McNemar/bootstrap inference | unavailable | unavailable |

There was no sample retry, replacement, repair, result filtering, reranking,
training, or RL.

## Preserved terminal evidence

| artifact | SHA256 |
|---|---|
| main `terminal_report.json` | `5006facf9a981f56a5a3aff8f5d886f0cb7ab2bd1b61ac22e3d1aedda446d2c9` |
| main `RESULTS_COMPLETE.md` | `1f58f81b35a75cff22410ce01210a6446252d7727154ba70299de20daf104348` |
| native `terminal_report.json` | `18256dd2514364c7781f6843eaa9eb56bbd13f49b8382ffacfd70f1bd24ad982` |
| native `CRYSLLMGEN_NATIVE1000_RESULTS_COMPLETE.md` | `beb2741b4b2b8b362093dc39d3e62b31afa27138ee68c595917547d2e4e2d190` |
| full failure-evidence bundle | `5f73eca2744fea97a103c2ca3e6d5d080263d9ee8f87a32bd56476a3cc0a1698` |
| returned terminal-artifact bundle | `f03307e19449a2068972a572ab053626b325137b83ccfd9860bc80f089f11a99` |

The ignored local evidence bundles are under
`.artifacts/h1_plan1200_r03_b3_v3_failure/`; A800, relay, and local hashes
match.

## Decision

V3 and native supplement V1 are immutable terminal failures. They must not
be repaired, requeued, or resubmitted in place. The requested full-1,000
post-diffusion CrysLLMGen basis remains unfulfilled.

Continuing requires explicit authorization for a new immutable repair. The
minimal engineering change is to make the frozen cohort schema and body
consumer agree on parsed validity and to add an exact producer/consumer
schema assertion to preflight. Reuse of the already frozen planner cohorts
or completed MP caches also requires explicit authorization and a recorded
scientific identity argument. No automatic V4 submission is authorized.
