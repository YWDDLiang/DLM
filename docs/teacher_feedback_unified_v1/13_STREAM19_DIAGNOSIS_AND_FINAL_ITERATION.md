# Stream19 diagnosis and the single final iteration

## Decision

Stream19 does not authorize paper1000. The failure is classified primarily as
`UNDERTRAINED`, with local transaction scope retained as a secondary limitation.
The one permitted final iteration is therefore one additional pass over the
same frozen 4,104 K10 groups, warm-started from the completed job-39799 policy.
No data, candidate, value, loss, learning rate, temperature, decoding rule or
refinement rule changes.

The new final endpoint is preregistered as Planner seed 26, stream 21, DLM seed
95117, refiner seed 105117 and tau800. It reports raw and refined results over
all 256 attempts. Exact success remains Strict/Meta 26/128; the preregistered
paper1000 launch rule remains 23/125 in the same endpoint.

## What stream19 established

The complete C3FD -> Planner-Llama -> SPAD DLM interface is not the failure:

- Planner composition validity: 256/256.
- DLM decoded, parsed, Plan-matched and graph-valid: 256/256.
- Raw and tau800 reconstructed: 256/256.
- Raw novel-unique: 256/256; refined novel-unique: 229/256.
- Fresh official MP coverage: 248/256, with eight explicit unresolved systems.

After the fresh fixed-thermo query, the outcome is:

| endpoint | Strict stable | Meta stable | Strict S.U.N. | Meta S.U.N. |
| --- | ---: | ---: | ---: | ---: |
| raw DLM | 12/256 | 61/256 | 12/256 (4.69%) | 61/256 (23.83%) |
| model494 tau800 | 19/256 | 133/256 | 14/256 (5.47%) | 107/256 (41.80%) |

The refiner helps rather than erases the aggregate stability signal: Meta
stable rises by 72 cells and Meta S.U.N. by 46. Its remaining limitation is
that it loses 27 novel-unique representatives and adds only two Strict S.U.N.
cells. `REFINER_WASHOUT` is therefore not the primary diagnosis.

`EVALUATION_COVERAGE` also cannot explain the miss. Even if all eight unresolved
rows were favorable, refined would be bounded by Strict 22 and Meta 115, still
below the fixed near-line gate 23/125.

## Why the teacher is adequate

The frozen K10 action set contains a large, reachable physical control signal:

| state type | groups | positive headroom | median K10 headroom | >10 meV | >50 meV | median selected-action E0 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cell | 2,052 | 1,733 | 149.51 meV/atom | 1,544 | 1,242 | -68.37 meV/atom |
| XYZ | 2,051 | 1,502 | 135.72 meV/atom | 1,335 | 1,157 | -1.10 meV/atom |

Thus the teacher is not dominated by ties or unreachable actions. Cell actions
also improve the unrelaxed energy strongly, while XYZ benefits are primarily
short-basin rather than instantaneous. This supports retaining the same K10
scientific object for the final iteration.

## Why one more pass is the least confounded final change

Job 39799 completed exactly one pass: 4,104 posterior exposures, 3,889
informative and 215 retained zero-information exposures, interleaved with 4,104
clean anchors. The gradient probe was well balanced: scaled posterior norm
32.55 versus clean norm 39.37, with median cosine 0.034 and no gradient conflict.
All 2,736 updates were finite and the KL budget remained at 0.05 nat.

However, one exposure per independent state did not make the sampled posterior
loss trend downward: first-quarter mean 0.01253 and last-quarter mean 0.01297.
This is evidence that the optimization stopped after coverage, not after
convergence. The raw endpoint nevertheless moved into a better stability regime
than the 128-state pilot, so the learned direction is not null. A second pass is
therefore better supported than adding an untrained test-time controller.

The second pass loads job39799 as both initial policy and trust-region reference,
uses the same LR 5e-6, warmup 128, clean interleave and K10 labels, and saves only
its terminal checkpoint. This is a training-sufficiency test, not a sweep.

## Evidence and recovery record

Job39803 completed both raw and refined CHGNet/newness computations but its old
cache finalizer stopped on an unseen chemsys. A fresh fixed-thermo query was then
run for the frozen 253 chemsys: 245 resolved and eight are explicit unknowns.
The credential entered through no-echo stdin, was popped before the first query,
was never serialized, and no query process or shell copy remained afterward.
The wrapper failure remains recorded; the recovered science result is
`HELDOUT_STREAM19_FINAL_FRESH.json` with `_SCIENCE_SUCCESS`.

## Final stream21 outcome

The registered warm-start iteration completed as job39805: 2,736 finite
updates, 1,368 clean-CE and 1,368 transaction-posterior updates, all 4,104
groups exposed exactly once, 3,889 informative exposures and only the terminal
policy saved. Planner seed26 jobs39806/39807 froze a new outcome-blind 256-row
cohort. Job39810 then decoded, parsed, Plan-matched, constructed and refined all
256 rows without retry or replacement; job39814 evaluated raw first and tau800
second without Direct.

A fresh official query covered 252/256 rows after mapping (247/251 unique
chemsys resolved; four explicit unknowns). The final fixed-denominator result is:

| endpoint | Reconstructed | Novel-unique | Strict stable | Meta stable | Strict S.U.N. | Meta S.U.N. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw DLM | 256/256 | 254/256 | 5/256 | 51/256 | 5/256 (1.95%) | 49/256 (19.14%) |
| model494 tau800 | 256/256 | 230/256 | 20/256 | 148/256 | 14/256 (5.47%) | 123/256 (48.05%) |

Neither the exact 26/128 target nor the preregistered 23/125 paper1000 gate is
met. Four unknown rows cannot rescue the Strict gate even under an all-favorable
bound. The final interpretation therefore supersedes the provisional
`UNDERTRAINED` diagnosis: one extra full exposure increases refined Meta S.U.N.
but does not improve Strict S.U.N., and raw stability regresses on the new
prospective stream. The remaining limitation is `LOCAL_CONTROL_INSUFFICIENT`:
the K10 teacher has real local headroom, yet moving probability among one cell
or XYZ transaction at a time does not reliably move complete generated crystals
into the strict low-energy tail. Per the registered contract, no further method
iteration or paper1000 scale-up is launched.
