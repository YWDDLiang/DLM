# DLM experiment lineage for the low-resource D3PO decision

Date: 2026-08-30

This is a read-only synthesis of the complete H1-A2/R5C/R03 and later
stability-program lineage. It prevents the next experiment from treating SGTC
L7 in isolation or repeating an earlier failure under a new name.

## Early representation and system evidence

| Period | Evidence | What carries forward |
|---|---|---|
| R5C fixed-slot preflight | The archived exact-length run explicitly reports `answer_token_count=107` under the old `fixed_slot` representation. The production successor uses variable `r5_exact_dynamic_v1`. | Do not return to a padded 107-token target. D3PO uses dynamic `7 + 4N`, masks only legal geometry positions, and normalizes every score by corruption probability/token count so N does not become a preference shortcut. |
| R5C exact dynamic | Requested1200 sampling parsed all Plans and produced 1167 graphs (`97.25%`). The refined1000 view reached about `90.7%` composition validity and `99.8%` structural validity. MP20/R5C conditioning could produce stable structures, but the replay-style result had low de-novo S.U.N. retention/novelty. | Exact length and structured conditioning are useful; replay/rich teacher conditioning is not a de-novo stability claim. Existing model494-refined outcomes may be preference teachers, not inference-time answers. |
| H1-B formula-only | Body success was 100%, but adjusted Strict/Meta S.U.N. were only `5.54/43.13%`; `81.57%` of lattices had all angles at 90 degrees and `53.33%` repeated two lengths. | Composition-only ordinary CE collapses toward frequent geometry templates. Keep C³FD minimal typed fields and let the DLM learn structural modes; do not confuse syntactic success with realization quality. |
| Historical H1-A2 | Frozen1000 compatibility was `94/474` Strict/Meta S.U.N., but exact replay fell to `8.58/46.08%` and continuous repeats to `7.63/45.47%`. A single seed cannot establish robustness. | Every new stability claim needs at least two paired generation seeds. Historical aggregate `105/488` remains a headline, not training microdata or a single-seed control. |
| R03 D1/D2 | D1 was `99/523` and Safe-axis D2 `117/496` over 1024. The apparent Strict gain traded Meta and was not reproduced: exact R03 was `8.42/46.58%`, continuous R03 `7.47/45.26%`. Four D2 repeats reused one Plan cohort and were process realizations, not independent Planner seeds. | Preserve the safe legality invariant (lattice, all X, all Y, all Z), but do not revive R03/Safe-axis as a claimed stability method. Separate process repeats, DLM seeds, and Planner seeds. |

## Later controlled experiments

| Experiment | Result type | Main result | Constraint on D3PO |
|---|---|---|---|
| Counterfactual grounding | Mechanism positive, endpoint negative | Four repeats gave only `+2/1024` Strict and reduced Meta; fresh requested1000 changed Strict `89→86`, Meta `487→467`. Intermediate checkpoint effects were non-monotonic. | A changed NLL/margin is insufficient. No checkpoint selection, and final evidence must be paired energy/hull plus S.U.N. |
| Sufficient CE training | Formal negative | Epoch2→3 improved body `985→992`, while Strict `81→79` and Meta `489→477`. | No more ordinary-CE epochs or checkpoint sweeps. |
| Conditioning/schedule factorial | Formal negative with legality lesson | Hard-axis did not improve stability; joint XYZ caused large body/Direct losses from duplicate coordinates. | Keep exact/full-axis legality. Do not reopen joint/mixed/atom-major schedules. |
| model494 tau800 | Strong absolute positive, attribution caveat | Full-axis raw→refined Direct `188→457`, Strict `10→48`, Meta `66→230`; novelty and stable-to-S.U.N. retention fell. | Preference labels must be post-refiner. Report raw and refined separately because model494 can erase DLM differences. |
| tau sweep | Formal short-tau negative | tau0/200/500/800 Strict `10/29/39/48`, Meta `66/171/222/230`; tau900 added only three Strict in a partial run, tau1000 was incomplete/OOM. | Freeze tau800. No tau search; tau900 is at most a later sensitivity. |
| Same-Plan preference data | Data construction narrowly missed old gate | Extreme-only 8-stream data produced `95/27` train/validation pairs at a hard 0.06 eV/atom gap, one below the old train minimum. | Preference learning was never tested. Rebuild all non-tie pairs with composition-normalized soft margins instead of one extreme pair per Plan. |
| Noisy critic assets | Data success, critic not run | 1752 eligible outcomes over 222 Plans; MatterSim dependency failure was engineering-only. | Reuse these raw body texts and post-refiner CHGNet labels. Do not reopen MatterSim or train a separate critic first. |
| CTV token Q | Formal scientific negative | Centered Spearman `0.0353`, pair AUC `0.5053`, continuation agreement `0.4915`, coverage `0.0781`. | Single-token value guidance is closed. Preference must act on complete body sequences with common corruption. |
| SGTC L6/L7 | Formal scientific negative | L7 base/G0/G1 Strict `60/55/53`, Meta `412/421/417`; G1 energy/hull moved slightly in the adverse direction. | Positive-only strict-stable continuation does not learn a fixed-composition energy boundary. Reference is the minimal base, never G1. |
| CCFD | Internal mechanism positive, external endpoint negative | Assignment rose to 99.15%, independent comp-valid was essentially unchanged and seed directions disagreed. | Internal preference accuracy cannot be the endpoint; independent Direct/energy/hull evaluation is mandatory. |
| C³FD-v2.5 | Strong formal positive | `2000/2000` independent comp-valid, NU `1756`, zero semantic dead ends, and improved N/arity/family distance. | Planner/composition is solved for this program. Freeze C³FD and spend the remaining budget only on DLM realization. |

## Corrected mistakes and engineering lessons

- CTV's first pooled absolute-energy correlation was invalid because composition
  baselines dominated. Only within-state/composition-centered statistics count.
- Pair, mask, continuation, and repeat rows are not independent samples. Each
  composition receives total statistical/training weight one.
- The first combined D3PO pair build also selected the lowest relaxation energy
  when identical texts repeated. The corrected v3 build averages repeat labels
  and then removes strict PBC-equivalent structures: `335` exact and `164`
  physical duplicates were removed from `4205` outcomes, with no CIF parse
  failures. This prevents refiner randomness and serialization differences from
  becoming pseudo-preferences.
- The old energy-pair split was formula/Plan-based rather than strict chemsys
  isolation. Rebuild a salted chemsys split before training.
- H1-A2/R03 historical high points are seed/process fragile. Two paired DLM seeds
  are required even in a fixed-256 pilot.
- Past pre-science failures included missing Python entry points, missing conda
  variables, denominator-schema rejection, duplicate-attempt readers, and OOM.
  New jobs need local/remote tests, adapter-equality checks, exact denominator
  checks, and fail-closed success markers before scientific execution.
- Current model494 sampling passes a clean proposal directly as an `x_tau`-like
  state although training used forward corruption. This remains an interface
  hypothesis, but it is not mixed into the first D3PO causal test.

## Reusable assets

- minimal `ctv_minimal_base step696` policy/reference and dynamic exact-axis
  tokenizer/executor;
- eight-stream raw body texts and 1752 post-refiner labels;
- completed L7 bodies/labels as optional training data only if L7 is retired and
  a fresh outcome-blind cohort is frozen;
- unused C³FD seed17 attempts for a fresh distribution-preserving test cohort;
- model494 tau800, Direct/CHGNet/official evaluators, and unknown-as-missing
  policy;
- L6 two-seed matched outputs for a zero-new-query retrospective pilot if they
  remain excluded from training.

## Routes that remain closed

No more ordinary CE, strict-positive-only continuation, single-token Q heads,
simple scalar geometry rank heads, formula-only templates, mixed XYZ, rich-Plan
answer leakage, tau/temperature/checkpoint grids, completed-sample reranking,
MatterSim-first dependency work, or uncorrected TraceRL/naive GRPO.

The surviving low-resource hypothesis is a reference-anchored,
margin-aware, composition-normalized, full-sequence **shared-noise masked-D3PO**
objective with a winner denoising anchor. It is a direct continuation of the
successful exact-length/safe-axis infrastructure and the previously untested
Same-Plan preference idea, while respecting every formal negative above.
