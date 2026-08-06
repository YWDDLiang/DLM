# DLM: plan-guided discrete diffusion for crystal generation

This repository contains the complete experimental record for a staged crystal
generator built from a learned Planner, a discrete diffusion body model, and a
continuous diffusion refiner. It versions source code, immutable contracts,
ledgers, tests, terminal decisions, and reproduction documents; large model
weights and run artifacts are referenced by path and SHA rather than committed.

## Current result in one minute

As of 2026-08-06:

- **Best validated end-to-end system:** H1-A2 epoch-2 Planner + R5-C
  exact-length body DLM + CrysLLMGen `model_494` refine800.
- **Most successful recent model change:** R03 safe-axis decoding for the body
  DLM. It passed the registered 32/64/256 body gates, eliminated the
  mixed-axis duplicate-coordinate collapse, and improved pooled completion and
  raw joint validity.
- **Important boundary:** R03 is a successful DLM component, not a promoted
  end-to-end replacement. Its registered four-repeat evaluation improved joint
  and strict S.U.N. counts but reduced meta S.U.N., so the scientific route
  stopped.
- **Current bottleneck:** formula/composition generation in the Planner. The
  next authorized experiment removes the causally useless generated `charge:`
  line and adds training-only ion/count auxiliary supervision while keeping
  the DLM and refiner frozen. It has passed local source/preflight tests but has
  not yet produced generation metrics.

### Frozen headline metrics

All values below use raw attempts as the denominator.

| System / experiment | Attempts | `comp_valid` | `struct_valid` | Joint valid | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|---:|---:|---:|---:|
| Local historical CrysLLMGen reference | 1,000 | 89.2% | 99.9% | 89.1% | 9.0% | 46.1% |
| **H1-A2 epoch-2 end-to-end fallback** | **1,000** | **87.8%** | **99.9%** | **87.7%** | **9.4%** | **47.4%** |
| R03 D1 control, four repeats | 1,024 | 848/1,024 (82.81%) | 982/1,024 (95.90%) | 846/1,024 (82.62%) | 99/1,024 (9.67%) | 523/1,024 (51.07%) |
| **R03 safe-axis, four repeats** | **1,024** | **852/1,024 (83.20%)** | **989/1,024 (96.58%)** | **851/1,024 (83.11%)** | **117/1,024 (11.43%)** | **496/1,024 (48.44%)** |

The R03 S.U.N. row is the later completed-snapshot lower-bound report
(`R03G`), in which 72 unresolved hulls are explicitly false-scored. The
preregistered frozen-cache decision (`R03E`) was candidate minus control
`+5/1024` joint, `+4/1024` strict S.U.N., and `-28/1024` meta S.U.N.; that
negative meta result remains the promotion decision.

The published CrysLLMGen composition-validity number is **93.55% on strict raw
attempts**, not a S.U.N. survivor-denominator metric. The local 89.2% reference
is not claimed to be an exact reproduction of the paper checkpoint and recipe.

## What the successful stack actually uses

```text
materials goal
  -> Meta-Llama-3-8B + H1-A2 epoch-2 Planner LoRA
       formula + anion/lattice/space-group/volume Plan
  -> LLaDA-8B-Instruct + R5-C exact-length body adapter
       discrete lattice/species/coordinates, exactly 7 + 4N tokens
  -> safe-axis constrained decoding
       lattice -> grouped X -> grouped Y -> grouped Z
  -> CrysLLMGen model_494 continuous refiner
       exactly 800 reverse steps, batch 1
  -> frozen Direct and common-snapshot S.U.N. evaluators
```

| Component | Concrete implementation | What it contributes / improves |
|---|---|---|
| Planner | Meta-Llama-3-8B + H1-A2 epoch-2 LoRA | Converts an unconstrained generation request into a model-generated rich Plan: formula, anion family, lattice family, space-group bucket, and volume bin |
| Body DLM | LLaDA-8B-Instruct + R5-C adapter | Realizes the sampled Plan as an exact-length discrete crystal body; count/element prefill and schema masks keep output aligned with the Plan |
| Body safety constraints | Duplicate-coordinate and lattice-volume masks | Fail closed on malformed geometry while preserving one raw attempt per ordinal |
| R03 decoding | Safe-axis reveal order | Restores the causal prerequisite of the duplicate-coordinate mask; improves completion without changing weights, prompts, Plans, seeds, or denominators |
| Continuous refiner | CrysLLMGen `model_494`, exact 800 steps | Converts the discrete proposal to continuous lattice and coordinates; conditional structure validity is approximately 99.7–99.8% |
| Evaluation | Frozen CrysLLMGen-compatible Direct + novelty/uniqueness/CHGNet/MP-hull | Reports raw composition, structure, joint validity, strict S.U.N., meta S.U.N., coverage, and unknown accounting |

Compared with the local historical CrysLLMGen reference, the frozen H1-A2
system gained `+0.4` percentage point strict S.U.N. and `+1.3` points meta
S.U.N., while losing `1.4` points in composition and joint validity. This is
why H1-A2 remains the end-to-end fallback and Planner chemistry is now the
primary optimization target.

Full identities and metric provenance are in
[H1_FALLBACK_MANIFEST.md](workstreams/plangraph_dlm_iclr_20260731/H1_FALLBACK_MANIFEST.md).

## Recent experiment ledger: success, stop, and pending work

| Experiment | Intervention | Main observed result | Decision |
|---|---|---|---|
| R03B/R03C/R03D safe-axis | Change only DLM coordinate reveal order | `31/32`, `63/64`, and `248/256` completion; zero duplicate-coordinate failures | **Component gate pass** |
| R03E/R03G/R03H | Four fixed refine800 repeats plus common-snapshot attribution | Raw joint `+5/1024`; completed-snapshot strict `+18/1024`; meta `-27/1024` | **Mechanism success, end-to-end scientific stop** |
| CR-Plan E1 | Exact full-prefix charge reachability, real-model physical probe | 100% trace/scalar parity; full/off median `1.468x`; all `14/14` applicable attempts affected before terminal | **Engineering feasibility pass only** |
| CR-Plan four-arm 512 | Off / grammar / terminal-only / full-prefix formula support | Full-prefix gained `+6/512` raw composition but only `+1/512` nonshortcut/primary and added five shortcuts | **Scientific stop; no downstream** |
| No-charge C0/C1 SFT | Delete generated `charge:`; matched neutral vs oxidation auxiliaries; `2x` chemistry-token loss | Source-repair V2 passed isolated `101/101` tests and preflight; no generated metrics yet | **Authorized and locally frozen; training pending** |

## Recent successful experiment: R03 safe-axis DLM decoding

### What changed

R03 kept all of the following fixed:

- H1-A2 epoch-2 Planner and sampled Plans;
- R5-C body-DLM weights and tokenizer;
- exact `7 + 4N` output length;
- count/element prefill and all schema constraints;
- per-ordinal seeds and raw denominators;
- CrysLLMGen `model_494` and exact 800-step refinement;
- Direct and S.U.N. evaluators.

Only the order in which masked body-coordinate tokens were revealed changed.

The rejected mixed-axis PlanGraph schedule could commit a `Z` coordinate before
the corresponding `X/Y` pair was known. That violated the causal precondition
of the duplicate-coordinate mask and created 18 new duplicate-coordinate
failures in the first 32 attempts.

The accepted safe-axis schedule keeps PlanGraph grouping but restores the
required axis order:

```text
count/elements prefilled
  -> lattice
  -> grouped X blocks
  -> grouped Y blocks
  -> grouped Z blocks
```

This guarantees `z_before_xy_count = 0` without changing the model,
composition, or sampling ledger.

### Body-generation ladder

| Stage | D1 control | Safe-axis candidate | Candidate delta | Duplicate failures |
|---|---:|---:|---:|---:|
| paired 32 | 31/32 | 31/32 | 0 | 0 / 0 |
| paired 64 | 61/64 | 63/64 | +2 | 0 / 0 |
| paired 256 | 246/256 | 248/256 | +2 | 0 / 0 |

The paired-256 completion gain is `+0.78125` percentage point. Its McNemar
`p=0.7266` supports a safety/non-inferiority conclusion and a positive point
estimate, not a claim of statistically established completion superiority.

### End-to-end Direct results

Four frozen-refiner process repeats produced 1,024 raw attempts per arm:

| Arm | Generation complete | Composition valid | Structure valid | Joint valid |
|---|---:|---:|---:|---:|
| D1 control | 984 | 848 | 982 | 846 |
| **Safe-axis** | **992** | **852** | **989** | **851** |
| Delta | **+8** | **+4** | **+7** | **+5** |

Among refine800-complete structures, conditional structure validity remained
near saturation: `99.7967%` for D1 and `99.6976%` for safe-axis. The apparent
raw composition gain comes from recovering more body-complete attempts; the
formula itself is fixed by the Planner and is not changed by the DLM schedule.

### S.U.N. result and claim boundary

The completed-snapshot lower-bound analysis retained 72 unresolved hulls as
false and reported:

| Arm | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|
| D1 control | 99/1024 = 9.67% | 523/1024 = 51.07% |
| **Safe-axis** | **117/1024 = 11.43%** | **496/1024 = 48.44%** |
| Delta | **+18 = +1.76 pp** | **-27 = -2.64 pp** |

Strict deltas were positive in all four repeats. The paired attribution found
`+16` strict cases from finite hull-threshold crossings and `+2` from
novel-unique eligibility. The meta loss came from finite `0.1 eV/atom`
threshold crossings, not from the residual unknowns.

Therefore the defensible conclusion is:

> Exact-length body diffusion is reliable, and decoding order is a real causal
> design variable. Safe-axis decoding removes the mixed-axis duplicate collapse
> and improves completion, raw joint validity, and strict S.U.N. lower bound,
> but it does not improve broad metastable stability.

The complete experiment ladder, identities, confidence intervals, and replay
commands are in
[H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md](workstreams/plangraph_dlm_iclr_20260731/H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md).

## What did not become the main method

### P* and B2 factorial experiment

The later Planner/body factorial experiment did not pass its preregistered
Phase-2 gates. On 256 raw attempts, the frozen baseline arm `M00=P0+B0`
produced:

```text
generation/composition/structure/joint = 243/203/241/201
strict/meta S.U.N.                    = 13/58
```

The trained body factor B2 sharply reduced generation and joint completion.
Neither P* nor B2 improved strict S.U.N.; those candidates were stopped and
were not promoted over H1-A2/R5-C.

### CR-Plan constrained decoding

CR-Plan tested charge-aware legal support at formula decoding time. Its
full-prefix implementation achieved exact tokenizer/scalar parity and a real
physical-cost ratio of `1.468x` median versus unconstrained decoding. However,
the final four-arm 512 experiment stopped:

- full-prefix versus terminal-only raw `comp_valid`: `+6/512`;
- nonshortcut/primary gain: only `+1/512`, below the frozen `+11/512` gate;
- shortcut-valid outputs increased by `+5`;
- terminal-only had seven generation errors.

No paired-64, Body, Direct, or S.U.N. experiment followed. CR-Plan is retained
as negative/mechanistic evidence, not presented as the current successful
method.

## Why composition is now the main bottleneck

The body/refiner path reaches approximately `99.7–99.8%` conditional structure
validity, while Planner formula validity is materially lower. In the frozen
historical H1 audit, 1,186 parsed Plans contained:

| Planner outcome | Count |
|---|---:|
| Composition valid | 1,044 |
| Charge-neutrality failure | 98 |
| Pauling failure | 37 |
| Missing usable oxidation state | 7 |

The generated historical `charge:` line is causally backward: the Planner
emits the formula first and only later reports `charge_ok` or `charge_fail`.
That label cannot influence the earlier element/count decisions. The compact
formula also supplies sparse count supervision compared with repeated
atom/ion tokens.

This diagnosis motivates a Planner-only SFT intervention while freezing the
successful DLM and refiner.

## Current experiment: minimal no-charge ion-auxiliary SFT

This is an authorized, locally frozen experiment design, **not yet a reported
generation result**.

The deployed Plan remains formula-first and close to MP-20. C0/C1 remove only
the generated `charge:` line:

```text
formula: Li2O
anion: oxide
lattice: cubic
spacegroup: sg_195_230
volume: volpa_016_020
end: plan
```

The three Planner arms are:

| Arm | Inference output | Training difference |
|---|---|---|
| P0 | Historical seven-line Plan | Frozen H1-A2 baseline |
| C0 | Six-line Plan without `charge:` | Continued SFT with neutral atom/count auxiliaries |
| C1 | Same six-line Plan as C0 | Matched auxiliaries use explicit oxidation-state/ion witnesses |

C0 and C1 consume the same 3,200-record, 400-update ledger:

- 30% stable-primary no-charge Plan targets;
- 5% matched atom-sequence/ion-sequence to formula;
- 5% matched element-count/element-oxidation infill;
- 40% full-MP20 conditional anchors with formula in the input;
- 20% conditional frozen-P0 KL anchors over nonformula fields.

Formula/chemistry payload tokens receive `2x` loss weight. Evaluator-invalid
MP-20 formulas are never repeated as unconditional positive formula targets,
but remain available as conditional anchors so the model does not drift away
from MP-20. SMACT 4.0.0 supplies a frozen secondary witness/audit contract;
the historical paper-comparable evaluator remains the primary metric.

At inference there is no repair, witness selection, retry, filtering,
replacement, or reranking. Invalid generated formulas remain raw failures.
The body DLM, exact-length contract, refiner, and evaluators are unchanged.

Local source-repair V2 passed exact file hashing, isolated `101/101` tests,
preflight, and shell/Slurm syntax checks. Real train-data materialization,
tokenizer/model smoke, and A800 training remain pending; no C0/C1
`comp_valid`, `struct_valid`, joint, or S.U.N. number is claimed yet.

See
[H1_NOCHARGE_ION_AUX_SFT_EXECUTION_ANNEX_V1.md](workstreams/plangraph_dlm_iclr_20260731/analysis/H1_NOCHARGE_ION_AUX_SFT_EXECUTION_ANNEX_V1.md)
for the immutable task mixture, evaluator identities, gates, and downstream
reporting contract.

## Metric conventions

Primary metrics use the raw all-attempt denominator:

- Planner failures remain false;
- body/refiner failures remain false;
- invalid composition or structure remains false;
- non-novel and non-unique samples remain false for S.U.N.;
- unknown hull values remain false and are reported explicitly.

The project reports:

- Planner parse/completion, composition validity, primary/nonshortcut and
  shortcut strata, charge/Pauling/missing-state failures, diversity, element
  and arity coverage;
- body generation/completion and exact-length compliance;
- CrysLLMGen-compatible `comp_valid`, `struct_valid`, joint `valid`,
  `wdist_density`, `wdist_num_elems`, `cov_recall`, and `cov_precision`;
- uniqueness, novelty, novel-unique rate, strict S.U.N., and meta S.U.N.;
- paired discordance, McNemar tests, bootstrap intervals, sign stability,
  coverage, and unknown accounting.

Completed/refined or survivor-denominator tables are secondary diagnostics and
are always labelled as such.

## Reproduction and code map

| Purpose | Entry point |
|---|---|
| Operational result index | [EXPERIMENT_TODO_INDEX_V3.md](workstreams/plangraph_dlm_iclr_20260731/EXPERIMENT_TODO_INDEX_V3.md) |
| Frozen end-to-end H1 baseline | [H1_FALLBACK_MANIFEST.md](workstreams/plangraph_dlm_iclr_20260731/H1_FALLBACK_MANIFEST.md) |
| Successful safe-axis DLM experiment | [H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md](workstreams/plangraph_dlm_iclr_20260731/H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md) |
| Planner composition diagnosis | [H1_COMP_VALID_ROOT_CAUSE_AND_SFT_PLAN_V2.md](workstreams/plangraph_dlm_iclr_20260731/analysis/H1_COMP_VALID_ROOT_CAUSE_AND_SFT_PLAN_V2.md) |
| Current no-charge SFT execution annex | [H1_NOCHARGE_ION_AUX_SFT_EXECUTION_ANNEX_V1.md](workstreams/plangraph_dlm_iclr_20260731/analysis/H1_NOCHARGE_ION_AUX_SFT_EXECUTION_ANNEX_V1.md) |
| H1 Planner implementation | [crystal_dlm/h1_llm_planner.py](crystal_dlm/h1_llm_planner.py) |
| No-charge/ion auxiliary implementation | [crystal_dlm/h1_nocharge_ion_aux.py](crystal_dlm/h1_nocharge_ion_aux.py) |
| Exact-length body DLM | [crystal_dlm/r5_plan_body.py](crystal_dlm/r5_plan_body.py) |
| DLM generation core | [crystal_dlm/llada_generation.py](crystal_dlm/llada_generation.py) |
| CrysLLMGen integration | [crystal_dlm/wqcodiff/crysllmgen/](crystal_dlm/wqcodiff/crysllmgen/) |
| Tests | [tests/](tests/) |

The older Wyckoff-quotient program and restored `legacy_dlm_r5c/` tree are
retained for provenance. They are not silently mixed into the current H1
headline.

## Repository boundaries

Source, contracts, ledgers, tests, terminal reports, and reproduction
documents are versioned. Large checkpoints, model weights, datasets, run
directories, caches, archives, and secrets are excluded by
[.gitignore](.gitignore) and are referenced by immutable path/SHA records.
