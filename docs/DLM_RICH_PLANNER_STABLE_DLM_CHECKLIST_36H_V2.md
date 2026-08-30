# Rich-Planner + Stable-DLM 36-hour checklist v2

Date: 2026-08-30
Status: revised after Skeptic review; no downstream field selected from test outcomes
Deadline: 2026-08-31 23:30 Asia/Shanghai

## Understanding lock

- Stability is the primary endpoint. The default recovery reference is the
  corrected H1-A2 exact replay (`8.58%` Strict, `46.08%` Meta), always reported
  with its denominator and alongside the continuous replay (`7.63/45.47%`) and
  historical high point (`9.40/47.40%`). Returning to that corrected range is
  an acceptable sprint outcome; an additional gain is desirable but not a
  result-deletion gate.
- C3FD-v2.5 remains the composition proposer because it already gives exact
  composition/N conservation and 100% legacy composition validity without
  outcome filtering.
- Audit and test the Planner's **existing seven-line rich interface** before
  inventing new DLM intent tokens. The old deployment sampled lattice system
  and volume bin from rich logits, hard-derived anion/charge, and compiled SG
  one-to-one from lattice. CPU validation now shows the separately supervised
  SG head carries target-label information and is substantially more faithful
  under top-1 validation. The proposed canary interface therefore predicts
  lattice, SG and volume, subject to sampling/RNG regression tests and downstream
  validation. The Planner may propose soft structural
  context, but it may not generate the CIF or consume test stability outcomes.
- The DLM remains responsible for the complete discrete crystal body. A frozen
  continuous diffusion model may refine lattice/coordinates but may not change
  composition or hide raw-DLM regressions.
- Every conclusion must be backed by an original run artifact, source code, or
  a newly reproducible statistic. Earlier synthesis documents are indexes, not
  sufficient evidence by themselves.
- No best-of-N, reranking, replacement, survivor filtering, composition tilt,
  checkpoint/temperature/tau grid, or test-set method selection.
- At most two Slurm jobs may be active or pending; **maximum concurrent**
  allocation is six A800s and every GPU receives 4--8 CPUs. Only wrappers named
  in the frozen sprint allowlist may submit work. Reports state cumulative
  A800-hours, peak concurrent GPUs and remaining estimate separately.

## Paper questions and candidate contributions

### Research questions

1. **Proposal versus realization:** when chemically valid and novel
   compositions increase, why can stable S.U.N. conversion decrease?
2. **Structural specification:** does a predicted, uncertainty-aware coarse
   structural Plan reduce the underdetermination of
   `p(structure | exact composition, N)` without leaking the target structure?
3. **Stable DLM learning:** can a masked DLM learn same-composition
   thermodynamic preference while preserving its raw periodic-geometry
   manifold, rather than relying on the diffusion refiner to rescue it?
4. **Pipeline attribution:** which gains exist in the raw DLM and which are
   introduced, erased, or amplified by continuous diffusion refinement?

### Candidate contribution stack

1. **C3FD constrained semantic Planner.** It solves exact composition/N
   realization at the proposal stage and exposes candidate-supply versus
   stability-conversion as separate quantities.
2. **Predicted-rich, exact-composition masked DLM.** A learned Planner supplies
   only coarse soft context; the DLM still generates every lattice and atomic
   coordinate token under exact composition/cardinality constraints.
3. **Same-composition stability alignment with raw safety.** Full-sequence
   continuous-energy ranking plus raw-validity/reference anchors teaches the
   DLM a relative stability boundary. This becomes a contribution only if two
   training seeds reproduce a raw continuous improvement without losing
   H1-A2-level validity/S.U.N., with refined/official outcomes directionally
   compatible. Refined-only improvement is a refiner-mediated pipeline effect,
   not a Stable-DLM learning contribution.

The raw/refined paired evaluation protocol is mandatory evidence. It is not
advertised as a third algorithm if contribution 3 is weak. Final reporting
marks every candidate contribution as **supported**, **candidate**, or
**unsupported**; it does not promise three contributions in advance.

## Evidence ledger to re-verify from original artifacts

- [ ] R5C exact-dynamic: requested/body/Direct and conditioning provenance.
- [ ] H1-B formula-only: raw geometry template-collapse statistics and S.U.N.
- [ ] H1-A2 historical, exact replay, and continuous replay: denominators,
  Planner/DLM/process seeds, raw/refined split, and official cache provenance.
- [ ] R03 D1/D2 and corrected replays: safe-axis legality, process versus
  independent seeds, Strict/Meta trade-off.
- [ ] Rich SFT epoch2/epoch3: body/Direct versus stability divergence.
- [ ] Counterfactual rich-field grounding: training margin, four repeats, fixed
  1000 endpoint and the exact fields changed.
- [ ] C3FD-v2.5: exact composition validity, NU, distribution TVDs, rich-head
  checkpoint/data hashes, and seven-line renderer.
- [ ] Minimal CTV/SGTC L6/L7: raw/refined Direct, energy, S.U.N., and objective.
- [ ] D3PO: raw adverse result versus post-refiner/official weak positive.
- [ ] model494 tau0/200/500/800 and tau900 sensitivity: absolute rescue and
  arm-effect attenuation.

Each row in the final ledger must contain: artifact path, immutable hash where
available, denominator, seed unit, exact statistic, interpretation, and a
specific forbidden over-claim.

## Rich-field suitability audit

The current C3FD checkpoint trains five soft heads but the deployed sampler does
not use all five. The audit must distinguish a trained head, an active sampled
field, a hard-derived field, and a compiler-derived rendering field.

| Field | Current role | Prior evidence | Required new evidence | Provisional use |
|---|---|---|---|---|
| `anion_framework` | hard-derived from proposal family; its rich head is not sampled | overlaps current minimal `family` | exact code/value mapping; head metric is diagnostic only | retain for seven-line compatibility; no predicted-rich credit |
| `charge_bucket` | hard-derived from the C3FD certificate; its rich head is not sampled | overlaps current minimal `charge` | exact code/value mapping; head metric is diagnostic only | retain as chemistry context; no predicted-rich credit |
| `lattice_system` | broad structural mode | old rich conditioning reduced template collapse, but lattice/SG matching was not lower-hull | val accuracy/NLL/ECE, majority-normalized accuracy, seed stability, matched prompt ablation | soft, with training dropout; never a hard geometry constraint |
| `spacegroup_bucket` | target symmetry range from metadata; old sampler compiled it from metric lattice | target H(SG\|metric lattice)=0.881 nats; separately supervised head top-1 accuracy 60.6% versus compiler 26.8--27.5% | sampled-distribution regression, RNG preservation and matched prompt effect | test sampling the existing SG head after volume; soft only, never hard-enforce a one-to-one metric/SG map |
| `volume_per_atom_bin` | global scale | modest favorable continuous/Meta association in historical diagnostic | val accuracy/NLL/ECE, ordinal error, seed stability, matched prompt ablation | highest-priority soft structural field, still uncertainty-aware |

CPU audit outputs:

- [ ] per-checkpoint and pooled validation NLL, top-1 accuracy, majority
  baseline, majority-normalized gain, 10-bin ECE, class entropy, confusion and
  ordinal error where ordered;
- [ ] independent SG-head accuracy, lattice-derived SG accuracy and prediction
  agreement;
- [ ] seed17/18 distribution and metric variation;
- [ ] exact code evidence showing which heads are sampled, calibrated, ignored,
  or deterministically compiled;
- [ ] train-only association of field correctness with raw Direct, raw energy,
  refined energy, Meta and Strict, clearly labelled observational;
- [ ] immutable JSON/CSV/MD, input hashes, tests and zero-outcome-leak marker.

No field is promoted because it has balanced classes or low teacher-forced NLL
alone. Prediction, nonredundancy, and downstream causal value are separate.

## Single hard operational gate: execution preflight

This is one compound operational gate, not a collection of scientific result
gates. No A800 job starts until a current asset/implementation manifest supersedes the
stale transfer ledger and proves, from the live execution environment:

- the CPU-only active-field calibration audit is terminal for both C3FD
  checkpoints, including NLL, accuracy, majority baseline, ECE, ordinal error,
  seed variation and the SG derivation audit; ambiguity is disclosed but is not
  converted into an outcome-tuned threshold;
- rich-compatible H1-A2 DLM, current minimal DLM, C3FD checkpoint/data,
  model494, MP20 split, parser and sampler paths exist and have SHA256 entries;
- all three canary arms consume one ledger SHA and preserve the same hard
  composition/N/sample ordinals;
- an explicit RCF renderer exists and is round-trip tested;
- the canary wrapper requests <=6 GPUs, 4--8 CPUs/GPU, checks the two-job ceiling
  and uses a frozen wrapper allowlist;
- a contract hash is atomically registered with its Slurm job ID before launch;
  an existing active, completed or failed contract cannot be silently reused;
- blocked MP20/D3PO/L6/L7 cohort hashes are loaded and exact identity plus
  chemsys overlap is recomputed, rather than trusting a Boolean manifest flag;
- every attempt receives a new immutable run directory and writes contract,
  code/checkpoint/ledger/input/output hashes, Slurm log/exit and exactly one
  SUCCESS or FAILED marker.

## First GPU experiment: package recovery plus rich-context attribution

Before any new stability training, locate and hash the exact rich-compatible
H1-A2 DLM checkpoint, prompt parser, frozen-composition entry point and sampler.
The code preflight must prove that all three arms accept one immutable ledger;
otherwise no GPU job starts. Freeze 256 outcome-blind, chemsys-held-out
development compositions, excluding every prior main/sealed/prospective test
identity. Use two common DLM/refiner streams and exactly three arms:

1. `M0`: current C3FD minimal prompt + current minimal DLM;
2. `RCF`: current C3FD exact composition + historical rich-compatible DLM with
   a pre-frozen marginal-preserving counterfactual assignment of the complete
   `(metric lattice, SG bucket, volume bin)` tuple from another development
   composition; empirical joint support and all hard anchors remain valid;
3. `R0`: the same rich-compatible DLM + current C3FD predicted seven-line Plan.

This uses six cells and no official MP query. The counterfactual permutation is
fixed before generation and preserves the predicted rich tuple jointly, not
three independent marginals; no `<UNKNOWN>` or other
out-of-support neutral token is introduced. `R0-RCF` estimates the effect of
aligning the three predicted soft structural fields to their originating
composition while holding the hard composition, DLM, prompt schema and streams
fixed. It does not estimate oracle field correctness, an SG-only effect, or an
overall composition-alignment effect.
`R0-M0` is explicitly a package-level recovery comparison, not a single-factor
causal effect.

Report body, raw Direct, raw CHGNet, periodic minimum-distance failures,
lattice/volume diversity, refined Direct/CHGNet, and raw-to-refined movement.
Do not reduce this to pass/fail or claim that 256 development compositions prove
population recovery. It estimates continuous raw/refined effects and exposes
large execution regressions. A return toward corrected H1-A2/R03 behavior is
actionable evidence for initialization, not a final paper claim.

## Stable-DLM training after recovery canary

Interpret the canary without a result-deletion gate:

- if `R0` improves continuous energy as well as raw execution, rich context has
  direct development evidence;
- if it restores raw execution/diversity but not energy, it is only a recovery
  initializer and the stability-alignment objective remains essential;
- if `R0-RCF` is null but both rich arms exceed `M0`, the evidence supports a
  rich-trained checkpoint/package, not composition-specific rich predictions;
- if both rich arms regress, archive the package result and use the already
  partially implemented minimal listwise+raw-safety fallback before any final cohort is
  opened.

If the rich package has useful recovery value, initialize from the frozen
rich-compatible DLM and build a
fresh train-only rich-conditioned candidate pool:

- MP20-train compositions only; chemsys-disjoint validation;
- current C3FD predicted rich Plan, never target-derived oracle rich fields;
- multiple predeclared BASE trajectories per composition and common model494
  seeds;
- one record per attempted trajectory, including invalid raw bodies;
- labels: raw parse/Direct/minimum periodic distance/raw energy where known,
  post-refiner energy, and within-composition ranks;
- group weight one; no absolute-energy comparison across compositions;
- immutable builder, source hashes, seed ledger and no-test-outcome guard.

Training objective:

```text
L = L_rich_body_CE
  + lambda_rank * L_same_composition_continuous_listwise
  + lambda_raw * L_raw_validity_and_raw_rank
  + lambda_ref * L_quadratic_reference_bound
  + 0.20 * L_best_raw_valid_anchor
```

Soft rich-field dropout is included so uncertain Planner predictions do not
become brittle hard constraints. Constants are calibrated once from train-only
gradient norms, then frozen. Two training seeds, exactly 348 updates, only the
final checkpoint, no seed/checkpoint selection.

An `R0` failure does not prove that oracle structural intent is useless; it only
classifies this predicted interface/checkpoint package. No oracle test field is
added to rescue the same development result. The predeclared fallback is the
partially implemented minimal listwise+raw-safety route and must be chosen
before any final-cohort outcome. Its loss/data utilities do not count as a
runnable method until the model, trainer, two-seed/348-update wrapper and
integration tests are complete.

## Prospective evaluation

- [ ] Freeze one new 256-composition C3FD ledger before policy outcomes.
- [ ] Verify exact-composition disjointness from MP20 train and all earlier
  D3PO/L6/L7 main/sealed cohorts; report Planner-seed provenance honestly.
- [ ] One six-cell generation: BASE/R0 policy seed A/R0 policy seed B by two
  common streams, one trajectory per Plan.
- [ ] One 12-cell raw/refined offline evaluation.
- [ ] One fresh official MP query for that immutable union.
- [ ] Composition-cluster bootstrap: average the two streams within each
  composition before uncertainty estimation; disclose both training seeds.
- [ ] Report continuous energy/hull distributions first, then Direct/NU and
  Strict/Meta S.U.N. thresholds.

Interpretation is graded rather than a deletion gate:

- **Recovered:** replicated raw/refined behavior is at least comparable to
  corrected H1-A2/R03 and no major validity loss;
- **Improved:** both policy seeds move continuous energy/hull favorably and
  retain recovery-level structure validity/S.U.N.;
- **Refiner-mediated:** only post-refiner endpoints improve;
- **Unstable/negative:** seed directions conflict or both raw and refined
  endpoints fail to improve.

## Resource and time schedule

| Work | Resource ceiling | Expected wall time | Can overlap |
|---|---:|---:|---|
| original-artifact audit + field metrics | 0 GPU, <=48 CPU | 1--3 h | yes, agents/read-only |
| three-arm recovery canary | 6 A800, 48 CPU | 1--2 h | reporting only |
| fresh rich candidate pool | up to 6 A800, 48 CPU | 2--4 h | CPU builder/tests |
| two-seed alignment training | 2 A800, 16 CPU | 1--3 h | report scaffolding |
| prospective generation/refinement | 6 A800, 48 CPU | 1--3 h | none |
| raw/refined offline evaluation | 6 A800, 48 CPU | 1--3 h | docs |
| official query/finalization | 0 GPU, <=8 CPU | 1--4 h | archive/tests |

CPU and API latency do not consume the GPU budget. The time boxes are planning
estimates, not scientific hard stops; only the final deadline and prevention of
duplicate jobs are hard operational constraints.

Official querying is a single immutable 0-GPU, <=8-CPU process with bounded
request concurrency and a query ledger. Credentials enter only a temporary
nonambient process environment, are never printed/hashed/written/placed on a
command line, and are unset immediately after process creation. Engineering
recovery creates a new immutable attempt ID and verifies source/cache state; it
never overwrites or repeats a completed external query.

## Decision checklist

The entries below are evidence/progress items, not independent scientific
result thresholds. Items explicitly included in the single execution preflight
are operational prerequisites. User-facing status uses exactly these stages: design
reviewed; execution preflight passed; recovery canary terminal; training
terminal; prospective cohort opened; prospective evaluation terminal. A stage
is never called active without a job ID or artifact path and marker.

- [ ] Original-artifact evidence ledger complete.
- [ ] Live asset and implementation SHA manifest complete; stale transfer
  ledger reconciled.
- [ ] Rich-field validation metrics complete for both Planner checkpoints.
- [ ] SG redundancy and sampling behavior explicitly documented.
- [ ] Historical rich-compatible DLM checkpoint and prompt parser hashed.
- [ ] Wrapper allowlist/resource/job-count/atomic-contract guards tested.
- [ ] Blocked-cohort overlap recomputed from hashed inputs.
- [ ] Three-arm recovery canary contract frozen and six cells run once.
- [ ] Recovery interpretation written without field cherry-picking.
- [ ] Fresh train-only rich candidate pool frozen.
- [ ] Two-seed stability alignment trained without selection.
- [ ] Prospective cohort frozen before outcomes.
- [ ] Raw/refined/official evaluation complete.
- [ ] Final paper RQs, 2--3 contribution claims and forbidden claims updated.
- [ ] Positive and negative runs archived with root cause and resource usage.
