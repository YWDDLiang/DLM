# Flagship Experiment Plan: Stratified Geometry-Adaptive Wyckoff Co-Diffusion

Date: 2026-07-10  
Revision: v3 after the evaluation-metric audit; hard-capped at four A800s and four weeks  
Target: one ICLR oral-quality crystal-generation paper  
Working name: **StratWQ-CoDiff**

## 0. Executive Decision

The project continues to target one crystal-generation paper. DLM is not a
separate poster track and is not treated as a replacement for continuous
diffusion. It is the prioritized discrete transition engine inside a
stratified discrete--continuous process.

The central thesis is upgraded from generic bidirectional co-denoising to:

> Crystal generation is a generative process on a stratified Wyckoff quotient
> space. Discrete Wyckoff topology determines the dimension and constraints of
> the continuous crystal manifold. A geometry-calibrated masked denoiser makes
> reversible orbit birth, death, and type corrections across strata, while a
> space-group-equivariant continuous model denoises lattice and free coordinates
> within the current stratum.

The paper has three method contributions:

1. a stratified Wyckoff-quotient state space for complete crystal generation;
2. trans-dimensional orbit birth, death, and type-change kernels that preserve
   crystallographic support;
3. geometry-adaptive true remasking of previously committed discrete states.

Relaxation consistency is a performance and mechanism extension. It is not a
co-equal fourth idea and is removed if it does not transfer to held-out MLIPs.

The experimental cycle is now a hard 28-calendar-day sprint on four A800 GPUs.
The theoretical maximum is 2688 GPU-hours; the usable registered ceiling is
2050 GPU-hours. No task is allowed to extend the experimental deadline.

## 1. Claim Hierarchy And Claim Boundaries

### Primary Claim

At matched data, parameter, training-compute, inference-compute, and refinement
budgets, stratified geometry-adaptive topology revision improves attempt-level
held-out `MLIP-SUN@0.1` over the frozen strongest matched Wyckoff baseline while
preserving novelty, uniqueness, crystallographic validity, symmetry, coverage,
and a defensible quality--compute Pareto frontier.

### Confirmatory Mechanism Claims

1. A typed masked/set denoiser is more effective than matched WQ-AR and WQ-D3PM
   for high-corruption reconstruction of complete Wyckoff protostructures.
2. Clean or informative geometry causally improves discrete topology recovery,
   and informative discrete topology causally improves continuous denoising.
3. True geometry-adaptive revision corrects more committed decisions than it
   corrupts and outperforms confidence-only, random-remask, fixed-schedule, and
   extra-compute controls.
4. Orbit-level birth/death adds value beyond fixed topology and cannot be
   explained by atom-level NULL-slot behavior alone.
5. Relaxation consistency is retained only if guide-MLIP gains transfer to
   every held-out MLIP without protected-distribution collapse.

### Prohibited Claims

- first discrete--continuous crystal diffusion;
- first masked categorical diffusion for crystal generation;
- first variable-cardinality crystal generator;
- first bidirectional atom-type/structure conditioning;
- DLM is universally better than autoregressive or continuous diffusion;
- exact likelihood, detailed balance, or a valid RJ-MCMC sampler unless proven;
- DFT-confirmed stability, thermodynamic stability, synthesizability, or
  experimental discovery;
- headline results from conditional, best-of-N, selected, or survivor-only runs.

The paper may claim novelty only for the complete combination that survives the
matched controls. It must describe MatterGen, WyckoffDiff, SymmCD, SGEquiDiff,
MCFlow, MiAD, and CrysLLMGen as direct neighboring work.

## 2. Scope

### In Scope

- fully unconditional, free-orbit-count, ordered inorganic crystal generation;
- MP20 as the main benchmark;
- Alex-MP-20 or grouped-OOD only when the main result freezes by Day 24 with
  unused registered buffer;
- generated space group, orbit topology, Wyckoff type, and species;
- symmetry-compatible lattice metrics and asymmetric-unit free coordinates;
- explicit orbit birth, death, type change, and committed-state rollback;
- attempt-level `MLIP-SUN@0.0/@0.1`, multi-MLIP robustness, symmetry, and efficiency;
- one guide MLIP for training-only relaxation labels and two held-out MLIPs;
- as-generated, common-refiner, and one-shot-relaxed evaluation stages;
- a frozen legacy-proxy panel and a LeMat-GenBench compatibility panel.

### Out Of Scope

- new DFT calculations or experimental synthesis claims;
- fractional or disordered occupancy in the first paper;
- property-conditioned inverse design, doping, molecules, or non-crystal tasks;
- beam search, rejection sampling, best-of-N, multiple final refinements, or
  output-dependent early stopping;
- a separate DLM paper during the flagship experimental cycle;
- MP-Doob as a co-equal contribution.

For ordered crystals, occupancy is fixed to one for every active orbit.
Multiplicity and total atom count are deterministic consequences of the space
group and Wyckoff types; they are not separately generated tokens.

## 3. Prior-Art Boundary

| Work | Already established | Remaining distinction required here |
|---|---|---|
| MatterGen | masked atom-type diffusion jointly denoised with coordinates and lattice | hard Wyckoff quotient and explicit cross-stratum topology revision |
| WyckoffDiff | D3PM on Wyckoff protostructures | complete continuous crystal, variable topology, and geometry feedback |
| SymmCD | joint discrete site-symmetry/atom-type and continuous ASU/lattice diffusion | exact free-coordinate strata, no terminal Wyckoff projection, generated orbit count, true rollback |
| SGEquiDiff | AR Wyckoff/species generation followed by equivariant coordinate diffusion | reversible rather than one-way topology--geometry generation |
| MCFlow | independent atom/structure times and bidirectional conditioning | hard quotient strata and constraint-aware adaptive route selection |
| MiAD | atom-level NULL birth/death and variable atom count | orbit-level symmetry-coupled birth/death and dimension-changing charts |
| CrysLLMGen / *LLM Meets Diffusion* | LLM proposes and freezes composition; diffusion refines LLM coordinates/lattice from an intermediate time | matched quotient proposal with geometry-to-topology feedback and true revision rather than a one-way frozen handoff |
| LLaDA-Rec | task-aligned parallel representation, multi-scale masking, adaptive commitment | crystal-specific topology, physical geometry feedback, and true rollback |

Neither independent time variables nor cross-state attention is sufficient
novelty. The method must demonstrate that topology-indexed continuous strata and
geometry-adaptive trans-dimensional revision are necessary.

LLaDA-Rec is used as a mechanism audit, not as evidence that DLM is intrinsically
better. Its useful tests are transferred faithfully: matched causal versus
bidirectional context, fixed versus confidence-adaptive generation order, and
quality versus generation steps. Its tokenizer, recommendation beam search, and
fixed-length assumption are not transferred.

CrysLLMGen is the direct sequential-hybrid baseline. Its useful idea is a
chemically strong discrete proposal followed by continuous diffusion injected at
an intermediate time. Its frozen composition, one-way handoff, 7B-versus-small
model confounding, and sampling-time removal of invalid compositions are not
evidence for bidirectional co-denoising. Our internal matched implementation must
retain every submitted attempt and isolate handoff, feedback, and rollback.

## 4. Formal Stratified State Space

Let the discrete topology be

\[
\tau =
\left(
G,\{(w_i,a_i)\}_{i=1}^{K}
\right) / S_K,
\]

where G is the space group, K is the number of active symmetrically unique orbit
instances, w_i is a Wyckoff type, a_i is the species, and S_K denotes that orbit
order is not physical. The quotient is a multiset: multiple orbit instances may
share the same Wyckoff type and species while having different free coordinates.

Multiplicity is m(G,w_i), and the atom count is

\[
N(\tau)=\sum_{i=1}^{K}m(G,w_i).
\]

The continuous stratum associated with topology tau is

\[
\mathcal M_\tau =
\mathcal L_G
\times
\prod_{i=1}^{K}\mathcal U_{G,w_i},
\]

where L_G is the positive-definite symmetry-compatible lattice-metric space and
U_{G,w_i} is the chart of free asymmetric-unit coordinates for orbit i. Thus,

\[
\dim \mathcal M_\tau =
\dim \mathcal L_G+
\sum_{i=1}^{K}\dim\mathcal U_{G,w_i}.
\]

The full state space is the disjoint union

\[
\mathcal Q=\bigsqcup_{\tau\in\mathcal T}\{\tau\}\times\mathcal M_\tau.
\]

Birth, death, or Wyckoff-type changes may change the dimension of the continuous
state. Canonical orbit order is permitted only for storage. Model inputs use
permutation augmentation and a permutation-equivariant set architecture.

### Transition Semantics

- birth: sample a new Wyckoff/species orbit and initialize its free coordinates
  from a conditional base or learned bridge on the target chart;
- death: remove the orbit and its continuous variables;
- Wyckoff change: implement as death followed by birth unless a valid chart
  transport is formally defined;
- species change: keep the chart but revise the discrete chemical state;
- same topology: apply only stratum-internal continuous denoising;
- space-group change: restrict to an explicitly defined compatible transition
  graph and early low-geometry-SNR steps in version 1.

The semantic state is a dynamic orbit multiset. Batching padding is an implementation
artifact: it must not enter the probability definition, loss, metrics, or model
output. A fixed K_max NULL canvas may be used only if tests prove padding
invariance and the decoded transition semantics remain explicit birth/death.

## 5. Formal Deliverables Before Full Training

The implementation must establish or exhaustively test:

1. every reverse transition remains in Q;
2. a topology transition produces a continuous state in its target stratum;
3. orbit permutation equivariance;
4. compatibility masks never remove ground-truth support;
5. categorical and event kernels are normalized;
6. projectors and chart round trips are idempotent within tolerance;
7. disabling birth/death reduces to a fixed-topology model;
8. disabling rollback reduces to a monotonic masked denoiser;
9. disabling geometry messages reduces to a discrete-only adaptive denoiser;
10. changing batching padding does not change loss, posterior, or reconstruction.

Required numerical gates:

- toy-kernel normalization error below 1e-6;
- chart/projector reconstruction error below 1e-6 on synthetic fixtures;
- permutation posterior difference below 1e-5;
- zero illegal transitions in at least one million synthetic transitions;
- zero ground-truth-support removals by compatibility masks;
- zero padding-dependence failures in the dedicated test suite.

The paper will not use the words stratified or trans-dimensional as a headline
claim unless these gates pass.

## 6. Model Matrix

### Historical And Atom-Level Controls

| ID | Definition | Role |
|---|---|---|
| H-ATOM-DLM | existing exact-length masked executor plus frozen refiner | historical diagnostic only |
| B-ATOM-AR | strongest reproducible atom/string AR plus shared final evaluator | atom-level control |
| B-ATOM-JOINT | strongest reproducible atom-level joint diffusion/flow | ordinary crystal-generation control |

### Matched Wyckoff Controls

| ID | Definition | Causal question |
|---|---|---|
| B-WQ-AR | same WQ multiset and continuous backend, random-order AR with birth/STOP | quotient versus DLM attribution |
| B-WQ-D3PM | same WQ multiset and event support with standard categorical diffusion | masked DLM versus ordinary discrete diffusion |
| B-WQ-DLM-MONO | masked DLM; a committed field cannot be remasked | value of true rollback |
| B-WQ-JOINT-NOREV | joint discrete--continuous denoising with no explicit committed-state revision | value beyond ordinary joint diffusion |
| B-WQ-DISC-ONCE | CrysLLMGen-style WQ/geometry proposal injected at \(\tau\), with topology frozen during continuous refinement | value beyond a strong one-way LLM/DLM--diffusion handoff |

### Proposed Variants

| ID | Definition | Role |
|---|---|---|
| M-WQ-STRAT-CONF | orbit birth/death plus confidence-triggered true remask | adaptive discrete control |
| M-WQ-STRAT-GEO | confidence plus geometry-calibrated true remask | core method |
| M-WQ-STRAT-RELAX | core plus transferable relaxation consistency | final method if promoted |
| M-WQ-STRAT-RELAX-MP | full plus marginal preservation | collapse-triggered optional control |

The development baseline champion is selected only among B-WQ-AR, B-WQ-D3PM,
B-WQ-DLM-MONO, B-WQ-JOINT-NOREV, and B-WQ-DISC-ONCE. It is frozen before the
core generation comparison. A published-only external number cannot become the
matched champion.

B-WQ-AR must be a strong order-agnostic baseline: train with randomized orbit
orders or a valid order-marginalized objective, generate orbit instances with an
explicit STOP event, and use no positional convention that leaks the storage
canonicalization. B-WQ-D3PM and B-WQ-DLM-MONO use the same allowed topology
event support as the proposed method. A deliberately weak canonical-order AR or
fixed-cardinality D3PM is not an admissible matched control.

For B-WQ-DISC-ONCE, freeze a validation-only injection grid
\(\tau/T\in\{0.25,0.50,0.75,1.00\}\). Compare proposed geometry at \(\tau\),
topology-only conditioning with fresh continuous noise, and matched random
geometry. The selected \(\tau\) freezes before core evaluation. No invalid
proposal is discarded or replaced. This tests whether gains come only from a
better starting point, as in CrysLLMGen, or from later geometry-to-topology
correction.

External baselines should include reproducible variants of CrysLLMGen,
DiffCSP/FlowMM, MatterGen, WyckoffDiff, SymmCD, SGEquiDiff, MCFlow, MiAD, and
Chemeleon2 when artifacts exist. Rerun-matched and published-reference-only
results must be separated.

## 7. Discrete Corruption And DLM Rescue Design

The typed discrete denoiser is trained on a mixture of:

- whole-orbit masking;
- within-orbit Wyckoff/species masking;
- deletion of a true orbit;
- insertion of a false orbit;
- wrong species or wrong Wyckoff substitution;
- coincident-orbit or crystallographically incompatible orbit corruption;
- correct discrete topology with corrupted geometry;
- plausible marginal fields with an incompatible joint discrete--continuous
  state.

Coordinate and lattice values are never converted into DLM tokens. The discrete
engine predicts topology and chemical categories only.

The old exact-length model is the monotonic historical baseline. Its fixed
N--elements--lattice--coordinate schedule and low-confidence commitment do not
qualify as true revision.

## 8. Continuous Model And Cross-Stratum Bridge

The continuous model denoises a symmetry-compatible lattice metric and orbit
free coordinates only in the current stratum. It receives soft discrete
posteriors where mathematically valid and hard topology only where needed to
instantiate a chart.

For every topology event, the bridge B_theta must define how continuous state
is created, removed, or transported:

\[
\tau' \sim R_\theta(\tau'\mid \tau,c,t),\qquad
c' \sim B_\theta(c'\mid c,\tau,\tau',t).
\]

This is a learned reverse generative process, not an RJ-MCMC claim. Space-group
changes may be limited to early steps; orbit topology remains revisable
throughout the registered revision window.

Two discrete/continuous time variables and off-diagonal corruption regimes are
required controls and training tools, but are not presented as novel because
MCFlow already establishes that design space.

## 9. Geometry-Adaptive True Remasking

For orbit or field i, the scheduler consumes:

\[
r_i=f_\phi(
H[p_\theta(d_i)],
s_i^{collision},
s_i^{coordination},
s_i^{strain},
s_i^{symmetry},
s_i^{force},
s_i^{basin},
t).
\]

The score decides whether to commit, retain, remask, birth, or delete an orbit
under a fixed call and revision budget. True remasking requires that:

- a committed species, Wyckoff type, orbit existence state, or early-stage
  space-group state can become masked again;
- its later value may differ from the committed value;
- orbit death removes its continuous variables;
- orbit birth creates new continuous variables;
- training includes apparently committed but incorrect states;
- every revision has a logged reason and shared-noise no-revision control.

Primary fixed-compute experiments disable adaptive early stopping. A separate
variable-compute Pareto track may stop early but cannot replace the fixed-budget
causal comparison.

## 10. Training Objectives

The full objective is

\[
\mathcal L =
\mathcal L_{event}
+\mathcal L_{field}
+\lambda_{geo}\mathcal L_{geometry}
+\lambda_{lat}\mathcal L_{lattice}
+\lambda_{bridge}\mathcal L_{bridge}
+\lambda_{cross}\mathcal L_{cross}
+\lambda_{sym}\mathcal L_{symmetry}
+\lambda_{cal}\mathcal L_{calibration}
+\lambda_{relax}\mathcal L_{relaxation}
+\lambda_{basin}\mathcal L_{basin}.
\]

- L_event: orbit birth/death and topology-event prediction;
- L_field: typed masking loss over space group, Wyckoff type, and species;
- L_geometry: score/flow loss in Wyckoff free-coordinate charts;
- L_lattice: score/flow loss on the symmetry-compatible lattice metric;
- L_bridge: target-stratum validity and coordinate initialization/transport;
- L_cross: compatibility of discrete posterior and geometry evidence;
- L_symmetry: equivariance, reconstruction, and support preservation;
- L_calibration: calibrate confidence and revision scores;
- L_relaxation: short training-only relaxation displacement/commutation targets;
- L_basin: calibrated probability that the state survives fixed refinement.

Training proceeds in three stages:

1. matched discrete recovery and continuous denoising pretraining;
2. joint bridge, bidirectional intervention, and adaptive revision training;
3. optional relaxation consistency only after the core passes its gate.

One guide MLIP may construct training labels. Held-out MLIPs are prohibited from
losses, early stopping, checkpoint selection, scheduler calibration, or
hyperparameter tuning. They may be queried only by a pre-registered one-shot
phase gate on validation attempts after the phase configs and checkpoints are
frozen. No tuning is allowed after that query. Benchmark-test attempts remain
unseen until P8. MP-Doob activates only after a pre-registered family/prototype
drift greater than 2 percentage points.

## 11. Fairness And Evaluation Contract

All matched WQ comparisons require:

- identical data split, augmentations, and canonicalization;
- parameter count within 5 percent;
- identical optimizer family and update count;
- training FLOPs within 10 percent and complete actual-compute reporting;
- same continuous backend, final frozen refiner, refinement calls, and steps;
- exact discrete, continuous, joint, bridge, projection, and refiner call logs;
- both fixed-call and realized-FLOP/wall-clock comparisons;
- same attempt IDs, initial noise, and paired corruption fixtures where possible;
- no beam, best-of-N, rejection, retry, or output-dependent truncation;
- every submitted attempt stays in the denominator.

NFE does not imply equal compute across a Transformer DLM and an equivariant
continuous model. Exact calls and actual FLOPs, wall time, peak memory, and GPU
hours must all be reported.

## 12. Experiment Phases And Gates

### P0 — Attempt, Evaluator, And Budget Contract

Deliverables:

- stable attempt_id through every generation and evaluation stage;
- world-size invariant per-attempt seeds;
- one terminal status and reason per attempt;
- no output-dependent selection or replacement;
- frozen data, matcher, evaluator, refiner, cache, and model hashes;
- tests for distributed merge, duplicate IDs, seed mismatch, denominator,
  timeouts, cache mismatch, and all failure paths;
- a metric registry freezing for every metric its denominator, evaluation
  stage, reference set, matcher/tolerance, hash-based subset, and selection rule;
- an as-generated evaluator with no learned refiner or geometry optimization;
- a pinned LeMat-GenBench evaluator and its MACE-MP/UMA/Orb-v3 stack, or a
  Day-3 written incompatibility record forbidding direct cross-paper ranking;
- frozen prototype, protostructure, structure, and substitution-aware novelty
  references with strict/standard/lenient matcher settings;
- measured units for full training, sampling 1000 attempts, and evaluating 1000
  attempts with each MLIP;
- pre-registered fixed-call grid and variable-compute Pareto grid.

Gate: every invariant passes. No model result is paper-admissible before P0.

### P1 — Stratified MP20-WQ Dataset

Build and compare at least three registered symmetry tolerances. Store
conventional and primitive cells, asymmetric units, G, unordered orbit
multisets, Wyckoff charts, free-coordinate bases, chart dimensions, canonical
multiset hashes, round-trip structures, ambiguity flags, symmetry tolerances, and the
complete canonicalization trace. Randomize orbit input order during training.

Gate:

- conversion coverage at least 95 percent;
- round-trip StructureMatcher at least 99 percent;
- multiplicity-derived atom-count consistency 100 percent;
- chart/projector consistency 100 percent within registered tolerance;
- zero canonicalized train/test leakage;
- orbit permutation does not change reconstruction;
- ambiguity flags exist for every tolerance-sensitive decomposition;
- no exclusion changes a headline material-family distribution by over 2 pp.

### P2 — Formal Process And Synthetic Tests

Implement dynamic-K batching, event kernels, target-chart bridges, a
set-equivariant discrete network, the true-remask state machine, and the formal
tests in Section 5.

Gate: every formal numerical gate passes. Failure blocks full training and the
stratified/trans-dimensional claim.

### P3 — DLM Rescue And Falsification

Use corruption-recovery tasks before expensive MLIP evaluation. Compare
B-WQ-AR in conditional random-order repair mode, B-WQ-D3PM,
B-WQ-DLM-MONO, M-WQ-STRAT-CONF, and M-WQ-STRAT-GEO at 30, 50, 70,
and 90 percent orbit/field corruption.

Required conditions:

- missing orbit, false orbit, wrong Wyckoff, wrong species, and joint corruption;
- clean, noisy, shuffled, and absent geometry;
- clean, partial, shuffled, and absent discrete context;
- fixed, discrete-first, continuous-first, confidence-adaptive, and
  geometry-adaptive schedules;
- causal left-to-right, causal random-order, bidirectional fixed-order, and
  bidirectional adaptive-order attention/schedule controls;
- shared corruptions and exactly matched calls.

Promotion gate:

- on both 70 and 90 percent corruption, the 95 percent CI lower bound of exact
  full-protostructure recovery gain over the best WQ-AR/D3PM is positive;
- at least one high-corruption point improves by at least 3 absolute pp;
- orbit edit distance is lower at three of four corruption levels;
- orbit-cardinality recovery is better than WQ-D3PM;
- both geometry-to-discrete and discrete-to-geometry interventions are positive;
- geometry-adaptive beats the best fixed schedule at equal calls;
- bidirectional adaptive order beats fixed left-to-right at equal calls, while
  bidirectional-fixed and causal-adaptive controls separate attention from order;
- the CI lower bound of wrong-to-right minus right-to-wrong is positive;
- shuffled geometry and matched random remasking do not reproduce the gain.

If this gate fails, stop the standalone DLM-superiority claim. One small pilot
may test stratified revision with the best discrete kernel; do not spend further
GPU on the legacy exact-length or CE-reweighting route.

### P4 — Frozen Matched Generative Baselines

Run 25-percent-update screening for B-WQ-AR, B-WQ-D3PM, B-WQ-DLM-MONO,
B-WQ-JOINT-NOREV, and B-WQ-DISC-ONCE. Only the best AR/diffusion engine,
B-WQ-JOINT-NOREV, and the strongest reproducible atom-level baseline may receive
one full seed. Run 256-attempt smoke evaluation followed by three sampling seeds
times 1000 development attempts only for promoted variants. All other external
models use public checkpoints or published samples; at most two are rerun.

Gate:

- parse and representation reconstruction at least 99 percent;
- graph acceptance at least 95 percent;
- illegal Wyckoff/multiplicity states equal zero;
- attempt accounting 100 percent;
- the WQ representation is not over 2 pp worse in `MLIP-SUN@0.1` than the atom
  baseline unless it has a clear symmetry/compute Pareto advantage.

Select and freeze the matched baseline champion before testing the core method.

### P5 — Stratified Geometry-Adaptive Topology Revision

Compare fixed topology, birth/death without geometry revision, confidence
revision, geometry revision, random count-matched revision, shuffled feedback,
no revision, and extra-call controls.

Promotion gate:

- M-WQ-STRAT-GEO improves held-out development `MLIP-SUN@0.1` by at least 2
  absolute pp over the frozen champion;
- three sampling seeds have the same direction;
- Novel&Unique drops by no more than 2 pp;
- graph acceptance remains at least 95 percent;
- geometry-adaptive beats the best fixed schedule at equal calls and realized
  compute;
- shared-noise revision has positive outcome delta versus retaining the old state;
- random remasking and extra calls cannot explain the gain;
- geometry-adaptive revision beats the frozen best-\(\tau\) CrysLLMGen-style
  one-way handoff at equal calls and corrects initially wrong topology;
- gains appear in both orbit-count-changed and unchanged subsets;
- revision churn remains below a pre-registered limit and net correction is
  positive.

### P6 — Relaxation Consistency

Only if P5 passes by Day 17, run one pre-registered relaxation-consistency
fine-tune initialized from M-WQ-STRAT-GEO. The objective is frozen on Day 3 from
terminal stability labels, signed Ehull utility, short displacement supervision,
or denoise--relax commutation; there is no four-way objective search. If the
fine-tune is not complete by Day 19, P6 is removed from the paper.

Promotion gate:

- guide gain transfers by at least 1 pp to each held-out MLIP;
- raw-to-relaxed coordinate RMSD and lattice strain decrease;
- at least three major material families improve;
- protected marginals drift by no more than 2 pp;
  - guide MLIP is absent from final checkpoint selection;
  - held-out transfer is a one-shot validation gate after all P6 variants and
    checkpoints are frozen, with no post-query tuning.

If P5 passes but P6 does not transfer, remove relaxation consistency from the
method and paper claim.

### P7 — Critical Ablation And Independent Replication

The main paper retains four train-affecting ablations:

1. atom space versus Wyckoff quotient;
2. fixed topology versus orbit birth/death;
3. monotonic/confidence revision versus geometry-adaptive true remask;
4. without versus with relaxation consistency.

Only the frozen matched champion and final method receive training seeds 11, 23,
and 47. Existing promoted seed-11 checkpoints count toward this requirement.
Each checkpoint uses sampling seeds 101 and 202 with 1000 attempts:

\[
3\ \text{training seeds}\times
2\ \text{sampling seeds}\times
1000=6000\ \text{attempts per method}.
\]

### P8 — Frozen Final Paper Evaluation

After configs and checkpoints are frozen:

- generate 10,000 pooled attempts for the full method and frozen champion,
  allocated deterministically as 3334/3333/3333 across the three train seeds;
- evaluate all attempts with the primary held-out MLIP;
- evaluate a predeclared attempt-ID-hash subset of at least 6000 with all MLIPs;
- report as-generated, common-refiner, and one-shot-relaxed results separately;
- run the exact frozen LeMat-GenBench panel on the same hash-selected 2500
  attempts if it passed the Day-3 gate; otherwise report the same fields under
  our registered MLIPs and label them non-comparable to the official leaderboard;
- report the two-held-out-MLIP consensus separately from all-MLIP 2/3 and 3/3;
- recompute uniqueness inside each bootstrap resample;
- evaluate strict, standard, and lenient novelty matchers;
- report prototype, protostructure, full-structure, and substitution-aware
  novelty/uniqueness separately;
- evaluate free-N and matched-N tracks separately;
- run Alex-MP-20/grouped OOD only if all main tables freeze by Day 24.

Oral-candidate gate:

- held-out `MLIP-SUN@0.1` gain at least 5 pp;
- hierarchical 95 percent CI lower bound at least 2 pp;
- all-MLIP unanimous `MLIP-SUN@0.1` gain at least 3 pp;
- `MLIP-SUN@0.0` non-inferiority greater than -1 pp;
- raw and common-refiner `MLIP-SUN@0.1` gains both have positive direction;
- Novel&Unique drop no more than 2 pp;
- substitution-aware `MLIP-SUN@0.1` direction is positive and the substitution-derived
  fraction among valid metastable outputs does not increase;
- space-group and occupied-Wyckoff-dimension JSD do not regress materially;
- positive direction under every matcher and major material family;
- not explained by WQ-D3PM, joint-no-revision, extra compute, or the refiner;
- competitive with rerun-matched strong external models;
- end-to-end compute no more than 2x the champion, or a clear Pareto advantage.

## 13. Metrics And Executable Definitions

### Metric Registry And SUN Formula

Every reported metric is keyed by
`{denominator, stage, evaluator, reference, matcher, tolerance, subset_hash}`.
The three stages are `raw` (deterministic crystallographic projection only),
`common_refiner` (the same frozen learned refiner), and `relaxed` (one fixed
evaluator-specific cell/position relaxation). No stage may silently replace
another.

For a registered batch of (N) submitted attempts, let (S_\tau) be the
attempts with evaluator-self-consistent (E_\mathrm{hull}\le\tau), (U) the
attempts whose duplicate-matcher component has size one, and (N_r) the
attempts not matching novelty reference (r). Then

\[
\mathrm{MLIP\!\mbox{-}SUN}_{\tau,r}
=\frac{|S_\tau\cap U\cap N_r|}{N_{\mathrm{submitted}}}.
\]

Every failure contributes zero. Every table writes the threshold explicitly as
`MLIP-SUN@0.0` or `MLIP-SUN@0.1`; unqualified SUN/mSUN labels are not used.
`<` versus `<=`, hull hashes, cell relaxation, force/stress convergence,
element support, and optimizer/step limits freeze in P0.

### Tier A: Mandatory Main-Paper Metrics

Headline and decomposition:

- relaxed primary-held-out `MLIP-SUN@0.1`, with `MLIP-SUN@0.0` secondary;
- raw and common-refiner `MLIP-SUN@0.1` to expose refiner dependence;
- held-out-only unanimous and all-MLIP 2/3 and 3/3 `MLIP-SUN@0.1`;
- marginal (S\), (U\), (N\), (U\cap N), (P(S\mid U\cap N)), and
  (P(U\cap N\mid S)), with count and submitted denominator;
- the LeMat-GenBench fields: validity, formation energy, Ehull, relaxation
  RMSD, stable, metastable, SUN, and MSUN. These MLIP quantities are never
  equated to DFT results in other papers.

Validity and chemistry:

- CIF/parse, finite/SPD lattice, positive `det(L)`, nonempty structure, legal
  Wyckoff/multiplicity, collision, density/lattice-bound, graph, and
  space-group-determinability pass rates;
- SMACT charge-neutrality and electronegativity validity separately, with a
  pinned oxidation-state table and code version;
- unsupported-element, relaxation-failure, nonconvergence, and missing-hull
  rates as visible metrics, not only ledger reasons.

Novelty and diversity:

- benchmark novelty against frozen MP20 train and extended discovery novelty
  against the available MP/Alexandria/ICSD reference, never conflated;
- prototype U/N (anonymous species), protostructure U/N (species--Wyckoff
  assignment), and full-structure U/N, each with number unique, percent seen in
  train, and number new;
- substitution-derived fraction among valid metastable samples and
  substitution-aware `MLIP-SUN@0.1`, because a new composition on a memorized
  prototype is not strong structural novelty;
- (U(n)), (N(n)), and (U\cap N(n)) at
  (n\in\{1000,2000,5000,10000\}) using repeated sampling without replacement;
- top-prototype mass, effective prototype count, and family entropy.

Symmetry and distribution:

- JSD to test for 230-bin space-group support, crystal system, and occupied
  Wyckoff free-coordinate dimension (d\in\{0,1,2,3\}) at every registered
  symmetry tolerance, with 0.1 A as the compatibility point;
- Wasserstein-1 distance for atomic density, number of elements, atom count,
  active-orbit count, and orbit multiplicity;
- intended versus redetected space group, Wyckoff multiset, Wyckoff letter, and
  site-symmetry preservation at raw and relaxed stages;
- chiral-space-group handedness preservation reported separately;
- CrystalNN/Magpie coverage recall/precision and CMD-50/fingerprint diversity,
  with all-attempt and legacy-valid-only denominators explicitly separated.

Recovery, DLM attribution, and revision:

- exact protostructure, space-group, orbit-multiset, species--Wyckoff, and
  orbit-cardinality recovery; topology-edit and tangent-coordinate errors;
- causal versus bidirectional attention, fixed left/right/random versus
  adaptive order, accuracy-versus-iteration, and recovery-versus-step curves;
- wrong-to-right, right-to-wrong, net correction, revision precision/recall,
  rollback, birth/death/type-change, dimension change, churn, and shared-noise
  counterfactual utility;
- invalid-transition, bridge, projection, collision, coordination, strain, and
  basin-score deltas.

Relaxation and efficiency:

- raw-to-relaxed element-aware periodic RMSD, normalized RMSD divided by
  ((V/N)^{1/3}), lattice strain, energy drop, force before/after, relaxation
  steps, cumulative displacement, convergence, and symmetry change;
- train GPU-hours/FLOPs/updates and parameters; component calls, sampling
  seconds/attempt, throughput, peak memory, relaxation/energy wall time, total
  GPU-hours/1000, successful candidates/GPU-hour, and GPU-hours/success;
- quality--steps, quality--FLOPs, quality--wall-time, and stability--novelty
  Pareto curves. This is the LLaDA-Rec-inspired test of whether adaptive DLM
  refinement is useful at equal budget.

### Tier B: Non-Blocking Appendix Metrics

- FWD on equal-size real/generated protostructure sets and FMD-inverse on a
  pinned encoder, only if both pass reproducibility tests by Day 5;
- continuous uniqueness/novelty and transport novelty distance;
- basin AUROC/AUPRC/Brier/ECE and dense family/threshold sweeps;
- full OOD metrics. None may delay Tier A or the final three-seed comparison.

## 14. Statistical Plan

- one primary hypothesis: final method versus the pre-frozen matched champion;
- report all three train seeds, their mean and standard deviation, and a 95
  percent hierarchical interval; attempts are not independent model replicates;
- compute duplicate connected components once on each fixed 10,000-attempt
  batch. Do not recompute uniqueness inside an ordinary with-replacement
  bootstrap, which would create artificial duplicates;
- estimate U/N/SUN uncertainty with train-seed hierarchy plus duplicate-component
  cluster resampling; estimate (U(n)) by repeated without-replacement sampling;
- use paired permutation or McNemar tests only where shared noise makes attempt
  pairs causally meaningful;
- primary comparison uncorrected; confirmatory secondary comparisons use Holm;
- matcher, symmetry, family, and threshold exploration use BH-FDR;
- report absolute pp effect, relative effect, CI, p-value, numerator,
  denominator, every seed, and every registered failure reason.

## 15. Four-A800, Four-Week Compute Funnel

Hardware is fixed at four A800 GPUs for at most 28 calendar days. The physical
maximum is 2688 GPU-hours. To leave room for data stalls, evaluator latency,
serial dependencies, and failures, the registered usable ceiling is 2050
GPU-hours. At least 800 GPU-hours are reserved for frozen champion/final runs in
Week 4; unused early-tier budget rolls forward, never backward into new ideas.

| Tier | Deadline | Work | Ceiling | Maximum promoted variants |
|---|---|---|---:|---:|
| T0 | Day 3 | contracts, dataset, formal and evaluator tests | 40 GPUh | CPU-heavy |
| T1 | Day 7 | 10% MP20 recovery and DLM falsification | 140 GPUh | five |
| T2 | Day 11 | full-data 25%-update screen | 280 GPUh | five |
| T3 | Day 17 | one-seed convergence and 256 smoke | 350 GPUh | four |
| T4 | Day 21 | development, causal interventions, optional one P6 fine-tune | 360 GPUh | champion, core, two controls |
| T5 | Day 28 | seeds 23/47, final 10k, multi-MLIP and statistics | 800 GPUh | champion and final only |
| buffer | Day 28 | recovery from failed jobs, not speculative variants | 80 GPUh | gate-controlled |

Hard cumulative stop-lines are 40 GPUh by Day 3, 180 by Day 7, 750 by Day 14,
1250 by Day 21, and 2050 by Day 28. P0 re-estimates job-unit costs but cannot
raise the final ceiling or deadline.

Rules:

- use one model per GPU and a queue; do not bind multiple GPUs to one model
  unless the P0 timing audit proves a shorter critical path;
- shard sampling and MLIP evaluation statically by attempt ID and fill idle GPUs
  dynamically rather than reserving a permanently idle evaluator card;
- T1/T2 use recovery and cheap structural metrics, not full MLIP refinement;
- a variant advances only through the immediately preceding gate;
- all invalid CrysLLMGen-style proposals remain in the denominator;
- debug refiner steps are never paper results;
- final attempts use one shared frozen refiner and identical fixed steps;
- first cuts are full OOD, FWD/FMD, basin calibration, and external retraining;
  never cut the two final methods, three train seeds, 10k denominator, second
  held-out MLIP, or Tier-A symmetry/novelty diagnostics.

## 16. Four-Week Critical Path

| Week | GPU lanes and deliverable | Hard decision |
|---|---|---|
| 1 | P0/P1/P2 on CPU plus four parallel lanes for WQ-DLM/confidence, WQ-AR/D3PM, continuous/joint smoke, and evaluator/recovery; run P3 at 10% data | Day 3 metric/formal freeze; Day 7 DLM continue/stop |
| 2 | 25%-update screen, then one full seed of the best AR/diffusion engine, joint-no-revision, CrysLLMGen-style discrete-once, and preliminary core | Day 11 cut to four; Day 14 freeze discrete engine and baseline champion |
| 3 | P5 core, shared-noise interventions and sampling-only controls; one P6 fine-tune only if P5 passes by Day 17; reuse promoted seed-11 checkpoints | Day 17 core gate; Day 19 P6 keep/drop; Day 21 freeze method, checkpoints and configs |
| 4 | train only missing seeds 23/47 for champion/final, generate pooled 10k/method, run 6k multi-MLIP and 2500 LeMat panels, statistics and artifact audit | Day 24 checkpoints complete; Day 26 generation complete; Day 28 evidence freeze |

No tuning is permitted after Day 21. If P0--P2 are not complete by Day 7, the
oral cycle stops rather than borrowing time from final replication. OOD is
allowed only when all main tables are frozen by Day 24.

## 17. Paper Evidence Layout

- Figure 1: the disjoint union of Wyckoff strata and topology-changing events;
- Figure 2: geometry-adaptive commit, rollback, birth, and death trajectories;
- Table 1: matched MP20 raw/refined/relaxed MLIP-SUN@0.0/@0.1, symmetry,
  prototype/protostructure novelty, and actual compute;
- Table 2: WQ-AR/D3PM/monotonic/CrysLLMGen-style one-way/confidence/geometry-
  adaptive causal matrix;
- Figure 3: recovery interventions and wrong-to-right versus right-to-wrong;
- Figure 4: raw-to-relaxed basin movement and multi-MLIP transfer;
- Figure 5: quality--compute and stability--novelty Pareto curves;
- Table 3: SG/Wyckoff-dimension/atom-orbit-count distribution distances,
  substitution-aware novelty, and LeMat compatibility;
- Appendix: formal tests, all seeds, matcher sensitivity, complete failure
  accounting, canonicalization ambiguity, configs, and artifact hashes.

## 18. Global Kill And Fallback Rules

Stop the oral claim if any condition holds:

1. quotient conversion or formal support-preservation gates fail;
2. high-corruption DLM recovery does not beat matched WQ-AR/D3PM;
3. either directional intervention is null or shuffled feedback matches it;
4. adaptive true remask is matched by fixed schedules, random remasking, or
   extra calls;
5. the core gains under 2 pp development MLIP-SUN@0.1 over the frozen champion;
6. guide-MLIP gains fail to transfer to either held-out MLIP;
7. Novel&Unique drops by over 2 pp, substitution-aware novelty regresses, or
   graph acceptance falls below 95 percent;
8. gains disappear under a lenient matcher, a major material family, or
   orbit-count-unchanged structures;
9. strong rerun-matched external baselines remain over 5 pp ahead;
10. final CI includes zero or compute exceeds 2x without Pareto advantage;
11. the core is matched by the frozen CrysLLMGen-style one-way handoff at equal
    calls, or its gain disappears before the common refiner/MLIP relaxation;
12. Tier-A metric/evaluator contracts or the Day-21 method freeze are missed.

Fallbacks:

- DLM recovery fails but quotient works: run one stratified pilot with the best
  discrete kernel; remove DLM-superiority language.
- topology revision fails: stop the flagship mechanism; retain WQ models only as
  baselines or a later repair/infilling project.
- relaxation consistency fails transfer: remove it without changing the core.
- MP-Doob is used only for documented collapse and never rescues a failed core.

Do not split the flagship into overlapping DLM and co-denoising papers during
this experimental cycle.

## 19. Evaluation Audit Provenance

The metric contract was checked against the primary evaluation protocols of
[MatterGen](https://www.nature.com/articles/s41586-025-08628-5),
[WyckoffDiff](https://arxiv.org/abs/2502.06485),
[SymmCD](https://arxiv.org/abs/2502.03638),
[SGEquiDiff](https://arxiv.org/abs/2505.10994),
[FlowMM](https://arxiv.org/abs/2406.04713),
[MCFlow](https://arxiv.org/abs/2602.20210),
[Chemeleon2](https://www.nature.com/articles/s42256-026-01262-4), and
[CrysLLMGen / LLM Meets Diffusion](https://arxiv.org/abs/2510.23040).
Recent substitution-based and continuous novelty audits motivate the stronger
novelty panel. LLaDA-Rec is retained only for the adaptive-order/step-budget
mechanism tests.

The literature-derived compatibility metrics never override the registered
attempt-level headline. In particular, survivor-only valid panels,
generate-until-novel protocols, filtered invalid LLM compositions, selected
low-energy candidates, and DFT S.U.N. values are reported only as clearly
labeled reference protocols, not mixed with our MLIP-predicted rates.
