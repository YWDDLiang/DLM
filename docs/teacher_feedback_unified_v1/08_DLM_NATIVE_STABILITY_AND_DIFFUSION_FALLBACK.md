# Stability-first safe evolution of the LLM+DLM system

Status: method rationale accepted; preflight execution approved on 2026-09-04.
The active efficiency checklist and formal-training launch conditions are in
[`09_EFFICIENCY_FIRST_POTENTIAL_CLOSURE_PLAN.md`](09_EFFICIENCY_FIRST_POTENTIAL_CLOSURE_PLAN.md).

Optimization amendment (2026-09-04): the scalar `0.5/0.5`, 348-update and
full-site reverse-sweep protocol in the original design has been superseded by
the audited protocol in the active checklist: objective-separated interleaved
updates for 2,048 optimizer steps, fixed first-unique action proposals,
independent gradient probes and one cell plus at most two Llama-anchor
closures. Where this rationale document and the active checklist differ, the
checklist governs execution.

## 0. Decision

The project will evolve toward a DLM-native stable crystal generator without
discarding components that already work.

Two endpoints remain visible during the transition:

```text
Native endpoint:
C3FD-constrained Llama -> scaffold program -> SPAD DLM -> crystal

System endpoint:
the same frozen native crystal -> frozen model494 continuous refinement
```

The native endpoint is where new stability improvements must first appear.
The system endpoint remains the current competitive fallback and is not hidden,
deleted, or weakened before the native endpoint is demonstrably sufficient.
The long-term goal is to reduce and eventually remove the external continuous
refiner, not to remove the discrete masked diffusion language model.

`DLM` below always means the discrete masked diffusion language model over
crystal tokens. `Diffusion` or `model494` means the external continuous crystal
diffusion model.

## 1. Preserved assets

### 1.1 SPAD transaction state machine

Preserve the existing implementation of:

- exact `7+4N` storage;
- exact composition/N prefill;
- lattice-scalar and XYZ transaction scheduling;
- periodic 000/100 alias handling;
- triclinic minimum-image geometry;
- catastrophic-collision rejection;
- suffix-visible non-causal revision;
- one Plan and one trajectory per request.

On the fixed two-stream comparison, raw Direct rises from B0 `401/512 =
80.27%` to BS `511/512 = 99.80%` in the development endpoint. That gain is a
joint effect of composition prefill, schema support, transaction scheduling,
periodic masking and schedule-matched training; it must not be attributed to
one line or one distance rule alone.

No proposed change may bypass the SPAD state machine or weaken exact
composition preservation.

### 1.2 C3FD chemical support

C3FD remains the support function for chemically reachable composition actions.
It preserves exact counts and the existing charge/valence reachability logic
without forcing Llama to become a deterministic enumerator. The current
composition-validity result remains an established contribution.

### 1.3 Llama scaffold pointer

Keep the trained pointer checkpoint. Its teacher is the periodic
maximum-contact-tree of relaxed MP20 structures, not CIF file order. Existing
validation reaches `73.50%` exact permutation, `80.41%` root accuracy and
`82.63%` pairwise order; `229/256` prospective programs are non-canonical.

This proves that Llama can produce and execute a nontrivial scaffold program.
It does not yet prove that the learned program lowers energy. The correct
interpretation is an **equilibrium coordination-scaffold program**, not a
physical nucleation trajectory.

### 1.4 Frozen model494 fallback

Keep model494 and its successful tau800 pipeline intact. For BS it raises the
reported Strict/Meta S.U.N. from raw `16/107` to refined `35/234` over 512
attempts. Earlier G2 evidence also shows a refined paired hull reduction of
approximately 16 meV/atom on one stream.

These are real system-level gains. Until the native endpoint crosses the
required stability level, model494 remains part of the reportable complete
system.

## 2. Scientific target

For composition `c`, Direct validity describes a broad feasible set:

\[
\mathcal V_c=
\{(L,X):\operatorname{comp}(L,X)=c,\ V(L)>0,
d_{\min}^{\rm PBC}(L,X)>d_0\}.
\]

The desired low-energy region is much smaller:

\[
\mathcal B_c(\delta)=
\{(L,X)\in\mathcal V_c:
E_{\rm hull}(L,X,c)\le\delta,
\|F(L,X,c)\|\approx0,
\|\sigma(L,X,c)\|\approx0\}.
\]

The current system has mainly increased \(P_\theta(\mathcal V_c)\). The new
work must increase

\[
P_\theta(\mathcal B_c(\delta)\mid\mathcal V_c,c)
\]

while preserving novelty and uniqueness. Stability is therefore the primary
objective; Direct is a safety and execution metric rather than a substitute for
stability.

The current official metric defines Strict stability by
\(E_{\rm hull}\le0\), Meta stability by
\(E_{\rm hull}\le0.1\) eV/atom, and S.U.N. by intersection with novelty and
uniqueness. It does not test phonons, kinetic barriers, finite-temperature
free energy or experimental synthesizability. The paper should target relaxed
low predicted-energy periodic structures and not claim actual nucleation
dynamics.

## 3. Repository-aware audit of the proposed changes

### 3.1 Lattice closure is not authorized as a zero-shot inference trick

The concern about predicting

\[
L^{(1)}\mid X^{(0)}
\]

out of distribution is valid. Ordinary masked DLM training can randomly mask
lattice tokens, and the current schedule data include lattice prediction
states, but the training distribution does not deliberately and frequently
present **all six lattice values masked while every site is visible**. A
zero-shot full-cell remask is therefore unsafe.

If cell closure is tested later, it must first receive an explicit MP20
training mask class with exactly the same state used at inference. The native
lattice and coordinates remain unchanged until that training is complete.

### 3.2 A periodic adapter is not a new-from-scratch GNN

The repository already contains:

- `periodic_relation_adapter.py`;
- `periodic_relation_runtime.py`;
- periodic metric/RDF/overlap/coordination objectives;
- element-radius tables and triclinic 27/125-image operators;
- trained G1/G2 checkpoints and matched evaluations.

Therefore the earlier draft overstated the engineering gap in one direction,
while the new audit overstates it in the other. Reusing an existing tested
operator is possible; inventing another geometry network is unnecessary.

The evidence boundary remains important: G2 improved one refined endpoint but
did not establish a reliable raw-energy gain. Existing geometry code is an
engineering asset, not automatically a new main contribution.

### 3.3 A continuous residual output head is genuinely new and high-risk

The existing periodic relation adapter modifies hidden states and logits; it
does not emit deterministic sub-token coordinates or a continuous lattice.
Adding a new regression head would introduce a new output contract, losses,
checkpoint fields, serialization and evaluation path.

It is therefore deferred. It becomes justified only if the representation
ceiling audit demonstrates that current token quantization itself materially
damages energy, force/stress or stable-status retention.

### 3.4 Fine lattice ownership must be stated correctly

The current Llama/C3FD Plan supplies soft lattice-system, space-group and
volume-per-atom information. The six fine lattice tokens are generated by the
DLM, not directly by Llama. Any deterministic volume correction would act on
the DLM cell using an existing Plan prior; it is not a correction to a
Llama-generated cell.

### 3.5 Low-tau model494 must be interpreted using the actual sampler

The current model494 sampler begins from the supplied clean proposal while
setting the reverse-time index to `tau`; it does not first apply the textbook
forward Gaussian corruption described in the audit. Consequently, reducing
tau means executing a shorter, lower-time stochastic reverse transition. It
must not be described as reducing an explicitly applied 90% forward-noise
mixture.

Historical matched evidence also shows that tau200 produced less S.U.N. than
tau800 on the old poor-quality raw cohort. A low-tau test on the new high-Direct
SPAD body is reasonable, but tau150 cannot replace tau800 by argument alone.

## 4. Current information flow and model responsibilities

The method must be designed around what the current code actually computes.

### 4.1 C3FD: chemical base distribution and reachable support

C3FD supplies calibrated base logits for:

- proposal stratum: family, atom count and number of species;
- each oxidation-state species/count action;
- lattice system, space-group bucket and volume-per-atom bin.

At every composition step it also constructs the legal action mask that
enforces the remaining atom-count, family, charge/valence and benchmark
reachability constraints. C3FD is therefore authoritative for **what chemical
actions may still lead to a complete composition**. It has no fine lattice or
coordinate logits.

### 4.2 Llama: preference inside support and scaffold programming

The current Planner Llama does two distinct jobs.

First, typed C3FD states are mapped into Llama embeddings. Llama emits
zero-initialized residual logits over the same proposal, species/count and soft
field actions. C3FD and Llama are normalized on the same legal support and
combined as a unit-weight product of experts:

\[
\log p(a\mid s)
\propto
\log p_{\rm C3FD}(a\mid s)
+\log p_{\rm Llama}(a\mid s),
\qquad a\in\mathcal A_{\rm chem}(s).
\]

Thus Llama changes the probability of legal compositions; it never reopens an
illegal action.

Second, a pointer head on the terminal Llama state predicts a permutation of
the final Plan elements. This `species_program` controls the order in which the
DLM establishes species anchors and later revisits them.

The current Planner was nominally conditioned on `meta_or_better`, but all
eligible training rows already belong to that tier. It therefore learned a
near-stable MP20 support prior, not a high-versus-low energy classifier. The
paper must not claim that the current Llama predicts hull energy.

### 4.3 The scientific program passed to the DLM

The current public Compact V2 Plan contains:

- hard `N`, `elements`, `counts` and anion family;
- soft LS, SG bucket and VPA bin;
- a separate `species_program` metadata field.

The proposed runtime program retains the internal oxidation-state/count trace
as additional typed metadata rather than adding another verbose text schema.
For mixed-valence elements it is an aggregate chemical prior, not a claim that
every generated site has a known oxidation state.

The program answers three questions:

1. which composition has been chemically certified;
2. which global lattice regime is plausible;
3. in which scaffold order the DLM should resolve and revisit the sites.

### 4.4 DLM: fine lattice and coordinate realization

The DLM owns every fine structural variable in the exact body:

\[
z=(N,L_1,\ldots,L_6,(E_i,X_i,Y_i,Z_i)_{i=1}^{N}),
\qquad |z|=7+4N.
\]

Exact `N` and element slots are prefixed from the Plan. The DLM predicts the
six lattice tokens and all coordinates. SPAD uses the Llama program to choose
the transaction order, enforces the periodic state machine, and permits
suffix-visible revision. This is where global chemical intent must be converted
into an actual periodic crystal.

### 4.5 External diffusion: retained system endpoint

Frozen model494 jointly changes lattice and coordinates after a complete native
body exists. It remains the current fallback and compatibility endpoint, but it
does not define the new DLM training target.

## 5. Scientific design principles for composition, lattice and coordinates

### 5.1 Composition is a support problem

Composition determines atom identities, multiplicities, charge/valence
feasibility and the set of competing structures. It does not determine a
unique stable polymorph. Chemical rules can therefore exclude unreachable
actions, while Llama should retain a learned distribution over the reachable
set.

Energy is not compared across arbitrary compositions: elemental references and
hull baselines differ. Stability learning begins only after `c` is fixed.

### 5.2 Lattice is a global stress-bearing variable

The lattice defines

\[
G=L^TL,\qquad V=|\det L|,\qquad R=XL.
\]

Changing one lattice component changes every Cartesian pair distance and the
cell stress. Six independently plausible scalar tokens can still form an
inconsistent joint cell. Lattice stability must therefore be learned and
evaluated as a complete six-token transaction.

VPA, LS and SG are useful global priors but are not energy. They may constrain
or softly score lattice proposals; the final stability credit must come from a
complete reconstructed structure.

### 5.3 Coordinates define periodic coordination under the chosen lattice

Fractional coordinates have physical meaning only together with `L`:

\[
d_{ij}^{\rm PBC}
=\min_{n\in\mathbb Z^3}
\|L(x_i-x_j+n)\|_2.
\]

The existing `0.5 A` rule removes catastrophic overlap. Species-aware distance,
coordination and valence information can identify additional risk, but no
pairwise rule captures the full many-body energy. XYZ must therefore be one
complete action, and its utility must be measured after reconstructing the
whole crystal.

### 5.4 Stability is a conditional potential problem

For fixed composition `c`, the proxy hull energy can be written as

\[
\widetilde E_{\rm hull}(L,X;c)
=\widetilde E(L,X;c)-H(c),
\]

where `H(c)` is constant across candidate structures in the same group.
Consequently, within-composition raw CHGNet energy ordering is aligned with its
proxy hull ordering. CHGNet is still an approximation to DFT and cannot replace
the final official calculation, but it provides a direct many-body training
signal that VPA and minimum distance do not.

## 6. Phase 0: determine the actual bottleneck

Three analyses run in parallel before method training.

### 6.1 Oracle representation ceiling

Start from untouched continuous MP20 validation structures. Encode and decode
them through the exact current `7+4N` representation without learned
generation. Require codec/parse/composition closure on all 9,047 rows, then use
one fixed stratified 512-row subset for paired potential and fast-validity
diagnostics. Compare original and quantized structures using:

- paired CHGNet energy per atom;
- force RMS and maximum force;
- stress norm;
- volume, metric strain and angle changes;
- PBC minimum-distance changes and Direct flips;
- Strict/Meta status only on the subset supported by an existing compatible
  phase-diagram cache.

For this cached diagnostic only, preserve the fixed-composition DFT hull
reference and add the paired CHGNet quantization delta to the cached DFT hull
value. Report both Strict and Meta retention, but gate only on Meta retention
because the exact-zero Strict boundary is unusually sensitive to proxy error.

Report distributions and tails, not only mean energy. A `15 meV/atom` change is
a useful reference near S.U.N. thresholds, not a stand-alone decision rule.

If the oracle remains near its original basin, do not build a continuous
residual head or new vocabulary. If quantization itself creates large
force/stress tails or many stability flips, representation precision becomes a
separate later problem; it is not silently mixed into the first training run.

### 6.2 Existing native-error decomposition

Use already generated BS raw bodies to determine whether high energy is mainly
associated with:

- VPA or lattice-metric mismatch and stress;
- species-normalized short distances;
- under/over-coordination and RDF deviation;
- valid geometry with no simple local explanation.

Measure relations within composition/family strata so a few extreme collisions
or common chemistries do not determine the conclusion.

### 6.3 Program and prior coverage

On full MP20-train, audit:

- LS/SG/VPA calibration and held-out coverage;
- oxidation/valence trace coverage;
- contact-tree program coverage;
- element/valence pair-distance support and rare-pair backoff.

These statistics decide which program metadata are reliable. They do not become
hard chemistry rules automatically.

## 7. Main method: Scientific-Programmed Potential-Closed Denoising

The method is one hierarchy:

\[
\boxed{
\mathcal A_{\rm chem}
\longrightarrow
P_{\rm Llama}
\longrightarrow
\mathcal A_{\rm struct}
\longrightarrow
q_E^*(a\mid s,c)
}
\]

It answers, in order: what can be generated, how it should be constructed, what
constitutes a legal structural action, and which legal action is lower energy.

### 7.1 Hard support and soft scientific risk

Hard support remains limited to properties whose violation is definitively
invalid:

- exact N/elements/counts;
- legal token family and complete action length;
- positive, finite cell/Gram matrix;
- catastrophic PBC distance below the existing Direct boundary.

VPA deviation and valence/element-conditioned pair geometry are soft risks.
They help propose or regularize transactions but do not carry the primary
stability claim. Their parameters are estimated on MP20-train and backed off
for rare chemistry.

In the first 2,048-group experiment they are diagnostics and candidate-
construction metadata, not an additional inference-time logit tilt. A soft
runtime prior is added only if Phase 0 shows a stable association with energy,
force or stress after conditioning on chemistry. This prevents two unverified
pairwise heuristics from being bundled with the first potential-learning result.

### 7.2 Deployment-matched closure states

No zero-shot lattice closure is allowed. The DLM first receives explicit
training states that exactly match deployment visibility and action semantics.

Two state types are used:

1. **Cell closure:** all sites visible; the complete six-token lattice block is
   masked and predicted.
2. **Site closure:** lattice and all other sites visible; one complete XYZ block
   is masked and predicted. Deployment revisits only the first two
   distinct-species anchors selected by the Llama program, in reverse program
   order; unary structures revisit one anchor.

“Deployment matched” means identical visible tokens, mask block, program order
and action definition. It does not claim that MP20 contexts and generated DLM
contexts have identical marginal distributions.

To reduce that remaining context shift, the first pool contains equal numbers
of two state sources:

1. **MP20-restoration states:** relaxed MP20 structures receive multi-block
   perturbations whose lattice strain, displacement and contact statistics are
   calibrated to the existing BS error audit. The clean MP20 block is available
   as a teacher action.
2. **On-policy states:** the frozen BS DLM generates complete structures for
   MP20-train compositions. These generated structures define deployment-like
   states but never become positive teachers; their action preference comes
   only from within-state raw energy comparisons.

### 7.3 Complete-transaction potential posterior

Use only MP20-train compositions and structures. Each source row is assigned to
one state. The efficiency-first study contains 2,048 groups:

- 512 MP20-restoration cell-closure states;
- 512 MP20-restoration site-closure states;
- 512 on-policy cell-closure states;
- 512 on-policy site-closure states.

The on-policy half reuses the already generated MP20-train programs and BS
predictor bodies from job 39556. Its confounded model494 endpoints and labels
are not reused. This avoids another full predictor generation pass while
preserving deployment-like native contexts.

Before masking, every state stores its current complete lattice or XYZ block as
the provisional block. `a_noop` means restoring that exact provisional value.
For an MP20-restoration state, `a_teacher` restores the corresponding clean
relaxed MP20 block. If teacher and no-op are identical after tokenization, they
are recorded as a duplicate and cannot create an artificial preference. An
on-policy state has no clean teacher action.

Each group retains at most four candidates. Fixed no-op/teacher actions are
inserted first, followed by the first distinct legal DLM proposals in request
order. The proposal temperature is frozen, at most eight request-keyed
proposals are drawn, and neither temperature nor proposal count responds to
energy. Thus the candidate sets are:

\[
\mathcal K_{\rm restore}(s)=
\{a_{\rm noop},a_{\rm teacher},a_{\rm DLM}^{(1)},a_{\rm DLM}^{(2)}\},
\]

or

\[
\mathcal K_{\rm onpolicy}(s)=
\{a_{\rm noop},a_{\rm DLM}^{(1)},a_{\rm DLM}^{(2)},a_{\rm DLM}^{(3)}\}.
\]

For every action:

1. insert the complete three- or six-token block;
2. reconstruct the entire `(c,L,X)` structure;
3. remove definitively invalid actions from support;
4. compute raw CHGNet energy, force and stress;
5. compare energy only inside the common composition and state.

Energy is measured in eV/atom. For action `a`, define

\[
\Delta E_a=E_a-E_{\rm noop}.
\]

Within each group, finite legal energies are median-centered and divided by a
MAD scale with the existing robust fallback and clipping. `q_ref` is the
current BS DLM probability over the retained complete actions. An action score
is the sum of the three or six conditional token log probabilities, matching
the probability of the complete transaction. Cell and site groups never
compete in the same softmax; their group losses are divided by transaction
length to normalize optimizer scale. The target posterior is

\[
q_E^*(a\mid s,c)
\propto
q_{\rm ref}(a\mid s,c)
\exp[-\eta_g\,\widehat{\Delta E}_a],
\qquad a\in\mathcal A_{\rm struct}(s,c).
\]

The group-specific tilt `eta_g` is obtained by the existing bisection routine so
that

\[
D_{\rm KL}(q_E^*\Vert q_{\rm ref})\le0.05\ \text{nat}.
\]

There is no free beta sweep. Illegal actions have zero support. Energy-unknown
actions retain no preference advantage. Duplicate action draws are merged
before normalization, so they cannot manufacture either reference mass or an
energy difference. Groups with fewer than two distinct legal energy-known
actions, or with less than 1 meV/atom energy spread, remain in accounting but
receive zero training weight.

Force and stress are not separate first-run losses. Previous force projection
was quantization-sensitive and failed to improve the realized student. They are
measured as basin diagnostics: an energy gain accompanied by adverse force or
stress is not interpreted as successful physical closure.

This differs from retired methods:

- it is not single-token Q prediction;
- it does not copy a continuous force into a quantized token;
- it does not use post-model494 energy;
- it does not train on positive rows without same-state alternatives;
- it assigns value to complete deployed lattice/XYZ transactions.

### 7.4 Objective-separated preservation and potential training

Continue from the existing schedule-matched SPAD DLM rather than rebuilding a
new generator. Do not add clean CE and posterior losses in one scalar mixture.
Use a fixed four-update cycle:

1. clean MP20 SPAD CE;
2. cell-transaction posterior;
3. clean MP20 SPAD CE;
4. site-transaction posterior.

Repeat 512 cycles for 2,048 optimizer updates. On-policy states receive only
posterior supervision; clean CE is computed only on MP20 teacher structures.
This gives the cell and site posterior pools three group exposures each while
clean CE occupies half of all optimizer updates. The first 100 total updates
warm up linearly to LR `5e-6`, after which the LR remains fixed. Release only
update 2048.

The equal-compute closure-only control uses the same initialization, optimizer,
update count and cell/site schedule, but its transaction slots use clean MP20
transaction-restoration CE and never consume energy labels or on-policy
generated structures. This preserves a direct attribution: the two models
differ in whether complete transactions receive same-composition potential
supervision.

Before either formal run, independently backpropagate clean CE, cell posterior
and site posterior on five fixed batches without updating weights. Require
finite, nonzero gradients, posterior/CE median norm ratios in `[1e-2,1e2]`,
median cosine similarity above `-0.5`, and target KL no greater than `0.05`
nat. The probe diagnoses implementation or data failures; it never tunes loss
weights, LR, KL or epoch count.

If the 2,048-group study produces a genuine native stability shift, extend the
same frozen construction to full MP20 and train **two independent full-data
seeds**. The pilot seed does not count as one of those replications. The pilot
is mechanism evidence, not the final scale claim.

### 7.5 Native inference

Inference remains one program and one trajectory:

1. C3FD-supported Llama samples one certified composition and one scaffold
   program.
2. Existing SPAD produces one complete native body.
3. The trained DLM executes one cell closure.
4. It revisits the first two distinct-species Llama anchors once, in reverse
   program order; a unary structure revisits one anchor.
5. The resulting native crystal is emitted without CHGNet, reranking,
   replacement or best-of-N.

### 7.6 Conditional pointer energy alignment

The current pointer checkpoint remains the starting point. Pointer DPO is not
part of the first critical path.

First freeze composition and Plan. Random variates are keyed by request, site
handle, token family and draw rather than by decode-step count, so changing
program order does not silently remap the random stream. Test whether distinct
programs cause a reproducible raw-energy difference. If the program effect
exceeds repeated-run noise, construct same-composition program pairs and make
one small DPO update to the existing pointer. Composition logits are not trained
from cross-composition absolute energy.

If the effect is not identifiable, keep the pointer as an executable
equilibrium scaffold prior and do not call it energy-aware.

### 7.7 Continuous refinement remains a dual endpoint

Freeze the native bodies before continuous refinement. Report separately:

- DLM-native;
- model494 tau200 as the single predeclared low-tau sensitivity;
- frozen model494 tau800 fallback.

No sample is selected between endpoints. Tau200 only tests whether a stronger
native proposal requires less continuous correction. Existing tau200 evidence
is weaker than tau800 on the historical poor-body cohort, so it is not assumed
superior and no third tau is added.

## 8. Implementation reuse and new engineering

### Reused directly

- C3FD reachability, calibration and typed states;
- Llama typed-residual PoE and current LoRA;
- current scaffold pointer checkpoint;
- C3FD Native Plan V2 and exact composition prefill;
- SPAD generation, MIC, aliases and suffix-visible site revision;
- physical corruption utilities;
- periodic metric/RDF/overlap/coordination operators;
- CHGNet `efsm` batching with explicit job-local device;
- KL-bounded four-action posterior primitives;
- native, Direct, official and model494 evaluation paths.

### New but bounded

- preserve typed valence trace in runtime program metadata;
- production six-token cell-closure mask/action path;
- one formal 2,048-group closure dataset builder;
- generalize the K4 trainer from fixed XYZ length three to transaction length
  three or six, with length-normalized scores;
- mix clean SPAD CE with transaction-posterior updates;
- thin wrapper that sends the same frozen native body to native, low-tau and
  tau800 evaluations.

### Explicitly deferred

- new GNN or geometry adapter architecture;
- continuous residual coordinate/lattice head;
- token-vocabulary expansion;
- online CHGNet guidance, MCTS, reranking or best-of-N;
- new model494 checkpoint or tau sweep;
- joint end-to-end Llama/DLM backpropagation.

## 9. Efficiency-first execution sequence

The critical path contains only four stages. Pointer DPO, full-MP20 expansion
and low-tau sensitivity are removed from the first result path.

### Stage A — one compact Phase 0 package, 0.5--1.5 hours

Run the three read-only analyses together:

- MP20-val representation ceiling;
- existing BS native-error decomposition;
- MP20-train program/VPA/pair coverage.

Use one batched CHGNet job and CPU workers. No generation or training waits on
an expensive Direct calculation.

### Stage B — implement only the required interfaces, 2--3 hours

Implement and test:

- six-token cell closure;
- shared three/six-token K4 schema and length-normalized scorer;
- objective-separated 2,048-update interleaved trainer and five-batch gradient
  probe;
- one thin native evaluator.

Do not implement Pointer DPO, new geometry layers, residual heads, vocabulary
changes or low-tau wrappers during this stage.

### Stage C — build 2,048 groups and train two cells, 1--2 hours

- Reuse job39556 MP20-train programs and BS predictor bodies for the 1,024
  on-policy groups.
- Build 1,024 MP20-restoration groups from the same train-only source domain.
- Generate at most eight fixed-temperature proposals per group on four A800,
  retaining the first distinct legal actions in request order.
- Label 8,192 raw candidates through the verified explicit-device CHGNet path.
- Train closure+clean-CE and potential-closed cells concurrently, one A800
  each, for 2,048 updates after the gradient probe passes.

This consumes the labelled pool approximately once and avoids producing a
27,136-row pool that the pilot would barely read.

### Stage D — one-stream native decision, 1.5--2.5 hours

Use one frozen 256-composition stream and compare:

1. existing programmed SPAD baseline;
2. closure+clean CE;
3. potential-closed DLM.

Compute parse/composition, raw CHGNet energy/force/stress and cached official
S.U.N. first. Run fast structural validity concurrently; defer expensive full
Direct components that do not affect the decision.

### Conditional completion after a native positive

Only if potential closure improves the native stability direction over the
closure-only control:

1. run the second fixed stream;
2. run frozen tau800 on the same native bodies;
3. expand the unchanged method to full MP20 and two full-data seeds;
4. consider Pointer DPO after program-value identifiability is shown;
5. run tau200 only if the native gain is erased by tau800.

Expected wall time:

- first native answer: **6--9 hours**;
- two-stream native plus tau800 fallback: **8--12 hours**;
- full-data/two-seed paper endpoint: additional work after the mechanism is
  positive, not part of the first overnight critical path.

The one-seed pilot is not sufficient for a final paper claim. Its purpose is to
decide whether the mechanism creates a large enough native signal to justify
full-data, two-seed replication.

## 10. Expected improvements and failure interpretation

### 10.1 Why this has a stronger expectation than validity tightening

- Chemical support preserves composition rather than trying to learn hard
  conservation from examples.
- Llama changes the global scaffold program rather than predicting individual
  coordinates.
- Cell closure gives the DLM a trained view of `L` after complete `X` exists.
- Site closure uses final lattice and future sites.
- Whole-transaction CHGNet energy captures attraction, repulsion,
  coordination and many-body effects that VPA or a distance cutoff cannot.
- Within-composition comparison is aligned with the direction of proxy hull
  improvement.

The first expected improvements are lower stress, force and raw energy. Raw
Meta S.U.N. should respond before Strict because the strict boundary is harder.
The current BS refined result is `35/512` Strict and `234/512` Meta; reaching
the target requires a net `+17` and `+22`. The 2,048-group study can establish
direction, while a large threshold shift may require the full-MP20 expansion.

The pilot evidence standard is paired rather than based on a single aggregate
mean:

- composition-cluster bootstrap for raw energy, force and stress;
- fixed-denominator Strict/Meta changes and paired discordances;
- Direct and N/U/NU disclosed on the same attempts;
- intervention accounting showing which complete transactions changed.

A useful mechanism signal is a paired raw-energy interval below zero together
with non-adverse force/stress and a positive S.U.N. direction. These are
evidence standards, not a rule for deleting an unfavorable result.

### 10.2 Principal risks

The largest remaining scientific risk is context shift:

\[
p_{\rm MP20\ closure}(s)
\ne
p_{\rm generated\ closure}(s).
\]

The DLM may learn to repair one block when the rest of an MP20 structure is
nearly correct, yet fail when several parts of a generated structure are wrong.
Other risks are:

- K4 candidates may contain little useful energy variation;
- one cell plus two anchor closures may be insufficient for collective modes;
- CHGNet improvement may not cross official hull thresholds;
- energy preference may reduce novelty;
- tau800 may erase native gains;
- Llama program energy may be unidentifiable.

Interpretation is direct:

- validity-only improvement is not a stability result;
- lower force/stress without energy improvement identifies incomplete
  coordination recovery;
- lower raw energy with unchanged official S.U.N. identifies proxy/threshold
  mismatch;
- native improvement erased by tau800 supports reducing external refinement,
  not denying the native effect;
- no native improvement stops pointer DPO and full-data expansion.

## 11. Paper framing

### 11.1 Scientific question

Given composition `c`, how can a language-based crystal generator turn
chemical reachability and complete periodic-structure potential into LLM-
programmable, DLM-executable and revisable decisions, so that probability mass
moves from merely valid structures toward low-energy basins without
inference-time energy search?

### 11.2 One core method

> **Scientific-programmed potential-closed denoising:** C3FD constrains the
> chemical support on which Llama selects a scaffold program; a non-causal DLM
> executes complete lattice/site transactions; within-composition potential
> supervision trains those same transactions toward valid, lower-energy native
> crystals.

### 11.3 Overall system contributions and this revision's new method

The complete paper can organize its system into three connected contributions,
but their evidence provenance remains explicit:

1. **Scientific support-to-program compilation.** C3FD changes the reachable
   Llama action distribution; Llama retains learned composition preference and
   compiles the certified state into an executable equilibrium scaffold
   program.
2. **Programmed non-causal crystal transactions.** SPAD uses that program to
   resolve and revisit complete lattice/XYZ blocks in the exact `7+4N` DLM,
   exploiting future-visible remasking while preserving chemistry and periodic
   legality.
3. **Potential-closed transaction learning.** Raw same-composition many-body
   energy trains the complete actions that construct the crystal; inference
   uses one program and one trajectory, while model494 remains a compatible
   system endpoint rather than the source of DLM credit.

Items 1 and 2 are preserved project assets with existing evidence. **Item 3 is
the sole new method contribution proposed by this revision.** The new result
must not relabel the earlier C3FD or SPAD experiments as if they were produced
by potential closure.

The contributions are one decision chain:

\[
\boxed{
\text{what is chemically reachable}
\rightarrow
\text{what order should resolve structure}
\rightarrow
\text{which complete legal action is lower energy}
}
\]

### 11.4 Main-paper evidence budget

Keep only three pieces of main evidence:

1. one end-to-end table with composition validity, native Direct/S.U.N. and
   fixed-refinement S.U.N./N/U;
2. one program intervention with fixed composition, Plan and request/site
   randomness, showing that Llama changes actual structures;
3. one mechanism panel comparing programmed SPAD, closure+clean CE and
   potential-closed DLM on raw energy, force, stress, S.U.N. and validity.

Quantization, VPA/pair diagnostics, extra tau settings and engineering failures
belong in the appendix or repository.

The existing pure-LLM/CrysLLMGen endpoint is retained as architectural context
rather than triggering another AR training campaign. The causal program panel
is the direct test of the Llama-to-DLM claim; a broad new AR factorial is not on
the critical path. Without a matched AR implementation of the same closure, the
paper claims only that a DLM is **architecturally suited** to future-visible
remasking and past revision. It does not claim empirical proof that every AR
alternative is inferior or incapable.

### 11.5 Relation to published systems

- [MatterGen](https://www.nature.com/articles/s41586-025-08628-5) jointly
  models atom types, periodic coordinates and lattice, and evaluates proximity
  to relaxed minima. It supports treating `(c,L,X)` as coupled rather than
  independent text fields.
- [DiffCSP](https://proceedings.neurips.cc/paper_files/paper/2023/hash/38b787fc530d0b31825827e2cc306656-Abstract.html)
  shows that joint lattice/coordinate generation and periodic equivariance are
  central to crystal structure prediction.
- [CrysLLMGen](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f789a628fca473e922c806657512a20f-Abstract-Conference.html)
  motivates retaining a hybrid continuous fallback while the language model
  owns discrete composition and proposal generation.
- [CrystaLLM](https://arxiv.org/abs/2307.04340) demonstrates that an energy
  predictor can guide language-based crystal generation through search. The
  proposed distinction is to distil potential preferences into complete DLM
  transactions, avoiding inference-time tree search.

## 12. Decision log

| Decision | Alternative | Resolution |
|---|---|---|
| Preserve C3FD, Pointer, SPAD and tau800 | rebuild the pipeline | rejected; established assets remain fixed |
| Train complete lattice/XYZ actions | single-token Q or force target | accepted; prior local-credit routes were negative |
| Use raw same-composition CHGNet | post-model494 energy | accepted; removes refiner random-credit confounding |
| First build 2,048 groups | immediate 27,136 x K4 | accepted; supplies three posterior exposures per cell/site pool under the 2,048-update interleaved budget and reuses job39556 native bodies |
| Explicitly train cell closure | zero-shot `L given X` | accepted; avoids an OOD inference trick |
| Keep VPA/pair geometry soft | hard 0.75 radius cutoff | accepted; pair priors are not universal energy laws |
| Conditional Pointer DPO | mandatory hierarchical value training | accepted; no training without identifiable program value |
| Keep tau800 | immediately replace with tau150 | accepted; low tau remains one paired sensitivity only |
| Defer new adapter/residual/vocab | implement all at once | accepted; Phase 0 must first show need |

Structured internal review disposition: **APPROVED** after revision. This only
approves the design and execution order. It is not experimental evidence and
does not imply that the first pilot reaches `10%/50%`.

## 13. Claim boundaries

Do not claim actual nucleation dynamics, exact Boltzmann sampling, phonon or
kinetic stability, experimental synthesizability, an energy-optimal Llama
program, end-to-end Llama/DLM backpropagation, or that model494 gains were
produced by the native DLM.

The intended claim is narrower and stronger: scientific support, learned
programming and same-composition potential supervision can be compiled into one
LLM-guided DLM that emits more stable native periodic structures while retaining
an established continuous-refinement fallback.
