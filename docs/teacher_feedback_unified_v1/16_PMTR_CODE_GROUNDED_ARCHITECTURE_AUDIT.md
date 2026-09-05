# PMTR code-grounded architecture audit and collaboration contract

Status: **code audit complete; PMTR implementation and GPU execution not started**

This document is the shared source of truth for implementing
[PMTR](15_PMTR_SCIENTIFIC_METHOD_AND_EXECUTION.md). It records the code that was
actually inspected, the capabilities that exist today, the interfaces that are
missing, and the smallest coherent implementation path.

## 1. Collaboration rules

These rules override informal recollection from prior discussions.

1. **Code before memory.** Any claim that a capability exists must cite its
   implementation and tests in the current branch. A historical document or
   prior experiment is not evidence that the current runtime has that feature.
2. **Write decisions here.** New architecture decisions, rejected alternatives,
   execution state, and scientific interpretation are appended to this document
   or the PMTR method document before they become instructions.
3. **Keep lineage explicit.** Current C3FD/Planner-Llama/SPAD assets are separate
   from historical R03, G2, BTRD, Force-Score, D3PO, K10, and PCTP work.
4. **One mechanism test, one formal run.** Expensive inference is not used for a
   collection of small ablations. Analysis and unit tests remove design risk;
   one integrated train-only preflight checks transfer; one fixed prospective
   run evaluates the method.
5. **Build reusable interfaces.** PMTR is implemented as general corruption,
   repair-context, logit-transform, and training interfaces rather than a chain
   of job-specific scripts.
6. **Keep controls proportional.** Preserve only essential scientific
   invariants, run configuration, terminal status, and compact metrics. Do not
   add repeated certificate layers, large SHA ledgers, or result-independent
   hard gates.
7. **No MLIP at inference.** CHGNet is permitted only in offline MP20-train
   corruption certification. The production runtime cannot import, call, or
   receive output from CHGNet or another MLIP.

## 2. Audited repository state

- Branch: `codex/llama-programmed-basin-closure`.
- PMTR design baseline: commit `3c7f8df`.
- The working architecture is C3FD compact Plan -> Llama species program ->
  SPAD predictor and basin closure -> optional frozen model494 tau800.
- No PMTR source module, checkpoint, dataset, Slurm job, or result currently
  exists.

The audit examined the current implementations in:

- `src/crystal_dlm/c3fd_native_plan.py`;
- `src/crystal_dlm/species_program_pointer.py`;
- `src/crystal_dlm/spad_program.py`;
- `src/crystal_dlm/spad_generation.py`;
- `src/crystal_dlm/llada_generation.py`;
- `src/crystal_dlm/dynamic_crystal.py`;
- `src/crystal_dlm/periodic_geometry_ops.py`;
- `src/crystal_dlm/periodic_geometry_objective.py`;
- `src/scripts/llada_sft.py`;
- `scripts/build_spad_basin_closure_sft_data.py`;
- their corresponding `tests/test_*` files.

## 3. What exists today

### 3.1 Compact Plan and exact chemistry

`c3fd_native_plan.py` defines `C3FD_NATIVE_PLAN_V2` with exact `N`, elements,
counts, anion family, lattice system, space-group bucket, and volume-per-atom
bin. `_payload` checks `N in 1..20`, unique canonical species/counts, and exact
count conservation. `build_native_inference_prompt` reuses the same renderer as
training.

PMTR does not change this schema. No force, energy, repair, or historical rich
field is added to the Plan text.

### 3.2 Llama-conditioned executable species program

`species_program_pointer.py` implements `PlanConditionedSpeciesPointer`.
Its input combines:

- the final Llama hidden state;
- element and count embeddings;
- the three compact structural fields;
- the already selected program prefix.

It emits only an exact permutation of available Plan species. The pointer cannot
change composition. Its deterministic MP20 teacher is a maximum-contact-tree
species order and does not read energy or hull.

`spad_program.py::program_from_element_order` compiles this permutation into
native site slots and rejects duplicates, omissions, and non-Plan species.
`reverse_species_block_revision_slots` converts it into the actual reverse
repair order.

Therefore the Llama program is already an executable control variable. PMTR
does not add a schedule head or claim interactive real-time control.

### 3.3 Dynamic crystal language

`dynamic_crystal.py` owns the exact body contract:

\[
1\ N + 6\ L + N(A,X,Y,Z)=7+4N.
\]

`llada_generation.py` and `sample_llada_dynamic_crystals.py` already expose
token-id/value maps for coordinate, length, and angle families. They also apply:

- schema-family masks;
- positive/nondegenerate lattice checks;
- periodic 0/1 coordinate alias aggregation;
- duplicate-coordinate masking;
- strict PBC minimum-distance support.

These maps are sufficient to build a general continuous-to-token renderer.
PMTR must not introduce a second tokenizer or representation.

### 3.4 SPAD predictor and basin-closure state machine

`spad_program.py::spad_predictor_position_groups` resolves:

1. hard-prefilled `N` and species;
2. six lattice components;
3. one anchor site for each Llama-program species;
4. remaining sites.

The predictor uses scalar coordinate components so that lattice and PBC support
can reject an illegal later component before commitment.

The current basin closure is implemented by:

- `spad_generation.py::revise_spad_cell`;
- `spad_generation.py::revise_spad_species_blocks`.

Important implementation detail:

- a cell proposal is predicted component-by-component but the old complete cell
  is restored if the final six-token proposal leaves geometric support;
- a species block is fully remasked, sites are visited in supplied order, each
  site's X/Y/Z are predicted sequentially, and the old XYZ or complete block is
  restored atomically when support fails;
- lattice and non-active species remain visible;
- suffix visibility is real, not a document-only claim.

PMTR must preserve this state machine. “Joint XYZ prediction” means one
continuous XYZ repair vector is computed and cached for the site, then used
during the existing X/Y/Z component predictions before atomic commit. It does
not mean replacing the current sequential token sampler with an untested
three-token sampler.

### 3.5 Existing training support for coherent visible errors

`llada_sft.py::JsonlSftDataset` already accepts a clean `answer` and a different
`source_answer`. `validate_paired_dynamic_ids` requires both bodies to preserve:

- identical prompt;
- exact `7+4N` length;
- identical `N` token;
- identical element token at every native slot.

`forced_rollout_process` can keep corrupted source geometry visible while
masking only the active repair positions and computing loss only on the desired
clean transaction.

This is the correct data path for PMTR. A new trainer should extend it rather
than implement another tokenizer/collator stack.

### 3.6 Reusable geometric infrastructure

`periodic_geometry_ops.py` provides differentiable 27/125-image triclinic
minimum distances. `periodic_geometry_objective.py` already contains:

- token family support tables with numeric values;
- lattice construction from lengths and angles;
- PBC pair distances;
- differentiable metric, pair/RDF, overlap, and coordination utilities.

The old G2 weights and runtime are excluded, but these stateless geometry and
token utilities are general code and may be reused after moving any required
primitive into a neutral module.

### 3.7 Hidden-state capture is feasible but not yet general

The base generation helper `_model_logits` returns logits only. PMTR therefore
cannot currently consume the final DLM hidden state.

`periodic_relation_runtime.py` demonstrates that the hidden tensor entering the
LM output head can be captured safely with a forward pre-hook and transformed
before a second output-head call. That wrapper belongs to historical G2 and its
weights/configuration must not be loaded, but the capture pattern proves the
backbone can expose the needed state without modifying LLaDA internals.

PMTR should extract this pattern into a neutral, tested `forward_with_hidden`
adapter.

## 4. What does not exist yet

The following are real implementation gaps, not completed capabilities:

1. no joint SPD-logmetric/PBC-Cartesian corruption kernel;
2. no post-quantization offline CHGNet corruption certifier;
3. no data builder that creates a complete coherent corrupted `source_answer`
   and the exact current repair-state sequence;
4. no general transaction context carrying old active geometry, Plan data,
   program rank, and transaction role;
5. no public minimum-image vector/argmin-image operator—only distance is public;
6. no PMTR lattice or coordinate repair-vector head;
7. no target-bin manifold-to-token renderer;
8. no optional transaction-logit transform hook in cell/species revision;
9. no PMTR continuous loss path in the trainer;
10. no explicit runtime test proving that inference cannot call an MLIP.

The PMTR document is a design, not evidence that any item above works.

## 5. Required corrections to the initial PMTR design

### 5.1 Cache a transaction-level continuous proposal

Current cell and XYZ tokens are sampled sequentially. Recomputing a new repair
vector after each component would change the scientific action and could create
order-dependent drift.

PMTR must compute once and cache:

- one six-dimensional corrected cell proposal before resolving `a,b,c,alpha,
  beta,gamma`;
- one three-dimensional corrected Cartesian site proposal before resolving
  `X,Y,Z`.

Each component forward receives the corresponding marginal logit transport.
The existing transaction rollback remains authoritative.

### 5.2 Preserve old geometry through an explicit side channel

The active block is replaced by mask tokens before the model forward. The repair
head nevertheless needs the old generated value it is attempting to revise.
This value is available in current runtime (`previous_token_ids`) but is stored
only after generation for logging.

Move it into an immutable `TransactionContext` before remasking. The DLM still
sees a mask; only the repair head receives old geometry. Training uses the same
contract.

### 5.3 Recompute geometry after cell commitment

Cell closure changes the mapping between fractional and Cartesian coordinates.
After a supported cell is committed, PMTR must rebuild:

- the lattice matrix and metric;
- Cartesian coordinates from the still-visible fractional values;
- triclinic minimum-image vectors and distances.

Coordinate repair cannot reuse pair geometry computed under the old cell.

### 5.4 Render around the predicted target, not the old token

Restricting residual logits to the old token's adjacent bins limits one repair
sweep to roughly one quantization step and is inconsistent with current raw
force/stress errors.

The renderer must locate the two legal bins bracketing the predicted corrected
continuous value. Existing hard support still decides whether either token can
be committed.

### 5.5 Do not put CHGNet in the loss

The final approved design uses CHGNet only to certify offline corruption input.
The clean MP20 transaction and its exact manifold retraction are the training
targets. This avoids repeating projected Force-Score and avoids a force-versus-
clean-target gradient conflict.

## 6. General architecture to implement

### 6.1 Neutral transaction interfaces

Introduce general types independent of PMTR:

```text
TransactionKind = cell | site_xyz

TransactionContext:
  kind
  active generation positions
  old active token ids and decoded geometry
  complete pre-remask token state
  prompt length / N
  Plan fields
  species program and program rank
  committed lattice version

ModelStep:
  logits
  final hidden states

TransactionLogitTransform:
  prepare(context, model_step) -> cached proposal
  apply(component, logits, cached proposal, legal support) -> logits
```

`revise_spad_cell` and `revise_spad_species_blocks` receive an optional
`TransactionLogitTransform`. With `None`, output and RNG behavior must remain
unchanged. PMTR is one implementation of this interface, not a hard-coded fork
of the sampler.

### 6.2 Neutral periodic manifold primitives

Create or extract stateless functions for:

- lattice parameters <-> canonical lattice <-> SPD metric;
- SPD matrix log/exp and relative tangent;
- wrapped torus displacement;
- triclinic minimum-image vector, selected image, and distance;
- Cartesian/fractional conversion;
- legal numeric token support and target-bin interpolation.

Both data construction and inference must call the same primitives.

### 6.3 PMTR modules

Build six reusable modules:

1. `manifold_corruption`;
2. `corruption_certification` (offline-only dependency boundary);
3. `programmed_repair_states`;
4. `manifold_repair_head`;
5. `manifold_token_transport`;
6. `pmtr_runtime` and `pmtr_trainer` adapters.

Only `corruption_certification` may import CHGNet-facing code. The inference
package and sampler must not depend on it.

## 7. Training architecture grounded in current code

### 7.1 Data row

Extend the existing paired-row contract rather than create a new format:

```text
prompt          = unchanged compact Plan prompt
answer          = clean MP20 dynamic body
source_answer   = coherent corrupted dynamic body
forced masks    = exact inference-matched remaining transaction positions
loss positions  = current active component
repair metadata = old active geometry, clean manifold target, program role
```

The existing dataset and collator already enforce prompt/N/element identity and
construct the masked source state.

### 7.2 State sampling

`closure_states` enumerates exactly `6+3N` component states for cell then reverse
Llama species blocks. Today one deterministic state is selected per source.

For PMTR, each of the two epochs creates a new coherent corruption and samples
one active component according to the empirical runtime visitation over these
same states. Across all 27,136 sources this yields 54,272 corrupted examples
without materializing every state for every crystal.

### 7.3 Model path

The current `compute_loss_components` already supports paired source tokens and
forced active loss. PMTR requires a model wrapper that returns base logits plus
captured hidden states and applies the optional logit transform.

The continuous SPD/torus loss should be implemented as an additional component
returned by a PMTR-specific trainer adapter. Do not put PMTR branching into
unrelated legacy loss profiles.

### 7.4 Optimization

- initialize from the retained pre-K10 SPAD basin-closure checkpoint;
- zero-initialize the PMTR output projections;
- use all 27,136 MP20-train sources for two fixed epochs;
- alternate clean SPAD and corrupted PMTR microbatches;
- normalize SPD and torus losses by their fixed corruption scales;
- use one seed and save only the fixed endpoint;
- validation is diagnostic and cannot select a checkpoint.

This is substantially denser supervision than the retired K10 route. It solves
the training-data insufficiency at the correct object, while not claiming that
two epochs mathematically guarantee target attainment.

## 8. Scientific consistency review

### 8.1 Composition

`N` and element slots are identical in clean and corrupted bodies and remain
hard-prefilled. PMTR has no pathway to change composition.

### 8.2 Lattice and coordinates

The corruption and repair live on

\[
\mathrm{SPD}(3)\times\mathbb T^{3N}.
\]

Lattice noise is applied in log-metric tangent space. Coordinate noise is
defined in Cartesian space, converted through the corrupted cell, and wrapped
fractionally. Cell updates are committed before coordinate geometry is
recomputed. These choices respect positive-definite cells and periodic
coordinates and explicitly couple `L` with `X`.

### 8.3 Stability

The clean target is an MP20 relaxed crystal. Offline CHGNet certification is
performed after exact token quantization and admits only corruptions for which
the clean retraction is locally downhill under that training MLIP. CHGNet does
not define the learned target and cannot be called at inference.

Thus PMTR learns a force-certified local relaxed-manifold retraction. It does
not learn or claim the exact DFT PES, global ground state, kinetic stability, or
synthesizability.

### 8.4 DLM necessity

PMTR operates only after a complete structure exists. SPAD can remask an early
cell or XYZ transaction while preserving all later sites as visible context.
An autoregressive generator would have to discard or regenerate that suffix.
The method therefore exercises a genuine non-causal DLM capability.

### 8.5 Llama necessity

The Llama-conditioned pointer program determines which species anchors are
constructed first and which complete species blocks are repaired first in the
reverse sweep. Program rank and active species condition the repair head. The
program is therefore part of the DLM transition state, not decorative text.

PMTR does not claim Llama observes live geometry or controls the DLM
interactively.

## 9. Architectural risk assessment

| Risk | Impact | Code-grounded response |
|---|---|---|
| Hidden states unavailable in current sampler | High | Extract the proven LM-head pre-hook pattern into a neutral adapter |
| Old active geometry disappears on remask | High | Construct immutable `TransactionContext` before masking |
| Cell change invalidates coordinate geometry | High | Version the lattice and recompute all transforms/MIC after commit |
| One-bin renderer cannot repair large errors | High | Interpolate around predicted corrected target bins |
| PMTR accidentally changes base RNG/output | High | Optional transform defaults to `None`; zero head must reproduce base output |
| Historical G2 leaks into new method | High | Reuse only stateless geometry utilities; no G2 model/config/checkpoint import |
| CHGNet leaks into inference | High | Separate offline certification package; runtime dependency test |
| Clean/corrupt loss imbalance | Medium | Dimensionless normalization and one gradient-scale sanity report |
| Synthetic corruption does not match raw SPAD | High | One integrated train-only transfer preflight on actual SPAD raw inputs |
| Repair copies training structures and loses novelty | Medium | One bounded local sweep, exact-composition prospective evaluation, novelty reported |

## 10. Minimal validation strategy

### 10.1 Unit and integration tests before GPU

Use deterministic CPU/Torch tests for:

- SPD log/exp round trip and positive definiteness;
- PBC vector/distance agreement under skew cells;
- global translation and same-species permutation covariance;
- target-bin interpolation, periodic aliases, and legal support;
- zero-output PMTR equality with current logits;
- only active transaction logits change;
- cached cell/XYZ proposal remains fixed across component predictions;
- cell commit triggers MIC recomputation;
- source/target preserve exact `N`, elements, and `7+4N`;
- inference runtime imports no CHGNet/MLIP package.

These tests validate mathematical and software contracts. They are not model
ablations.

### 10.2 The only small model experiment

Use the one integrated 512-row MP20-train package already defined in the PMTR
method document:

- 384 train / 128 held-out coherent corruptions;
- one small PMTR learnability run;
- one paired one-sweep application to 128 fixed, train-only current-SPAD raw
  structures;
- report reconstruction/PBC validity and paired raw energy, force RMS, stress;
- no Direct, S.U.N., tau sweep, alternative arm, seed, or hyperparameter search.

This experiment answers one question only: does the learned repair transfer
from coherent MP20 corruptions to actual SPAD errors? If yes, proceed directly
to the one formal full-corpus run. If no, diagnose the demonstrated interface or
distribution mismatch rather than opening multiple experimental branches.

### 10.3 Essential invariants only

The implementation has four non-negotiable invariants:

1. exact `N`/elements and `7+4N` remain unchanged;
2. decoded cell and PBC geometry remain finite and valid;
3. disabled or zero-initialized PMTR reproduces current SPAD behavior;
4. inference has no MLIP dependency or call.

All other quantities are reported as diagnostics or outcomes, not multiplied
into a large gate matrix.

## 11. Formal execution shape

After the integrated preflight supports transfer:

1. build/certify the full two-epoch dynamic corruption stream;
2. train one PMTR seed to one fixed final endpoint;
3. generate one fresh fixed256 prospective cohort with one Plan and trajectory;
4. evaluate raw validity and paired physical metrics first;
5. calculate raw and fixed-tau800 official S.U.N. once;
6. enter paper1000 only under the already declared final success condition.

Maximum use is four A800 and four CPU cores per GPU. Direct is not on the
critical path. No multi-tau, multi-arm, multi-seed, checkpoint sweep, or
result-triggered method iteration is part of this plan.

## 12. Paper architecture

The paper should present one scientific logit-compilation hierarchy:

```text
C3FD reachable chemistry
        -> constrains Planner-Llama chemical actions
Llama species program
        -> compiles semantic crystal dependencies into repair order
SPAD masked DLM
        -> exposes future geometry and revises earlier transactions
PMTR
        -> compiles continuous SPD/PBC repair into legal token probabilities
```

The central problem is not “how to add another force loss.” It is:

> How can continuous periodic repair knowledge be executed by a discrete
> crystal language model without breaking exact chemical language constraints?

The core contribution is:

> A Llama-programmed non-causal crystal repair process that transports a learned
> SPD/PBC manifold retraction into native legal special-token probabilities.

Prior work separately covers continuous crystal diffusion, hybrid
continuous/discrete diffusion, visible-token correction, and learned unmasking
order. PMTR does not claim those ingredients individually. The proposed novelty
is their crystal-language integration under exact C3FD/SPAD semantics.

Main-paper evidence remains compact:

1. existing C3FD/SPAD composition and execution result;
2. one PMTR paired native raw result;
3. one final raw/tau800 S.U.N. result.

Historical R03, G2, BTRD, Force-Score, D3PO, K10, and rejected PCTP results stay
outside the current-method figure and main contribution list.

## 13. Current implementation decision

The architecture is **approved for implementation**, subject to the code facts
and corrections above. Approval means the scientific and software objects are
coherent enough to build; it does not assert that PMTR already exists or that
`10%/50%` is guaranteed.

Next implementation order:

1. neutral manifold/vector/token primitives;
2. transaction context and optional logit-transform hook;
3. PMTR head and renderer;
4. paired corruption/repair-state data;
5. trainer adapter and unit tests;
6. the single integrated preflight.

No GPU experiment is authorized by this document alone; execution begins only
after the user approves the audited implementation plan.

