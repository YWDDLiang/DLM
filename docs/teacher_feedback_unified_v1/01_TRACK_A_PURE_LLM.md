# Track A: C3FD-Conditioned LLM-Only Crystal Executor

Status: **design awaiting approval**

## 1. Definition of “pure LLM”

Track A uses no masked DLM when producing the raw discrete crystal. Its
trainable generator is one Llama backbone with:

- the retained C3FD typed scientific-state interface for composition;
- a species-program ranking head;
- a body LoRA trained on canonical CrysLLMGen text;
- Semantic Logit Adapter heads for lattice and coordinate fields.

The backbone has two named stages: frozen `PlannerAdapter-P` for chemical
planning and the newly trained `BodyAdapter-A + ProgramHead-A + SLA-A` for
structure execution. The deployed scientific sampler uses SLA-A semantic
actions. Native text tokens provide the causal context and text-training loss;
canonical CrysLLMGen text is the deterministic output serialization, not a
second competing deployment sampler.

A deterministic codec and periodic commit controller are decoding
infrastructure, not learned structure generators. The common terminal
model494 run is reported separately as `A1→494`; it is not part of the
“pure-LLM raw generator” claim. In the paper the precise term is
**LLM-only executor**: C3FD still supplies chemical support, but no DLM or
continuous generator produces its raw structure.

## 2. Why the existing fused Planner is retained

The current typed C3FD–Llama Planner already satisfies the first teacher
requirement:

- C3FD owns prefix state, conservation and reachable support;
- Llama hidden states produce residual proposal/action/soft-field logits;
- one trajectory produces exact composition plus LS/SG/VPA;
- measured Llama-vs-C3FD KL confirms that Llama changes decisions.

Retraining that successful chemical component is unnecessary for the first A/B
test. It is frozen while the body controller is trained. The new work begins
where the existing Planner stops: converting its state into an AR structural
program.

## 3. Training representation

Use the full MP20 training split. Every teacher structure is converted into:

1. two same-schema conditioning views: the teacher Compact-V2 Plan and the
   frozen C3FD–Llama predicted Plan for that MP20-train composition;
2. the Llama species program;
3. canonical CrysLLMGen text:

```text
a.b b.b c.c
alpha beta gamma
Element
x.xx y.yy z.zz
...
```

Both conditioning views share the same body target and together retain total
source weight one. Teacher view teaches field semantics; predicted view matches
deployment noise. MP20 validation reports both views but does not select one.

Before reordering those records, one teacher-forced pass of the frozen starting
AR-body checkpoint measures native-text margin per species block. Those fixed
soft labels train ProgramHead-A and determine the training serialization. They
are not recomputed after BodyAdapter-A changes.

The numeric precision matches the common semantic grid. Site records are
grouped by the same species program used at inference. Global random origin
translation and within-species site permutation are data augmentations;
cross-species program order is never randomized.

## 4. Body Llama and Semantic Logit Adapter

The body Llama is trained with three losses over the same teacher state:

\[
\mathcal L_A
=\mathcal L_{\mathrm{text}}
+\lambda_{\mathrm{sem}}\mathcal L_{\mathrm{field}}
+\lambda_{\mathrm{program}}\mathcal L_{\mathrm{order}}.
\]

- `L_text`: ordinary causal loss on canonical CrysLLMGen text;
- `L_field`: field-family CE from Llama boundary hidden states to canonical
  length, angle and coordinate bins;
- `L_order`: soft species-ranking supervision.

The semantic heads are small linear/low-rank projections. They do not share
weights or IDs with the DLM tokenizer. Their purpose is to expose the Llama's
field distribution in a tokenizer-independent space.

One fresh `BodyAdapter-A` is a LoRA with rank 8, alpha 32 and dropout 0.05 on
the Llama attention and feed-forward projections. It trains together with
`ProgramHead-A` and `SLA-A` for exactly 1,696 optimizer updates: effective
source batch 16 over 27,136 MP20-train rows. Teacher/predicted Plan losses are
averaged inside each source group, so two views do not double the epoch.
The frozen optimizer recipe is LR `5e-5`, cosine decay, 100 warmup updates and
minimum LR ratio 0.2. Two A800 GPUs use per-device source batch one and gradient
accumulation eight. `PlannerAdapter-P` is not overwritten.

The three Track-A components are then frozen as one endpoint. Validation
measures text likelihood, semantic-bin accuracy/calibration, exact parse and
program consistency; it does not choose a checkpoint.

## 5. Inference algorithm

```text
plan, chemical_state = C3FD_Llama.sample_one()
program = Llama.order_head(chemical_state)
state = CanonicalCrystal(plan, program)

lattice_logits = Llama_SLA.six_field_logits(state)
lattice_beam = progressive_beam(lattice_logits, width=32)
state.commit_lattice(sample(valid_lattice_distribution(lattice_beam)))

for site in sites_in_program_order:
    xyz_logits = Llama_SLA.xyz_logits_from_one_site_boundary(state, site)
    xyz_beam = cartesian_topK_per_axis(xyz_logits)  # K frozen from train audit
    legal_beam = exact_PBC_filter(xyz_beam, state.completed_sites)
    state.commit_site(sample(legal_beam))

return render_crysllmgen_text(state)
```

Exact `N` and the species multiset come from the Plan. Species lines are
compiled in the Llama program order, so body generation cannot silently change
composition. Llama remains active in choosing the Plan, program, lattice and
every coordinate.

## 6. Scientific commit controller

### Lattice

- lengths must be positive;
- a completed angle tuple must define positive volume;
- LS and VPA provide soft scores, never false hard symmetry;
- extreme volume and condition number receive bounded penalties.

### Sites

- site identity and species are fixed by the program;
- coordinate values wrap on the unit torus;
- one Llama/SLA call produces factorized X/Y/Z semantic distributions from the
  site-boundary hidden state;
- the default is top four per axis, giving 64 triplets; one train-only coverage
  and throughput audit may freeze top eight (512 triplets) before evaluation;
- the normalized joint score is the sum of the three semantic log
  probabilities plus soft risk; all triplets are checked against completed
  sites with an exact nearest-lattice-vector routine;
- triplets below 0.5 Å are removed; species-aware near-collision risk is soft;
- if the beam has no legal triplet, the site/request fails before any axis is
  committed.

The lattice is also an atomic semantic block: one SLA call produces six field
distributions, a progressive width-32 beam removes non-positive-volume tuples,
and exactly one lattice is sampled. These are internal constrained action
sets, not multiple emitted crystals or output reranking.

If no value remains, the request fails. The method does not append a replacement
sample or repair an emitted structure.

## 7. Required comparisons

| Cell | Generator | Question |
|---|---|---|
| A0 | frozen LLM program + SLA; syntax/exact composition only | What does the LLM-only semantic executor achieve? |
| A1 | exact same weights/program + lattice/PBC commit controller | Does scientific-state commit control improve it? |
| A1→494 | frozen A1 raw body + common terminal model494 | How much does the shared continuous refiner add? |

A native-text-only decode is reported on the mechanism subset as a
representation diagnostic, not as the A0 causal baseline. A0 and A1 use the
same Plan/program ledger, weights, body seed, temperature and requested
denominator. Track-A composition validity is evaluated on every request;
`comp_valid>=95%` is the retention target, not a license to delete failures.

## 8. Primary evidence

- requested-denominator proposal composition validity and downstream body
  composition retention, reported separately;
- text parse/body/CIF success;
- raw Direct plus minimum-distance ECDF, collision count, volume and condition
  number;
- raw CHGNet as a surrogate development endpoint;
- novelty/uniqueness and composition-distribution drift;
- terminal model494 and surrogate MP-reference S.U.N. reported separately.

## 9. Expected outcome

Track A should retain the Planner's high composition validity because exact
chemistry is compiled into the body state. Its main uncertainty is raw
geometry: AR causality cannot condition the current site action on unresolved
future sites. Track B tests whether DLM bidirectional evidence improves the
same action before it is committed.
