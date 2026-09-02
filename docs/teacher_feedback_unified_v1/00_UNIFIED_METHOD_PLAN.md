# Unified Method Plan: Scientific-State Commit Decoding

Status: **proposal awaiting user approval**

## 1. Understanding lock

The teacher feedback is interpreted as one connected problem, not three
independent modules:

1. The current C3FD–Llama Planner is a form of decoding-time knowledge
   injection. C3FD maintains a chemical state, reachable action set and learned
   action scores; Llama supplies the learned residual that chooses among
   scientifically reachable actions.
2. Crystal DLM decoding should not commit all fields in an arbitrary fixed
   order. Llama should provide a composition-dependent execution program and
   local value policy, while the DLM contributes bidirectional evidence to the
   same pre-commit semantic distribution.
3. Periodic geometry and continuous diffusion should act on the same crystal
   state. Periodic geometry constrains a lattice/site transaction before
   commitment; the optional continuous response acts only after a complete
   predictor and reopens one block before final output acceptance.

The two required methods remain distinct:

- **Track A:** LLM-only crystal execution (the requested “pure LLM” route); no
  masked DLM participates in the discrete generator.
- **Track B:** LLM-guided masked DLM; the Llama controls order and a semantic
  value policy, while the DLM maps its native special-token evidence into the
  same pre-commit semantic distribution.

## 2. Central scientific question

> How can an explicit scientific state be exposed to a language model across
> chemical composition, discrete crystal syntax and periodic geometry, so the
> model can decide both which variables to commit and which values remain
> geometrically feasible?

The answer is **Scientific-State Commit Decoding (SSCD)**:

```text
C3FD chemical state and reachable support
             ↓
Llama scientific action residuals
             ↓
exact composition + coarse Plan + species program
             ↓
canonical semantic crystal state
        ↙                         ↘
Track A: AR Llama             Track B: masked DLM
semantic block commits        guided by Llama program/prior
        ↘                         ↙
periodic-feasibility commit controller
             ↓
complete raw crystal
             ↓
optional one-step response corrector + common terminal continuous refiner
```

The canonical runtime maintains the state; Llama reads that state and chooses
actions. The tight connection is the state and commit protocol. C3FD, Llama,
DLM and the frozen continuous refiner (checkpoint 494) never exchange raw
token IDs.

## 3. Three representations, one scientific state

The current implementation uses three genuinely different spaces:

| Component | Native representation | What it can score |
|---|---|---|
| AR Llama | native BPE/SentencePiece text tokens; CrysLLMGen numeric text | causal textual continuations |
| masked DLM | dedicated one-token-per-field `7+4N` vocabulary | all unresolved special-token positions |
| model494 | atom types, fractional coordinates and lattice matrices | continuous denoising transition |

They meet in a canonical crystal state:

\[
\mathcal C=(N,\mathbf z,\mathbf n,\pi,
a,b,c,\alpha,\beta,\gamma,
\{(i,Z_i,x_i,y_i,z_i)\}_{i=1}^{N}).
\]

`pi` is the Llama species-block program and `i` is a request-local,
transaction-stable serialization-slot handle rather than a physical atom
identity. The AR text codec, DLM-token codec and model494-array codec are
separate deterministic functions around this state.

The existing representations align in precision but not in raw support:

- AR length text permits a wider range, while DLM length tokens use 0.1 Å bins
  through 50.0 Å; the shared generation domain is the positive intersection
  `0.1–50.0 Å`;
- AR angles are integral degrees and DLM angle tokens use one-degree bins;
- AR coordinates use two decimals and DLM coordinate tokens use 0.01 bins, but
  DLM bins `000` and `100` are the same point on the periodic torus.

The bridge therefore canonicalizes values, not raw bins: probability assigned
to coordinate aliases `000/100` is combined by log-sum-exp and a committed
zero coordinate is encoded canonically as `000`. This permits exact physical
value-level mapping without pretending that tokenizer IDs, supports or
sequence likelihoods are identical.

## 4. Shared C3FD–Llama controller

### 4.1 Chemical actions

At chemical state `s_t`, C3FD defines reachable support and calibrated base
scores. Llama receives typed state embeddings and emits residual action logits:

\[
\ell_t^{\mathrm{chem}}(a)
=\ell_t^{\mathrm{C3FD}}(a)
+\Delta\ell_t^{\mathrm{Llama}}(a)
+\log\mathbf 1[a\in\mathcal A_{\mathrm{reachable}}(s_t)].
\]

This deliberately changes the sampling distribution and is described as
scientific-state logit fusion, not exact speculative decoding. C3FD supplies
scientific reachability; Llama remains responsible for ranking the legal
choices. Existing evidence—C3FD-v2.5 `2000/2000` composition-valid and fused
Planner `1200/1200` at scale—supports this component.

### 4.2 Coarse Plan

The controller emits:

- exact `N/elements/counts`;
- lattice-system, space-group bucket and volume-per-atom bin;
- a permutation of unique species blocks.

The three structural fields are soft conditions. No target prototype, Wyckoff
position or fine coordinate is invented by C3FD.

### 4.3 Species program

A small ranking head on the terminal C3FD–Llama chemical-state representation
assigns one priority per species. It is trained on MP20-train only. One
teacher-forced forward of the frozen starting AR-body checkpoint supplies mean
native-text margin for every species block; high-confidence anchor blocks
precede lower-confidence blocks. Labels are frozen before BodyAdapter-A
training. No DLM counterfactual, dynamic relabeling or energy label is used.
Same-composition polymorphs produce soft pairwise preferences rather than a
false unique order.

The program changes commitment/serialization order but not the final
`7+4N` schema. Track A and Track B consume exactly the same program.

### 4.4 Exact checkpoint ownership

The shared controller is a staged Llama system with explicit named weights:

- `PlannerAdapter-P` plus the current typed C3FD residual heads generate
  chemistry and coarse Plan; they remain frozen;
- `BodyAdapter-A`, `ProgramHead-A` and `SLA-A` are trained together on the
  body task, then frozen as one Track-A endpoint;
- every Track-B cell loads that exact frozen Track-A endpoint;
- core Track B through B2 may train only DLM weights and a separate agreement
  gate. Candidate E1 later owns a separate frozen `Confidence-E1` module.
  Neither updates `BodyAdapter-A`, `ProgramHead-A` or `SLA-A`.

“Same Llama controller” therefore means the same base backbone and these
frozen stage-specific adapters, not an unspecified monolithic checkpoint.

## 5. Cross-token Semantic Logit Adapter

Raw AR vocabulary logits cannot be added to DLM logits. A numeric value such as
`0.37` may be several AR tokens but exactly one DLM token. SSCD therefore
introduces a **Semantic Logit Adapter (SLA)** attached to the body Llama hidden
state at each field boundary.

For field family `f`:

\[
\ell_{\mathrm{AR}}^{\mathrm{sem},f}
=W_f h_{\mathrm{Llama}}+b_f,
\]

where the output domain is the canonical semantic set:

- element symbols for an element field;
- length bins for `LA/LB/LC`;
- angle bins for `AA/AB/AG`;
- coordinate bins for `X/Y/Z`.

The SLA is trained jointly with the native AR text loss on the same MP20 target
state. It is a separately supervised **typed semantic action policy**, not the
mathematical pushforward of native BPE sequence probabilities and not a
tokenizer translation table. Llama hidden state determines the distribution. A
deterministic lookup maps each semantic value to its DLM special token and to
its canonical AR text rendering.

Track A samples the SLA value and appends its canonical text rendering to the
AR context. Track B fuses the same SLA distribution with DLM special-token
logits in semantic space. This makes Llama's role identical across both
methods while respecting the tokenizer mismatch.

An exact cross-tokenizer string-likelihood scorer over a canonical candidate
trie is retained as a diagnostic of agreement between two Llama output heads.
Agreement does not make the distributions identical; disagreement is reported.

## 6. Shared scientific commit controller

Before either executor samples a semantic value:

1. schema and exact-composition support remove impossible actions;
2. a complete lattice must have positive volume and valid angles;
3. `X/Y/Z` form one atomic site-commit transaction; a fixed small semantic
   beam is constructed from one-forward factorized axis logits, and exact triclinic
   minimum-image calculation removes complete triplets with a PBC distance
   below 0.5 Å before any axis is committed;
4. species-aware near-collision, volume-per-atom and lattice-conditioning risks
   add bounded soft penalties;
5. the distribution remains within a per-action KL trust region of at most
   0.05 nats from the executor's support-normalized distribution.

Unknown future coordinates never cause hard rejection. The same controller is
used by A and B, so structural validity is part of the language-generation
process rather than post-hoc filtering.

## 7. Two core contributions and one candidate extension

### C1 — Scientific-state Llama decoding

C3FD exposes reachable chemical actions and a learned scientific state; Llama
supplies context-dependent residual preferences. This explains high
composition validity as knowledge injected into an active LLM policy, not as
an external composition enumerator.

Two metrics remain separate: **proposal composition validity** measures the
learned C3FD–Llama chemical policy; **body composition retention** measures
whether the executor preserves the already selected Plan inventory.
Deterministic element-slot compilation may make the latter high, but it is not
counted as a new learned chemistry result.

### C2 — Llama-programmed cross-token DLM commitment

The Llama chooses species-block order and emits field-level semantic priors.
The DLM retains its special vocabulary and bidirectional masked context, maps
its logits into the same semantic action space, and participates in one joint
pre-commit distribution. A fusion change means its argmax differs from the
Llama-only argmax; this is not described as editing an already committed
sample. Jointly committed states are rendered back to the AR context before the
next block.

The contribution is the live shared-state commit protocol—not generic additive
logits, learned unmasking alone or the claim that two tokenizers are identical.

### Candidate E1 — PBC-feasible continuous-response corrector

After Track B produces a complete graphable predictor crystal, the frozen
continuous refiner's
actual deployed first transition is treated as an empirical refiner response,
not automatically as an in-distribution score. Confidence is trained only on
frozen B2-generated MP20-train states produced by the same runtime. A
train-only force/stress module determines whether that response is locally
energy-descending and feasibility-safe. The block with the highest frozen
standardized sum of analytic risk and predictor uncertainty is re-masked; block
selection never reads the continuous response. The
response is projected onto adjacent legal semantic values and the DLM performs
a full-context correction.

E1 is promoted as a third contribution only if it improves held-out raw
geometry/stability over B without relying on terminal full refinement to erase
the difference. Otherwise it
remains a mechanism study; the A/B paper still stands on C1+C2.

## 8. Existing evidence that motivates the redesign

- Composition is largely solved: C3FD-v2.5 reached `2000/2000`, and the fused
  Planner produced `1200/1200` composition-valid scale Plans.
- Compact-V2 plus exact `7+4N` restored high body execution, but raw Direct
  remained around `118/256` in the prospective base.
- Site/commit order is consequential: canonical ordering previously moved raw
  Direct from `128` to `143` in a matched development audit.
- G2-PBC-R showed that periodic relations can improve raw Direct
  (`118→128/256` in its full-epoch comparison), while its interaction with a
  different canonical DLM could reverse part of that gain. Geometry must
  therefore enter through one shared state and matched schedule.
- Terminal model494 strongly repairs geometry and stability, but endpoint
  distillation and a projected-force microstudent failed. The next bridge uses
  local continuous drift with abstention, not endpoint imitation.

## 9. Minimal evidence matrix

Planner:

- P0: Llama typed actions with syntax support;
- P1: P0 + C3FD reachable support;
- P2: P1 + C3FD learned action scores.

Executors on one frozen Plan/program ledger:

| ID | Method | Added mechanism |
|---|---|---|
| A0 | frozen LLM-only executor | Program + SLA; syntax/exact composition only |
| A1 | same A0 weights/program | + lattice and joint-site PBC commit controller |
| BC | frozen Compact-V2 DLM | common block schedule, canonical species order, no Llama prior |
| BO | same BC weights/schedule | + Llama species-block order only |
| BG | same BO weights/order | + frozen Llama SLA and learned agreement gate |
| BP | same BG weights | + shared lattice/joint-site PBC controller |
| B2 | BP runtime | + one program/schedule-matched MP20-train DLM epoch |
| B2C0 | complete B2 predictor | compute-matched zero-response corrector control |
| B3 | complete B2 predictor | + one empirical model494-response DLM corrector block |
| A1→494 | A1 raw output | common terminal model494 baseline |
| B2→494 / B2C0→494 / B3→494 | B raw output | same terminal model494 protocol |

The first result-bearing pass uses one frozen model seed and two common
sampling streams on fixed256. A0/A1 and BC/BO/BG/BP are adjacent decoder
comparisons; B2 isolates schedule-matched DLM adaptation. They are not expanded
into temperature or checkpoint sweeps. If exact-composition body execution
fails, the interface is diagnosed before expensive CHGNet/model494 work.
Otherwise all registered cells receive concurrent raw evaluation.

## 10. Paper story

The paper is not “Planner + DLM + refiner.” It is one hierarchical decoder in
which the scientific state changes resolution:

1. C3FD–Llama decides a reachable composition;
2. Llama converts that state into a structure-generation program and semantic
   value priors;
3. either Llama alone or a masked DLM commits crystal fields through the same
   periodic controller;
4. the DLM's masked infilling ability permits an optional complete-state block
   reopening driven by a continuous-refiner response before final acceptance.

The LLM-only route establishes that scientific-state decoding improves an AR
crystal generator. The LLM+DLM route tests whether bidirectional pre-commit evidence adds
value beyond that same Llama controller. The common canonical state makes the
comparison interpretable despite different tokenizers.

## 11. Approval boundary

This document authorizes design, local documentation and read-only validation
only. Training, sampling, CHGNet, model494, external queries and remote job
submission begin only after explicit user approval.
