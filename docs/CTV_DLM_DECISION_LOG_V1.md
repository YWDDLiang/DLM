# CTV-DLM Structured Design Review — Decision Log V1

Date: 2026-08-28

Final disposition: **APPROVED for prerequisites and one engineering resource
canary; formal science remains NO-GO.**

## Understanding lock

- C³FD-v2.5 is frozen as the composition-correctness contribution.
- The new question is fixed-composition geometry stability during masked DLM
  generation.
- Existing MatterSim validation continues and must gate any thermodynamic
  training.
- No stability prompt, terminal reranking, replacement, RL or rich-soft-field
  dependence is allowed in the primary design.

## Decisions and alternatives

| Decision | Rejected alternative | Rationale |
|---|---|---|
| Real forced-action continuations supervise terminal cost | corrupt completed structures and copy their terminal labels backward | corrupted-terminal classification is not a policy cost-to-go estimand |
| Two fixed milestones, one base-selected position each | guide every step, change position order and remask | keeps treatment identifiable and Branch support finite |
| Full-support normalized exponential token guidance | heuristic Doob delta, uncertainty division or top-K reranking | matches discrete classifier-style guidance and has an explicit partition |
| Absolute eV/atom cost plus state-centered advantage | within-Plan rank alone | preserves threshold calibration while cancelling composition offsets for action choice |
| Feasibility diagnostic only | learned feasibility hard mask | keeps legal support identical in gamma0 and guided arms |
| Separate Q heads; generator logits bit-identical | shared generator/Q LoRA | prevents critic training from silently changing the base policy |
| Plan/reduced-composition statistical unit | 1752 structures or repeated corruptions as independent rows | avoids pseudo-replication |
| Counter-based RNG | shared sequential seed | path changes otherwise consume different random streams |
| Minimal certified specification | unvalidated rich lattice/SG/volume fields | removes a measured non-predictive conditioning confound |
| Downstream holdout terminology | globally fresh L7 claim | C³FD compositions were already evaluated upstream |
| Defer distillation and RL | immediate guided-winner CE or GRPO | prevents proxy self-confirmation and invalid old TraceRL reuse |

## Reviewer objections and resolutions

### Skeptic / Challenger

Initial disposition: **REJECT**.

Accepted and resolved:

- terminal corruption was not value-to-go -> replaced by real branch returns;
- training/guided state mismatch -> interventions and training states now share
  two fixed frozen-base milestones;
- MatterSim single point was not the same quantity -> retained only as Gate A
  and added frozen MatterSim branch-relaxation validation;
- heuristic “Doob” rule was unjustified -> replaced by normalized discrete
  energy guidance and removed the claim;
- rank-only labels lost absolute information -> added absolute cost;
- critic/search/remask effects were confounded -> primary changes token
  probabilities only with compute-matched gamma0;
- old raw1000 was adaptively reused -> C³FD seed18 is honestly labelled a
  downstream holdout.

Accepted as boundaries rather than eliminated:

- success-conditioned Q does not estimate failure-inclusive cost;
- MLIPs can share data/theory bias;
- text geometry is not fully E(3)/cell invariant;
- fixed C³FD witness may be conservative for mixed-valence chemistry.

### Constraint Guardian

Disposition: **conditional GO for resource canary**.

Required and incorporated:

- freeze minimal-spec base before Branch data;
- Gate A before thermodynamic training;
- one <=256-completion canary and <75% host-RAM projection;
- reduced-composition identity isolation;
- numeric Q/support/gamma/gate definitions;
- bit-identical base logits with Q heads loaded;
- idempotent branch ids and complete denominator accounting;
- Plan-clustered inference and honest 10/50 benchmark-attainment language.

### User Advocate

Disposition: **conceptually aligned**.

Required terminology:

- expand CTV-DLM as Counterfactual Terminal-cost Value DLM;
- call the learned quantity downstream proxy-energy cost, not true hull
  stability;
- say no **terminal** reranking, not no preference reweighting;
- distinguish the frozen base DLM from the stability-aware CTV-DLM system;
- report guided coverage/fallback and structural, not formula-only, diversity;
- keep the main-text story to counterfactual action value -> normalized token
  guidance -> compute-matched S.U.N. evaluation.

### Integrator / Arbiter

First disposition: **REVISE** because intervention scope, cost, support, gamma,
minimal prompt, symmetry transforms and MatterSim failure rules were not fully
numeric.

After revision: **APPROVED** for prerequisite implementation and one resource
canary. The canary must assert eight distinct actions per state and 256 unique
branch ids. Its first engineering run exposed literal quantile collisions
before any terminal outcome existed. The frozen correction reserves argmax
and projects the seven unchanged CDF targets one-to-one onto nearest unused
legal-token CDF midpoints. This changes neither gamma nor scientific outcomes;
fewer than eight legal tokens still fails closed. Formal Branch generation,
Q training, L6 and L7 remain stage-gated after the canary ledger.

No reviewer objection was silently rejected. The only rejected “fatal” claim
was that compute confounding is unavoidable: with identical heads, masks,
positions, forwards and counter RNG, gamma0 versus guided isolates the
predeclared two-intervention policy.
