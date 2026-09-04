# Full-MP20 Basin Action-Value Distillation

Status: implementation authorized; formal training starts after the scientific
object preflight below confirms that the frozen labels rank the deployed
transaction outcomes they claim to rank.

Execution: shared full-source program export, transaction ledger and frozen
reference on-policy body job `39658` is running on `4 A800 + 16 CPU`. It reads
no energy or outcome and is shared by both routes. Formal A/B training has not
started.

## Question

For the same Llama-programmed DLM transaction and the same finite legal action
set, does a value defined after entering the local relaxed basin teach stability
better than a value defined at the instantaneous unrelaxed structure?

This is a controlled comparison of value definitions, not a comparison of two
unrelated candidate generators.

## Deployed object

The frozen native trajectory is

\[
s_0=\text{SPAD body},\qquad
a_L\sim q_{\rm ref}(a\mid s_0),
\]

\[
s_1=T(s_0,a_L),\qquad
a_2\sim q_{\rm ref}(a\mid s_1),
\]

\[
s_2=T(s_1,a_2),\qquad
a_1\sim q_{\rm ref}(a\mid s_2),
\qquad x_T=T(s_2,a_1).
\]

The actions are exactly the deployed complete transactions:

- `cell`: all six lattice tokens;
- `anchor_second`: one complete XYZ triplet;
- `anchor_first`: one complete XYZ triplet.

For a candidate action at stage `t`, all later transactions are executed by the
frozen reference DLM with candidate-shared random numbers. The shared terminal
structure is

\[
x_T(s_t,a;\xi)=T_{q_{\rm ref}}^{t+1:T}(T(s_t,a);\xi).
\]

This continuation is essential: evaluating a lattice candidate before the two
anchor transactions would optimize a different object from deployment.

## Full-MP20 source population

Every standard MP20 training row (`27,136/27,136`) receives one immutable source
record and one reference on-policy body attempt. Failures remain in the source
ledger and are never silently removed.

One stage is assigned by source index modulo three:

```text
0 -> cell
1 -> anchor_second
2 -> anchor_first
```

This is **full-source coverage**, not three-times-larger full-transaction
coverage. It matches the deployed one-cell/two-anchor transaction frequency in
expectation while giving every source weight one.

The trained Planner-Llama pointer is used for the 24,558 rows with supported
typed inputs. The existing canonical fallback is retained for the other 2,578
rows; no missing scientific certificate is invented.

`species_program` is an executable control variable, not descriptive metadata.
It controls the initial SPAD species transaction order, identifies
`anchor_second` and `anchor_first`, fixes their revision order, and determines
the remaining reference-DLM continuation after every candidate. Consequently
the learned value is attached to the transaction selected by Llama's program.
Replacing this order with a global canonical schedule would define a different
method and is not allowed in either route.

## One shared action support

Both routes consume byte-identical states, action token blocks, reference
continuations and terminal structures. Each finite candidate set contains up to
four first-distinct legal actions:

1. no-op;
2. one frozen reference-DLM transaction;
3. one positive force/stress transaction;
4. the opposite force/stress transaction.

The physics proposals are generated without reading their energies. For a site,

\[
\Delta r_i=\pm\delta\frac{F_i}{\lVert F_i\rVert+\epsilon},
\qquad
\Delta f_i=\Delta r_iL^{-1}.
\]

For a cell, using CHGNet's convention
\(\sigma=V^{-1}\partial E/\partial\varepsilon\),

\[
D_\sigma=-\frac{\operatorname{sym}(\sigma)}
{\lVert\operatorname{sym}(\sigma)\rVert+\epsilon},
\qquad
L'=L\exp(\pm\delta_\varepsilon D_\sigma).
\]

Every proposal is quantized back through the existing 7+4N vocabulary, then
checked for exact species/order, positive cell volume, valid angles and strict
triclinic MIC support. A quantized duplicate is merged; an invalid proposal is
recorded rather than replaced by an energy-aware search.

## Two routes

The finite candidate distribution is

\[
q_C(a\mid s)=
\frac{\exp\ell_\theta(a\mid s)}
{\sum_{a'\in C(s)}\exp\ell_\theta(a'\mid s)}.
\]

The KL trust region is therefore explicitly a candidate-set KL, not a claim
about projection over the full vocabulary action space.

### Route A: terminal single-point control

\[
V_A(s,a)=-E_{\rm CHGNet}(x_T(s,a;\xi)).
\]

This is the full-source scale control. The 512-source pilot already shows that
it is not sufficient evidence of relaxed stability, so it is not the proposed
main contribution.

### Route B: basin-consistent action value

\[
V_B^{(K)}(s,a)=
-E_{\rm CHGNet}(R_K(x_T(s,a;\xi))).
\]

`R_K` uses the same CHGNet model, cell degrees of freedom and force tolerance as
the final relaxation, but a frozen short step count. Residual force/stress may
be reported or used only as a deterministic tie-break within numerical energy
tolerance. It never replaces basin endpoint energy.

Each route distils its value through the same candidate-set target:

\[
q_j^*(a\mid s)\propto q_{\rm ref}(a\mid s)
\exp(\beta V_j(s,a)),\qquad j\in\{A,B\}.
\]

Invalid actions have zero support before either value is considered.

## Scientific-object preflight

Before generating all labels:

1. verify the chain state for each stage against the actual inference call
   sequence;
2. verify byte-identical A/B states, candidates, continuations and terminal
   structures;
3. verify post-quantization `+force` versus `-force` and `-stress` versus its
   opposite on a frozen train-only set; do not assume a sign from notation;
4. on a frozen train-only calibration set, compare single-point and one fixed
   short-relaxation ranking against the normal full relaxation ranking;
5. approve Route B only if the short value has better paired ranking agreement
   than the single-point value and retains meaningful within-group variation;
6. report all full-source attempts and the effective gradient-producing subset.

The preflight chooses no result-facing seed, Plan, checkpoint or cohort. It only
tests whether the proposed teacher measures the claimed object.

### Preflight addendum: quantization, relaxation horizon and optimizer signal

The deployed coordinate vocabulary is `000..100`, i.e. a `0.01` fractional
grid, rather than a 1,000-bin coordinate grid. A Cartesian force displacement
can nevertheless quantize to a no-op in a large or skewed cell. Physics actions
therefore use an outcome-blind, fixed ascending geometric scan bounded by
`0.0025..0.40` Angstrom for a site and `0.00025..0.05` strain for a cell, and
retain the first quantized non-noop action in each direction. The upper bounds
keep the intervention local; they are not expanded to make the audit pass. This
is geometry-aware support construction, not energy-based candidate search. On
a frozen train-only preflight set, report separately by stage:

- the fraction of finite nonzero force/stress directions that change at least
  one transaction token after quantization (target at least 85%);
- the fraction for which both signed directions are distinct from no-op and
  from one another;
- the selected-step histogram and hard-cap hit rate, which must not exceed 5%
  in any stage (`0.40` Angstrom site cap; `0.05` cell-strain cap);
- legality after exact-species, positive-volume, angle and strict triclinic-MIC
  checks.

Every deployment stage (`cell`, `anchor_second`, `anchor_first`) must pass the
85% token-change target; a pooled average cannot hide a failed lattice or site
channel. If any stage is below 85%, no labels or formal training may start. The
only permitted correction is to derive a deterministic minimum step from the
state lattice, direction and tokenizer bin width, cap it by a predeclared local
displacement bound, freeze that rule on MP20-train, and repeat the preflight.
Energy, relaxation outcomes, prospective data and stochastic retry are
forbidden in this correction.

The short-relaxation horizon is not assumed to be 64 steps. On 100 frozen
MP20-train terminal groups, evaluate checkpoints `K in {3,5,10,20}` against a
fixed 50-step local-relaxation reference using the same initial candidate and
optimizer trajectory. Report pooled and per-stage Kendall tau-b, pairwise
ordering agreement, tie rate and within-group energy spread. Select the
smallest K whose rank agreement is within 0.02 of the best tested K and whose
non-tied pair coverage is at least 80% of the best tested K. Route B is approved
only if the selected short value also improves paired ranking agreement over
E0. The 50-step endpoint is an operational local-basin reference, not a claim
of DFT or fully converged truth.

The trainer does not add a same-minibatch `0.5 * CE + 0.5 * posterior` scalar
loss. It alternates one full-MP20 clean-CE update with one on-policy
transaction-posterior update; clean CE is never computed on generated states.
Both complete-action objectives are normalized by active transaction length,
so six-token cells do not receive twice the weight of three-token XYZ actions.
Before the formal jobs, five frozen paired batches at the common initialization
must report unclipped clean/posterior gradient norms, their ratio and cosine,
post-clip norms, informative-group count and maximum candidate-set teacher KL.
Keep the unit posterior multiplier and the `0.05`-nat target-KL budget when the
median gradient ratio lies in `[0.2, 5]` and all quantities are finite. Do not
perform batch-adaptive inverse-gradient weighting or change a coefficient in
response to downstream loss/S.U.N. If the probe fails, diagnose candidate
degeneracy and normalization first; any single replacement coefficient or KL
budget must be declared once, shared by routes A/B, and re-probed before either
formal route starts. A median clean/posterior cosine at or below `-0.5` is a
scientific objective-conflict failure, not an invitation to add PCGrad or tune
weights until the conflict disappears.

## Training

Both routes start from the same BS checkpoint and share seed, source order,
candidate order, update count, clean-CE schedule, optimizer, learning rate and
candidate-set KL budget.

- Route A: `2 A800 + 8 CPU`;
- Route B: `2 A800 + 8 CPU`;
- total: `4 A800 + 16 CPU`, at most two jobs.

No early stopping, checkpoint selection or route-specific Plan sampling is
allowed. Inference uses no CHGNet, force, stress, relaxation, candidate ranking,
reranking or replacement: one Plan produces one DLM trajectory.

## Evaluation and claim boundary

The existing frozen prospective Plan/program/cohort is reused. Report native
raw validity, relaxed energy, Strict/Meta S.U.N. and the fixed tau800 fallback
separately. Route A versus B is attributable only to instantaneous versus basin
value.

Because CHGNet is both teacher and the generated-structure relaxation proxy,
the immediate claim is **CHGNet-basin-aligned transaction distillation**. The MP
cache supplies official reference phases but does not make generated energies
independent DFT. MatterSim failure does not block this experiment; an eventual
frozen DFT or independent-MLIP subset is external validation, not part of
inference.
