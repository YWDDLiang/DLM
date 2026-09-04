# Full-MP20 Basin Action-Value Distillation

Status: implementation authorized; formal training starts after the scientific
object preflight below confirms that the frozen labels rank the deployed
transaction outcomes they claim to rank.

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

