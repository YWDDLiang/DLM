# Token-native PBC geometry executor V1

Status: design only. Implementation starts only if the rollout-matched pilot is
terminal or if that pilot improves Direct but leaves a clear geometry gap.

## Scientific role

The DLM remains the crystal executor. C3FD fixes composition and `N`; the
masked DLM produces the dynamic `7+4N` body; G2 supplies periodic relational
state. The new interface adds one identity-preserving continuous operation
inside each denoising forward pass:

```text
Plan + masked 7+4N state
        -> DLM hidden H
        -> q0 token logits
        -> committed/soft periodic geometry (L0, U0, species)
        -> existing G2 site/pair states Z
        -> SPD lattice + torus coordinate residual
        -> adjacent legal geometry-token logit residual
        -> qfinal
        -> unchanged exact-axis token commitment
```

It is not a second generator, completed-sample repair, energy filter, reranker
or replacement for model494. It does not modify N/element logits and requires
no CHGNet call at inference.

## MP20-only supervision boundary

Every positive geometry target is an original MP20-train crystal serialized in
the same dynamic `7+4N` language. A generated structure, model494 output,
CHGNet-relaxed structure or selected low-energy sample is never used as a
teacher.

Real DLM rollouts are used only to construct the input distribution. For an
MP20 row, run the frozen base policy with its deployment C3FD Plan, retain the
tokens already committed at a lattice/X/Y/Z stage (including wrong tokens),
and keep the remaining fields masked. The paired MP20 body is the sole target.
Thus the interface learns to recover from its own deployment errors without
declaring those generated errors to be correct labels. This is supervised
rollout-state correction, not self-training or preference learning.

Train/validation/holdout splits are by MP20 source composition. No generated
outcome, Direct result, CHGNet value, hull value or model494 trajectory may
influence row selection, targets or loss weights.

## Inputs and q0 geometry

The interface consumes the state already constructed by
`periodic_relation_runtime.py`:

- final DLM hidden states;
- q0 legal-family distributions for six lattice and `3N` coordinate tokens;
- committed token values for fields already emitted by exact-axis decoding;
- element identities, site mask and prompt length;
- strict triclinic lattice and periodic pair graph from G2.

For a committed field, use its exact token value. For a masked lattice field,
use a MAP-anchored local expectation rather than an unrestricted mean across
multiple modes. For a masked fractional coordinate, use a circular/torus mean;
if the resultant is low, anchor the local expectation at the MAP bin. This
avoids averaging two periodic modes into a nonexistent midpoint.

Each soft field also carries a confidence: circular resultant for coordinates
and local posterior mass for lattice values. A stage-aware gate attenuates pair
messages when one of the still-masked axes is multimodal. The executor can
therefore use predicted future axes as context without treating an uncertain
mean as a real atom position.

The canonical row-vector lattice is `L0`; its metric is

\[
G_0=L_0L_0^\top.
\]

Fractional coordinates are

\[
U_0\in[0,1)^{N\times3}.
\]

## SPD lattice residual

Pool the final rank-64 G2 site states over active sites. A small MLP emits six
parameters for a symmetric tangent matrix `S`; its final layer is exactly zero
initialized.

\[
S=\operatorname{sym}(\tanh(f_{metric}(\bar Z))).
\]

Apply an SPD congruence update:

\[
G_1=L_0\exp(\eta_S S)L_0^\top.
\]

This guarantees positive definiteness for every finite output. At step0,
`S=0`, hence `G1=G0`. Cholesky factorization recovers a canonical lattice and
therefore updated length/angle values. The lattice residual is active only
while lattice fields remain masked; later coordinate stages always use the
actually committed lattice and never change it secretly.

## PBC torus coordinate residual

Extend the shared bounded-image operator so it returns the exact minimum-image
Cartesian displacement as well as distance. For sites `i,j`:

\[
r_{ij}^{MIC}=\arg\min_{n\in[-2,2]^3}
\left\|(u_j-u_i+n)L_0\right\|.
\]

A shared pair MLP predicts a symmetric scalar from G2 site states and radial
features. Scatter anti-symmetric pair contributions along the minimum-image
unit direction:

\[
\Delta r_i={\eta_r\over\deg(i)}\sum_j
a_{ij}{r_{ij}^{MIC}\over\|r_{ij}^{MIC}\|+\epsilon}.
\]

The opposite contribution is applied to site `j`, so the global translation
component is exactly zero. Convert to fractional tangent coordinates and wrap:

\[
\Delta u_i=\Delta r_iL_0^{-1},\qquad
u_{1,i}=(u_{0,i}+\Delta u_i)\bmod1.
\]

Only currently masked coordinate axes receive a logit residual. Previously
committed X/Y/Z values remain immutable under exact-axis decoding.

Training follows the same rule. Previous committed fields and future masked
fields are stop-gradient context; geometric gradients enter only the current
active lattice or coordinate group. Consequently, the executor cannot repair a
completed CIF secretly and cannot learn from target values that were not yet
available at that denoising stage.

## Continuous-to-token renderer

The executor does not hard-round geometry. For every legal family bin `v_k`,
construct a local triangular basis around the original and updated continuous
value:

\[
\phi_k(y)=\max(0,1-d(y,v_k)/h).
\]

- fractional coordinates use torus distance and `h=0.01`;
- lengths use ordinary distance and `h=0.1 A`;
- angles use ordinary distance and `h=1 degree`.

The residual is

\[
\Delta q_k=g_f[\phi_k(y_1)-\phi_k(y_0)].
\]

Only adjacent legal lattice/coordinate bins are modified. N, elements and the
rest of the vocabulary are untouched. When the continuous head is zero,
`y1=y0` and `Delta q=0` exactly. `qfinal=qG2+Delta q` therefore preserves the
current model at initialization while retaining a gradient path through the
renderer.

## Training losses

The primary target remains dynamic token CE against the paired MP20 body.
Continuous losses act on the active executor output, not on a post-hoc
generated or reconstructed teacher CIF:

\[
L=L_{CE}+0.1L_{SPD}+0.1L_{torus}+0.1L_{pair}
  +0.2L_{collision}+0.05L_{coord}+0.01L_{step}.
\]

- `L_SPD`: target lattice log-metric/geodesic error;
- `L_torus`: `1-cos(2*pi*(U1-Utarget))`;
- `L_pair`: same/different-species smooth RDF target;
- `L_collision`: strict triclinic PBC soft barrier with a 0.55 A training
  buffer around the 0.50 A Direct threshold;
- `L_coord`: smooth coordination target;
- `L_step`: bounded metric/coordinate residual norm to protect valid states.

`L_SPD` is active only during the lattice stage. `L_torus` is active only on
the current X/Y/Z group. Pair, collision and coordination terms assemble a
full soft geometry from committed values plus stop-gradient q0 context, but
backpropagate only through the active group. This matches exact-axis execution
while retaining a physically meaningful local signal.

The executor first targets geometry validity. Lower energy is not claimed from
these losses alone. If Direct improves and raw CHGNet does not, stability must
use a separate raw-safe same-composition objective.

## Symmetry and invariance boundary

Guaranteed by construction:

- periodic translation through torus wrap/minimum image;
- Cartesian rotation equivariance of pair-vector coordinate updates;
- site permutation equivariance in the shared graph/scatter path;
- SPD lattice validity;
- exact composition preservation.

The interface does not claim arbitrary `GL(3,Z)` cell-basis equivariance; the
dynamic token language retains its canonical cell gauge. Training may apply a
shared global fractional translation and same-element site permutation to
source, target and masks, but random orbit augmentation is not required for
the first pilot.

## Complexity

Expected additional parameters are below 20k: one metric MLP, one pair-scalar
MLP, family gains and small normalizations. For batch16 and `N<=20`, strict
125-image PBC remains `O(N^2)` and reuses the existing G2 graph. Expected
incremental activation memory is 20--40 MB, peak GPU memory below +10%, and
sampling latency below +15%. No second Transformer forward is introduced.

## Repository integration points

- `periodic_geometry_ops.py`: return bounded minimum-image vectors/indices;
- new `continuous_geometry_executor.py`: SPD head, pair-vector head, torus
  update and adjacent-bin renderer;
- `periodic_relation_adapter.py`: expose final site states and pair graph;
- `periodic_relation_runtime.py`: `q0 -> G2 -> executor -> qfinal`, checkpoint
  load/save and exact step0 equality;
- `periodic_geometry_objective.py`: losses on `G1,U1`;
- `llada_sft.py`: executor-only parameter partition and diagnostics;
- `sample_sgtc_l6.py`: optional executor checkpoint, with sampler order unchanged.

## Future pilot contract

Use 512 independent MP20-train structures, not `64×8` synthetic states. Save
four current-G2 rollout input states per structure, paired with the original
MP20 target, split 384/128 by composition, freeze Planner/DLM LoRA/existing G2,
and train only the executor for 256 updates on 2 A800. Generated states are
inputs only; the teacher remains MP20 throughout.

Promotion requires:

- exact step0 logit equality and 100% finite SPD outputs;
- body loss at most 1/256 and composition unchanged;
- raw Direct `+8/256` and net invalid→valid `+8`;
- valid→invalid at most 2;
- peak memory below +10% and latency below +15%.

Only after geometry passes, run paired raw CHGNet: at least 240 known pairs,
median candidate-minus-BASE at most `-0.01 eV/atom`, and lower-energy fraction
above 0.55. A geometry-only positive is reported as such and is never promoted
to a stability claim without the energy gate.

## Related-method boundary

DiffCSP jointly diffuses lattice and coordinates with a periodic E(3)-equivariant
score model; FlowMM integrates a Riemannian flow; MatterGen jointly diffuses
atom types, wrapped coordinates and lattice. This interface instead keeps the
masked token DLM and exact symbolic composition as the generator, applying one
local continuous tangent update inside its existing denoising pass:

- DiffCSP: https://arxiv.org/abs/2309.04475
- FlowMM: https://arxiv.org/abs/2406.04713
- MatterGen: https://www.nature.com/articles/s41586-025-08628-5
