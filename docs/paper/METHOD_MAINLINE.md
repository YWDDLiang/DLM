# Method: science-constrained hierarchical crystal language generation

## 1. Joint model

Given a stability intent `y`, our generator factors crystal design into a
scientifically supported Plan and two coupled realization scales:

\[
z\sim p_\phi(z\mid y,\mathcal C),\qquad
x_0\sim p_\theta(x\mid z;\mathcal R),\qquad
x_*\sim K_{494}(x\mid x_0,z).
\]

`C` is C3FD's prefix-dependent scientific support, `R` is the G2 periodic
relation operator inside the masked DLM, and `K494` is a frozen terminal
diffusion kernel. The Plan `z` is the shared state connecting all transitions.

## 2. Science-Constrained LLM Planner

### 2.1 Typed scientific state

A Planner state contains target atom count, chemical family, target arity,
selected valence species/counts, remaining atoms and charge. C3FD computes the
set of actions that can still reach a benchmark-compatible terminal state.
This support is prefix dependent: an action is available only when the
remaining atom and charge budget can be completed.

### 2.2 Learned product-of-experts decoding

Llama receives typed embeddings of the same partial scientific state and
predicts residual logits for proposal and composition actions. For legal mask
`M_t`, decoding uses

\[
\pi_\phi(a_t\mid s_t)
=\operatorname{softmax}
\left(\ell_{\mathrm{C3FD}}(s_t)+\ell_{\mathrm{LLM},\phi}(s_t)
+\log M_t\right).
\]

The coefficients are fixed at one. The residual output is zero initialized,
so the normalized step-zero distribution equals C3FD. The completed action
sequence and Llama soft-field predictions are serialized as one
`C3FD_NATIVE_PLAN_V2` Plan. Sampling uses one trajectory, with no rejection,
repair, reranking or best-of-N.

## 3. Plan as an exact cross-scale interface

The Plan exposes exact `N`, ordered elements and counts, composition-derived
family, and compact lattice-system, space-group-bucket and volume-per-atom
hints. The same canonical serializer is used in MP20 training and deployment.
Thus the Planner does not merely precede the DLM; it defines the global state
conditioning every DLM prediction.

## 4. Plan-Conditioned Crystal Diffusion Language

### 4.1 Dynamic crystal language

For N atoms, a body contains

\[
[N,L_a,L_b,L_c,\alpha,\beta,\gamma]
+\bigoplus_{i=1}^{N}[E_i,X_i,Y_i,Z_i].
\]

This is exactly `7+4N` tokens. Global lattice parameters use 0.1 Å/1° tokens;
fractional coordinates use the frozen coordinate vocabulary. Exact N and
species multiplicities are visible/prefilled, while lattice and coordinates
are masked under the exact-axis schedule.

### 4.2 Masked denoising objective

The base DLM is trained on MP20 teacher Plans and crystal bodies using masked
token likelihood:

\[
\mathcal L_{\mathrm{DLM}}
=-\mathbb E_{x,z,t,M_t}
\sum_{k\in M_t}\log p_\theta(x_k\mid x_{\bar M_t},z,t).
\]

This separates long-horizon scientific intent from detailed realization while
allowing all unknown geometric tokens to be denoised in parallel.

## 5. Periodic-relational denoising

### 5.0 Global visibility is not relational priority

The masked Transformer has global self-attention, so lattice and coordinate
tokens are nominally visible to every position. That does not make G2
redundant. Visibility only makes a computation possible; it neither encodes
the correct periodic computation nor assigns it sufficient optimization
priority.

In particular, triclinic minimum-image distance is a nonlinear function of six
lattice parameters and two fractional sites. Its physical meaning also depends
on the species pair. Ordinary token CE offers no guarantee that the network
will reconstruct this coordinate-aware relation at an uncertain q0 state. A
single colliding pair can invalidate an entire crystal but contributes only one
of `N(N-1)/2` pair relations and competes with every masked token. The global
network can therefore represent the information while systematically
underweighting the event that decides validity and basin quality.

G2 is a scientific-salience path: it computes a small set of high-consequence
relations in the correct periodic frame and gives them a direct, zero-
initialized route back to logits. It does not reduce the receptive field or
replace the Transformer. It preserves global language reasoning while making
the most important crystal relations easy to compute, explicitly normalized
and difficult for average token gradients to ignore.

### 5.1 Soft geometry from q0

At a denoising state q0, legal geometry-token probabilities define expected
lattice lengths, angles and circular fractional-coordinate means. The lattice
metric couples these expectations into a single periodic frame.

### 5.2 Strict periodic relation graph

For each atom pair, the operator evaluates the bounded triclinic minimum image
over 125 neighboring cells and constructs species-pair radial features. The
promoted packing margin is

\[
m_{ij}=\operatorname{clamp}(0.55(r_i+r_j),0.60,1.40)\;\text{Å}.
\]

Metric, pair-RDF, normalized penetration and coordination losses supervise the
same relational state. The 125-image implementation agrees with pymatgen over
the complete frozen audit.

### 5.3 Residual return to crystal tokens

For site states `h_i`, pair feature `e_ij` and global metric `g`, G2 computes

\[
m_{ij}=f_m(h_i,h_j,e_{ij},g),\qquad
\bar m_i=\frac{1}{N-1}\sum_{j\ne i}m_{ij},\qquad
\Delta q_i=W_{\mathrm{out}}f_u(h_i,\bar m_i,g).
\]

`W_out` is initialized to zero, giving exact equality to the base DLM before
training. The updated logits `q1=q0+Delta q` participate in the next denoising
decision. No relation projection is applied after a structure has been sampled.

### 5.4 Stability transport through the same residual

BTRD, when promoted by its fixed stability gate, adds no new inference module.
It uses model494 tau200 train-only teacher geometries to supervise normalized
metric and minimum-image coordinate transport through the existing G2
residual. Backbone and Compact-V2 LoRA remain frozen. Thus the same
scientific-salience path first represents periodic relations and then learns
which local direction enters a better structural basin.

## 6. Terminal diffusion realization

The raw exact-composition structure enters frozen model494:

\[
x_*\sim K_{494}^{(800)}(\cdot\mid x_0,z).
\]

The refiner seed is fixed by requested sample index, and missing attempts remain
missing. The pipeline reports raw `x0` to identify what the DLM learned and
refined `x*` to measure the complete generator.

## 7. Training and inference invariants

- C3FD and Llama jointly sample one Plan; no candidate selection.
- Plan keys, types, order and rendering are equal at train and inference.
- DLM N/composition are exact and remain visible.
- G2 is initialized as a function-preserving residual.
- Each Plan produces one DLM trajectory.
- model494 uses a fixed step count and sample-index seed.
- All metrics use the requested denominator; failures are not replaced.

These invariants make the hierarchy a single probabilistic generator with
auditable information flow.
