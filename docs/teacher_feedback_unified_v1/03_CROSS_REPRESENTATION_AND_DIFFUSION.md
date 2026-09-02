# Cross-Representation and Diffusion Contract

Status: **design awaiting approval**

## 1. Non-negotiable boundary

There is no valid operation called “add the AR token logits to the DLM token
logits.” The models use different vocabularies and probability factorizations:

- the AR model assigns probabilities to native subword sequences;
- the DLM assigns one probability to each crystal special token at a fixed
  semantic position;
- model494 has no language vocabulary and operates on continuous tensors.

All communication therefore passes through typed semantic values and a
canonical crystal state.

## 2. Canonical crystal state

Each request owns one mutable state:

```text
request_id
Plan:
  N, elements, counts
  sampled LS/SG/VPA labels
  full LS/SG/VPA probabilities, confidence, model/version
Program:
  species order, ephemeral serialization-slot IDs
Lattice:
  a, b, c, alpha, beta, gamma
Sites:
  site_id, species, x, y, z, completion mask
Provenance:
  committed_by, AR confidence, DLM confidence, risk summary
```

Scientific values are stored numerically, never as tokenizer IDs. A commit is
valid only when all affected views can be regenerated from this state.

There are exactly three transaction units:

1. one typed composition action in the Planner;
2. one complete six-value lattice block;
3. one complete XYZ site triplet.

A species block is only a scheduling container holding one or more site
transactions. Individual scalar field logits are internal scores, not commits.

## 3. Codec A: canonical state and AR text

The AR representation follows the existing CrysLLMGen grammar:

- first line: three lengths at one decimal;
- second line: three integral angles;
- alternating element line and three two-decimal fractional coordinates.

`render_ar(state)` emits only canonical spellings. `parse_ar(text)` returns
the canonical state or a parse failure; it does not infer missing atoms or
repair malformed numbers.

During decoding, `render_ar_partial(state, mask)` emits the Plan, complete
lattice if available and only fully committed sites in program order. It is a
control prefix, not a parseable final crystal. Full `parse_ar` is called only
after every transaction is complete.

Native tokenizer segmentation is deliberately opaque to the scientific
controller. For example, the text value `0.37` may contain one or several
subword tokens. The SLA reads the Llama hidden state at the field boundary and
scores a separately supervised semantic coordinate action directly. It is not
asserted to equal native-string probability.

For diagnostics, a candidate-trie scorer computes the native AR likelihood of
each canonical value string, including its delimiter. It measures agreement
between two Llama output heads; it does not turn SLA into the pushforward of
the native AR distribution. Its frozen scope is 512 MP20-train rows and at most
one available position from each field family per row.

## 4. Codec B: canonical state and DLM tokens

The DLM representation is one token per semantic field:

```text
<N_004>
<LA_041><LB_041><LC_042>
<AA_090><AB_090><AG_120>
<E_Li><X_000><Y_000><Z_000> ...
```

`encode_dlm(state)` quantizes canonical numeric values to the existing bins.
`decode_dlm(tokens)` requires exact `7+4N` length and field families.
`encode_dlm_partial(state, mask)` instead emits the fixed `7+4N` canvas with
unresolved positions represented by the DLM mask ID. Partial consistency checks
compare resolved semantic fields, slot handles and the mask—not a full decode.

The lookup table is explicit:

| Semantic value | AR rendering | DLM rendering |
|---|---|---|
| length 4.1 Å | `4.1` | `<LA_041>` / axis-specific family |
| angle 90° | `90` | `<AA_090>` / axis-specific family |
| coordinate 0.37 | `0.37` | `<X_037>` / axis-specific family |
| lithium | `Li` | `<E_Li>` |

Mapping uses semantic field plus value, never string equality between tokens.
Lengths use the shared positive domain `0.1–50.0 Å`, even though the AR parser
accepts a wider range and the DLM vocabulary contains a zero-length token.
Angles share `1–179°`.

Coordinates require alias handling. DLM bins `000` and `100` represent
fractional coordinates 0 and 1, which are the same point under PBC, while AR
parsing wraps `1.00` to `0.00`. The semantic bridge:

1. maps both DLM aliases to the torus value `0.00`;
2. combines their probability mass by log-sum-exp;
3. emits canonical DLM bin `000` after commitment;
4. uses physical values in `[0,1)` for geometry.

All other round trips use half the registered quantization step as tolerance.

## 5. Site identity and order

Atom order is not assumed equal across models. The canonical state assigns
ephemeral serialization-slot IDs before geometry generation:

1. species blocks follow the Llama program;
2. slots are stable within each species;
3. DLM positions retain their original fixed indices;
4. the Llama control transcript visits sites in program order;
5. codecs carry `site_id` so a corrected DLM slot updates the right AR record.

These IDs are software handles, not physical atom identities. Training applies
within-species permutations and global translations. Teacher/fusion metrics
match same-species sites as sets using minimum-cost matching rather than
requiring arbitrary row-wise identity. Canonicalization may be applied at
input/output boundaries, but a live transaction keeps its assigned handles
stable.

## 6. Semantic Logit Adapter contract

For each field family, the SLA output dimension exactly equals the **unique
physical semantic** candidate set, not necessarily the number of DLM token
aliases. Separate heads are used for:

- lattice lengths;
- lattice angles;
- fractional coordinates;
- optional element/program decisions.

Axis identity, site species, Plan state and commitment stage enter the head
features. At one site boundary, a shared hidden state produces three
axis-specific coordinate distributions; at the lattice boundary it produces
six field distributions. Coordinate heads share weights with an axis
embedding; lattice heads share within length and angle families.

Training targets come from teacher-Plan and frozen-predicted-Plan views of the
same canonical MP20 body; the two views share source weight one. The SLA is
evaluated by:

- semantic top-1 and NLL;
- calibration error;
- agreement with native-text candidate likelihoods;
- invariance to tokenizer segmentation of the same canonical value;
- train/serve equality at field boundaries.

## 7. Scientific support in semantic space

Hard support is evaluated on candidate values:

- exact Plan inventory and field type;
- legal numeric range;
- positive-volume completed lattice;
- no exact PBC-equivalent duplicate;
- atomically proposed complete-site minimum distance at least 0.5 Å.

For a triclinic cell,

\[
d_{ij}^{\mathrm{PBC}}
=\min_{n\in\mathbb Z^3}
\left\|(f_i-f_j+n)L\right\|_2.
\]

Simple fractional `delta-round` and an uncertified finite image radius are not
guaranteed to solve this problem for a highly skewed cell. Every site-triplet
candidate uses a validated nearest-lattice-vector/MIC routine. The GPU path
batches candidate/site displacements and enumerates a per-lattice image radius
certified from a singular-value lower bound; a reduced basis may tighten that
bound. Near-singular or excessive-radius cells use the exact CPU backend or are
already outside the lattice hard support. No approximate screen decides
whether exact evaluation is needed.

Soft risk includes:

- species-aware distance margin above 0.5 Å;
- low volume per atom;
- extreme metric condition number;
- disagreement with soft LS/VPA Plan distributions;
- high AR/DLM uncertainty.

Hard support never uses unknown future coordinates. One forward produces X/Y/Z
factor distributions. Default top four per axis yields 64 triplets; a single
MP20-train coverage/throughput audit may freeze top eight before evaluation.
The joint log score is the sum of axis scores. The site is committed only after
all three coordinates pass vectorized exact PBC checks. This avoids both an
early X/Y trap and per-branch DLM/LLM forwards.

## 8. Codec C: canonical state and model494

`to_diffusion(state)` produces:

- integer atom types in canonical site-ID order;
- fractional-coordinate tensor;
- lattice matrix;
- atom counts and batch indices.

`from_diffusion(output)` preserves atom/site identity and wraps coordinates
onto the torus. It never converts a continuous result directly into text
without passing through the same canonical quantizer.

The model494 network outputs differently parameterized coordinate and lattice
quantities. The bridge therefore uses the actual zero-noise deterministic
`tau800→799` transition:

\[
v_{494}(\mathcal C)
=\operatorname{Log}_{\mathcal C}
T^{\mathrm{det}}_{494,800\rightarrow799}(\mathcal C).
\]

- coordinate displacement uses strict PBC minimum images and removes global
  translation;
- lattice displacement uses a log-metric tangent;
- the result is called the deployed-refiner transition response, not a physical
  force or a guaranteed in-distribution score.

## 9. Force-calibrated complete-state corrector

Only a complete, parsed and graphable B2 predictor state may be queried.
Calibration uses 1,024 frozen-B2 MP20-train generated states and 256 disjoint
MP20-validation development states with the same predicted-Plan and sampling
contract used at serve time; clean teacher structures are not substituted for
them. CHGNet
force/stress and finite-difference energy provide labels for one confidence
module with site and lattice output heads. The module predicts whether the
observed model494 response is energy-descending and does not increase geometry
risk. CHGNet is absent from production decoding and cannot serve as independent
promotion evidence.

For adjacent semantic candidate `a`:

\[
S(a,s)=c_\phi^{f(a)}(s)
\langle\Delta y_a,v_{494}(y_s)\rangle_M,
\]

\[
R(a,s)=\max\{0,
U_{\mathrm{feas}}(y_s^a)-U_{\mathrm{feas}}(y_s)\}.
\]

The DLM corrector distribution is:

\[
q(a\mid s)\propto p_{\mathrm{AR+DLM}}(a\mid s)
\exp\{\eta_sS(a,s)-\lambda_sR(a,s)\}
\mathbf 1[a\in\mathcal A_{\mathrm{hard}}(s)].
\]

Risk and KL multipliers use an established primal-dual constrained-decoding
solver; that optimizer is not claimed as novel. The current semantic value is
always a no-op candidate. If drift, graph or confidence is unavailable, the
corrector abstains and returns B2 unchanged.

## 10. Atomic commit transaction

One successful lattice-block or site-triplet commit performs:

1. choose one complete semantic action: six lattice values or one XYZ triplet;
2. update all corresponding canonical numeric fields together;
3. write the corresponding six or three DLM special tokens together;
4. append the complete canonical lattice/site text block to the AR control
   transcript;
5. invalidate future AR distributions;
6. update only affected periodic graph/cache entries;
7. verify that the AR partial rendering, DLM partial canvas and canonical
   resolved fields agree.

Any failure rolls back the in-memory transaction before the block is considered
committed. It does not generate a substitute scientific sample.

## 11. Runtime placement

Llama and DLM inference weights are BF16 and explicitly placed on one assigned
A800 with `device_map=None`, no CPU offload and initial batch one. A memory
canary records peak allocation and requires 8 GiB headroom before batch size is
changed.

The B3 complete-state corrector does not require triple model residency. It is
executed as two deterministic stages on the same frozen states:

1. a model494 worker writes the one-step continuous response in canonical
   site-ID order;
2. a Llama+DLM worker reads that response and performs the single re-mask
   correction.

This preserves the algorithm while keeping at most two 8B language models on
one GPU.

## 12. Required interface tests

- all MP20-train/validation teacher states round-trip through AR and DLM codecs;
- AR and DLM values agree despite different token lengths/IDs;
- exact composition and site IDs survive every partial commit;
- tokenizer segmentation changes do not change semantic support;
- coordinate wrapping, global translation and same-species output
  canonicalization preserve the physical structure;
- skewed-cell MIC agrees with an exact backend near the 0.5 Å boundary;
- deterministic model494 transition preserves site identity;
- no CHGNet, hull or test outcome is read by the production decoder.
