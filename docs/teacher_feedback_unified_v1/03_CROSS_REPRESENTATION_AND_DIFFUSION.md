# Cross-Representation Contract

Status: **approved implementation contract**

## 1. Spaces remain separate

| Module | Native space |
|---|---|
| C3FD–Llama Planner | typed chemical states, action logits and ordinary Plan text |
| Track-A AR body | native text tokens |
| Track-B DLM | dedicated `7+4N` crystal tokens |
| continuous refiner | atom types, fractional coordinates and lattice tensors |

No raw token ID or logit is compared across these spaces.

## 2. Canonical state

```text
Plan:
  N, elements, counts
  sampled + probabilistic LS/SG/VPA
Program:
  ordered unique elements from Planner semantic_trace
Canvas:
  six lattice values
  request-local site handles, species, XYZ
  committed/masked flags
Provenance:
  predictor/backfill stage, transaction index, old value
```

The runtime owns this state. Models read views of it.

## 3. Planner view

The Planner action trace is authoritative for program order. Canonical Plan
serialization may sort `elements/counts` for identity, but the separate
`species_program` field preserves sampled order.

Validation requires:

- program elements are unique;
- program set equals Plan element set;
- trace counts collapse to exact Plan counts;
- N equals the sum of counts.

## 4. DLM view

The DLM canvas is exact length `7+4N`. Unresolved fields use the real
checkpoint mask ID. N and E tokens are prefilled.

The special-token physical mapping is:

| Family | Physical mapping |
|---|---|
| N | integer 1–20 |
| LA/LB/LC | bin × 0.1 Å; bin 000 illegal in production |
| AA/AB/AG | integer degrees 1–179 |
| E | H–Pu |
| X/Y/Z | bin / 100 on the periodic torus |

Coordinate tokens 000 and 100 are physical aliases. Compatibility with old
checkpoints combines both logits by log-sum-exp; new commits use 000.

Partial validation compares resolved fields and masks. Full parsing occurs only
when no registered mask remains and rejects any non-whitespace text outside the
schema tokens.

## 5. Track-A text view

Track A renders the same Plan/program and canonical arrays as CrysLLMGen text.
Its BPE segmentation is irrelevant to Track B. The shared objects are the Plan,
program and final physical arrays, not token probabilities.

## 6. Continuous view

A complete raw state maps to:

- integer atom types in request-local site-handle order;
- fractional-coordinate tensor wrapped to `[0,1)`;
- lattice matrix;
- atom counts and batch indices.

Continuous output returns through the same state before text or special-token
serialization. Site order is preserved or matched within species.

The current refiner's raw `pred_x/pred_l` are differently parameterized and
are not called force. Candidate E1 uses either the actual deployed transition
response or a separately defined MLIP force teacher on complete structures.

## 7. Transaction semantics

Exactly three transactions exist:

1. Planner species/count action;
2. complete six-value lattice;
3. complete XYZ site.

A transaction updates canonical state, DLM canvas and geometry cache together.
During backfill the old XYZ remains available as a no-op/provisional candidate.
Non-active suffix tokens are immutable.

## 8. Periodic geometry

For a lattice L and fractional displacement d:

\[
d_{ij}^{\mathrm{PBC}}=
\min_{n\in\mathbb Z^3}\|(d+n)L\|_2.
\]

Fractional rounding or an uncertified fixed image radius is not generally exact
for a skewed triclinic cell. Production uses a validated nearest-lattice-vector
implementation or a reduced-cell enumeration with a certified bound.

Hard support:

- positive-length, positive-volume lattice;
- exact N/elements;
- no PBC-equivalent duplicate;
- complete-site minimum distance at least 0.5 Å.

Soft risk:

- species-aware near-collision;
- extreme volume per atom or lattice condition;
- disagreement with probabilistic LS/VPA hints.

Unknown future coordinates never cause hard rejection.

## 9. Required audits

- all 2,481 special-token strings atomic on the real tokenizer;
- all dynamic IDs present in input/output checkpoint rows;
- full MP20 train/validation range, clipping and alias counts;
- exact `7+4N` tokenizer length;
- teacher/predicted prompt context margin;
- strict parser rejects surrounding garbage;
- exact-length sampler has no EOS tail;
- program schedule covers every predictor position once;
- remask changes exactly one registered anchor;
- changing visible suffix changes earlier-anchor logits;
- continuous round-trip preserves composition and site mapping.
