# Track B: Llama-Programmed Anchor–Backfill Crystal DLM

Status: **approved priority route**

## 1. Core claim

Track B uses the Planner Llama to control DLM execution without sharing token
IDs or inventing a cross-model logit bridge:

- predicted Compact Plan is the condition;
- a Plan-conditioned pointer on the Planner Llama predicts the construction program;
- the DLM owns lattice/coordinate values;
- suffix-visible anchor backfill is the DLM-specific operation.

This is **Scientific Programmed Anchor–Backfill Denoising (SPAD)**.

## 2. Planner-to-DLM signal

The existing typed Planner returns a certified Plan and a canonical provenance
trace. A lightweight species pointer additionally returns:

```text
plan_state:
  N, elements, counts, anion framework, LS, SG bucket, VPA bin
semantic_trace:
  proposal,
  species(Z, oxidation, count),
  ...,
  EOS
species_program:
  an exact permutation of unique plan_state elements
```

The pointer/compiler path:

1. reads the terminal Planner-Llama hidden state, final elements/counts and
   selected LS/SG/VPA fields;
2. masks every element not in the certified Plan and every already selected
   element;
3. predicts an immutable unique-element permutation;
4. verifies exact agreement with Plan elements/counts;
5. maps that permutation to canonical DLM site positions.

The C3FD action state enforces increasing species keys, so its semantic trace
is deliberately not claimed as learned order. The pointer is trained on a
periodic maximum-contact-tree target derived only from MP20-train structures.

The Plan text is encoded by the DLM tokenizer. The program remains structured
metadata and maps elements to DLM site positions. This is the complete
Llama→DLM interface.

## 3. Why no SLA/gate in the first path

AR text values and DLM special tokens have different probability spaces.
Although a separate semantic head could be trained, current code has no such
head, rollout calibration or agreement gate. Adding all three before the first
B result would introduce avoidable train/serve shift.

SPAD needs none of them. Llama controls global condition and schedule; DLM
controls values. A later semantic-value prior remains possible but is not part
of the initial method or claim.

## 4. Exact DLM state

For N sites the body is:

\[
N,LA,LB,LC,AA,AB,AG,(E_i,X_i,Y_i,Z_i)_{i=1}^{N}.
\]

- N and all E positions are prefilled from the exact Plan;
- lattice and XYZ positions begin masked;
- storage order remains canonical;
- the Llama-pointer species program chooses non-contiguous active positions;
- every forward sees the whole canvas with bidirectional attention.

Only exact `7+4N` canvases are production inputs. The historical fixed
87-position EOS-tail path is excluded.

## 5. Stateful sampler API

The existing monotone `generate()` is split into:

```python
canvas = initialize_canvas(prompt, n, element_prefill)
logits = constrained_forward(model, canvas, active_positions)
canvas = commit_transaction(canvas, transaction, value)
canvas = remask_transaction(canvas, transaction)
canvas = resume(model, canvas, remaining_transactions)
```

Required semantics:

- `constrained_forward` refreshes the full model after every transaction;
- `commit_transaction` changes an entire lattice or XYZ transaction;
- `remask_transaction` may mask already committed XYZ;
- every non-active token remains bitwise unchanged;
- completion fails if any registered mask remains.

## 6. Predictor schedule

1. **Inventory prefill:** N and all species tokens.
2. **Lattice transaction:** six fields, final positive Gram determinant.
3. **Species anchors:** one site for each unique species in Planner order.
4. **Future completion:** remaining sites in the same program, with existing
   anchors visible.

An anchor can live later in the stored sequence than an unresolved site. The
program therefore exercises non-contiguous future-first generation.

## 7. Backfill schedule

After a complete predictor:

1. enumerate first anchor of each species in reverse Planner order;
2. preserve its old XYZ as a no-op/provisional candidate;
3. re-mask exactly that XYZ block;
4. keep the lattice and every other site visible;
5. run the DLM with the full suffix;
6. commit one periodic-feasible XYZ transaction;
7. continue through the fixed one-sweep anchor list.

No AR logits are used in backfill because a causal AR distribution for an early
position cannot condition on the fixed future suffix. Llama has already
supplied the program; DLM alone performs posterior infilling.

## 8. Geometry support

### Lattice

- zero-length tokens are excluded;
- six fields form one transaction;
- final Gram determinant must be positive;
- LS/VPA remain soft Plan conditions; SG is not treated as a Wyckoff guarantee.

### Coordinates

- coordinate 000/100 aliases are combined as one torus value;
- XYZ forms one atomic transaction;
- old XYZ remains provisional during remask;
- complete candidates use validated triclinic minimum-image distance;
- below 0.5 Å is illegal; species-aware near-collision is soft.

The first implementation may use current tokenwise commits internally, but the
transaction is not externally visible until all XYZ coordinates are valid.

## 9. Training masks

One source row contributes a weighted mixture:

- ordinary random mask;
- program predictor mask with committed anchors/current/future state;
- full-body correction mask with one earlier anchor hidden and suffix visible.

MP20 teacher Compact-Plan prompts provide SFT input. Predicted Plans are used
at inference through the identical schema. Exact N/elements stay visible.

The schedule-matched endpoint is one LoRA:

- initialization: retained Compact-V2 DLM;
- rank 8, alpha 32, dropout 0.05;
- LR 5e-6;
- effective source batch 16;
- 1,696 optimizer updates;
- one model seed and one endpoint;
- up to 4 A800.

## 10. Required cells

| Cell | Weights | Decoder |
|---|---|---|
| B0 | retained | historical confidence-ordered exact-plan |
| BC | retained | canonical species under SPAD transactions |
| BP | retained | learned Llama-pointer anchor-first |
| BR | retained | BP + suffix-visible remask sweep |
| BS | matched LoRA | BR decoder |

`BR-no-suffix` is a mechanism subset: it remasks the same anchor but also
masks later sites. BR−BR-no-suffix measures the information supplied by future
context.

## 11. DLM necessity evidence

Mechanism:

- future-position step map precedes an earlier stored position;
- changing a visible later site changes earlier-anchor logits;
- remask changes only the selected anchor;
- BR recovers teacher anchor tokens/geometry better than BR-no-suffix.

System:

- requested-denominator body/composition retention;
- raw Direct, graphability, collision distribution and lattice condition;
- raw stability surrogate;
- terminal Strict/Meta S.U.N.;
- calls, NFE and wall time.

An AR generator cannot preserve the suffix while revising an early atom. This
is the non-replaceable role of the DLM.

## 12. Expected result and iteration

B0/BC tests transaction ordering and BC/BP isolates the Llama species program.
BR targets early-site geometric mistakes
using the completed future. BS removes the random-mask versus structured-decoder
mismatch.

Small failures are localized to token coverage, program compilation, remask
state, geometry support or learned mask adaptation. Each receives one adjacent
repair. The final target is Strict/Meta S.U.N. above 10%/50% without selecting
rows, streams or checkpoints.
