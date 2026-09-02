# Unified Method Plan: Scientific Programmed Anchor–Backfill Denoising

Status: **approved for implementation; B route has priority**

## 1. Scientific question

> How can a scientific LLM program the generation order of a diffusion
> language model so that exact chemistry, future crystal context and periodic
> geometry jointly determine each committed structure?

The answer is **Scientific Programmed Anchor–Backfill Denoising (SPAD)**.

```text
C3FD chemical state/reachable support
             +
Llama residual action preferences
             ↓
exact composition + Compact Plan + Llama species-pointer permutation
             ↓
species construction program
             ↓
exact 7+4N masked-DLM canvas, N/elements prefilled
             ↓
lattice → non-contiguous species anchors → remaining future sites
             ↓
re-mask early anchors while the completed suffix remains visible
             ↓
periodic-feasible backfill → raw crystal → common continuous refinement
```

The single core contribution is not a loose Planner/DLM cascade. C3FD first
certifies the composition support; the terminal Planner-Llama state then
predicts a permutation of exactly those elements, controlling where the DLM
denoises and where it later revisits.

## 2. Model boundaries

### C3FD

C3FD maintains typed chemical state, conservation and reachable actions. It
provides learned base scores over proposal and species/count actions plus
coarse lattice-system, space-group-bucket and volume-per-atom distributions.
It has no fine lattice-length, angle or coordinate logits.

### Llama

The retained Planner Llama reads typed C3FD state embeddings and supplies
residual action preferences. After the final Plan is certified, a small masked
pointer head reads its terminal hidden state, Plan element/count embeddings and
soft structural fields and emits `species_program`. Llama therefore affects
both **what composition is selected** and **which species anchors the DLM
resolves first**.

### DLM

The DLM owns all `7+4N` special-token lattice and coordinate distributions.
It sees the full masked canvas with bidirectional attention, may generate
non-contiguous future positions first and can re-mask an earlier anchor while
keeping later sites visible.

### Continuous refiner

The frozen continuous refiner receives atom types, fractional coordinates and
lattice arrays only after a complete raw crystal exists. Its terminal
trajectory is a common system component. A one-step response/force-to-DLM
feedback mechanism is a later candidate contribution, not part of the SPAD
core.

## 3. Two routes

### Route A — LLM-only executor

C3FD–Llama emits the same Plan and species program. A separate AR body Llama
generates ordinary CrysLLMGen text. It never shares raw token IDs with the DLM.
Route A is the pure-LLM system comparison requested by the teacher; it does not
block B implementation.

### Route B — LLM-programmed DLM

The DLM prompt receives the predicted Compact Plan through its own tokenizer.
The species program is runtime metadata compiled into native DLM position
groups. No AR hidden state, BPE token or AR logit enters the DLM.

Route B is the paper-priority method and receives up to four A800 GPUs.

## 4. Shared state and token boundary

Three spaces remain separate:

| Space | Representation |
|---|---|
| Planner/AR | typed Planner states and ordinary text tokens |
| DLM | one special token per `7+4N` field |
| continuous refiner | integer atom types, fractional-coordinate tensors and lattice matrix |

They meet in a canonical crystal state containing:

- exact N/elements/counts;
- sampled and full-distribution LS/SG/VPA Plan fields;
- species program;
- six lattice values;
- request-local site handles, species and XYZ;
- committed/masked state.

The program is metadata rather than a new DLM token. Canonical element sorting
may format a formula but cannot erase program order.

## 5. Special-token scope

The source defines 2,481 crystal tokens; dynamic bodies use 2,457:

- N: 1–20;
- LA/LB/LC: 0.0–50.0 Å at 0.1 Å;
- AA/AB/AG: 1–179 degrees;
- elements: H–Pu;
- X/Y/Z: 0.00–1.00 at 0.01.

This vocabulary targets ordered MP20 structures with at most 20 sites. It is
not claimed to represent arbitrary disordered/magnetic CIF information.

Production excludes zero-length actions, canonicalizes coordinate 000/100
periodic aliases and uses exact `7+4N` canvases. Full train/validation
coverage and checkpoint embedding rows are audited before training.

## 6. Species program

The current C3FD action trace is canonical by construction and is retained for
composition provenance, not mislabelled as a learned order. A lightweight
masked pointer attached to the same Planner Llama emits an exact permutation
of the final Plan elements; it cannot add, remove or change their counts.

Its MP20-train teacher is a deterministic maximum-contact-tree order from the
periodic element contact graph: start from the element with largest average
contact degree, then add the element most strongly connected to the selected
scaffold. This uses geometry but no energy, hull or evaluation outcome.

## 7. Anchor–backfill DLM

### Predictor

1. Prefill N and every element slot from the Plan.
2. Resolve a valid six-value lattice transaction.
3. Visit species in Planner order and generate one anchor site for each.
4. Complete remaining sites while keeping those anchors visible.

The visited positions can occur anywhere in storage order. Future-first
generation is therefore genuine, not left-to-right text generation disguised
as diffusion.

### Backfill

1. After the full predictor is complete, choose registered early anchors in
   reverse program order.
2. Re-mask one XYZ anchor transaction.
3. Keep lattice and every other site—including positions to its right—visible.
4. Run the DLM again and fill the anchor from full bidirectional context.
5. Preserve the previous triplet as a no-op candidate and apply the same
   periodic geometry support.

An AR model would have to regenerate its suffix after changing an early atom.
SPAD preserves that suffix. This suffix-preserving revision is the key evidence
for DLM necessity.

## 8. Geometry transactions

The generation state uses exactly three transaction types:

1. one Planner composition action;
2. one six-value lattice block;
3. one complete XYZ site triplet.

No X or Y is irreversibly committed before a legal triplet exists. A complete
lattice must have positive volume. A complete site is tested with a validated
triclinic minimum-image calculation; PBC distances below 0.5 Å are illegal,
and species-aware near-collision risk is soft.

During backfill, the old committed coordinate remains the provisional geometry
until a replacement transaction is accepted. This avoids soft-probability
averages that place an atom at a nonexistent mode.

## 9. Training

The species pointer first trains with the Planner/C3FD composition model frozen.
The retained Compact-V2 DLM is then tested without new weights. One later
schedule-matched LoRA uses:

- full MP20 train;
- MP20 teacher Compact-Plan prompts; predicted same-schema Plans are used only
  at inference;
- ordinary random-mask states;
- program-matched anchor/future predictor states;
- complete-state anchor-remask states with teacher suffix visible;
- exact N and element slots always visible.

No CHGNet, hull, continuous-refiner endpoint or generated test outcome enters
SPAD training.

## 10. Adjacent experiments

| ID | Method |
|---|---|
| B0 | retained DLM, historical confidence-ordered exact-plan schedule |
| BC | same weights, canonical species under SPAD transactions |
| BP | same weights, learned Llama-pointer anchor-first schedule |
| BR | BP + one suffix-visible anchor-remask sweep |
| BS | BR + one schedule-matched MP20 LoRA epoch |

B0→BC tests the crystal transaction order. BC→BP isolates learned Llama order. BP→BR tests
the DLM-specific ability to use generated future context. BR→BS tests
train/serve mask alignment.

A small `BR-no-suffix` mechanism cell masks later sites during the same
anchor remask. BR must outperform it on anchor recovery/geometry for the
suffix-context claim.

## 11. Existing evidence and expectation

- C3FD-v2.5 reached `2000/2000` composition-valid and the fused Planner
  reached `1200/1200` at scale.
- Exact dynamic DLM bodies already achieve high execution, but raw Direct has
  remained the main weakness.
- Prior order changes materially changed raw Direct, confirming that DLM
  commitment order is scientifically consequential.
- Full terminal continuous refinement strongly restores validity/stability,
  while endpoint distillation and a force microstudent failed.

SPAD directly targets the unresolved issue: early geometry decisions made
without future context. The intended final target is Strict S.U.N. above 10%
and Meta S.U.N. above 50% on a fixed requested denominator and two common
streams.

## 12. Candidate later contribution

After BS is frozen, Candidate E1 may regress a continuous-refiner response or
CHGNet force into a DLM geometry residual/confidence module on BS-generated
MP20-train states. It must improve raw structure/stability against an
equal-compute zero-response remask before terminal refinement. Failure does not
weaken the SPAD core.
