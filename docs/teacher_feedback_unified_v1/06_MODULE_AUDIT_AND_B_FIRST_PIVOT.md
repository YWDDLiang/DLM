# Module Audit and B-First Method Pivot

Status: **implementation audit accepted; this document supersedes SLA/gate as the first Track-B path**

## 1. Audit outcome

Three independent implementation audits converged on one result:

- the typed C3FD–Llama Planner is reusable;
- the DLM special-token representation and bidirectional backbone are reusable;
- the production sampler does not yet expose completed-state re-masking;
- direct AR-token/DLM-token logit fusion is neither implemented nor necessary
  for the first strong B result.

Track B is therefore simplified to:

```text
C3FD reachable chemical state
        -> Llama residual action decisions
        -> exact composition + predicted Compact Plan
        -> terminal Llama species-pointer permutation
        -> species construction program
        -> exact 7+4N DLM canvas with N/E prefilled
        -> non-contiguous anchor-first generation
        -> future completion
        -> suffix-visible anchor remask/backfill
        -> periodic-feasible raw crystal
```

This is named **Scientific Programmed Anchor–Backfill Denoising (SPAD)**.

## 2. Teacher-feedback correspondence

| Teacher requirement | SPAD implementation |
|---|---|
| scientific knowledge modifies Llama decoding | C3FD support/base scores and Llama residuals jointly choose typed chemical actions |
| Llama should decide DLM order | a masked pointer on terminal Llama state predicts the exact Plan-element permutation compiled into DLM positions |
| use DLM rather than a second AR model | DLM commits non-contiguous future positions and later infills an earlier remasked anchor with the suffix still visible |
| lattice/coordinates should receive prior knowledge | predicted LS/SG/VPA condition the prompt; lattice and PBC state constrain transactions |
| connect to continuous diffusion | complete raw SPAD state can later receive a force-calibrated continuous-response correction |

The pointer is attached to the same Planner Llama and cannot alter the
C3FD-certified candidate set. It connects chemistry and DLM execution without
sharing token IDs.

## 3. Special-token audit

Source vocabulary contains 2,481 crystal special-token strings. Dynamic
`7+4N` uses 2,457:

| Family | Range | Count |
|---|---:|---:|
| atom count | 1–20 | 20 |
| LA/LB/LC | 0.0–50.0 Å, 0.1 Å step | 1,503 |
| AA/AB/AG | 1–179 degrees | 537 |
| elements | H–Pu | 94 |
| X/Y/Z | 0.00–1.00, 0.01 step | 303 |

The real retained step-3392 tokenizer has vocabulary size 128,830.
Boundary probes for N, lattice, angle, Pu and coordinate 000/100 each encode as
one token. Its pad/eos ID is 126081 and differs from the current DLM mask ID
126336.

CPU job 39507 completed the full empirical audit: all 27,136 MP20 train and
9,047 validation rows parse, tokenize to exact `7+4N`, and fit the supported
range. All 2,481 special tokens are atomic and unique; checkpoint input/output
rows cover the 128,830-token vocabulary. Teacher contexts peak at 238/234
tokens, leaving 144/148 tokens under the 382-token limit. Coordinate 100 occurs
7,829/2,518 times and is handled as a periodic alias of 000.

### Required fixes before B training

1. verify all 2,481 token strings are atomic and all 2,457 dynamic IDs have
   input/output rows after checkpoint reload;
2. exclude zero-length tokens from production support and count any MP20
   length clipping at 50 Å;
3. combine coordinate 000/100 probability as one torus value and emit canonical
   000;
4. use only exact `7+4N` canvases; the old universal 87-position EOS-tail
   sampler is historical compatibility code;
5. make strict parsing reject non-whitespace text outside schema tokens;
6. require every programmed schedule to cover each dynamic position exactly
   once and leave no mask;
7. verify prompt + `7+4N` stays within the 382-token contract for teacher and
   predicted Plans.

## 4. What the DLM already supports

- full bidirectional attention;
- arbitrary non-contiguous generation-position groups;
- all-position logits on every forward;
- a fresh full forward after each monotone commit;
- exact N and element prefill;
- forced-mask training states.

The limitation is the runtime state machine: current production generation is
mask-to-token only. It cannot turn a committed token back into a mask.

## 5. SPAD execution program

The Planner `semantic_trace` records species/count provenance but C3FD enforces
strictly increasing species keys. It is therefore canonical, not a learned
permutation. A small masked pointer reads terminal Planner-Llama state,
elements/counts and soft Plan fields and emits `species_program`.

The MP20-train target is a periodic maximum-contact-tree order: choose the
largest average-contact-degree element as root, then repeatedly choose the
element most connected to the current scaffold. C3FD, Llama and composition
heads remain frozen while this pointer trains.

The program must be an exact permutation of unique Plan elements.

## 6. Anchor–backfill schedule

1. Prefill exact N and every element slot.
2. Generate and validate the six-value lattice transaction.
3. For each species in Planner order, generate one anchor site at its native
   DLM positions, even when those positions occur later in storage order.
4. Complete all remaining sites, preserving the generated anchors.
5. Select the first anchor of each species in reverse Planner order.
6. Re-mask one anchor XYZ transaction while keeping all other lattice/sites
   visible.
7. Run DLM again and backfill that anchor from full suffix context.
8. Keep the old triplet as an explicit no-op option and accept a correction
   only through the same geometry support.

The first experiment uses exactly one anchor-remask sweep. It does not tune the
number of sweeps from evaluation outcomes.

## 7. Why DLM is necessary

An AR body model must invalidate and regenerate its suffix after changing an
early atom. SPAD instead keeps the later crystal visible and changes only an
earlier masked site. The DLM contribution is therefore:

> suffix-preserving, non-causal crystal revision under a Llama-generated
> scientific construction program.

The required mechanism test changes a visible later site and verifies that an
earlier masked anchor's logits change. A context-independent toy model is not
sufficient.

The real retained checkpoint now passes this test (job 39513): a one-bin
change to a later site's Z token changes the earlier masked anchor logits on
all three coordinate axes. The maximum absolute changes are 0.125, 0.09375
and 0.15625 for X/Y/Z.

## 8. Train/serve signal

There are only two Llama-to-DLM signals:

1. **condition:** predicted Compact Plan text is tokenized by the DLM's own
   tokenizer;
2. **program:** the Llama pointer permutation is compiled into DLM position
   groups and remask transactions.

No AR BPE token, AR hidden state or AR logit enters the DLM.

DLM training uses:

- MP20 teacher same-schema Plan prompts; predicted Plans are inference inputs;
- ordinary random masks to retain general denoising;
- program-matched predictor masks;
- complete-state anchor-remask masks with teacher suffix visible.

The target remains the MP20 special-token body. Generated/refined evaluation
outcomes do not enter training.

## 9. Adjacent evidence

| Cell | Difference |
|---|---|
| BC | retained DLM, canonical monotone schedule |
| BH | same weights, deterministic chemistry heuristic schedule |
| BP | same weights, learned Llama-pointer anchor-first schedule |
| BR | BP + one suffix-visible anchor-remask sweep |
| BS | BR after one schedule-matched MP20 LoRA epoch |
| BR-no-suffix | same remask position but mask the later sites too; mechanism diagnostic |

BC/BH/BP isolates learned Llama order from fixed order. BP→BR tests DLM-only suffix-visible revision.
BR→BS tests matched training. BR-no-suffix is a small mechanism comparison, not
a full headline cell.

Track A is the LLM-only system comparison and no longer supplies a prerequisite
body adapter to B.

The learned program is also nontrivial in deployment: on the fixed 256-Plan
ledger, 229 programs differ from canonical atomic-number order while all 256
Plan texts remain bit-identical to the control.

## 10. Later stability contribution

After SPAD is established, a complete raw BS crystal may receive Candidate E1:

- use the deployed continuous refiner's one-step response on BS-generated
  MP20-train states;
- regress confidence/direction into a DLM geometry residual or use one
  response-guided remask;
- compare against an equal-compute zero-response correction;
- require improvement before terminal full refinement.

This stage is downstream of the core contribution and cannot delay the first
SPAD result.
