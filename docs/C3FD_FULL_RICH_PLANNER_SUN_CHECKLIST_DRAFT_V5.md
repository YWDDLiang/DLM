# C3FD + Llama Rich Planner S.U.N. checklist draft v5

Date: 2026-08-31

Status: **approved concept; superseded by active execution checklist v6**

This draft supersedes deterministic field completion, compact V2 reruns, and
alignment. Job38914 is cancelled and its partial outputs are excluded.

## Shared scientific objective

Preserve C3FD's demonstrated composition validity while restoring the exact
historical H1-A2 Planner→parser→full-plan-state→rich-DLM interface. The external
Planner output remains exactly:

```text
formula: ...
anion: ...
charge: ...
lattice: ...
spacegroup: ...
volume: ...
end: plan
```

The historical parser may derive `N/elements/counts` from the emitted formula
and add historical technical bookkeeping. It may not invent a missing
scientific rich field. Missing/invalid output is one failed attempt with no
retry, replacement, survivor filter, rerank, or top-up.

## Candidate F — feasible cascade: C3FD composition → Llama Rich Expander

Status: **feasible engineering route and fallback**.

1. C3FD samples a benchmark-compatible composition and locks
   `formula/N/elements/counts`.
2. The canonical `formula:` line is prefilled and cannot be changed.
3. A Llama+LoRA Rich Expander autoregressively emits
   `anion/charge/lattice/spacegroup/volume/end`.
4. The canonical seven-line Plan passes through the historical parser and
   `build_body_prompt()` into the old H1-A2 rich DLM.

Training uses MP20 teacher compositions as inputs and the corresponding
teacher rich suffixes as targets. Inference substitutes C3FD compositions into
the identical prompt template. This is learned conditional Plan expansion, not
deterministic result completion.

## Candidate M — integrated conditioning interface: C3FD-Conditioned Rich Planner (C3FD-CRP)

Status: **supporting conditioning hypothesis, not the paper headline**.

Candidate F passes only the formula. C3FD-CRP additionally exposes the
composition-generation state to Llama as learned prefix embeddings:

- exact `N` and species/count multiset;
- valence-resolved C3FD action trace and charge ledger;
- anion-family/proposal state and terminal certificate;
- calibrated Planner uncertainty summaries, but no structure, energy, hull,
  DLM, refiner, or test outcome.

A small projector maps this frozen C3FD state to `K` soft prefix tokens. The
prefix tokens condition the same Llama backbone and LoRA while it generates the
H1-A2 rich suffix autoregressively. The formula remains a hard C3FD anchor.

```text
C3FD constrained composition state
        ├── canonical formula hard anchor
        └── learned semantic prefix projector
                         ↓
                Llama + LoRA
                         ↓
       anion/charge/lattice/SG/volume/end
                         ↓
        historical H1-A2 parser + old rich DLM
```

C3FD remains frozen during Rich-Planner training, so its composition guarantee
cannot be traded away. Train only the prefix projector and Llama LoRA on MP20
teacher full-rich Plans. Candidate M is therefore not a repair model: C3FD
controls hard chemical feasibility, while Llama models the multimodal
composition-conditioned structural prior.

## DLM centrality lock

This remains a DLM paper. Candidate F and Candidate M only define where the
DLM's conditioning comes from; neither is sufficient as the main contribution.

The method-facing object is a **C3FD-grounded hard–soft plan-conditioned masked
DLM**:

- hard channel: C3FD exact `N/elements/counts` are frozen during masked body
  generation;
- soft channel: the Llama-generated full H1-A2 rich Plan conditions structural
  token prediction;
- realization: the DLM, not the Planner, generates lattice parameters,
  species placement, and fractional coordinates;
- model494 remains a fixed downstream refiner and is reported separately.

Planner validity and rich-field accuracy are interface diagnostics. The central
mechanism evidence is raw DLM realization—body/Direct/CHGNet and the conversion
from valid Plan to valid/stable structure. Refined-only gains cannot support a
new DLM claim.

## Paper interpretation

The method separates two kinds of planning uncertainty:

- **hard chemistry:** exact cardinality, species/count conservation, charge
  reachability, and benchmark composition validity, handled by C3FD;
- **soft structural prior:** lattice family, symmetry range, and volume regime,
  modeled autoregressively by Llama.

Candidate F is the formula-only conditioning ablation; Candidate M tests
whether C3FD's valence/certificate state adds information beyond the formula.
These first establish a reliable conditioner. The paper contribution is made
only if a full-rich-conditioned masked DLM converts that conditioning into
better raw structure realization while model494 is fixed.

## Frozen training boundary

- MP20 train/validation teacher rich Plans only;
- C3FD-v2.5 frozen for both candidates;
- same Llama base, output schema, LoRA configuration, epochs, and two training
  seeds for F and M;
- no compact V2, deterministic completion, predicted-view SFT, DLM retraining,
  alignment/listwise data, energy, hull, S.U.N., or test outcomes;
- final checkpoint only; no seed/checkpoint selection.

## Prospective comparison after approval

- Freeze one new all-attempt C3FD fixed256 composition ledger before executor
  outcomes.
- Generate Candidate F and Candidate M Plans on the same compositions/order.
- Feed both through the same old H1-A2 rich DLM, body streams17/18, and
  model494 tau800.
- Preserve every failed Plan/body/refinement row on the fixed denominator.
- Evaluate raw first and refined second; then execute one official MP query.
- Report F and M separately plus historical H1-A2 `9.40/47.40%` and corrected
  exact `8.58/46.08%` references. Targets are not tuning or deletion gates.

Pending faithful H0/R0S and compact-V2 development cells may join the same
official-input union only as separately labelled development evidence. The
train-only alignment pool, malformed canary, cancelled job38914, and
already-official D3PO cells stay excluded.

## Ordered actions after approval

- [ ] Freeze the composition-prefill/full-rich-suffix training schema.
- [ ] Build Candidate F MP20 teacher train/validation data.
- [ ] Build the frozen C3FD-state prefix representation and projector for M.
- [ ] Add one shared Llama+LoRA trainer supporting formula-only and prefix modes.
- [ ] Train F and M with identical two-seed contracts and no selection.
- [ ] Run one small interface-only canary without tuning either method.
- [ ] Freeze a new C3FD fixed256 all-attempt prospective ledger.
- [ ] Run F/M × streams17/18 generation and model494 refinement.
- [ ] Run one raw-first/refined offline evaluation and one official MP query.
- [ ] Finalize S.U.N., ablation, contribution grade, and paper story.

## Approval boundary

- `F only`: fastest fallback, formula-only conditional Rich Expander.
- `F + M`: recommended; F is the ablation and C3FD-CRP is the paper method.

No training, generation, evaluation, or query begins before explicit approval.
