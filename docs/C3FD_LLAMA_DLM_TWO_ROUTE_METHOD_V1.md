# C3FD–Llama conditioning for the H1-A2 crystal DLM

Date: 2026-08-31

Status: approved method specification

## Method center

This is a DLM work. C3FD and Llama produce conditions; the masked crystal DLM
is the model that realizes lattice parameters, species placement, and
fractional coordinates. Model494 is a fixed continuous refiner.

```text
C3FD composition proposal
        ↓
Llama Rich-Plan conditioner (F or M)
        ↓ exact historical seven-line H1-A2 Plan
historical parser → full plan_state JSON
        ↓
exact-composition masked crystal DLM
        ↓
raw structure → fixed model494 tau800 → refined structure
```

## Shared hard contract

C3FD samples and locks a benchmark-compatible
`formula/N/elements/counts`. The Llama conditioner cannot change the formula.
It autoregressively predicts exactly:

```text
anion: ...
charge: ...
lattice: ...
spacegroup: ...
volume: ...
end: plan
```

The formula line is prefilled. Concatenation produces the byte-compatible
historical `h1_rich_plan_v1` seven-line interface. Missing/invalid fields are
failed attempts; no repair, default insertion, retry, replacement, survivor
filter, rerank, or top-up is permitted.

## Route F — formula-conditioned Rich Expander

F is the feasible fallback and formula-only conditioning ablation.

Input:

```text
fixed_formula: NaCl
formula: NaCl
```

Target: the five rich fields plus `end: plan`. Training pairs come only from
MP20 train/validation teacher structures. At inference, MP20 formula is
replaced by the frozen C3FD formula using the identical prompt template.

## Route M — C3FD semantic-prefix Rich Expander

M integrates C3FD into the Llama conditioner with the smallest reliable code
change: serialize the already available C3FD semantic state as a fixed-format
text prefix consumed by the normal tokenizer and LoRA path.

Input:

```text
c3fd_state:
N=2
species_valence_counts=Na:+1:1|Cl:-1:1
family=halide
net_charge=0
certificate=benchmark_compatible
formula: NaCl
```

The output target is identical to F. M therefore tests whether C3FD's
valence/certificate state informs the rich structural prior beyond the formula,
without adding a second backbone, custom attention, or a post-hoc completion
model. C3FD remains frozen; only the existing Llama LoRA is trained.

## Why this belongs in a DLM paper

F and M are conditioning interfaces, not independent headline generators. The
scientific question is whether a hard-valid proposal and a richer soft Plan
improve the DLM's Plan-to-structure conversion.

- Planner metrics establish condition availability and correctness.
- Raw DLM body/Direct/CHGNet metrics are the primary mechanism evidence.
- Refined and official S.U.N. are system endpoints.
- Refined-only improvement cannot support a new DLM claim.

The minimal controlled comparison holds the C3FD composition ledger, Llama
base, LoRA recipe, output schema, DLM checkpoint, body noise, model494, and
evaluation fixed. `M−F` isolates the value of the C3FD semantic prefix.

## Training contract

- same historical H1-A2 rich Planner tokenizer and base for F/M;
- same historical rich Planner adapter initialization for interface recovery;
- MP20 train/validation teacher rich suffixes only;
- two fixed training seeds per route, one fixed adaptation epoch, final
  checkpoint only;
- no predicted-view SFT, compact V2, energy, hull, S.U.N., DLM outcome,
  alignment/listwise objective, or checkpoint/seed selection;
- categorical output grammar may constrain syntax, but never insert a
  scientific value after generation.

## Evaluation contract

After a small syntax/interface canary, freeze one new C3FD all-attempt
fixed256 composition ledger. Generate F/M Plans on the same compositions and
run both through the same old H1-A2 rich DLM, streams17/18, and model494 tau800.
Evaluate raw first, refined second, and query official MP once for the final
prospective cohort. Preserve every failed row on the fixed denominator.

Historical H1-A2 `9.40/47.40%` and corrected exact `8.58/46.08%` are references,
not tuning or result-deletion gates.
