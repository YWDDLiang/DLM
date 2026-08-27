# Planner Physics-Inspired Element Embedding Feasibility V1

Date: 2026-08-27

Status: isolated design/audit only; not part of the active DLM training arm.

## Source mechanism

CrysVCD represents a composition as element–oxidation-state/count pairs. Its
formula Transformer combines three vectors for every pair:

```text
learned element-valence token embedding
+ projected full/valence-shell electronic configuration
+ learned count embedding (0..20)
```

The model then generates a charge-balanced composition before the geometric
diffusion stage.

## Why it is not a direct drop-in

The current H1 Planner:

- is a pretrained text LLaMA with LoRA;
- emits the six-field/seven-line `h1_rich_plan_v1` text;
- uses ordinary BPE text tokens for element symbols and digits;
- freezes base embeddings under the current LoRA target modules;
- computes formula-derived N/elements/counts only after parsing.

Adding a numerical electronic-configuration vector to ordinary BPE rows would
modify text tokens shared with unrelated contexts and would not reliably bind a
symbol to one oxidation state. A faithful adaptation needs explicit
element-valence/count tokens or an auxiliary head.

## Recommended adaptation

Keep the public Planner output unchanged. Add a train-only auxiliary sequence:

```text
valence_plan:
<EL_Fe_OX_2P> <COUNT_1>
<EL_Fe_OX_3P> <COUNT_2>
<EL_O_OX_2N>  <COUNT_4>
end: valence_plan

rich_plan:
formula: Fe3O4
anion: oxide
charge: neutral_plausible
lattice: ...
spacegroup: ...
volume: ...
end: plan
```

Implementation options, in order:

1. **Auxiliary valence head:** share the LLaMA hidden state, predict explicit
   valence/count tokens with a small trainable head, and retain the existing
   rich-Plan text head. Lowest risk to the public tokenizer.
2. **Special-token prefix:** add explicit element-valence/count tokens to the
   tokenizer, initialize their trainable deltas from electronic configurations,
   supervise the prefix, then deterministically strip it before exposing the
   seven-line rich Plan.
3. **Dedicated composition Transformer:** reproduce the CrysVCD architecture and
   feed its formula into the existing rich-field Planner. Highest fidelity but a
   different model and contribution; out of current scope.

Option 1 is the recommended first implementation because it preserves the
existing text Planner and prevents special-token embedding changes from
contaminating general LLaMA tokens.

## Required data audit

Before implementation, measure on the exact Planner train/val and frozen raw1000:

- oxidation-state assignment coverage;
- unique element–oxidation-state pairs;
- mixed-valence frequency;
- count range and count-token coverage;
- charge-neutral assignment rate;
- formulas with multiple valid oxidation assignments;
- all-metal cases where oxidation state zero is the appropriate representation;
- lanthanide/actinide/exotic-state exclusions relative to the current Plan
  support.

No missing assignment may silently become `unstable` or be dropped from final
evaluation.

## Feasibility gates

- at least 95% train/validation formula coverage, with every uncovered category
  disclosed;
- raw1000 coverage no worse than train by more than 3 pp;
- rich-Plan parse rate and formula/N distribution noninferior to P0;
- no increase in all-metal or unary shortcuts;
- auxiliary valence accuracy positive on exact formula/count matching;
- downstream DLM evaluation remains a separate factorial arm.

## Experimental isolation

The active DLM sufficiency and stability-preference program uses the frozen P0
Plans. A physics-embedding Planner candidate may be evaluated only after:

1. the CPU-only coverage audit passes;
2. the DLM sufficient base is frozen;
3. the candidate is compared against P0 with the same frozen DLM/refiner;
4. proposal-mix changes and within-Plan conversion are reported separately.

It must not be used to rescue a failed DLM candidate or to replace the frozen
raw1000 two-round test cohort.
