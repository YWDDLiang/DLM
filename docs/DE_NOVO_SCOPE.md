# Plan-source taxonomy and R5-C scope

## Main distinction

Training-time extraction of Plan labels from MP-20 is supervision. Inference-
time replay of an MP-20-derived or frozen Plan is conditioning. The main
H1-A2 route is fully de novo at the Plan level only when its Plan is sampled
from the learned condition source.

## Three Plan sources

### A_learned: main fully de novo system

```text
learned rich Plan
    -> composition/N anchors
    -> Crystal DLM geometry completion
    -> continuous refinement
```

This route generates formula and coarse fields rather than retrieving a
dataset row. It carries the fully de novo claim.

### C_gold / R5-C: conditional executor reference

R5-C derives a global specification from a held-out MP-20 structure. The Plan
may contain exact composition/cardinality and coarse lattice, space-group, and
volume bins, but withholds exact lattice values, coordinates, material ID, and
energy. The DLM must still realize quantized geometry and the refiner must
produce a continuous structure.

Recommended name:

> **Gold Plan (R5-C; conditional executor reference)**

It is not Planner-free, not fully de novo at the Plan level, and not a
mathematical upper bound on novelty, diversity, or S.U.N.

### C_replay: frozen generated-Plan control

A frozen H1-A2 or R03 Plan bundle fixes the scientific conditions so body and
refiner changes can be paired. It is a downstream reproducibility and mechanism
control, not a gold Plan and not a new learned-Plan cohort.

## Paper use

| Source | Main text | Appendix | Valid interpretation |
|---|---:|---:|---|
| `A_learned` | yes | yes | complete de novo system |
| `C_gold / R5-C` | yes, separate conditional task/reference | yes | given-specification executor capacity |
| `C_replay` | mechanism only | yes | fixed-condition paired control |

Historical R5-C adjusted results remain legacy context because their evaluator,
survivor selection, and denominator differ from the modern H1-A2 contract.
They cannot be subtracted from H1-A2 and called a causal Planner effect.

## Matched A-vs-C contract

If a matched comparison is run later:

- both sources use the same rich Plan schema;
- gold Plans expose no exact target geometry or energy;
- DLM checkpoint, temperature, anchors, support, field schedule, refiner, and
  evaluator are identical;
- one body attempt is made per Plan; no retry, filtering, replacement, or
  reranking;
- every body success is refined rather than selecting a first-success prefix;
- all failures remain in the requested-attempt denominator;
- analyses report both raw source differences and chemistry-standardized
  conversion differences.

The table title should be:

> **Condition-source decomposition: learned de novo Plans versus MP-20-derived
> gold Plans**

## No-separate-Planner alternatives

A future self-planning DLM could generate a minimal formula/N header and then
instantiate the exact body in a second pass. That can remain fully de novo but
is not implemented. Feeding empirical or user Plans directly to the existing
body is specification-conditioned generation, not a no-Plan generative model.
