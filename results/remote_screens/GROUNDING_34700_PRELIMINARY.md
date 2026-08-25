# Counterfactual grounding preliminary screen

## Identity

- Slurm job: `34700`
- Candidate: composition-matched counterfactual Plan grounding
- Control: same B0 initialization and factual CE continuation
- Schedule: standard H1-A2 lattice -> X -> Y -> Z
- Status: training active; fixed-256 body/refine not complete

## Data contract

| Split | Rows | Grounding eligible | Coverage |
|---|---:|---:|---:|
| train | 27,136 | 27,117 | 99.930% |
| validation | 9,047 | 9,025 | 99.757% |
| test | 9,046 | 9,031 | 99.834% |

Factual prompt reconstruction is byte-identical to the frozen prompt. The
counterfactual keeps formula, N, elements, counts, anion and charge unchanged
and changes only lattice family, space-group bucket and volume bin.

## First matched checkpoint

At optimizer step 500:

| Arm | Factual validation CE |
|---|---:|
| CE-only control | 1.915292 |
| Grounding candidate | 1.623997 |

The candidate is 15.2% lower at the first matched checkpoint. The step-500
candidate training batch had a factual-minus-counterfactual geometry margin of
+0.867.

This is a mechanism screen, not a final result. Training batches use random
masks and their individual margins vary. Promotion still requires the completed
training endpoint and fixed-256 body/refiner comparison.

