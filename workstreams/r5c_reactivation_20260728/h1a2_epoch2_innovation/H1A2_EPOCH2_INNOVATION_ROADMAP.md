# H1-A2 epoch-2 innovation roadmap

Status: recommended ICLR main line, prepared 2026-07-29

## Scientific center

The paper line is:

> A language-model Planner first commits to a chemically coherent material Plan; a diffusion language model realizes the discrete crystal body; a Plan-aware continuous diffusion residual moves the result toward the corresponding structural basin.

H1-A2 epoch-2 is the frozen starting point because it is the last fully-de-novo model whose direct metrics and original A100/CHGNet S.U.N scale have both been reproduced.

The two bottlenecks should not be conflated:

- composition validity is primarily a Planner/formula problem;
- strict/meta S.U.N beyond the small denominator gain from comp-valid is primarily a Plan-to-geometry basin problem.

## Workstream A — chemistry-coherent Planner

### A1: ValidReplay

Continue the epoch-2 LoRA briefly on chemistry-valid MP20 Plan targets, with stratified epoch-2 anchor replay.

This is a low-risk causal diagnostic. It tests whether the 142 chemistry-invalid Plans are partly caused by learning targets that fail the frozen validator.

### A2: JointChem

Add deterministic full-sequence preference terms:

- real MP20 Plan over a charge/Pauling-invalid formula negative;
- real MP20 Plan over a matched-marginal but cross-material rich-field negative.

This attacks formula validity and same-material tuple coherence without inference filtering or S.U.N supervision.

### A3: formula-first factorization, only if A2 is insufficient

Generate explicit element/count/oxidation slots first. Derive `formula`, total atom count, and body slot counts deterministically. Then generate the remaining rich fields conditional on that immutable chemistry state.

This is the larger representation contribution, but it requires tokenizer/schema and R5-C compatibility work. It should be justified by A1/A2 evidence rather than attempted immediately.

## Workstream B — Plan-aware continuous diffusion residual

### Motivation

H1-A2 already has 99.9% structure validity, but only 9.4% strict and 47.4% meta S.U.N. The failure is therefore not mostly parseability or geometric validity. Comparable experiments locate the remaining gap in low-hull structural basins, especially oxides, chalcogenides, and intermetallics.

### Architecture

Freeze every parameter in CrysLLMGen `model_494`. Add:

1. a small Plan encoder for formula/counts, lattice class, space-group bucket, and volume bin;
2. a zero-initialized per-atom residual head for the coordinate score;
3. a zero-initialized graph residual head for lattice noise;
4. optional FiLM modulation after the stock atom/time projection.

At initialization:

```text
pred_coord = stock_pred_coord
pred_lattice = stock_pred_lattice
```

After training:

```text
pred_coord = stock_pred_coord + residual_coord(plan, t, stock_features)
pred_lattice = stock_pred_lattice + residual_lattice(plan, t, stock_graph_features)
```

The lowest-risk implementation wraps the two stock decoder outputs. A later ablation may add FiLM inside the node features.

### Exact local insertion points

Recovered source:

- `crystal_dlm/wqcodiff/crysllmgen/upstream/models_ddpm/diffusion.py`
  - `CSPDiffusion.forward`: stock decoder call around line 78;
  - `CSPDiffusion.sample`: corrector and predictor decoder calls around lines 152 and 168.
- `crystal_dlm/wqcodiff/crysllmgen/upstream/models_ddpm/cspnet.py`
  - per-atom representation after `atom_latent_emb` around line 282;
  - graph representation before `lattice_out` around lines 292–294.

The stock denoiser has hidden width 512, six CSP layers, sinusoidal distance features with 128 frequencies, fully-connected graph edges, maximum 20 neighbors, cutoff 7, LayerNorm, and inner-product lattice output.

### Training

Use only MP20 teacher structures and the stock diffusion corruption process:

- sample the same lattice and wrapped-coordinate noise as the stock objective;
- minimize coordinate-score and lattice-noise reconstruction;
- keep atom types, atom count, and order fixed;
- stratify batches by ground-truth chemistry family;
- never stratify by generated S.U.N, CHGNet energy, hull, MP API result, or MLIP score.

Required invariants:

- null Plan yields exactly the stock model output;
- zero-initialized residual reproduces stock output before training;
- shuffled Plan is worse than matched Plan on held-out reconstruction;
- stock output drift stays within a preregistered bound;
- no topology, composition, retry, replacement, or rerank path exists.

### Selection

Select without S.U.N using:

- held-out matched reconstruction loss;
- matched-minus-shuffled Plan identity;
- exact-null equality;
- stock-output drift;
- lattice-volume and minimum-distance sanity;
- chemistry-family worst-group loss.

Run S.U.N only after the checkpoint is frozen.

## Factorial attribution

The confirmatory design is 2×2:

| Arm | Planner | Continuous refiner |
|---|---|---|
| A | frozen H1-A2 epoch-2 | stock `model_494` |
| B | selected chemistry-coherent Planner | stock `model_494` |
| C | frozen H1-A2 epoch-2 | Plan-aware residual |
| D | selected chemistry-coherent Planner | Plan-aware residual |

This separates:

- Planner effect on composition validity: `B - A`;
- residual effect on structural basin/S.U.N: `C - A`;
- combined effect and interaction: `D - A` and `(D - C) - (B - A)`.

All arms share attempt IDs, source prompts, Planner seeds where applicable, R5-C reverse noise, parent-refiner noise, and evaluation order. No arm gets retry or replacement.

## Execution sequence

1. Freeze all baseline identities and row ledgers.
2. Implement and unit-test A1/A2 data and sequence scoring locally.
3. Run Plan-only training and selection.
4. Run paired-256 A vs B with stock refiner.
5. Stop if composition validity does not improve without distribution gaming.
6. Implement the zero-initialized residual behind an exact-null feature flag.
7. Verify stock parity before any residual training.
8. Train residual candidates and select on held-out reconstruction only.
9. Run paired-256 A/B/C/D.
10. Promote a frozen design to 1,000 attempts and at least three Planner seeds.

The initial paired-256 is a mechanism screen. Paper claims require the full-1000 panel and seed robustness.

## Success criteria

Minimum full-panel targets:

- composition validity at least 90% and higher than epoch-2;
- structure validity at least 99%;
- strict S.U.N at least 9.4% and preferably above 10%;
- meta S.U.N at least 49.4%, with a goal above 50%;
- no all-metal shortcut inflation;
- novelty, uniqueness, atom-count, arity, and chemistry-family coverage noninferior;
- matched-vs-shuffled Plan identity remains positive;
- A/B/C/D interaction is reported rather than hiding a negative component.

## Stop rules

- Stop A1/A2 if comp-valid gains come primarily from all-metal inflation.
- Stop Planner continuation if atom count or chemistry marginals reproduce the H1-A3/A4 drift.
- Stop the residual if exact-null fails or stock parity is not exact before training.
- Stop the residual if matched identity rises but meta S.U.N falls, as happened in the old continuation branch.
- Do not use more epochs as the default response to a failed gate.
- Do not introduce MLIP guidance, energy filtering, candidate pools, repair, or reranking.

## Immediate implementation deliverable

The next code change should be limited to Workstream A data construction, sequence-level scoring, and tests. No A800 job should be launched until:

- the exact historical source-row ledgers are frozen;
- negative generation is deterministic;
- prompt tokens are excluded from sequence ranking;
- S.U.N/MLIP/API leakage tests pass;
- the checkpoint-selection rule is serialized.

