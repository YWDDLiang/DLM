# Method at a glance

## The failure mode

A crystal language model must make two kinds of decisions:

- **global scientific decisions:** which composition and coarse structural
  regime should exist;
- **relational geometric decisions:** whether lattice and all sites form a
  valid, low-energy periodic object.

Flattening both into one token sequence creates a salience problem. The model
can see every token, yet the few relationships that decide failure compete
with hundreds of ordinary token predictions.

## Why introduce a residual if the DLM already has global attention?

Global visibility answers **“can information be accessed?”** It does not answer
**“which computation is guaranteed, in which coordinate system, and with what
gradient priority?”**

1. A Transformer can attend to every lattice and coordinate token, but exact
   triclinic minimum-image distance is a nonlinear periodic computation. It is
   not explicitly represented in token CE.
2. One catastrophic pair can invalidate a crystal. In a 20-atom structure it
   competes with 190 pairs and dozens of token losses, so global training can
   underweight it even when all tokens are visible.
3. The relevant relation is species dependent: the same distance can be safe
   or unsafe for different atom pairs.
4. Early masked states are uncertain. The model needs to reason over expected
   lattice/site geometry before committing tokens, not only inspect finished
   coordinates afterward.

G2 therefore acts as a **scientific-salience residual**. It deterministically
reconstructs metric, strict-PBC distance, species-pair, RDF and coordination
features from q0; aggregates them into site messages; and adds a
zero-initialized correction to q1 logits. The global DLM remains responsible for
all broad context. The residual gives high-consequence periodic relations a
short, explicit gradient path without replacing global attention.

This is analogous to giving a vision Transformer an equivariant geometric
path: the input was always visible, but the scientifically correct relation is
made easy to compute and hard to ignore.

## The coupled generator

```text
C3FD reachable support × Llama learned prior
                    ↓
             one global Plan z
                    ↓ exact train/serve interface
       global masked-DLM context q0
                    ↕ G2 scientific-salience residual
              raw crystal x0
                    ↓ model494 terminal diffusion
              final crystal x*
```

## Four coupled modules

### Science-Constrained LLM Planner

- **Input:** a partial typed composition state.
- **Output:** one composition-valid Compact-V2 Plan.
- **Function:** C3FD supplies reachable actions; Llama learns which reachable
  path resembles the materials distribution and predicts soft structural hints.
- **Evidence:** C3FD reaches `2000/2000` composition-valid proposals and the
  fused Planner reaches `1200/1200`; nonzero action KL on `87.05%` of decisions
  confirms that Llama remains active.

### Plan-Conditioned Crystal DLM

- **Input:** exact N/elements/counts and global Plan hints.
- **Output:** one raw lattice plus N ordered species/fractional sites.
- **Function:** the dynamic `7+4N` language makes atom cardinality an exact
  condition and lets parallel masked denoising focus on geometry.
- **Evidence:** `248/248` strict round trips preserve species order and validity;
  the Plan1200 profile yields `1139/1159` valid CIFs.

### G2 Periodic-Relation Residual

- **Input:** uncertain q0 lattice/site token distributions and the shared Plan.
- **Output:** a zero-initialized correction to geometry-token logits.
- **Function:** strict triclinic PBC, species-pair distance, RDF, overlap and
  coordination relations receive a short, explicit path inside denoising.
- **Evidence:** fresh BASE→G2 refined S.U.N. changes `19/111→24/117`, paired
  official hull improves by `16.43 meV/atom`, and full-epoch raw Direct changes
  `118→128/256`.

### Frozen model494 Terminal Diffusion

- **Input:** the raw exact-composition DLM structure.
- **Output:** one fine-scale refined structure with the same attempt identity.
- **Function:** isolate a fixed basin transition from learned language
  realization; it is not a candidate selector.
- **Evidence:** a matched 512-row mechanism run changes raw→tau800 Strict/Meta
  `10/66→48/230`. The independent Plan1200 tau800 profile reaches `81/486` on
  main1000, supporting the same terminal transition at scale.

## What each result measures

- `comp_valid`: whether the scientific LLM Planner worked.
- raw Direct/energy/hull: whether the DLM learned crystal realization and
  stability before downstream refinement.
- refined hull and S.U.N.: whether the complete coupled generator works.

The decomposition is diagnostic; the method remains one conditional
probability flow connected by the Plan state.

On the independent scale profile, the complete tau800 pipeline reaches
`81/1000` Strict and `486/1000` Meta S.U.N. This supports
scale transfer while identifying Strict stability as the remaining bottleneck.
