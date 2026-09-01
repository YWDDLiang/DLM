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

- C3FD prevents the LLM from spending mass outside chemically reachable
  action space.
- Llama chooses among reachable paths using a learned materials prior.
- The Plan fixes exact chemistry and conditions every geometric prediction.
- `7+4N` makes variable atom count an exact language contract.
- G2 prioritizes periodic relations inside denoising.
- BTRD teaches that same residual a low-cost model494 basin-transport direction
  when and only when its stability gate passes.

## What each result measures

- `comp_valid`: whether the scientific LLM Planner worked.
- raw Direct/energy/hull: whether the DLM learned crystal realization and
  stability before downstream refinement.
- refined hull and S.U.N.: whether the complete coupled generator works.

The decomposition is diagnostic; the method remains one conditional
probability flow connected by the Plan state.
