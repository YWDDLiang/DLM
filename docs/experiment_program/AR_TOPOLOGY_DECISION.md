# CrysLLMGen-Derived AR Proposal After the Day-7 DLM Decision

Date: 2026-07-20

## Decision

The active method is an extension of the official CrysLLMGen code path.  It no
longer initializes discrete topology from a global all-`MASK` state.  As in
CrysLLMGen, the AR model proposes a coarse crystal before continuous
refinement; the proposal is expressed on the Wyckoff quotient as:

```text
BOS -> space group -> lattice chart -> orbit/type/species/free-coordinate
tuples -> STOP
```

The emitted space group is fixed after the first token.  Orbit tuples contain
Wyckoff type, species, and the orbit's 0--3 dimensional free coordinate; their
serialized order is randomized during training and is canonicalized only for
storage, hashing, and evaluation.

The Llama proposal is expanded through the Wyckoff affine maps and injected at
a registered intermediate timestep into the CrysLLMGen CSPDiffusion sampler.
Continuous geometry therefore begins from the Llama proposal, not fresh random
noise.  The exact atom-wise, topology-frozen CrysLLMGen path is preserved as an
upstream reproduction and matched baseline.

## Revision semantics

Geometry-adaptive feedback remains active after initialization.  The current
state is edited through an explicit event policy:

- `birth(target Wyckoff type, species)`;
- `death(orbit id)`;
- `type_change(orbit id, target type)`, implemented as death plus birth;
- `species_change(orbit id, target species)`;
- `no_op`.

The event and payload heads condition on the complete current topology and the
current continuous geometry evidence.  A successful edit changes the target
stratum; the target-stratum bridge initializes any newly introduced continuous
chart before continuous denoising resumes.

## Role of MASK

`MASK` is no longer a de novo initial state or a semantic NULL canvas.  It may
remain only as an implementation-local training corruption or conditional
infilling marker.  Direct event-based revision is the paper semantics, so a
committed field does not need to pass through a globally masked generation
state before it can be changed.

## Claim change

The method is not described as two diffusion processes.  Its accurate scope is
a CrysLLMGen-derived autoregressive coarse proposer coupled to its continuous
diffusion refiner through geometry-triggered, dimension-changing Wyckoff edits.
The candidate paper contribution remains the stratified Wyckoff quotient,
target-stratum bridges, and closed-loop geometry-to-topology revision, not the
choice of AR or CSPDiffusion itself.

Frozen protocol v3 and its Day-7 artifacts are retained unchanged for
provenance.  Any executable protocol implementing this decision must receive a
new version and new hashes.
