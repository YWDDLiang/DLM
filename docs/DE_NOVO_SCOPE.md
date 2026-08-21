# Fully de novo scope and Plan sources

## Main task

H1-A2 treats fully de novo crystal generation as a three-level generative
model:

```text
P ~ p_phi(P)              learned global Plan prior
B ~ p_theta(B | P)       typed discrete crystal realization
M ~ p_psi(M | B, P)      continuous periodic refinement
```

Equivalently,

```text
p(M) = sum_P p_phi(P) sum_B p_theta(B | P) p_psi(M | B, P).
```

The Plan contains composition, atom count, and coarse structural intent, but
does not contain the exact lattice, site realization, or fractional
coordinates. Its role is to generate what chemical and structural region to
explore while leaving multiple body realizations possible.

## Training versus inference

During training, an MP-20 crystal may be deterministically mapped to a Plan
label. This provides supervision for both the learned Plan prior and the
conditional body model; it does not make inference conditional on that row.

During fully de novo inference, the Plan is sampled from the trained Planner.
The body and refiner then operate only on the sampled Plan and their own random
states.

## Plan-source taxonomy

| Plan source | Purpose | Fully de novo at Plan level? |
|---|---|---:|
| learned Planner sample | main H1-A2 route | yes |
| MP-20-derived or empirical Plan replay | downstream control | no |
| frozen Plan bundle | deterministic reproduction of body/refiner | no |
| user-provided Plan | conditional material generation | no |

An empirical or frozen Plan can still lead to a structurally novel crystal,
but it does not test whether the model can generate the global composition,
cardinality, and coarse mode. It must therefore not be used as the sole source
for a fully de novo headline.

## Why the Planner is structurally necessary

The exact-cardinality DLM constructs a state of length `7 + 4N`. It therefore
needs `N` before body generation begins, and composition must also be available
before species anchors are instantiated. A learned global prior closes this
unconditional loop without returning to a fixed maximum canvas or replaying a
dataset condition.

The paper does not claim that the particular Planner backbone is independently
novel. The contribution is the hierarchical factorization and the semantics of
the generated Plan: it is global enough to define the body state, but
underdetermined enough to admit multiple compatible structures.

## Required interpretation of controls

Plan-replay experiments isolate `p_theta(B | P)` and `p_psi(M | B, P)`. They
answer how reliably a fixed specification can be realized and refined. The
full H1-A2 experiment additionally evaluates `p_phi(P)` and answers the fully
de novo question.

If replayed Plans outperform learned Plans downstream, the correct conclusion
is that global Plan generation is the bottleneck—not that replay should replace
the main fully de novo route.

## Architectural alternatives retained for the paper decision

“Without a Planner model” can refer to two scientifically different routes and
must not be used as a single label.

### A. Separate learned Planner

```text
P ~ p_phi(P)
B ~ p_theta(B | P)
M ~ p_psi(M | B, P)
```

This is the current fully de novo H1-A2 route. It gives global chemistry and
cardinality an explicit learned prior, and lets `N` instantiate the exact body
length before masked completion.

### B. No separate Planner, but self-planning generation

A single model could first generate a minimal header `(composition, N, coarse
mode)` and then instantiate and complete the `7 + 4N` body in a second pass.
This can remain fully de novo because the global variables are still generated,
but it removes the independent Planner only at the model level—not at the
functional level. It requires a different training and inference design and is
not claimed as the current H1-A2 implementation.

### C. No learned Planner and empirical Plan replay

MP-20-derived, frozen, or user-provided Plans can be fed directly to the body.
This is the simplest way to isolate the Crystal DLM and refiner, but it is a
conditional realization task rather than fully de novo Plan generation.

| Route | Separate Planner model | Generates global Plan | Fully de novo | Current role |
|---|---:|---:|---:|---|
| A. learned Planner | yes | yes | yes | current main route |
| B. self-planning DLM | no | yes | yes | alternative architecture |
| C. empirical/frozen Plan | no | no | no at Plan level | downstream control |

The paper may ultimately present both A and C, with C isolating downstream
realization, or present only A as the main method. Route B should be discussed
only if implemented and evaluated; otherwise it is a future architectural
alternative rather than an experimental result.
