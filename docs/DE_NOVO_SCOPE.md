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
