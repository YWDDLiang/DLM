# Basin-Transport Residual Distillation (BTRD)

Status: frozen stability-primary candidate. PSTR is inactive.

## Scientific target

G2 already improves periodic validity, while model494 provides the dominant
continuous stability transition. BTRD moves a low-cost version of that
transition upstream: the DLM learns the direction from its current geometric
state toward a model494 basin before inference-time refinement.

BTRD does not train the Planner, alter Compact-V2, change `7+4N`, or use CHGNet
as a training label. Planner composition validity is therefore exactly
preserved.

## Frozen train-only teacher

- Select 8,192 MP20-train rows by a fixed content hash; exclude every exact
  composition in the 1200 evaluation ledger.
- Apply one deterministic `p_mask=0.5` geometry corruption per row.
- Use frozen A/G2-PBC-R to obtain the q1 soft lattice and coordinates.
- For 6,144 rows, apply frozen model494 tau200 once to obtain a local basin
  teacher; retain 2,048 original MP20 geometries as a 25% anchor.
- Preserve every teacher failure. A failed tau200 row falls back to its original
  MP20 anchor and remains explicitly labeled; no row is selected by energy,
  displacement, validity or success.
- N, elements and site order never change. Only lattice and fractional
  coordinate positions receive transport supervision.

Tau200 is pre-registered because the matched historical endpoint moves median
hull from `2.1767` to `0.1337 eV/atom` and Meta S.U.N. from `66/512` to
`171/512`, at approximately one quarter of tau800 refinement cost.

## Objective

Let current q1 soft geometry be `(L_hat,f_hat)` and the tau200 teacher be
`(L_plus,f_plus)`, with metric `G=LL^T`. The transport loss is

\[
\mathcal L_{\mathrm{transport}}
=\frac{\|\hat G-G^+\|_F^2}{\|G^+\|_F^2+\epsilon}
+\frac1N\sum_i
\frac{\|\operatorname{MI}(\hat f_i-f_i^+)L^+\|_2^2}
{(V^+/N)^{2/3}+\epsilon}.
\]

The sole loss is

\[
\mathcal L=\mathcal L_{\mathrm{masked\ CE}}(Q_{7+4N}(x^+))
+0.10\mathcal L_{\mathrm{metric}}
+0.10\mathcal L_{\mathrm{RDF}}
+0.20\mathcal L_{\mathrm{overlap}}
+0.05\mathcal L_{\mathrm{coord}}
+0.25\mathcal L_{\mathrm{transport}}.
\]

Existing species-aware mean-overlap remains unchanged. No PSTR or alternate
safety term is added, preserving clean stability attribution.

## Training

- Initialize from promoted A/G2-PBC-R step1696.
- Freeze backbone and Compact-V2 LoRA; train only the existing G2 residual.
- Seed fixed once; 512 updates, effective batch16.
- LR1e-6 cosine, warmup10, sole step512 checkpoint.
- Teacher construction uses at most6 A800; training uses2 A800/16 CPU.
- No grid, checkpoint selection, energy threshold, group winner or online RL.

## Fixed256 promotion

A and BTRD use identical first256 official-known Plans, DLM noise and sampling
parameters. First evaluate raw generation, fast validity, raw CHGNet and
official hull; do not run model494.

All conditions are required:

- paired raw CHGNet and official-hull means are each at most
  `-10 meV/atom`;
- both paired bootstrap 95% CI upper bounds are below zero;
- raw Meta S.U.N. increases by at least5/256;
- raw Strict S.U.N. does not decrease;
- raw fast Direct decreases by at most3/256;
- body/composition-valid decreases by at most5/256 and remains at least95%;
- Planner composition validity changes by exactly zero.

Only a passing checkpoint generates the remaining official-known Plans. The
first256 remains a development block; the untouched remainder is reported as
the confirmation block. Failure stops BTRD before any downstream refiner run.

## Contribution claim on success

BTRD would show that the local transport of a frozen physical basin projector
can be absorbed by a masked crystal DLM through its existing periodic residual,
improving raw thermodynamic stability without increasing inference
multiplicity or exposing energy labels to the language model.
