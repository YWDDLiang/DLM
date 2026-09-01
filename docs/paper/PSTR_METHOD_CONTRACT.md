# Periodic Sitewise Tail Risk (PSTR)

Status: superseded before model loading or training. PSTR remains a documented
geometry-risk idea, but the user clarified that the new method must primarily
improve thermodynamic stability and S.U.N. It is not part of the active run.

## Motivation

G0 found that every parsed raw Direct failure was a periodic atomic collision.
The legacy overlap objective averages all `N(N-1)/2` pairs, so one catastrophic
pair contributes only `1/190` when `N=20`. PSTR aligns DLM training with the
sitewise extreme event that actually determines structural validity.

For normalized species-aware penetration

\[
v_{ij}=\left[(m_{ij}-d^{\mathrm{PBC}}_{ij})/m_{ij}\right]_+^2,
\]

PSTR defines

\[
T_i^\tau=\tau\left(\log\sum_{j\ne i}\exp(v_{ij}/\tau)-\log(N-1)\right)
\]

and

\[
L_{\mathrm{PSTR}}=(1-\beta)\,\mathrm{mean}_{i<j}v_{ij}
+\beta\,\mathrm{mean}_iT_i^\tau.
\]

The sole setting is `tau=0.10`, `beta=0.50`. The total overlap weight stays
`0.20`; metric/RDF/coordination remain `0.10/0.10/0.05`. Strict triclinic
125-image PBC and species margins are unchanged. PSTR adds no inference module
or cost.

## Training contract

- Initialize from promoted A / G2-PBC-R step1696.
- Freeze backbone and Compact-V2 LoRA; train only the existing periodic
  relation residual.
- MP20 train-only teacher data, seed81017, 2 A800, effective batch16.
- 256 updates, LR1e-6 cosine, warmup10, sole step256 checkpoint.
- No search, alternate temperature/mix, checkpoint selection or Planner change.

## Gate

After the official-reference-known 1200 Plan queue is frozen, use its first256
rows and identical A/PSTR DLM noise. Evaluate only body accounting and fast
Direct validity.

PSTR expands only when all hold:

- raw fast Direct is at least matched A +6/256;
- generated body/composition-valid loss is at most5/256 and candidate
  composition-valid remains at least95%;
- all Planner inputs are identical, so Planner composition-valid loss is zero;
- no retry, replacement, top-up, reranking or failed-row filtering.

Failure stops PSTR before model494, CHGNet, N/U or hull. Success runs the
remaining known Plans without rerunning the first256.
