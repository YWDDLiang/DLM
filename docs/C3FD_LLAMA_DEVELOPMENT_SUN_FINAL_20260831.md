# C3FD–Llama development S.U.N. final

Date: 2026-08-31

Scope: development only. Faithful H0/R0S are historical-interface diagnostics;
compact V2 uses MP20 train/validation compositions and is not prospective.

## Official query

- allowed cells: faithful H0/R0S and compact-V2 seeds82017/82018, raw+refined,
  streams17/18; 16 cells total;
- frozen unique chemsys: `739`;
- resolved/unresolved: `723/16`; unknowns remain missing and are never mapped
  to unstable;
- transport retries: `0`;
- database version: `2026.04.13`;
- official input SOURCE SHA-256:
  `f738f7b225f7d9813a02390d334841c2bbbc2cd4ad227e2ade3ec78a019d25d7`;
- completion manifest SHA-256:
  `e934889e8349722beada5aa43ecf94b283fb91ec921654f7ba81710d37f7c1a7`;
- final JSON SHA-256:
  `84ea1984d2ccd56a76b271b06f6dc7eb212d2c0670869fa49950982ab2bc1219`.

Excluded: train-only alignment pool38881, malformed canary38420,
cancelled deterministic-completion job38914, and already-official D3PO.

## Stream-aggregated S.U.N.

| Group | Stage | Arm | Strict S.U.N. | Meta S.U.N. | Strict s17/s18 | Meta s17/s18 |
|---|---|---|---:|---:|---:|---:|
| faithful | raw | H0 | 2.930% | 12.500% | 11/4 | 38/26 |
| faithful | raw | R0S | 2.930% | 13.672% | 8/7 | 33/37 |
| faithful | refined | H0 | 8.789% | 41.992% | 24/21 | 107/108 |
| faithful | refined | R0S | 7.031% | 42.188% | 20/16 | 108/108 |
| compact V2 | raw | seed82017 | 1.562% | 11.719% | 5/3 | 32/28 |
| compact V2 | raw | seed82018 | 1.367% | 12.305% | 5/2 | 32/31 |
| compact V2 | refined | seed82017 | 8.203% | 54.883% | 24/18 | 141/140 |
| compact V2 | refined | seed82018 | 8.203% | 55.273% | 21/21 | 138/145 |

## Interpretation

- Canonical full-schema repair did not restore H1-A2: R0S lowers refined
  Strict S.U.N. relative to H0 and leaves Meta near 42%.
- Compact V2 plus fresh SFT exceeds the 50% Meta reference in this overlapping
  development cohort but remains below 10% Strict; it cannot support a
  prospective or generalization claim.
- Raw S.U.N. is low for every route. The raw/refined gap confirms that the
  frozen refiner remains a dominant contributor, so raw DLM realization is the
  primary mechanism endpoint for the new F/M routes.
- Seeds82017/82018 are both disclosed and neither is selected.

Remote immutable output:
`$ROOT/runs/c3fd_llama_development_sun_final_20260831_v1`.
