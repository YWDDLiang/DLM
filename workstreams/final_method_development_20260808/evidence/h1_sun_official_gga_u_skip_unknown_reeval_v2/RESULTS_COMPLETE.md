# Official-MP S.U.N. re-evaluation V2 — explicit MP unknowns

This report changes only MP phase-diagram references and `E_hull`. Generation, reconstruction, novelty, uniqueness, model-494 refinement, and CHGNet relaxed energies are byte-frozen inputs.

Thermo contract: reuse only the completed fresh official `MPRester.get_entries_in_chemsys` spool (`compatible_only=True`, explicit `GGA_GGA+U`). No new MP query is issued. The 80 Yb systems whose official response lacks a Yb unary reference are explicit `hull_unknown` and are excluded only from the columns labelled `skip MP unknown`; fixed all-attempt and reconstructed denominators remain visible for comparability.

## Strict S.U.N.

| Panel | Arm | Repeat | Stage | N+U | MP unknown | Old / all | Clean / all fixed | Clean / all skip MP unknown | Old / reconstructed | Clean / reconstructed fixed | Clean / reconstructed skip MP unknown | 0→1 | 1→0 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R03_REFINED256_CURRENT_REPLAY | R03 | 0 | post_model494 | 227 | 9 | 28/256 (10.94%) | 28/256 (10.94%) | 28/247 (11.34%) | 28/248 (11.29%) | 28/248 (11.29%) | 28/239 (11.72%) | 1 | 1 |
| R03_REFINED256_CURRENT_REPLAY | R03 | 1 | post_model494 | 224 | 9 | 31/256 (12.11%) | 32/256 (12.50%) | 32/247 (12.96%) | 31/248 (12.50%) | 32/248 (12.90%) | 32/239 (13.39%) | 2 | 1 |
| R03_REFINED256_CURRENT_REPLAY | R03 | 2 | post_model494 | 226 | 9 | 29/256 (11.33%) | 30/256 (11.72%) | 30/247 (12.15%) | 29/248 (11.69%) | 30/248 (12.10%) | 30/239 (12.55%) | 2 | 1 |
| R03_REFINED256_CURRENT_REPLAY | R03 | 3 | post_model494 | 227 | 9 | 29/256 (11.33%) | 30/256 (11.72%) | 30/247 (12.15%) | 29/248 (11.69%) | 30/248 (12.10%) | 30/239 (12.55%) | 2 | 1 |
| V4_ALL_ATTEMPT_1000 | B3 | 0 | post_model494 | 873 | 26 | 70/1000 (7.00%) | 71/1000 (7.10%) | 71/974 (7.29%) | 70/981 (7.14%) | 71/981 (7.24%) | 71/955 (7.43%) | 3 | 2 |
| V4_ALL_ATTEMPT_1000 | B3 | 0 | pre_model494 | 975 | 27 | 20/1000 (2.00%) | 21/1000 (2.10%) | 21/973 (2.16%) | 20/981 (2.04%) | 21/981 (2.14%) | 21/954 (2.20%) | 1 | 0 |
| V4_ALL_ATTEMPT_1000 | B3 | 1 | post_model494 | 865 | 25 | 58/1000 (5.80%) | 60/1000 (6.00%) | 60/975 (6.15%) | 58/978 (5.93%) | 60/978 (6.13%) | 60/953 (6.30%) | 4 | 2 |
| V4_ALL_ATTEMPT_1000 | B3 | 1 | pre_model494 | 973 | 28 | 11/1000 (1.10%) | 11/1000 (1.10%) | 11/972 (1.13%) | 11/978 (1.12%) | 11/978 (1.12%) | 11/950 (1.16%) | 1 | 1 |
| V4_ALL_ATTEMPT_1000 | B3 | 2 | post_model494 | 884 | 24 | 60/1000 (6.00%) | 61/1000 (6.10%) | 61/976 (6.25%) | 60/983 (6.10%) | 61/983 (6.21%) | 61/959 (6.36%) | 5 | 4 |
| V4_ALL_ATTEMPT_1000 | B3 | 2 | pre_model494 | 977 | 29 | 21/1000 (2.10%) | 24/1000 (2.40%) | 24/971 (2.47%) | 21/983 (2.14%) | 24/983 (2.44%) | 24/954 (2.52%) | 4 | 1 |
| V4_ALL_ATTEMPT_1000 | R03 | 0 | post_model494 | 870 | 24 | 70/1000 (7.00%) | 73/1000 (7.30%) | 73/976 (7.48%) | 70/969 (7.22%) | 73/969 (7.53%) | 73/945 (7.72%) | 4 | 1 |
| V4_ALL_ATTEMPT_1000 | R03 | 0 | pre_model494 | 968 | 28 | 26/1000 (2.60%) | 27/1000 (2.70%) | 27/972 (2.78%) | 26/969 (2.68%) | 27/969 (2.79%) | 27/941 (2.87%) | 2 | 1 |
| V4_ALL_ATTEMPT_1000 | R03 | 1 | post_model494 | 864 | 22 | 65/1000 (6.50%) | 67/1000 (6.70%) | 67/978 (6.85%) | 65/978 (6.65%) | 67/978 (6.85%) | 67/956 (7.01%) | 3 | 1 |
| V4_ALL_ATTEMPT_1000 | R03 | 1 | pre_model494 | 978 | 27 | 22/1000 (2.20%) | 22/1000 (2.20%) | 22/973 (2.26%) | 22/978 (2.25%) | 22/978 (2.25%) | 22/951 (2.31%) | 0 | 0 |
| V4_ALL_ATTEMPT_1000 | R03 | 2 | post_model494 | 875 | 24 | 60/1000 (6.00%) | 60/1000 (6.00%) | 60/976 (6.15%) | 60/977 (6.14%) | 60/977 (6.14%) | 60/953 (6.30%) | 5 | 5 |
| V4_ALL_ATTEMPT_1000 | R03 | 2 | pre_model494 | 976 | 30 | 22/1000 (2.20%) | 22/1000 (2.20%) | 22/970 (2.27%) | 22/977 (2.25%) | 22/977 (2.25%) | 22/947 (2.32%) | 1 | 1 |

## Meta-S.U.N. (≤0.1 eV/atom)

| Panel | Arm | Repeat | Stage | N+U | MP unknown | Old / all | Clean / all fixed | Clean / all skip MP unknown | Old / reconstructed | Clean / reconstructed fixed | Clean / reconstructed skip MP unknown | 0→1 | 1→0 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R03_REFINED256_CURRENT_REPLAY | R03 | 0 | post_model494 | 227 | 9 | 122/256 (47.66%) | 122/256 (47.66%) | 122/247 (49.39%) | 122/248 (49.19%) | 122/248 (49.19%) | 122/239 (51.05%) | 1 | 1 |
| R03_REFINED256_CURRENT_REPLAY | R03 | 1 | post_model494 | 224 | 9 | 123/256 (48.05%) | 125/256 (48.83%) | 125/247 (50.61%) | 123/248 (49.60%) | 125/248 (50.40%) | 125/239 (52.30%) | 2 | 0 |
| R03_REFINED256_CURRENT_REPLAY | R03 | 2 | post_model494 | 226 | 9 | 125/256 (48.83%) | 126/256 (49.22%) | 126/247 (51.01%) | 125/248 (50.40%) | 126/248 (50.81%) | 126/239 (52.72%) | 1 | 0 |
| R03_REFINED256_CURRENT_REPLAY | R03 | 3 | post_model494 | 227 | 9 | 126/256 (49.22%) | 127/256 (49.61%) | 127/247 (51.42%) | 126/248 (50.81%) | 127/248 (51.21%) | 127/239 (53.14%) | 1 | 0 |
| V4_ALL_ATTEMPT_1000 | B3 | 0 | post_model494 | 873 | 26 | 450/1000 (45.00%) | 454/1000 (45.40%) | 454/974 (46.61%) | 450/981 (45.87%) | 454/981 (46.28%) | 454/955 (47.54%) | 4 | 0 |
| V4_ALL_ATTEMPT_1000 | B3 | 0 | pre_model494 | 975 | 27 | 129/1000 (12.90%) | 130/1000 (13.00%) | 130/973 (13.36%) | 129/981 (13.15%) | 130/981 (13.25%) | 130/954 (13.63%) | 2 | 1 |
| V4_ALL_ATTEMPT_1000 | B3 | 1 | post_model494 | 865 | 25 | 444/1000 (44.40%) | 445/1000 (44.50%) | 445/975 (45.64%) | 444/978 (45.40%) | 445/978 (45.50%) | 445/953 (46.69%) | 3 | 2 |
| V4_ALL_ATTEMPT_1000 | B3 | 1 | pre_model494 | 973 | 28 | 118/1000 (11.80%) | 118/1000 (11.80%) | 118/972 (12.14%) | 118/978 (12.07%) | 118/978 (12.07%) | 118/950 (12.42%) | 2 | 2 |
| V4_ALL_ATTEMPT_1000 | B3 | 2 | post_model494 | 884 | 24 | 414/1000 (41.40%) | 414/1000 (41.40%) | 414/976 (42.42%) | 414/983 (42.12%) | 414/983 (42.12%) | 414/959 (43.17%) | 4 | 4 |
| V4_ALL_ATTEMPT_1000 | B3 | 2 | pre_model494 | 977 | 29 | 148/1000 (14.80%) | 147/1000 (14.70%) | 147/971 (15.14%) | 148/983 (15.06%) | 147/983 (14.95%) | 147/954 (15.41%) | 0 | 1 |
| V4_ALL_ATTEMPT_1000 | R03 | 0 | post_model494 | 870 | 24 | 449/1000 (44.90%) | 455/1000 (45.50%) | 455/976 (46.62%) | 449/969 (46.34%) | 455/969 (46.96%) | 455/945 (48.15%) | 6 | 0 |
| V4_ALL_ATTEMPT_1000 | R03 | 0 | pre_model494 | 968 | 28 | 130/1000 (13.00%) | 133/1000 (13.30%) | 133/972 (13.68%) | 130/969 (13.42%) | 133/969 (13.73%) | 133/941 (14.13%) | 3 | 0 |
| V4_ALL_ATTEMPT_1000 | R03 | 1 | post_model494 | 864 | 22 | 435/1000 (43.50%) | 435/1000 (43.50%) | 435/978 (44.48%) | 435/978 (44.48%) | 435/978 (44.48%) | 435/956 (45.50%) | 2 | 2 |
| V4_ALL_ATTEMPT_1000 | R03 | 1 | pre_model494 | 978 | 27 | 122/1000 (12.20%) | 123/1000 (12.30%) | 123/973 (12.64%) | 122/978 (12.47%) | 123/978 (12.58%) | 123/951 (12.93%) | 2 | 1 |
| V4_ALL_ATTEMPT_1000 | R03 | 2 | post_model494 | 875 | 24 | 441/1000 (44.10%) | 444/1000 (44.40%) | 444/976 (45.49%) | 441/977 (45.14%) | 444/977 (45.45%) | 444/953 (46.59%) | 5 | 2 |
| V4_ALL_ATTEMPT_1000 | R03 | 2 | pre_model494 | 976 | 30 | 138/1000 (13.80%) | 139/1000 (13.90%) | 139/970 (14.33%) | 138/977 (14.12%) | 139/977 (14.23%) | 139/947 (14.68%) | 1 | 0 |

## Hull coverage and shift

| Panel | Arm | Repeat | Stage | Old unknown | Clean hull unknown | Relax unknown | Old unknown resolved | Mean ΔE_hull (clean-old) |
|---|---|---:|---|---:|---:|---:|---:|---:|
| R03_REFINED256_CURRENT_REPLAY | R03 | 0 | post_model494 | 9 | 9 | 0 | 0 | -0.000938 |
| R03_REFINED256_CURRENT_REPLAY | R03 | 1 | post_model494 | 9 | 9 | 0 | 0 | -0.000666 |
| R03_REFINED256_CURRENT_REPLAY | R03 | 2 | post_model494 | 9 | 9 | 0 | 0 | -0.000563 |
| R03_REFINED256_CURRENT_REPLAY | R03 | 3 | post_model494 | 9 | 9 | 0 | 0 | -0.000699 |
| V4_ALL_ATTEMPT_1000 | B3 | 0 | post_model494 | 26 | 26 | 0 | 0 | -0.000790 |
| V4_ALL_ATTEMPT_1000 | B3 | 0 | pre_model494 | 29 | 27 | 2 | 0 | -0.000933 |
| V4_ALL_ATTEMPT_1000 | B3 | 1 | post_model494 | 25 | 25 | 0 | 0 | -0.000042 |
| V4_ALL_ATTEMPT_1000 | B3 | 1 | pre_model494 | 28 | 28 | 0 | 0 | 0.000019 |
| V4_ALL_ATTEMPT_1000 | B3 | 2 | post_model494 | 24 | 24 | 0 | 0 | -0.000278 |
| V4_ALL_ATTEMPT_1000 | B3 | 2 | pre_model494 | 29 | 29 | 0 | 0 | -0.000015 |
| V4_ALL_ATTEMPT_1000 | R03 | 0 | post_model494 | 24 | 24 | 0 | 0 | -0.000790 |
| V4_ALL_ATTEMPT_1000 | R03 | 0 | pre_model494 | 29 | 28 | 1 | 0 | -0.000768 |
| V4_ALL_ATTEMPT_1000 | R03 | 1 | post_model494 | 22 | 22 | 0 | 0 | 0.000075 |
| V4_ALL_ATTEMPT_1000 | R03 | 1 | pre_model494 | 28 | 27 | 1 | 0 | -0.000016 |
| V4_ALL_ATTEMPT_1000 | R03 | 2 | post_model494 | 24 | 24 | 0 | 0 | -0.000279 |
| V4_ALL_ATTEMPT_1000 | R03 | 2 | pre_model494 | 31 | 30 | 1 | 0 | -0.000167 |

## Paired V4 inference

All four V4 comparisons retain the frozen repeat/ordinal pairing; a pair is omitted only when either endpoint is explicit MP `hull_unknown`. The terminal JSON records each omitted-pair count, per-repeat exact McNemar tests and 50,000-draw hierarchical paired bootstraps for strict and meta endpoints.

The historical refined-256 panel is descriptive because it has no paired B3 arm in this repair run.
