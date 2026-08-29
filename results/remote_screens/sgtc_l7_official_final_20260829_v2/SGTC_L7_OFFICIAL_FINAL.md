# SGTC-DLM-v1 official L7

SGTC L7 pass: **False**

| Arm | Requested | Body | Direct J | N/U/NU | Hull K/U | Strict stable/SUN | Meta stable/SUN |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1000 | 998 | 996 | 922/995/922 | 979/19 | 81/60 | 486/412 |
| g0_all | 1000 | 1000 | 997 | 933/999/933 | 981/19 | 78/55 | 486/421 |
| g1_strict | 1000 | 1000 | 996 | 930/998/930 | 981/19 | 73/53 | 485/417 |

## Official e_hull distribution (all known reconstructed attempts)

| Arm | Known | q10 | q50 | q90 | <=0.01 | <=0.05 | <=0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 979 | 0.0082 | 0.1014 | 0.3417 | 109 | 269 | 486 |
| g0_all | 981 | 0.0102 | 0.1012 | 0.3548 | 98 | 261 | 486 |
| g1_strict | 981 | 0.0092 | 0.1016 | 0.3476 | 102 | 245 | 485 |

## Matched continuous effects

Positive candidate-minus-control energy is adverse; lower fractions above 0.5 are favorable.

| Pair | Metric | N | Mean delta | Median delta | Fraction lower | Normal 95% CI |
|---|---|---:|---:|---:|---:|---:|
| g0_all-base | official_e_above_hull | 979 | 0.003708 | 0.000000 | 0.4831 | [-0.000831, 0.008248] |
| g0_all-base | chgnet_energy_per_atom_refined | 998 | 0.003883 | 0.000000 | 0.4820 | [-0.000587, 0.008354] |
| g1_strict-base | official_e_above_hull | 979 | 0.004525 | 0.000000 | 0.4801 | [-0.000339, 0.009390] |
| g1_strict-base | chgnet_energy_per_atom_refined | 998 | 0.004238 | 0.000000 | 0.4850 | [-0.000553, 0.009028] |
| g1_strict-g0_all | official_e_above_hull | 981 | 0.000848 | 0.000000 | 0.4771 | [-0.004175, 0.005870] |
| g1_strict-g0_all | chgnet_energy_per_atom_refined | 1000 | 0.000385 | 0.000000 | 0.4810 | [-0.004572, 0.005343] |

## Exact paired binary effects

| Pair | Metric | Known both | Delta | Wald 95% CI | McNemar p |
|---|---|---:|---:|---:|---:|
| g0_all-base | strict_sun | 979 | -0.5107% | [-1.4708%, +0.4494%] | 0.404873 |
| g0_all-base | meta_sun | 979 | +0.8172% | [-1.5182%, +3.1525%] | 0.548502 |
| g1_strict-base | strict_sun | 979 | -0.7150% | [-1.7927%, +0.3627%] | 0.264931 |
| g1_strict-base | meta_sun | 979 | +0.4086% | [-1.8397%, +2.6568%] | 0.789396 |
| g1_strict-g0_all | strict_sun | 981 | -0.2039% | [-1.0519%, +0.6441%] | 0.814529 |
| g1_strict-g0_all | meta_sun | 981 | -0.4077% | [-2.7558%, +1.9403%] | 0.798545 |

## Seen/unseen support relative to G1 strict training chemsys

| Arm | Seen reconstructed/known | Seen Strict/Meta SUN | Unseen reconstructed/known | Unseen Strict/Meta SUN |
|---|---:|---:|---:|---:|
| base | 319/308 | 18/158 | 679/671 | 42/254 |
| g0_all | 320/309 | 18/158 | 680/672 | 37/263 |
| g1_strict | 320/309 | 16/167 | 680/672 | 37/250 |

The gate uses the requested-1000 denominator. Paired intervals and exact McNemar results are in the JSON artifact.
The frozen L7 contract evaluated only reconstructed/refined cells. Raw CHGNet and raw official hull are unavailable and are not inferred post hoc.
The existing public 105/488 headline is unchanged by this internal confirmation.
