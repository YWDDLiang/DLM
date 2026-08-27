# model494 intermediate-timestep L6 calibration

Source arm: `full_axis`. Selected tau: **800**.

| tau | Requested | Reconstructed | Direct J | N/U/NU | Strict stable/SUN | Meta stable/SUN | E_hull q50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 512 | 505 | 188 | 505/505/505 | 10/10 | 66/66 | 2.1767 |
| 200 | 512 | 505 | 456 | 488/505/488 | 32/29 | 187/171 | 0.1337 |
| 500 | 512 | 505 | 458 | 466/504/465 | 45/39 | 259/222 | 0.1019 |
| 800 | 512 | 505 | 457 | 451/503/449 | 61/48 | 281/230 | 0.0913 |

## tau0 versus tau800

- strict_positive: `False`
- meta_positive: `False`
- reconstructed_noninferior_1pp: `True`
- direct_noninferior_1pp: `False`
- novel_noninferior_1pp: `True`
- unique_noninferior_1pp: `True`
- strict_retention_noninferior_1pp: `True`
- meta_retention_noninferior_1pp: `True`
- both_seeds_strict_noninferior_1pp: `False`
- both_seeds_meta_noninferior_1pp: `False`
- eligible: `False`

## tau200 versus tau800

- strict_positive: `False`
- meta_positive: `False`
- reconstructed_noninferior_1pp: `True`
- direct_noninferior_1pp: `True`
- novel_noninferior_1pp: `True`
- unique_noninferior_1pp: `True`
- strict_retention_noninferior_1pp: `True`
- meta_retention_noninferior_1pp: `True`
- both_seeds_strict_noninferior_1pp: `False`
- both_seeds_meta_noninferior_1pp: `False`
- eligible: `False`

## tau500 versus tau800

- strict_positive: `False`
- meta_positive: `False`
- reconstructed_noninferior_1pp: `True`
- direct_noninferior_1pp: `True`
- novel_noninferior_1pp: `True`
- unique_noninferior_1pp: `True`
- strict_retention_noninferior_1pp: `True`
- meta_retention_noninferior_1pp: `True`
- both_seeds_strict_noninferior_1pp: `False`
- both_seeds_meta_noninferior_1pp: `False`
- eligible: `False`
