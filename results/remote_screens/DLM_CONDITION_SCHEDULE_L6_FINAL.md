# DLM conditioning × coordinate schedule L6

Promote hard_joint: **False**
Selected valid axis-condition arm: **full_axis**

| Arm | Requested | Body | Direct J | N/U/NU | Hull K/U | Strict stable/SUN | Meta stable/SUN |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_axis | 512 | 505 | 457 | 451/503/449 | 489/16 | 61/48 | 281/230 |
| full_joint | 512 | 319 | 295 | 286/317/284 | 310/9 | 38/29 | 179/146 |
| hard_axis | 512 | 512 | 463 | 458/510/456 | 496/16 | 61/47 | 282/230 |
| hard_joint | 512 | 417 | 376 | 381/417/381 | 406/11 | 43/35 | 214/181 |

## hard_joint minus full_axis

- parse_rate: `+0.0000%`
- body_rate: `-17.1875%`
- direct_joint_rate: `-15.8203%`
- novel_rate: `+2.0600%`
- unique_rate: `+0.3960%`
- strict_attempt_rate: `-2.5391%`
- meta_attempt_rate: `-9.5703%`
- strict_retention: `+2.7068%`
- meta_retention: `+2.7289%`

## Factorial effects

### conditioning_hard_minus_full

- parse_rate: `+0.0000%`
- body_rate: `+10.2539%`
- direct_joint_rate: `+8.4961%`
- novel_rate: `+0.9290%`
- unique_rate: `+0.3162%`
- strict_attempt_rate: `+0.4883%`
- meta_attempt_rate: `+3.4180%`
- strict_retention: `+1.7201%`
- meta_retention: `+1.3625%`

### schedule_joint_minus_axis

- parse_rate: `+0.0000%`
- body_rate: `-27.4414%`
- direct_joint_rate: `-24.3164%`
- novel_rate: `+1.1310%`
- unique_rate: `+0.0799%`
- strict_attempt_rate: `-3.0273%`
- meta_attempt_rate: `-12.9883%`
- strict_retention: `+0.9867%`
- meta_retention: `+1.3664%`

### interaction_hard_x_joint

- parse_rate: `+0.0000%`
- body_rate: `+17.7734%`
- direct_joint_rate: `+14.6484%`
- novel_rate: `+1.5655%`
- unique_rate: `+0.6215%`
- strict_attempt_rate: `+1.3672%`
- meta_attempt_rate: `+6.8359%`
- strict_retention: `+6.7189%`
- meta_retention: `+3.3054%`


## Frozen gate

- pooled_strict_positive: `False`
- pooled_meta_positive: `False`
- parse_noninferior_1pp: `True`
- body_noninferior_1pp: `False`
- direct_noninferior_1pp: `False`
- novel_noninferior_1pp: `True`
- unique_noninferior_1pp: `True`
- strict_retention_noninferior_1pp: `True`
- meta_retention_noninferior_1pp: `True`
- both_seeds_strict_noninferior_1pp: `False`
- both_seeds_meta_noninferior_1pp: `False`
- eligible: `False`
- promote_hard_joint: `False`
