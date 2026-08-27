# DLM raw-body versus model494 L6 diagnostic

| Source arm | Stage | Requested | Reconstructed | Direct J | N/U/NU | Strict stable/SUN | Meta stable/SUN |
|---|---|---:|---:|---:|---:|---:|---:|
| full_axis | raw | 512 | 505 | 188 | 505/505/505 | 10/10 | 66/66 |
| full_axis | model494 | 512 | 505 | 457 | 451/503/449 | 61/48 | 281/230 |
| hard_axis | raw | 512 | 512 | 187 | 511/512/511 | 2/2 | 8/7 |
| hard_axis | model494 | 512 | 512 | 463 | 458/510/456 | 61/47 | 282/230 |

## full_axis: model494 minus raw

- reconstructed_rate: `+0.0000%`
- direct_joint_rate: `+52.5391%`
- novel_rate: `-10.6931%`
- unique_rate: `-0.3960%`
- strict_attempt_rate: `+7.4219%`
- meta_attempt_rate: `+32.0312%`
- strict_retention: `-21.3115%`
- meta_retention: `-18.1495%`

## hard_axis: model494 minus raw

- reconstructed_rate: `+0.0000%`
- direct_joint_rate: `+53.9062%`
- novel_rate: `-10.3516%`
- unique_rate: `-0.3906%`
- strict_attempt_rate: `+8.7891%`
- meta_attempt_rate: `+43.5547%`
- strict_retention: `-22.9508%`
- meta_retention: `-5.9397%`
