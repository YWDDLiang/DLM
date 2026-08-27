# DLM sufficient-training raw1000 result

Selected total epoch: **2**
Strict/Meta absolute target met: **False**

| Epoch | Scope | Body | Direct C/S/J | N/U/NU | Hull K/U | Strict stable/SUN | Meta stable/SUN | Strict retention | Meta retention |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | pooled1000 | 985/1000 | 871/985/871 | 885/982/883 | 958/27 | 102/81 | 587/489 | 79.41% | 83.30% |
| 3 | pooled1000 | 992/1000 | 879/992/878 | 889/987/886 | 965/27 | 100/79 | 578/477 | 79.00% | 82.53% |
| 2 | round1 | 497/500 | 437/497/437 | 446/496/445 | 482/15 | 53/41 | 289/240 | 77.36% | 83.04% |
| 2 | round2 | 488/500 | 434/488/434 | 439/486/438 | 476/12 | 49/40 | 298/249 | 81.63% | 83.56% |
| 3 | round1 | 498/500 | 438/498/437 | 444/497/443 | 483/15 | 45/34 | 282/230 | 75.56% | 81.56% |
| 3 | round2 | 494/500 | 441/494/441 | 445/490/443 | 482/12 | 55/45 | 296/247 | 81.82% | 83.45% |

## Frozen epoch-3 selection gate

- body_noninferior_1pp: `True`
- direct_noninferior_1pp: `True`
- novel_noninferior_1pp: `True`
- unique_noninferior_1pp: `True`
- strict_noninferior: `False`
- meta_noninferior: `False`
- strict_retention_noninferior_1pp: `True`
- meta_retention_noninferior_1pp: `True`
- select_epoch3: `False`

## Absolute targets on selected checkpoint

- strict_at_least_10pct: `False`
- meta_at_least_50pct: `False`
- both_met: `False`

Known-both exact McNemar: Strict epoch3-only/epoch2-only `19/21`, p=`0.8746`; Meta `81/96`, p=`0.2926`.

Round uniqueness uses the pooled-1000 representative definition restricted to each half; pooled1000 is the primary estimand.
