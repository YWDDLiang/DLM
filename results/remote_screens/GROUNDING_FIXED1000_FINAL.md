# Counterfactual grounding — fixed requested-1000 confirmation

Contribution candidate pass: **False**

| Arm | Body | Direct C/S/J | N/U/N∩U | Hull K/U | Strict stable/SUN | Meta stable/SUN | Strict retention | Meta retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 994/1000 | 880/994/877 | 892/991/890 | 966/28 | 109/89 | 589/487 | 81.65% | 82.68% |
| candidate | 990/1000 | 876/989/874 | 887/987/885 | 962/28 | 106/86 | 569/467 | 81.13% | 82.07% |

## Candidate minus control

- body_rate: `-0.4000%`
- direct_joint_rate: `-0.3000%`
- novel_rate: `-0.1425%`
- unique_rate: `-0.0012%`
- strict_attempt_rate: `-0.3000%`
- meta_attempt_rate: `-2.0000%`
- strict_known_rate: `-0.2735%`
- meta_known_rate: `-1.8694%`
- strict_stable_to_sun_retention: `-0.5193%`
- meta_stable_to_sun_retention: `-0.6087%`

## Frozen criteria

- mechanism: `True`
- strict_direction: `False`
- meta_direction: `False`
- body_noninferiority_1pp: `True`
- direct_joint_noninferiority_1pp: `True`
- novelty_noninferiority_1pp: `True`
- strict_retention_noninferiority_1pp: `True`
- meta_retention_noninferiority_1pp: `True`
- contribution_candidate_pass: `False`

Known-both exact McNemar: Strict candidate-only/control-only `13/15`, p=`0.8506`; Meta `78/95`, p=`0.2237`.

The public 105/1000 Strict and 488/1000 Meta headline remains unchanged and is not redefined by this cohort.
