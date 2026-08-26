# Difficulty-Decomposed Self-Improving Planner V2 — final two-seed screen

Route-B screen useful: **False**

| Seed | Arm | Planner parsed | Body | Refined | Reconstructed | Direct C/S/J | N/U/N∩U | Hull K/U | Strict (attempt; known) | Meta (attempt; known) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | control | 255/256 | 255/256 | 255 | 255 | 227/255/226 | 226/255/226 | 244/11 | 13/256=5.08%; 13/244=5.33% | 113/256=44.14%; 113/244=46.31% |
| 17 | candidate | 255/256 | 254/256 | 254 | 254 | 230/254/229 | 222/254/222 | 245/9 | 19/256=7.42%; 19/245=7.76% | 117/256=45.70%; 117/245=47.76% |
| 18 | control | 256/256 | 254/256 | 254 | 254 | 225/254/223 | 227/253/226 | 250/4 | 24/256=9.38%; 24/250=9.60% | 116/256=45.31%; 116/250=46.40% |
| 18 | candidate | 255/256 | 254/256 | 254 | 254 | 226/254/226 | 224/254/224 | 248/6 | 17/256=6.64%; 17/248=6.85% | 107/256=41.80%; 107/248=43.15% |

## Pooled 512-attempt comparison

| Arm | Planner parsed | Body/refined/reconstructed | Direct C/S/J | N/U/N∩U | Hull K/U | Strict (attempt; known) | Meta (attempt; known) |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 511/512 | 509/509/509 | 452/509/449 | 453/508/452 | 494/15 | 37/512=7.23%; 37/494=7.49% | 229/512=44.73%; 229/494=46.36% |
| candidate | 510/512 | 508/508/508 | 456/508/455 | 446/508/446 | 493/15 | 36/512=7.03%; 36/493=7.30% | 224/512=43.75%; 224/493=45.44% |

## Pooled deltas (candidate minus control)

- planner_parse_rate: `-0.1953%`
- body_rate: `-0.1953%`
- direct_joint_rate: `+1.1719%`
- novel_rate: `-1.2028%`
- unique_rate: `+0.1965%`
- novel_unique_rate: `-1.0063%`
- strict_known_rate: `-0.1876%`
- meta_known_rate: `-0.9202%`
- strict_attempt_rate: `-0.1953%`
- meta_attempt_rate: `-0.9766%`

## Decision criteria

- strict_direction: `False`
- meta_noninferiority_1pp: `True`
- planner_parse_noninferiority_1pp: `True`
- body_noninferiority_1pp: `True`
- direct_joint_noninferiority_1pp: `True`
- novelty_noninferiority_1pp: `False`
- route_b_screen_useful: `False`

## Seed stability and statistics

- seed 17: Strict attempt delta `+2.3438%`; Meta attempt delta `+1.5625%`.
- seed 18: Strict attempt delta `-2.7344%`; Meta attempt delta `-3.5156%`.
- Pooled known-both exact McNemar: Strict candidate-only/control-only=29/29, p=1; Meta candidate-only/control-only=105/108, p=0.8910.

Proposal-mix changes and downstream conversion are reported separately. Because the Planner arms sample different compositions, ordinal pairing is only an end-to-end common-random-number comparison and is not evidence of a fixed-composition realization effect.

The normalized V2 screen is not retained as a positive method result: seed 17 improved Strict/Meta, seed 18 reversed both, and pooled Strict plus novelty were negative despite improved Direct joint validity.

The public 105/1000 Strict and 488/1000 Meta headline remains unchanged pending user confirmation.
