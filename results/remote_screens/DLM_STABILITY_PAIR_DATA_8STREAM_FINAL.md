# DLM same-Plan stability pair data

Preference training authorized: **False**

- Plans/streams: `256/8`
- Train/validation pairs: `95/27`
- Minimum gap: `0.060 eV/atom`
- Failure reasons: `{'fewer_than_two_eligible_streams': 34, 'gap_below_threshold': 97, 'identical_extreme_body_text': 3}`
- Gap quantiles: `{'q10': 0.07019004821777343, 'q25': 0.08535027503967285, 'q50': 0.1184701919555664, 'q75': 0.16494059562683105, 'q90': 0.25134882926940927}`

Unknown CHGNet energies are missing and never become negatives. Each Plan contributes at most one primary low/high-energy pair; novelty is not a training label.
