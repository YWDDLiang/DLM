# DLM same-Plan stability pair data

Preference training authorized: **False**

- Plans/streams: `256/4`
- Train/validation pairs: `67/22`
- Minimum gap: `0.060 eV/atom`
- Failure reasons: `{'fewer_than_two_eligible_streams': 34, 'gap_below_threshold': 128, 'identical_extreme_body_text': 5}`
- Gap quantiles: `{'q10': 0.06955671310424805, 'q25': 0.08346748352050781, 'q50': 0.10422945022583008, 'q75': 0.1439194679260254, 'q90': 0.2204892158508301}`

Unknown CHGNet energies are missing and never become negatives. Each Plan contributes at most one primary low/high-energy pair; novelty is not a training label.
