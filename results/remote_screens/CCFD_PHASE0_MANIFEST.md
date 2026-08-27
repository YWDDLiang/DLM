# CCFD Phase 0 manifest

Phase 1 authorized: **True**

| Dataset | Plans | Assignment | CCFD terminal | Round-trip | Legacy comp-valid | Legacy false reject among CCFD | Mixed valence |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 27136 | 96.74% | 96.74% | 100.00% | 90.50% | 8.54% | 5.70% |
| val | 9047 | 96.46% | 96.46% | 100.00% | 90.24% | 8.57% | 5.76% |
| test | 9046 | 96.76% | 96.76% | 100.00% | 90.95% | 8.25% | 5.52% |
| raw1000 | 1000 | 94.10% | 94.10% | 100.00% | 88.50% | 9.67% | 10.80% |

## Frozen gates

- train_assignment_at_least_95pct: `True`
- val_assignment_at_least_95pct: `True`
- train_roundtrip_100pct: `True`
- val_roundtrip_100pct: `True`
- raw_within_3pp_of_train: `True`
- legacy_false_rejection_audited: `True`
- frozen_legacy_rates_reproduced: `True`
- phase1_authorized: `True`
