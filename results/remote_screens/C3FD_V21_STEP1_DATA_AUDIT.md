# C³FD-v2.1 Step-1 data audit

Step 1 pass: **True**

| Split | Rows | Proposal exact | Ledger exact | Reachable weighted mass |
|---|---:|---:|---:|---:|
| train | 27136 | 100.00% | 100.00% | 99.76% |
| val | 9047 | 100.00% | 100.00% | 99.52% |
| test | 9046 | 100.00% | 100.00% | 99.58% |

## Gates

- train_val_present: `True`
- all_rows_have_exact_proposal_labels: `True`
- all_rows_have_exact_or_inapplicable_ledger: `True`
- train_reachable_weighted_mass_at_least_99pct: `True`
- val_reachable_weighted_mass_at_least_99pct: `True`
- step1_pass: `True`
