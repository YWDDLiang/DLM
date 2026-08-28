# CCFD-v2 semantic pair-prior audit

Soft prior authorized: **True**

The prior is train-only typed element/valence co-occurrence. It is not BPE and never changes the hard CCFD legality mask.

| Split | Records | Positive pairs | Known nodes | Seen edges | AUC | Positive mean | Negative mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation_all | 8164 | 27076 | 99.98% | 94.46% | 0.6405547414225179 | 0.9293566161283349 | 0.24816209002473968 |
| validation_formula_disjoint | 6995 | 23539 | 99.98% | 93.63% | 0.6316340778991009 | 0.8937226386353895 | 0.2536582242460562 |

## Frozen gates

- train_only_fit: `True`
- validation_known_nodes_at_least_95pct: `True`
- formula_disjoint_pairs_at_least_1000: `True`
- formula_disjoint_auc_above_0_60: `True`
- formula_disjoint_positive_mean_above_negative: `True`
- hard_mask_unchanged: `True`
- soft_pair_prior_authorized: `True`
