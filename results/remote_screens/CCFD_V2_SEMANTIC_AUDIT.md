# CCFD-v2 semantic compiler audit

Semantic compiler authorized: **True**

| Dataset | Plans | Legacy valid | Semantic compile | N/charge | Composition RT | Rich RT | Benchmark cert | Extended-only | Cert agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 27136 | 90.50% | 96.84% | 100.00%/100.00% | 100.00% | 100.00% | 90.50% | 6.34% | 100.00% |
| val | 9047 | 90.24% | 96.84% | 100.00%/100.00% | 100.00% | 100.00% | 90.24% | 6.60% | 100.00% |
| test | 9046 | 90.95% | 97.02% | 100.00%/100.00% | 100.00% | 100.00% | 90.95% | 6.07% | 100.00% |
| raw1000 | 1000 | 88.50% | 96.80% | 100.00%/100.00% | 100.00% | 100.00% | 88.50% | 8.30% | 100.00% |

## Frozen gates

- train_val_present: `True`
- train_val_exact_invariants: `True`
- dual_certificate_alignment: `True`
- semantic_coverage_at_least_legacy: `True`
- raw1000_within_3pp_of_train: `True`
- semantic_compiler_authorized: `True`
- planner_gpu_training_authorized: `False`

`extended_only` rows remain unknown and cannot count as comp-valid. This audit does not authorize Planner GPU training by itself.
