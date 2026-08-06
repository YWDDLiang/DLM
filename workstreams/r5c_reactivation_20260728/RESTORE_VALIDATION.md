# R5-C Restore Validation

```text
baseline_archive: r5c_frozen_baseline_20260728.tar.gz
bytes: 4528007
sha256: ad1b7f5b9ee0df0c06396ef1d3865f7a5e7b2e4d3f4b46216445288e04be8325
reactivation_bundle: r5c_reactivation_bundle_20260728.tar.gz
bundle_bytes: 4530138
bundle_sha256: 63f699f670ab8c450e7e196ec824f7009ea5a4e9f6e7aee7f743d81f51d25d1b
bundle_gzip_test: PASS
bundle_tar_listing: PASS
source_regular_files_excluding_bytecode: 330
restored_regular_files_excluding_bytecode: 330
recursive_diff_excluding_bytecode: PASS
core_tests: 47/47 PASS
```

Restored key SHA256 anchors:

```text
3478ddf657873ea055e5816c423ce36be5ecf0cd1a73c6ee1e5514648047be83  crystal_dlm/r5_plan_body.py
5239a9a9ef9c078911a03ddb4791a217433ad62cf524caba99d0b6cc12c913b0  scripts/sample_llada_r5c_plan_body.py
a5cc3ff4459eada0cabed19199a2b3a7464887ba87af6a68d8490883ae5e61ad  evidence/reports/20260531_r5c_full_pipeline_reproducibility_and_technical_evolution.md
```

Remote read-only preflight also confirmed that the historical R5-C checkpoint,
full-1000 run, A100/CHGNet evaluation evidence, exact-length test JSONL,
CrysLLMGen refiner checkpoint, LLaDA model, evaluator scripts, and
`diff_meets_diff` environment remain present on A800.

The same preflight recovered the completed H1-A4 terminal evidence.  Both
epochs ended successfully; epoch 2 passed its hybrid target gate but remained
below the conditional R5-C S.U.N. anchor.  The restored experiment index records
the values and prevents H1-A4 from being mistaken for unfinished work.
