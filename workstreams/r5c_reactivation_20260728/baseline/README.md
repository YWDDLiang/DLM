# Legacy R5-C Crystal DLM

This directory is the frozen, non-Wyckoff snapshot of the earlier crystal
diffusion-language-model program.  It is preserved because the R5-C pipeline
produced valid structures and reproducible diagnostic results, even though it
is no longer the discrete engine of the active Wyckoff paper.

## Scientific status

- The canonical method is the R5-C exact-length/dynamic LLaDA crystal body
  generator, optionally followed by the historical CrysLLMGen refiner.
- Its results remain valid under their recorded legacy evaluation contracts.
- They are **not** interchangeable with the new attempt-level, multi-MLIP
  Wyckoff protocol and must not be used as headline results for that paper.
- No claim is made here that DLM is universally better than autoregressive or
  discrete-diffusion generators.

## Contents

- `crystal_dlm/`: snapshot of the 28 pre-Wyckoff Python modules.
- `scripts/`: snapshot of the 98 legacy data, training, sampling, refinement,
  metric, and analysis entry points.
- `tests/`: 29 non-Wyckoff unit-test files.
- `launchers/pre_wyckoff/`: 95 archived launch and analysis scripts.
- `evidence/reports/`: canonical R5/R5-C reports and full metric artifacts.
- `evidence/runs/`: the preserved R5-C full-1000 and A100 evaluator-sensitivity
  run evidence.

The larger 768 MB superseded experiment archive remains at
`../archive/20260710_pre_wyckoff/` and is indexed rather than duplicated here.
Datasets, third-party references, and model weights are intentionally not
copied.

## External assets

Historical scripts may require explicit paths to:

- `../data/dlm_sft/`;
- `../reference/crysllmgen/`;
- `../reference/LLaDA/`;
- `/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/` on the server.

The server environment used by the project is `diff_meets_diff`.  Do not
download or silently substitute a missing model checkpoint.

## Local verification

Run legacy tests from this directory so the frozen package is imported instead
of the active workspace copy:

```bash
cd legacy_dlm_r5c
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py'
```

Some tests require optional reference repositories or large-model libraries;
their absence should be reported rather than repaired by changing the frozen
method.

## Provenance anchors

- `crystal_dlm/r5_plan_body.py` SHA256:
  `3478ddf657873ea055e5816c423ce36be5ecf0cd1a73c6ee1e5514648047be83`
- `scripts/sample_llada_r5c_plan_body.py` SHA256:
  `5239a9a9ef9c078911a03ddb4791a217433ad62cf524caba99d0b6cc12c913b0`
- `evidence/reports/20260531_r5c_full_pipeline_reproducibility_and_technical_evolution.md`
  SHA256:
  `a5cc3ff4459eada0cabed19199a2b3a7464887ba87af6a68d8490883ae5e61ad`

The root-level legacy files are retained temporarily as compatibility copies.
The active Wyckoff source bundle already excludes them.  Remove the duplicates
only after all frozen legacy tests and entry-point imports have been verified.
