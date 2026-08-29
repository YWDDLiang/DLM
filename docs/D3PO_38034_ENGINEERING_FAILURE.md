# D3PO generation job 38034 engineering failure

Date: 2026-08-30

## Terminal status

- Slurm job `38034`, six A800 / 48 CPUs.
- All six cells stopped in Plan validation before model sampling, body output,
  proposal graphs, model494 refinement, or any scientific result.
- The immutable run contains `_FAILED` and `ENGINEERING_FAILURE.tsv`; no cell
  `_SUCCESS` and no run `_SUCCESS` exists.

## Root cause

The frozen test selection correctly stored `d3po_test_ordinal=0..255`, but its
generation-facing certified rows retained the original C³FD seed17
`sample_idx` values (the first selected row was source index `265`). The shared
SGTC sampler requires execution `sample_idx` to be exactly global ordinals
`0..255`, while preserving the original ledger index separately.

This is a cohort serialization/interface error. It does not change which 256
Plans were selected, their order, composition, prompt, training data, DLM
checkpoint, random seeds, denominator, or scientific contract.

## Minimal recovery prepared, not automatically submitted

For every already selected row:

```text
source_sample_idx = original seed17 sample_idx
sample_idx = d3po_test_ordinal = 0..255
```

Regenerate and hash only the certified execution view, rerun local/remote tests,
and update the wrapper hash. Preserve job38034 permanently. Per the automation
contract, a recovery submission requires notification rather than an automatic
retry.
