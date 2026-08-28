# CTV frozen-feature extraction success archive

This immutable snapshot preserves the successful, reusable engineering stage
that exactly reproduced frozen CTV states and legal action probabilities. It
does **not** archive the failed CTV Q head as a scientific contribution.

Contents:

- `code/`: frozen feature helper and extractor;
- `docs/`: feature/Q and overall CTV contracts;
- `tests/`: focused feature invariants;
- `slurm/`: one-GPU extraction contract;
- `results/`: feature manifest;
- `ARTIFACT_POINTERS.json`: remote artifact path, identity and hash;
- `SHA256SUMS.txt`: hashes for every archived file except itself.

Scientific boundary: the extractor is bit-faithful engineering evidence. The
subsequent single-token value method failed and remains documented separately
in `docs/CTV_DLM_V1_FINAL_NO_GO.md`.
