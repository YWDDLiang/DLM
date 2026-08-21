# Evaluation adapter status

The historical evaluator wrappers contained cluster-specific environment and
source-hash gates, so they are intentionally not copied into the public
runtime. The upstream metric implementation is available under
`src/vendor/crysllmgen/`.

A relative-path Direct/N/U/CHGNet/S.U.N. adapter will be finalized after the
A800 environment and official-MP runtime are copied. Until then,
`slurm/60_evaluate.sbatch` stops with a readable placeholder message and
preserves all upstream outputs.

