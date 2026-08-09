# H1 current-run four-cell Direct/S.U.N.

V2 is a new immutable engineering revision authorized after V1 failed before
`sbatch`: shell and SBatch bytes are normalized to LF. No scientific input,
model, ledger, seed, sampling, refinement, evaluator, or gate is changed.

This immutable package evaluates the complete Planner x Body matrix requested
by the user. It reuses the successful R03 safe-axis and refinement/evaluation
shell while replacing neither missing attempts nor failed stages.

The four current-run cells are `M00=P0+B0`, `M10=SFT-v2+B0`, `M01=P0+B3`,
and `M11=SFT-v2+B3`. Each has 256 raw-denominator ordinals, the same body and
refiner seed ledger, D2 safe-axis body decoding, model_494 refine800, Direct,
and frozen-cache S.U.N. Historical summaries are context only and cannot
stand in for a cell.

The cell array is `0-3%2` on the `gpu` partition. The dependent normal-CPU
assembly uses `afterany` so an incomplete cell becomes explicit fail-closed
terminal evidence rather than an indefinitely pending report. Nothing in the
package trains a model, selects a checkpoint, launches downstream work, or
enters RL.
