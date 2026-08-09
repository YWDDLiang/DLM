# H1 current-run four-cell Direct/S.U.N. V3

V3 is a new immutable, user-authorized revision after V1 and V2 both stopped
before `sbatch`. It corrects V2's single B3 terminal-report path, proves that
the byte-frozen successful R03 Direct scorer reduces composition counts by GCD
before `comp_valid`, and completes exactly the frozen Planner-union Materials
Project cache gap before Slurm. It does not change a Planner raw row, model,
checkpoint, ledger, seed, body schedule, refiner step, or scoring algorithm.

The four current-run cells are `M00=P0+B0`, `M10=SFT-v2+B0`, `M01=P0+B3`,
and `M11=SFT-v2+B3`. Each has 256 all-attempt ordinals, the same body and
refiner seed ledger, D2 safe-axis decoding, model_494 refine800, Direct, and
S.U.N. Historical summaries cannot replace a current-run cell.

Before submission, the package derives the 455-system union directly from the
byte-frozen P0 and SFT-v2 raw256 files. The frozen base cache covers 198;
exactly the remaining 257 systems are queried using the previous successful
R03F completion code. The credential arrives only through a private one-time
file, is consumed and deleted before `sbatch`, and is never serialized. Slurm
jobs are offline and use one completed cache SHA shared by all four cells.

The cell array is `0-3%2` on `gpu`. The dependent normal-CPU assembly uses
`afterany`, so any incomplete cell becomes explicit fail-closed terminal
evidence. There is no sample retry, replacement, repair, filter, rerank,
training, automatic downstream work, or RL.
