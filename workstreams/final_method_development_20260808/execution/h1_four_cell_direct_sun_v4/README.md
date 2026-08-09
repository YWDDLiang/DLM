# H1 current-run four-cell Direct/S.U.N. V4

V4 is a new immutable, user-authorized revision after V3 stopped before any
external query or `sbatch`. V3 correctly froze the byte-exact base cache but
compared its 11,661 physical nonempty rows with the 11,163 distinct chemical
systems returned by the frozen R03F loader. V4 freezes and validates those as
separate identities. The cache SHA and bytes are unchanged, as are the frozen
455-system wanted union and 257-system query gap. V4 does not change a Planner
raw row, model, checkpoint, ledger, seed, body schedule, refiner step, Direct
or S.U.N. scoring algorithm.

The four current-run cells are `M00=P0+B0`, `M10=SFT-v2+B0`, `M01=P0+B3`,
and `M11=SFT-v2+B3`. Each has 256 all-attempt ordinals, the same body and
refiner seed ledger, D2 safe-axis decoding, model_494 refine800, Direct, and
S.U.N. Historical summaries cannot replace a current-run cell.

Before submission, the package derives the 455-system union directly from the
byte-frozen P0 and SFT-v2 raw256 files. The frozen base cache covers 198;
exactly the remaining 257 systems are queried using the previous successful
R03F completion code. The credential is consumed through a one-time carrier
and deleted before `sbatch`; it is not serialized into source, results, or the
Slurm environment. Slurm jobs are offline and use one completed cache SHA
shared by all four cells.

The cell array is `0-3%2` on `gpu`. The dependent normal-CPU assembly uses
`afterany`, so any incomplete cell becomes explicit fail-closed terminal
evidence. There is no sample retry, replacement, repair, filter, rerank,
training, automatic downstream work, or RL.
