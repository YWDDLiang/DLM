# R03 raw Plan × B3 repeated Direct/S.U.N. evaluation — V4

V3 failed closed before submitting Slurm work because the reused four-cell
validator requires a Planner-by-Body matrix, whereas this experiment is an
intentional four-repeat P0+B3 panel. V4 copies the five successful V4 runtime
scripts byte-for-byte and binds them to a strict repeat-panel protocol shim.
Scientific inputs, model bytes, cache, repeats, seeds, scorers, and statistics
are unchanged.

This immutable execution package answers one narrow question: what happens when
the frozen R03 raw planner cohort is kept fixed and the frozen B0 body is
replaced by B3?

The registered candidate consists of four independent A800 process repeats.
Every repeat uses the first 256 records of the byte-frozen R03 P0 raw Plan,
the common H1 body/refiner seed ledger, D2 safe-axis body generation,
CrysLLMGen `model_494` with 800 refinement steps, the byte-frozen R03 Direct
scorer (GCD before SMACT composition validity), and the exact 227-system R03F
Materials Project snapshot. No MP network query is made.

The control is not rerun. Each new B3 repeat is paired by repeat and ordinal
with the corresponding completed historical R03G B0 safe-axis artifact.
Pooled 1024-attempt counts are descriptive only. Inferential reporting uses
per-repeat exact McNemar tests and a hierarchical paired bootstrap that
resamples repeat blocks and then ordinals within repeats.

There is no sample retry, replacement, repair, filter, rerank, checkpoint
selection, training, downstream automation, or RL. Any engineering failure is
preserved fail-closed and is not automatically retried.
