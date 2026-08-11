# Official-MP S.U.N. skip-unknown re-evaluation V2

This immutable repair reuses the complete 2,630-fragment spool produced by the
fresh official `GGA_GGA+U` query. It performs no new Materials Project request.
The spool contains 2,550 fully resolved phase-diagram reference sets and 80
contract failures, all with the exact error `missing unary references: ['Yb']`.

The 80 affected chemical systems are not silently labelled unstable. They are
recorded as explicit `hull_unknown`. Reports expose both:

- fixed all-attempt and reconstructed denominators for comparison with earlier
  runs; and
- `skip MP unknown` denominators that remove only those explicit MP-reference
  unknowns.

Generation, reconstruction, novelty, uniqueness, model-494 refinement, CHGNet
relaxation, and relaxed energies remain byte-frozen. No planner, body,
diffusion, energy, training, or RL task is rerun.

V2 covers the 12 V4 all-attempt cells (R03/B3, three repeats, pre/post
model-494) and four frozen historical R03 refined-256 repeats. Paired V4 tests
retain repeat and generation-ordinal pairing and omit a pair only when either
endpoint is an explicit MP `hull_unknown`.
