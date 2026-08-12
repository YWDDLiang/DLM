# Archived H1-A2/R03 first256 official E_hull completion V1

This evaluation-only bundle reuses the terminal 256-attempt ledgers from the
archived H1-A2 D1 versus R03 D2 safe-axis reproduction. It does not rerun the
planner, body, model-494 refinement, CHGNet relaxation, Direct metrics,
novelty, or uniqueness.

The union of both arms contains 221 novel-unique chemical systems. The frozen
clean-official cache already resolves 211. Exactly ten systems are queried
again through official `MPRester.get_entries_in_chemsys()`: nine previously
unresolved Yb systems and one genuinely new system. The primary contract is
`compatible_only=True` with `GGA_GGA+U`. Any system that still lacks a complete
official unary reference set remains explicit `hull_unknown`; it is never
silently classified as unstable.

The MP credential is accepted only through a mode-0600 one-time carrier. The
carrier is destroyed before the first HTTP request, and the credential is not
serialized into source, logs, Slurm state, results, or evidence.
