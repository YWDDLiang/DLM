# Archived H1-A2 vs R03 first256 — official E_hull completion

Only stability was recomputed. Generation, model-494 refine800, Direct, U, N, and CHGNet relaxed energies are byte-frozen.

- Official query: `mp_api.client.MPRester.get_entries_in_chemsys`, `compatible_only=True`, `GGA_GGA+U`.
- Chemical systems: 221 total; 211 reused resolved; 10 freshly queried; 9 remain explicit hull-unknown.

| Arm | Generated | Joint valid | Novel+unique | Hull evaluated | Hull unknown | Strict S.U.N. | Meta-S.U.N. | Historical expected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original_H1A2_DLM | 246/256 | 212/256 | 223/256 | 214 | 9 | 24/256 (9.38%) | 118/256 (46.09%) | 27 strict / 133 meta |
| R03_safe_axis_DLM | 248/256 | 213/256 | 224/256 | 215 | 9 | 28/256 (10.94%) | 128/256 (50.00%) | 28 strict / 122 meta |

## Exact paired McNemar

| Endpoint | H1-A2 only | R03 only | Discordant | Two-sided exact p |
|---|---:|---:|---:|---:|
| strict S.U.N. | 2 | 6 | 8 | 0.2890625 |
| meta-S.U.N. | 14 | 24 | 38 | 0.14330665 |

Any remaining official-reference failure is preserved as `hull_unknown` and is not silently counted as an evaluated unstable structure.
