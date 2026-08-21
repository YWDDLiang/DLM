# Seed inventory

The release distinguishes confirmed historical seeds, upstream defaults, and
unrecorded values.

| Component | Seed | Status |
|---|---:|---|
| Planner epoch-2 Python/Torch training RNG | `17` | confirmed historical |
| H1-A2 Planner sampling | base `17`, rank seeds `17/18` | confirmed historical |
| Quick-route Planner sampling | `17029` | confirmed historical |
| Quick-route ordinal ledger naming input | `17` | confirmed; not a B0 training seed |
| Quick-route ordinal ledger sampling root | `17029` | confirmed historical |
| B0/DLM data ordering | `20260515` | confirmed `data_seed`, not global training RNG |
| B0/DLM global training RNG | — | A800 run config/log audit found no recorded value |
| `model_494` Torch seed | `1234` | adjacent upstream source default; no checkpoint-specific run config found |
| `model_494` NumPy timestep RNG | — | upstream training code did not seed it |

The public training launchers do not relabel `20260515` as the B0 training
seed. If the historical values cannot be recovered, a future numeric value
will be explicitly called a **release seed** rather than a historical seed.
