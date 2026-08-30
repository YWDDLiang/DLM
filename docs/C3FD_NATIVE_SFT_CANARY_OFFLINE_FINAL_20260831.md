# C3FD-native SFT canary offline final

Date: 2026-08-31

Slurm job: `38768`

Status: **SUCCESS**

The fixed development canary completed all four raw cells followed by all four
model494-refined cells. Job38768 finished `COMPLETED 0:0` in `01:58:44` on
`4 A800 / 32 CPU`, for `7.9156 A800-hours`. No official MP query was run, and
neither training seed was selected or discarded.

## Main result

| Policy | Body parsed | Raw Direct joint | Refined Direct joint | Raw NU | Refined NU |
|---:|---:|---:|---:|---:|---:|
| 82017 | 502/512 (98.05%) | 154/512 (30.08%) | 456/512 (89.06%) | 502/512 | 391/512 |
| 82018 | 506/512 (98.83%) | 191/512 (37.30%) | 461/512 (90.04%) | 505/512 | 400/512 |

Fresh teacher-rich SFT therefore restores prompt/body execution under the
C3FD-predicted V2 interface, but it does not restore the raw structural
manifold. Most structural validity is supplied by model494 rather than by the
raw DLM.

The paired model494-refined-minus-raw CHGNet deltas are:

- policy82017: `-2.7722 eV/atom`, 95% composition-bootstrap CI
  `[-3.0344, -2.5124]`, fraction lower `0.9752`;
- policy82018: `-2.6143 eV/atom`, 95% CI `[-2.8920, -2.3509]`, fraction lower
  `0.9719`.

Independent-policy sensitivity is uncertain in raw space
(`82018-82017=-0.1646 eV/atom`, CI `[-0.4178,+0.0832]`) and effectively zero
after refinement (`-0.00010 eV/atom`, CI `[-0.0225,+0.0226]`). These comparisons
are seed-sensitivity diagnostics, never a basis for choosing one policy.

## Decision

The preregistered condition “execution recovered but raw stability remains
insufficient” is met. The next route is one fresh MP20-train-only, on-policy,
same-composition safety-aware alignment. It must preserve every raw-invalid
candidate, use within-composition continuous ranking, and retain both training
seeds. The historical 3,614 candidates remain excluded from formal training.

This canary makes no S.U.N. or official hull claim. Final Strict/Meta S.U.N. is
reserved for the one prospective 256-composition evaluation.

Final report SHA-256:
`9b19b458852f750970e723200cfa2eb93695411e31ae23fadb16ede990264758`.
Positive archive:
`archive/native_sft/canary_offline_success_38768/_ARCHIVE_SUCCESS`.
