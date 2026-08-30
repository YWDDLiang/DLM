# C3FD-native teacher-rich SFT data final

CPU job `38686` completed successfully in `00:00:15` on 16 CPUs and 0 GPUs.
The immutable output is `data/c3fd_native_teacher_sft_v1_20260831`.

Each MP20 structure contributes exactly one ground-truth
`C3FD_NATIVE_PLAN_V2` rich JSON prompt and one dynamic crystal-body target.
C3FD-predicted Plans, masks, minimal prompts, Planner certificates, valence,
charge, prototypes, and oxidation fields are absent from formal SFT.

| Split | Teacher prompts | Target bodies |
|---|---:|---:|
| MP20 standard train | 27,136 | 27,136 |
| MP20 standard validation | 9,047 | 9,047 |

The original MP20 train/validation chemsys overlap is 3,469 and is disclosed.
Every row has prompt schema `C3FD_NATIVE_PLAN_V2`, view `teacher-native`, sample
weight one, and no `prediction_checkpoint` field.

Frozen SHA-256:

- manifest: `b77a1d0126d19576b2a6e381fe177de792b230cab2eca58e2c5c8097ff23ae06`;
- train: `f61cd177f252ce44ba15669a187a5f6fed22e2d3f50218088124fd0b5b836068`;
- validation: `bea7d8e67476bcdde545f13971cb87ea1e2931954ff8948410dc21ce38e509e1`.

The positive archive is
`archive/native_sft/teacher_data_success_38686/_ARCHIVE_SUCCESS`.

This freezes the H1-A2-faithful separation: teacher-rich MP20 SFT first;
C3FD-predicted rich JSON only at inference.
