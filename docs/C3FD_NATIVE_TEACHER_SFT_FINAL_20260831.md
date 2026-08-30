# C3FD-native teacher-only SFT final

Date: 2026-08-31

Slurm job: `38703`

Run: `runs/c3fd_native_teacher_sft_38703`

Status: **SUCCESS**

The two independently seeded, fresh-LoRA DLM policies completed the frozen
teacher-only MP20 SFT contract. The job finished `COMPLETED 0:0` in `01:14:51`
on `4 A800 / 32 CPU`, corresponding to `4.9900 A800-hours` observed. The
scientific configuration was unchanged from the preregistered recovery of the
pre-run environment failure in job38701.

## Frozen inputs and method

- shared pretrained LLaDA-8B backbone; no old rich/minimal adapter loaded;
- MP20 teacher-rich `C3FD_NATIVE_PLAN_V2` only: train `27,136`, validation
  `9,047`;
- LoRA `r=8`, alpha `32`, dropout `0.05`, targets `q/k/v/ff/up`;
- seeds `82017` and `82018`;
- effective batch `16` per seed;
- stage 1: `1,696` updates, LR `5e-5`; stage 2: `1,696` updates, LR `1e-5`;
- only step `3392` is eligible. Step `1696` was monitoring-only; no seed,
  checkpoint, or early-stop selection was performed.

Both step-zero canaries recorded `checkpoint=null`, `fresh_lora=true`,
`lora_B_max_abs=0`, across `160` LoRA-B tensors. All `339` logged train events
per seed had finite loss, task loss, gradient norm, and learning rate.

## Terminal policies

| Seed | Eligible checkpoint | Validation loss @1696 | Validation loss @3392 | Adapter SHA-256 |
|---:|---:|---:|---:|---|
| 82017 | step3392 only | 2.7781652107 | 2.7043892791 | `06cd5465d1c12dc70c695a12602a4bd99f6c5561f2dc44daa431aadeef962b20` |
| 82018 | step3392 only | 2.1264180888 | 1.8198185447 | `7f22580a78b4823b44ee841a25cce9fd39541bb0a2192171b0adb4788725526c` |

Each adapter is `6,391,016,776` bytes. Both adapter hash manifests were
rechecked successfully. Each seed directory contains one saved checkpoint,
`checkpoints/step-3392`, and no `final` alias.

## Interpretation and next action

This terminal establishes reproducible Planner-interface SFT, not a stability
claim. Validation CE improved from the monitoring point for both seeds, but it
cannot establish raw structural validity, energy stability, or S.U.N. The next
scientific step is the fixed, raw-first train/MP20-standard-validation canary,
with both seeds retained. It is followed by the already frozen one-shot
prospective cohort, generation, raw/refined evaluation, and official MP query.

Positive archive marker:
`archive/native_sft/train_success_38703/_ARCHIVE_SUCCESS`. The archive records
the immutable run metadata, reports, hashes, logs, configurations, canaries,
and policy path/hash references; the large adapters remain in the immutable
source run rather than being duplicated.
