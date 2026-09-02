# Failed methods

This is the sole retained record for retired BTRD/model494-distillation work.
The corresponding checkpoints, generated datasets, run directories, Slurm
wrappers, implementation code, detailed contracts and duplicated analyses were
deleted after the results became terminal. None of these methods is part of the
promoted G2 or Plan1200 pipeline.

## Basin-Target Residual Distillation (BTRD)

| Attempt | What was tried | Matched first-256 result | Verdict |
|---|---|---|---|
| BTRD-tau200 | Train only the G2 residual for 512 updates on 6,038 model494 tau200 endpoint targets plus 2,154 MP20 anchors. | BASE body/Direct/Strict/Meta `254/111/11/44`; BTRD `251/115/5/45`. Paired CHGNet delta `+1.187 eV/atom`, 95% CI `[-0.318,+3.977]`. | **Failed.** Direct improved by 4, but Strict fell by 6 and stability did not improve. |
| BTRD-800T | Repeat with 6,038 tau800 endpoint targets and the registered collision-tail loss, again for 512 residual-only updates. | BASE `254/111/9/42`; BTRD-800T `253/117/5/46`. Paired CHGNet delta `+0.931 eV/atom`, 95% CI `[-0.570,+3.699]`. | **Failed.** Direct improved by 6 and Meta by 4, but Strict fell by 4 and the energy direction remained adverse/uncertain. |
| Full-corpus basin distillation | Proposed model494 tau800 targets for all 27,136 MP20-train rows, followed by residual-only and joint residual/LoRA training. | Not run; estimated cost was about 106 A800-hours. | **Cancelled before execution.** The two smaller BTRD experiments already falsified the assumed stability-transfer mechanism. |

### Failure reason

The training data imitated model494 endpoint geometries but did not encode an
explicit raw-structure-to-endpoint correction. The same endpoint sequence was
used as the denoising target and noisy input basis. Consequently, the objective
could improve structural execution/Direct without learning a direction toward
a lower-energy basin. The result does not support claims of basin transport,
vector-field recovery or raw stability distillation.

### Engineering-only failures

Jobs 39186 (delegated-script execute permission), 39187 (`PYTHONPATH`) and
39189 (shell local binding) stopped before science. Job 39185 completed all 512
training updates but its wrapper failed a post-training log assertion; the
valid endpoint was finalized CPU-only. These incidents did not alter the
scientific verdict above.

## Projected-force microstudent

The residual-only microstudent checkpoint from job39230 changed 86/128 holdout
texts but produced six invalid→valid and six valid→invalid Direct flips, for net
zero. It was not promoted and its 6 GiB step128 checkpoint was removed on
2026-09-02. Logs, the frozen data audit and the failure reason remain.

## Canonical-site G2 interaction

Canonical MP20 site ordering improved fixed256 raw Direct from old G2 `128` to
canonical DLM `143`. Adding a new G2-PBC-R epoch reduced it to `133`, and neither
canonical arm established a paired raw CHGNet advantage. The canonical-G2
step1696 checkpoint was therefore classified as failed and its 6 GiB checkpoint
was removed on 2026-09-02. The canonical DLM step3392 checkpoint is retained as
the supported Direct-improving artifact; the historical promoted G2-A remains
untouched.
