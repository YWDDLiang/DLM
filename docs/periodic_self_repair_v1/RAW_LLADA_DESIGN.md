# Raw-initialized periodic numerical DLM

This supersedes the proposed K4/K8 warm start. Load only the original LLaDA-8B-Instruct and tokenizer. Fresh crystal tokens have independent trainable FP32 input/output row increments; original vocabulary rows stay frozen. Fresh LoRA uses r8/alpha32/dropout0.05 on q/k/v/ff/up projections.

Preserve the original Planner, species program, exact composition/7+4N canvas and geometric legal logits. Numeric-logit increments use eight Fourier modes for coordinates, log-length/RBF for lengths and ordered nonperiodic RBF for angles. A zero-initialized MLP supplies actual old periodic pair geometry directly as Transformer attention bias. Unknown coordinates cannot create geometry pairs.

Task/mask/noise are explicit; unknown error is -1, regardless of visibility. Clean construction uses its masked canvas as old geometry, preventing target leakage. Repair uses a structured corruption and its correctly conditioned clean prefix. Known-noise metadata is hidden with probability0.5 to cover the deployment unknown-noise condition.

Read only original-MP20 teacher-only train/val data (27136/9047 sources), with prompt, clean answer, Plan/program and source metadata. Each source has a dense random-mask construction view and a scalar full-cell repair view per epoch. Two full epochs produce6784 global-batch16 updates. No K4/K8 generated paths, labels, weights or previous DLM tables are used.

Loss is family-balanced schema denoising CE plus0.25 times a sampled actual-legal-support CE. Each contains90% exact CE and10% numerical CE, narrowing from0.75 to0.25 bins. Schema temperature is1; legal-policy temperature is0.7, with the same alias folding and raw dtype as deployment. An unsupported teacher remains in schema training; its legal contribution is zero and the conflict is recorded. Targets are never added back to legal support.

Dense CE is conditional denoising, not full trajectory likelihood. Raw-model deployment uses scalar construction followed by full-cell repair and whole-transaction rollback. Construction conditions all remaining masked numeric positions consistently with training and trace replay. Older half-cell/cooperative/closure phases remain optional legacy interfaces.

Reference hyperparameters: measured canonical153/39282 recipe; two A800, microbatch1/rank, accumulation8, global16, max length382, seed82017/data_seed20260515; LR5e-5 then1e-5, warmup100/stage, cosine floors0.2/0.1, AdamW weight decay0, clip1. Two views double the old3392 updates to6784. Runtime is measured, not promised from the old speed.

Integrated checks cover raw initialization, new-row/module gradients, finite loss and complete source/view coverage. The same100 original validation sources monitor both epoch boundaries without checkpoint selection. Final fixed256 native and frozen model494 tau800 evaluations remain separate. Periodic features use the declared125-image shell; strict whole-model equivariance and universally exact nearest images are not claimed.

Self-improvement begins only after this raw model has an eligible final checkpoint and a fixed native evaluation. Job243 will run this model itself on1024 frozen train-only Planner/program conditions, retain all4096 K4 requests, label every native endpoint with the common protocol, quantize each verified R(x), and independently label Q(R(x)) before admission. The existing K4/K8 generated paths, labels, teachers and checkpoints are never read. This future collection is a new model-specific dataset, not part of base training.
