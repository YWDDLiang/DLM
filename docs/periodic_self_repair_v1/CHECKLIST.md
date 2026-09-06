# Periodic DLM implementation and execution checklist

Latest: 2026-09-06 07:58 Asia/Shanghai. Independent branch codex/periodic-self-repair; remote root grounding/experiments/periodic_self_repair_20260906/. Two A800/eight CPU reserved for this new method, four A800/sixteen CPU for original K8. Count only this experiment's jobs; do not manage unrelated projects' jobs.

- [x] Preserve paper core: C³FD/typed Llama Planner, species program, exact composition/canvas, legal-set definition and conditional token sampling.
- [x] Implement zero-initialized numerical output increments and internal periodic attention bias.
- [x] Supply explicit mask/task/noise information, with unknown errors distinct from zero noise.
- [x] Retain exact conditional full-cell draws and whole-transaction rollback.
- [x] Complete three independent read-only audits and fix identified correctness problems.
- [x] Pass36 local related CPU tests plus independent BF16/FP32 probability/gradient checks.
- [x] Commit and push design/code; deploy to separate remote source.
- [x] Run two-GPU actual-model preflight39930: zero initial logits difference, zero replay difference, finite gradients, ~34.73GiB/GPU.
- [x] Prepare all4096 K4 requests:972 target candidates; preserve3124 unavailable requests.
- [x] Complete fresh target labels39930:695 verified,277 not converged,3124 unavailable.
- [ ] Finalize raw-target admission and start formal repair round0 on two GPUs from39892, four passes, original CE anchors and actual-policy KL.
- [ ] In parallel, complete original K8 labels and original four-GPU teacher/training pipeline.
- [ ] Prepare and label all8192 K8 quantized-target requests on this method's two GPUs.
- [ ] Continue the round0 repair model on complete K4+K8 data, four passes; frozen reference becomes original final K8.
- [ ] Verify the fixed-start repair results with the unfiltered repair-validation denominator.
- [ ] Evaluate the final new model on the frozen256 native requests and the same requests with frozen model494 tau800; preserve all failures.
- [ ] Record native physical distributions, paired A/B, Stable, N, U and SUN separately.
- [ ] At18:49, report completed new-method results and missing work; original mainline deadline remains19:19.

The immediate-start amendment changes the previous wait-for-K8 starting point. It does not add database construction targets, change the existing K8 recipe, or treat engineering checkpoints as trained scientific models. Early repair scores, if reported, are round0 diagnostics.
