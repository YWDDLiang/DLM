# New DLM execution checklist

Authoritative direction: original LLaDA initialization, original MP20, no K4/K8 weights/data. Priority: new DLM, original K4/K8, diffusion joint work, then self-improvement. New model2A800/8CPU; original mainline4A800/16CPU.

- [x] Retain Planner, species program, exact canvas and geometric legal logits.
- [x] Implement periodic coordinate Fourier and ordered lattice numerical logits.
- [x] Add actual periodic geometry inside Transformer attention.
- [x] Add explicit mask/task/noise and prevent hidden-GT geometry leakage.
- [x] Implement fresh LoRA and trainable new input/output token rows, compact save/reload.
- [x] Implement original-MP20 dense construction and structured repair views.
- [x] Reuse measured r8/alpha32/dropout.05, global16, max382, LR5e-5 to1e-5, warmup100/stage, weight decay0.
- [x] Implement two full epochs/two views,6784 updates, final checkpoint, fixed validation monitor.
- [x] Keep full-source schema CE and separately gated legal CE with conflict ledger.
- [x] Cancel and invalidate incorrect K4 warm-start39934; do not resume239.
- [ ] Submit raw training240 on two A800 and record job ID.
- [ ] Check initialization, first ten updates, gradients, throughput and coverage during training.
- [ ] Finish declared endpoint and evaluate fixed256 native generation.
- [ ] Evaluate same inputs with frozen model494 tau800 separately.
- [ ] Keep original K4/K8 monitoring active without mixing its data into this model.
- [ ] Report new-model completed results and gaps by18:49; original mainline deadline19:19.

Design: RAW_LLADA_DESIGN.md. Older warm-start documents are historical and superseded by this checklist.
