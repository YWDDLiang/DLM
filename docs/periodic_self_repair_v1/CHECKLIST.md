# New DLM execution checklist

Authoritative direction: original LLaDA initialization and original MP20 base training. Active work is the new DLM, original K4/K8 continuation, and a conditional new-DLM K4/K8 post-training stage. New model2A800/8CPU; original mainline4A800/16CPU.

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
- [x] Submit raw training240 on two A800: job39937, started from original LLaDA with original MP20 only.
- [ ] Check initialization, first ten updates, gradients, throughput and coverage during training.
- [x] Prepare fixed256 native evaluation241 with matched reference comparison; submit only after39937 succeeds.
- [x] Prepare same-input frozen model494 tau800 evaluation242; submit only after native241 succeeds.
- [ ] Finish declared endpoint and run fixed256 native generation.
- [ ] Run same-input frozen model494 tau800 separately.
- [x] Freeze the pre-result OR gate: native>=252/6/55 or tau800>=252/18/124 for reconstructed/Strict/Meta.
- [ ] If the raw native result passes and original K8 job39938 is complete, collect this new model's own K4 paths and train two passes.
- [ ] From that new K4 policy, collect its own K8 paths and train two continuation passes.
- [ ] Do not substitute the previous DLM's K4/K8 paths for either new-model collection.
- [ ] Keep original K4/K8 monitoring active without mixing its data into this model.
- [ ] Report new-model completed results and gaps by18:49; original mainline deadline19:19.

Design: RAW_LLADA_DESIGN.md. Older warm-start documents are historical and superseded by this checklist.

Latest observed healthy startup: steps1–3 losses5.8872/6.0712/5.9611, finite clipped gradients, peak memory16.03GiB/GPU. Per user instruction, let39937 run naturally and check again only for failure, stage completion, or before consuming its checkpoint.
