# New periodic DLM from original LLaDA

Current direction: original LLaDA-8B-Instruct plus fresh LoRA and crystal-token rows, trained on original MP20. No K4/K8 weights, data or feedback.

- [Current design](docs/periodic_self_repair_v1/RAW_LLADA_DESIGN.md)
- [Current checklist](docs/periodic_self_repair_v1/CHECKLIST.md)
- Entry: slurm/240_train_periodic_dlm_from_base.sbatch
- Branch: codex/periodic-self-repair
- Outputs: grounding/experiments/periodic_self_repair_20260906/from_original_llada_v1/

The retained core is Planner/species program, exact composition and geometric legal logits. New mechanisms are typed numerical output learning and periodic geometry inside Transformer attention.

Previous238/239 warm-start plans are superseded;39934 was cancelled/invalidated.39930 remains historical engineering/data work and supplies no input or result to this new model. Original K4/K8 work continues separately.
