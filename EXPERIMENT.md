# Periodic DLM self-repair

This is the separate implementation workspace for the user's 2026-09-06 DLM architecture experiment.

- Branch: codex/periodic-self-repair
- [Design, mathematical scope, and audit](docs/periodic_self_repair_v1/DESIGN_AND_AUDIT.md)
- [Execution registration and live status](docs/teacher_feedback_unified_v1/29_PERIODIC_SELF_REPAIR_12H.md)
- Local workspace: D:/codex_work/ai4s/DLM_periodic_self_repair
- Separate remote source: /public/home/jiaosz/ywliang/ai4s/.sscd_periodic_self_repair_20260906_v1
- Separate experiment outputs: grounding/experiments/periodic_self_repair_20260906/

The original K8 experiment retains its own source, recipe, checkpoints, and evaluation. This method's formal training starts from the completed final K8 policy. Earlier checkpoints may be used only for explicitly ineligible engineering checks, with all temporary updated weights discarded.
