# Periodic DLM self-repair

This is the separate implementation workspace for the user's 2026-09-06 DLM architecture experiment.

- Branch: codex/periodic-self-repair
- [Design, mathematical scope, and audit](docs/periodic_self_repair_v1/DESIGN_AND_AUDIT.md)
- [Execution registration and live status](docs/teacher_feedback_unified_v1/29_PERIODIC_SELF_REPAIR_12H.md)
- Local workspace: D:/codex_work/ai4s/DLM_periodic_self_repair
- Separate remote source: /public/home/jiaosz/ywliang/ai4s/.sscd_periodic_self_repair_20260906_v1
- Separate experiment outputs: grounding/experiments/periodic_self_repair_20260906/

The original K8 experiment retains its own source, recipe, checkpoints, and evaluation. Following the user's 07:58 instruction to start the new two-GPU training immediately, repair round0 starts from completed original K4 policy39892 using complete K4 feedback. Repair round1 continues that new model on complete K4+K8 feedback, with final K8 as the frozen reference policy. Engineering weights from job39930 were discarded. This two-stage amendment supersedes the earlier plan to wait for K8 before any scientific repair training.
