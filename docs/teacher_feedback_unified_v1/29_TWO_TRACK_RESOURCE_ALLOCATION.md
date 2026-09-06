# User-authorized separate periodic DLM experiment

At 06:49 on 2026-09-06 the user allocated two A800/eight CPU and twelve hours to an independent periodic DLM architecture experiment, retaining four A800/sixteen CPU for the existing K8 and final evaluation. The user then explicitly allowed immediate use of currently free cards and a multi-agent audit.

Original K8 data, teacher, last two training passes, optimizer continuation and scientific evaluation remain this branch's responsibility. Script233 now defaults to four GPUs and four matching processes so the later independent experiment respects the user's allocation. Total across both workstreams stays at six A800, twenty-four CPU and two simultaneous jobs.

The separate implementation and its detailed design/audit live in:

- Local: D:/codex_work/ai4s/DLM_periodic_self_repair
- [Independent branch](https://github.com/YWDDLiang/DLM/tree/codex/periodic-self-repair)
- [Design and audit](https://github.com/YWDDLiang/DLM/blob/codex/periodic-self-repair/docs/periodic_self_repair_v1/DESIGN_AND_AUDIT.md)
- Remote code: /public/home/jiaosz/ywliang/ai4s/.sscd_periodic_self_repair_20260906_v1
- Remote artifacts: grounding/experiments/periodic_self_repair_20260906/

Formal new-model adaptation starts from the final completed K8 checkpoint. A clearly ineligible engineering check may use existing39892 and discard all updated weights. New training targets come from the model's own generated structures and verified terminals; the rejected extra MP20 construction CE is not restored.

The inherited237 four-GPU support-only proposal is superseded and has never been submitted. The new two-GPU script238 first verifies the actual architecture, then prepares all K4 targets; a second invocation prepares all K8 targets. Data, failures, provenance and results remain separately accounted. No new scientific SUN result exists at design submission.

The preceding two paragraphs are superseded by the user's later clarification: the new periodic DLM starts from original LLaDA and original MP20, with no K4/K8 weights or feedback. Its job is39937 and current design is maintained in the independent workspace. Historical39930 is not an input; mistaken39934 is cancelled and invalidated.

Original K8 completed both label shards. The first teacher was not trained because an unresolved Gd2 energy scale supplied89.24% of positive B gain. The explicit credibility variant fixes group13087 at uniform probability over its six verified finite occurrences while retaining its path supervision and every original record. It reports deltaA=deltaB=-0.0347304475eV/atom, KL0.009071156 and ESS1177.459 across the same662 supervised conditions. Four-GPU final training job39938 uses this teacher and continues the registered optimizer state from39892.
