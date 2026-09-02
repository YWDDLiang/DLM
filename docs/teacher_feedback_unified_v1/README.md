# Teacher-Feedback Unified Method Package

Status: **execution approved; implementation and preflight active**

Branch: `codex/unified-scientific-decoding`
Worktree: `D:\codex_work\ai4s\DLM_unified_scientific_decoding`

This package turns the teacher feedback into two connected methods. The
implementation audit in
[`06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md`](06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md)
is authoritative where it supersedes the initial SLA/gate design:

1. **Track A — LLM-only executor** (the requested pure-LLM route): a
   C3FD-conditioned Llama generates a complete crystal through typed semantic
   actions and emits ordinary CrysLLMGen text.
2. **Track B — LLM-programmed DLM:** a small Plan-conditioned pointer on the
   Planner Llama predicts a species permutation, which controls a masked DLM's
   non-contiguous anchor order; the DLM later re-masks early anchors with the
   generated suffix visible.

The models do not share token IDs. They communicate through a canonical crystal
state and typed semantic action space. The frozen continuous refiner (internal
checkpoint name: model494) receives arrays, not language tokens.

In one sentence: **C3FD constrains reachable chemistry, Llama predicts a
geometry-supervised construction program over the certified Plan, and the DLM
uses that program to build future context before backfilling earlier atoms.**

“Same Llama” refers to one base backbone with frozen stage-specific weights:

| Stage | Active Llama components |
|---|---|
| chemical planning | retained PlannerAdapter-P + typed C3FD residual heads |
| Track-A body | independent Plan-conditioned AR body adapter |
| Track-B guidance | PlannerAdapter-P Plan text + Llama pointer permutation |

Track B does not depend on Track-A body weights and receives execution priority.
Its first path trains only a DLM LoRA after decoder-only program/remask cells.
Candidate E1 later owns a separate stability module.

Short glossary:

| Term | Meaning here |
|---|---|
| C3FD | the project's typed chemical-state model with learned action scores and reachable support |
| AR | autoregressive, left-to-right Llama decoding |
| DLM | masked diffusion language model using dedicated crystal tokens |
| SSCD | Scientific-State Commit Decoding, the complete proposed framework |
| LS / SG / VPA | lattice system / space-group bucket / volume-per-atom bin; all are soft Plan conditions |
| PBC / MIC | periodic boundary conditions / minimum-image calculation |
| Direct | existing composition/structural validity evaluator; joint validity requires both |
| S.U.N. | stable, unique and novel generated crystal |
| N / U / NU | novel / unique / both novel and unique counts |
| NFE | neural-network forward evaluation |
| MLIP | machine-learned interatomic potential |
| Compact-V2 | retained composition-plus-coarse-Plan interface used by the current DLM |
| G2-PBC-R | historical periodic-relation residual; only an ablation in this plan |
| checkpoint 494 | current frozen continuous crystal refiner, internally called model494 |
| A0/A1 | LLM-only executor without/with periodic commit control |
| SPAD | Scientific Programmed Anchor–Backfill Denoising, the B-route core |
| B0/BC/BP/BR/BS | retained schedule / canonical SPAD / Llama-pointer SPAD / suffix-visible remask / matched-SFT endpoint |
| E1 | later continuous-response or force-to-DLM stability contribution |

Read in this order:

1. [Unified method and paper story](00_UNIFIED_METHOD_PLAN.md)
2. [Track A: LLM-only executor](01_TRACK_A_PURE_LLM.md)
3. [Track B: LLM-guided DLM](02_TRACK_B_LLM_GUIDED_DLM.md)
4. [Cross-representation and diffusion contract](03_CROSS_REPRESENTATION_AND_DIFFUSION.md)
5. [Execution checklist](04_EXECUTION_CHECKLIST.md)
6. [Decision log](05_DECISION_LOG.md)
7. [Implementation audit and B-first pivot](06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md)

The active ten-minute heartbeat is `sscd-a-b-approval-and-execution`. Execution
is approved; it stays quiet except for terminal audit, training, generation,
evaluation, S.U.N. or blocking events.
