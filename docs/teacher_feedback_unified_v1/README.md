# Teacher-Feedback Unified Method Package

Status: **design and review only; no scientific run starts before explicit user approval**

Branch: `codex/unified-scientific-decoding`
Worktree: `D:\codex_work\ai4s\DLM_unified_scientific_decoding`

This package turns the teacher feedback into two connected methods:

1. **Track A — LLM-only executor** (the requested pure-LLM route): a
   C3FD-conditioned Llama generates a complete crystal through typed semantic
   actions and emits ordinary CrysLLMGen text.
2. **Track B — LLM-guided DLM:** the same Llama controls a masked DLM's
   species-block order and semantic value priors while the DLM retains its own
   special `7+4N` vocabulary.

The models do not share token IDs. They communicate through a canonical crystal
state and typed semantic action space. The frozen continuous refiner (internal
checkpoint name: model494) receives arrays, not language tokens.

In one sentence: **C3FD constrains reachable chemistry, Llama chooses the
scientific program and semantic actions, and the masked DLM joins the
pre-commit decision using bidirectional context; an optional later corrector
may explicitly reopen one completed block.**

“Same Llama” refers to one base backbone with frozen stage-specific weights:

| Stage | Active Llama components |
|---|---|
| chemical planning | retained PlannerAdapter-P + typed C3FD residual heads |
| Track-A body | frozen BodyAdapter-A + ProgramHead-A + SLA-A |
| Track-B guidance | the exact same frozen Track-A body components |

Core Track B through B2 may train only its DLM LoRA and agreement gate.
Candidate E1 later owns one separate `Confidence-E1` module trained after B2;
it never changes the frozen Llama controller or B2 weights.

Short glossary:

| Term | Meaning here |
|---|---|
| C3FD | the project's typed chemical-state model with learned action scores and reachable support |
| AR | autoregressive, left-to-right Llama decoding |
| DLM | masked diffusion language model using dedicated crystal tokens |
| SSCD | Scientific-State Commit Decoding, the complete proposed framework |
| SLA | supervised Llama Semantic Logit Adapter over physical field values; not raw BPE probability |
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
| BC/BO/BG/BP | frozen DLM under canonical order, Llama order, Llama guidance and then PBC control |
| B2 | BP after one schedule-matched DLM LoRA epoch |
| B3 / E1 | optional B2 complete-state response corrector; distinct from terminal refinement |

Read in this order:

1. [Unified method and paper story](00_UNIFIED_METHOD_PLAN.md)
2. [Track A: LLM-only executor](01_TRACK_A_PURE_LLM.md)
3. [Track B: LLM-guided DLM](02_TRACK_B_LLM_GUIDED_DLM.md)
4. [Cross-representation and diffusion contract](03_CROSS_REPRESENTATION_AND_DIFFUSION.md)
5. [Execution checklist](04_EXECUTION_CHECKLIST.md)
6. [Decision log](05_DECISION_LOG.md)

The ten-minute heartbeat may inspect status and maintain this package, but it
must remain quiet and must not submit training, generation, evaluation or
external-query work until the user explicitly approves execution.
