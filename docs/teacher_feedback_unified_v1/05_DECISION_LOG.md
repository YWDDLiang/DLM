# Decision Log

Status: **multi-agent design approved; execution awaits explicit user approval**

| Decision | Alternatives considered | Reason |
|---|---|---|
| One shared canonical crystal state connects all modules | pass JSON/text/token arrays ad hoc between stages | prevents silent composition, site-order and quantization drift |
| AR and DLM logits meet in semantic value space | add raw token logits; force a shared tokenizer | their vocabularies and factorizations differ |
| Shared support is the physical-domain intersection | assume matching decimal precision means identical vocabularies | AR length range is wider and DLM coordinate bins 000/100 are PBC aliases |
| Keep AR native text and add SLA semantic action heads | enumerate all BPE strings online; retrain AR on DLM token IDs | preserves LLM text context and gives efficient DLM-aligned actions; SLA is not claimed to be native probability |
| Retain the successful C3FD–Llama chemical Planner | retrain Planner immediately | composition validity is already strong; current gap is structure commitment |
| Llama controls species-block order | let C3FD invent geometry order; fixed DLM schedule only | Llama has learned context, while C3FD has no coordinate logits |
| DLM positions stay stable while program chooses active blocks | physically reorder DLM tokens at every request | avoids tokenizer/position retraining merely to change commitment order |
| Exact N/elements are visible to the DLM | ask DLM to regenerate composition | composition is already solved and should condition geometry |
| Shared PBC controller is used in both A and B | post-hoc drop invalid CIFs | makes structural validity part of generation and permits a fair executor comparison |
| XYZ is one atomic site transaction | commit X/Y independently; reject them using arbitrary future values | exact PBC is defined on complete sites and joint commitment avoids an empty final-Z support |
| BC/BO/BG/BP are frozen-weight adjacent cells; B2 adds one matched epoch | bundle order, guidance, PBC and training | isolates decoder effects before adaptation |
| Existing G2 is an ablation, not automatically stacked | declare G2 final; discard it entirely | prior evidence is mixed and schedule-dependent |
| Diffusion feedback is a complete-state DLM corrector | query force/energy on masked structures; endpoint distillation | incomplete structures lack a unique physical graph and prior distillation failed |
| model494 supplies an empirical deployed-transition response | call `pred_x/pred_l` physical force or assume an in-distribution score | heads have different parameterizations/signs and raw B2 is not a formal t=800 draw |
| CHGNet trains confidence but is absent online | one CHGNet call per token; full force-vector student | lower runtime and avoids repeating the failed microstudent |
| Track-B independent work overlaps Track A; controller-dependent work waits | claim fully parallel A/B training; serialize everything | Track B depends on the frozen Track-A controller but BC implementation/baseline work does not |
| No scientific jobs before user approval | let the heartbeat start preparatory runs | the user explicitly reserved execution approval |

## Skeptic objections and resolutions

| Objection | Resolution |
|---|---|
| “Shared Llama” had ambiguous adapters and could change inside Track B | Freeze PlannerAdapter-P; train and freeze BodyAdapter-A, ProgramHead-A and SLA-A; Track B never updates them |
| Program labels required impossible DLM counterfactuals and contaminated Track A | Use one frozen starting-body-Llama teacher forward and fixed per-species text margins; no DLM or energy labels |
| Teacher Plan training versus predicted Plan inference | Train same-body teacher/predicted Plan views with combined source weight one |
| A0/A1 and B0/B1 bundled multiple interventions | Replace them with adjacent A0/A1 and BC/BO/BG/BP cells at fixed weights and schedule |
| SLA was falsely described as native AR probability | Define it as a separately supervised Llama semantic action policy; candidate trie is diagnostic only |
| Gate lacked an objective | Fused teacher-value CE plus alpha-to-zero regularization under a field/stage KL cap |
| SLA/gate see generated or jointly committed prefixes at inference | Train the gate on rollout-matched joint prefixes; keep SLA frozen and permit uncertainty-based abstention |
| DLM logits were compared across different mask schedules | Use one common block schedule and calibrate by family, remaining-mask ratio and stage |
| Runtime hard support could delete the MP20 CE label | Geometry hard support does not truncate CE; conflicting quantized teachers are disclosed and omitted only from geometry auxiliary loss |
| Site IDs ignored same-species symmetry | Treat IDs as ephemeral handles; augment within species and score with permutation-aware matching |
| Finite-image screen could miss triclinic collisions; X/Y could trap Z | Use exact or certified MIC for an atomic XYZ candidate beam before any axis commits |
| model494 response had unjustified score semantics and teacher-state calibration | Call it an empirical deployed-transition response and calibrate on frozen B2-generated MP20-train states |
| B3 used extra model494 and correction compute | B2C0 executes the same response call, fresh Llama/SLA call and DLM corrector with zero response; B2C0→B3 is compute-matched |
| CHGNet teacher/evaluator loop | CHGNet is development-correlated; promotion requires a held-out second MLIP and thermodynamic claims require registered DFT |
| One stream was too noisy | Use one model seed with two matched sampling streams for every fixed256 cell |
| “pure LLM”, “world model” and “verifier” were overstated | Paper uses LLM-only executor, scientific state and bidirectional pre-commit executor |

The complete-state response corrector remains **Candidate E1**: it is
implemented only after A/B and becomes a third contribution only through
registered independent evidence. Remaining sequential reviews may still
request revisions.

Skeptic disposition after revision: **APPROVED**.

## Constraint Guardian objections and resolutions

| Objection | Resolution |
|---|---|
| CHGNet worker counts exceeded CPU allocation | 4/8/10 process benchmark stays inside 8 allocated cores/GPU with one thread/worker |
| Seven cells × two streams had an optimistic ETA | fixed256 generation/raw evaluation revised to 5–8 hours |
| MIC candidate cost and exactness were unspecified | default 64 triplets, 256-row train-only throughput canary, certified GPU image bound and exact CPU fallback |
| model494 calibration could imply all 27,136 rows | fixed to 1,024 B2-generated train states plus 256 validation states |
| dual-8B placement and triple-model memory were unspecified | BF16, explicit device placement, no offload, batch1, 8 GiB headroom; model494 response and Llama+DLM correction are separate stages |
| B2 could mean full-parameter training | Track A and B2 are LoRA r8/alpha32/dropout0.05 on two A800 |
| “one epoch” was ambiguous with two Plan views | both views are averaged per source; effective source batch16 and exactly 1,696 updates |
| claimed A/B parallelism violated controller dependency | only BC work overlaps Track A; controller-dependent BO/BG/BP wait |
| candidate-trie diagnostic was unbounded | fixed to 512 rows and at most one position per family |

Constraint Guardian disposition after revision: **APPROVED**.

## Reader/User Advocate objections and resolutions

| Objection | Resolution |
|---|---|
| Track A sounded like an unconstrained ordinary LLM | Use “LLM-only executor” and define it as no DLM/continuous raw generator |
| Runtime state sounded internal to Llama | State explicitly belongs to the canonical runtime; Llama reads it |
| Geometry checks were called general physical admissibility | Restrict language to geometric feasibility |
| SLA/native text roles were unclear | SLA is the deployed semantic action head; native text supplies context/loss and deterministic serialization |
| BG/BP “revision” was operationally false | Define them as joint pre-commit fusion with argmax fusion-change metrics; only Candidate-E1 cells B2C0/B3 reopen a block |
| species/field/site commit units conflicted | Freeze three transactions: composition action, six-value lattice, one XYZ site; species is a scheduling container |
| partial codecs were undefined | Add partial AR rendering and masked DLM canvas contracts |
| response corrector and terminal refiner were conflated | Name and evaluate the one-step response-corrector and full terminal-refiner roles separately |
| E1 appeared as an already supported third contribution | Mark it globally as Candidate E1 with explicit calibration and promotion criteria |
| composition validity attribution was blurred | Separate learned proposal validity from deterministic body inventory retention |
| acronyms overloaded first-time readers | Add a complete short glossary and checkpoint-ownership table to README |

Reader/User Advocate disposition after revision: **APPROVED**.

## Final arbitration

The Integrator verified checkpoint ownership, semantic-space token bridging,
transaction units, teacher/predicted Plan handling, adjacent causal cells,
resource scheduling, E1 compute controls and the approval boundary.

Final disposition: **APPROVED**.

This disposition approves the plan for user review. It does not authorize
training, generation, evaluation, external queries or remote job submission.
