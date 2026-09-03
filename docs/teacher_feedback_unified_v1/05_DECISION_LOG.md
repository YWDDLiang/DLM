# Decision Log

Status: **SPAD executed; SPAD-E design approved and awaiting implementation approval**

## Design-review decisions

| Decision | Rejected alternative | Rationale |
|---|---|---|
| C3FD is a typed chemical-state model | call it a fine-geometry world model | it has composition/action and coarse Plan logits, not coordinate logits |
| Track A is an LLM-only raw executor | call the whole refined pipeline pure LLM | the terminal continuous refiner is separate |
| Track B is the priority route | make A's body adapter a prerequisite for B | B only needs the frozen Planner Plan and action order |
| Spaces meet through Plan/program/state | add AR BPE logits to DLM special-token logits | token spaces and probability factorizations differ |
| DLM owns all geometry values | add an unimplemented SLA/gate first | Plan/program are sufficient to establish Llama guidance |
| A Plan-conditioned Llama pointer predicts species order | call the canonical C3FD trace a learned order | C3FD enforces increasing species keys; the pointer is the smallest honest learned permutation |
| DLM storage remains canonical | physically reorder token positions | a program can activate arbitrary non-contiguous positions |
| Anchor–backfill is the core DLM operation | use only monotone schedule changes | suffix-visible re-masking demonstrates non-causal DLM necessity |
| Exact N/elements are prefilled | ask DLM to regenerate solved composition | protects C3FD's demonstrated contribution |
| Only exact `7+4N` is production | use universal 87-token EOS tail | the latter mismatches SFT for N<20 |
| 000/100 are one torus action | treat them as independent physical values | they are periodic aliases |
| Lattice/XYZ are transactions | commit arbitrary individual fields | full geometry is needed for validity decisions |
| SPAD-E learns terminal preference on suffix-visible backfill | detached response residual or inference-time critic | the downstream transition supervises the same DLM-specific operation offline |

## Initial review disposition

The original canonical-state/SLA design passed skeptic, constraint, reader and
arbiter review as a coherent proposal. Implementation audit then found that
SLA, gate and body-program heads did not exist and were not the shortest
Llama→DLM signal.

The user authorized implementation and requested B priority plus explicit use
of DLM future-first/backward revision.

## Implementation-audit findings

### Accepted

- 2,481 crystal special-token strings exist in source; dynamic bodies use
  2,457.
- Real step-3392 boundary probes encode atomically.
- DLM attention and position-group API support arbitrary non-contiguous
  future-first generation.
- Each monotone commit already triggers a fresh full DLM forward.
- Existing forced-mask training is reusable.

### Must be fixed

- full checkpoint/tokenizer and MP20 coverage passed in CPU job 39507;
- configured mask ID 126336 is valid but must remain explicitly checked;
- strict parser now rejects surrounding garbage;
- schedule validation now requires complete coverage;
- old universal sampler appends visible EOS tails;
- committed dynamic tokens cannot yet be re-masked in production;
- current collision mask detects duplicate bins, not general 0.5 Å triclinic
  distance;
- random-mask training does not match program predictor/backfill states.

## B-first pivot

The first three module audits recommended:

> C3FD–Llama Plan + species construction program → non-contiguous
> anchor-first DLM → complete future context → suffix-visible anchor backfill.

This became SPAD and supersedes SLA/gate as the first Track-B path. A subsequent
code audit proved the action trace canonical; SPAD therefore uses a
geometry-supervised pointer on the terminal Planner-Llama state.

## Evidence policy

- B0→BC isolates transaction order; BC→BP isolates the learned Llama-pointer program.
- BP→BR isolates one suffix-visible remask sweep.
- BR→BS isolates schedule-matched LoRA training.
- BR-no-suffix tests whether visible future context carries the gain.
- proposal composition validity and body composition retention are separate.
- one model seed and two common streams are fully disclosed.
- Strict/Meta S.U.N. >10%/50% is the target, not a row-selection rule.
- small failures receive a root-cause repair; scientific negatives remain
  visible and motivate only adjacent revisions.

## Post-SPAD stability decision

The prospective BS endpoint reached refined Direct `512/512` but Strict/Meta
S.U.N. `6.84%/45.70%`.  Structured skeptic, constraint, paper-reader and
arbiter review therefore accepted one tightly coupled extension:

- SPAD-E acts only on a Llama-programmed suffix-visible XYZ backfill state;
- `K=4` always includes the existing XYZ no-op plus three PBC-legal reference
  DLM actions;
- model494 tau800 and CHGNet endpoint energy label those actions offline on
  MP20-train only;
- a 0.05-nat trust region and explicit reference KL define the target;
- SPAD-CE is the equal-compute control;
- force is diagnostic only; no force/stress loss or inference-time critic;
- at inference the DLM first completes the future canvas, then rewrites the
  earlier site from learned weights, and model494 runs once only after discrete
  generation ends.

The existing official cohort is development evidence for this design.  A
positive result requires a second training seed and newly frozen confirmatory
cohort.

## Resource decision

- maximum 6 A800, 2 jobs, 8 CPU/GPU;
- B gets up to 4 A800;
- A/evaluation gets up to 2;
- current source sync uses local Git push and A800 HTTPS clone/pull;
- non-Git scp transfers are batched and separated by at least five minutes.
