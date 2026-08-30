# Rich-Planner + Stable-DLM decision log v2

Date: 2026-08-30
Review state: APPROVED by final Arbiter review

## Understanding lock

The user prioritizes stability, accepts recovery to the historical H1-A2 range,
wants the existing Planner/DLM/diffusion story restored, permits up to six
A800s, and requires conclusions to be verified against original artifacts.
The Planner may recover predicted rich structural context, but the DLM must
remain the structure generator and no target/test rich labels may leak into
inference.

## Initial decisions

| Decision | Alternatives considered | Rationale | Status |
|---|---|---|---|
| Audit and test current C3FD's existing seven-line interface first | keep minimal composition+N; build DLM self-intent heads; restore old free-text Planner | C3FD already has the renderer and active lattice/volume heads; this is the shortest story-consistent recovery test and preserves 100% composition validity | proposed |
| Treat SG bucket as compiler-derived, not an independent structural signal | independently sample SG head; hard-enforce SG; remove the line | current sampler maps lattice one-to-one to SG bucket, so it adds no sampled information; the line may still preserve historical prompt compatibility | proposed pending metric audit |
| Give volume highest field priority and lattice an uncertainty-aware soft role | use all fields equally; use VPA/CN self-intent; no structural context | historical volume association is modestly favorable; lattice/SG endpoint evidence is mixed/adverse; field-level validation and canary are still required | proposed pending audit |
| Use a three-arm six-cell recovery canary | full rich versus minimal only; unknown-token neutralization; many field subsets; train immediately | `R0-RCF` uses in-vocabulary, marginal-preserving counterfactual fields within one rich DLM; `R0-M0` measures practical package recovery; avoids a downstream field sweep | revised |
| Initialize stability training from a demonstrated rich-compatible DLM | continue current minimal BASE; train new architecture from scratch | historical rich lineage retained better raw/S.U.N. behavior; the sprint should recover first and then optimize stability | proposed pending checkpoint verification |
| Use full-sequence same-composition continuous-energy alignment plus raw safety | ordinary CE/SGTC; D3PO post-refiner only; token critic; RL | ordinary CE and token critic failed; D3PO supplied weak refined signal but damaged raw geometry; the revised objective targets both facts | proposed |
| Keep model494 tau800 fixed and report raw/refined separately | tune tau900; alter refiner now; omit raw | model494 is a large absolute contributor and attribution confound; changing it now would obscure whether the DLM improved | locked unless interface-only evidence appears |
| Use graded interpretation, not many hard gates | 10/50 deletion gate; one p-value gate; best cell selection | recovery itself is scientifically useful; continuous effects and replication carry more information than a threshold | locked |

## Known objections awaiting review

1. Current minimal and historical rich DLM checkpoints differ, so package-level
   recovery is not a causal field comparison.
2. Predicted lattice/volume may be inaccurate or multimodal; hard conditioning
   can harm valid polymorphs.
3. The seven-line interface may recover raw validity without improving
   thermodynamic stability.
4. Existing historical listwise candidates are not a clean-room, from-scratch
   dataset for the final paper claim.
5. Six-GPU canary time may delay the only training run if checkpoint/wrapper
   compatibility is poor.
6. A third algorithmic contribution may be unjustified; the evaluation and
   attribution framework should not be oversold.

## Skeptic review resolution

Disposition received: **REVISE**. All eight objections were accepted.

| Objection | Resolution |
|---|---|
| Competing rich/listwise/SI documents permit hidden route selection | This v2 checklist supersedes the sprint actions in earlier drafts. Rich is a frozen development recovery test; minimal listwise is the sole pre-outcome fallback; SI heads are deferred. |
| Unknown-token neutralization is OOD and not implemented | Replaced `RH` with a pre-frozen, in-vocabulary, marginal-preserving `RCF` permutation; no unknown token. |
| Three arms cannot fully identify field quality | Claim narrowed: `R0-RCF` tests composition alignment, `R0-M0` is package recovery, and neither is oracle accuracy. |
| 32--64 rows cannot establish recovery | Development cohort fixed at 256 chemsys-held-out rows; interpretation is continuous orientation, not a final population claim. |
| Five trained heads were misdescribed as five active predictions | Corrected: only lattice and volume are rich-logit sampled; anion/charge are hard-derived; SG is compiled. |
| Active rich fields lack calibration evidence | Per-field NLL/accuracy/majority/ECE/ordinal/seed audit is now required before GPU. |
| Raw execution alone could promote a thermodynamically useless route | Split interpretation: execution-only recovery may initialize training but is never called stability evidence; energy alignment remains required. |
| Frozen-ledger and Stable-DLM code paths are unproven | Added fail-closed code preflight. No GPU starts until ledger support, parser, wrapper, hashes, and tests are demonstrated. |

## Constraint Guardian review resolution

Disposition received: **REJECT until execution preflight**. All operational
requirements were accepted; the stale asset-transfer ledger is treated as an
item to reconcile against the live cluster, not as proof that assets are absent.

| Requirement | Resolution |
|---|---|
| Current assets/hashes are not proven | Added a live asset/implementation SHA manifest as the first execution task; no GPU before it passes. |
| Six-GPU/two-job limits are not executable | Added wrapper allowlist, <=6 GPU, 4--8 CPU/GPU and active+pending <=2 fail-closed guards. |
| RCF/R0 wrapper is absent | GPU remains unauthorized until a concrete same-ledger wrapper and parser/sampler integration tests exist. |
| Listwise fallback was overstated as implemented | Corrected to partially implemented; runnable status requires trainer/wrapper/integration tests. |
| Output-exists is not atomic duplicate prevention | Added atomic contract-hash/job-ID registration and immutable attempt directories. |
| Leakage guard trusts Boolean provenance | Added recomputation against hashed blocked identities and chemsys sets. |
| API/recovery contract underspecified | Added one 0-GPU/<=8-CPU bounded query process, query ledger, nonambient credential handling, immutable recovery attempts and no repeat after completion. |

## User Advocate review resolution

Disposition: **REVISE for unambiguous expectations**. All six clarity
objections were accepted.

| Objection | Resolution |
|---|---|
| “H1-A2 recovery” named several incompatible references | Default frozen to corrected exact replay `8.58/46.08%`; continuous and historical views remain disclosed context. |
| Design text could be mistaken for execution progress | Added six explicit progress states and required job/artifact/marker evidence before reporting a stage active. |
| Too many apparent hard gates | Clarified that there is one compound execution gate plus the deadline; all other checklist rows are evidence items or graded result classes. |
| Six-A800 language was ambiguous | Corrected to six maximum concurrent A800s and added cumulative GPU-hour/peak/remaining reporting. |
| Refined-only gain conflicted with Stable-DLM claim | Stable-DLM contribution now requires replicated raw improvement; refined-only is labelled refiner-mediated. |
| Two-to-three contributions looked precommitted | Final report must mark each as supported/candidate/unsupported; only lattice and volume receive predicted-rich credit, and frozen diffusion is not automatically a new algorithmic contribution. |

## First Arbiter review

Disposition: **REVISE**. The only unresolved objection was an inconsistency over
whether active rich-field calibration is required before GPU execution. It is
now explicitly part of the single compound operational preflight. The audit has
no outcome-tuned pass threshold: it must be complete and disclosed, while the
later canary supplies causal development evidence.

## Final Arbiter disposition

**APPROVED.** The calibration audit is now unambiguously part of the single
execution preflight, must be completed and disclosed before GPU work, and has no
outcome-tuned threshold. All Skeptic, Constraint Guardian and User Advocate
objections are resolved. Implementation may proceed under the checklist.

## Resolution policy

Each objection must be accepted, rejected with original evidence, or converted
into a pre-outcome contract change. Test outcomes may classify a frozen method
but may not select rich-field subsets, seeds, checkpoints, temperatures or
refiner steps.
