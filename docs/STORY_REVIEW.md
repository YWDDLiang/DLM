# Concept-only proposer–reviewer verdict

## Final disposition

> **APPROVED — approximately 7/10 at the concept level.**

This approves the research-question, method and contribution logic. It does
not pre-approve positive empirical claims. Zero or negative mechanism results
remain valid answers and must remove the corresponding method claims.

## Approved main question

> **When different crystal-validity checks can only be evaluated after
> different information has been generated, do restricting invalid choices
> whenever the prerequisite information is available and choosing which
> geometric variables are eligible for commitment at each stage affect how
> reliably a model-proposed composition is realized as a periodic crystal?**

The composition and atom count are held fixed. The scope is the eligible Plan
distribution sampled by the learned source.

## Why the question passes review

- It describes an empirical unknown rather than naming the proposed model.
- It has explicit positive, null and negative answers.
- It leads naturally to partial-state execution, selected support and a
  testable commitment policy.
- It is limited to the three implemented checks and does not claim a general
  constraint theory.
- It separates the primary discrete mechanism from learned-Plan scope and the
  downstream fixed refiner.

## Approved contribution hierarchy

1. **Problem formulation:** prerequisite-dependent selected checks and
   geometry commitment as testable variables in crystal realization.
2. **Core method:** a composition-anchored typed masked executor with selected
   state-conditional support and a grouped confidence-adaptive policy.
3. **Paired evaluation protocol:** Plan-level estimates of support, policy,
   their interaction, heterogeneity and fixed-refiner downstream conversion;
   this becomes an empirical contribution only after results are complete.

The learned condition source and `model_494` have defined roles in the intended
end-to-end route; neither is an algorithmic contribution.

## Strongest rejection

> The method may still be viewed as a hand-designed lattice-to-X-to-Y-to-Z
> policy plus a few local inference masks around a pretrained masked model and
> an inherited refiner. Unless strict paired results show that selected support
> and commitment policy materially affect realization, and that the proposal
> differences remain meaningful after refinement, the problem framing may be
> judged a renaming of constrained-decoding details.

This objection cannot be solved by further prose.

## Defensible boundaries

The paper may claim that:

- selected crystal checks become evaluable under different partial states;
- the masked interface makes commitment policy an explicit inference variable;
- the implemented support and policy can be evaluated under one frozen
  checkpoint with Plan-level pairing; and
- a fixed refiner permits downstream pre/post conversion analysis.

The paper may not claim:

- DLM superiority over AR;
- a universal or learned optimal commitment order;
- general constraint satisfaction or support-consistent training;
- minimum-distance, exact-space-group or Plan-volume guarantees;
- revision or reopening of committed tokens;
- hard execution of soft rich-Plan fields; or
- algorithmic novelty of the Planner backbone or continuous refiner.

## Evidence status

The existing implementation matches the approved conceptual scope, but the
strict mechanism evidence still requires a fixed positional skip-anchor
control, equal model-call budgets, call-indexed paired randomness, stable
Plan/attempt metadata and identical pre/post evaluators. These are experiment
wiring changes rather than checkpoint retraining.

The `105/1000` Strict and `488/1000` Meta future paper results, and the external
CrysLLMGen context, remain end-to-end results. They cannot substitute for the
paired mechanism evidence.
