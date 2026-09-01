# ICLR paper story

## Title candidates

1. **From Chemical Support to Periodic Realization: Science-Constrained
   Language Diffusion for Crystal Generation**
2. **CrystalBridge: Coupling Mechanism-Constrained LLM Planning with Periodic
   Relational Denoising**
3. **Planning Chemistry, Denoising Geometry: A Hierarchical Crystal Language
   Model**

## One-sentence thesis

Scientific crystal generation becomes reliable when chemical feasibility is
embedded in the LLM's action distribution and carried through an exact Plan
interface into a DLM that reasons over periodic relations during denoising.

## Abstract draft

Generating crystals with language models requires more than syntactically
valid CIF text: a model must choose chemically reachable compositions and
realize them as coupled periodic geometries. We introduce a science-constrained
hierarchical crystal language model that unifies these decisions across scales.
A typed C3FD support is fused inside a pretrained Llama Planner, allowing the
language model to reweight chemically reachable actions rather than filtering
completed samples. The resulting Compact-V2 Plan conditions a masked crystal
diffusion language model through an exact variable-length `7+4N` interface.
Within denoising, a zero-initialized periodic relation residual reconstructs
triclinic minimum-image, species-pair and coordination interactions from soft
token states and feeds them back into crystal logits. A fixed terminal
diffusion transition completes coarse-to-fine realization. The constrained
Planner achieves perfect composition validity in 2000-sample and independent
1200-sample evaluations while exhibiting substantial Llama-induced action
reweighting. On a fresh prospective cohort, periodic-relational denoising
improves refined Strict/Meta S.U.N. from `19/111` to `24/117` and lowers paired
official hull by `16.43 meV/atom`; a full-epoch implementation raises raw
Direct validity from `118` to `128` of 256. These results show that scientific
constraints and language generation need not be competing paradigms: when
connected through exact interfaces and relational denoising, they form a
single auditable generator for chemically valid and physically realizable
crystals.

## Introduction arc

1. **Language is attractive but flat.** Crystal text offers a universal
   generative interface, yet composition and geometry obey different laws and
   operate at different scales.
2. **Post-hoc validity is the wrong abstraction.** Filtering cannot recover
   probability mass spent on unreachable chemistry and does not teach the
   body model why periodic coordinates interact.
3. **Scientific information must enter twice, through one state.** It should
   constrain the LLM's global decisions and condition the DLM's local
   realization without fragmenting the generator.
4. **Our answer is a coupled hierarchy.** C3FD-supported Llama planning,
   exact Plan-conditioned crystal language, and periodic-relational denoising
   solve the three failure modes in one probabilistic flow.
5. **The evidence follows the hierarchy.** Composition validity tests the
   Planner, raw Direct tests the DLM, and refined hull/S.U.N. tests the complete
   coarse-to-fine generator.

## Method narrative

Do not introduce the method as four sequential modules. Introduce one latent
Plan model, then unfold how each transition receives the scientific state:

```text
scientific support × learned LLM prior
                 ↓
          global Plan z
                 ↓ exact train/serve contract
       masked crystal language state q0
                 ↕ periodic relational residual
             raw crystal x0
                 ↓ fixed terminal diffusion
            final crystal x*
```

The key rhetorical move is that C3FD and G2 act at complementary ends of the
same Plan-conditioned distribution: one shapes what crystals may be requested,
the other shapes how that request is geometrically realized.

## Headline claims

- **Scientific constraints can be native to an LLM generator.** C3FD support
  is inside the action distribution, and measured KL proves Llama still makes
  learned choices.
- **A Plan can be an exact neural interface, not prompt decoration.** `7+4N`
  turns global chemical intent into a cardinality-correct diffusion language.
- **Periodic geometry must be relational during denoising.** G2 improves raw
  realization and the complete system's official stability without changing
  inference multiplicity.

## Figure plan

1. **Figure 1 — Coupled hierarchy.** Render
   `docs/paper/architecture_mainline.mmd`; color the shared Plan state across
   Planner, DLM and terminal diffusion.
2. **Figure 2 — Science-Constrained LLM action.** Show a partial composition,
   C3FD reachable support, Llama reweighting, and one sampled action. Include
   action KL and comp_valid.
3. **Figure 3 — Periodic-relational denoising.** Show q0 soft lattice/sites,
   125-image species relation graph, zero-init residual and q1 logits.
4. **Figure 4 — Evidence by scale.** Composition validity, raw Direct, official
   hull ECDF and Strict/Meta S.U.N. in one aligned panel.

## Table plan

- **Table 1:** main comparison with composition-valid, raw/refined Direct,
  N/U/NU, Strict and Meta S.U.N.
- **Table 2:** Planner ablations—C3FD predecessor, v2.5, fused Llama.
- **Table 3:** geometry ablations—BASE, G1, G2, A, B.
- **Table 4:** reproducibility—data, seeds, checkpoints, hashes and compute.

## Reviewer-facing answers

**Is this merely constrained enumeration?** No. C3FD defines the reachable
support, while Llama reweights every typed action and predicts global
structural hints; nonzero action KL directly measures the learned LLM effect.

**Is the Plan just extra metadata?** No. It determines body length, exact
composition and global geometric conditioning, and its serializer is identical
at training and inference.

**Is G2 post-processing?** No. It transforms uncertain q0 distributions and
changes q1 token logits inside every DLM denoising step; model494 is the later
fixed terminal transition.

**Are gains caused only by the refiner?** Raw Direct isolates learned DLM
realization, while paired refined hull and S.U.N. measure the complete system.
The paper reports both because they answer different scientific questions.

## Positioning

The paper is fundamentally a DLM contribution. The Planner supplies a
scientifically valid, learned conditioning state; the `7+4N` crystal language
and periodic relation residual determine whether the masked DLM can realize
that state. The final story is therefore not “LLM plus diffusion,” but a
language hierarchy whose global and local distributions are constrained at the
scientific scale where each constraint is meaningful.
