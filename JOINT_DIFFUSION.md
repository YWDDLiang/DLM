# DLM/Planner-conditioned continuous diffusion

**Status: paused before any data preparation or GPU work.** The user directed
that original diffusion training and its role in the paper be established
before considering architectural changes. The prototype below is retained only
as unexecuted code; it is not part of the current method.

This independent branch prepares the third-priority extension while the new discrete DLM and original K8 policies train.

The frozen scientific interface is composition, typed Planner program, a complete DLM proposal and a continuous diffusion state. The model494 decoder receives an explicit zero-initialized residual before its CSP layers. Site features contain periodic proposal-to-current displacement, proposal pair environment, element-independent program rank and noise time. Graph features contain current/proposal lattice metrics and scale/volume changes.

The existing decoder keeps its normal diffusion-time input. The added adapter accepts no relaxed target, energy, force or hull value. Offline targets may later use common verified relaxation, but those values cannot enter deployment conditioning. Step-zero equality preserves model494 exactly; the historical frozen model494 remains the primary comparison.

The current module only establishes the explicit conditional architecture and checkpoint interface. It does not claim a trained diffusion model, a valid SDE change of measure, or SUN improvement. No dataset builder, trainer, Slurm job, or deployment change is active.

The verified original training semantics are:

- Original MP20 clean structures are the data distribution.
- Each batch samples a time in 1..1000. Lattice matrices receive Gaussian
  noise; fractional coordinates receive wrapped-normal noise.
- The model regresses lattice noise and the normalized negative wrapped score,
  using two MSE terms. It does not read energy, force, relaxation, A/B, Planner
  state, or a DLM error.
- The bundled trainer uses Adam at 0.001, value-wise gradient clipping at 1,
  and ReduceLROnPlateau. A checkpoint name such as model494 does not by itself
  establish its exact data snapshot or selection history.
- At deployment, the existing sampler uses a DLM proposal directly as the
  t=800 state. This is not the forward noising marginal used in training, so
  standard denoising theory alone does not prove refinement behavior.

The current paper story therefore keeps the new discrete DLM as the learned
proposal/repair contribution and frozen model494 as a separately scored
continuous post-processing reference. A joint extension can be reconsidered
only after the discrete results, with its own data and attribution.
