# DLM/Planner-conditioned continuous diffusion

This independent branch prepares the third-priority extension while the new discrete DLM and original K8 policies train.

The frozen scientific interface is composition, typed Planner program, a complete DLM proposal and a continuous diffusion state. The model494 decoder receives an explicit zero-initialized residual before its CSP layers. Site features contain periodic proposal-to-current displacement, proposal pair environment, element-independent program rank and noise time. Graph features contain current/proposal lattice metrics and scale/volume changes.

The existing decoder keeps its normal diffusion-time input. The added adapter accepts no relaxed target, energy, force or hull value. Offline targets may later use common verified relaxation, but those values cannot enter deployment conditioning. Step-zero equality preserves model494 exactly; the historical frozen model494 remains the primary comparison.

The current module only establishes the explicit conditional architecture and checkpoint interface. It does not yet claim a trained diffusion model, a valid SDE change of measure, or SUN improvement. The first training protocol will freeze the t=1 model494 mapping, train only t>=2 stochastic denoising transitions, retain original MP20 denoising anchors, and compare on identical DLM inputs/noise seeds.
