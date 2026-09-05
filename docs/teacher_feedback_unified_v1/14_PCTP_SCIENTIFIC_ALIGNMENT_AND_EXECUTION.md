# PCTP reward design: superseded record

Status: **rejected before implementation; no GPU job or checkpoint exists**

PCTP proposed assigning a fixed tau800 terminal S.U.N./energy reward to a
complete SPAD trajectory and using group-relative policy optimization to update
the DLM and then the Llama species pointer.

The proposal corrected one real implementation defect in the retired K10 path:
K10 normalized only inside a small candidate set and therefore did not increase
the absolute deployed probability of preferred actions. Nevertheless, PCTP is
not retained as the next paper method.

## Why it was rejected

1. Terminal-reward optimization of diffusion trajectories is established by
   DDPO, Diffusion-DPO, D3PO, and related group-relative methods. Applying that
   recipe to the current crystal pipeline is system adaptation, not a strong
   new central contribution.
2. A post-tau800 reward primarily trains basin entry under a frozen refiner. It
   does not by itself show that the raw DLM learned periodic geometry or native
   stability.
3. Joint program/trajectory reward makes Llama-versus-DLM credit difficult to
   identify unless the two policies are updated sequentially with expensive new
   rollouts.
4. The route would require two large terminal-rollout collections and would
   repeat the broad scientific object of the historical, non-robust
   post-refiner D3PO experiment.
5. The advisor-facing story must center on a crystal-DLM mechanism, not on a
   generic reward algorithm.

PCTP remains a useful negative design record. The approved successor is
[Programmed Manifold-to-Token Repair](15_PMTR_SCIENTIFIC_METHOD_AND_EXECUTION.md),
which changes the DLM's native repair mechanism and contains no reward/RL stage.

