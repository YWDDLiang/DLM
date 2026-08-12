# Archived R03 vs H1-A2 first256 downstream repair V3

User-authorized one-time engineering repair after job 31931 proved both fresh body arms byte-identical to the archived successful run but failed before refinement because the body runtime shadowed the refiner runtime `scripts` package.

This immutable bundle:

- reuses the verified V2 body artifacts without body generation or planner sampling;
- resolves `scripts.refine_dlm_with_crysllmgen` exclusively from the archived refiner runtime and verifies that resolution before and inside the Slurm job;
- runs model_494 refine800, Direct, archived frozen-cache S.U.N., validation, and paired terminal assembly for control then candidate;
- submits exactly one 1×A800/8-CPU job;
- performs no retry, replacement, scientific repair, filtering, reranking, training, or RL;
- does not rehash the large model_494 checkpoint, using its registered path/size/mtime identity.
