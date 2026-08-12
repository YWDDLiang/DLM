# H1-A2/R03 V8 official finalization continuation V5

This immutable continuation preserves the failed combined run and reuses its nine completed post-model494 preliminary cells plus its completed official MP cache. It performs only input freezing, preliminary assembly, cache adoption, and official S.U.N. finalization.

The parent Slurm job completed all nine CUDA/CHGNet cells, then failed before input freezing because Bash expanded an empty array under `set -u`. This continuation replaces that orchestration branch with two explicit collector calls. It submits no Slurm job, performs no generation, diffusion refinement, preliminary evaluation, MP query, training, RL, retry, replacement, repair, filtering, or reranking, and never evaluates pre-refine structures.

The parent run remains untouched as failure evidence. Reused artifacts are copied as same-filesystem hard links into a new run root, then audited before any finalization output is written.
