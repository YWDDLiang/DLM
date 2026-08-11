# S.U.N. skip-unknown submission continuation V1

The prepared V2 run completed cache adoption but its first `sbatch` created no
job because a 16-element array exceeded the user's Slurm submit-count limit.
This immutable continuation verifies that there is no partial job or cell
output, then submits one normal job that runs all 16 independent cells in
parallel on 16 CPUs. A second normal job assembles the results after completion.

It does not query MP, modify the prepared cache, rerun generation/refinement,
or change any scientific input or evaluation code.
