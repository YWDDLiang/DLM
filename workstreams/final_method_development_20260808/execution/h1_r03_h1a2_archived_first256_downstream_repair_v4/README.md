# Archived R03 vs H1-A2 first256 downstream repair V4

V3 stopped in preflight and submitted no Slurm job because its audit configuration confused the two-byte `exit_code` file size with the file content. V4 corrects the expected upstream exit-code content from `2` to `1`; its scientific workflow is unchanged.

V4 reuses the V2 body artifacts that already passed the exact archived byte-identity gate. It resolves the refiner helper from the archived refiner runtime, then runs model_494 refine800, Direct, archived frozen-cache S.U.N., validation, and paired assembly in one 1×A800/8-CPU job. It does not rerun body generation, resample plans, rehash the large checkpoint, retry samples, replace failures, filter, rerank, train, or use RL.
