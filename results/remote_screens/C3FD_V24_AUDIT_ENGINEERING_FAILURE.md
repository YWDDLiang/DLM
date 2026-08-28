# C³FD-v2.4 CPU audit — engineering NO-GO

Date: 2026-08-28

The bitset implementation still exceeded its frozen 60-second exhaustive
all-strata audit limit (126 seconds observed, about 856 MB RSS), so v2.4 did
not authorize GPU work. Separately, the constructive certificate covered
`24,558/24,558` train and `8,159/8,159` validation benchmark-supervised teacher
trajectories in 1.95 seconds.

No chemistry or effect gate is relaxed. The separately preregistered v2.5
candidate tests actual learned proposals through a fixed 32-request-per-seed
canary before automatically extending the same checkpoints to requested 256.
