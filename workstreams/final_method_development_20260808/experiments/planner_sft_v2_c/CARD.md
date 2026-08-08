# Experiment card: Planner SFT-v2-C

Status: `LOCAL_CONTRACT_TESTS_PASS_SOURCE_FREEZE_PENDING`

Candidate uses exactly the SFT-v2 record multiset, optimizer, update count, and
seed role. Only its deterministic curriculum order differs, as frozen in
`WORKSTREAM_SPEC_V1.md`. It executes regardless of all other Planner results.

The order implementation passes deterministic multiset identity and 10%
direct/aux alternating-prefix tests. Full-data order SHA remains pending.
