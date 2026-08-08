# Experiment card: Planner SFT-v2

Status: `LOCAL_CONTRACT_TESTS_PASS_SOURCE_FREEZE_PENDING`

Candidate starts from protected P0 and uses the common SFT-v2 record multiset
defined in `WORKSTREAM_SPEC_V1.md`. It uses deterministic hash-shuffle order,
one complete ledger epoch, fixed endpoint selection, and the registered
raw64/raw256 literal-positive gates. It executes regardless of C0/C1 results.

Implemented source: `crystal_dlm/h1_chemistry_first_sft.py`,
`scripts/build_h1_chemistry_first_sft_data.py`,
`scripts/llama_h1_chemistry_first_sft.py`, and the dedicated audit/gate
assemblers. Full-data evidence is pending.
