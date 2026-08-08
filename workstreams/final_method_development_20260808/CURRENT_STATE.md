# Current state

Updated: 2026-08-08 (Asia/Shanghai)

Overall status: `PLANNER_SFT_V2_SOURCE_CANDIDATE_READY_FOR_A800_AUDIT`

The Evidence-First workstream has begun on branch
`codex/evidence-first-sun-msun`. No C0/C1, SFT-v2, SFT-v2-C, B3, integration,
or final-evaluation scientific result has been read or created under this
workstream.

## Read-only execution audit

- 5090 host: reachable through the configured port-2213 key path.
- Existing A800 tmux sessions `ssha800` and `ssha800_2`: present,
  `pane_dead=0`, `pane_current_command=ssh`.
- A800 user queue at audit time: empty.
- `gpu` and `gpu_long`: available/up; exact allocation is rechecked immediately
  before each submission.
- No installed A800 Python runtime satisfies SMACT 4.0's Python
  `>=3.11,<3.14` contract; discovered Conda runtimes are Python 3.10 with
  SMACT 3.1 or 3.2.
- Frozen C0/C1 remote source and run roots: absent.
- `final_method_development_20260808` remote root: absent.
- `final_paper_closure_20260808` remote root: absent.
- Duplicate submission risk found: none.

Audit marker: `__EF_AUDIT_DONE__` at 2026-08-08T12:59:52+08:00.

## Immediate critical path

1. Transfer the committed source snapshot and frozen portable SMACT4 bundle in
   one SCP batch, then freeze the canonical source on A800.
2. Atomically prepare the run-local SMACT4 runtime and run source, isolated-
   archive, exact-tokenizer, and evaluator preflights on A800.
3. Submit the SFT-v2/SFT-v2-C data/smoke/training/raw64 DAG only after those
   gates and a fresh `sinfo`/`squeue` audit.
4. Freeze and execute C0/C1, then audit B0/B1/B2 and construct B3 state panels.

The chemistry-first implementation includes the common-record data builder,
deterministic base/curriculum orderings, fixed-endpoint trainer, named-PEFT
endpoint resolver, separate exact-SMACT4 audit, and raw64/raw256 gate
assembler. Before the portable-runtime changes, 47 focused local/regression
tests passed. Those changes are intentionally not re-run locally: by explicit
user instruction, all subsequent project programs and tests run on A800 only.

The one-time authorized local runtime exception produced a portable CPython
3.12.13 + 54-wheel bundle. Offline installation passed `pip check`; the probe
reported SMACT 4.0.0, Transformers 4.54.0, 93 oxidation elements, and contract
SHA `ad070f3a...`. Frozen bundle archive SHA is
`4ffac0ce561483fcacbb592cb9287b2e24bb4fbca67217396f7a2743a3de44bc`.
The full 27,136-row census, post-bundle A800 tests, exact-tokenizer audit,
isolated archive test, and A800 smoke remain pending and are not represented
as passed.

Transfer batch v1 reached 5090 only and was superseded before any A800 copy or
source/run creation because static Git-tree review found the frozen SMACT wheel
was excluded by the repository-wide `*.whl` ignore rule. Commit `0ec8940`
force-adds the exact 2 MB wheel; the unused v1 inputs remain immutable evidence
and are never submitted.

Corrected transfer batch v3 is frozen from commit `1b7b51d`: source archive
SHA `e084fbd66093f158662dd4ada3a0c3fad8be4d3008a64aa16bff1acfc2578eb4`
(10,179,180 bytes). A single local-to-5090 SCP completed at
`2026-08-08T14:55:25+08:00`; both source and runtime bundle hashes matched on
5090. The A800 transfer completed and both inputs matched there.

The first A800 bootstrap then failed before creating staging, source-freeze,
runtime, or scientific run paths. Exact byte inspection found CRLF in the
archived shell scripts (`bootstrap_source_on_a800.sh` SHA `b682ff01...`), so
`set -Eeuo pipefail` was parsed with a trailing carriage return. This is a
packaging-only engineering failure. Transfer input v1 is retained read-only;
the bounded v2 repair freezes `*.sh` and `*.sbatch` to LF and changes only the
transfer-input path. All scientific contracts and the first formal run
identity remain unchanged because none was created.

The repair archive is frozen from commit `b6b9ae3`: SHA
`5fbc73259011a34041f101dd5031d800d69a083806fdb63647bf461b9b0d0635`
(10,177,969 bytes). A packaging-only audit found zero carriage-return bytes
across all 369 archived shell/sbatch files and confirmed the exact SMACT4 wheel
is present. Its single local-to-5090 SCP completed at
`2026-08-08T15:16:43+08:00` with matching SHA; A800 transfer input v2 remains
empty pending the required next SCP interval.

No Planner or DLM RL is authorized.
