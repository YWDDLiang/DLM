# Current state

Updated: 2026-08-08 (Asia/Shanghai)

Overall status: `PLANNER_SFT_V2_LOCAL_CONTRACT_TESTS_PASS`

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
- Frozen C0/C1 remote source and run roots: absent.
- `final_method_development_20260808` remote root: absent.
- `final_paper_closure_20260808` remote root: absent.
- Duplicate submission risk found: none.

Audit marker: `__EF_AUDIT_DONE__` at 2026-08-08T12:59:52+08:00.

## Immediate critical path

1. Freeze and test SFT-v2/SFT-v2-C data, ledger, trainer, and evaluator code
   before reading C0/C1 scientific outputs.
2. Freeze execution packages for the already-authored C0/C1 path and both new
   SFT candidates.
3. Audit B0/B1/B2 artifacts and construct the B3 state panels/package.
4. Submit only after source/archive/SCP/runtime/GPU-smoke identity gates.

The local chemistry-first implementation now includes the common-record data
builder, deterministic base/curriculum orderings, fixed-endpoint trainer,
separate exact-SMACT4 audit, and raw64/raw256 gate assembler. A focused
34-test suite (10 new chemistry-first tests plus 24 no-charge regressions)
passes, as does Python bytecode compilation. The full 27,136-row census,
exact-tokenizer audit, isolated archive test, and A800 smoke remain pending and
are not represented as passed.

No Planner or DLM RL is authorized.
