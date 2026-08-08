# Current state

Updated: 2026-08-08 17:26 (Asia/Shanghai)

Overall status: `PLANNER_SFT_V2_PACKAGING_REPAIR_V3_IN_REVIEW`

The Evidence-First workstream is active on branch
`codex/evidence-first-sun-msun`. No C0/C1, SFT-v2, SFT-v2-C, B3,
integration, or final-evaluation scientific result has been read or created
under this workstream. No training, generation, Direct, refiner, S.U.N., or RL
job has been submitted.

## Connection and read-only audit

- 5090 is reachable only through port 2213 and the configured private key.
- Existing A800 tmux sessions `ssha800` and `ssha800_2` were present with
  `pane_dead=0` and `pane_current_command=ssh` at the last audit.
- A800 access remains restricted to those existing sessions. Neither may be
  recreated or reconnected; if both fail the workstream stops.
- A800 had no user jobs at the initial audit. Every submission still requires
  a fresh `sinfo`/`squeue` snapshot.
- Audit marker: `__EF_AUDIT_DONE__` at 2026-08-08T12:59:52+08:00.

## Frozen evidence already obtained

- Protected P0/B0/model_494 identities remain unchanged.
- Transfer v1 failed before source creation because archived executable files
  contained CRLF. The immutable failure evidence is retained.
- Corrected archive from commit `ef82ffc` has SHA
  `79d1e6e60b06e61e0654ebdafcfad828cb86888b7f17fff9bcbfeae2a97e42b9`.
- The corrected A800 source bootstrap passed under run
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_v1`:
  source inventory SHA
  `f429f63ef42ead9162149ddce135bb35da7fc2ad94d3e86c8135506063f6a801`,
  source archive SHA
  `555fdcc59901aad4bd4ceae685b28827ab3d5b312434706ce5a319d2c815de25`,
  source manifest SHA
  `b0fab254166928df999ffeb81199aeca699904508895fb98f73645687e8faf5d`,
  590 files.
- The attempted A800 portable SMACT4 runtime failed before project tests or
  science because `contourpy-1.3.3` required a newer manylinux tag. Its partial
  build and logs remain immutable; it is not repaired or reused.
- Two independent engineering reviewers converged on the same safe remedy:
  exact SMACT4 must be machine-separated, complete, SHA-bound, and joined
  one-to-one; stale, missing, duplicate, extra or substituted rows must fail.

## User-frozen evaluator split

A800 now uses only its existing Python 3.10/SMACT 3.1 environment. Exact
SMACT 4.0 must never be installed or executed on A800. The earlier foreground
zip-overlay probe was already active when this rule was frozen; it is ignored,
is not a gate, and must not be interrupted under the tmux safety contract.

The replacement identity is `h1_chemistry_first_sft_v2_smact_split_v2`:

1. A800 exports the complete immutable MP20 legacy snapshot using SMACT 3.1.
2. The local machine runs only the explicitly authorized exact-SMACT4 witness
   builder from the same frozen source inventory and returns a sealed
   one-row-per-source ledger.
3. A800 verifies parent/source/ordinal/material/formula/witness hashes and uses
   the exact POS intersection to build data, smoke, train, and generate raw64.
4. Raw64 generation stops. The local machine produces complete exact-SMACT4
   per-arm audit reports; A800 verifies every report against its current raw
   bytes before SMACT3 gate assembly.
5. Raw256 repeats the same generation/local-audit/assembly split only for
   raw64-passing candidates.

No A800 submission contains `SMACT4_PYTHON` or an exact-SMACT4 executable
path. The local exception is limited to witness and raw secondary-audit
production; all model tests, tokenizer checks, data assembly, training,
generation, and primary evaluation remain on A800.
The replacement source also excludes the retired portable-runtime builders,
requirements, and launcher, so the A800 snapshot contains no dormant SMACT4
runtime entry point. The immutable failed runtime run remains preserved as
historical evidence outside this source.

## Split-source audit terminal and minimal repair

The first split-source run reached an engineering terminal before Slurm:

- run root:
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2`;
- source tests `48/48`, isolated tests `48/48`, and preflight-focused tests
  `35/35` all passed;
- the only failed check was `legacy_evaluator_source_sha`;
- archived `composition_validity.py` had CRLF SHA `c078c1ca...`, while its LF
  normalization exactly matched the frozen evaluator SHA `ca1c94f5...`;
- terminal SHA:
  `1fc5776c66c3ba34f9afc991200cec81734a35168092e14fd5da2b633956765d`;
- no Slurm, GPU, training, generation, SMACT4-on-A800, or science occurred.

The allowed repair is packaging-only. It writes the Git archive directly to
its output file, requires LF for source text classes, checks the evaluator
member SHA before extraction, and uses new transfer/staging/freeze/run paths.
The new run root is
`20260808_h1_chemistry_first_sft_v2_smact_split_v2_packaging_repair_v3`.
Following the user's audit-budget instruction, its source gate runs one
protocol test, one isolated inventory/SHA check, and one focused preflight;
the duplicate broad source/isolated test pass is not repeated.

## Immediate critical path

1. Complete static review, commit, and push packaging repair v3.
2. Transfer it after the ten-minute SCP interval, freeze it into the new
   immutable A800 run root, and run only the reduced SMACT3 source gate.
3. Submit only the legacy snapshot job after fresh partition checks.
4. Pull that snapshot, build the local exact-SMACT4 witness ledger, return it,
   and submit data plus minimal GPU smoke only.
5. Submit fixed-endpoint training and raw64 generation only after both smoke
   tasks are `COMPLETED 0:0`; do not queue assembly before the local audit.
6. Continue C0/C1 and the B3 portfolio after this source family reaches its
   registered terminal.

No Planner or DLM RL is authorized.
