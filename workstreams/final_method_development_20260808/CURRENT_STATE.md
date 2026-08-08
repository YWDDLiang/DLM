# Current state

Updated: 2026-08-08 18:39 (Asia/Shanghai)

Overall status: `PLANNER_SFT_V2_SOURCE_GATE_PATH_REPAIR_V5_PREPARING`

The Evidence-First workstream is active on branch
`codex/evidence-first-sun-msun`. No C0/C1, SFT-v2, SFT-v2-C, B3,
integration, or final-evaluation scientific result has been read or created
under this workstream. The sealed data passed, while the first minimal GPU
smoke reached a repairable engineering failure before model forward; no formal training, raw generation,
Direct, refiner, S.U.N., or RL job has been submitted.

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

## Packaging repair v3 and evaluator-split inputs

The packaging-only v3 repair passed its reduced A800 source gate and is now
the immutable active run root:

- run root:
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_packaging_repair_v3`;
- source-input archive SHA:
  `d64bf8790fe9a2df4926e25e045883f0f6077d2b8c1d8c86d73b3b3bb7c8f0cf`;
- source inventory SHA:
  `410fb34d2543f620fff012fde574a75a935bcaf54e41449fb529c12eb913c20c`;
- frozen source archive SHA:
  `3c5d93f697e97e29ee233bcefab41c8636599711e6decaf971eeb383f8137559`;
- source manifest SHA:
  `50f9e541a8fcafe13a4d953b1907f949259be4564bffd576379db72fcdbcf89a`;
- archived legacy evaluator member SHA:
  `ca1c94f583e0c97a172b5c9b7ba96505257fd74dedfc618b584c34486ac1f178`;
- the one protocol test, isolated inventory/evaluator identity check, and
  focused 35-test preflight all passed; A800 used Python 3.10.18 and
  SMACT 3.1.0, and did not import or execute SMACT 4.

Legacy snapshot job `31025` completed `0:0` in `00:03:38`. Its sealed bundle
SHA is `d65a682c8e29820938a8e4637963dd56f18f2f6d893abd77add02043fb6a13bf`
and report SHA is
`351a13f8c462a9a0ee377c1d07ebaafb426a0b64cffc2c9c3ad950725e3deefb`.
It contains exactly 27,136 train, 9,047 validation, and 9,046 test rows.

The explicitly authorized local-only exact-SMACT4 witness build also passed.
Its persistent isolated runtime terminal SHA is
`0853bbea0c714f8a3150b08f3c94bdf1ec03b9cd49585d3a16533a9f88ea67d6`.
The witness manifest SHA is
`d21698e29664c607541d7ab644250e93e18cfb2a0cd03d1687a270b42c8ccd32`,
ledger SHA is
`ab687c5f16dc64887de2446c3aae20c0e15021a3bfa4ffe71dd6cfe09b482c93`,
and deterministic archive SHA is
`b896516351a76b869cc10d8c95a321ee53bb1cf25e3ed38d207d1dd40b7322c3`.
The train census is 7,079 legacy-primary and 4,451 exact uniform-primary;
validation is 2,264 and 1,406 respectively. Official/witness parity is true.
The sealed ledger was transferred and frozen as data bytes only; SMACT 4 was
not run on A800.

Data job `31035` completed `0:0` in `00:15:47`. Both candidates contain
36,038 train and 9,047 validation records. The record multiset is identical;
base and curriculum order SHAs differ as registered, the 3,603-record
curriculum prefix alternates correctly, and all data/tokenizer audits pass.
There are zero invalid unconditional formula targets, generated charge-field
leaks, token-weight failures, or truncation failures. Frozen optimizer
geometry is 4,505 updates, 135 warmup steps, and six microbatches in the final
accumulation group. Audit-report SHA is
`1c9a5e2cba51a1258acf107255ca308c7c9f7122d50f5ae3a09ed97d2681612a`;
order-ledger SHA is
`c31df9dd44bbe1ea75a99131899d4e6ea7131d1f230b39ecfb25f730d5406a43`.

Minimal smoke array `31036_[0-1]` reached `FAILED 1:0` for both tasks after
`00:13:08`. It loaded the base model but stopped before the first model forward:
all 448 candidate/reference tensor keys matched and no noncandidate parameter
was trainable, but PEFT 0.16 had loaded the trainable candidate copy in FP32
and the frozen reference copy in the BF16 base dtype. The resulting maximum
absolute difference was `6.103515625e-05` for both candidates. No optimizer
step, formal training, scientific generation, or SMACT4-on-A800 occurred.

The frozen V4 repair loads both copies through the same protected-P0 FP32 PEFT
path, then freezes every reference parameter before forward or optimizer
construction. It changes no model payload, data, task order, prompt, seed,
optimizer, ledger, evaluator, or gate. The V3 data/legacy snapshot will be
reused byte-for-byte under a new manifest; a fresh dual-arm GPU smoke remains
mandatory before training.

V4 source freezing itself completed with source inventory
`2e7997cfc9894db5ba099d4fc3bfa18440b10d0c29e36927768dc35eaef03968`,
archive `193816fb1aa6919f0bed755f5fae9459f7a0bd63afc797b579dc741f43cb8b73`,
and manifest `02fcb50da21bd54e314aeaa0e09cf2b9cbb71300f24a9dddeaca009f01beffc1`.
Before executing its A800 source gate, a static check found that the isolated
archive path was still bound to the existing immutable V3 directory. The gate
was not executed, no pass marker was written, and no job was submitted.

Two independent propose/red-team reviews agreed that deleting or reusing V3,
hand-writing the pass marker, or changing only one path would be indefensible.
The active repair is therefore a new path-only V5 identity:
`20260808_h1_chemistry_first_sft_v2_smact_split_v2_source_gate_path_repair_v5`.
Every active script/SBatch/config path moves to V5 and isolated extraction is
scoped under `${RUN_ROOT}/isolated_archive_test`. V4's PEFT repair, all model
and science code, data, prompts, seeds, optimizers, ledgers, evaluators, and
gates remain unchanged. SMACT4 remains forbidden on A800.

## Immediate critical path

1. Preserve V3 job `31036` plus V3/V4 source evidence unchanged.
2. Freeze, transfer, and minimally verify the V5 path-only source plus byte-
   identical parent-data reuse record; submit only the replacement dual-arm
   smoke.
3. Submit fixed-endpoint training and raw64 generation only after both V5 smoke
   tasks are `COMPLETED 0:0`; do not queue assembly before the local audit.
4. Continue C0/C1 and the B3 portfolio after this source family reaches its
   registered terminal.

No Planner or DLM RL is authorized.
