# WTB-256 job28194 pre-scientific Gate-A failure

## Outcome

The one authorized WTB-256 evaluation-only submission was accepted as Slurm
job `28194`, received the intended `1×A800 + 8 CPU + 96 GiB` allocation on
`node99`, passed the installed-source and exact `diff_meets_diff` CUDA
environment checks, and then stopped after eight seconds.

This is **not a scientific failure**. The run produced zero source attempts,
zero R/U/T samples, zero CrysLLMGen evaluations, zero S.U.N evaluations, and
zero training attempts.

The immutable terminal audit is:

`runs/remote_audit/20260727_wq_wyckoff_chart_retraction_confirmatory256_job28194_terminal_failure_v1.json`

## Exact failure

The first scientific stage was announced as
`pipeline_stage=freeze_wq_sources_256`. Before writing any source artifact,
`GateALock.load(...)` raised:

```text
ValueError: authorized execution patch changed after install
```

The detailed Gate audit resolves the generic exception to:

```json
{
  "ok": false,
  "errors": ["patch_authorization"],
  "changed": [],
  "registered_files": 150
}
```

All 150 installed files remained byte-identical. The base source bundle SHA
also matched. The mismatch is purely between two authorization registries:

- `scripts/a800/install_authorized_patch.py` accepts
  `user_wq_wyckoff_chart_retraction_confirmatory256_v1_local_preparation_2026-07-26`.
- `crystal_dlm/wqcodiff/crysllmgen/gate.py` does not include that same label in
  `PATCH_ALLOWED_AUTHORIZATIONS`.

The installer therefore correctly installed the archive, while the runtime
Gate-A audit rejected the install record's otherwise valid authorization
label.

## Why pre-submission tests did not catch it

The final remote regression suite passed all 13 selected test files, but its
coverage had a registry-consistency hole:

1. The WTB-256 test checked that the installer contained the new label.
2. Gate-A tests checked prior labels, not the new WTB-256 label.
3. No test asserted equality between the installer and Gate-A authorization
   registries.
4. The final read-only preflight verified installed bytes, resources, hashes,
   imports, and absence of an existing claim, but did not call
   `GateALock.load(...)` with the new patch identity.

## Preserved evidence

- Submission record SHA:
  `e660623c3fcdb606667379a1d3659d16defdfda8340f64cce3ebb68cc54399a7`
- Claim SHA:
  `a7870e476ddb2d69dfb9a517bbb6d811c64a4021695ae9cab6c5acf82223caea`
- Stdout SHA:
  `5790afb3a6c6d372e1429348b95b0d8e350ce663c7d0d5fe4d4ba404fa380ace`
- Stderr SHA:
  `8100c391370939d00edc497be6fad497ca576a3a771d400a787148ac3ce2c723`
- GPU CSV SHA:
  `faa3fc31e2849bd3785ca113edda9b4d0712146ddcf1a93c3aea4dd19c200ef8`
- Installed patch record SHA:
  `56d281873d2def3f6bf18d7974482c339349af848a64bf6c73f8c5a23397b4b4`

The fixed output directory exists but contains zero regular files. Nothing
will be deleted, overwritten, retried, or repurposed.

## Safe next step

A future supersession requires a new explicit authorization. The minimum
change should be an audit-sidecar correction, not a scientific redesign:

1. Register the exact WTB-256 label in Gate-A.
2. Add a test for that label and a general installer/Gate registry parity
   test.
3. Add a pre-claim `GateALock.load(...)` check so this class of error consumes
   no GPU allocation.
4. Build a new cumulative patch and immutable submission identity.
5. Preserve job `28194` and all evidence.

No further Slurm claim, evaluation, or training is authorized by the consumed
one-shot permission.
