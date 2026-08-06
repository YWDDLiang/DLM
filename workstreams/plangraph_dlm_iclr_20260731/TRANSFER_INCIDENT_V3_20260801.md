# V3 preflight source-transfer incident — 2026-08-01

Status: `resolved_by_user_restore_and_scp`

Scope: transfer of the frozen V3 CPU data/tokenizer-preflight source archive
through the user-maintained nested `ssha800:1.0` tmux pane.

## Evidence

- Local archive before transfer:
  - path: `/tmp/plangraph_dlm_v3_h1a2_preflight_20260801.tar.gz`
  - SHA-256:
    `f9c659d2245c2b07559cd8568303cebae3d303d9817d6c6f0d79314db2ac93ac`
  - bytes: `107698`
- The local-to-outer transfer completed with the same SHA-256 and byte count.
- During the outer-to-nested paste, terminal EOF overtook pasted base64 input.
  The input was consequently interpreted by the nested login shell and the
  trailing EOF logged that shell out.
- The nested pane returned to an outer-host shell; it no longer has
  `pane_current_command=ssh`.

## Impact

- No Slurm job was submitted.
- No registered run root was created or populated.
- No H1 asset, model, dataset, seed ledger, denominator, threshold, or
  scientific result was modified.
- Any partial file is confined to an unregistered temporary `/tmp` path and is
  not admissible execution evidence.

## Frozen recovery

After the user restores `ssha800:1.0` and its pane again reports
`pane_dead=0` and `pane_current_command=ssh`:

1. transfer the archive using a fixed-byte-count reader, without terminal EOF;
2. verify the exact SHA-256 and byte count on the nested host;
3. create the new preflight run root and unpack the archive;
4. run `sha256sum -c` against the frozen source manifest;
5. submit exactly one authorized CPU materialization/tokenizer-preflight job.

No automatic reconnect or direct A800 connection is permitted.

## Resolution

- The user restored `ssha800:1.0` and explicitly authorized `scp`.
- Both outer and nested panes were rechecked as
  `pane_dead=0`, `pane_current_command=ssh`.
- `scp` transferred the frozen archive; nested SHA-256 and byte count matched
  exactly.
- The source manifest passed in the new run root.
- CPU preflight job `29318` was submitted once; no blind resubmission occurred.
