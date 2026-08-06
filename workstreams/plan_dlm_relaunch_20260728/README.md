# Active Plan + DLM relaunch assets

This directory is the active allowlist for the 2026-07-28 ICLR relaunch.

The scientific trunk is:

```text
H1-A2 epoch-2 Planner
  -> frozen R5-C exact-length DLM body
  -> one immutable shared Plan
  -> original CrysLLMGen parent / repaired S2 Plan-conditioned refiner
```

`ACTIVE_RECOVERY_MANIFEST.json` is authoritative for model identities, remote
paths, source roles and prohibited branches.  The portable active source bundle
and the inclusive historical archives are stored under:

```text
archive/20260728_plan_dlm_relaunch/
```

The immediate local implementation is a single-trajectory exact-null protocol:

- existing FiLM/lattice post-transform masks remain required;
- `force_null` must route the frozen parent decoder;
- parent and exact-null consume one reverse execution and share one immutable
  result;
- matched and shuffled conditions use the same draft and reverse-noise
  manifest;
- no training or GPU submission is authorized by this directory.

The new conditional mechanism panel is already frozen:

- 256 eligible R5-C drafts;
- zero source-position overlap with the earlier observed gold-gate 256;
- selection seed `20260728`;
- 245 shuffled-control eligible rows and 171 rows with a genuinely distinct
  lattice-family intervention;
- lock:
  `NULL_REPAIR_SELECTION_LOCK.json`.

The prepared execution uses the absolute
`/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python`, one A800 and
four CPU cores.  It runs no S.U.N., CHGNet, API, MLIP or training stage; those
remain downstream of this causal mechanism gate.
