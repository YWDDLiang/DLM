# G2 full-epoch periodic-relation A/B final

Date: 2026-09-01  
Scope: matched post-outcome development; fixed256; one frozen Plan/noise stream.

## Outcome

Job39172 completed `0:0` in `01:45:21` on four A800 GPUs, for an observed
`7.0233 A800-h`. All A/B raw/refined generation, Direct, full-reconstructed,
CHGNet and hash gates passed. No sample, Plan, body, failed row, seed,
checkpoint or result was replaced.

| Stage | Arm | Method | Body | Direct | N/U/NU | Hull known | Strict S.U.N. | Meta S.U.N. |
|---|---|---|---:|---:|---:|---:|---:|---:|
| raw | A | G2-PBC-R | 255 | 128 | 254/255/254 | 247 | 7/256 (2.734%) | 41/256 (16.016%) |
| raw | B | G2-PBC-RU | 254 | 130 | 253/254/253 | 245 | 6/256 (2.344%) | 41/256 (16.016%) |
| refined | A | G2-PBC-R | 255 | 254 | 223/255/223 | 248 | 23/256 (8.984%) | 117/256 (45.703%) |
| refined | B | G2-PBC-RU | 254 | 254 | 221/254/221 | 248 | 23/256 (8.984%) | 115/256 (44.922%) |

## Paired B−A effects

| Endpoint | Common known | Mean | Bootstrap 95% CI |
|---|---:|---:|---:|
| raw CHGNet | 251 | +129.19 meV/atom | [-107.06, +362.41] |
| raw official hull | 245 | +128.05 meV/atom | [-112.37, +370.26] |
| refined CHGNet | 254 | +8.10 meV/atom | [-4.59, +24.71] |
| refined official hull | 248 | +8.32 meV/atom | [-4.65, +25.13] |

Raw Strict discordance is A-only/B-only `2/1`; raw Meta `6/6`; refined Strict
`3/3`; refined Meta `15/13`. None favors B.

## Registered decision

The promoted full-epoch implementation is **A / G2-PBC-R**. B neither keeps
body while gaining at least eight Direct outcomes nor establishes a paired raw
energy advantage. This promotion chooses between the two registered
full-epoch mechanisms; it does not replace the earlier fresh prospective G2
headline (`24/117` refined Strict/Meta S.U.N.).

The cached official union omits `Ag-Ca-Pb` and `Co-O-V`. Those rows remain
unknown in the fixed256 denominator and are never counted stable. No new MP
query was issued.

## Immutable artifacts

- Evaluation: `runs/g2_full_ab_refine_eval_39172/_OFFLINE_SUCCESS`.
- Final: `runs/g2_full_ab_cached_official_final_20260901_v2/_SUCCESS`.
- Archive: `archive/c3fd_g2/g2_full_epoch_ab_final_20260901_v1/_ARCHIVE_SUCCESS`.
- Evaluation OUTPUTS SHA: `6ca71897b28d425780e4f4bbe9a5693502c5be4bb7c74a970624bf9a4efcaa00`.
- Final JSON SHA: `b50dd8d291daf46d29ff916e0b34395e7f252a11cbaaa8e99e6378f4a8819881`.
- Final OUTPUTS manifest SHA: `b5b03ae357f07543b49e1d01eeedfa3a597e97a70993340353cae83e00a81a37`.
- Archive manifest SHA: `513864501df488f409e95aca351af026dbafc2dc4f6c5d44f0c07842dddb85d7`.
- Finalizer/tests: commit `fee4c2c`.

Method development is now closed. The next operation is Git consolidation and
paper-mainline construction; no additional scientific experiment is launched.
