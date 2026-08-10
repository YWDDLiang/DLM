# Result: DLM B3

Status: `B3_R03_RAW_REPEATS_TERMINAL_STRICT_SUN_REGRESSION_NO_PROMOTION`

The immutable artifact inventory is complete without rehashing the protected
6.39 GB payloads. B0 remains
`5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d`.
B1 and B2 are the historical one-epoch/1,696-update endpoints with adapter
SHAs `ace1a0d1...bea8c` and `e3451c94...f8cb8`, respectively. Their common
100-row fixed-panel NLLs changed from about 1.969803 to 1.460710 (B1) and
1.466090 (B2).

B2 remains a closed scientific stop: dependency margins were 0.259809 (B1)
and 0.233067 (B2), so B2-B1 was -0.026741 with paired-bootstrap 95% interval
[-0.055592, 0.001986]. This inventory does not revive B1 or B2 and does not
select B3. IID, D1, synthetic safe-axis, and actual-rollout state panels are
the next required evidence boundary.

B0-v4 job `31308` produced 1,773 synthetic states, 2,208 actual-rollout
states, and 64 actual attempts, then failed with maximum producer/rescore NLL
delta `0.01802229881286621`. Read-only comparison with the successful R03
rollout and historical B1/B2 panel established that this was a BF16 batch
geometry mismatch: producer rollout batches were at most eight, whereas the
rescore regrouped serialized states by length.

B0-v5 changed neither protected B0 nor state-generation policy. It records
producer batch membership/order, replays each actual state in that exact
batch under the unchanged `5e-4` identity tolerance, and scores all frozen
B0/B3 states at the historical fixed-panel batch size one. Source commit is
`2803439`, archive SHA is
`ce94d79311c468d7f3aa11a881cce1f9bcd2ee0e51732fa09d9bc68434d1c3c5`,
and gpu job `31323` completed `0:0` on node99. Producer replay passed with
maximum, mean, and p95 absolute deltas all `0.0`; panel-manifest SHA is
`6cc3d81074a3e472b39c93090d3d4a85c6565d92eb7eb3c5c18a57cf9f966937`.
The frozen census is IID 100, D1 600, synthetic safe-axis 1,073, and actual
B0 rollout 2,208 states. B0 token-weighted mean NLL is 2.1609953261,
2.0610669718, 2.1251216630, and 4.6503425199 respectively.

B3-v2 failed before SBatch because its strict source adapter expected the old
run-root once in each sbatch file; both frozen files contain it three times
(stdout, stderr, and `RUN_ROOT`). No training or scorer job was submitted.
V3 commit `fa405b9` changed only those two identity-count expectations from
one to three and passed all source identities, then stopped before SBatch
because the A800 tar lacks `--sort=name`. V4 commit `3b5d775` reuses the
successful B0 `tar -czf` compatibility pattern and changes no science or
training setting.

Two-A800 `gpu` training job `31330` completed `0:0` in `00:56:59` at exactly
1,696 updates. Its dependent batch-one frozen-panel scorer `31331` completed
`0:0` in `00:10:42`. Submission-record SHA is
`d1c78cbed83863498640bd2f48f7db1c6458ce26554050065e7424fa244a21dc`;
training-terminal SHA is
`1f9ab27d4d2286f55088448af2940387d9e0c55e7757a4a64f695034fa5da514`;
score-terminal SHA is
`755b5b8687ea117aaf491b298f52bf15ce22594fc905468ccb7faadfff394143`.
The terminal adapter SHA is
`ab4f3b82dfcafd0d111bc7ee424ff08ea0932a1a1466beaf91539917922bc7`.

| frozen panel token-weighted mean NLL | B0 | B3 | B3-B0 |
|---|---:|---:|---:|
| IID | 2.1609953261 | 1.8843846605 | -0.2766106656 |
| D1 | 2.0610669718 | 1.7240345461 | -0.3370324257 |
| synthetic safe-axis | 2.1251216630 | 1.7615879962 | -0.3635336669 |
| actual B0 rollout | 4.6503425199 | 4.8122329398 | +0.1618904199 |

B3 transfers in the desired direction on all three synthetic/fixed panels
but regresses on the actual protected-B0 rollout states. The registered
requirement that both synthetic safe-axis and actual-rollout point estimates
improve is false. B3 is a mixed diagnostic terminal, not a promoted body
checkpoint; no ratio sweep, automatic downstream, or RL was launched.

The user-authorized complete same-pipeline matrix also completed independently
of promotion. It used new 256-attempt cells throughout; no historical summary
substituted for a current-run arm.

| cell | comp/joint | structure | COV-P | COV-R | novel-unique | strict S.U.N. | meta S.U.N. |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0+B0 (M00) | 216/256 | 247/256 | 96.4844 | 85.2200 | 219/256 | 21/256 | 116/256 |
| SFT-v2+B0 (M10) | 214/256 | 246/256 | 96.4844 | 67.5768 | 207/256 | 13/256 | 91/256 |
| P0+B3 (M01) | 223/256 | 252/256 | 98.4375 | 83.2854 | 223/256 | 19/256 | 119/256 |
| SFT-v2+B3 (M11) | 213/256 | 247/256 | 96.4844 | 71.6781 | 210/256 | 16/256 | 102/256 |

At fixed P0, B3 improves composition/joint by 7/256 (+2.734375 pp;
paired-bootstrap 95% CI [0.78125, 4.6875] pp; exact McNemar p=0.015625),
but strict S.U.N. changes by -2/256 and meta S.U.N. by +3/256, neither
significant. At fixed SFT-v2, B3 changes composition/joint by -1/256,
strict S.U.N. by +3/256, and meta S.U.N. by +11/256, again without a
promotion-grade interval. The joint M11 endpoint is below protected M00 on
composition, novelty, strict S.U.N., and meta S.U.N. B3 remains a mixed,
unpromoted diagnostic endpoint; no ratio sweep, checkpoint reselection, or RL
is authorized.

Array `31374` and assembler `31375` both completed `0:0`. Terminal report SHA:
`cdd23113f86e97c5f747e7c97cf24a531231d68b32420cdf03909d8de2806fb6`.

## Frozen R03 raw-Plan contrast

The requested fixed-Plan test removes the Planner change from the contrast.
It byte-freezes R03 P0 raw first256 and compares B3 against the historical
R03G protected-B0 control under D2 safe-axis, model_494 refine800, and R03E
S.U.N. Historical B0 was not rerun. Array `31549` supplied four independent
A800 process repeats of the same 256 paired ordinals; assembler `31550`
completed `0:0`. Every raw failure remains in the denominator, and there is
no retry, replacement, repair, filter, rerank, MP query, or RL.

| repeat | B3 Direct comp/struct | B0→B3 novel | B0→B3 unique | B0→B3 strict | B0→B3 meta | B0→B3 hull evaluated |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 216/251 | 227→234 | 248→251 | 28→26 | 122→132 | 218→220 |
| 1 | 216/251 | 224→227 | 248→251 | 31→26 | 123→120 | 215→214 |
| 2 | 216/251 | 226→225 | 248→251 | 29→24 | 125→127 | 217→213 |
| 3 | 216/251 | 227→228 | 248→251 | 29→23 | 126→125 | 218→215 |

Inference is repeat-aware: exact McNemar is reported separately per repeat,
and the registered 50,000-draw hierarchical paired bootstrap resamples
repeat blocks and then paired ordinals. The pooled 1,024 rows are only a
descriptive aggregation.

| paired B3-B0 endpoint | Δ percentage points | hierarchical 95% CI | repeat count differences | conclusion |
|---|---:|---:|---:|---|
| generation complete | +1.171875 | [0, 2.34375] | +3,+3,+3,+3 | small completion gain |
| unique representative | +1.171875 | [0, 2.34375] | +3,+3,+3,+3 | small uniqueness gain |
| novel / novel-unique | +0.9765625 | [-0.9765625, 2.9296875] | +7,+3,-1,+1 | uncertain |
| meta full S.U.N. | +0.78125 | [-2.24609375, 4.1015625] | +10,-3,+2,-1 | mixed, 2+/2- |
| hull evaluated | -0.5859375 | [-2.24609375, 0.9765625] | +2,-1,-4,-3 | uncertain |
| strict full S.U.N. | **-1.7578125** | **[-3.22265625, -0.390625]** | **-2,-5,-5,-6** | reproducible regression |

For strict full S.U.N., `P(Δ>0)=0.00414` and `P(Δ<0)=0.99352`; the
descriptive pooled count is 117→99 (pooled exact McNemar p=0.0113516).
Meta is 496→504 descriptively but has no repeat-stable improvement. This is
the requested answer from the S.U.N. mouth: B3 slightly improves completion
and uniqueness, but loses strict successes in every repeat and therefore does
not improve the end-to-end scientific endpoint.

Direct composition validity is shown only to document execution (B3 is
216/256 composition-valid and 251/256 structure-valid in each repeat).
`comp_valid` is not a DLM metric, and no causal DLM claim is made from that
number. B3 remains a terminal diagnostic and is not promoted; B0 remains the
protected body. Terminal report SHA:
`101382719310c35f643dfd5b9051834946582058c7a42c0fbd524741b4da6f91`.
