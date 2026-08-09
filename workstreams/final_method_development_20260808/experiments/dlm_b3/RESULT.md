# Result: DLM B3

Status: `B3_V4_TERMINAL_MIXED_NO_PROMOTION_FOUR_CELL_EVALUATION_AUTHORIZED`

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

The user-authorized complete same-pipeline matrix still proceeds independently
of promotion: P0+B0, SFT-v2+B0, P0+B3, and SFT-v2+B3 will each produce new
256-attempt safe-axis + model_494 refine800 + Direct + frozen-cache S.U.N.
evidence. Historical summaries cannot substitute for a current-run cell.
