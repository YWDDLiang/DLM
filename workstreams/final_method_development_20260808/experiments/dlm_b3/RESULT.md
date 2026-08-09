# Result: DLM B3

Status: `B0_B1_B2_INVENTORY_COMPLETE_B0_V5_PANEL_FROZEN_B3_V4_TRAINING_RUNNING`

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
training setting. It uniquely submitted two-A800 `gpu` training job `31330`
and dependent frozen-panel scorer `31331` (`afterany:31330`); submission-record
SHA is `d1c78cbed83863498640bd2f48f7db1c6458ce26554050065e7424fa244a21dc`.
Training is running on node99 while the scorer waits on dependency. The B0
initialization, R5-C bytes/order/seeds, `dynamic_v1`, IID:safe-axis 2:1
sampling, 1,696 updates, LR `5e-5`, terminal-checkpoint-only contract, and
batch-one panel score remain unchanged. No body64, ratio sweep, downstream,
S.U.N., or RL job was auto-submitted.
