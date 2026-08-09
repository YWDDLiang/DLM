# Result: DLM B3

Status: `B0_B1_B2_INVENTORY_COMPLETE_STATE_PANELS_PENDING`

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
