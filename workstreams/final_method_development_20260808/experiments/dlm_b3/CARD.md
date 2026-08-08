# Experiment card: DLM B3

Status: `SPEC_FROZEN_AUDIT_PENDING`

B3 is mandatory after the B0/B1/B2 and state-panel inventory. It starts from
protected B0, preserves the historical two-GPU/optimizer/one-epoch contract and
R5-C prompt/answer bytes, uses `dynamic_v1`, IID:safe-planned 2:1, no mixed-axis
groups, and `z_before_xy=0`. Synthetic and actual-rollout NLL point estimates,
body64 completion, duplicate failures, and new failure classes determine
whether B3-R is eligible. S.U.N. cannot choose its checkpoint.

