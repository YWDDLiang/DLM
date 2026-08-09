# Experiment card: DLM B3

Status: `SPEC_FROZEN_AUDIT_PENDING`

B3 is mandatory after the B0/B1/B2 and state-panel inventory. It starts from
protected B0, preserves the historical two-GPU/optimizer/one-epoch contract and
R5-C prompt/answer bytes, uses `dynamic_v1`, IID:safe-planned 2:1, no mixed-axis
groups, and `z_before_xy=0`. Synthetic and actual-rollout NLL point estimates,
body64 completion, duplicate failures, and new failure classes determine
whether B3-R is eligible. S.U.N. cannot choose its checkpoint.

The user-required complete comparison uses the documented best pipeline as a
fixed shell around the DLM factor. At fixed P0, B0 and B3 both run safe-axis,
model_494 refine800, Direct, and S.U.N. with common identities and raw
denominators. A separate SFT-v2+B3 cell measures interaction. Thus the report
must distinguish the Planner main effect, DLM main effect, and joint effect;
historical aggregate metrics cannot substitute for any requested new arm.
