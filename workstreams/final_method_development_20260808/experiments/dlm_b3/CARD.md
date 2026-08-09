# Experiment card: DLM B3

Status: `SPEC_AND_MINIMAL_IMPLEMENTATION_FROZEN_STATE_PANEL_EXECUTION_PENDING`

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

The common state panel is frozen before B3 training results. Synthetic IID,
D1, and legal safe-axis states use the first 100 validation rows. The actual
B0+safe-axis panel uses the first 64 rows and records every decoder forward,
active group, active width, remaining-mask fraction, commit count/confidence,
visible wrong commitments, and final token error. Wrong committed tokens stay
visible; they are never repaired or remasked. The scoring target is the frozen
ground-truth token only at positions still masked in the active group. Primary
NLL is token-weighted raw-logit cross entropy; state-weighted NLL, top-1
accuracy/confidence, ECE10, Brier score, and group/state distributions are
co-reported. B3 must be rescored on the byte-identical B0-frozen states.

The only training-code extension is policy `d2_safe_axis`: a composition
group, a lattice group, all PlanGraph X groups, all Y groups, then all Z
groups. It reuses the existing stateless mask sampler and loss path. No data,
prompt/answer bytes, model, optimizer, update count, inference decoder, or
checkpoint-selection rule changes.

The frozen execution package directly reuses the completed B1/B2 shell:
2xA800 on `gpu`, eight CPUs total, one full 27,136-row epoch, global batch 16,
1,696 updates, LR 5e-5, cosine/warmup100/min-ratio0.2, the same data and
corruption seeds, and terminal checkpoint only. A dependent one-A800 scorer
uses the byte-frozen B0 state panel. Both jobs explicitly disable automatic
body64, ratio sweep, downstream, S.U.N., and RL. The package performs only
syntax/runtime identity checks needed for training and scoring; it does not
repeat the earlier broad unit-test suite.
