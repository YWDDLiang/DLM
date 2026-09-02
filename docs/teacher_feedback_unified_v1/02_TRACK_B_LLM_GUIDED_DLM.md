# Track B: Llama-Guided Masked Crystal DLM

Status: **design awaiting approval**

## 1. Purpose

Track B tests the paper's main hypothesis: a Llama scientific controller can
tell a masked DLM **what to resolve next** and **which semantic values are
plausible**, while the DLM contributes bidirectional evidence before those
semantic actions are committed.

Track B reuses one exact frozen Track-A controller endpoint:

- C3FD–Llama Plan;
- species program;
- `BodyAdapter-A`, `ProgramHead-A` and `SLA-A`.

It adds the masked DLM as a bidirectional pre-commit executor. It does not assume that Llama
and DLM share token IDs. No Track-B optimization updates any Track-A weight.

## 2. Native DLM representation

For `N` sites, the DLM body remains exactly:

\[
\langle N\rangle,
\langle LA,LB,LC\rangle,
\langle AA,AB,AG\rangle,
\{\langle E_i,X_i,Y_i,Z_i\rangle\}_{i=1}^{N}.
\]

Every field is one dedicated token. The canonical state supplies:

- one exact `N` token;
- all element tokens with the exact Plan multiplicities;
- request-local, transaction-stable slot handles connecting DLM positions to
  Llama program blocks;
- masks only for lattice and coordinates.

The DLM tokenizer and embeddings remain unchanged. Llama native text IDs are
never inserted into the DLM sequence.

## 3. Llama control has three channels

### Global condition

The exact composition and coarse LS/SG/VPA Plan condition every DLM pass.

### Commitment order

The Llama program selects the next species block. DLM storage positions remain
stable; the program chooses which subset of those positions becomes active.
Thus Llama changes denoising order without requiring the two models to share a
serialization.

The fixed dependencies are:

```text
N and exact element inventory
      -> complete six-field lattice
      -> species blocks in Llama order
      -> stable serialization slots within species
      -> propose and atomically commit one XYZ site triplet
```

The Llama chooses only the species-block permutation inside this physical DAG;
the method does not claim a general learned unmasking policy. Crystal semantics
retain lattice first and atomic site closure because PBC validity is not
decidable before those dependencies are complete.

### Local value prior

At the current field boundary, the body Llama produces one SLA distribution
over semantic values. The DLM produces logits over its allowed special tokens.
A deterministic codec maps the latter to the same semantic value domain.

## 4. Semantic fusion instead of token-ID fusion

For active field `f` and semantic candidate `a`:

\[
\ell_f^{B}(a)=
\frac{\ell_{\mathrm{DLM}}^{\mathrm{sem}}(a)}{T_{D,f}}
+\alpha_f(s)
\operatorname{center}\left(
\frac{\ell_{\mathrm{AR}}^{\mathrm{sem}}(a)}{T_{A,f}}
\right).
\]

`T_D` and `T_A` are calibrated by field family. A small agreement gate
`alpha` reads:

- field family and commitment stage;
- Plan embedding;
- AR and DLM entropy/margin;
- top-value agreement and Jensen–Shannon divergence.

The shared scientific commit controller then applies hard support and soft
risk **after** the language-only agreement gate. The sampled semantic value is encoded as exactly one DLM special token
and rendered into the Llama control transcript.

BG/BP make a joint **pre-commit** decision; they do not first commit a Llama
sample and then edit it. After every joint commit, any unconsumed future Llama
distribution is discarded and the next block is scored from the updated
canonical state.

## 5. One blockwise decoding step

```text
1. Commit one six-value lattice block.
2. Read the next species block from the frozen Llama program.
3. For each serialization slot belonging to that species:
   a. render the committed canonical state as the Llama control prefix;
   b. obtain one Llama/SLA and one DLM factorized XYZ distribution;
   c. map both to semantic values and form the joint triplet beam;
   d. apply exact PBC support and soft risk;
   e. atomically commit one complete site triplet.
4. Advance to the next species only after all of its site slots are complete.
```

For the lattice block, one Llama/SLA call and one DLM forward provide six
field distributions; a progressive width-32 beam forms valid tuples. For a
site block, one call from each model provides three factorized X/Y/Z
distributions. The default top four per axis gives 64 jointly scored triplets;
a train-only coverage/throughput audit may freeze top eight (512) before any
evaluation. Triplets are committed together only after exact PBC evaluation.
For BG/BP diagnostics, the **Llama proposal** is the support-normalized SLA
argmax, the **DLM proposal** is the support-normalized DLM argmax, and a
**fusion change** occurs when the fused argmax differs from the Llama proposal.
Sampling differences alone are not called revisions. The explicit
post-predictor revision is reserved for B3, which re-masks a completed block.
The paper reports proposal agreement, fusion-change frequency and whether
fusion changes improve permutation-aware teacher accuracy and raw geometry.
DLM is called a bidirectional pre-commit executor, not a
certificate-producing verifier.

## 6. Training

### Frozen-weight adjacent decoder cells

All four cells use the retained two-epoch Compact-V2 DLM weights and one common
block-mask schedule. DLM temperatures are calibrated by field family,
remaining-mask ratio and block stage on MP20-train rollout states.

- `BC`: canonical species-block order, no Llama semantic prior, syntax/exact
  composition only;
- `BO`: BC + Llama species-block order only;
- `BG`: BO + frozen SLA semantic policy through the agreement gate;
- `BP`: BG + shared lattice/joint-site PBC commit controller.

These adjacent cells isolate order, Llama semantic guidance and periodic
support without changing DLM weights or mask-time semantics.

The agreement gate is trained with Llama and DLM frozen. For each
rollout-matched MP20-train field state with teacher semantic value `y`:

\[
\mathcal L_{\mathrm{gate}}
=-\log p_{\mathrm{fused}}(y)
+\lambda_\alpha\alpha(s)^2,
\]

subject to a field/stage KL cap of 0.05 nats from the DLM distribution.
Generated/committed
prefixes are visible; teacher values are used only as labels. This defines
exactly what the gate learns.

### B2: schedule-matched adaptation

Starting once from the same retained Compact-V2 checkpoint, train one LoRA
with rank 8, alpha 32 and dropout 0.05 for exactly 1,696 optimizer updates.
Effective source batch is 16 over 27,136 rows; paired Plan-view losses are
averaged inside each source group. LR is `5e-6`, cosine decay with 100 warmup
updates and minimum LR ratio 0.2. Two A800 GPUs use per-device source batch one
and gradient accumulation eight. The actual BP mask pattern is used:

- exact `N` and element slots visible;
- lattice resolved first;
- one Llama-selected species block active;
- later blocks masked;
- CE on eligible teacher DLM special tokens;
- teacher and frozen predicted same-schema Plan views, sharing source weight;
- schema/exact-composition support active in the loss;
- geometric support audited against quantized teacher values but not used to
  delete a teacher label from CE; incompatible teacher rows are disclosed and
  excluded only from the geometry-controller auxiliary term;
- no CHGNet, hull, model494 endpoint or generated-test outcome in the loss.

Only this DLM LoRA trains in B2; base DLM weights remain frozen. The frozen SLA
is evaluated on jointly committed MP20-train prefixes; the separately trained
gate may abstain when Llama
uncertainty or distribution shift is high. C3FD, Planner and the complete
Track-A controller remain frozen. B2 is the final learned LLM-guided DLM.

The existing G2 residual implementation is retained as a mechanism ablation,
not automatically stacked into B2. Previous results show that periodic
residuals can help but interact with serialization/order. If used, it receives
the exact B2 mask schedule and is compared on the shared mechanism subset
before becoming part of the main executor.

## 7. Optional complete-state continuous-response corrector

After B2 resolves a complete graphable predictor crystal:

1. on frozen B2-generated MP20-train states from the same runtime, measure
   model494's actual deterministic first deployed transition;
2. compute PBC torus displacement for coordinates and log-metric displacement
   for lattice;
3. use the Candidate-E1-owned `Confidence-E1` module trained after B2 from
   MP20-train force/stress labels; a component is active only
   when its calibrated probability of energy descent without risk increase is
   at least 0.60;
4. select one block by a frozen analytic-risk/predictor-uncertainty rule that
   never reads the model494 response;
5. re-mask it while leaving all other final fields visible to the DLM;
6. obtain fresh Llama SLA logits for that block;
7. project the continuous drift onto adjacent legal semantic values;
8. run one full-context DLM correction with the current value retained as a
   no-op candidate.

This produces B3. It is the only proposed place where the continuous refiner
changes a discrete DLM decision. It is described as an empirical
deployed-refiner response, not an exact score at an in-distribution `t=800`
state.

Implementation is two-stage rather than triple-resident: checkpoint 494 first writes
the canonical one-step response for frozen complete B2 states, then a
co-resident BF16 Llama+DLM worker performs the correction. No scientific state
changes between those stages.

This one-step **response-corrector role** is distinct from the common
**terminal-refiner role**, which runs the full registered continuous trajectory
after raw A/B structures are frozen. The former changes one discrete block; the
latter outputs the final continuous crystal.

For attribution, `B2C0` performs the same response-independent block
selection, model494 call,
fresh Llama/SLA call and DLM corrector pass as B3, but fixes the continuous
response residual to zero. B2→B2C0 measures the effect of reopening one block;
B2C0→B3 isolates response-residual steering at equal model calls and correction
compute. It does not claim to isolate the existence of a response call, which
is deliberately present in both. All cells report NFE and latency.

## 8. Minimal experiment design

Full fixed256 cells:

| Cell | Purpose |
|---|---|
| BC | frozen DLM + common block schedule, canonical order |
| BO | BC + Llama species order |
| BG | BO + frozen SLA/gate |
| BP | BG + shared PBC commit controller |
| B2 | one-epoch schedule-matched LLM-guided DLM |
| B2C0 | B2 + compute-matched zero-response corrector |
| B3 | B2 + one complete-state drift corrector |

BC/BO/BG/BP are the required adjacent decoder comparisons and use the same
fixed256 ledger. A fixed-weight gate and optional G2 residual are mechanism
diagnostics on one shared subset; they are not mixed into the primary
attribution.

Each fixed256 cell uses one model seed and two common sampling streams. Matched
A1, BC, BO, BG, BP and B2 raw generation runs before terminal diffusion. B3
starts only if its generated-state validation gives directional AUC above 0.55
and more than half of usable one-step responses lower the calibration energy
without increasing feasibility risk.

## 9. Metrics

- requested-denominator body and exact-composition validity;
- raw Direct, graphability, minimum-distance ECDF and collision count;
- raw lattice volume/condition and VPA agreement;
- raw CHGNet and a held-out second-MLIP robustness endpoint;
- Llama/DLM argmax agreement, fusion-change frequency and
  permutation-aware teacher accuracy;
- DLM NFE, Llama calls and wall time;
- common terminal model494, surrogate MP-reference Strict/Meta S.U.N.;
- a registered DFT subset only if an ab initio stability claim is needed.

## 10. Expected effect and main risk

The strongest expected effect is raw structural validity: Llama supplies a
causal local prior, DLM sees unresolved global context, and PBC support prevents
completed-site collisions. Stability improvement is less certain until the
drift corrector is measured.

The main engineering risk is not model size but state synchronization. Every
commit must update four linked objects atomically: canonical state, DLM canvas,
Llama control transcript and periodic graph cache. The cross-representation
contract defines that transaction.
