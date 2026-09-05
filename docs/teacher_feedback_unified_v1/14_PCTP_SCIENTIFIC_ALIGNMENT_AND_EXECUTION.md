# PCTP: Scientific Alignment, Training Sufficiency, and Execution Plan

Status: **design approved; implementation and new GPU execution have not started**

## 1. Executive decision

The next method is **Program-Conditioned Trajectory Preference (PCTP)**. It is
not another K10 epoch, a renamed historical D3PO run, or a return to R03/G2.
PCTP starts only from the current successful architecture and changes the unit
of stability learning from a local candidate transaction to a complete SPAD
trajectory with a fixed terminal physical transition.

The design review reached `APPROVED` after four material corrections:

1. optimize the full deployed action probability, not a K<=4 conditional
   softmax;
2. assign value from a complete generated trajectory, not a frozen local K10
   continuation;
3. update the DLM and Llama pointer sequentially so terminal credit is
   identifiable;
4. use one fixed tau800 terminal objective and report raw behavior separately,
   while retaining a raw-stability ordering inside equal terminal tiers.

This design removes the known mathematical mismatch. It does **not** guarantee
the numerical target before execution. The defensible expectation is that it
has a materially higher chance of reaching the fixed `10%/50%` endpoint than
more training of the failed local posterior because it optimizes the right
random variable.

## 2. Lineage boundary

### 2.1 Current architecture retained by PCTP

Only the following recent, implemented path is current:

\[
\text{C3FD support}
\rightarrow
\text{compact Plan}
\rightarrow
\text{Llama hidden state + species pointer}
\rightarrow
\text{SPAD DLM}
\rightarrow
\text{fixed model494 }\tau=800.
\]

- **C3FD support** restricts Planner-Llama to chemically reachable
  compositions.
- **Compact Plan** carries the exact composition and the existing structural
  conditioning fields.
- **Llama species pointer** emits a permutation of Plan species. It cannot
  alter the composition.
- **SPAD DLM** executes exact `7+4N`, six-token cell and three-token XYZ
  transactions, triclinic PBC support, and suffix-visible non-causal revision.
- **model494 tau800** is a frozen terminal transition applied once to every
  method. It is not trained by PCTP and is not claimed as a new contribution.

### 2.2 Historical or failed work not included in PCTP

The following are not current modules and must not be presented as such:

- R03 safe-axis scheduling;
- the historical full-sequence D3PO experiment;
- G2, BTRD, Force-Score, and Potential-Closure;
- the final local K10 candidate-posterior route.

They can appear only as historical evidence or negative diagnosis. In
particular, PCTP does not import an R03 schedule head and does not reuse the old
heterogeneous D3PO preference pool.

## 3. What the completed result establishes

The final stream21 endpoint was:

| Endpoint | Reconstructed | Novel | Unique | Strict stable | Strict S.U.N. | Meta stable | Meta S.U.N. |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw DLM | 256/256 | 254/256 | 256/256 | 5/256 | 5/256 | 51/256 | 49/256 |
| fixed tau800 | 256/256 | 230/256 | 256/256 | 20/256 | 14/256 | 148/256 | 123/256 |

This localizes the remaining problem:

- composition, serialization, reconstruction, and cohort uniqueness are not
  the bottleneck;
- raw DLM structures are valid but rarely occupy low-energy basins;
- tau800 already places 148 structures below the Meta stability threshold, but
  25 stable structures are not novel;
- Strict is limited first by stability itself: even before novelty
  intersection, only 20/256 structures are Strict stable.

Therefore the new objective must increase low-energy terminal probability and
retain novelty. More validity engineering cannot supply the missing Strict
count.

## 4. Why the local K10 route failed

### 4.1 Candidate-conditional probability was the wrong training object

The deployed scorer computes

\[
m_C(s)=\sum_{a\in C(s)}p_\theta(a\mid s),
\]

but the training loss re-normalized only over the legal K<=4 candidates:

\[
\mathcal L_{\rm old}
=D_{\rm KL}\!\left(
q^*(a\mid a\in C,s)
\Vert
p_\theta(a\mid a\in C,s)
\right).
\]

Consequently, adding the same constant to every candidate log score leaves the
loss unchanged. The model could learn which candidate is preferred if it
already entered `C(s)`, but the objective did not require those actions to gain
probability against the rest of the deployed vocabulary. This differs from the
absolute deployed-action objective stated in the design document.

### 4.2 Local K10 action value was not complete-trajectory value

The old label estimated one action followed by a frozen reference
continuation:

\[
Q_{10}^{\pi_0}(s,a)
=-E_{\rm CHGNet}
\left(R_{10}(T_{\pi_0}^{\rm remain}(s,a))\right).
\]

Crystal energy is not additive over the cell and isolated XYZ transactions.
Once the policy changes, both the visited states and the remaining continuation
change, so reusing the same labels for a warm-start pass makes the local value
increasingly stale.

### 4.3 The old objective did not implement its intended raw-energy constraint

The Route-B design intended a simultaneous K10 improvement and non-increasing
expected raw energy. The implemented posterior accepted one energy vector at a
time and used K10 energy alone. Raw validity was preserved by SPAD, but raw
energy was not an optimization constraint.

### 4.4 Training ran, but more of the same was not justified

The two completed passes produced 5,472 finite optimizer updates and 8,208
group exposures. Adapter weights changed, gradients were finite, and the
warm-start path was correct. Posterior loss nevertheless had no decreasing
trend, and the second prospective stream did not show a consistent Strict
gain. Training insufficiency may have existed, but convergence of the old
surrogate would still not imply improved free-generation S.U.N.

The final diagnosis is therefore:

> Numerical optimization succeeded, but the objective and control variable did
> not match the complete deployed trajectory or the terminal S.U.N. event.

## 5. Scientific problem and mathematical object

For a composition `c`, the current hierarchy is

\[
P\sim\pi_\phi(P\mid c),
\qquad
\tau\sim p_\theta(\tau\mid c,P),
\qquad
x_f=G_{800}(\tau).
\]

- `P` is the existing Llama-conditioned species permutation;
- `tau=(a_1,...,a_T)` is the actual SPAD sequence of cell, XYZ, and revision
  transactions;
- `G_800` is the fixed model494 transition followed by the fixed evaluation
  relaxation;
- `x_f` is the structure on which stability and novelty are evaluated.

The scientific target is the probability of a favorable terminal event:

\[
J_\delta(\phi,\theta)
=\Pr\left[
E_{\rm hull}(x_f,c)\le\delta
\land N(x_f)
\land U(x_f)
\right],
\]

where `delta=0` for Strict and `delta=0.1 eV/atom` for Meta.

PCTP optimizes the program-conditioned terminal trajectory rather than treating
composition planning, DLM decoding, and terminal physics as unrelated modules.

## 6. Terminal outcome construction

### 6.1 Fixed evaluator

Training compositions are selected only from MP20 train. The terminal value
uses:

1. one current-SPAD raw trajectory;
2. one frozen model494 tau800 transition;
3. the same CHGNet relaxation and energy convention used by final evaluation;
4. a predeclared Materials Project phase-diagram snapshot;
5. the same novelty definition against the frozen training reference set.

The MP snapshot is disclosed as external scientific knowledge, not hidden
train-only supervision. It contains no prospective candidate outcome. Missing
training chemsys receive no energy preference and remain clean-CE examples;
they are not backfilled after a test result is observed.

### 6.2 Rank outcome rather than tune many scalar weights

For each same-composition K4 group, candidates are ordered lexicographically:

1. raw-invalid or terminal-invalid candidate;
2. valid but non-novel terminal candidate;
3. valid novel terminal above 0.1 eV/atom;
4. novel Meta terminal;
5. novel Strict terminal.

Within the same terminal tier, use the raw stability tier and then lower raw
`E_hull`; only after those ties use lower terminal `E_hull`. The resulting order
is converted into centered, clipped rank advantages. No outcome-dependent
weight sweep is required.

This ordering has two deliberate properties:

- the final tau800 S.U.N. target remains primary;
- when terminal outcomes are tied, which will be common, the DLM receives a
  native raw-stability learning signal rather than ordinary imitation alone.

The method directly optimizes novelty because stream21 showed that all
stable-to-S.U.N. losses came from novelty (`U=256/256`). It does not claim to
optimize general cohort-level uniqueness; uniqueness remains a reported fixed
evaluation metric.

## 7. DLM policy improvement

### 7.1 Data

Freeze the current Llama pointer. For 1,024 MP20-train compositions, sample four
complete trajectories from the current SPAD DLM under the same program. Retain
every valid and invalid attempt. There is no inference-time selection: K4 exists
only to estimate a training advantage.

### 7.2 Correct deployed probability

For a logged transaction `(s_t,a_t)`, recompute

\[
\log p_\theta(a_t\mid s_t,P)
\]

under the complete dynamic legal support, including transaction temperature,
PBC masks, suffix-visible state, and all token normalizers. Do not perform a
second softmax over the four sampled trajectories or over a hand-built local
candidate set.

### 7.3 Terminal-preference-weighted transaction objective

Let `A_i` be the detached group-relative terminal advantage of trajectory `i`.
The complete trajectory score is

\[
\log p_\theta(\tau_i\mid c,P)
=\sum_{t=1}^{T_i}
\log p_\theta(a_{it}\mid s_{it},P).
\]

The policy term is

\[
\mathcal L_{\rm PG}
=-\frac1K\sum_i A_i
\sum_{t=1}^{T_i}
\log p_\theta(a_{it}\mid s_{it},P).
\]

To avoid retaining an 8B-model graph for the complete trajectory, sample a
logged transaction uniformly and multiply its contribution by `T_i`. This is
an unbiased estimator of the transaction-sum gradient for the fixed clipped
surrogate:

\[
\mathbb E_{t\sim U(1,T_i)}
\left[-A_iT_i\nabla\log p_\theta(a_{it}\mid s_{it},P)\right]
=-A_i\sum_t\nabla\log p_\theta(a_{it}\mid s_{it},P).
\]

This is described as **terminal-preference-weighted deployed-transaction policy
optimization**, not exact trajectory DPO and not continuous online RL.

### 7.4 Distribution preservation and sufficient training

Use the existing LoRA policy with:

- separate, alternating policy-gradient and clean MP20 CE microbatches;
- a frozen reference adapter and per-transaction KL regularization;
- three fixed passes over the 1,024 groups;
- fresh deterministic transaction sampling in each pass;
- one training seed and only the pass-3 terminal checkpoint;
- validation diagnostics for mechanism reporting, never checkpoint selection.

Three passes provide 12,288 trajectory exposures before transaction sampling,
instead of two repetitions of 4,104 local candidate states. More importantly,
the exposure now supervises actual complete-trajectory outcomes and complete
deployed action probabilities. Training sufficiency is assessed by absolute
preferred-action log-probability movement and held-out group advantage
agreement, not by comparing losses from different state quarters.

This addresses the previous undertraining risk without pretending that extra
epochs can repair an incorrect objective.

## 8. Llama program policy improvement

After DLM training, freeze the new DLM. The Llama backbone remains frozen; only
the existing species pointer is updated.

For 512 MP20-train compositions with at least three species:

1. sample up to four distinct legal species permutations from the current
   pointer;
2. execute each permutation with two semantic common random streams under the
   frozen DLM;
3. key randomness by composition identity, physical site, transaction role,
   and draw identity rather than schedule ordinal;
4. average the two terminal outcomes per permutation;
5. compute the same terminal rank advantage;
6. optimize the complete permutation log probability with a frozen-pointer KL
   anchor.

Unary and binary compositions remain supported at inference; they do not need
preference supervision when their program action space is trivial or nearly
trivial.

Sequential freezing gives interpretable credit:

- during DLM optimization, the program is fixed;
- during pointer optimization, the DLM is fixed.

The paper can therefore claim that the same terminal value improves DLM
execution and then the Llama-conditioned program policy. It must not claim
simultaneous end-to-end training or real-time Llama control.

## 9. Why this is not scientifically misplaced

| Previous mismatch | PCTP correction |
|---|---|
| Local K10 action | Complete generated terminal outcome |
| K<=4 conditional softmax | Full deployed action probability |
| Frozen reference continuation | One current-policy trajectory collection |
| Artificial 1:1 cell/XYZ states | Transactions sampled from real SPAD visitation |
| K10 energy ranking | Fixed tau800 evaluator-style Strict/Meta tiers |
| No novelty objective | Terminal novelty is part of the preference order |
| Simultaneous Llama/DLM ambiguity | Sequentially freeze and update each policy |
| More epochs on stale labels | One policy-iteration cycle, no refresh |

The remaining approximation is explicit: CHGNet plus an MP phase diagram is an
operational stability evaluator, not DFT, kinetic stability, or experimental
synthesizability. PCTP aligns to the same operational endpoint used in the
reported S.U.N. benchmark. That is scientifically coherent as long as the paper
does not broaden the claim beyond this evaluator.

## 10. What “the model learns stability” may mean

PCTP can establish three different levels of evidence:

1. **Policy learning:** preferred terminal trajectories gain absolute deployed
   probability on unseen MP20-train validation groups.
2. **Native basin learning:** raw energy or raw S.U.N. improves while validity
   is retained. This supports the claim that the DLM itself moves toward lower
   energy geometry.
3. **System basin learning:** tau800 S.U.N. improves under the frozen transition.
   This supports the claim that the programmed DLM produces better basin-entry
   states for the fixed physical transition.

The design directly trains levels 1 and 3 and supplies raw tie-breaking signal
for level 2. It cannot guarantee level 2 before measurement. If only tau800
improves, the correct claim is “better basin entry under a fixed transition,”
not “raw DLM samples equilibrium crystals.”

This distinction prevents a new scientific-object mismatch while preserving a
strong, useful paper claim.

## 11. Single preflight and single formal run

To minimize inference cost, run exactly one 64-composition x K4 preflight. It
checks only:

- logged SPAD transactions can be replayed with the exact deployed legal
  support;
- changing policy weights changes absolute selected-action probability;
- the frozen MP snapshot covers enough train groups to construct value;
- complete trajectories contain terminal rank or continuous-energy variation;
- tau800 does not erase all trajectory variation;
- the pointer produces nontrivial permutations where the composition permits;
- raw reconstruction and validity remain intact.

This is a mechanism check, not a SUN trial and not a hyperparameter search. No
multi-tau, multi-arm, multi-seed, or repeated small endpoint evaluation follows.
If the interface and learning signal exist, execute one formal PCTP
configuration. If they do not, fix only the demonstrated implementation error;
do not alter the scientific target to force a pass.

## 12. Execution DAG and resources

Maximum resources remain four A800 and four CPU cores per GPU.

| Stage | Work | GPU plan | Expected wall time |
|---|---|---:|---:|
| A | General program/trajectory/reward/replay interfaces and tests | CPU plus small GPU replay | 4-8 h |
| B | One 64xK4 preflight | 4 A800 | 1-2 h |
| C | 1,024xK4 DLM trajectory rollout, tau800, CHGNet | 4 A800 | 7-10 h |
| D | Three-pass DLM policy training | 2 A800 | 2-4 h |
| E | 512xK4x2 pointer-value rollout | 4 A800 | 6-9 h |
| F | Pointer training and one fixed256 endpoint | pointer light, then 4 A800 | 1-2 h |

Expected total is 24-30 hours; I/O or cache problems can extend it to roughly
35 hours. A paper1000 endpoint, if authorized by the predeclared final rule,
adds approximately 3-5 hours.

The implementation should expose reusable interfaces rather than one-job
scripts:

- program sampling and permutation log probability;
- trajectory and semantic-RNG logging;
- deployed transaction replay;
- terminal outcome ranking;
- DLM and pointer policy-gradient trainers.

Only essential run configuration, terminal status, and compact result summaries
are retained. Repetitive certificates and SHA-heavy gates are intentionally
excluded.

## 13. Expected target movement

The fixed tau800 endpoint starts from:

- Strict S.U.N. `14/256`; target `26/256`;
- Meta S.U.N. `123/256`; target `128/256`.

Meta requires five additional outcomes and has dense training signal. Strict
requires twelve additional outcomes, an approximately 86% relative increase,
and is the decisive risk.

Conditional on the single preflight showing terminal variation that survives
tau800, the working range is:

- Meta S.U.N.: approximately `125-140/256`;
- Strict S.U.N.: approximately `20-30/256`.

These are planning ranges, not confidence intervals or promised results. The
simultaneous target has a realistic but not high-confidence chance. PCTP merits
the run because it is the first proposal that directly optimizes the complete
deployment endpoint while preserving the current successful architecture.

## 14. Paper formulation

### 14.1 Scientific question

> How can a chemically constrained LLM program a non-causal crystal DLM so that
> its executable denoising trajectories preserve exact periodic structure and
> increasingly terminate in low-energy, novel crystal basins?

### 14.2 One core method claim

> PCTP learns a terminal-value-aware, program-conditioned denoising policy: a
> C3FD-constrained Llama species program determines how SPAD visits crystal
> construction, while terminal basin preference sequentially improves the
> non-causal DLM execution policy and the program policy without inference-time
> energy evaluation or candidate reranking.

### 14.3 Evidence chain for the main paper

The main text needs only three connected results:

1. **Executable support:** C3FD and SPAD establish exact chemistry and nearly
   complete geometric execution.
2. **Value-bearing program:** under a frozen DLM, terminal-value training changes
   the Llama pointer toward better programs.
3. **Terminal discovery:** the final PCTP policy improves raw diagnostics and/or
   fixed-tau800 S.U.N. on one independent prospective endpoint.

These are not three unrelated contributions. They are one causal sequence:

\[
\text{chemically reachable program}
\rightarrow
\text{non-causal executable trajectory}
\rightarrow
\text{terminal basin value}.
\]

model494 is the fixed environment transition. Historical R03, D3PO, G2, and
K10 results remain outside the main method.

### 14.4 Claim boundary

Safe claims:

- the species permutation is an executable latent program that changes SPAD
  state visitation;
- terminal preference is assigned to actual deployed DLM transactions;
- sequential policy improvement gives both the DLM execution and Llama pointer
  access to the same terminal value;
- inference uses one program and one trajectory without CHGNet or reranking;
- the system improves basin-entry probability under a fixed physical
  transition if the final endpoint is positive.

Unsupported claims unless new evidence establishes them:

- Llama controls the DLM in real time;
- the Llama backbone and DLM are trained end to end;
- the model has learned the true DFT potential-energy surface;
- raw DLM structures are stable when only tau800 improves;
- PCTP predicts kinetic stability or experimental synthesizability;
- model494 is a new contribution.

## 15. Decision log

| Decision | Alternatives rejected | Reason |
|---|---|---|
| Use complete terminal preference | More K10 epochs; local absolute-mass patch alone | Local value cannot represent the complete trajectory endpoint |
| Use full deployed transaction probability | K4 candidate-normalized loss | Candidate-relative ranking did not move absolute deployed mass |
| Use group-relative policy gradient | Naive simultaneous joint DPO; full-trajectory autograd | Correct terminal credit with tractable memory and clear probability semantics |
| Sequentially train DLM then pointer | Simultaneous Llama/DLM update | Identifiable credit assignment |
| Keep only the current species program | Import R03 or add a schedule head | Preserves recent architecture and avoids historical conflation |
| Optimize fixed tau800 endpoint | Multi-tau search | Matches the registered terminal system and avoids selection |
| Use raw stability only within terminal ties | Ignore raw; fixed weighted raw/refined sum | Adds native signal without changing the primary endpoint or tuning weights |
| One preflight, one formal run | Many small ablations, seeds, arms, and sweeps | Compute and paper focus |
| Treat MP phase diagrams as disclosed external knowledge | Hidden proxy or post-test query backfill | Clear scientific and data boundary |

Final design disposition: **APPROVED**.

## 16. Relation to published methods

- [DDPO: Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301)
  motivates treating denoising as a multi-step policy optimized by terminal
  reward.
- [D3PO: Preference-Based Alignment of Discrete Diffusion Models](https://arxiv.org/abs/2503.08295)
  supports reference-regularized preference alignment for discrete diffusion,
  while PCTP uses the actual SPAD deployed transaction process.
- [Diffusion-DPO](https://arxiv.org/abs/2311.12908) provides the broader
  precedent for aligning diffusion generation with terminal preferences.
- [MatterGen](https://www.nature.com/articles/s41586-025-08628-5) supports the
  need to treat atom types, coordinates, and the periodic lattice as a coupled
  crystal-generation object and to evaluate stability together with novelty and
  uniqueness.

