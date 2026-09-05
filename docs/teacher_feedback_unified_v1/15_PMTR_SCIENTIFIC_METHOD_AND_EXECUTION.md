# PMTR: Programmed Manifold-to-Token Repair

Status: **scientific design approved; implementation and GPU execution not yet started**

## 1. Decision

The next method is **Programmed Manifold-to-Token Repair (PMTR)**. It replaces
the superseded reward-based PCTP proposal and does not extend the failed K10,
D3PO, G2, BTRD, or Force-Score routes.

PMTR addresses the measured gap directly:

> The current DLM can produce chemically exact, parseable, PBC-valid crystals,
> but its symbolic lattice and coordinate tokens do not expose a learned local
> direction back toward a relaxed periodic crystal manifold after generation
> errors have already become visible context.

The core method is a bridge, not another reward:

\[
\text{continuous SPD/PBC repair vector}
\longrightarrow
\text{legal probability transport over crystal tokens}.
\]

A C3FD-constrained Llama program determines the semantic repair order. SPAD
provides suffix-visible non-causal revision. PMTR predicts a local periodic
repair vector and compiles it into residual logits for the active cell or XYZ
transaction. Inference uses no energy model, MLIP, force, stress, filter,
reranker, or replacement.

The final multi-role design review disposition is **APPROVED**. The numerical
`10%/50%` target remains an experimental goal, not a guaranteed consequence of
the design.

## 2. Strict method lineage

### 2.1 Current architecture retained

Only this recent pipeline is the starting point:

\[
\text{C3FD composition support}
\rightarrow
\text{compact Plan}
\rightarrow
\text{Llama species program}
\rightarrow
\text{SPAD DLM}
\rightarrow
\text{optional fixed model494 }\tau800.
\]

- C3FD constrains chemically reachable composition actions.
- Compact Plan is shared by training and inference.
- The existing Llama-conditioned pointer emits a permutation of Plan species.
- SPAD executes exact `7+4N`, six-token cell and three-token XYZ transactions,
  triclinic PBC support, and suffix-visible basin-closure revision.
- model494 remains an optional, frozen terminal fallback and is not part of
  PMTR training or its novelty claim.

### 2.2 Historical work excluded

The following are not PMTR components:

- R03 safe-axis schedules;
- historical full-sequence D3PO;
- G2 periodic-relation adapters or checkpoints;
- BTRD/model494 endpoint imitation;
- projected Force-Score targets;
- SGTC positive-only training;
- Potential-Closure and local K10 posterior;
- the superseded PCTP terminal-reward plan.

No historical weight, generated/refined positive teacher, reward loss, or
outcome-selected structure enters PMTR.

## 3. Empirical problem being solved

The final recent stream21 result is:

| Endpoint | Reconstructed | Novel | Unique | Strict stable | Strict S.U.N. | Meta stable | Meta S.U.N. |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw DLM | 256/256 | 254/256 | 256/256 | 5/256 | 5/256 | 51/256 | 49/256 |
| fixed tau800 | 256/256 | 230/256 | 256/256 | 20/256 | 14/256 | 148/256 | 123/256 |

Composition and structural execution are already saturated. Raw failure is an
energy/force/stress problem rather than a parser or exact-composition problem.
At tau800, all stable-to-S.U.N. losses come from novelty, not uniqueness:

- Strict: `20 stable -> 14 novel/unique stable`;
- Meta: `148 stable -> 123 novel/unique stable`;
- uniqueness itself is `256/256`.

The method must therefore change native periodic geometry while preserving the
existing exact-support machinery. More format gates, ordinary CE, terminal
reward, or composition work cannot supply the missing Strict count.

## 4. Why current masked training misses this object

Standard absorbing-mask training predicts a masked clean token from other clean
visible tokens. Once inference commits a wrong lattice or coordinate token,
that token becomes visible context and is treated as fact. The model was not
trained to interpret a coherent set of wrong-but-visible periodic geometry
tokens and retract them toward the relaxed-data manifold.

Current closure CE improved the visibility pattern, but its teacher contexts
are clean MP20 structures. Real SPAD closure receives generated, off-manifold
contexts. Historical rollout-matched CE attempted to combine generated prefixes
with clean MP20 suffix targets; it failed because later clean Y/Z coordinates
were incompatible with already committed lattice/X/Y errors.

PMTR closes this gap without splicing incompatible structures. It starts from a
single MP20 polymorph, corrupts its complete lattice and coordinates coherently,
and repairs every corrupted variable along one program-matched path.

## 5. Scientific object

A crystal with fixed species `A` is represented by

\[
x=(G,U,A)\in \mathrm{SPD}(3)\times\mathbb T^{3N}\times\mathcal A^N,
\]

where

\[
G=LL^\top
\]

is the lattice metric and `U` contains fractional coordinates on the periodic
three-torus. C3FD freezes `A` and `N`; PMTR acts only on `G` and `U`.

For a relaxed MP20 structure `x0`, PMTR constructs a valid corrupted structure
`xt` on the same manifold and learns a local retraction

\[
v^*(x_t,x_0)
=\operatorname{Log}_{x_t}(x_0)
=(S^*_{t\to0},\Delta R^*_{t\to0}).
\]

`S*` is an SPD log-metric tangent and `Delta R*` is the minimum-image Cartesian
repair displacement. The method does not claim this vector is the exact score
of a Boltzmann distribution or the true potential-energy gradient.

The scientific hypothesis is narrower and testable:

> A DLM trained to compile force-certified local retractions of coherent
> relaxed-crystal corruptions into legal token probability transport will
> produce lower-force, lower-stress, lower-energy raw crystals under the same
> Llama-programmed SPAD repair process.

## 6. Offline force-certified corruption

### 6.1 Clean target boundary

Every positive target is one original MP20-train relaxed structure. Generated
SPAD structures, model494 endpoints, CHGNet-relaxed structures, selected
low-energy candidates, validation rows, and prospective outcomes never become
targets.

### 6.2 Joint periodic corruption

For each clean structure, sample a symmetric lattice tangent `H` and a
zero-centre-of-mass Cartesian displacement `Delta R`:

\[
G_t=G_0^{1/2}\exp(H)G_0^{1/2},
\qquad
L_t=\operatorname{chol}(G_t),
\]

\[
U_t=\operatorname{wrap}\left((U_0L_0+\Delta R)L_t^{-1}\right).
\]

This jointly changes cell metric and coordinates while preserving atom
identity. It avoids treating the same fractional displacement as the same
physical move in differently sized or tilted cells.

The complete corrupted crystal is serialized through the exact current
`7+4N` codec before any certification. Parser validity, positive-definite cell,
composition, and triclinic minimum-image distance are checked on the decoded
quantized structure, not on an unobservable continuous precursor.

### 6.3 CHGNet is a data certificate, not a reward or inference module

For each row, generate at most four deterministic corruption proposals and
retain the first proposal satisfying all of the following after quantization:

1. it remains parser/PBC valid;
2. its CHGNet single-point energy exceeds the clean structure by a bounded
   positive amount;
3. the CHGNet coordinate descent direction has positive inner product with the
   PBC retraction toward `x0`;
4. when the lattice is changed, the CHGNet stress-derived descent tangent agrees
   with the SPD retraction toward `G0`.

No lowest-energy or largest-gap proposal is selected. If none of the four is
certified, the source row remains a normal clean SPAD-CE example.

CHGNet values are never included in the optimization loss. This eliminates the
force-versus-clean-target gradient conflict seen in earlier ideas: the learned
target is always `x0`, while the certificate establishes that the target
direction is locally energy descending under the disclosed training MLIP.

## 7. Program-matched non-causal repair trajectory

Let the Llama pointer output species program

\[
P=(a_{p_1},\ldots,a_{p_m}).
\]

Starting from the complete corrupted structure `xt`, construct exactly the
current inference repair path:

1. remask the six-token cell transaction;
2. repair the cell toward `G0`;
3. recompute Cartesian/fractional transforms and all triclinic MIC relations
   under the newly committed lattice;
4. traverse species blocks in reverse Llama-program order;
5. remask and atomically repair each XYZ transaction;
6. end only after every corrupted degree of freedom has been revisited.

At each teacher state:

- earlier repaired blocks contain their clean `x0` values;
- the active block is masked for the DLM;
- all unrepaired blocks remain visibly corrupted values from the same `xt`;
- the clean active transaction is the target.

This is not ordinary masking. Wrong-but-coherent geometry remains visible while
the DLM learns to revise one transaction using both earlier and later context.
It is also not the failed rollout splice: every visible value belongs to one
known path between the same corrupted and clean polymorph.

Before remasking, runtime stores the old active cell or XYZ value. The DLM sees
the mask, while the PMTR head receives the old generated geometry as an explicit
side channel. This information is available at inference because PMTR is
revising an already generated structure.

## 8. Repair-vector head

PMTR adds one small, zero-output-initialized head. It consumes:

- DLM hidden states at the active transaction;
- compact Plan context;
- active species and Llama-program rank;
- the saved old active geometry;
- exact committed lattice and other visible sites;
- strict triclinic minimum-image pair vectors and distances.

It does not consume CHGNet, a learned reward, G2 hidden states, or model494.

### 8.1 Lattice branch

The cell branch predicts six values defining a symmetric tangent `S_hat` and a
candidate corrected metric

\[
\widehat G
=G_t^{1/2}\exp(\widehat S)G_t^{1/2}.
\]

Positive definiteness is guaranteed for every finite output. Zero output gives
`G_hat=Gt` exactly.

### 8.2 Coordinate branch

For an active site `i`, shared PBC pair messages predict antisymmetric
Cartesian corrections:

\[
\Delta\widehat r_i
=\frac{1}{\deg(i)}
\sum_j a_{ij}
\frac{r_{ij}^{\rm MIC}}
{\lVert r_{ij}^{\rm MIC}\rVert+\epsilon}.
\]

Pairwise antisymmetry removes global translation. The correction is converted
under the currently committed lattice:

\[
\Delta\widehat u_i=\Delta\widehat r_iL^{-1},
\qquad
\widehat u_i=\operatorname{wrap}(u_i+\Delta\widehat u_i).
\]

XYZ is predicted as one vector and committed atomically. After a cell repair,
all coordinate conversions and MIC relations are recomputed with the new cell.

## 9. Manifold-to-token probability transport

The head does not overwrite or round a crystal. It compiles a continuous repair
proposal into the active DLM token distribution.

For a predicted scalar coordinate or lattice value `y_hat`, find the two legal
quantization bins that bracket `y_hat`, and form a differentiable triangular
basis `phi_k(y_hat)`. These are bins around the predicted corrected value, not
merely the old token's `+/-1` neighbours. A single repair sweep can therefore
make a finite correction rather than being limited to one 0.01 fractional bin.

The residual is

\[
\Delta\ell_k
=g_f\left[\phi_k(\widehat y)-\phi_k(y_{\rm old})\right],
\qquad
\ell_{\rm final}=\ell_{\rm DLM}+\Delta\ell.
\]

Only legal bins in the active lattice or XYZ transaction receive residual
mass. `N`, elements, counts, prompt tokens, committed non-active tokens, and the
rest of the vocabulary are unchanged. Existing exact composition, token-family
and PBC hard support is applied after the residual.

Zero initialization gives `Delta ell=0`; PMTR starts exactly as the retained
SPAD DLM rather than replacing it with a new generator.

## 10. Training objective

The loss contains no reward, policy gradient, ranking, selected winner, or
terminal outcome:

\[
\mathcal L
=\mathcal L_{\rm token}
+\mathcal L_{\rm SPD}
+\mathcal L_{\rm torus}
+\mathcal L_{\rm step}
+\mathcal L_{\rm reference}.
\]

- `L_token`: active-transaction CE to the original MP20 token block;
- `L_SPD`: dimensionless geodesic error between predicted and clean lattice
  metric tangents;
- `L_torus`: dimensionless PBC Cartesian/torus repair error to `U0`;
- `L_step`: bounded repair magnitude regularization;
- `L_reference`: clean SPAD CE and reference-logit preservation.

Continuous errors are normalized by the fixed corruption scales so lattice and
coordinate units cannot dominate one another. Clean and corrupted examples use
alternating microbatches rather than a single unverified `0.5:0.5` scalar sum.
A short gradient-scale diagnostic checks for an order-of-magnitude imbalance;
it is an engineering sanity check, not a hyperparameter sweep.

CHGNet energy, force, and stress never appear in this loss.

## 11. Training sufficiency

Initialize from the retained pre-K10 SPAD basin-closure checkpoint. Do not use
the failed K10 warm-start policy.

Training uses:

- all `27,136` MP20-train source structures;
- one newly generated coherent corruption per source in each epoch;
- two fixed source epochs;
- active transactions sampled according to actual SPAD basin-closure
  visitation over cell, species rank, and XYZ roles;
- one training seed;
- one final checkpoint after epoch two;
- no early stopping, checkpoint selection, or validation-guided tuning.

This supplies `54,272` coherent corrupted-source exposures, plus alternating
clean SPAD anchors. It replaces the old 4,104 sparse, local-value states with
dense token and geometric supervision over the complete train distribution.

The two epochs solve the previous training-sufficiency problem only in the
proper sense: the new repair head receives enough examples of the correct
scientific object. They are not claimed to guarantee convergence or the final
S.U.N. threshold.

## 12. Inference contract

Production inference is exactly:

```text
C3FD-constrained compact Plan
    -> one Llama species program
    -> one current SPAD raw trajectory
    -> one PMTR cell + reverse-species repair sweep
    -> one raw crystal
    -> optional separately labelled fixed model494 tau800 fallback
```

At inference PMTR uses:

- no CHGNet or other MLIP;
- no force or stress calculation;
- no energy or hull score;
- no reward/value model;
- no candidate pool, filter, rerank, replacement, or best-of-N;
- no online relaxation inside the DLM.

The repair-vector head and DLM logits alone perform the revision.

## 13. Why PMTR is scientifically aligned

| Prior failure or mismatch | PMTR correction |
|---|---|
| Masked model never sees coherent visible geometry errors | Physically coherent replacement geometry remains visible during repair |
| Rollout MP20 suffix incompatible with committed errors | Corrupted and clean endpoints are the same polymorph and every block is repaired |
| G2 relation loss improved validity but not energy | PMTR learns an explicit clean-manifold retraction certified locally downhill |
| Force-Score copied quantized force targets | Force is not a target; exact post-quantization retraction to real MP20 is the target |
| BTRD imitated model494 endpoints | No model494 or generated structure is a positive teacher |
| K10 ranked one local candidate | No ranking; every supervised state belongs to a complete repair path |
| PCTP reused generic terminal reward | No reward/RL; PMTR changes native DLM geometry inference |
| Transformer must infer `L`, PBC and Cartesian relations symbolically | SPD/PBC arithmetic is explicit before transport into token logits |
| Llama program is decorative prompt text | Its species order determines the actual repair-state sequence and conditions the head |

The remaining approximation is disclosed: CHGNet certifies a locally downhill
training direction, while the ultimate scientific reference remains DFT. PMTR
learns a local relaxed-data-manifold retraction, not the exact PES, global hull,
kinetics, or synthesizability.

## 14. What would prove that the DLM learned stability

Evidence must be separated into three levels:

1. **Synthetic repair:** held-out force-certified MP20 corruptions are retracted
   more accurately. This is necessary but not sufficient.
2. **Native free-generation repair:** applying one PMTR sweep to actual current
   SPAD raw structures lowers paired CHGNet energy, force RMS, and stress while
   preserving composition and PBC validity. This establishes native local
   stability learning.
3. **Discovery endpoint:** raw and/or fixed-tau800 official S.U.N. improves on a
   fresh prospective cohort. This establishes system-level utility.

Only levels 2 and 3 support the paper headline. If PMTR improves only synthetic
corruption reconstruction, the method has not solved the real problem. If only
tau800 improves, the claim is better basin entry under a fixed fallback, not
raw equilibrium generation.

## 15. One preflight, one formal run

To avoid expensive small-experiment proliferation, use one integrated
MP20-train preflight:

- 512 force-certified corruption rows from independent MP20-train structures;
- 384 train and 128 held out by source composition;
- one small repair-head learnability run;
- the same package applies one free PMTR sweep to a fixed 128 actual current
  SPAD train-only raw structures;
- compare paired raw energy, force RMS, stress, reconstruction, composition,
  and PBC validity;
- no Direct, S.U.N., tau search, multiple arm, seed, or hyperparameter sweep.

The preflight answers only whether the mechanism transfers from coherent
corruptions to actual SPAD errors. A non-adverse, nonzero raw physical shift
authorizes one full training run. Token-loss improvement alone does not.

After the single preflight:

1. build the full two-epoch dynamic corruption stream;
2. train one PMTR endpoint;
3. run one fresh fixed256 prospective raw generation/repair;
4. compute raw validity and CHGNet energy/force/stress first;
5. complete raw and fixed-tau800 official S.U.N. once;
6. launch paper1000 only if the predeclared final condition is reached.

No multi-tau, multi-arm, multi-seed, checkpoint sweep, or result-directed
iteration is added.

## 16. Engineering plan

Implement reusable modules rather than job-specific wrappers:

1. `manifold_corruption`: joint SPD/PBC corruption and offline certification;
2. `programmed_repair_states`: exact Llama-program/SPAD teacher trajectory;
3. `manifold_repair_head`: lattice and coordinate vector branches;
4. `manifold_token_transport`: legal differentiable token-logit renderer;
5. `pmtr_runtime`: side-channel capture, cell commit, MIC recomputation, XYZ
   atomic commit;
6. `pmtr_trainer`: clean/corrupted alternating training and compact diagnostics.

Reuse only generic current assets:

- dynamic `7+4N` codec and legal token families;
- strict triclinic PBC operators;
- SPAD state machine;
- Llama species pointer and compact Plan;
- current LoRA/trainer loading and distributed infrastructure.

Do not load historical G2, BTRD, Force, D3PO, K10, or PCTP weights or labels.
Keep run metadata compact: source split, configuration, final status, and result
summary. Repeated SHA/certificate layers are unnecessary.

## 17. Resource and time estimate

Maximum resource use remains four A800 and four CPU cores per GPU.

| Stage | Expected wall time |
|---|---:|
| General PMTR implementation and tests | 5-9 h |
| Single integrated 512-row preflight | 1-2 h |
| Full corruption certification/data build | 2-5 h |
| Two-epoch PMTR training | 2-4 h |
| Fixed256 raw physical evaluation | 0.5-1.5 h |
| tau800 + official finalization | 0.5-1.5 h |
| Total before paper1000 | approximately 12-22 h |
| Conditional paper1000 | additional 3-5 h |

The data builder and training are expected to dominate. CHGNet certification
uses batched offline inference only; it never changes the production runtime.

## 18. Expected result and failure interpretation

PMTR has a stronger native-stability mechanism than K10/PCTP because its dense
supervision acts on the actual lattice and coordinate repair field rather than
on sparse terminal ranks. Existing hard support and zero initialization make a
large validity collapse unlikely.

Reasonable planning expectations are:

- raw reconstruction/composition/PBC validity remains near the current
  `98-100%` range;
- paired raw force/stress and median energy move in the favorable direction if
  the repair field transfers;
- fixed-tau800 Meta S.U.N. has a credible path from `123/256` across `128/256`;
- Strict `14/256 -> 26/256` remains the difficult outcome and is not guaranteed.

A planning range conditional on a positive real-raw preflight is approximately:

- Strict S.U.N. `20-30/256` after the fixed fallback;
- Meta S.U.N. `126-142/256` after the fixed fallback.

These are not confidence intervals or promised results.

Failure diagnosis is simple:

- synthetic repair positive but actual raw unchanged: corruption-to-deployment
  mismatch;
- actual raw force/energy improves but tau800 unchanged: fallback erases native
  gains;
- raw and tau800 energy improve but Strict does not: local basin improvement is
  insufficient to cross the official zero threshold;
- validity declines: renderer/support implementation failure;
- novelty declines after tau800 only: fixed fallback, not PMTR, controls the
  stable-to-S.U.N. loss.

## 19. Paper story

### 19.1 Scientific question

> How can a scientific LLM turn chemical and structural knowledge into an
> executable repair program for a discrete crystal DLM, when physical
> corrections live on a continuous periodic manifold but generation decisions
> must remain legal language tokens?

### 19.2 One core contribution

> PMTR compiles continuous SPD/PBC manifold repair vectors into legal
> probability transport over native crystal-language tokens inside a
> Llama-programmed, suffix-visible non-causal repair process.

This single claim binds the modules tightly:

1. C3FD restricts chemical logits to reachable composition actions.
2. Llama emits the species-semantic program that determines repair order.
3. SPAD exposes completed future geometry and remasks earlier cell/XYZ
   transactions.
4. PMTR computes periodic repair in the correct continuous geometry and
   transports it back to legal discrete token probabilities.

The DLM is necessary because it can preserve the completed suffix while
revising earlier lattice or coordinate transactions. An autoregressive LLM
would need to regenerate the suffix. Llama is necessary because its learned
species program supplies the semantic dependency order; PMTR is not a generic
fixed sweep detached from the Planner.

### 19.3 Main-paper evidence

The main paper needs only a compact connected chain:

- existing C3FD/SPAD evidence: exact chemistry and near-complete execution;
- one paired native PMTR result: raw force/stress/energy and validity;
- one final raw/tau800 S.U.N. result.

No large reward ablation, multi-tau table, historical method collage, or many
small inference experiments are required in the main text.

### 19.4 Claim boundary

Safe claims after positive evidence:

- Llama's species program controls the actual non-causal repair order;
- PMTR learns to retract coherent periodic geometry errors toward the relaxed
  MP20 data manifold;
- the learned repair vector is compiled into native legal DLM token logits;
- no MLIP or energy evaluator is used at inference;
- raw stability improves if paired raw energy/force/stress and raw S.U.N. do;
- the full system improves fixed-transition S.U.N. if the final endpoint does.

Do not claim:

- physical corruption, force guidance, or learned denoising order alone is new;
- PMTR learns the exact DFT PES or a global ground-state search policy;
- Llama controls the DLM interactively in real time;
- model494 is a new contribution;
- experimental synthesizability or kinetic stability;
- guaranteed `10%/50%` before observation.

## 20. Novelty relative to prior work

Prior work already covers the ingredients separately:

- [DiffCSP](https://arxiv.org/abs/2309.04475) and
  [MatterGen](https://www.nature.com/articles/s41586-025-08628-5) perform
  continuous periodic crystal denoising of lattice/coordinates;
- [CANDI](https://arxiv.org/abs/2510.22510) combines discrete and continuous
  diffusion generally;
- corrective masked-DLM work trains or invokes visible-token self-correction;
- learned-order/path-planning work optimizes generic masked-token schedules;
- force-guided diffusion uses physical force during generation or score
  guidance.

PMTR does not claim these ingredients. Its proposed novelty is their
crystal-language bridge:

\[
\text{Llama semantic repair program}
+\text{suffix-visible SPAD state}
+\text{SPD/PBC repair vector}
+\text{legal special-token probability transport}.
\]

The closest internal token-native geometry executor and manifold-repair notes
were design-only and never produced a checkpoint or experimental result. PMTR
is their SPAD-era consolidation with coherent visible-error training and a
strictly MLIP-free inference path.

## 21. Decision log

| Decision | Rejected alternative | Reason |
|---|---|---|
| Replace PCTP with PMTR | Generic terminal reward/RL | Reward alignment already has strong prior art and does not establish native DLM geometry learning |
| Target coherent visible errors | Ordinary absorbing-mask CE | Real repair context contains wrong visible lattice/coordinate tokens |
| Use one same-polymorph corruption/retraction path | Generated-prefix plus clean-suffix splice | Historical Y/Z realizability failure |
| Use continuous SPD/PBC vector before token projection | Expect Transformer attention to infer all metric arithmetic | Directly addresses measured lattice-coordinate blind spot |
| Certify force consistency offline | Put force/stress in loss | Avoids conflict with the real MP20 target and old Force-Score failure |
| Render around predicted target bins | Restrict to old token +/-1 | Allows finite one-sweep correction |
| Keep current Llama species program | Import R03 or add a new schedule head | Preserves current architecture and method focus |
| Full MP20, two epochs | Another sparse 4,104-state pilot | Addresses training sufficiency with dense correct-object supervision |
| One integrated preflight and one formal run | Many arms, tau, seeds, sweeps | Efficiency and paper focus |
| No inference MLIP | CHGNet force/energy guidance | Required deployment and scientific boundary |

Final design disposition: **APPROVED**.

