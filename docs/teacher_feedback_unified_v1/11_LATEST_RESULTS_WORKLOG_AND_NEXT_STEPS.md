# Latest results, completed work, and full-MP20 next steps

Last updated: 2026-09-04 14:10 Asia/Shanghai

Branch: `codex/unified-scientific-decoding`

This document separates three kinds of statements:

- **Observed**: completed experiment or audited repository fact;
- **Interpretation**: inference supported by the observed evidence;
- **Planned**: implementation or experiment that has not produced a result.

## 1. Executive summary

The current system has solved composition validity and nearly solved structural
validity, but native DLM stability remains the main bottleneck.

The completed 512-source Potential-Closure pilot produced the best fixed
stream17 result at model494 tau800:

- composition validity: `256/256`;
- parsed structural validity: `256/256`;
- Strict S.U.N.: `20/256 = 7.81%`;
- Meta S.U.N.: `125/256 = 48.83%`.

Relative to the equal-compute closure control, this adds three Strict and ten
Meta S.U.N. structures after tau800. Relative to frozen BS, it adds two Strict
and nine Meta structures. The target remains `26/256 = 10.16%` Strict and
`128/256 = 50.00%` Meta, so the observed gaps are six and three structures.

However, native raw evidence is not yet strong enough to say that the DLM has
learned a consistently lower-energy basin policy. The pilot changed most
crystals, but its potential-control raw single-point energy median was almost
zero and its clustered confidence interval crossed zero. This identifies the
central error in the old objective: an instantaneous energy after one local
transaction is not the value of that transaction after the remaining DLM
program and physical relaxation.

The next experiment therefore does **not** simply enlarge the old 512-source
pool. It trains two value definitions on one shared full-MP20 deployment-chain
dataset:

1. terminal single-point value, retained as the scale control;
2. terminal short-relaxation basin value, the proposed main method.

Both routes will use the same 27,136 sources, Llama programs, DLM states,
candidate token blocks, future DLM continuations, terminal raw structures,
initial checkpoint, seed and compute. The only causal difference will be the
definition of physical value.

## 2. Unified architecture

The current paper-facing architecture is one hierarchical generation policy:

```text
C3FD chemical support
        ↓
Planner-Llama composition and structural Plan
        ↓
Llama species/scaffold program
        ↓
SPAD DLM initial non-autoregressive crystal construction
        ↓
cell transaction → Llama anchor-2 transaction → Llama anchor-1 transaction
        ↓
native raw crystal
        ↓
optional frozen model494 refinement
        ↓
validity, relaxed hull, Strict/Meta S.U.N.
```

The factorization is

\[
p_{\phi,\theta}(P,x\mid c)
=
\pi_\phi(P\mid c,\mathcal A_{\rm C3FD})
\prod_t q_\theta(x_{S_t}\mid x_{\bar S_t},c,P).
\]

### 2.1 C3FD

C3FD restricts the reachable chemical action support. It prevents illegal
stoichiometry, charge and element-space actions before Llama samples them. It
does not generate lattice or coordinate logits.

Observed consequence: composition validity is consistently close to 100%; in
the latest fixed256 raw and tau800 experiment it is exactly `256/256` for every
arm.

### 2.2 Planner-Llama and scaffold pointer

Llama is not merely a text preprocessor. Its `species_program` is an executable
control variable that determines:

- the species transaction order used during initial SPAD body construction;
- which species is the first structural anchor;
- which species is the second structural anchor;
- the reverse anchor revision order after the cell transaction;
- which future DLM transactions remain after a candidate training action.

The existing pointer asset has observed exact-permutation accuracy `73.50%`,
root accuracy `80.41%`, and produces a predominantly non-canonical scaffold
program. It is retained rather than retrained in the current stability sprint.

For full MP20, 24,558 train rows have the typed input required by the trained
pointer. The other 2,578 rows use the already declared
`canonical_missing_pointer_semantics` fallback. No typed transcript or chemical
certificate is fabricated.

### 2.3 SPAD DLM

The crystal body is represented by exactly `7+4N` special tokens:

\[
[N,L_a,L_b,L_c,\alpha,\beta,\gamma,
(A_i,x_i,y_i,z_i)_{i=1}^N].
\]

SPAD acts on complete physical transactions:

- all six lattice tokens are one cell transaction;
- one atom's three coordinate tokens are one XYZ transaction;
- an invalid transaction is rolled back as a unit;
- strict triclinic periodic minimum-image support rejects hard collisions;
- suffix-visible revision lets the DLM preserve later atoms while rewriting an
  earlier anchor.

This is where the DLM is necessary. A fixed autoregressive factorization cannot
naturally revise an early lattice or atom while retaining the complete future
crystal. The DLM learns conditional kernels over arbitrary masked transactions:

\[
q_\theta(z_S\mid z_{\bar S},c,P).
\]

### 2.4 model494

model494 remains an explicit, frozen continuous refinement fallback. It is not
used to choose a DLM sample at inference and is not hidden inside the native
result. Raw and tau800 results are always reported separately.

The long-term objective is to remove this dependency by improving native raw
stability. Current evidence does not yet justify removing it: tau800 still
raises the best Strict/Meta result from `3.91/22.66%` raw to `7.81/48.83%`.

## 3. Data facts

### 3.1 Base SPAD SFT

The base SPAD DLM was trained with the complete MP20 split:

| Split | Rows | Treatment |
|---|---:|---|
| train | 27,136/27,136 | all retained |
| validation | 9,047/9,047 | all retained |

The 2,578/889 train/validation rows without typed pointer semantics are not
dropped; they use canonical program fallback. Therefore the base DLM training
is full-MP20 training.

### 3.2 Potential-Closure pilot

The newly added physical alignment in the completed pilot was not full MP20:

| Quantity | Count |
|---|---:|
| MP20-train source structures | 512 |
| matched transaction states | 2,048 |
| retained legal candidates | 6,883 |
| informative energy groups | 2,003 |
| state strata | clean-cell, clean-site, on-policy-cell, on-policy-site |

This distinction matters: the final adapter inherited a full-MP20 SFT base,
but its new energy supervision was a 512-source pilot.

## 4. What the completed pilot trained

Each group masked exactly one complete transaction. Candidate structures shared
the same composition and all non-active tokens. Raw CHGNet single-point energy
defined the finite candidate-set target:

\[
q_A^*(a\mid s)
\propto
q_{\rm ref}(a\mid s)
\exp[-\eta E_{\rm CHGNet}(T(s,a))],
\]

subject to a candidate-set KL budget of `0.05 nat`.

The phrase **candidate-set KL** is important. The target normalizes over the
two-to-four retained complete actions; it is not an exact projection over the
full vocabulary action space.

Training used 2,048 optimizer updates:

```text
clean MP20 CE
→ cell transaction objective
→ clean MP20 CE
→ site transaction objective
```

The equal-compute control replaced potential targets with clean complete-
transaction targets. Both arms used the same BS initialization, seed, optimizer
and endpoint-only checkpoint policy.

Inference used no CHGNet, no force, no energy lookup, no candidate search, no
reranking and no replacement. It generated one body, then one cell transaction
and at most two Llama-selected anchor transactions.

## 5. Latest quantitative results

All rows below use one fixed stream17 cohort and denominator 256. Eight
unresolved official chemical systems remain unknown in the official hull cache;
they are not silently converted into known unstable entries.

| Endpoint | Arm | Comp-valid | Struct-valid | N∩U | Hull-known | Strict S.U.N. | Meta S.U.N. |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | BS | 256/256 | 255/256 | 255/256 | 248/256 | 7/256 (2.73%) | 54/256 (21.09%) |
| raw | closure control | 256/256 | 255/256 | 256/256 | 248/256 | 12/256 (4.69%) | 55/256 (21.48%) |
| raw | potential closed | 256/256 | 255/256 | 256/256 | 248/256 | 10/256 (3.91%) | 58/256 (22.66%) |
| tau800 | BS | 256/256 | 256/256 | 222/256 | 248/256 | 18/256 (7.03%) | 116/256 (45.31%) |
| tau800 | closure control | 256/256 | 255/256 | 220/256 | 248/256 | 17/256 (6.64%) | 115/256 (44.92%) |
| tau800 | potential closed | 256/256 | 256/256 | 223/256 | 248/256 | **20/256 (7.81%)** | **125/256 (48.83%)** |

### 5.1 Potential versus closure control

| Endpoint | Strict wins/losses | Meta wins/losses | Net count |
|---|---:|---:|---:|
| raw | 4/6 | 21/18 | Strict -2, Meta +3 |
| tau800 | 6/3 | 20/10 | Strict +3, Meta +10 |

### 5.2 Single-point energy evidence

Potential minus control before model494:

- paired mean: `-0.236 eV/atom`;
- paired median: approximately `0`;
- lower/higher/equal: `133/118/5`;
- composition-clustered mean 95% interval: `[-0.599,+0.131] eV/atom`.

After tau800:

- paired mean: `+0.0055 eV/atom`;
- paired median: approximately `0`;
- lower/higher/equal: `128/125/3`;
- clustered interval: `[-0.0149,+0.0245] eV/atom`.

The tau800 single-point distributions are effectively equal, even though
Potential-Closed has ten more Meta S.U.N. successes than control. This indicates
that threshold/basin occupancy and N/U changes, not a broad endpoint energy
shift, produced the observed S.U.N. difference.

### 5.3 The policy did change structures

Weak raw stability cannot be explained by a no-op adapter:

- control cell transactions changed: `237/256`;
- potential cell transactions changed: `237/256`;
- control anchor transactions changed: `393`;
- potential anchor transactions changed: `389`;
- control and potential raw structures exactly equal: only `16/256`.

The adapter changed most outputs, but those changes were not consistently
directed toward lower relaxed basins.

## 6. What worked

### 6.1 Strongly supported

1. **Full-MP20 SPAD SFT is operational.** The complete train/validation split is
   retained.
2. **C3FD preserves chemical validity.** Latest composition validity is
   `256/256` in every arm.
3. **Llama programs control actual DLM execution.** They determine initial
   species groups and closure anchors, rather than being decorative prompt text.
4. **SPAD complete transactions preserve geometry validity.** Raw structural
   validity is `255/256`; the best tau800 arm is `256/256`.
5. **Potential supervision has a positive downstream signal.** At tau800 it
   improves Meta S.U.N. by ten structures over equal compute and reaches 48.83%.

### 6.2 Promising but not yet established

1. Native potential alignment may improve average raw energy, but the observed
   median and clustered interval do not establish a population-wide shift.
2. The tau800 S.U.N. improvement is a useful positive signal, but it does not
   prove that native DLM endpoints entered lower basins before refinement.
3. A second stream has not been used to claim replication for this pilot.

### 6.3 Not supported by the pilot

1. Merely increasing the weight or number of updates of the instantaneous
   single-point objective is not justified.
2. The pilot's independently constructed cell/site training states are not an
   exact representation of the sequential deployment chain.
3. It is not yet justified to remove model494 from the final system.

## 7. Engineering work completed

### 7.1 Representation and schedule

- exact `7+4N` serializer and parser;
- full MP20 schedule-matched SFT data;
- C3FD-supported Llama programs;
- trained species/scaffold pointer;
- SPAD initial program execution;
- complete six-token cell revision;
- suffix-visible XYZ anchor revision;
- strict triclinic MIC support and atomic rollback.

### 7.2 Pilot physical alignment

- matched train-only state freezer;
- deterministic first-distinct variable-K candidate sampler;
- raw CHGNet E/F/stress labeller;
- finite-candidate posterior with legal support and candidate-set KL;
- objective-separated clean CE and transaction updates;
- equal-compute closure control;
- fixed-stream native and tau800 evaluation;
- cached-official Strict/Meta S.U.N. finalizer.

### 7.3 Efficiency improvements

- DLM generation uses batch 8;
- CHGNet single-point prediction uses batch 16;
- deterministic tau800 refinement was sharded with two workers per GPU;
- control/potential tau800 completed in 13 minutes 39 seconds;
- matched BS tau800 completed in 7 minutes 2 seconds;
- expensive Direct/fingerprint evaluation was omitted from the latest critical
  path while exact N/U needed by S.U.N. was retained.

### 7.4 Important recovered engineering failures

These failures did not change scientific parameters or erase negative evidence:

| Failure | Root cause | Resolution |
|---|---|---|
| first Potential-Closed training stopped at update 580 | float64 posterior was down-cast before a strict float32 sum check | keep the K<=4 loss in float64; parameter-identical recovery completed 2,048 steps |
| first tau800 summary rejected BS pairing | reused BS refinement belonged to another cohort | refine the matching fixed256 BS once; preserve control/potential outputs |
| full S.U.N. wrapper ended failed after four cells completed | final summary expected `report.json`, actual artifact was `summary.json` | reuse all completed cells and perform CPU-only finalization |

## 8. Why the old scientific object was wrong

The deployed trajectory is sequential:

\[
s_0\xrightarrow{a_L}s_1
\xrightarrow{a_2}s_2
\xrightarrow{a_1}x_T.
\]

The pilot independently constructed cell and site states from one source
answer. A lattice candidate was evaluated immediately, without asking what
happens after the two Llama-programmed anchor revisions. Therefore its label was

\[
-E(T(s_0,a_L)),
\]

while the value needed by deployment is

\[
Q_L(s_0,a_L)
=
\mathbb E_{a_2,a_1\sim q_{\rm ref}}
[-E(x_T)].
\]

Scaling the old independent groups to 27,136 sources would reduce sampling
variance but preserve this objective bias. The next design explicitly fixes
the bias before increasing scale.

## 9. Corrected full-MP20 experiment

### 9.1 Shared deployment-chain source

Every MP20 train row receives:

- one frozen Llama/fallback program;
- one frozen reference SPAD body attempt;
- one source weight;
- one assigned deployment stage;
- all preceding reference transactions needed to construct the correct state;
- all remaining reference transactions needed to construct terminal action
  value.

Stages are assigned by source index modulo three:

```text
0 → cell
1 → anchor_second
2 → anchor_first
```

This gives full-source coverage and matches the deployed one-cell/two-anchor
frequency. It is intentionally different from full-transaction coverage, which
would create three groups per source and triple the cost.

Actual XYZ slots are resolved from each generated body's species order. Teacher
atom order is retained only for auditing and can never select an on-policy slot.

### 9.2 Shared candidate support

Every A/B group shares up to four legal complete actions:

1. current no-op;
2. one reference-DLM action;
3. one positive physical direction;
4. its opposite direction.

For a site,

\[
\Delta r_i=\pm\delta\frac{F_i}{\lVert F_i\rVert+\epsilon},
\qquad
\Delta f_i=\Delta r_iL^{-1}.
\]

For a cell, CHGNet defines
\(\sigma=V^{-1}\partial E/\partial\varepsilon\), so the descent proposal is

\[
L'=L\exp\left(
-\delta_\varepsilon
\frac{\operatorname{sym}(\sigma)}
{\lVert\operatorname{sym}(\sigma)\rVert+\epsilon}
\right),
\]

with the opposite proposal also retained. Both are quantized through the
deployed special-token vocabulary and checked after quantization. No candidate
energy is read while constructing support.

### 9.3 Shared terminal continuation

For a candidate at stage `t`, the frozen reference DLM executes every remaining
Llama-programmed transaction using candidate-shared random numbers:

\[
x_T(s_t,a;\xi)
=
T_{q_{\rm ref}}^{t+1:T}(T(s_t,a);\xi).
\]

Thus A and B receive byte-identical terminal structures.

### 9.4 Route A: full-source single-point control

\[
V_A(s,a)=-E_{\rm CHGNet}(x_T(s,a;\xi)).
\]

Route A is not presented as the new main method. It answers whether the pilot
was limited primarily by 512-source coverage.

### 9.5 Route B: Basin-Consistent Transaction Posterior

\[
V_B^{(K)}(s,a)
=
-E_{\rm CHGNet}(R_K(x_T(s,a;\xi))).
\]

`R_K` uses the same CHGNet potential, cell degrees of freedom and force
tolerance as final evaluation, with one fixed short step count. Residual force
and stress are diagnostics or deterministic tie-breakers only; basin endpoint
energy is the optimized value.

The comparison therefore changes only

```text
instantaneous terminal potential  ↔  short-relaxation basin value
```

and not dataset, candidates, continuation, model or compute.

## 10. Is this inference cheating?

No, under the frozen contract:

- CHGNet, force, stress and short relaxation appear only in train-only label
  construction;
- inference calls none of them;
- one Plan produces one DLM trajectory;
- no candidate is generated, scored or selected at inference;
- no test/prospective outcome trains either route.

This is offline physical-value distillation, analogous in role to an offline
reward teacher.

There is nevertheless an evaluation-coupling limitation: CHGNet is both the
training teacher and the generated-structure relaxation proxy. The immediate
claim is therefore **CHGNet-basin-aligned stability**, not universal DFT or
experimental stability. Official MP data provide reference phases but do not
turn CHGNet-generated energies into independent DFT measurements. MatterSim is
currently unavailable and does not block this experiment; a later frozen DFT
or independent-MLIP subset would strengthen external validity.

## 11. Scientific-object preflight

Formal route training starts only after the following are observed on frozen
MP20-train calibration data:

1. stage states exactly follow body → cell → anchor-2 → anchor-1;
2. later anchor slots are resolved from the current generated body;
3. every candidate executes all remaining reference-DLM stages;
4. A and B receive identical source/state/action/continuation/terminal bytes;
5. post-quantization `+F/-F` and `-stress/+stress` signs agree with direct
   finite-energy perturbation more often than chance;
6. one predeclared short-relaxation value ranks normal full-relaxation outcomes
   better than the single-point value;
7. each stage retains measurable post-relaxation candidate variation;
8. all 27,136 sources remain in accounting, including failed body/action rows;
9. effective gradient-producing groups are reported rather than silently used
   as a new denominator.

This preflight does not select a test seed, Plan, checkpoint or evaluation
cohort. It only decides whether the teacher measures the claimed scientific
object.

## 12. Formal training contract

If the preflight approves Route B:

| Route | GPUs | CPUs | Role |
|---|---:|---:|---|
| A: terminal single-point | 2 A800 | 8 | scale/value control |
| B: basin-consistent | 2 A800 | 8 | proposed main route |

Both routes use:

- the same BS initialization;
- the same training seed and source permutation;
- global batch 16;
- 1,696 posterior updates, one full 27,136-source pass;
- 1,696 alternating clean-CE updates, one full MP20 pass;
- 3,392 total optimizer updates;
- LR `5e-6`, warmup 100, fixed candidate-set KL budget;
- only the final step-3392 checkpoint;
- no early stopping or route selection.

## 13. Current execution state

### 13.1 Completed now

- full design and scientific-object audit;
- full-MP20 pointer/fallback export implementation;
- full-source transaction-stage ledger implementation;
- force/stress transaction proposal implementation;
- strict post-quantization triclinic PBC validation;
- 24 relevant local and remote tests passing;
- trained pointer checkpoint schema verified against the live artifact.

### 13.2 Active job

Job `39658`, using `4 A800 + 16 CPU`, is shared by both future routes.

Observed at 14:10:

- programs: `27,136/27,136` complete;
- trained Llama pointer programs: `24,558`;
- canonical fallback programs: `2,578`;
- transaction source ledger: `27,136/27,136` complete;
- reference body generation: running;
- rank0 progress: approximately `54/6,784`;
- current observed throughput implies roughly 70–80 minutes for the body stage
  if sustained.

This job reads no CHGNet energy, official outcome or model494 result. Its output
is common input, so it cannot favor A or B.

### 13.3 Implementation in progress

Three non-overlapping modules are being completed and reviewed:

1. exact sequential state/candidate/reference-continuation builder;
2. shared single-point/short/full CHGNet value labeller and ranking preflight;
3. two-GPU full-MP20 transaction-value trainer.

No formal A/B checkpoint is being trained before these contracts pass.

## 14. Remaining execution sequence

1. Finish the 27,136 reference bodies and audit retained failures.
2. Build exact sequential states from generated body species order.
3. Construct shared no-op, DLM and ±physics candidates.
4. Execute candidate-shared remaining DLM stages and freeze terminal structures.
5. Compute batched terminal single-point values.
6. On a frozen train-only subset, compare the predeclared short-relaxation
   ranking with normal full relaxation.
7. If the basin proxy is valid, label all shared candidates with short basin
   value.
8. Launch A and B concurrently at `2 A800 + 8 CPU` each.
9. Evaluate both on the already frozen Plan/program/cohort, first native raw and
   then the explicit tau800 fallback.
10. Finalize official-cache Strict/Meta S.U.N. and update the paper narrative.

## 15. Time estimate

The following are estimates, not gates:

| Remaining stage | Expected wall time |
|---|---:|
| full reference body completion | about 1–1.5 h from current progress |
| shared sequential actions and terminal continuation | 4–8 h |
| fixed short-vs-full basin calibration | 0.5–1.5 h |
| full shared value labelling | 6–12 h, throughput-dependent |
| two concurrent 2-GPU training routes | 4–6 h |
| fixed raw and tau800 evaluation/finalization | 1.5–3 h |
| total from current state | approximately 17–31 h |

The dominant cost is no longer pointer inference or tau800. It is generating
deployment-matched terminal counterfactuals and assigning a basin-consistent
value to every full-MP20 source.

## 16. Expected outcomes and decision use

### Strong outcome

- Route B produces a reproducible left shift in native relaxed energy;
- native Strict and Meta improve without validity loss;
- tau800 reaches or exceeds `10%/50%`;
- Route B exceeds Route A on the same shared candidate data.

Paper interpretation: Llama programs the sequence of non-causal DLM crystal
transactions, and basin action-value distillation teaches those same
transactions to enter lower-energy regions without an inference critic.

### Moderate outcome

- native energy or Meta improves, but Strict remains below 10%;
- tau800 exceeds 50% Meta but not 10% Strict.

Paper interpretation: the method improves metastable-basin occupancy while
exact hull stability remains the limiting physical frontier. model494 remains
an explicit co-generation/refinement stage.

### Negative outcome

- Route B does not outperform the shared single-point control;
- or the short basin value fails to predict normal relaxation ordering.

Interpretation: the finite local transaction action space is insufficient to
control final basin identity. The negative result is retained; it does not
authorize inference-time reranking, test-outcome adaptation or hidden candidate
selection.

## 17. Key files

- `docs/teacher_feedback_unified_v1/10_FULL_MP20_BASIN_ACTION_VALUE_PLAN.md`:
  frozen scientific and execution contract;
- `src/scripts/export_full_mp20_pointer_programs.py`: full learned-pointer plus
  canonical-fallback program export;
- `scripts/freeze_full_mp20_transaction_sources.py`: one-stage-per-source full
  MP20 ledger;
- `src/crystal_dlm/transaction_physics.py`: deterministic ±force/±stress
  complete transaction proposals;
- `slurm/193_full_mp20_reference_bodies.sbatch`: current shared full-source job;
- `docs/teacher_feedback_unified_v1/09_EFFICIENCY_FIRST_POTENTIAL_CLOSURE_PLAN.md`:
  completed 512-source pilot and final S.U.N. evidence.

