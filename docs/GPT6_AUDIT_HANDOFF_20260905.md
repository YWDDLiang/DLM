# GPT-6 audit handoff: C3FD–Llama–SPAD crystal generation

Date: 2026-09-05  
Branch: `codex/llama-programmed-basin-closure`  
Local worktree: `D:\codex_work\ai4s\DLM_llama_programmed_basin_closure`  
Snapshot HEAD: `3f67509`  
Execution state: **all jobs stopped; no PMTR checkpoint or PMTR scientific result**

This document is a read-only audit handoff. It separates:

1. code that exists;
2. experiments that actually completed;
3. narrative claims that are not backed by the same evidence;
4. proposed methods that were never scientifically evaluated;
5. unresolved contradictions that GPT-6 must adjudicate.

It does not authorize another experiment or prescribe a replacement method.

## 1. Recommended audit order

Read these sources in order:

1. this handoff;
2. the user's full physical-gap audit at
   `C:\Users\admin\.codex\attachments\7041083c-fbfd-4997-a5dc-2dede1e549b3\pasted-text.txt`;
3. [current README](../README.md);
4. [current execution checklist](teacher_feedback_unified_v1/04_EXECUTION_CHECKLIST.md);
5. [code-grounded architecture audit](teacher_feedback_unified_v1/16_PMTR_CODE_GROUNDED_ARCHITECTURE_AUDIT.md);
6. [native stability audit](teacher_feedback_unified_v1/08_DLM_NATIVE_STABILITY_AND_DIFFUSION_FALLBACK.md);
7. [latest completed SPAD/K10 worklog](teacher_feedback_unified_v1/11_LATEST_RESULTS_WORKLOG_AND_NEXT_STEPS.md);
8. [final K10 diagnosis](teacher_feedback_unified_v1/13_STREAM19_DIAGNOSIS_AND_FINAL_ITERATION.md);
9. [PMTR method proposal](teacher_feedback_unified_v1/15_PMTR_SCIENTIFIC_METHOD_AND_EXECUTION.md);
10. the code entry points listed in Section 10 below.

Evidence priority is:

```text
run-level immutable result
    > current source code and tests
    > final result document
    > README/checklist summary
    > planning document
    > conversation-level intention
```

Old plans are useful for reconstructing intent, but they are not proof that the
planned objective, data path, or inference behavior was implemented.

## 2. The scientific target and the actual failure

For a fixed composition `c`, define a broad executable region

\[
\mathcal V_c=
\{(L,X):\operatorname{comp}(L,X)=c,\;V(L)>0,\;
d_{\min}^{\rm PBC}(L,X)>d_0\}.
\]

The physically useful region is much smaller:

\[
\mathcal B_c(\delta)=
\{(L,X)\in\mathcal V_c:
E_{\rm hull}(L,X,c)\le\delta,
\|F(L,X,c)\|\approx0,
\|\sigma(L,X,c)\|\approx0\}.
\]

The completed system has strongly increased

\[
P_\theta(\mathcal V_c),
\]

but has not established an increase in

\[
P_\theta(\mathcal B_c(\delta)\mid \mathcal V_c,c).
\]

That distinction is the central scientific fact. The repository's current
metric defines Strict stability as `E_hull <= 0`, Meta stability as
`E_hull <= 0.1 eV/atom`, and S.U.N. as the stable, unique, novel intersection.
It does not evaluate phonons, kinetic barriers, finite-temperature free energy,
or experimental synthesis. See
[native stability audit:94](teacher_feedback_unified_v1/08_DLM_NATIVE_STABILITY_AND_DIFFUSION_FALLBACK.md#L94).

Another essential metric detail is that the reported “raw S.U.N.” uses the raw
DLM structure as the starting endpoint for the common CHGNet relaxation and
official hull calculation. It is not the CHGNet single-point energy of the raw
token geometry. Raw single-point energy, force, stress, and distance tails are
separate mechanism diagnostics.

The strongest direct evidence is the fixed 512-attempt comparison:

| Endpoint | Execution evidence | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|---:|
| B0 raw schedule | Raw Direct 80.27% | not the reported endpoint | not the reported endpoint |
| BC crystal transactions | Raw Direct 99.22% | not the reported endpoint | not the reported endpoint |
| BP Llama program | Raw Direct 98.05% | not the reported endpoint | not the reported endpoint |
| BS schedule-matched SPAD | Raw Direct 511/512 = 99.80% | raw 16/512 = 3.12% | raw 107/512 = 20.90% |
| B0 + model494 tau800 | Direct 508/512 | 38/512 = 7.42% | 245/512 = 47.85% |
| BC + model494 tau800 | Direct 512/512 | 33/512 = 6.45% | 238/512 = 46.48% |
| BS + model494 tau800 | Direct 512/512 | 35/512 = 6.84% | 234/512 = 45.70% |

Sources: [README:112](../README.md#L112) and
[README:128](../README.md#L128).

Therefore:

- composition and discrete execution are close to solved;
- native thermodynamic stationarity is not solved;
- model494 remains a real system-level contributor;
- model494 also weakens attribution because methods with very different raw
  execution converge to similar refined S.U.N.

The user's audit gives the corresponding physical picture: the generated
structures can pass a parser and a `0.5 Å` collision floor while remaining far
from a force/stress stationary low-energy basin. Repository measurements report
approximately `22 eV/Å` median force RMS versus `0.176 eV/Å` for quantized MP20,
approximately `100 GPa` stress norm versus `1.85 GPa`, minimum-distance median
`1.089 Å` versus `2.458 Å`, and Plan-VPA agreement `100/512` versus `396/512`.
See [basin-closure audit:26](teacher_feedback_unified_v1/12_LLAMA_PROGRAMMED_BASIN_CLOSURE.md#L26).

## 3. Current implemented architecture

The current branch implements this chain:

```text
C3FD calibrated chemical logits + reachable-action support
                  |
                  v
Llama residual logits on the same legal action set
                  |
       unit-weight product of experts
                  |
                  v
exact composition + Compact-V2 Plan
                  |
terminal Llama hidden state -> species permutation pointer
                  |
                  v
7+4N masked crystal DLM
                  |
                  v
SPAD predictor -> cell closure -> reverse species-block closure
                  |
                  v
raw native crystal
                  |
          optional PMTR (untrained)
                  |
                  v
optional frozen model494 tau800 -> refined crystal
                  |
                  v
CHGNet / official MP evaluation -> Direct, N/U/NU, S.U.N.
```

### 3.1 C3FD

C3FD supplies calibrated distributions and reachable support for:

- proposal family, atom count, and arity;
- species/count actions;
- lattice system, space-group bucket, and volume-per-atom bin.

It controls which chemical actions remain reachable. It has no fine lattice or
coordinate logits. The typed sampler normalizes C3FD and Llama on the same legal
support and adds their log probabilities. Illegal actions remain unavailable.
Code: [c3fd_llama_typed_planner.py:159](../src/crystal_dlm/c3fd_llama_typed_planner.py#L159)
and [sample_c3fd_llama_typed_planner.py:381](../src/scripts/sample_c3fd_llama_typed_planner.py#L381).

### 3.2 Planner-Llama and species pointer

Llama has two implemented roles:

1. residual preference over C3FD-legal typed actions;
2. an exact permutation of final Plan species.

The current pointer teacher is a periodic maximum-contact-tree derived from an
MP20 relaxed structure. It is not a hull-energy ordering and is not an observed
nucleation trajectory. Code:
[species_program_pointer.py:115](../src/crystal_dlm/species_program_pointer.py#L115).

The pointer is nontrivial—73.50% exact permutation, 80.41% root accuracy,
82.63% pairwise order, and 229/256 prospective programs are noncanonical—but BP
Raw Direct is 98.05%, below canonical BC at 99.22%. This proves that Llama can
change the executable order; it does not prove that the learned order is more
stable.

### 3.3 Compact Plan

`C3FD_NATIVE_PLAN_V2` contains exact `N`, elements, counts, anion family, lattice
system, space-group bucket, and VPA bin. Training and inference use the same
renderer. Code: [c3fd_native_plan.py:11](../src/crystal_dlm/c3fd_native_plan.py#L11)
and [c3fd_native_plan.py:94](../src/crystal_dlm/c3fd_native_plan.py#L94).

### 3.4 Dynamic crystal language

The exact body is

\[
z=(N,L_1,\ldots,L_6,(E_i,X_i,Y_i,Z_i)_{i=1}^{N}),
\qquad |z|=7+4N.
\]

`N` and element positions are fixed from the Plan. The DLM predicts six fine
lattice tokens and every fractional coordinate. Code:
[dynamic_crystal.py:214](../src/crystal_dlm/dynamic_crystal.py#L214).

The current token audit covers all MP20 `27,136/9,047` train/validation rows.
The DLM base uses the full split. The Planner pointer has only `24,558/8,158`
typed rows; its missing rows use canonical fallback in full-SPAD training. Do
not say that every module was trained on the same full subset.

### 3.5 SPAD

SPAD compiles the species program into noncontiguous position groups:

- lattice transaction;
- one anchor per species in Llama order;
- remaining sites;
- suffix-visible revisiting of the cell and reverse species blocks.

Code: [spad_program.py:196](../src/crystal_dlm/spad_program.py#L196),
[spad_generation.py:1006](../src/crystal_dlm/spad_generation.py#L1006), and
[spad_generation.py:1560](../src/crystal_dlm/spad_generation.py#L1560).

The current implementation still predicts six lattice components and X/Y/Z
components sequentially. Cell and site/block changes are committed atomically;
invalid complete transactions roll back. Strict triclinic periodic support is
applied during decoding. The evidence for 99.8% execution is the combined SPAD
system; it is not an isolated estimate of the Llama order or one logit mask.

### 3.6 model494

The actual sampler starts from supplied proposal coordinates and lattice, sets
the reverse-time index to `diff_steps`, and updates both coordinates and lattice
unless `keep_coords` or `keep_lattice` is enabled. It does not explicitly apply
the textbook forward corruption before the reverse loop. Code:
[diffusion.py:90](../src/vendor/crysllmgen/models_ddpm/diffusion.py#L90).

Thus two common statements are wrong for this code:

- “tau800 explicitly replaces 90% of the input by forward Gaussian noise”;
- “model494 keeps the input lattice fixed.”

Empirically tau800 can still wash out distinctions, but that interpretation
must follow observed before/after behavior, not the textbook forward formula.

### 3.7 PMTR

PMTR is an optional, newly implemented but untrained route:

```text
coherent SPD/PBC corruption of an MP20 train structure
    -> offline CHGNet certification after exact token quantization
    -> frozen SPAD hidden states
    -> small repair head predicts a symmetric metric tangent and Cartesian site vector
    -> continuous target is rendered as legal old-to-target token-logit transport
    -> existing SPAD hard support, commit, and rollback remain authoritative
```

The production runtime contains no CHGNet, energy, force, stress, reward,
candidate pool, or reranking call. Code:
[manifold_repair_head.py:83](../src/crystal_dlm/manifold_repair_head.py#L83),
[pmtr_runtime.py:126](../src/crystal_dlm/pmtr_runtime.py#L126), and
[pmtr_training.py:402](../src/crystal_dlm/pmtr_training.py#L402).

The first planned run freezes the existing SPAD DLM/LoRA and trains only the
repair head. If successful, the precise claim would be that a
DLM-conditioned repair head learned a native token repair—not that the whole
DLM backbone learned a physical potential.

## 4. What is already supported

| Claim | Evidence | Status |
|---|---|---|
| C3FD can construct chemically valid compositions | C3FD-v2.5 2000/2000; fused Planner 256/256 and scale 1200/1200 | supported |
| Llama materially changes probabilities within C3FD support | mean fused-vs-C3FD KL 0.06819; 87.05% of 2,038 actions have nonzero KL | supported |
| Llama emits noncanonical executable species programs | pointer metrics and 229/256 noncanonical programs | supported |
| DLM/SPAD can execute exact chemistry and nearly complete discrete geometry | BS Raw Direct 511/512 | supported |
| Suffix-visible DLM revision is a real capability | real-checkpoint later-token intervention changes earlier XYZ logits | supported |
| model494 substantially improves absolute stability/execution | BS raw 16/107 -> tau800 35/234 Strict/Meta | supported system effect |
| High Direct validity implies high stability | contradicted by 99.8% Direct and 3.12% raw Strict | unsupported |
| Llama's learned order is better than canonical | BP 98.05% Raw Direct vs BC 99.22%; no energy gain established | unsupported |
| Current native DLM reaches 10%/50% | no current prospective endpoint does | unsupported |
| PMTR improves anything | no optimizer update or checkpoint exists | untested |

## 5. Complete experimental iteration record

This table records scientific iterations, not every shell-level recovery. Major
engineering negatives are included when they changed interpretation or consumed
the only attempted execution.

| Stage | Intervention and scale | Main result | Decision |
|---|---|---|---|
| R5C fixed-slot | padded 107-token target | exact-length engineering only | retired; padding was not stability |
| R5C exact dynamic | variable exact length with structured/replay conditions | 1167/1200 graphs; refined structural validity about 99.8% | proves dynamic length can execute; not a de-novo stability claim |
| H1-B formula-only | composition without rich structural context | 5.54% Strict, 43.13% Meta; 81.57% all-90° cells and repeated-length collapse | direct warning against composition-only CE |
| H1-A2 historical | learned rich Plan + DLM + model494 | traceable compatibility view 94/1000 Strict and 474/1000 Meta; repository also carries aggregate headline 105/488 | useful high point, provenance conflict remains |
| H1-A2 exact/continuous replay | all requested 1200 and pooled 3840 views | 8.58/46.08% and 7.63/45.47% | historical effect is process/accounting fragile but nonzero |
| R03 D1/D2 | global X→Y→Z and Safe-axis process variants | D1 9.67/51.07%; D2 11.43/48.44%; corrected replays 8.42/46.58 and 7.47/45.26 | schedule effects are fragile; D2 repeats are not independent seeds |
| More rich CE | rich SFT epoch2 vs epoch3 | body 985→992 and Direct 871→878, while Strict 81→79 and Meta 489→477 | CE/execution and stability are different objectives |
| Counterfactual rich grounding | factual vs counterfactual global Plan fields | token margin +0.759, but fixed1000 Strict 89→86 and Meta 487→467 | richer fields affect likelihood, not reliably low energy |
| CTV token value | 320 states, 3,072 branches, one-token action head | centered Spearman 0.0353, AUC 0.5053, continuation agreement 0.4915 | global stability cannot be assigned to one token; no-go |
| SGTC positive-only continuation | stable MP20 positives vs all MP20 | body/Direct retained; Strict/Meta 53/417 vs base 60/412 and adverse energy/hull | positive-only CE lacks a same-composition boundary |
| D3PO fixed256 | full-sequence preference from post-refiner energy | refined mean about -2 to -3 meV and Meta +1.76/+3.52 pp; Strict unchanged; raw +0.200 eV and one seed worse | weak refiner-mediated signal; raw exploitation risk |
| C3FD-v2.5 | constructive exact composition/reachability | composition 1724/2000→2000/2000; NU 1530→1756 | composition support solved and frozen |
| Compact-V2 fresh SFT | job38703; full MP20 27,136/9,047; two seeds; 3,392 updates = two epochs | both seeds finite; canary raw Direct 30.08/37.30%, refined 89.06/90.04% | schema executes, raw manifold remains poor |
| Compact-V2 development S.U.N. | MP20-overlapping 512-row development | raw about 1.4–1.6%/11.7–12.3%; refined 8.203% Strict and 54.88–55.27% Meta | useful development signal, not prospective |
| Faithful H0/R0S | historical full-schema diagnostics | refined H0 8.789/41.992%; R0S 7.031/42.188% | canonical rich completion did not restore H1-A2 |
| F/M rich expanders | C3FD formula-only F vs C3FD soft-prefix M, 512 attempts | comp-valid 97.66/98.24%; raw Direct 42.38/44.73%; refined Strict/Meta F 6.641/38.281%, M 7.617/37.695% | M kept for integrated conditioning, not energy; later superseded |
| Fused typed Planner | jobs39046/39051/39088; 24,558 train rows; one epoch, 1,535 updates | 256/256 composition-valid; body/comp 248/256; raw Direct 106/256 | Llama reweights legal composition, but Planner data are not full MP20 |
| G0 representation audit | fused fixed256 | parse 8, composition 0, lattice 0, PBC collision 142, pass 106; exact roundtrip | token quantization was not the Direct failure source |
| G1 geometry loss | job39103, one seed, 348 updates | body 245/256, Direct 115/256 vs base 248/106 | Direct +9 but body floor failed; downstream stopped |
| G2 periodic residual | job39107, one seed, 348 updates | body/comp 248; Direct 119 vs base106; large raw→refined energy repair | promising execution adapter, raw energy not established |
| Fresh prospective G2 | jobs39128/39137/39139, fixed256 | raw BASE/G2 Strict 5/9, Meta 41/47; refined BASE 19/111, G2 24/117 | 9.375/45.703%; misses 10/50; refined hull -16.43 meV |
| Full-epoch G2 A/B | job39172, fixed256 | refined A/B 23/117 and 23/115; B no advantage | keep A as historical development; G2 removed from teacher-feedback mainline |
| Plan1200 scale | jobs39183/39199/39204/39205 | 1200 valid Plans; 1139 valid CIFs; documented composite1000 result 81/486 | scale result below 10/50; conflicts with 105/488 narrative |
| Teacher-feedback SPAD | B0→BC→BP→BR→BS | Raw Direct 80.27→99.22→98.05→98.83→99.80% | exact structured execution solved; stability open |
| Basin-closure CE | job39700; full MP20; 1,696 updates | raw BS 7/54 vs closure CE 7/56; median CHGNet -0.3244 eV, post-relax hull -0.02274 eV | strong continuous shift, only 0/+2 threshold counts |
| Response-backfill | fixed 512 | raw Strict 16→21, Meta 107→106; refined Strict about35→36 | single-site correction is too weak/inconsistent |
| Potential-Closure pilot | 512 sources, 2,048 states, 6,883 candidates, 2,048 updates | tau800 20/125 vs control17/115 and BS18/116; raw potential10/58 | best near-line result, but refiner-mediated and not replicated |
| Tau bridge | closure system fixed256 | tau200 15/98; tau400 11/110; tau600 19/121; tau800 23/119 | low tau did not replace tau800; cohorts must not be mixed |
| K10 pilot | 128 groups, K≤4, 512 updates | local headroom real; implementation learned candidate-relative ranking | motivated scale-up but objective/deployment mismatch remained |
| K10 scale pass 1 | jobs39792/39794/39796/39799; 4,104 states, 15,348 candidates, 2,736 updates | stream19 raw 12/61; tau800 14/107 | exact execution; below target |
| K10 scale pass 2 | job39805 plus jobs39806/39807/39810/39814 | stream21 raw 5/49; tau800 14/123 | more training raises refined Meta but not Strict; raw regresses |
| PCTP reward proposal | design only | recognized as generic reward/value distillation with prior-art and attribution problems | rejected before implementation |
| PMTR | current branch, design and preflight only | data certificate succeeded; all training attempts failed before first update | hypothesis remains untested; implementation currently nonfunctional |

Primary sources:

- [capability regression audit](DLM_CAPABILITY_REGRESSION_36H.md);
- [stability program audit](DLM_STABILITY_PROGRAM_AUDIT_20260830.md);
- [Compact-V2 training final](C3FD_NATIVE_TEACHER_SFT_FINAL_20260831.md);
- [Compact-V2 canary](C3FD_NATIVE_SFT_CANARY_OFFLINE_FINAL_20260831.md);
- [F/M checklist](C3FD_LLAMA_DLM_SUN_CHECKLIST_V6.md);
- [fresh G2 final](36H_FINAL_REPORT_C3FD_G2_20260901.md);
- [full-epoch G2 final](G2_FULL_EPOCH_AB_FINAL_20260901.md);
- [Plan1200 final](PLAN1200_TAU800_FINAL_20260902.md);
- [SPAD/K10 checklist](teacher_feedback_unified_v1/04_EXECUTION_CHECKLIST.md).

## 6. Current PMTR proposal and execution record

### 6.1 Intended scientific object

PMTR replaces reward/ranking with supervised retraction to the original MP20
relaxed structure. It constructs a coherent joint corruption on

\[
\mathrm{SPD}(3)\times\mathbb T^{3N},
\]

quantizes it through the exact native codec, and uses CHGNet only to certify
offline that the corrupted state is uphill and the clean retraction is locally
downhill. CHGNet values do not enter the loss or production inference.

At a program-matched repair state, the head predicts:

- a symmetric lattice-metric tangent;
- a translation-free Cartesian site correction.

The target is rendered into the active legal special-token family as

\[
\Delta\ell=g\,[\phi(\hat y)-\phi(y_{\rm old})].
\]

The DLM/LoRA is frozen in the first run. Clean identity and corrupt-repair
microbatches alternate. The original formal proposal called for all 27,136
MP20 train sources and two epochs.

### 6.2 Implemented modules

Implemented and tested at unit level:

- SPD/PBC geometry: `manifold_geometry.py`, `periodic_geometry_ops.py`;
- coherent corruption: `manifold_corruption.py`;
- offline certification: `offline_pmtr_certification.py`;
- transaction hook: `transaction_logits.py` plus SPAD integration;
- repair head: `manifold_repair_head.py`;
- token renderer: `manifold_token_transport.py`;
- runtime/checkpoint: `pmtr_runtime.py`, `pmtr_checkpoint.py`;
- data builder and trainer: `build_pmtr_preflight_data.py`,
  `pmtr_training.py`, `train_pmtr.py`;
- production opt-in integration: `sample_llada_r5_exact_length.py`;
- paired physics evaluator: `evaluate_pmtr_transfer_physics.py`.

The fixed-body application script and its test remain **untracked working-tree
files**:

- `src/scripts/apply_pmtr_fixed_bodies.py`;
- `tests/test_apply_pmtr_fixed_bodies.py`.

They were stopped before review or commit.

### 6.3 Actual PMTR jobs

| Job | Resources/time | Terminal | Meaning |
|---|---|---|---|
| 39824 | 4 A800, 16 CPU, 41 s | COMPLETED 0:0 | 2,048 proposals; 1,966 certified; 510/512 sources have ≥1 certificate |
| 39825 | 4 A800, 2:09 | FAILED before update | bfloat16 frozen hidden state entered float32 LayerNorm before casting |
| 39826 | 4 A800, 1:11 | FAILED before update | float32 head Cartesian delta and bfloat16 lattice violated geometry dtype contract |
| 39827 | 4 A800, 1:34 | FAILED before update | first gradient probe produced non-finite PMTR parameter gradient on all ranks |

Job39824 produced:

- fit: 384 rows, 382 repair + 2 clean fallback;
- holdout: 128 rows, 128 repair;
- disjoint actual-SPAD transfer cohort: 128 rows;
- no selection by minimum energy; the builder used the first certified proposal
  in fixed proposal order.

Jobs39825/39826 were deterministic mixed-precision integration failures and
were patched. Job39827 is more consequential: the failure occurred in the first
gradient-scale probe, before any optimizer update, so no checkpoint or
scientific result exists.

The traceback only establishes a non-finite gradient. A plausible but unproven
cause is differentiation through SPD eigendecompositions at zero tangent or
repeated eigenvalues, possibly compounded by piecewise token interpolation at
quantization knots. GPT-6 should inspect the exact autograd path before accepting
that diagnosis.

### 6.4 What the successful certificate does and does not show

`510/512` certificate coverage shows that the chosen synthetic corruption
generator can usually produce a quantized MP20-neighbor state whose clean
direction is locally downhill under CHGNet.

It does not show that:

- the head can train;
- real SPAD errors resemble those corruptions;
- PMTR lowers force, stress, energy, or hull;
- raw S.U.N. improves;
- refined S.U.N. improves;
- model494 can be removed.

PMTR is therefore not a scientific negative yet. It is an untested hypothesis
with a currently broken differentiable implementation.

### 6.5 Design-to-code mismatches already visible

These are current code facts, not speculative future concerns:

1. **Corruption is not regenerated each epoch.** The design says each source
   receives a new coherent corruption in each epoch. The builder currently
   materializes one fixed corruption and one transaction row; the trainer's two
   epochs replay that fixed row.
2. **One source supervises one transaction, not a full sweep.** The selected
   scalar state is promoted to its full cell or XYZ transaction, but a source
   does not teach the complete cell-plus-all-sites repair sequence executed at
   inference.
3. **Teacher-forced repaired prefixes differ from deployment prefixes.** The
   builder inserts clean MP20 values for earlier repaired transactions. At
   inference, later repairs condition on earlier PMTR samples, which can be
   imperfect or rolled back.
4. **The certificate does not explicitly enforce the SPAD `0.5 Å` PBC floor.**
   Its post-quantization validity checks token changes, clipping, and species;
   successful CHGNet evaluation is not identical to the deployed PBC support.
5. **Stress-aligned lattice supervision was replaced.** Stress is recorded as a
   diagnostic. The compatibility field named
   `lattice_descent_dot_spd_retraction` stores the energy decrease of a small
   joint toward-clean probe, not a stress/tangent contraction.
6. **The documented reference loss is absent.** The implemented total is token
   CE + SPD MSE + torus MSE + step regularization. Clean identity batches give
   an indirect anchor, but there is no explicit reference-logit KL.
7. **Gradient scales are observed, not balanced.** The first five probes print
   component gradient norms. No calibration is applied before directly adding
   token and continuous losses.
8. **Training and inference head precision differ.** The trainer keeps the head
   in float32; checkpoint loading casts it to the DLM output embedding dtype,
   normally bfloat16. Only subsequent periodic algebra is forced to float32.
9. **The head does not explicitly receive the numerical current metric.** It
   receives DLM hidden states, Plan context, species/program rank, and MIC pair
   vectors. The old lattice is decoded and used outside the head when applying
   the predicted tangent.
10. **The coordinate parameterization is center-force-like.** A site correction
    is a sum of MIC unit directions weighted by symmetric pair scalars. This is
    periodic and permutation structured, but it may not span arbitrary angular
    or low-coordination retractions.

Code evidence:
[builder:263](../scripts/build_pmtr_preflight_data.py#L263),
[trainer loop:285](../src/scripts/train_pmtr.py#L285),
[certificate:200](../src/crystal_dlm/offline_pmtr_certification.py#L200),
[actual loss:421](../src/crystal_dlm/pmtr_training.py#L421), and
[head messages:242](../src/crystal_dlm/manifold_repair_head.py#L242).

## 7. Why the prior stability routes missed the scientific object

### 7.1 Validity was treated as stability

Schema CE, exact chemistry, PBC collision rejection, and rollback raise
`P(V_c)`. They do not supply the force/stress/energy information required to
raise conditional basin occupancy. The 99.8% Direct versus 3.12% raw Strict
gap is the decisive counterexample.

### 7.2 Ordinary masking conditions on the wrong state

Ordinary masked CE usually sees clean visible geometry and predicts clean
masked tokens. Deployment repair sees many wrong-but-visible lattice and
coordinate tokens. A DLM trained on clean context has no reason to interpret a
visible high-energy token as something to revise.

### 7.3 Some rollout states spliced incompatible geometries

Historical generated-prefix/clean-suffix states could combine an erroneous
cell or coordinate prefix with later clean MP20 tokens that no longer represent
one coherent periodic crystal. The target was statistically defined but not
necessarily geometrically realizable under the visible context.

### 7.4 Local actions received global terminal credit

CTV, response-backfill, Potential-Closure, and K10 attempted to assign a global
or post-refiner endpoint to one token, one XYZ, one cell, or a few anchors. A
collective lattice–coordinate basin can require many coupled changes. Real
candidate headroom does not imply that a local control surface can move the
complete free-generation distribution.

### 7.5 Candidate-relative ranking was not deployment probability

The K≤4/K10 implementation normalized action probabilities within retained
candidates. It could learn which candidate is better without materially raising
that complete action's probability against the full deployed vocabulary. The
planned raw `E0` safety constraint was also not implemented in the final local
objective. More epochs could not repair this estimand mismatch.

### 7.6 Post-model494 reward confounded credit

A tau800 terminal value combines the chosen DLM action, subsequent DLM
generation, stochastic model494 dynamics, and CHGNet approximation. Because
tau800 empirically compresses differences between raw methods, an improved
refined threshold count does not identify which earlier discrete decision was
responsible.

### 7.7 Pointer accuracy was not physical-order accuracy

The current contact-tree teacher is more physical than CIF serialization order,
but it still does not label low-energy program order, a site-level construction
trajectory, a kinetic barrier, or nucleation dynamics. Pointer accuracy proves
teacher imitation, not stability.

## 8. PMTR's scientific alignment and remaining mismatch

PMTR improves the object definition in three ways:

1. cell and coordinates are corrupted jointly in a periodic continuous space;
2. unrepaired future geometry remains coherently wrong rather than being spliced
   from another state;
3. continuous repair is compiled into native legal token probabilities, so
   inference needs no MLIP.

However, the central extrapolation remains unproven:

\[
\text{reconstruct a known MP20 polymorph from a local corruption}
\quad\not\Rightarrow\quad
\text{sample a low-hull basin for a novel free-generated composition}.
\]

That implication requires at least all of the following assumptions:

- real SPAD errors lie near the relaxed MP20 manifold rather than in a different
  topology or basin;
- the local retraction is identifiable from frozen DLM hidden states;
- one cached cell vector and sequential cached XYZ vectors form a sufficient
  control surface for collective basin errors;
- the repair head generalizes from seen MP20 structures to new compositions;
- discretized token transport preserves the continuous direction;
- improved local force/stress is large enough to cross Strict/Meta hull
  thresholds;
- model494 does not erase the gain.

None is experimentally closed. The planned actual-SPAD transfer test was never
reached.

## 9. Internal factual conflicts requiring adjudication

### 9.1 `105/488` versus traceable result views

The repository freezes a paper headline of `105/1000` Strict and `488/1000`
Meta. It also states that these are aggregate headline values and must not be
used to fabricate row-level microdata. Separately traceable views include:

- H1-A2 historical compatibility: `94/1000`, `474/1000`;
- H1-A2 exact replay: `8.58%`, `46.08%`;
- Plan1200 composite1000: `81/1000`, `486/1000`.

The Plan1200 document itself opens by mentioning `105/488` but its result table
reports `81/486`. These numbers must not be averaged, relabeled, or presented as
the same cohort. GPT-6 should require run-level provenance for the headline.

Collaboration history also contains an outcome-directed instruction to move
known Strict successes from a remainder block into main1000 while preserving a
larger headline. The final artifact claims parser-only first-valid selection.
The selection ledger, not prose, must decide which account is correct.

### 9.2 Strict/Meta definition

The user's pasted audit describes Strict using `E_hull < 0.1` plus a phonon
condition. The implemented official metric uses Strict `E_hull <= 0` and Meta
`E_hull <= 0.1`, with no phonon evaluation. The latter is the code/repository
definition for current numbers.

### 9.3 Lattice metric convention

The repository uses row-vector fractional coordinates, `R = X L`, so its
fractional metric is

\[
G=L L^T.
\]

Several prose documents write `G=L^T L`. GPT-6 must audit every PMTR tangent,
stress, and Cartesian/fractional conversion under one convention. Code:
[manifold_geometry.py:155](../src/crystal_dlm/manifold_geometry.py#L155).

### 9.4 model494 semantics

The pasted audit assumes textbook forward Gaussian corruption and a fixed cell.
Current code starts the reverse process at the supplied proposal and updates
both lattice and coordinates. Empirical washout may be real, but the stated
mechanism in the pasted audit is not the executed code path.

### 9.5 Pointer teacher

Some historical descriptions say the pointer learns CIF serialization order.
The current branch uses a maximum-contact-tree teacher from relaxed structures.
The latter is the current code fact; neither is direct stability supervision.

### 9.6 Direct baselines

`511/512`, `504/512`, `255/256`, and `256/256` all appear as SPAD structure
results. They refer to different cohorts, endpoint definitions, or evaluator
coverage. They cannot be treated as repeated measurements of one identical
baseline without a row-level mapping.

### 9.7 PMTR document status

The PMTR documents say GPU preflight had not started. Actual execution reached
certificate construction and three failed pre-update training jobs. This
handoff supersedes those status headers but not their method descriptions.

### 9.8 “Strict triclinic MIC” scope

The current operator searches a centered radius-two image shell (125 images).
That is the deployed finite search, not a proof of the global minimum image for
arbitrarily pathological skew cells. Code:
[periodic_geometry_ops.py:65](../src/crystal_dlm/periodic_geometry_ops.py#L65).

## 10. Current code and workflow map

| Layer | Entry points | Input | Output/effect | Current evidence |
|---|---|---|---|---|
| C3FD legal support | `c3fd_planner_model.py`, `sample_c3fd_llama_typed_planner.py` | typed partial chemical state | legal masks + calibrated logits | comp-valid established |
| Llama residual Planner | `c3fd_llama_typed_planner.py` | C3FD typed state and Llama hidden | residual proposal/action/soft-field logits | nonzero reweighting established |
| species pointer | `species_program_pointer.py` | terminal Llama state + Plan | exact element permutation | prediction metrics; no stability causality |
| Plan serializer | `c3fd_native_plan.py` | exact composition + coarse fields | Compact-V2 prompt | train/serve interface audited |
| crystal codec | `dynamic_crystal.py` | lattice/species/coordinates | exact `7+4N` body | full MP20 coverage |
| SPAD program | `spad_program.py` | Plan + species order | predictor/revision slot groups | code and execution verified |
| SPAD runtime | `spad_generation.py` | masked body and support | transaction commit/rollback | Raw Direct near saturation |
| PMTR corruption | `manifold_corruption.py` | clean MP20 train structure | coherent quantized neighbor | 510/512 source certificate coverage |
| PMTR certificate | `offline_pmtr_certification.py` | clean/corrupt/probe structures | train-only local-downhill certificate | data path works |
| PMTR head | `manifold_repair_head.py` | frozen DLM hidden + PBC geometry + program rank | metric/site vector | unit tests only; real gradient fails |
| PMTR renderer/runtime | `manifold_token_transport.py`, `pmtr_runtime.py` | continuous repair + legal token family | active token logit residual | unit tests only |
| PMTR trainer | `pmtr_training.py`, `train_pmtr.py` | paired coherent states | intended head checkpoint | no optimizer update completed |
| model494 | `models_ddpm/diffusion.py` | raw graph | stochastic refined lattice/coords | strong system-level benefit |
| evaluators | Direct, CHGNet, official MP finalizers | raw/refined structures | validity, energy, hull, SUN | multiple completed cohorts |

### 10.1 Normal current training flow

```text
MP20 train structure
  -> Compact Plan / exact 7+4N teacher body
  -> Llama contact-tree species program
  -> inference-matched SPAD mask state
  -> masked CE on clean transaction
  -> SPAD LoRA checkpoint
```

### 10.2 Normal current inference flow

```text
C3FD+Llama samples one legal Compact Plan and one species program
  -> N/elements prefilled
  -> DLM produces one predictor body
  -> SPAD cell closure
  -> reverse Llama species-block closure with suffix visible
  -> raw body
  -> optional model494 tau800
  -> evaluation
```

There is one Plan and one trajectory. No retry, best-of-N, energy reranking, or
inference CHGNet is part of the retained contract.

### 10.3 Intended PMTR training flow

```text
MP20 train x0
  -> coherent lattice+Cartesian corruption xt
  -> exact token roundtrip
  -> offline CHGNet local-downhill certificate
  -> program-matched transaction-start state
  -> frozen SPAD forward and hidden states
  -> train repair head with token + SPD/torus targets
  -> pmtr_final.pt
```

This flow stopped at the first gradient probe.

### 10.4 Intended PMTR inference flow

```text
one completed SPAD raw body
  -> PMTR cell proposal cached over six scalar token predictions
  -> commit or rollback
  -> recompute geometry
  -> PMTR XYZ proposal cached per site in reverse Llama program order
  -> existing support and rollback
  -> one repaired raw body
```

No trained checkpoint exists, so this flow has only fake-model/unit-test
coverage.

## 11. Questions GPT-6 should answer

### Scientific object

1. Is the paper's true target local relaxed-manifold reconstruction, low-hull
   conditional generation, or end-to-end discovery utility?
2. Under what assumptions does supervised `x_t -> x_0` retraction increase
   free-generation basin occupancy for new compositions?
3. Are current raw SPAD errors local geometric perturbations or wrong global
   topologies/basins?
4. Is a fixed-composition low-energy polymorph distribution identifiable from
   the current MP20 teacher data without generated low-energy teachers?

### Architecture

5. Does static Llama species order plus program-rank conditioning constitute a
   genuinely integrated LLM+DLM method, given BP does not beat canonical order?
6. Does head-only PMTR preserve the claimed DLM contribution, or is it another
   external adapter?
7. Is one cell transaction plus sequential site transactions expressive enough
   for collective periodic modes?
8. Are `7+4N` scalar tokens a suitable action space for coupled lattice and
   coordinate stationarity?

### Mathematics and implementation

9. Is every metric, tangent, MIC vector, and Cartesian/fractional conversion
   consistent with row-vector `G=L L^T`?
10. What exact operation produced the non-finite token-gradient in job39827?
11. Is eigendecomposition-based SPD log/exp differentiable enough near cubic or
    repeated-eigenvalue cells?
12. Does `searchsorted`/piecewise bracket interpolation provide a stable
    gradient at exact quantization bins and periodic aliases?
13. Does zero initialization preserve exact output while still giving useful
    gradients to every output branch?
14. Does transaction-start training match cached-proposal inference for all six
    cell and three XYZ components?
15. Are the fixed-corruption epochs, one-transaction-per-source supervision,
    and teacher-forced clean prefixes materially different from the method
    described in the PMTR paper draft?
16. Is the current center-force coordinate head and hidden-only lattice head an
    expressive enough control surface for the observed force/stress errors?
17. Does casting a float32-trained head to bfloat16 at inference preserve the
    learned repair field?

### Evaluation and paper claims

18. Which exact run and row ledger supports `105/488`?
19. Which numbers are fresh prospective, overlapping development, historical
    replay, or parser-conditioned subsets?
20. Can the current paper honestly claim one causal chain beyond three separate
    facts—C3FD validity, SPAD execution, and model494 refinement?
21. What is the smallest evidence set needed to demonstrate one connected core
    contribution rather than multiple independent modules?
22. Can model494 remain in the final system without obscuring the DLM result,
    and what paired raw/refined attribution is required?
23. Given the remaining time, which already-proven assets must be frozen and
    which unproven branch should be abandoned before more compute?

## 12. Requested GPT-6 deliverable

Please return five explicit verdicts:

1. **Scientific-question verdict:** one mathematical task that matches the data,
   model, and metric rather than restating the architecture.
2. **Architecture verdict:** which existing modules form one necessary chain,
   and which are merely adjacent add-ons.
3. **PMTR verdict:** `NUMERICAL_FIXABLE`, `SCIENTIFICALLY_MISALIGNED`,
   `CONTROL_SURFACE_INSUFFICIENT`, or a clearly defined combination, with code
   evidence.
4. **Evidence verdict:** a reconciled result table with provenance and a list of
   claims that must be removed or weakened.
5. **Next-action verdict:** at most one high-value implementation/experiment,
   plus a stop condition; do not propose a broad experiment matrix.

## 13. Frozen operational state

- No Slurm job is active.
- PMTR job39824 data are preserved.
- Jobs39825/39826/39827 are preserved failures.
- No full PMTR corpus, formal PMTR training, transfer evaluation, prospective
  generation, model494 run, official query, or paper1000 run was started.
- The Materials Project credential is intentionally absent from this document.
- The implementation snapshot before adding this audit document is `3f67509`
  and was synchronized with its remote.
- Two untracked fixed-body PMTR files remain for audit; they are not an approved
  or completed implementation.

This is the state from which GPT-6 should audit. It should not assume that the
latest proposed method is valid merely because earlier modules are individually
successful.
