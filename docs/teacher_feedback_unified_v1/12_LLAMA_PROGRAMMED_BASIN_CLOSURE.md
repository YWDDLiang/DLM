# Llama-Programmed Basin Closure

Status: **implementation approved; scientific training waits for closure-state replay tests**

## 1. Problem

The validated SPAD system has nearly saturated benchmark execution validity,
but not native thermodynamic conversion. On the fresh prospective cohort, BS
has raw Direct `504/512 = 98.44%`, while its no-model494 Strict/Meta S.U.N. is
only `16/512 = 3.12%` and `107/512 = 20.90%` after the common CHGNet
relaxation.

For fixed composition `c`, the broad executable set

\[
\mathcal V_c=\{(L,X):d_{\min}^{\rm PBC}\ge0.5\,\text{A},\;V>0\}
\]

is much larger than the useful low-energy basin set

\[
\mathcal B_c(\delta)=\{(L,X)\in\mathcal V_c:
E_{\rm hull}(R(L,X),c)\le\delta\}.
\]

Current evidence shows that SPAD puts most mass in `V_c`, not in `B_c`.
Generated stream17 structures have median CHGNet force RMS around `22 eV/A`
and stress norm around `100 GPa`, versus roughly `0.176 eV/A` and `1.85 GPa`
for quantized MP20 references. A fixed 512-row same-composition and
teacher-Plan sample from the full-MP20 predictor run has minimum-distance
median `1.089 A` versus teacher `2.458 A`, and Plan-VPA agreement `100/512`
versus teacher `396/512`.

The target is therefore not another validity rule. It is a trainable,
programmed DLM closure that reduces lattice-coordinate nonstationarity and
then prefers legal closure actions whose completed trajectories enter lower
local basins.

## 2. One connected method

```text
C3FD reachable chemical support
  -> Planner-Llama Compact Plan + species_program
  -> SPAD 7+4N predictor
  -> full crystal visible
  -> trained six-token cell closure L | X
  -> reverse Llama-program species-block coordinate closure
  -> native raw crystal
  -> optional frozen model494 fallback
```

C3FD keeps its existing role: exact chemical state and reachable actions.
Llama keeps its existing role: residual Plan preferences and an immutable
permutation of the certified species. The masked DLM remains the sole owner of
lattice and coordinate values.

The same `species_program` controls both construction and closure. Llama
supplies the chemical block factorization and order; the DLM executes
conditional kernels over those blocks while retaining the complete future
crystal.

## 3. Runtime contract

Current BS remains unchanged:

```text
predictor -> reverse unique-species anchor backfill
```

The new method is an opt-in path and must not alter `--spad-backfill`:

```text
predictor -> cell closure -> reverse species-block closure
```

### 3.1 Cell closure

After all coordinates are committed, mask the six lattice fields and resolve
them in native order while every coordinate remains visible:

\[
L^{(1)}\sim q_\theta(L\mid X^{(0)},c,P,\pi).
\]

The six values are externally atomic. A non-positive or PBC-unsupported final
cell restores the complete previous cell. Formal inference requires checkpoint
metadata proving that full-coordinate-visible cell closure was trained.

### 3.2 Species-block closure

Visit species blocks in reverse `species_program` order. At the start of a
block, mask all XYZ fields belonging to that species. Resolve its sites in the
exact reverse of their predictor order, with `X -> Y -> Z` inside each site.
The lattice, every other species block and the completed textual suffix stay
visible. A site with no legal Z completion restores its old XYZ transaction.

Only one fixed closure sweep is allowed. There is no repeated repair,
candidate selection, reranking or result-dependent stopping.

## 4. Geometry-recovery training

Create a new dataset; never mutate the retained BS corpus or checkpoint. Use
all MP20 train rows with teacher Compact Plans and the frozen Llama/fallback
programs.

Each row deterministically contributes one runtime-matched state:

- cell step `j`: mask lattice fields `j..5`, supervise only field `j`, and keep
  all coordinates visible;
- species-block step: keep other species and the full suffix visible, mask the
  active component plus every not-yet-committed coordinate in the active
  block, and supervise only that component.

Exact `N` and element positions are always visible. `answer` is the clean MP20
teacher. `source_answer` is either that teacher or a bounded perturbation of
the same structure. Perturbations may include cell strain, periodic coordinate
jitter and one injected short pair, but they never change composition, body
length, site order or target polymorph.

The existing LLaDA SFT trainer already supports `source_answer`,
`forced_mask_positions` and `loss_positions`; no backbone, tokenizer or new
continuous head is required.

## 5. Basin-value stage

This stage is conditional. First use 128 frozen on-policy closure states to
measure whether the existing 6-token cell and 3-token XYZ actions have useful
terminal headroom.

For an action `a` at state `s`, execute every remaining frozen closure step and
then one calibrated short CHGNet relaxation:

\[
Q_K(s,a)=-E_{\rm CHGNet}
\left(R_K(T_{\rm remain}(T(s,a)))\right).
\]

This replaces the rejected instantaneous label `-E(T(s,a))`. Candidate support
contains at most four outcome-blind legal actions: no-op, one reference-DLM
action and two quantized force/stress-direction actions. Invalid actions have
zero support before value is read.

The preferred finite posterior remains close to the frozen closure policy:

\[
q^*(a\mid s)\propto p_{\rm ref}(a\mid s)
\exp[-\beta\widehat E_K(a)]\mathbf1[a\in\mathcal A_{\rm legal}],
\qquad D_{\rm KL}(q^*\Vert p_{\rm ref})\le\epsilon.
\]

Do not impose a hard instantaneous-energy constraint: a useful basin action
can temporarily rise before the fixed continuation and relaxation. Raw energy,
force and stress remain mandatory diagnostics. Score complete actions using
the actual temperature, sequential component conditioning and schema/PBC masks.

If the preflight shows material headroom and usable short-K/full-relax ranking,
train one seed on 2,048--4,096 value sources. Alternate clean closure CE and
posterior updates; never calculate clean CE on generated states. Full-source
species-block K4 training is outside the first pass because its sequential
forward/backward cost is prohibitive.

## 6. Evaluation

The first matrix has only three arms:

| Arm | Question |
|---|---|
| BS | Is high execution validity sufficient? |
| closure-CE | Does runtime-matched non-causal closure reduce physical nonstationarity? |
| closure-basin | Does terminal basin value improve native stability beyond equal-compute closure CE? |

Reuse frozen BS. Generate only the two new arms. Evaluate raw first:

- composition and fast structural validity;
- minimum-distance distribution and the sub-`0.75 A` tail;
- Plan-VPA agreement;
- CHGNet force RMS and stress norm;
- energy removed by the common relaxation;
- raw Strict/Meta S.U.N. and N/U;
- paired energy and stable/unstable wins and losses.

Do not rerun expensive Direct on the first screen. Run fixed model494 only if a
native improvement exists, and label that result as the refined-system
fallback rather than native-DLM evidence.

## 7. Claim and stop boundaries

The method succeeds scientifically only if closure-CE reduces physical
nonstationarity without losing execution validity and closure-basin improves a
native stability endpoint over equal-compute closure-CE. Loss reduction alone,
proxy improvement alone or tau800-only improvement is insufficient.

If the 128-state preflight has negligible terminal headroom, stop the value
path. More epochs or more sources cannot rescue an action space that cannot
express a useful basin transition. If canonical order matches or beats the
Llama program, keep the DLM closure result but do not claim that Llama learned
an energy-optimal order. Program-value training is a later conditional step.

The paper description is **Llama-programmed non-causal basin closure**, not
thermal-equilibrium sampling, exact force integration, online CHGNet guidance
or end-to-end Llama-DLM training.
