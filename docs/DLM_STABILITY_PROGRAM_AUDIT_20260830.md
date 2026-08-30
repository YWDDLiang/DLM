# DLM stability program audit: R5C through SI-LWA

Date: 2026-08-30  
Status: frozen evidence synthesis before the next GPU training run

## Executive conclusion

The stability regression is not caused by one bad checkpoint or by the new
Planner simply sampling the wrong coarse chemistry distribution. It is the
combined result of three changes:

1. the realization interface moved from a rich, training-supported structural
   context to exact composition plus `N` with almost all global structural
   variables latent;
2. C3FD removed composition invalidity and composition drift while increasing
   exact-formula novelty, exposing a genuinely harder conditional crystal
   structure prediction problem;
3. DLM training continued to optimize positive-only token reconstruction, not
   a same-composition thermodynamic boundary or periodic spatial relations.

The historical H1-A2/R03 high points are partly real and partly amplified by
rich/replay support, process seeds, survivor-compatible accounting, and a very
strong model494 refiner. Corrected replay still outperforms the current minimal
base, so the capability loss cannot be dismissed as statistics alone.

The most evidence-backed sprint action is **sequence-level, same-composition
continuous-energy alignment with raw validity and periodic geometry safety**.
Structural intent is promoted only if train/held-out oracle and predictability
audits prove that a selected field carries incremental signal. Reinstating the
old rich Planner wholesale is not supported.

## Stagewise bottleneck

Historical H1-A2 compatibility supplied about `890/1000` novel-and-unique
candidates and converted them to `94` Strict and `474` Meta S.U.N.; conversion
within NU was approximately `10.56%/53.26%`. Current C3FD/minimal L7 supplied
`922/1000` NU but only `60/412` Strict/Meta S.U.N.; conversion within NU was
approximately `6.51%/44.69%`.

Candidate supply therefore improved while stable realization conditional on
that supply fell. This is the central regression seam.

## Experiment lineage and causal reading

| Period | Intervention / contract | What worked | What failed | Causal conclusion |
|---|---|---|---|---|
| R5C fixed-slot | padded 107-token answer | exact-length engineering was possible | padding was artificial and not a stability mechanism | Do not return to 107 tokens. |
| R5C exact dynamic | variable exact length with structured MP20/replay conditioning | 1167/1200 graphs; refined structure validity about 99.8% | replay-style result did not establish de-novo stable novelty | Dynamic exact length is useful; structured support mattered. |
| H1-B formula-only | composition without useful structural context | body execution 100% | Strict/Meta 5.54/43.13%; 81.57% all-90-degree lattices; repeated-length templates | Composition-only CE collapses to common geometry modes. |
| H1-A2 historical | learned rich Plan, D1 exact-axis, model494 | 94/474 internal and 105/488 public headline | one historical process view; survivor-compatible artifact | A real useful basin existed, but headline is not a replicated architecture effect. |
| H1-A2 exact/continuous replay | all attempts and repeated processes | 8.58/46.08% exact; 7.63/45.47% continuous | below historical high point | Seed/process/accounting explain part, not all, of the current 6.0/41.2 result. |
| R03 D1/D2 | coordinate commitment changes on one frozen Plan cohort | D1 9.67/51.07%; D2 Strict 11.43% | D2 Meta 48.44%; exact/continuous R03 8.42/46.58 and 7.47/45.26 | Safe-axis high point is process-fragile and not a general stability method. |
| Counterfactual rich-field grounding | same formula/N; factual versus counterfactual lattice family, SG bucket, volume | true-vs-counterfactual margin +0.759; body/Direct safe | fixed1000 Strict 89->86, Meta 487->467; four repeats only +2/1024 Strict and Meta trade-off | Rich fields affect token likelihood but do not by themselves identify lower-energy geometry. |
| More ordinary CE | rich-Plan epoch2 versus epoch3 | body 985->992, Direct 871->878 | Strict SUN 81->79, Meta 489->477 | Better CE/validity is not thermodynamic optimization. |
| Count/valence rich Planner | add correct-looking physics text | teacher assignment coverage about 96% | emitted neutral only 50.31%; lattice-SG 41.14%; all-metal 45.82% | Correct labels in a BPE text head do not guarantee coupled physical execution. |
| CCFD | internal reachability/assignment preference | assignment 94.90%->99.15% | independent comp-valid 1724->1725; seed directions disagree | Internal constraint accuracy is not an external endpoint. |
| C3FD v2.5 | constructive exact N/composition/charge/Pauling witness | 2000/2000 comp-valid; NU 1530->1756; zero semantic dead ends | no structural stability supervision | Composition is solved; freeze it. |
| C3FD distribution audit | compare to full MP20 train | N/arity/family TVD 0.0349/0.0185/0.0204; all-metal 37.25% versus train 34.91% | exact formulas are much more novel and mostly unsupported identities | Gross distribution drift is not the main cause; exact-level conditional OOD remains. |
| Minimal-spec base | remove rich lattice/SG/volume/prototype fields; exact C3FD composition | body/refined Direct near saturation | L7 base Strict/Meta only 60/412 | Removing global context exposed a harder p(structure|exact C,N) problem. |
| Condition/schedule factorial | full versus hard fields; axis versus joint XYZ | hard-axis improved body/Direct slightly | joint schedule lost 15.8--24.3 pp Direct/body through duplicates; no stability promotion | Preserve exact full-axis legality; schedule search is closed. |
| model494 tau | tau0/200/500/800 | Direct 188->457; Strict 10->48; Meta 66->230; q50 hull 2.1767->0.0913 | novelty/retention trade-off; DLM differences can be washed out | Refiner is a major absolute contributor and an attribution confound. |
| Same-Plan energy data | multiple trajectories per composition | energy gaps are substantial: median 0.118 eV/atom; 1752 outcomes over 222 Plans | old extreme-pair gate retained only 95/27 pairs | There is within-composition headroom; use all candidates and continuous weights. |
| CTV token action head | predict one token action's terminal energy | all 3072 branches evaluated cleanly | centered Spearman 0.0353, AUC 0.5053, continuation agreement 0.4915 | Stability is global/sequence-level; one-token value guidance is closed. |
| SGTC | all-MP20 versus strict-stable positive-only continuation | body/Direct/NU retained | G1 Strict/Meta 53/417; official and CHGNet moved adversely | Positive-only stable subset CE has no same-composition negative boundary. |
| D3PO | paired full-sequence shared-mask preference, post-refiner labels | both seeds refined and official mean moved about -2--3 meV; Meta +1.76/+3.52 pp; Direct/NU retained | Strict unchanged; raw fixed +0.200 eV; one seed raw significantly worse | Full-sequence preference has a weak real signal, but post-refiner-only reward permits raw off-manifold exploitation. |

## What the old Planner did and did not do

The old rich Planner reduced uncertainty about global lattice scale and broad
structure mode. That likely explains part of the better raw execution and why
the original DLM/refiner interface stayed closer to the distribution model494
was trained to repair.

It did **not** reliably predict stable structural basins:

- lattice/space-group matching was only about 40--43% in later matched audits;
- the stability diagnostic found lattice-SG matched rows had worse median hull
  (`0.0881` versus `0.0683 eV/atom`);
- fully soft-consistent rows also had worse median hull (`0.0879` versus
  `0.0739`);
- volume matching had a modest Meta/continuous association (`0.0733` versus
  `0.0809`) but little Strict effect;
- counterfactual grounding changed the intended mechanism without producing a
  reliable final stability gain.

Therefore restoring the seven-line rich Plan can plausibly improve syntactic or
global structural plausibility, but it is not an evidence-backed stability
objective. A deterministic composition-to-rich-Plan model also ignores
polymorphism and risks majority-plan collapse.

## What the new 100%-valid Planner changed

C3FD-v2.5 changed upstream candidate supply, not structural realization:

- comp-valid increased from `1724/2000` to `2000/2000`;
- Novel/Unique/NU changed from `1538/1961/1530` to
  `1763/1985/1756`;
- the new cohort is closer than P0 to full-train N/arity/family marginals;
- all-metal mass is close to full MP20 train, not an obvious easy-chemistry
  shortcut;
- outcome/stability labels were never used.

It simultaneously removed two historical shortcuts:

1. invalid formulas no longer disappear before the structure stage;
2. exact composition is anchored, so the generator cannot drift toward a
   familiar stable formula.

The resulting benchmark is scientifically cleaner and harder. Its lower stable
rate does not mean C3FD is a worse Planner; it means the DLM's conditional
realization deficit is no longer hidden by invalid-attempt attrition or
composition self-selection.

## Why the old DLM appeared better

Five effects acted in the favorable direction:

1. **Rich/replay support:** lattice/volume/family-scale context narrowed the
   structural basin and often resembled MP20 teacher conditions.
2. **Easier exact identities:** old sampled or replayed formulas had more direct
   support than current exact novel identities.
3. **Composition flexibility:** historical joint generation could sometimes
   avoid difficult fixed-composition realizations.
4. **Refiner dominance:** model494 converted many poor raw proposals into valid,
   lower-hull structures; historical reporting often emphasized the final
   system rather than raw DLM attribution.
5. **Seed/process/accounting:** the strongest H1-A2/R03 points decreased under
   exact all-attempt and continuous replay.

Nevertheless, corrected rich-lineage replay remains above the current minimal
base, so a genuine interface/task capability regression remains after removing
the favorable artifacts.

## Root causes ranked by evidence

1. **Objective mismatch — very high confidence.** Ordinary CE and SGTC do not
   contrast stable and unstable structures within one composition. D3PO's small
   post-refiner success is the first direct evidence that changing the objective
   can move energy.
2. **Exact-composition conditional OOD — high confidence.** Coarse marginals are
   close to MP20, but exact formulas are more novel and composition drift is
   forbidden.
3. **Loss of global structural basin context — medium/high confidence.** H1-B,
   rich versus minimal history, and volume diagnostics support it; rich-field
   counterfactual and SG diagnostics show it is not sufficient.
4. **DLM-to-refiner interface / reward leakage — high confidence.** tau800 is
   dominant; D3PO improves post-refiner outcomes while raw generation worsens;
   clean body is injected as an `x_tau`-like state without matched forward noise.
5. **Masked-token spatial inductive bias — medium/high confidence.** Token CE has
   no built-in periodic distance, species coordination, rotation, or global
   lattice coupling. Joint XYZ's failure shows that merely committing more
   coordinates together is not the fix.
6. **Seed/process artifacts — high confidence but not primary.** They inflate
   old high points but cannot explain the full rich-to-minimal gap.

## Immediate decision under the remaining time

### Freeze what already works

- keep C3FD-v2.5 composition+N and its outcome-blind distribution;
- keep dynamic `7+4N`, exact full-axis legality, temperature 0.7;
- keep model494 tau800 and report raw/refined separately;
- keep two training seeds, two common streams, one immutable prospective cohort;
- keep composition-level weighting and cluster bootstrap.

### Primary training intervention

Run **listwise full-sequence continuous-energy alignment with raw and geometry
safety**, initialized from minimal BASE:

- all `K=2..8` candidates per composition;
- robust centered post-refiner energy weights;
- raw-invalid candidates lexicographically worst;
- raw-energy rank where known;
- quadratic reference bound and best-valid denoising anchor;
- periodic species-pair distance-bin and coordination-consistency auxiliary
  targets at shared masked states;
- no cross-composition absolute-energy comparison.

This route directly follows the only replicated positive signal (D3PO refined /
official direction) while correcting its demonstrated raw failure.

### Structural intent decision

VPA/CN and richer fields remain a **train/validation information audit**, not a
prerequisite for the next GPU run. Promote intent only if an oracle field adds
raw Direct or continuous-energy value beyond minimal/listwise and is
predictable without majority collapse. Old SG text, prototype, and full rich
Plan are not restored by default.

If the intent audit is incomplete or ambiguous at the six-hour deadline, the
two-seed run is listwise-only. This is not a compromise in scientific focus:
the evidence says the missing energy boundary is more certain than the proposed
intent representation.

## Longer-term architecture if the sprint remains weak

The next DLM architecture should add periodic geometric inductive bias rather
than more text labels:

- a periodic pair-distance/species-pair auxiliary decoder at random mask times;
- a coarse metric-tensor and species-conditioned coordination latent;
- an E(3)/periodic graph adapter between selected Transformer blocks;
- teacher distillation from relaxed winners, including the correction from raw
  body to model494-relaxed structure;
- a valid forward-noise DLM-to-refiner bridge.

These are paper-compatible DLM extensions. They preserve the central claim that
the DLM chooses discrete structural realization while the continuous model
refines geometry.

## Falsifiable sprint outcome

Success is not a selected threshold count. It requires both independently
trained policies to move same-composition raw and refined energy in the favorable
direction, retain Direct/NU, and agree with official hull. Meta/Strict S.U.N.
then quantify threshold conversion. If only post-refiner energy moves, the
method is pipeline-aware but the raw DLM remains unsolved. If neither raw nor
refined energy moves, the current token representation/objective is a negative
result and further CE, Planner tuning, or tau search is not justified.

