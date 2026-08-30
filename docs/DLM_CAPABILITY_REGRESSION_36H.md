# DLM capability-regression audit for the 36-hour sprint

Date: 2026-08-30  
Status: frozen zero-GPU diagnosis before SI-LWA-v1 training

## Executive finding

The current failure is not evidence that masked-SFT is intrinsically unable to
produce valid crystal structures.  Historical R5C, H1-A2, and R03 artifacts
show that the DLM was structurally reliable when it executed a rich structural
Plan.  The large regression occurred after the interface was reduced to exact
composition plus `N`, while thermodynamic supervision remained either ordinary
positive-only denoising CE or post-refiner preference learning.

The current system still recovers approximately 99% post-refiner Direct joint
validity.  What regressed is the raw DLM geometry manifold and its conversion to
Strict/Meta S.U.N.  This distinction is mandatory in every sprint report.

## Comparable evidence ledger

| Period / system | Conditioning and evaluation view | Structure / execution evidence | Strict S.U.N. | Meta S.U.N. | Interpretation |
|---|---|---:|---:|---:|---|
| R5C fixed-slot preflight | old padded 107-token target | representation-only preflight | n/a | n/a | Retired; padding is not the historical source of stability. |
| R5C exact dynamic | MP20/replay-style structured conditioning | 1167/1200 graphs; refined1000 comp-valid about 90.7%, structure-valid about 99.8% | not a clean de-novo headline | not a clean de-novo headline | Dynamic exact length and structured conditioning can support valid structures. |
| H1-B formula-only | composition without useful global structural intent | body 100%; 81.57% all-90-degree angles; 53.33% repeated two lengths | 5.54% | 43.13% | Direct warning that composition-only CE collapses to frequent geometry templates. |
| H1-A2 historical compatibility | learned rich Plan, historical frozen1000 | Novel/Unique/NU 89.2/99.7/89.0% | 9.40% | 47.40% | Real high point, but one historical process view. |
| H1-A2 exact replay | rich-Plan lineage, all requested attempts | requested1200 | 8.58% | 46.08% | Seed/process change reduces the high point but does not explain the full current regression. |
| H1-A2 continuous repeats | rich-Plan lineage, pooled3840 raw view | pooled3840 | 7.63% | 45.47% | Historical effect is fragile, not fictitious. |
| R03 D1 | rich Plan, global X then Y then Z | requested1024 | 9.67% | 51.07% | Strong Meta point. |
| R03 D2 Safe-axis | same Plan cohort, process repeats rather than independent Planner seeds | requested1024 | 11.43% | 48.44% | Strict high point trades Meta and is not an independent-seed claim. |
| R03 exact / continuous replay | corrected replay views | requested1200 / pooled3840 | 8.42% / 7.47% | 46.58% / 45.26% | Confirms fragility while remaining above current minimal L7. |
| Rich-Plan sufficient SFT, epoch2 | frozen raw1000 rich-Plan cohort | body985; Direct joint871; stable102/587 | 8.10% | 48.90% | More CE is not stability optimization, but rich conditioning preserves a useful basin. |
| Rich-Plan sufficient SFT, epoch3 | same cohort | body992; Direct joint878; stable100/578 | 7.90% | 47.70% | Body improves while stability falls: CE and thermodynamic ranking are distinct objectives. |
| Minimal L6 full-axis | composition+N, dynamic `7+4N` | refined body/direct 505/457 of 512; raw Direct188; model494 raises raw->refined Direct 188->457 | 48/512; raw10/512 | 230/512; raw66/512 | model494 is the dominant validity/stability repair stage and can erase DLM differences. |
| Minimal L7 base | C3FD composition+N, exact novel cohort | body998; refined Direct996; N/U/NU 922/995/922 | 60/1000 | 412/1000 | Composition/execution is saturated after refinement, but S.U.N. is below every corrected H1-A2/R03 replay view. |
| SGTC L7 G1 | positive-only stable continuation | body1000; Direct996 | 53/1000 | 417/1000 | Better teacher-forced NLL does not learn a same-composition energy boundary. |
| D3PO fixed256, offline | post-refiner shared-noise preference | refined Direct 254--255/256; raw Direct base151/164, seed81017 124/137, seed81018 150/144 | official pending | official pending | Four refined means are slightly favorable, but raw validity/energy is non-replicated or adverse. |

The public `105/488` headline remains a separate frozen aggregate.  It is not
used as training microdata and is not silently replaced by any row above.

## Ranked, falsifiable causes

1. **Rich-to-minimal conditioning regression (highest confidence).**  H1-A2 and
   R03 exposed lattice/volume/family-scale soft information; CTV/SGTC/D3PO ask
   the DLM to infer the global structural basin from exact composition and `N`.
   H1-B's formula-only template collapse is the direct historical warning.
   Prediction: an internal oracle structural-intent condition should improve
   raw geometry or energy on matched train/chemsys-validation compositions.
2. **Exact-composition conditional difficulty.**  C3FD is close to MP20 in
   coarse N/arity/family marginals, but exact formulas are mostly unseen and no
   composition drift is allowed.  Prediction: degradation is larger on unseen
   exact identities/chemsys than on supported ones after conditioning is held
   fixed.
3. **Post-refiner objective permits off-manifold raw bodies.**  D3PO rewards
   model494 outcomes and only weakly anchors the reference.  Prediction: a loss
   that lexicographically penalizes raw invalid candidates and jointly ranks
   raw/refined energy preserves Direct while retaining the refined left shift.
4. **Seed/process instability (real but secondary).**  Exact and continuous
   replays lower H1-A2/R03 high points.  Prediction: any valid new claim must
   reproduce across two training seeds and two common generation/refiner
   streams; one favorable cell is insufficient.
5. **Dynamic representation is not the main cause.**  R5C exact-dynamic and
   current post-refiner Direct remain strong, while the retired 107-token and
   mixed-axis variants have independent defects.  Prediction: changing length
   representation or coordinate schedule without restoring intent/energy
   supervision will not recover S.U.N.

## Prospective correction

SI-LWA-v1 keeps C3FD at composition plus `N` and keeps the original dynamic body
positions.  The DLM first predicts two masked field-specific intents,
`VPA_Q8 = quantile(log(volume/N))` and `CN_ENV8 = coarse coordination medoid`,
then executes the unchanged lattice -> all-X -> all-Y -> all-Z schedule.  The
training objective combines masked body CE, intent prediction/reconstruction,
same-composition refined-energy listwise alignment, raw-validity/raw-rank
safety, a quadratic reference bound, and a best-valid anchor.

The route is rejected before prospective test generation if an oracle intent
does not improve either raw Direct or raw/refined energy on train-only or
chemsys-held-out validation.  If oracle intent works but self-prediction
collapses, the finding is a composition-to-intent bottleneck; external rich
Plan labels are not substituted as the paper-facing method.

## Claim boundary

- Supported now: historical structured conditioning maintained structure
  validity; minimal exact-composition execution loses raw quality and S.U.N.;
  ordinary CE and positive-only continuation do not solve thermodynamic rank.
- Not yet supported: removing rich fields *caused* the entire S.U.N. change;
  SI-LWA improves a prospective C3FD cohort; model494 is optimally bridged.
- Required next evidence: matched oracle/self-predicted/minimal canary, two
  training seeds, two common streams, raw and refined continuous effects, and
  one immutable prospective official-hull evaluation.

