# No-training evidence plan

This plan adds analysis and small inference panels without training a Planner,
DLM, refiner, reward model, or scheduler.

## E0: existing-artifact analysis

### E0-A — immutable attempt funnel

Every cohort reports, in raw-ordinal order:

```text
requested -> decoded -> Plan eligible -> body attempted -> body success
          -> refined -> reconstructed -> N/U -> hull known -> Strict/Meta SUN
```

The requested count is the primary denominator. Survivor-prefix and
hull-known views are secondary. Frozen generation, evaluator replay, and new
generation are never counted as independent cohorts when they reuse structures.

### E0-B — chemistry mix versus conditional conversion

For method `m` and chemistry stratum `h`:

```text
P_m(Y=1) = sum_h p_m(h) mu_m(h).
```

Primary strata are composition-derived family, arity, atom-count bin, and
all-metal/unary shortcut status. Generated anion, exact element set, element
presence, and coarse Plan fields are sensitivity analyses.

Report:

- distributions at every funnel stage;
- adjacent-stage TVD/JSD and stratum survival;
- symmetric Kitagawa/Oaxaca mix and within-stratum terms;
- standardization to both learned-P0 and MP-20 reference mixes;
- common-support coverage, effective sample size, maximum weight, and any
  trimming rule;
- overlap-restricted results when support is insufficient.

### E0-C — Plan quality

For generated Plans, report formula and full-tuple train collisions, nearest
Plan distance, marginal and tuple entropy, family/arity/N distributions, and
downstream conversion by Plan taxonomy. Plan metrics support the de novo task
boundary; they are not a separate algorithmic contribution.

```bash
python scripts/audit_plans.py \
  --generated runs/analysis/generated_plans.jsonl \
  --train data/plans/mp20_train_rich.jsonl \
  --output runs/analysis/e0/plan_audit.json
```

### E0-D — DLM-to-refiner attribution

Using aligned pre/post artifacts, report atom-count/composition invariance,
lattice and coordinate displacement, StructureMatcher match, minimum distance,
space group/P1 changes, energy changes, and a body-quality-to-final-quality
conversion matrix.

```bash
python scripts/build_refiner_pairs.py \
  --proposal-graphs runs/story_e2/prepared/e2_proposal_graphs.pt \
  --refined runs/story_e2/refined/dlm_refined_mp_192.pt \
  --metadata-jsonl runs/story_e2/prepared/e2_selected_metadata.jsonl \
  --attempt-ledger runs/story_e2/prepared/e2_attempt_ledger.jsonl \
  --chgnet-model checkpoints/chgnet/chgnet_0.3.0_e29f68s314m37.pth.tar \
  --chgnet-device cuda \
  --output runs/story_e2/refiner_pairs.jsonl

python scripts/audit_refiner.py \
  --input runs/story_e2/refiner_pairs.jsonl \
  --output runs/story_e2/refiner_attribution.json
```

The refined filename encodes the number of body successes, so replace `192`
when E1 contains failures. Optional evaluator fields can be joined with
`--evaluation-jsonl`; missing evaluator values remain unknown rather than being
treated as failures.

### E0-E — discovery Pareto

Plot Unique-and-Novel supply against Strict/Meta conversion within U&N; encode
final S.U.N. separately. Cross-paper points with different evaluators appear in
a separate protocol-context panel and are not ranked against local points.

## E1: rich-Plan adherence and multiplicity body panel

The panel contains 48 preregistered Plans: 24 learned H1-A2 and 24 held-out
R5-C gold Plans, exactly matched where possible on `N`, arity, and anion family.
Plans are selected without energy or downstream outcomes.

| Arm | Condition | Replicates per Plan |
|---|---|---:|
| `full` | intact rich Plan | 8 |
| `formula` | formula/N/composition only | 4 |
| `shuffle` | formula fixed; lattice/SG/volume permuted within matched strata | 4 |

Total: `48 * 16 = 768` requested body attempts. First four replicates are
paired across arms through deterministic per-task seeds. B0 weights,
temperature, composition/count anchors, support, and the standard exact-Plan
field schedule remain fixed. No retry, filtering, replacement, or reranking is
allowed.

The panel root seed is fixed in code and Slurm as `20260822`; task-specific
body and refiner seeds are deterministically derived from pair, source, and
replicate identity.

Primary outputs:

- parse, Plan match, graph yield, duplicates, and lattice legality;
- realized lattice-family, space-group-bucket, and volume-bin adherence;
- StructureMatcher/local-environment clusters and effective multiplicity;
- NFE, wall time, and learned-vs-gold source interaction.

```bash
python scripts/analyze_story_panel.py \
  --tasks runs/story_e1/contracts/e1_tasks.jsonl \
  --raw runs/story_e1/body/raw_generations.jsonl \
  --sample-metrics runs/story_e1/body/sample_metrics.json \
  --output-json runs/story_e1/e1_analysis.json \
  --output-jsonl runs/story_e1/e1_records.jsonl
```

Fail-closed interpretation:

- no full-vs-formula/shuffle adherence difference: describe conditioning as
  formula/composition/N only;
- at least 75% of Plans yield one valid cluster at K=8: remove the
  multiple-realization claim;
- gold Plans outperform learned Plans: report condition-source compatibility
  as the main bottleneck;
- learned and gold Plans are similar: do not attribute losses primarily to the
  Planner.

## E2: blind 192-proposal refinement panel

Before E1 outcomes are viewed, select 16 Plans (eight per source). For each of
the three E1 arms, take the first four registered replicates:

```text
16 Plans * 3 arms * 4 replicates = 192 requested proposals.
```

Body failures remain failures in the 192 denominator. All successful selected
proposals enter model_494 refinement with 800 reverse steps and batch size one.
Report invariance, Plan adherence retention, multiplicity retention,
displacement, CrysLLMGen-compatible Direct composition/structure/joint
validity, and paired CHGNet single-point energy. Small-panel S.U.N. is
descriptive rather than a promotion gate.

If refinement erases body/condition differences, restrict the DLM claim to
proposal reliability rather than final structural family or stability.

## Work explicitly deferred

- new Planner or DLM training;
- matched constrained AR training;
- support-consistent/legal-mass training;
- species-site assignment;
- violation-guided reopening or revision;
- another full R03/safe-axis suite;
- full 1,000/1,200 S.U.N. confirmation before E1/E2 support the estimand.

## Baseline inclusion policy

The same-protocol main table uses local H1-A2, CrysLLMGen, and clearly labelled
gold/replay controls. One additional strong method may be unified-re-evaluated
only when public outputs/checkpoints can be used without training and integrated
within one day. All conceptually relevant stronger methods remain in a
published-protocol context table regardless of score. Inclusion is never based
on whether a method scores above or below H1-A2.
