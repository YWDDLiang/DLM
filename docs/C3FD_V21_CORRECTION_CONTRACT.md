# C³FD-v2.1 Correction Contract

Date: 2026-08-28

Status: Step 0 frozen. No v2.1 GPU job is authorized until Steps 1--4 pass.

Contribution point 1 and public `105/488` are unchanged. C³FD remains a
composition-correctness candidate, not a stability method.

## Step 0 — immutable v2 snapshot

Formal v2 run: `c3fd_planner_v2_36427`.

| Metric | P0 / 2000 | C³FD-v2 / 2000 | Delta |
|---|---:|---:|---:|
| formula / parsed | 1989 | 1972 | -17 |
| independent comp-valid | 1724 | 1972 | +248 (+12.4 pp) |
| Novel | 1538 | 1605 | +67 |
| Unique | 1961 | 1919 | -42 |
| Novel × Unique | 1530 | 1583 | +53 |
| all-metal | 575 | 724 | +149 |

Both seeds improved independent comp-valid by `+12.4 pp`; paired 95% CI is
`[+10.81,+13.99] pp`. The original preregistered v2 screen remains NO-GO and
must not be rewritten as a pass.

## Correct interpretation of proposal drift

Not every difference from P0 is a regression.

| Reference | all-metal |
|---|---:|
| full MP-20 train | 34.91% |
| benchmark-compatible train | 38.57% |
| C³FD-v2 parsed | 36.71% |
| P0 parsed | 28.91% |

C³FD is closer to the real training distribution than P0 on all-metal. Its N
distribution is also closer to full train (`TVD=0.0866`) than P0
(`TVD=0.1017`). These quantities must not be forced back to P0.

The true collapses are:

- arity >=5: C³FD `13.4%`, P0 `2.1%`, benchmark train `2.9%`;
- arity TVD to full train: C³FD `0.1818`, P0 `0.0319`;
- family TVD to full train: C³FD `0.1582`, P0 `0.0948`;
- structured species top-50 mass: train `53.5%`, C³FD `62.5%`;
- 104 of 432 train species nodes were never sampled;
- 28/2000 requests ended in semantic dead ends.

The pair prior is not the main global cause: mean pair score is lower in C³FD
than P0. It is retained only as recorded evidence and is disabled in v2.1.

## v2.1 single-candidate change

Keep unchanged:

- typed element/oxidation/count actions;
- Aufbau/physics species features;
- locked atom count and exact charge ledger;
- online reachability and independent benchmark certificate;
- one request, one trajectory; no repair, replacement or reranking.

Change only:

1. predict `family`, `N` and exact `arity` before species generation;
2. train these three proposal heads on **all** MP-20 training rows, not only
   benchmark-compatible rows;
3. train species/count only on benchmark-compatible semantic rows;
4. inject normalized remaining atoms, remaining charge and remaining species
   into every semantic decoder step;
5. require exactly `arity` distinct elements at terminal state;
6. remove species `top_k=50`; use top-p plus independently calibrated
   temperatures for proposal/species/count heads;
7. set global pair-prior weight to zero for the v2.1 candidate.

No external CrysVCD comparison or component grid is added.

## Revised scientific gates, frozen before v2.1

Effect gates:

- pooled and per-seed independent comp-valid delta >0;
- paired 95% CI lower bound >0;
- ionic-only independent comp-valid delta >0;
- parse noninferior within 1 pp;
- Novel × Unique noninferior within 1 pp.

Distribution gates use full MP-20 train as the reference:

- all-metal absolute gap from full train <=3 pp;
- candidate N TVD to full train <= P0 N TVD to full train +0.01;
- candidate arity TVD to full train <= P0 arity TVD to full train +0.01;
- candidate family TVD to full train <= P0 family TVD to full train +0.01;
- no train species-support loss greater than 5 pp relative to validation
  support under calibrated sampling.

Standalone Plan Unique is reported but is not a promotion gate when
Novel × Unique improves. The original v2 gates remain archived unchanged.

## Phase ledger

Every step writes one MD/JSON record and is independently reversible.

### Step 1 — data and support audit

- add exact family/N/arity labels for all rows;
- add per-step ledger targets for semantic rows;
- audit reachable mass for every `(family,N,arity)` stratum;
- require >=99% weighted train/validation proposal mass reachable;
- output `C3FD_V21_STEP1_DATA_AUDIT.{md,json,csv}`.

### Step 2 — model regression tests

- write failing tests for exact arity and ledger conditioning first;
- add family and arity heads plus ledger projection;
- require exact terminal arity and unchanged N/charge certificates;
- output `C3FD_V21_STEP2_MODEL_AUDIT.{md,json}`.

### Step 3 — calibration

- fit one temperature per proposal/species/count head on validation NLL;
- species top-k disabled and pair-prior weight fixed to zero;
- verify entropy and species support without sampling final outcomes;
- output `C3FD_V21_STEP3_CALIBRATION.{md,json,csv}`.

### Step 4 — CPU proposal simulation

- draw at least 100,000 proposal tuples `(family,N,arity)`;
- apply no structure or stability outcome labels;
- require revised distribution gates before GPU authorization;
- output `C3FD_V21_STEP4_PROPOSAL_SIM.{md,json,csv}`.

### Step 5 — L5 pilot

- exactly two seeds × requested256;
- full candidate only, no grid;
- all failures retained in the denominator;
- promote only if every effect and distribution direction is eligible.

### Step 6 — L7 confirmation

- two seeds × requested1000 only after Step 5 passes;
- report both seeds and pooled; no selected-seed headline.

## Stop conditions

Stop without relaxing gates if:

- any CPU step fails after one engineering-only repair;
- exact N/charge/arity or benchmark certificate regresses;
- the requested256 pilot loses comp-valid direction in either seed;
- arity/family collapse remains after proposal calibration.
