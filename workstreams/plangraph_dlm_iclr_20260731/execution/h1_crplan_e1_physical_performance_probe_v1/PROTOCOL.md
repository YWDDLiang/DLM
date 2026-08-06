# H1 CR-Plan E1 physical-performance probe V1

Status: `authorized_exploratory_engineering_probe_not_a_v4_repair`

Date: 2026-08-05

## Question

V4 proved exact support parity but failed its immutable logical-work gate:
real paired-32 cursors reached 415,689 cumulative semantic DP states versus a
registered limit of 100,000. That failure remains final for the frozen V4
route.

E1 asks a different, physical question: is the already-audited exact
combined-trie/bitset implementation cheap enough during real Planner
generation to justify proposing a new preregistered scientific route?

E1 cannot turn V4 into a pass, authorize four-arm 512, or support a scientific
chemistry claim.

## Frozen probe

- one A800 node and one resident frozen P0 Planner;
- 18 engineering-only ordinals, independent of every scientific ledger;
- common base seed `26080517`, stateless ordinal role `shared`;
- balanced three-period order:
  `off→terminal→full`, `terminal→full→off`, `full→off→terminal`;
- `h1_rich_plan_v1`, no sample ID, temperature `0.9`, top-p `0.95`,
  top-k `50`, maximum 96 new tokens, and `1 <= N <= 20`;
- terminal/full share the frozen Direct-aligned `fail_closed` missing-state
  policy;
- each constrained mode owns an independent reachability/token-support cache;
- one common no-sampling forward pass warms only the resident model kernels
  before any ordinal; it does not traverse CR-Plan support or consume a seed;
- all attempts, including failures and first/cold occurrences, count;
- CUDA is explicitly synchronized immediately before and after every timed
  generation;
- model/tokenizer loading, support setup, trace audit, scalar reference
  reruns, RSS, GPU peak memory, and total job wall are reported.

There is no retry, replacement, repair, filter, rerank, fallback, Body,
refiner, Direct evaluation, S.U.N., network call, training, checkpoint
selection, promotion, or downstream action.

## Exactness checks

The V4 exact-tokenizer/support audit remains the release anchor. E1 adds:

1. optimized-versus-scalar legal IDs, terminal IDs, and rejection-count
   equality on every unique formula-value cursor actually visited by the
   terminal/full E1 traces;
2. proof that every sampled constrained token belonged to its exact support;
3. exact prompt hash, input-token hash, and ordinal seed identity across all
   three modes;
4. full optimized-versus-scalar end-to-end token/text/certificate reruns for
   preregistered engineering ordinals 2 and 11.

The scalar reruns occur after primary timing and are excluded from the latency
ratio, but remain inside total allocated GPU wall time.

## Immutable kill-or-go gate

E1 is a physical-feasibility pass only if:

- full-prefix median generation latency is at most `1.5x` same-job P0;
- full-prefix p95 generation latency is at most `2.0x` same-job P0;
- trace support parity and the two scalar token reruns are `100%`;
- prompt/input/seed identity is `100%`;
- at least 5% of charge-applicable full-prefix attempts contain a genuine
  preterminal full-versus-terminal support difference;
- no dead end, identity error, OOM, timeout, exception, silent fallback, or
  forbidden sample operation occurs.

The V4 100,000-state threshold is intentionally not reused or redefined.
Logical states remain reported under their original definition.

Pass means only `eligible_for_new_preregistered_scientific_route_amendment`.
Fail means stop CR-Plan and retain frozen H1. Neither outcome automatically
submits 512 or any downstream experiment.
