# Current state

Updated: 2026-08-13 (Asia/Shanghai)

Overall status: `ARCHIVED_FIRST256_REPRODUCTION_COMPLETE_FROZEN_CACHE_COVERAGE_LIMITED`

## Terminal diagnostic: archived first256 generation chain reproduced

The requested single-repeat, 256-attempt archived-code reproduction is
complete. Job `31963` completed `0:0` in `04:10:02` using one A800 and eight
CPUs. It reused the body artifacts from job `31931`; the control and candidate
attempt ledgers and proposal graphs are all byte-identical to the 2026-08-02
successful archive. The downstream repair changed Python import precedence
only. Both arms then ran model-494 exact refine800, Direct, and the archived
R5-C/A100 frozen-cache S.U.N. evaluator without retry, replacement, repair,
filtering, reranking, training, or RL.

| arm | generated | comp/joint | structure | COV-P | COV-R | strict S.U.N. | meta-S.U.N. | hull evaluated/unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original H1-A2 D1 | 246/256 | 212/256 | 246/256 | 96.09% | 84.31% | 12/256 | 70/256 | 122/101 of 223 |
| R03 D2 safe-axis | 248/256 | 213/256 | 248/256 | 96.48% | 83.76% | 14/256 | 73/256 | 120/104 of 224 |

Paired exact McNemar is non-significant: strict has R03-only/control-only
`4/2`, `p=0.6875`; meta has `14/11`, `p=0.690038`. The exact historical
absolute counts (`27/133` for H1-A2 and `28/122` for R03) were not recovered
under this bundled cache because 101 and 104 novel-unique structures have no
hull value. Report-only coverage-adjusted estimates are 8.92/52.01% for
H1-A2 and 10.54/54.95% for R03. Thus the archived generation/refinement
machinery is reproduced, but this incomplete frozen cache cannot support an
exact best-count reproduction. This result is not the official-clean MP
protocol and does not supersede the completed current-cache historical replay.

Full interpretation is in
`H1_R03_H1A2_ARCHIVED_FIRST256_REPRODUCTION_RESULT_V1.md`; returned evidence
is under `evidence/h1_r03_h1a2_archived_first256_downstream_repair_v4/`.
Terminal JSON SHA is `13570083...00fd3` and the returned archive SHA is
`a5582814...bca35`.

## Terminal diagnostic: official MP stability repair does not close the Plan1200 gap

The stability-only repair is complete. Normal jobs `31737` (16 frozen cells
evaluated in parallel) and `31738` (assembly) both completed `0:0`. The run
issued no new MP request and reran no Planner, body/DLM, model-494, CHGNet,
reconstruction, novelty, or uniqueness work. It adopted the completed fresh
official `MPRester.get_entries_in_chemsys()` spool with
`compatible_only=True` and `GGA_GGA+U`: 2,550/2,630 systems have complete
official references. The other 80 all lack a Yb unary reference and are
reported as explicit `hull_unknown`; they are excluded only from denominators
labelled `skip MP unknown`.

The official-clean correction is small. Historical R03 refined256 strict
S.U.N. is 10.94--12.50% on fixed all-attempt denominators and
11.34--12.96% after skipping explicit MP unknowns. V4 post-model494 strict
S.U.N. remains 6.00--7.30% for R03 and 6.00--7.10% for B3 on fixed
all-attempt denominators; skip-unknown values are 6.15--7.48% and
6.15--7.29%, respectively. Mean clean-minus-old `E_hull` shifts range only
from -0.000938 to +0.000075 eV/atom. Thus neither the 80 Yb systems nor the
old compatibility path explains the large historical-versus-Plan1200 gap.

The paired clean analysis confirms model-494 as a large positive effect:
strict post-minus-pre is +4.425 pp for R03 (95% CI +3.402 to +5.449) and
+4.665 pp for B3 (+3.536 to +5.763). B3-minus-R03 is not resolved on these
three cohorts, either pre (-0.514 pp, CI -1.338 to +0.378) or post
(-0.274 pp, CI -1.094 to +0.479). The remaining diagnostic focus is therefore
the Plan sampling/cohort distribution, not further S.U.N. cache repair.

Complete strict and meta tables are in
`evidence/h1_sun_official_gga_u_skip_unknown_reeval_v2/RESULTS_COMPLETE.md`;
the interpretation and paired table are in the adjacent `ANALYSIS.md`.
Terminal JSON SHA is `651588b7...f254b` and Markdown SHA is
`b6daf991...53544`.

## Terminal diagnostic: historical R03 remains high under current S.U.N./MP cache

An immutable evaluation-only replay of the four historical byte-frozen R03
refined256 process realizations is complete. Repeat array `31650` and assembly
`31651` completed `0:0`. The replay reran no Planner, body/DLM, model_494, or
CHGNet work; all generation and relax-energy caches remained byte-identical.
It completed the historical cohort's 224-system MP snapshot by resolving the
only 92 missing systems, with zero transport retry, then applied the current
exact S.U.N. path offline.

The current result is exactly identical to the historical result at every
attempt. Strict counts are `[28,31,29,29]/248` reconstructed and meta counts
are `[122,123,125,126]/248`; every strict and meta old/current comparison has
zero discordant pairs and exact McNemar `p=1`. Thus the low Plan1200 values
are not caused by the current evaluator or MP cache.

The Plan prompt audit also found the active H1A2/current seven-line
`h1_rich_plan_v1` branch byte-identical with the same P0 adapter and sampling
knobs. The meaningful change is the sampling/cohort protocol: legacy used one
global RNG stream at seed 17029 and reused one first256 cohort, whereas
Plan1200 uses stateless ordinals, three base seeds, and three disjoint
raw1200→first1000 parse-success cohorts. Only 19/254 comparable historical
formula identities overlap the current 3000-plan union, and only 4/254 after
including prototype identity.

All six V4 arm×repeat tasks (`31583`/`31584`) and all twelve pre/post stage
reports completed successfully. model_494 raises structure validity from
46.1–53.4% to 96.8–98.1%, strict S.U.N. from 1.12–2.68% to 5.93–7.22%, and
meta S.U.N. from 12.07–15.06% to 42.12–46.34%. Current R03 post-model494 is
still below historical R03 (strict 6.14–7.22% versus 11.29–12.50%; meta
44.48–46.34% versus 49.19–50.81%), making cohort/RNG distribution the leading
remaining explanation rather than prompt text, cache, or lack of refinement.

V4 assembly `31585` failed closed `3:0` after the twelve successful stage
reports because its statistics path raised `OverflowError: int too large to
convert to float`. Therefore the per-stage point estimates are complete but
V4 has no valid assembled three-repeat inference, and native1000 was not
submitted. The complete diagnostic is
`H1_R03_REFINED256_CURRENT_SUN_CACHE_REPLAY_AND_PLAN_SAMPLING_AUDIT_V1.md`;
the replay terminal report SHA is `b4f449c4...9ea2`.

## Terminal engineering failure: Plan1200 V3 and native post-refine 1000

The authorized V3 planner stage completed: array `31565` and assembler
`31566` finished `0:0`, raw1200 repeats yielded 1,189/1,193/1,194 parse
successes, and three distinct first-1000 cohorts were frozen with each repeat
shared between R03 and B3. The main 2,464-row MP cache and native 2,865-row
extension are complete at SHAs `bf0dc8ed...` and `bc622ae4...`; the latter
resolved 401 missing systems with zero transport retries and destroyed its
one-time credential carrier.

All main body tasks then failed before generation. R03 array `31569` and B3
array `31570` each have three `FAILED 1:0` tasks; assembler `31571` failed
closed `3:0`. Every frozen cohort row lacks a top-level `parsed` key, while
`run_body_safeaxis1000.py` requires `row.get("parsed") is True`. The body
preflight checked the other prompt invariants but omitted this field, so the
producer/consumer schema mismatch first surfaced at ordinal 0 in every task.

The requested CrysLLMGen-native supplement was frozen separately: select the
first 1,000 body/`process_one` successes in planner order and pass all 1,000
through model_494. Its arrays `31576`/`31577` and assembler `31578` also
failed closed because the required main successes were absent; reserve
generation and diffusion refine never began.

Therefore V3 has no pre-refine or post-refine CrysLLMGen/S.U.N. metric and no
three-repeat inference. This is an engineering failure, not a P0, R03, B3,
refiner, or evaluator result. V1, V2, V3, and native supplement V1 are sealed;
none may be repaired or resubmitted in place. The complete terminal record is
`H1_P0_PLAN1200_R03_B3_PREPOST_REPEATS3_EXECMODE_FAILURE_V3.md`.

## Terminal engineering failure: P0 plan1200 three-batch route stopped before sampling

The requested raw1200/first1000 three-repeat P0 plan experiment is terminal
as an engineering failure. GPU planner array `31561` launched all three
repeats with seeds `17029`, `27183`, and `31415`. Every task passed the A800,
PyTorch `2.4.0+cu121`, and SMACT `3.1.0` gates, entered
`planner_sample1200`, and then failed `1:0` at the same import:
`ModuleNotFoundError: No module named
'scripts.sample_llada_dynamic_crystals'`. The referenced module file exists
in the frozen source inventory, but the packaged `scripts` directory was not
made an unambiguous regular Python package. No plan row was emitted.

Dependent normal-CPU assembler `31562` sealed `status=failed`, an empty
repeat list, and `three_independent_plan_batches=false`. It did not submit
the separate R03 or B3 body arrays. Consequently no pre-refine or post-refine
CrysLLMGen/S.U.N. metric exists for this route, and the failure must not be
interpreted as a P0, R03, B3, refiner, or metric result. No repair, retry,
replacement, filter, rerank, training, or RL occurred.

The two-hop-returned minimal evidence bundle is
`execution/h1_p0_plan1200_r03_b3_prepost_repeats3_v1/evidence/planner_failure_evidence_31561_31562.tar.gz`
(3377 bytes; SHA
`ecadf983ac9f1676d637c7c239299729f39cf0e667c0823962c981452efa2da3`).
The detailed terminal record is
`H1_P0_PLAN1200_R03_B3_PREPOST_REPEATS3_FAILURE_REPORT_V1.md`. Continuing
requires explicit authorization for a new immutable V2; V1 will not be
modified or resubmitted.

## Terminal edge: R03 raw Plan × B3 repeated S.U.N. comparison complete

The requested fixed-Plan comparison is terminal. GPU repeat array `31549`
completed four independent A800 process realizations at 256 all-attempt
ordinals each, and normal-CPU paired assembler `31550` completed `0:0`.
The candidate is byte-frozen R03 P0 raw first256 + B3 + D2 safe-axis +
model_494 refine800; the control is the historical R03G result with the same
raw Plan and protected B0. The control was not rerun. Pairing is by repeat and
generation ordinal; each repeat has exact McNemar results, and formal
uncertainty uses a 50,000-draw hierarchical paired bootstrap over repeat
blocks and then ordinals. Pooled 1,024-attempt counts are descriptive only.

| repeat | B3 Direct comp/struct | B0→B3 novel | B0→B3 unique | B0→B3 strict S.U.N. | B0→B3 meta S.U.N. | B0→B3 hull evaluated |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 216/251 | 227→234 | 248→251 | 28→26 | 122→132 | 218→220 |
| 1 | 216/251 | 224→227 | 248→251 | 31→26 | 123→120 | 215→214 |
| 2 | 216/251 | 226→225 | 248→251 | 29→24 | 125→127 | 217→213 |
| 3 | 216/251 | 227→228 | 248→251 | 29→23 | 126→125 | 218→215 |

Strict full S.U.N. is the decisive result: B3 changes the endpoint by
`-1.7578125` percentage points, with hierarchical 95% CI
`[-3.22265625, -0.390625]`, `P(Δ>0)=0.00414`, and repeat differences
`[-2,-5,-5,-6]`. Meta full S.U.N. changes by `+0.78125` pp but is mixed
across repeats (`[+10,-3,+2,-1]`; 95% CI `[-2.24609375,4.1015625]`).
Novel and novel-unique each change by `+0.9765625` pp with intervals crossing
zero; unique representatives increase by `+1.171875` pp with a lower CI
bound of zero; hull-evaluated changes by `-0.5859375` pp with an interval
crossing zero. The descriptive pooled strict count is 117→99, whereas meta
is 496→504 and novel is 904→914.

Thus B3 improves completion/uniqueness slightly but reproducibly converts
fewer R03 raw plans into strict full-S.U.N. successes. It is not promoted.
Composition validity and other Direct metrics are retained only as
descriptive pipeline diagnostics: `comp_valid` is not a DLM endpoint and is
not used to attribute this S.U.N. difference to B3. There was no MP query,
retry, replacement, repair, filter, rerank, checkpoint reselection, automatic
downstream, or RL. The frozen terminal report is
`execution/h1_r03_raw_plan_b3_repeats4_direct_sun_v5/TERMINAL_REPORT.json`,
SHA `101382719310c35f643dfd5b9051834946582058c7a42c0fbd524741b4da6f91`.

## Terminal edge: current-run four-cell Direct/S.U.N. complete

The user-authorized V14 diagnostic now has exactly 256 all-attempt rows for
both P0 and SFT-v2. The local-only exact SMACT4 audit is complete and
SHA-bound. V18's normal-CPU SMACT3.1 assembly computed the complete formal
legacy result before an old-P0-schema identity gate labeled the job failed:
P0 is 128/256 (50.0%) and SFT-v2 is 195/256 (76.171875%), a gain of 67/256.
The same ledger shows exact-SMACT4 valid 135 versus 122, but uniform-primary
56 versus 101 and all-metal shortcuts 72 versus 10. Thus SFT-v2 strongly
changes the chemistry mode while not improving broad exact validity.

V20 job `31318` failed closed because the local audit manifest correctly
binds the raw-generation source inventory `4d8e7bde...`, while the repaired
assembly source inventory is `319b7591...`. No science bytes changed. V21 and
V22 then stopped before SBatch on two generator-anchor checks. V23 also
stopped before SBatch because its new root retained `_v20_` and therefore
triggered the launcher's exact stale-marker guard. None was rerun or created
an assembly job. V24 kept the same raw/assembly source separation and exact
guard, but used a clean immutable root without a parent-version marker.
Normal-CPU assembly job `31329` completed `0:0`; its SFT terminal-report SHA
is `cf51e406...` and stage-summary SHA is `445d58c2...`. It preserves the
complete V18 science above, emits the scientific-stop marker, and launches no
downstream or RL work. No A800 SMACT4 execution occurred.

DLM B0-v4 job `31308` produced all 1,773 synthetic states, 2,208 actual
rollout states, and 64 actual attempts, then failed a rescore identity gate.
The cause is BF16 batch geometry: rollout production used registered batches
up to eight, while the rescore regrouped states differently. B0-v5 kept
producer rollout bytes/batching unchanged, replays each serialized state in
its exact producer batch under the original `5e-4` gate, and uses the
historical fixed-panel batch size one for B0/B3 scientific scores. Its frozen
archive SHA is `ce94d793...`; gpu job `31323` completed `0:0` on node99. The
producer replay passed with maximum/mean/p95 absolute delta all zero, and the
frozen panel-manifest SHA is `6cc3d810...`. B3-v2 then stopped before SBatch
because its strict adapter expected the old run-root once in each sbatch file,
whereas the frozen train and scorer files each contain it three times. V3
fixed only those counts and passed every source identity, then stopped before
SBatch because A800's tar lacks `--sort=name`. V4 commit `3b5d775` reuses the
already successful B0 portable `tar -czf` pattern. Training job `31330`
completed `0:0` in `00:56:59` at exactly 1,696 updates, and dependent scorer
`31331` completed `0:0` in `00:10:42`. The terminal adapter SHA is
`ab4f3b82...`; training and score terminal SHAs are `1f9ab27d...` and
`755b5b86...`. B3 improves token-weighted NLL on IID (-0.276611), D1
(-0.337032), and synthetic safe-axis (-0.363534), but worsens the actual
protected-B0 rollout (+0.161890). The required two-panel transfer condition
is false, so B3 is not promoted and no ratio sweep was launched. No automatic
downstream, S.U.N., or RL job was submitted.

The user-mandated current-run matrix is now terminal. GPU array `31374`
completed all four tasks `0:0`, and its `afterany` normal-CPU assembler
`31375` completed `0:0`. Every cell used the same 256-ordinal seed ledger,
D2 safe-axis body generation, model_494 refine800, GCD-before-comp-valid R03
Direct scoring, and the same completed-cache R03E S.U.N. path. All raw
failures remain in the 256-attempt denominator; there was no retry,
replacement, repair, filter, or rerank.

| cell | comp/joint | structure | COV-P | COV-R | novel | unique | novel-unique | strict S.U.N. | meta S.U.N. | hull evaluated/unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M00 P0+B0 | 216/256 | 247/256 | 96.4844 | 85.2200 | 220 | 246 | 219 | 21/256 | 116/256 | 216/3 |
| M10 SFT-v2+B0 | 214/256 | 246/256 | 96.4844 | 67.5768 | 207 | 246 | 207 | 13/256 | 91/256 | 201/6 |
| M01 P0+B3 | 223/256 | 252/256 | 98.4375 | 83.2854 | 223 | 252 | 223 | 19/256 | 119/256 | 220/3 |
| M11 SFT-v2+B3 | 213/256 | 247/256 | 96.4844 | 71.6781 | 211 | 245 | 210 | 16/256 | 102/256 | 203/7 |

M01 is the descriptive leader on composition/joint, structure, novelty,
uniqueness, and meta S.U.N.; M00 remains the strict-S.U.N. leader and the
protected incumbent. At P0, B3 improves composition/joint by 7/256
(+2.734375 pp; paired-bootstrap 95% CI [0.78125, 4.6875] pp; exact McNemar
p=0.015625), while strict S.U.N. changes by -2/256 and is not significant.
SFT-v2 does not improve composition or strict S.U.N. at either body and
reduces meta S.U.N. at B0 by 25/256 (-9.765625 pp; 95% CI
[-17.96875, -1.5625] pp; p=0.0260756). The frozen decision is therefore
`complete_evidence_report_no_automatic_promotion`; formal promotion is false
and M00 remains protected.

The common Planner-union MP cache is complete at 455 chemical systems after
resolving 257 missing systems. Its SHA is
`165458334295920c7d15769895d5c56d152c932ba02e27c15c6e20b64db18d0d`;
the API credential was destroyed before Slurm and no MP API access occurred
inside the jobs. The complete terminal report is frozen at
`execution/h1_four_cell_direct_sun_v4/TERMINAL_REPORT.json`, SHA
`cdd23113f86e97c5f747e7c97cf24a531231d68b32420cdf03909d8de2806fb6`.

The Evidence-First workstream is active on branch
`codex/evidence-first-sun-msun`. The V8 optimizer audit repair passed both
two-update smoke arms, and V10 completed both fixed-endpoint 4,505-update
training arms plus the common-ledger P0/SFT-v2/SFT-v2-C raw64 generation.
V12 then completed the requested formal SMACT 3.1 recomputation. On the same
64-attempt ledger, P0 is 34/64 legacy composition-valid, while SFT-v2 and
SFT-v2-C are both 52/64: an absolute gain of 18/64 (+28.125 pp). SFT-v2's
paired exact McNemar p-value is 0.000912234.

Neither candidate is formally promoted. SFT-v2 misses the registered element
coverage, mean-N drift, and no-new-failure-class gates; SFT-v2-C additionally
misses parse and unique-formula gates and has larger distribution drift. The
V12 evaluator also exposes a schema-asymmetric P0 embedded-validator identity
check; this does not alter the recomputed SMACT 3.1 counts. Per the user's
explicit override, SFT-v2 alone proceeds through a diagnostic raw256 and
protected B0+safe-axis+model_494 Direct/S.U.N. chain. Independently of its
outcome, the mandatory DLM B0/B1/B2 inventory and B3 route now proceed. No RL
is authorized.

V13 preparation passed but its one-time submit command stopped before the
lock/SBatch boundary because it checked relative manifest entries outside the
run root; no V13 job or generation output exists. Immutable V14 changes only
that working directory. Its diagnostic raw256 array `31236_[0-1]%2`
completed on `gpu` for P0 and SFT-v2 (256 raw attempts per arm, common ledger
SHA `d5a3ac87458969816a0b27313fd9deecae47d2ddb10289ec08b9d93c5db48669`).
V24 completed formal assembly; downstream remains deliberately manual.

The mandatory DLM artifact inventory is also complete. Protected B0 is bound
to its frozen checkpoint/SHA; historical B1/B2 are bound to their terminal
1,696-update checkpoints and fixed-panel/dependency evidence. B2 remains a
non-revivable scientific stop because its dependency margin did not exceed
B1. The four B0-v5 state panels are now frozen, and B3 V4 training plus its
dependent scorer were submitted only after those identities were sealed.

The user has additionally required complete same-pipeline comparisons, not
candidate-only diagnostics. The frozen matrix is P0+B0 (protected control),
SFT-v2+B0 (Planner main effect), P0+B3 (DLM main effect), and SFT-v2+B3
(interaction), all using safe-axis, model_494 refine800, Direct, and S.U.N.
Historical summary rows remain context only; every requested cell must produce
its own current-run raw-denominator evidence. Scientific gate failures remain
visible labels, while engineering failures still fail closed.

The DLM state-panel contract and minimal B3 grouping implementation are now
frozen locally before reading any B3 result. Synthetic IID/D1/safe-axis panels
use validation ordinals 0..99; the real B0+safe-axis trace uses 0..63. Actual
wrong commitments remain visible and are counted, while NLL targets only the
frozen ground-truth tokens still masked in the active group. B3 will be scored
on these exact frozen state bytes. The new `d2_safe_axis` training policy is
composition, lattice, every X group, every Y group, then every Z group, with
zero mixed-axis groups and no Z-before-XY. B0-v5 panel job `31323`, B3
training `31330`, and dependent scorer `31331` all completed. Their mixed
terminal evidence is frozen without changing the contract; the complete
four-cell evaluation now consumes B0 and B3 read-only.

The B3 execution package is also frozen before any state-panel or B3 result.
It is a one-arm reuse of the successful historical two-A800, 1,696-update
training shell: B0 initialization, the same R5-C bytes/order/seeds, LR 5e-5,
and terminal checkpoint only. The sole scientific change is
`d2_safe_axis` at IID:planned 2:1. Its dependent job scores B3 on the exact
B0-frozen panel bytes; neither job can submit body64, a ratio sweep, S.U.N.,
or downstream work. The B0 terminal and manifest are frozen. V4 training and
its scorer are terminal; B3 remains an unpromoted diagnostic checkpoint used
only in the explicitly authorized four-cell comparison.

## Connection and read-only audit

- 5090 is reachable only through port 2213 and the configured private key.
- Existing A800 tmux sessions `ssha800` and `ssha800_2` were present with
  `pane_dead=0` and `pane_current_command=ssh` at the last audit.
- A800 access remains restricted to those existing sessions. Neither may be
  recreated or reconnected; if both fail the workstream stops.
- Local-to-5090 transfers are unrestricted. Only 5090-to-A800 SCP attempts are
  rate-limited, with at least ten minutes between attempts.
- A800 had no user jobs at the initial audit. Every submission still requires
  a fresh `sinfo`/`squeue` snapshot.
- Audit marker: `__EF_AUDIT_DONE__` at 2026-08-08T12:59:52+08:00.

## Frozen evidence already obtained

- Protected P0/B0/model_494 identities remain unchanged.
- Transfer v1 failed before source creation because archived executable files
  contained CRLF. The immutable failure evidence is retained.
- Corrected archive from commit `ef82ffc` has SHA
  `79d1e6e60b06e61e0654ebdafcfad828cb86888b7f17fff9bcbfeae2a97e42b9`.
- The corrected A800 source bootstrap passed under run
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_v1`:
  source inventory SHA
  `f429f63ef42ead9162149ddce135bb35da7fc2ad94d3e86c8135506063f6a801`,
  source archive SHA
  `555fdcc59901aad4bd4ceae685b28827ab3d5b312434706ce5a319d2c815de25`,
  source manifest SHA
  `b0fab254166928df999ffeb81199aeca699904508895fb98f73645687e8faf5d`,
  590 files.
- The attempted A800 portable SMACT4 runtime failed before project tests or
  science because `contourpy-1.3.3` required a newer manylinux tag. Its partial
  build and logs remain immutable; it is not repaired or reused.
- Two independent engineering reviewers converged on the same safe remedy:
  exact SMACT4 must be machine-separated, complete, SHA-bound, and joined
  one-to-one; stale, missing, duplicate, extra or substituted rows must fail.

## User-frozen evaluator split

A800 now uses only its existing Python 3.10/SMACT 3.1 environment. Exact
SMACT 4.0 must never be installed or executed on A800. The earlier foreground
zip-overlay probe was already active when this rule was frozen; it is ignored,
is not a gate, and must not be interrupted under the tmux safety contract.

The replacement identity is `h1_chemistry_first_sft_v2_smact_split_v2`:

1. A800 exports the complete immutable MP20 legacy snapshot using SMACT 3.1.
2. The local machine runs only the explicitly authorized exact-SMACT4 witness
   builder from the same frozen source inventory and returns a sealed
   one-row-per-source ledger.
3. A800 verifies parent/source/ordinal/material/formula/witness hashes and uses
   the exact POS intersection to build data, smoke, train, and generate raw64.
4. Raw64 generation stops. The local machine produces complete exact-SMACT4
   per-arm audit reports; A800 verifies every report against its current raw
   bytes before SMACT3 gate assembly.
5. Raw256 repeats the same generation/local-audit/assembly split only for
   raw64-passing candidates.

No A800 submission contains `SMACT4_PYTHON` or an exact-SMACT4 executable
path. The local exception is limited to witness and raw secondary-audit
production; all model tests, tokenizer checks, data assembly, training,
generation, and primary evaluation remain on A800.
The replacement source also excludes the retired portable-runtime builders,
requirements, and launcher, so the A800 snapshot contains no dormant SMACT4
runtime entry point. The immutable failed runtime run remains preserved as
historical evidence outside this source.

## Split-source audit terminal and minimal repair

The first split-source run reached an engineering terminal before Slurm:

- run root:
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2`;
- source tests `48/48`, isolated tests `48/48`, and preflight-focused tests
  `35/35` all passed;
- the only failed check was `legacy_evaluator_source_sha`;
- archived `composition_validity.py` had CRLF SHA `c078c1ca...`, while its LF
  normalization exactly matched the frozen evaluator SHA `ca1c94f5...`;
- terminal SHA:
  `1fc5776c66c3ba34f9afc991200cec81734a35168092e14fd5da2b633956765d`;
- no Slurm, GPU, training, generation, SMACT4-on-A800, or science occurred.

The allowed repair is packaging-only. It writes the Git archive directly to
its output file, requires LF for source text classes, checks the evaluator
member SHA before extraction, and uses new transfer/staging/freeze/run paths.
The new run root is
`20260808_h1_chemistry_first_sft_v2_smact_split_v2_packaging_repair_v3`.
Following the user's audit-budget instruction, its source gate runs one
protocol test, one isolated inventory/SHA check, and one focused preflight;
the duplicate broad source/isolated test pass is not repeated.

## Packaging repair v3 and evaluator-split inputs

The packaging-only v3 repair passed its reduced A800 source gate and is now
the immutable active run root:

- run root:
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_packaging_repair_v3`;
- source-input archive SHA:
  `d64bf8790fe9a2df4926e25e045883f0f6077d2b8c1d8c86d73b3b3bb7c8f0cf`;
- source inventory SHA:
  `410fb34d2543f620fff012fde574a75a935bcaf54e41449fb529c12eb913c20c`;
- frozen source archive SHA:
  `3c5d93f697e97e29ee233bcefab41c8636599711e6decaf971eeb383f8137559`;
- source manifest SHA:
  `50f9e541a8fcafe13a4d953b1907f949259be4564bffd576379db72fcdbcf89a`;
- archived legacy evaluator member SHA:
  `ca1c94f583e0c97a172b5c9b7ba96505257fd74dedfc618b584c34486ac1f178`;
- the one protocol test, isolated inventory/evaluator identity check, and
  focused 35-test preflight all passed; A800 used Python 3.10.18 and
  SMACT 3.1.0, and did not import or execute SMACT 4.

Legacy snapshot job `31025` completed `0:0` in `00:03:38`. Its sealed bundle
SHA is `d65a682c8e29820938a8e4637963dd56f18f2f6d893abd77add02043fb6a13bf`
and report SHA is
`351a13f8c462a9a0ee377c1d07ebaafb426a0b64cffc2c9c3ad950725e3deefb`.
It contains exactly 27,136 train, 9,047 validation, and 9,046 test rows.

The explicitly authorized local-only exact-SMACT4 witness build also passed.
Its persistent isolated runtime terminal SHA is
`0853bbea0c714f8a3150b08f3c94bdf1ec03b9cd49585d3a16533a9f88ea67d6`.
The witness manifest SHA is
`d21698e29664c607541d7ab644250e93e18cfb2a0cd03d1687a270b42c8ccd32`,
ledger SHA is
`ab687c5f16dc64887de2446c3aae20c0e15021a3bfa4ffe71dd6cfe09b482c93`,
and deterministic archive SHA is
`b896516351a76b869cc10d8c95a321ee53bb1cf25e3ed38d207d1dd40b7322c3`.
The train census is 7,079 legacy-primary and 4,451 exact uniform-primary;
validation is 2,264 and 1,406 respectively. Official/witness parity is true.
The sealed ledger was transferred and frozen as data bytes only; SMACT 4 was
not run on A800.

Data job `31035` completed `0:0` in `00:15:47`. Both candidates contain
36,038 train and 9,047 validation records. The record multiset is identical;
base and curriculum order SHAs differ as registered, the 3,603-record
curriculum prefix alternates correctly, and all data/tokenizer audits pass.
There are zero invalid unconditional formula targets, generated charge-field
leaks, token-weight failures, or truncation failures. Frozen optimizer
geometry is 4,505 updates, 135 warmup steps, and six microbatches in the final
accumulation group. Audit-report SHA is
`1c9a5e2cba51a1258acf107255ca308c7c9f7122d50f5ae3a09ed97d2681612a`;
order-ledger SHA is
`c31df9dd44bbe1ea75a99131899d4e6ea7131d1f230b39ecfb25f730d5406a43`.

Minimal smoke array `31036_[0-1]` reached `FAILED 1:0` for both tasks after
`00:13:08`. It loaded the base model but stopped before the first model forward:
all 448 candidate/reference tensor keys matched and no noncandidate parameter
was trainable, but PEFT 0.16 had loaded the trainable candidate copy in FP32
and the frozen reference copy in the BF16 base dtype. The resulting maximum
absolute difference was `6.103515625e-05` for both candidates. No optimizer
step, formal training, scientific generation, or SMACT4-on-A800 occurred.

The frozen V4 repair loads both copies through the same protected-P0 FP32 PEFT
path, then freezes every reference parameter before forward or optimizer
construction. It changes no model payload, data, task order, prompt, seed,
optimizer, ledger, evaluator, or gate. The V3 data/legacy snapshot will be
reused byte-for-byte under a new manifest; a fresh dual-arm GPU smoke remains
mandatory before training.

V4 source freezing itself completed with source inventory
`2e7997cfc9894db5ba099d4fc3bfa18440b10d0c29e36927768dc35eaef03968`,
archive `193816fb1aa6919f0bed755f5fae9459f7a0bd63afc797b579dc741f43cb8b73`,
and manifest `02fcb50da21bd54e314aeaa0e09cf2b9cbb71300f24a9dddeaca009f01beffc1`.
Before executing its A800 source gate, a static check found that the isolated
archive path was still bound to the existing immutable V3 directory. The gate
was not executed, no pass marker was written, and no job was submitted.

Two independent propose/red-team reviews agreed that deleting or reusing V3,
hand-writing the pass marker, or changing only one path would be indefensible.
The active repair is therefore a new path-only V5 identity:
`20260808_h1_chemistry_first_sft_v2_smact_split_v2_source_gate_path_repair_v5`.
Every active script/SBatch/config path moves to V5 and isolated extraction is
scoped under `${RUN_ROOT}/isolated_archive_test`. V4's PEFT repair, all model
and science code, data, prompts, seeds, optimizers, ledgers, evaluators, and
gates remain unchanged. SMACT4 remains forbidden on A800.

V5 source inventory `7cef386be864eef760088fe8bb7c7073b7d1e908ca92e33f7d8a9a951ffebc91`,
archive `e37b073b34f1d55f55e96683534bcdeb56ed33f52b60baa1d7fa579e849e1958`,
and manifest `10ddf8a7be3678b3440842ba66f4f97a7943862d4983be32ed62b7012a41829a`
are frozen. The reduced A800 source gate passed: 16 protocol tests, isolated
inventory/evaluator identity and 35 focused preflight tests all passed under
Python 3.10.18 and SMACT 3.1.0; SMACT4 was not executed. V3 data reuse also
passed with reused-tree manifest `951aa186b37fd821d35f6a9fe63919abfc8753c4bd4ffce2d96968a039457981`.
Fresh dual-arm smoke array `31064_[0-1]` also reached `FAILED 1:0` for both
tasks after `00:15:02`. Both arms stopped at the protected-P0 identity gate
before the first forward or optimizer construction. Their identity reports
are byte-identical to V3 (SHA
`ac04b54094136dddb3fe5f6bbe9b10b369ec76b0f736345d00b858ae3e29889c`):
all 448 keys match, all 448 values differ, and maximum absolute difference is
`6.103515625e-05`. The V4 loader-flag change therefore did not alter the
runtime tensors. The immutable V5 failure report SHA is
`3f161979bef1de77351ba9178aa59cbbaa794cfcb29e9f9bb7b11884022d9be8`.
At that historical V5 boundary, formal training and scientific generation
remained unsubmitted; later V10/V14/V24 evidence supersedes that status.

## Immediate critical path

1. Preserve V1, V2, V3, native supplement V1, all Slurm records, terminal
   reports, frozen cohorts, caches, and returned evidence byte-for-byte; do
   not repair, requeue, or resubmit any of those identities.
2. Make no scientific interpretation from V3: body generation never started,
   so all requested CrysLLMGen/S.U.N. and native full-1,000 post-refine metrics
   remain unavailable.
3. Continuing requires explicit authorization for a new immutable repair that
   aligns the cohort `parsed` schema with the body consumer and extends
   preflight to assert the exact producer/consumer schema. Reuse of planner
   cohorts or MP caches must be separately justified and authorized; there is
   no automatic submission.
4. Retain P0+B0 (M00) as the protected incumbent. V3 promotes neither B3,
   SFT-v2, nor any downstream route.

No Planner or DLM RL is authorized.
