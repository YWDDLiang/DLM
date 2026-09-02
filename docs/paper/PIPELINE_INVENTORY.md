# Paper-mainline pipeline inventory

Status: active inventory on `codex/c3fd-g2-paper-mainline`. Scientific
experiments are frozen; this document maps the terminal implementation into a
single reproducible method without changing behavior.

## Coupled method

The mainline is one hierarchical conditional crystal generator:

\[
z\sim p_\phi(z\mid \mathcal C_{\mathrm{C3FD}}),\qquad
x_{\mathrm{raw}}\sim p_\theta(x\mid z;\mathcal R_{\mathrm{G2}}),\qquad
x_{\mathrm{final}}\sim K_{494}(\cdot\mid x_{\mathrm{raw}},z).
\]

`C3FD` defines typed scientific support inside the Llama decision process;
Llama turns that support into a coherent Compact-V2 Plan `z`; dynamic `7+4N`
is the exact Plan-to-structure contract; G2 is an internal periodic relation
path in DLM denoising; model494 is the fixed terminal transition of the same
coarse-to-fine realization.

## Canonical stage map

| Stage | Scientific role | Canonical implementation | Frozen entrypoints | Primary tests | Immutable evidence |
|---|---|---|---|---|---|
| Typed scientific support | Defines reachable atom-count, family, charge/valence and benchmark-compatible composition actions | `src/crystal_dlm/c3fd_planner_model.py`, `src/crystal_dlm/family_reachability.py`, `src/crystal_dlm/ccfd_v2.py` | `scripts/build_c3fd_planner_data.py`, `scripts/train_c3fd_planner.py`, `slurm/53_c3fd_v25_requested1000.sbatch` | `tests/test_c3fd_planner_model.py`, `tests/test_train_c3fd_planner.py` | `results/remote_screens/C3FD_V25_REQUESTED1000_FINAL.*`; `2000/2000` composition-valid |
| Science-Constrained LLM Planner | Reweights the typed C3FD support with a learned Llama materials prior and emits one global Plan | `src/crystal_dlm/c3fd_llama_fused_plan.py`, `src/crystal_dlm/c3fd_llama_typed_planner.py` | `scripts/build_c3fd_llama_fused_data.py`, `src/scripts/train_c3fd_llama_typed_planner.py`, `src/scripts/sample_c3fd_llama_typed_planner.py`, Slurm `109/110/125` | `tests/test_build_c3fd_llama_fused_data.py`, `tests/test_c3fd_llama_fused_plan.py`, `tests/test_c3fd_llama_typed_planner.py`, `tests/test_train_c3fd_llama_typed_planner.py`, `tests/test_sample_c3fd_llama_typed_planner.py` | jobs39046/39051/39128; final Plan `256/256` composition-valid; mean fused-vs-C3FD KL `0.06819`; `87.05%` nonzero-KL decisions |
| Shared Plan contract | Serializes global chemical intent and Compact-V2 structural fields identically at train and inference | `src/crystal_dlm/c3fd_native_plan.py`, `src/crystal_dlm/composition_identity.py` | Plan output from Slurm111/125 | `tests/test_c3fd_native_plan.py`, `tests/test_materialize_c3fd_llama_prospective_conditions.py` | `248/248` strict round trips; no species-order or validity flips |
| Plan-conditioned crystal DLM data | Teaches the same Plan schema to map to an exact crystal body | `scripts/build_c3fd_native_sft_data.py` | `slurm/86_build_c3fd_native_sft_data.sbatch` | `tests/test_build_c3fd_native_sft_data.py`, `tests/test_audit_c3fd_native_teacher_sft.py` | teacher data job38686, train/val `27136/9047`, manifest SHA `b77a1d0...3ae06` |
| Plan-Conditioned Crystal Diffusion Language | Generates lattice, species and fractional coordinates by parallel masked denoising | `src/scripts/llada_sft.py`, `src/crystal_dlm/dynamic_crystal.py`, `src/crystal_dlm/llada_generation.py` | `slurm/87_train_c3fd_native_teacher_sft.sbatch`; inference body path in Slurm112/130 | DLM SFT and dynamic-representation tests; exact round-trip tests to be grouped by paper wrapper | job38703 seed82017 step3392, adapter SHA `06cd5465...62b20` |
| Exact dynamic `7+4N` representation | Binds one lattice header and exactly N species/XYZ tuples to Plan N/composition | `src/crystal_dlm/dynamic_crystal.py`, geometry token support in `src/scripts/llada_sft.py` | `representation=dynamic_v1`, `max_length=382`, exact-axis sampler | frozen G0 round-trip audit; paper-facing parity test pending | G0 round-trip `248/248`, exact species order, zero validity flips |
| Periodic-Relational Denoising | Lifts q0 soft lattice/coordinates to a strict-PBC species relation graph and returns a zero-initialized residual to q1 logits | `src/crystal_dlm/periodic_geometry_ops.py`, `periodic_geometry_objective.py`, `periodic_relation_adapter.py`, `periodic_relation_runtime.py` | `slurm/129_train_g2_full_epoch_ab.sbatch`; promoted setting A / G2-PBC-R | `tests/test_periodic_geometry_objective.py`, `tests/test_periodic_relation_adapter.py`, `tests/test_periodic_relation_runtime.py`, geometry contract audit | audit job39150; A training39164; terminal A/B report `docs/G2_FULL_EPOCH_AB_FINAL_20260901.*` |
| One-trajectory realization | Applies the same frozen Plan/noise with no retry, replacement, reranking or best-of-N | `src/scripts/sample_sgtc_l6.py`, `scripts/assemble_raw_body_repeat.py` | headline Slurm126; full-epoch ablation Slurm130 | `tests/test_slurm126_g2_final_generation.py`, exact fast-Direct regression | prospective BASE/G2 raw Direct `118/121`; full-epoch A/B `128/130` on fixed256 |
| Fixed terminal diffusion | Refines the raw structure with model494 while preserving attempt identity and atom multiset | `src/scripts/refine_dlm_with_crysllmgen.py`, `scripts/assemble_grounding_repeat.py` | prospective Slurm126; full-epoch Slurm131; scale tau800 run39199 | `tests/test_slurm121_g2_model494_refine.py`, `tests/test_ctv_refiner_seed.py` | L6 raw→tau800 `10/66→48/230`; Plan1200 tau800 scale profile |
| Physical evaluation | Joins Direct validity, N/U/NU, CHGNet and official hull on the fixed profile denominator | `eval_runtime/run_full_reconstructed_eval.py`, `eval_runtime/finalize_official.py`, paper finalizers under `scripts/finalize_*` | prospective Slurm127; full-epoch Slurm131; scale Slurm145 | `tests/test_slurm127_g2_final_offline_eval.py`, `tests/test_full_reconstructed_eval_helpers.py`, finalizer/protocol tests | main `105/488`; prospective report; full-epoch report; Plan1200 `81/486` |

## Frozen mainline checkpoints and choices

- Planner: C3FD-v2.5 seed17 + Llama typed Planner seed85017 final.
- Plan schema: `C3FD_NATIVE_PLAN_V2`; exact prospective Plan SHA above.
- Base DLM: Compact-V2 job38703 seed82017 step3392.
- Periodic DLM: A / `G2-PBC-R`, full 1696-update endpoint.
- Sampling: stream17, DLM seed91117, temperature0.7, exact-axis, one trajectory.
- Refiner: model494, tau800, refiner seed101117 by sample index.
- Metrics: profile-specific fixed denominators; no survivor filtering or result selection.

The main reported result is Strict/Meta S.U.N. `105/488` per 1,000. The fresh
prospective G2 endpoint is the matched causal profile: refined `24/117` and
paired official hull `-16.43 meV/atom`. Full-epoch A is the promoted
implementation evidence (`128/256` raw Direct versus BASE `118/256`). The
Plan1200 profile supplies the independent scale result (`81/486` at tau800).

## Evidence boundaries retained by the paper branch

- The prospective `24/117` endpoint uses the registered step348 G2 from Slurm116;
  the full-epoch A endpoint is a later mechanism ablation and is never relabeled
  as the headline checkpoint.
- The fused Planner optimization uses the typed-witness `24558/8158` subset.
  The DLM SFT uses the full MP20 standard `27136/9047` split. No document may
  claim that the current Planner was trained on all MP20 rows.
- model494 and the Direct implementation are frozen inherited transitions.
  Their role is integrated into the coarse-to-fine method, while
  learned raw-DLM and complete-system evidence remain separately measurable.

Detailed content identities live in the execution manifests. Reader-facing
evidence is indexed by the named profiles in `configs/paper/mainline_v1.json`.
