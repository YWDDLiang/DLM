# Legacy-to-mainline compatibility map

The paper branch adds stable interfaces around audited code. It does not erase
historical experiments or move immutable scientific assets.

## Active lineage

| Paper-facing stage | Existing source of truth | Planned stable wrapper |
|---|---|---|
| Build typed Planner data | `scripts/build_c3fd_llama_fused_data.py` / Slurm109 | `paper-build-planner-data` |
| Train Science-Constrained LLM Planner | `src/scripts/train_c3fd_llama_typed_planner.py` / Slurm110 | `paper-train-planner` |
| Sample one Plan | `src/scripts/sample_c3fd_llama_typed_planner.py` / final Slurm125 | `paper-sample-plan` |
| Build Plan-conditioned DLM data | `scripts/build_c3fd_native_sft_data.py` / Slurm86 | `paper-build-dlm-data` |
| Train Compact-V2 DLM | `src/scripts/llada_sft.py` / Slurm87 | `paper-train-dlm` |
| Train periodic-relation G2 | `src/scripts/llada_sft.py` + periodic relation core / Slurm129 method A | `paper-train-periodic-dlm` |
| Generate raw structures | `src/scripts/sample_sgtc_l6.py` + assembler / Slurm112/130 | `paper-generate` |
| Run fixed diffusion transition | `src/scripts/refine_dlm_with_crysllmgen.py` / Slurm131 | `paper-refine` |
| Evaluate and finalize | `eval_runtime/*`, Direct wrapper and `scripts/finalize_*` | `paper-evaluate`, `paper-finalize` |

Wrappers must load one versioned manifest and delegate to these implementations;
they must not duplicate model mathematics.

## Retained ablations, not active entrypoints

- C3FD-only proposal and old H1-A2 rich Planner: composition and interface
  baselines.
- F/M rich expander route (`slurm/101--108`): Planner-integration ablation.
- Deterministic H1-A2 fusion (`slurm/99--100`): cancelled interface route.
- Teacher/predicted/masked/minimal multiview Compact-V2 builders and alignment
  (`slurm/85--95`, old3614): development assets only.
- Auxiliary-only G1 (`slurm/115/117`): isolates geometry losses without the
  periodic residual.
- Original short G2 (`slurm/116/126/127`): fresh prospective headline and
  shorter-training ablation.
- G2-PBC-RU method B: uncertainty-gate ablation; not promoted.
- D3PO, SGTC, CTV, AR and basin-distillation branches: historical negative or
  alternative mechanisms, not part of the final paper pipeline.

## Compatibility policy

1. Keep legacy files and historical run contracts addressable by their old
   paths.
2. Put paper defaults in a new versioned manifest/config namespace.
3. Implement thin CLI adapters with explicit arguments and checksum checks.
4. Never place machine paths, credentials, checkpoints or generated datasets
   in Git.
5. Test each wrapper against the frozen legacy command it represents.
6. Update the top-level README only after the paper wrappers pass parity tests.
