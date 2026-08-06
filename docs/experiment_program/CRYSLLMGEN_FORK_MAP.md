# CrysLLMGen Fork and Modification Map

Date: 2026-07-20

## Upstream identity

- Repository: `https://github.com/kdmsit/crysllmgen.git`
- Local reference: `reference/crysllmgen/`
- Upstream commit: `94bb287751cd20a882c7c1df7ca736633d78e5e1`
- License: MIT; the upstream copyright and license notice must remain in every
  copied or substantially derived source file.

The reference working tree is not an active development directory.  Ignoring
Windows line-ending changes, its only substantive source change is the existing
`compute_metrics.py` worker-cap patch controlled by
`CRYSLLMGEN_METRICS_NUM_CPUS`; that patch is retained because all BLAS and
metric worker counts are restricted to one.  The missing MP20 training CSV is a
data-state difference, not a code fork.

## Engineering decision

The active method is a CrysLLMGen fork, not a from-scratch model that merely
uses CrysLLMGen as an external baseline.  The reference checkout stays under
`reference/crysllmgen/`; its 26-file source-only snapshot is byte-preserved
under `crystal_dlm/wqcodiff/crysllmgen/upstream/` and bound by
`crystal_dlm/wqcodiff/crysllmgen/UPSTREAM_MANIFEST.json`.  Active derived
modules live beside, never inside, that snapshot.  The existing attempt,
Wyckoff chart, bridge, evaluator, and audit modules in
`crystal_dlm/wqcodiff/` are attached to the fork as extensions.

The old `WQCoDenoiser` and Day-7 DLM/D3PM machinery are retained for provenance
and diagnostics but are not the primary paper backbone.

## Source mapping

| Upstream source | Active derived responsibility | Required preservation |
|---|---|---|
| `llm_finetune.py` | Llama-3-Instruct LoRA on atom and Wyckoff proposal/edit targets | SFT data flow and LoRA entry point |
| `crysllmgen_sample.py` | Attempt-accounted Llama proposal followed by diffusion refinement | one-pass proposal-to-refiner orchestration |
| `models_ddpm/cspnet.py` | expanded-atom CSP message passing inside a Wyckoff wrapper | six-layer CSPNet computation and checkpoint compatibility |
| `models_ddpm/diffusion.py` | lattice/free-coordinate training and predictor-corrector sampling | beta/sigma schedules and refinement semantics |
| `models_ddpm/diff_utils.py` | registered CrysLLMGen noise schedules | equations and indexed timestep convention |
| `models_ddpm/data_utils.py` | MP20 structure/graph conversion and WQ-expanded atom graphs | lattice/PBC conventions |
| `compute_metrics.py` and `eval_utils.py` | official CrysLLMGen metrics | metric definitions; only bounded workers are allowed |

Native `index_add_` or fully connected edge implementations may replace
`torch_scatter`/PyG kernels only as an engineering port.  Such a replacement
must pass deterministic tensor-parity tests against the upstream computation
and must be shared by every matched baseline and proposed method.

## Method changes on top of CrysLLMGen

1. Replace unrestricted atom-by-atom text with a constrained Wyckoff proposal
   containing space group, lattice chart, orbit type, species, and orbit free
   coordinates.
2. Expand the proposal through the Wyckoff affine maps before the inherited
   CSPNet; project atom-coordinate scores back to the free-coordinate tangent.
3. Preserve the CrysLLMGen one-way, topology-frozen path as
   `C-ATOM-OFFICIAL`, `C-ATOM-MATCHED`, and `C-WQ-HANDOFF` controls.
4. Add geometry-triggered Llama edit commands during refinement, with explicit
   orbit birth/death/type/species changes and a target-stratum bridge.
5. Replace CrysLLMGen's generate-until-valid loop with exactly one terminal
   attempt record.  Parser, bridge, timeout, and refinement failures remain in
   every denominator.
6. Load the user-supplied Llama checkpoint only from its registered offline
   path; remove network login and implicit Hugging Face model resolution.
7. Add stable seed derivation, token/forward/FLOP accounting, artifact hashes,
   and the frozen R5-C MLIP-SUN evaluation contract.

## Parity gate

Before any Wyckoff training, disabled-extension mode must reproduce the
upstream atom pipeline under the immutable contract
`configs/experiments/wyckoff_codiffusion/crysllmgen_parity_v1.json`.  The
dependency-light selector/comparator/auditor lives in
`crystal_dlm/wqcodiff/crysllmgen/parity.py`; the exact non-repairing atom-text
boundary is `crystal_dlm/wqcodiff/crysllmgen/atom_text.py`.  Disabled-extension
mode must reproduce the upstream atom pipeline on 256 hash-fixed proposals:

- identical parsed atom types and atom counts;
- lattice and fractional-coordinate maximum absolute difference below
  `1e-6` for deterministic one-step/tensor tests;
- identical beta/sigma tables and timestep traversal;
- identical checkpoint parameter names or a complete deterministic mapping;
- no hidden retry, survivor-only denominator, or network access.

Failure of this gate means the project is not yet legitimately based on the
CrysLLMGen implementation and blocks the main experiment cycle.
