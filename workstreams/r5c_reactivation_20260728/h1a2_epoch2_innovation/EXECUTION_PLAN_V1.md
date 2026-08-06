# H1-A2C JointChem v1 execution plan

Status: implementation complete; local and remote preflight in progress  
Baseline: frozen H1-A2 epoch-2 adapter  
First scientific scope: Planner only

## Runtime identity

The local restored baseline and the A800 project-root runtime use different
directory layouts. The executable resolves the restored baseline when present
and otherwise the project root, but launches only after
`RUNTIME_REQUIRED_SHA256.txt` verifies every imported H1-A2 sampling,
composition-validation, and Plan-analysis source file. This path fallback
changes no model, data, loss, sampling, or selection semantics.

## Fixed sequence

1. Verify the three historical H1-A2 JSONL source SHAs.
2. Build a byte-addressed row ledger.
3. Recompute formula validity with the frozen CrysLLMGen-compatible SMACT validator.
4. Freeze deterministic chemistry and joint-tuple negatives.
5. Build exactly 3,200 training microbatches:
   - 2,560 chemistry-valid positive replay rows;
   - 640 epoch-2 anchor rows from the unmodified source distribution.
6. Build a fixed 1,024-row validation panel.
7. Train:
   - P1 ValidReplay;
   - P2 JointChem.
8. Save adapters and teacher-forced Plan diagnostics every 50 updates through update 400.
9. Evaluate the unchanged epoch-2 initialization on the same fixed 128-row
   validation prefix, then preselect one checkpoint per arm only if its
   positive NLL is noninferior within 1% relative. P2 margins are the mean of
   per-row `negative_NLL - positive_NLL` values on identical chemistry-valid
   rows, never a difference between separately aggregated populations.
10. Sample frozen P0 and the preselected P1/P2 checkpoints on a common
    all-attempt 512 Plan ledger; select the Plan arm without S.U.N, CHGNet,
    MLIP, MP API, hull data, or generated crystals.
11. Only the selected candidate and frozen P0 proceed to exact paired-256 crystal generation.

## Training contract

| Parameter | Value |
|---|---:|
| GPU | 1×A800 |
| Maximum CPU | 8 |
| Base | Meta-Llama-3-8B, non-Instruct |
| Initialization | frozen H1-A2 epoch-2 adapter |
| Batch | 1 |
| Gradient accumulation | 8 |
| Updates | 400 |
| Maximum length | 768 |
| LR | `2e-6` |
| Warmup | 25 |
| Scheduler | cosine |
| Weight decay | 0 |
| Gradient clip | 1.0 |
| Seed | 17 |
| Precision | BF16 |
| Checkpoint/eval cadence | 50 updates |

P2 active-positive loss:

```text
0.75 * positive target-only CE
+ 0.15 * softplus(0.10 + positive_NLL - chemistry_negative_NLL)
+ 0.10 * softplus(0.10 + positive_NLL - joint_negative_NLL)
```

Missing negative terms are omitted and the active weights are renormalized. Anchor rows use positive CE only.
Ranking is also disabled for every composition-invalid validation row.

## Leakage firewall

Training artifacts may not contain:

- S.U.N outputs or labels;
- CHGNet, MatterSim, or other MLIP fields;
- formation energy or hull fields;
- MP API keys or query results;
- generated candidate rankings.

The builder emits only whitelisted Plan text, source-row hashes, deterministic negative text, and non-energetic chemistry diagnostics.

## Stop conditions

- source SHA mismatch;
- a prompt contains a sample ID when historical no-ID mode is required;
- chemistry negative changes total atom count or element arity;
- joint negative changes formula, anion, or charge;
- output leakage scan fails;
- deterministic rebuild changes an output SHA;
- target-only masking includes prompt tokens;
- the real-tokenizer/A800 loss preflight fails prompt-perturbation invariance,
  target sensitivity, exact pairwise numerics, or bitwise repeat;
- a checkpoint worsens fixed-panel positive NLL by more than 1% relative to
  the unchanged epoch-2 initialization;
- a chemistry or joint likelihood margin is missing, non-positive, or not
  aggregated from matched rows;
- GPU count is not exactly one;
- allocated CPU exceeds eight;
- checkpoint identities or update counts are incomplete.
- any required frozen H1-A2 runtime source SHA mismatches.
- the execution source-manifest SHA or epoch-2 adapter SHA mismatches.

No automatic crystal evaluation is authorized by the training job.
