# Build status

## Implemented

- Independent paper-facing Git repository and branch.
- Relative project layout with no dependency on the historical workspace.
- Public headline limited to `105/1000` Strict and `488/1000` Meta S.U.N.
- Planner prompt and confirmed scientific defaults.
- Planner, DLM, body inference, safe-axis core, and CrysLLMGen upstream source.
- Slurm stage layout and automatic dependency submission wrappers.
- Missing-checkpoint train/download policy and Planner frozen-Plan fallback.
- User-supplied `MP_API_KEY` policy with resumable final evaluation.
- Data, checkpoint, environment, and Plan placeholders with documentation.
- Standard-library contract tests and local syntax validation.

## Read-only A800 audit

A read-only audit was completed on 2026-08-21 through the existing A800 SSH
session. No remote file was created, modified, copied, or hashed. Private
cluster source paths are recorded only in the personal repository's transfer
ledger.

### Environment observed

The historical `diff_meets_diff` Conda environment is present. The following
installed versions were read from the environment metadata:

| Package | Observed version |
|---|---|
| Python | `3.10.18` |
| PyTorch | `2.4.0+cu121` |
| torch-scatter | `2.1.2+pt24cu121` |
| NumPy | `1.26.4` |
| Transformers | `4.54.0` |
| PEFT | `0.16.0` |
| Accelerate | `1.9.0` |
| pymatgen | `2025.6.14` |
| CHGNet package | `0.4.2` |

The official Materials Project query environment is also present with Python
`3.10.18`, `mp-api 0.45.13`, and `emmet-core 0.85.1`; it inherits the system
site packages from `diff_meets_diff`.

No committed Conda export or pip lock file was found. The live environments
are sufficient to produce those files later, but the read-only audit did not
create them.

### Model assets verified

| Asset | Verified contents |
|---|---|
| Planner base | Meta-Llama-3-8B directory present |
| H1-A2 epoch-2 Planner | adapter `167,832,240` bytes plus tokenizer/config files |
| DLM base | LLaDA-8B-Instruct directory present |
| B0 exact-length DLM | adapter `6,391,016,776` bytes plus tokenizer/config files |
| Continuous refiner | `model_494.pt`, `147,645,242` bytes |
| CHGNet evaluation model | packaged checkpoint, `4,863,221` bytes |

The checkpoint files remain on A800 and have not yet been transferred to this
repository.

### Frozen data verified

| Data | Rows | Observed bytes |
|---|---:|---:|
| MP-20 train CSV | 27,136 | 83,154,915 |
| MP-20 validation CSV | 9,047 | 27,607,644 |
| MP-20 test CSV | 9,046 | 27,588,701 |
| Planner train JSONL | 27,136 | 153,756,946 |
| Planner validation JSONL | 9,047 | 51,223,744 |
| Planner test JSONL | 9,046 | 51,234,563 |
| Exact-length DLM train JSONL | 27,136 | 111,422,704 |
| Exact-length DLM validation JSONL | 9,047 | 37,082,760 |
| Exact-length DLM test JSONL | 9,046 | 37,074,802 |

The row counts were read from the frozen data-builder statistics. A direct
full-file line scan was intentionally avoided after the public NFS proved
slow; no scientific count was inferred from file size.

### Plan and seed-ledger assets verified

- The historical H1-A2 Planner output contains 1,200 raw attempts and 1,186
  parsed Plans. Its recorded sampler is world size 2, batch size 4,
  temperature `0.9`, top-p `0.95`, top-k `50`, and base seed `17`.
- The H1-A2 raw and parsed Plan files are present with sizes `4,435,456` and
  `3,638,335` bytes.
- The R03 source Plan run contains 512 raw and parsed outputs with seed
  `17029`, world size 1, and batch size 4. The source files are `1,942,201`
  and `1,564,564` bytes.
- The canonical R03 attempt ledger is present, has exactly 256 rows, and is
  `830,351` bytes. It contains the frozen per-ordinal body/refiner seed
  records used by all four process replays.
- Standalone public `raw_256`, `parsed_256`, and seed-ledger files still need
  to be extracted/copied from these verified sources.

### Seed evidence after the audit

- Planner epoch-2 training seed `17` is explicitly recorded in the A800
  recovery configuration.
- B0 records `data_seed=20260515`; neither its run configuration, training
  log, nor full log records a global Python/Torch training seed. That seed
  remains **unrecorded** and must not be replaced by `20260515`.
- The source tree adjacent to `model_494` defines `SEED=1234` and passes it to
  `torch.manual_seed`. The training timestep sampler uses unseeded
  `numpy.random.choice`, and no checkpoint-specific override/config was found
  beside the checkpoint. Therefore `1234` remains an upstream default rather
  than proof of full checkpoint-level determinism.
- The early H1-A2 body and refiner inference commands expose no seed argument;
  their initial global RNG states remain unrecorded. The later R03 ordinal
  ledger must not be used to rewrite this history.

### Runtime sources verified

- The H1-A2 exact replay/fresh-official source contains cohort collection,
  paired body generation, official-MP query, and finalization modules, plus
  the frozen best-code archive.
- The original safe-axis 256 body source and four-repeat refine/evaluate
  source trees are both present.
- The official-MP query runtime and the quick-reproduction implementation can
  therefore be rebuilt without inventing code, but their historical absolute
  paths and source-identity gates must be replaced by the public relative-path
  interface.

## Remaining local release work

- Export and transfer the two confirmed Python environments.
- Transfer the base models, adapters, refiner, and CHGNet checkpoint.
- Transfer the frozen MP-20 CSV/JSONL datasets.
- Transfer the H1-A2 Plan fallback and extract the R03 first-256 Plan bundle
  plus ordinal seed ledger.
- Decide and document a new release seed for any from-scratch B0 training;
  the historical global seed cannot currently be recovered.
- Document `model_494`'s partial seed coverage and avoid a bitwise training
  reproducibility claim.
- Rebuild the Direct/N/U/CHGNet/S.U.N. adapter around relative paths and the
  confirmed package versions.
- Wire and validate the public quick body/refiner/evaluator route against the
  transferred 256-row ledger on A800.

The placeholder-aware launchers stop with readable messages for these items;
they do not silently invent assets, seeds, or package versions.
