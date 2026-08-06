# H1-A2-Aligned PlanGraph-DLM Charter V2

Status: `locally_implemented_not_submitted`

Redesign date: 2026-08-01

Amendment: `D2-shuffle` was removed before submission. D1 is the registered
planned-order control; no shuffle arm is part of V2.

The failed G1 v1 result remains immutable scientific evidence. It is not
repaired, merged, or reused. No v1 PG checkpoint is eligible for a later stage.
H1-A2 epoch 2 remains the frozen submission fallback.

## 1. Corrected research question

With the successful H1-A2 seven-line Planner, R5-C representation,
model-visible prompt, tokenizer, refiner, and evaluation protocol held fixed,
does aligning the DLM corruption process with a deterministically compiled
chemical dependency graph improve crystal-body generation and final S.U.N.?

V2 does **not** ask an LLM to emit PlanGraph JSON. PlanGraph is a deterministic
training/inference sidecar compiled from the existing H1/R5-C `plan_state`.
It is consumed only by the corruption/denoising scheduler. The same dependency
order is used for a candidate's training masks and body-generation schedule.

## 2. H1-A2 identity lock

### Planner

- family: H1-A2 rich seven-line Planner;
- base: `/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/`;
- frozen adapter SHA-256:
  `65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a`;
- exact output contract:

```text
formula: ...
anion: ...
charge: ...
lattice: ...
spacegroup: ...
volume: ...
end: plan
```

- official recovered sampler only:
  `baseline/scripts/sample_llama_h1_formula_plans.py`;
- temperature `0.9`, top-p `0.95`, top-k `50`, maximum new tokens `96`;
- sampling enabled; no sample ID, retry, repair, filter, replacement, or rerank;
- stop/truncate only after the generated `end: plan` boundary;
- no PlanGraph JSON, auxiliary explanation, or extra Planner field.

The frozen first-256 P0 ledger from the completed H1 exploratory run is the
primary paired-generation plan ledger. A newly sampled Planner baseline may
be used only for runtime parity and must not silently replace it.

### Body DLM

- base: frozen LLaDA-8B-Instruct identity;
- representation: `dynamic_v1`;
- R5-C source prompt and answer bytes are immutable;
- answer length remains `7 + 4*N`, maximum `87`;
- PlanGraph is attached as non-visible JSONL metadata;
- the model never sees serialized PlanGraph text;
- D0 and every planned arm receive identical ordered rows, prompts, answers,
  tokenizer tokens, and sample weights.

### Refiner and evaluation

- frozen R5-C sampler constraints;
- frozen CrysLLMGen `model_494` refiner;
- exact 800 reverse steps, one evaluation per input;
- same ordinal Plan ledger, R5-C noise, refiner noise, and evaluation order;
- raw all-attempt denominator is primary;
- no S.U.N., energy, hull, CHGNet, MLIP, or Materials Project signal enters
  training, checkpoint selection, retry, filtering, or reranking.

## 3. Deterministic sidecar compiler

For every original R5-C JSONL row:

1. preserve `prompt`, `answer`, and the reconstructed training `text` exactly;
2. validate the original dynamic answer against its existing `plan_state`;
3. compile `composition`, `symmetry_lattice`, and element-multiplicity site
   groups without reading source metadata;
4. strip material IDs, metadata, energies, stability labels, and sample IDs;
5. write the graph only under `plangraph`;
6. register a plan-condition hash derived only from the model-visible H1/R5-C
   prompt and graph, never from the target answer;
7. fail the entire atomic publication if any source row fails;
8. reject train/validation/test training-pair overlap.

The required builder is
`scripts/build_h1a2_sidecar_plangraph_data.py`. The old
`plangraph_v1_dynamic_body` prompt and its long JSON are prohibited in V2.

## 4. DLM arms

All scientific arms train from the same raw LLaDA base with the original R5-C
one-epoch setup. This makes D0 a baseline reproduction rather than an
unmatched continuation.

| Arm | Training corruption | Inference schedule | Purpose |
|---|---|---|---|
| H0 | frozen historical R5-C | frozen H1 exact-plan | immutable H1-A2 body reference |
| D0 | iid only | frozen H1 exact-plan | exact current-code R5-C reproduction |
| D1 | iid:current-order = 2:1 | frozen H1 exact-plan | train/inference alignment control |
| D2 | iid:compiled-PlanGraph = 2:1 | compiled PlanGraph | proposed matched method |

D1 is the sufficient control for the primary claim: it receives the same
planned-corruption mixture and budget as D2 but uses the existing H1-A2/R5-C
dependency order. This isolates the value of the compiled PlanGraph without
adding a synthetic shuffle distribution.

## 5. Exact training standard

The scientific full-epoch runs use:

- frozen `mp_20_r5_exact_length` train/validation/test identities;
- source row counts `27,136 / 9,047 / 9,046`;
- two A800 processes, batch size `1` per process, gradient accumulation `8`;
- global effective batch `16`, one epoch, expected `1,696` optimizer updates;
- BF16, gradient checkpointing, gradient clipping `1.0`;
- LoRA rank `8`, alpha `32`, dropout `0.05`;
- target modules `q_proj,k_proj,v_proj,ff_proj,up_proj`;
- modules to save `model.transformer.wte,model.transformer.ff_out`;
- AdamW, learning rate `5e-5`, weight decay `0`;
- cosine schedule, warmup `100`, minimum LR ratio `0.2`;
- data seed `20260515`;
- maximum sequence length `382`, not the obsolete PlanGraph-JSON length `768`;
- fixed validation panel every `500` updates, maximum `50` batches;
- D0 retains the legacy iid training RNG path.

A bounded 32-row engineering smoke may precede the full runs, but its outputs
are never scientific results or initialization.

## 6. Gates

### G0 — sidecar and baseline parity

- all 45,229 source rows publish without removal or mutation;
- every output prompt/answer has the registered model-visible SHA;
- every schedule key is reproducible from inference-available plan content and
  is invariant to the target answer;
- no metadata/energy/ID leakage;
- D0 completes one epoch with finite loss and gradients;
- D0 fixed-panel NLL and conditional-256 R5-C body metrics reproduce the
  historical scale before D1/D2 results are interpreted.

If D0 does not reproduce R5-C, stop and diagnose the runtime. Do not tune D2
against a broken baseline.

### G1 — likelihood and body screen

- final fixed-panel NLL no worse than `+1%` relative to D0;
- conditional body completion no worse than D0 by more than `1` point;
- composition locking and formula/body agreement remain exact;
- D2's registered matched dependency margin is positive;
- D2 strictly beats D1 on the same registered dependency margin;
- no checkpoint is selected with generation S.U.N. or energy labels.

### G2 — paired 256 end-to-end

Use the frozen H1 P0 first-256 plan ledger for H0, D0, D1, and D2. Every
failure remains in the denominator.

Continue only if:

- generation completion is at least `97%`;
- structure validity is no more than `2` points below H0;
- D2 beats both D0 and D1 on at least one registered direct metric;
- strict S.U.N. improves by at least `1` point or exceeds `10%`;
- meta S.U.N. improves by at least `3` points or exceeds `50%`;
- the likelihood gate still holds.

Composition validity is expected to be plan-limited and is reported with
Planner-caused and DLM-caused failure decomposition. It is not claimed as a
Planner improvement in V2.

## 7. Stop rules

- Do not train or sample a new JSON Planner.
- Do not reuse the failed G1 v1 PG/PG-shuffle checkpoints.
- Do not change the H1 prompt, seven-line schema, temperature, or tokenizer.
- Do not alter the R5-C model-visible body prompt.
- Do not derive any training/inference schedule from the target answer.
- Do not adapt thresholds after observing D2 generation or S.U.N.
- Do not enter a refiner modification or G4 automatically.
- Any new job requires a new run root, source manifest, authorization record,
  and explicit submission decision.

## 8. Paper claim if successful

The main contribution becomes:

> H1-A2 plans are compiled into a non-visible dependency graph that aligns
> discrete diffusion corruption and denoising while leaving the proven
> Planner language and crystal representation unchanged.

This isolates a real DLM mechanism and avoids attributing long-JSON
serialization failure to chemical planning.
