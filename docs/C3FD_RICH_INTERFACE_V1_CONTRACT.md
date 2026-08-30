# C3FD-Rich interface v1 contract

Date: 2026-08-30  
Priority: first  
Status: legacy deployed-interface audit; SG semantics superseded by
`C3FD_RICH_FIELD_SEMANTICS_AUDIT_V2.md`

The V1 `lattice_spacegroup_compatible_100pct` gate only verified the old
sampler's deterministic one-to-one compiler. It did not validate fidelity to
the original metric-lattice and symmetry labels and must not be reused as a
physical compatibility claim. The corrected canary design is frozen in
`DLM_RICH_PLANNER_STABLE_DLM_CHECKLIST_36H_V2.md`.

The current C3FD-v2.5 checkpoint already contains trained rich soft heads. This
contract restores their existing outputs to the DLM rather than training a new
free-text Planner.

## Frozen rich fields

- `anion_framework`
- `charge_bucket`
- `lattice_system`
- `spacegroup_bucket`, deterministically compatible with lattice system
- `volume_per_atom_bin`

Composition, `N`, elements, counts, certificate, and exact formula remain hard
and byte-identical between minimal and rich arms. Rich fields are soft context;
they do not prefill any body token or constrain exact coordinates.

## Existing Planner evidence

- C3FD-v2.5 requested2000 composition validity: 100%.
- The training data and model already include all five rich soft heads with
  per-field loss weight 0.2.
- Sampling already emits the historical seven-line rich Plan.
- Space-group bucket is selected by the frozen lattice-system compatibility map,
  so internal lattice/SG bucket compatibility is deterministic.

## Matched canary

Freeze 32 unused, outcome-blind C3FD compositions after excluding D3PO
main/sealed identities. Run:

- M0: minimal prompt -> BASE;
- R0: predicted-rich prompt -> BASE;
- two common DLM/refiner streams;
- temperature 0.7, full-axis, model494 tau800;
- one attempt per Plan, no retries/replacement/rerank;
- raw/refined Direct and CHGNet only, no official query.

Primary estimands are paired raw Direct and raw CHGNet delta. Refined metrics are
secondary. Rich is promoted if body/Direct do not materially regress and at
least one continuous raw/refined endpoint has concordant direction across both
streams. All results are disclosed; no rich-field subset or checkpoint is
selected from downstream outcomes.

## Final attribution if promoted

- M0 minimal Planner + BASE DLM;
- R0 predicted-rich Planner + BASE DLM;
- R1 predicted-rich Planner + listwise-aligned DLM;
- two streams, exactly six cells.

`R0-M0` estimates the rich Planner interface effect; `R1-R0` estimates the DLM
stability-alignment effect; pre/post model494 estimates continuous refinement.
