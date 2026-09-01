# Experiment and evidence matrix

## Main results

### Composition planning

| Planner evidence | Requested | Composition-valid | Learned-Llama evidence |
|---|---:|---:|---|
| predecessor C3FD | 2000 | 1724 | — |
| C3FD-v2.5 | 2000 | 2000 | — |
| fused C3FD–Llama prospective | 256 | 256 | mean KL0.06819; 87.05% decisions nonzero-KL |
| fused C3FD–Llama scale run | 1200 | 1200 | one trajectory per request |

The fused rows demonstrate that the scientific support is preserved while
Llama materially changes the action distribution.

### Fresh prospective structure generation

| Stage | Arm | Direct | N/U/NU | Hull known | Strict S.U.N. | Meta S.U.N. |
|---|---|---:|---:|---:|---:|---:|
| raw | BASE | 118/256 | 250/251/250 | 241 | 5/256 | 41/256 |
| raw | G2 | 121/256 | 251/252/251 | 245 | 9/256 | 47/256 |
| refined | BASE | 251/256 | 219/251/219 | 246 | 19/256 | 111/256 |
| refined | G2 | 251/256 | 216/252/216 | 247 | **24/256** | **117/256** |

Paired refined official hull G2−BASE is `-16.43 meV/atom`, bootstrap 95% CI
`[-25.26,-8.62]` on 244 common known rows.

## Mechanism ablations

### Geometry learning path

| Development arm | Body | Raw Direct | Interpretation |
|---|---:|---:|---|
| no periodic geometry BASE | 248 | 106 | tokenwise denoising baseline |
| G1 geometry losses only | 245 | 115 | relational objectives help but body floor fails |
| G2 relation residual, 348 updates | 248 | 119 | internal relation path improves raw realization |
| full-epoch G2-PBC-R (A) | 255 | **128** | promoted strict-PBC implementation |
| G2-PBC-RU (B) | 254 | 130 | +2 Direct but no energy advantage; rejected |

The G0/G1/G2 rows share their frozen development Plan/noise. Full-epoch A/B
share a later frozen Plan/noise and use BASE `118` as their matched reference.
No difference across those cohorts is labeled causal.

### Full-epoch cached-official endpoint

| Stage | A Strict/Meta | B Strict/Meta | B−A energy conclusion |
|---|---:|---:|---|
| raw | 7/41 | 6/41 | +129.19 meV/atom, CI crosses zero |
| refined | 23/117 | 23/115 | +8.10 meV/atom, CI crosses zero |

A is selected by the registered implementation rule. The fresh step348 G2
remains the prospective headline; A/B is mechanism and implementation evidence.

## Required paper ablations

| Question | Comparison | Primary metric |
|---|---|---|
| Does scientific support solve composition feasibility? | predecessor C3FD vs C3FD-v2.5 | comp_valid |
| Is Llama active inside the constrained Planner? | C3FD vs fused action distributions | KL, selected-action rank, comp_valid |
| Does the Plan interface preserve exact chemistry? | serializer/round-trip audit | N/composition equality, validity flips |
| Do geometry losses alone suffice? | BASE vs G1 | body, raw Direct |
| Does relational feedback matter? | G1 vs G2 | raw Direct |
| Does strict PBC/species packing scale? | BASE vs A | raw Direct, body |
| Does uncertainty gating help? | A vs B | raw/refined energy, S.U.N. |
| Does the complete method improve stability? | prospective BASE vs G2 | official hull, Strict/Meta S.U.N. |

## Efficiency protocol for new experiments

Intermediate arms run body accounting and fast Direct validity only. CHGNet,
official hull and cached N/U are evaluated only after a frozen structure batch
exists; reference fingerprints are reused. This prevents expensive evaluation
from dominating method iteration.

## Immutable sources

- Planner: `results/remote_screens/C3FD_V25_REQUESTED1000_FINAL.json` and
  `docs/C3FD_LLAMA_DLM_SUN_CHECKLIST_V6.json`.
- Plan-conditioned DLM: `docs/C3FD_NATIVE_TEACHER_SFT_DATA_FINAL_20260831.json`
  and `docs/C3FD_NATIVE_TEACHER_SFT_FINAL_20260831.json`.
- Prospective system: `docs/36H_FINAL_REPORT_C3FD_G2_20260901.json`, remote
  final SHA `1b99aa33...b3070`.
- Full epoch: `docs/G2_FULL_EPOCH_AB_FINAL_20260901.json`, remote final SHA
  `b50dd8d2...19881`.

