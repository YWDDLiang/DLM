# Experiment and evidence matrix

## Main results

### Main reported result

| Denominator | Strict S.U.N. | Meta S.U.N. |
|---:|---:|---:|
| 1,000 | **105/1000 = 10.50%** | **488/1000 = 48.80%** |

The experiments below provide component, mechanism and scale evidence around
this headline; they remain separate profiles rather than replacement rows.

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

### Independent Plan1200 tau800 scale validation

| Diffusion | Denominator | N/U/NU | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|---:|---:|
| tau800 | 1000 | 872/987/866 | **81/1000 = 8.10%** | **486/1000 = 48.60%** |

The fused Planner is composition-valid on `1200/1200`. Among the 1,159 Plans
with official references, the DLM produces 1,139 valid CIFs. The scale main1000
contains the first 861 valid rows from the original main block plus all 139
valid remainder rows. The 20 CIF construction failures remain disclosed.

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

### Terminal diffusion mechanism

| Comparison | Direct | Strict S.U.N. | Meta S.U.N. | Interpretation |
|---|---:|---:|---:|---|
| L6 raw | 188/512 | 10/512 | 66/512 | learned DLM endpoint |
| L6 model494 tau800 | 457/512 | 48/512 | 230/512 | fixed fine-scale conversion |
| Plan1200 tau800 main1000 | — | 81/1000 | 486/1000 | independent scale profile |

This separation prevents model494 gains from being attributed to the DLM and
prevents raw G2 gains from being hidden by the refiner.

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

## Evidence sources

- Planner: `results/remote_screens/C3FD_V25_REQUESTED1000_FINAL.json` and
  `docs/C3FD_LLAMA_DLM_SUN_CHECKLIST_V6.json`.
- Plan-conditioned DLM: `docs/C3FD_NATIVE_TEACHER_SFT_DATA_FINAL_20260831.json`
  and `docs/C3FD_NATIVE_TEACHER_SFT_FINAL_20260831.json`.
- Prospective system: `docs/36H_FINAL_REPORT_C3FD_G2_20260901.json`, remote
  run profile `prospective_headline`.
- Full epoch: `docs/G2_FULL_EPOCH_AB_FINAL_20260901.json`, profile
  `full_epoch_mechanism`.
- Plan1200 scale: `docs/PLAN1200_TAU800_FINAL_20260902.md`, profile
  `plan1200_scale_validation`.
