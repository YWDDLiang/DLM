# Programmed-path regression analysis: native

256 complete paired requests; method minus reference. This is a descriptive diagnostic, not a replacement evaluation or a causal attribution.

| Outcome | Reference | Method | Lost | Gained | Neither | McNemar descriptive p |
|---|---:|---:|---:|---:|---:|---:|
| reconstructed | 255 | 252 | 3 | 0 | 1 | 0.25 |
| native_execution_success | 255 | 252 | 3 | 0 | 1 | 0.25 |
| endpoint_execution_success | 255 | 252 | 3 | 0 | 1 | 0.25 |
| novel | 254 | 251 | 3 | 0 | 2 | 0.25 |
| unique_representative | 254 | 251 | 3 | 0 | 2 | 0.25 |
| novel_unique | 253 | 250 | 3 | 0 | 3 | 0.25 |
| terminal_verified | 63 | 76 | 29 | 42 | 151 | 0.153913 |
| strict_stable | 7 | 12 | 1 | 6 | 243 | 0.125 |
| meta_stable | 56 | 67 | 18 | 29 | 171 | 0.143865 |
| strict_sun | 6 | 11 | 1 | 6 | 244 | 0.125 |
| meta_sun | 55 | 66 | 18 | 29 | 172 | 0.143865 |
| verified_strict_sun | 1 | 4 | 0 | 3 | 252 | 0.25 |
| verified_meta_sun | 26 | 30 | 17 | 21 | 209 | 0.627103 |

## finite_physical_intersections

| Metric | n / excluded | Mean delta | Median | p10 | p90 | 10% each-tail trimmed | Without abs top 1 / 3 / 5 |
|---|---:|---:|---:|---:|---:|---:|---|
| A_gap_eV_atom | 252 / 4 | 0.207334 | 0.0287582 | -4.74701 | 4.48731 | 0.162448 | -0.242936 / 0.101283 / 0.181814 |
| eR_eV_atom_delta_equals_B_delta | 252 / 4 | -0.409361 | -0.000463843 | -2.99796 | 2.06711 | -0.192753 | -0.306849 / -0.389639 / -0.393701 |
| raw_energy_eV_atom | 252 / 4 | -0.202026 | 0 | -4.73055 | 3.75517 | -0.126018 | -0.549785 / -0.213484 / -0.134847 |
| raw_force_max_eV_A | 252 / 4 | -33.2173 | -6.50796e-07 | -85.2526 | 62.6821 | -5.37126 | 3.06734 / -10.2186 / -7.95842 |
| raw_stress_max_GPa | 252 / 4 | -68.7704 | 6.49868 | -86.0548 | 124.098 | 12.6133 | 65.9625 / 12.3181 / 12.1871 |
| terminal_force_max_eV_A | 252 / 4 | 0.0240206 | -0.00379321 | -4.56125 | 2.36441 | -0.248996 | -1.42787 / -0.4629 / -0.517619 |
| terminal_stress_max_GPa | 252 / 4 | 17.8626 | 0.000298382 | -14.4667 | 14.3465 | 0.0842447 | -6.37715 / -0.23364 / -0.232113 |
| relaxation_steps | 252 / 4 | -29.504 | 0 | -377.5 | 50.9 | -26.203 | -31.4661 / -31.7189 / -31.9636 |

## common_verified_intersections

| Metric | n / excluded | Mean delta | Median | p10 | p90 | 10% each-tail trimmed | Without abs top 1 / 3 / 5 |
|---|---:|---:|---:|---:|---:|---:|---|
| A_gap_eV_atom | 34 / 222 | 0.165684 | 3.57628e-07 | -5.27862 | 4.61602 | 0.125294 | -0.11644 / -0.161733 / 0.294759 |
| eR_eV_atom_delta_equals_B_delta | 34 / 222 | -0.0186265 | -3.57628e-07 | -0.243879 | 0.134348 | -0.0073244 | -0.00102346 / -0.00242166 / 0.0182111 |
| raw_energy_eV_atom | 34 / 222 | 0.147057 | 0 | -5.47629 | 4.18643 | 0.110383 | -0.135822 / -0.171534 / 0.277532 |
| raw_force_max_eV_A | 34 / 222 | -3.01904 | 2.83497e-06 | -40.0369 | 21.1607 | -2.15568 | -1.00664 / -1.09972 / 1.79016 |
| raw_stress_max_GPa | 34 / 222 | 9.44962 | 3.8147e-06 | -77.9077 | 114.845 | 7.06463 | -1.00207 / 12.5749 / 2.79445 |
| terminal_force_max_eV_A | 34 / 222 | -0.00626473 | -0.00516566 | -0.0272823 | 0.0171132 | -0.00602299 | -0.00388898 / -0.00737292 / -0.00500266 |
| terminal_stress_max_GPa | 34 / 222 | 0.0462029 | 0.0457805 | -0.0789018 | 0.204314 | 0.0470468 | 0.0404356 / 0.0282682 / 0.0166917 |
| relaxation_steps | 34 / 222 | -4.94118 | 0 | -24.5 | 19.2 | -1.5 | 0.363636 / -3.22581 / -0.413793 |

Trimming and absolute-tail deletion are sensitivity descriptions. Full-request counts and original finite-pair means remain unchanged.

## Same-pair energy decomposition

- finite_complete_energy_pairs: n=252; mean ΔA=0.207334, mean Δraw=-0.202026, mean ΔeR=ΔB=-0.409361 eV/atom; maximum identity residual=0.
- common_verified_complete_energy_pairs: n=34; mean ΔA=0.165684, mean Δraw=0.147057, mean ΔeR=ΔB=-0.0186265 eV/atom; maximum identity residual=0.

Absolute terminal-energy means are eR; only the same-composition differences equal ΔB.

## Execution, verification and missingness

- native_execution_success: lost 3, gained 0; method statuses on lost requests: {"generation_failure": 3}.
  - sample 0 / group eval:0 / O8K8Zn4: generation_failure; recorded details {"parser_error": "ValueError: FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|><|mdm_mask|><|mdm_mask|><|md'", "path_artifact_error": "FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|><|mdm_mask|><|mdm_mask|><|md'", "path_trace_failure": "construct_empty_support"}.
  - sample 69 / group eval:69 / H4C12Re4: generation_failure; recorded details {"parser_error": "ValueError: FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|><|mdm_mask|><|mdm_mask|><|md'", "path_artifact_error": "FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|><|mdm_mask|><|mdm_mask|><|md'", "path_trace_failure": "construct_empty_support"}.
  - sample 136 / group eval:136 / Cu3In3Pr3: generation_failure; recorded details {"parser_error": "ValueError: FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|>'", "path_artifact_error": "FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|>'", "path_trace_failure": "construct_empty_support"}.
- endpoint_execution_success: lost 3, gained 0; method statuses on lost requests: {"generation_failure": 3}.
  - sample 0 / group eval:0 / O8K8Zn4: generation_failure; recorded details {"parser_error": "ValueError: FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|><|mdm_mask|><|mdm_mask|><|md'", "path_artifact_error": "FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|><|mdm_mask|><|mdm_mask|><|md'", "path_trace_failure": "construct_empty_support"}.
  - sample 69 / group eval:69 / H4C12Re4: generation_failure; recorded details {"parser_error": "ValueError: FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|><|mdm_mask|><|mdm_mask|><|md'", "path_artifact_error": "FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|><|mdm_mask|><|mdm_mask|><|md'", "path_trace_failure": "construct_empty_support"}.
  - sample 136 / group eval:136 / Cu3In3Pr3: generation_failure; recorded details {"parser_error": "ValueError: FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|>'", "path_artifact_error": "FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|>'", "path_trace_failure": "construct_empty_support"}.
- reconstructed: lost 3, gained 0; method statuses on lost requests: {"generation_failure": 3}.
  - sample 0 / group eval:0 / O8K8Zn4: generation_failure; recorded details {"parser_error": "ValueError: FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|><|mdm_mask|><|mdm_mask|><|md'", "path_artifact_error": "FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|><|mdm_mask|><|mdm_mask|><|md'", "path_trace_failure": "construct_empty_support"}.
  - sample 69 / group eval:69 / H4C12Re4: generation_failure; recorded details {"parser_error": "ValueError: FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|><|mdm_mask|><|mdm_mask|><|md'", "path_artifact_error": "FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|><|mdm_mask|><|mdm_mask|><|md'", "path_trace_failure": "construct_empty_support"}.
  - sample 136 / group eval:136 / Cu3In3Pr3: generation_failure; recorded details {"parser_error": "ValueError: FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|>'", "path_artifact_error": "FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|>'", "path_trace_failure": "construct_empty_support"}.
- terminal_verified: lost 29, gained 42; method statuses on lost requests: {"not_converged": 19, "invalid_terminal": 9, "generation_failure": 1}.
  - sample 10 / group eval:10 / Cu2In2Ce2: not_converged; recorded details {}.
  - sample 15 / group eval:15 / Si6Cr2Y: invalid_terminal; recorded details {}.
  - sample 21 / group eval:21 / YPd3: not_converged; recorded details {}.
  - sample 29 / group eval:29 / O8Rb4Mo2: invalid_terminal; recorded details {}.
  - sample 33 / group eval:33 / NaAu2Tl: not_converged; recorded details {}.
  - sample 39 / group eval:39 / Si2AsBaTl: not_converged; recorded details {}.
  - sample 47 / group eval:47 / F6K2YIn: not_converged; recorded details {}.
  - sample 63 / group eval:63 / Ge4Ag6Sb8: invalid_terminal; recorded details {}.
  - sample 69 / group eval:69 / H4C12Re4: generation_failure; recorded details {"parser_error": "ValueError: FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|><|mdm_mask|><|mdm_mask|><|md'", "path_artifact_error": "FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|><|mdm_mask|><|mdm_mask|><|md'", "path_trace_failure": "construct_empty_support"}.
  - sample 77 / group eval:77 / Ag2Ce: not_converged; recorded details {}.
  - sample 80 / group eval:80 / AgCe2Pt: not_converged; recorded details {}.
  - sample 82 / group eval:82 / MgSr3: not_converged; recorded details {}.
  - sample 91 / group eval:91 / O8La2Pb2: not_converged; recorded details {}.
  - sample 93 / group eval:93 / O2ZnSr: not_converged; recorded details {}.
  - sample 101 / group eval:101 / LiF6K2Ag: invalid_terminal; recorded details {}.
  - sample 107 / group eval:107 / Cl6K2AsRb: not_converged; recorded details {}.
  - sample 133 / group eval:133 / Co4Ge4Pd4: not_converged; recorded details {}.
  - sample 140 / group eval:140 / O10Cu2Sr4Cs2: invalid_terminal; recorded details {}.
  - sample 148 / group eval:148 / Li2SnEr: not_converged; recorded details {}.
  - sample 155 / group eval:155 / ScPtAu2: not_converged; recorded details {}.
  - sample 168 / group eval:168 / AlGaSm2: not_converged; recorded details {}.
  - sample 179 / group eval:179 / Mg4Nd2: invalid_terminal; recorded details {}.
  - sample 181 / group eval:181 / O8K4Nd4: invalid_terminal; recorded details {}.
  - sample 183 / group eval:183 / Pr6Hg2: invalid_terminal; recorded details {}.
  - sample 194 / group eval:194 / Si2Ti4As2: not_converged; recorded details {}.
  - sample 203 / group eval:203 / Mg4Pb2: not_converged; recorded details {}.
  - sample 204 / group eval:204 / Pd3Gd: not_converged; recorded details {}.
  - sample 210 / group eval:210 / O6As2Ag2: invalid_terminal; recorded details {}.
  - sample 230 / group eval:230 / Li6AlAs2In: not_converged; recorded details {}.

Only stored terminal statuses, force/stress values, parser/artifact errors and path failures are used. If optimizer-stop or representation-consistency details were omitted from attempt results, this script cannot reconstruct them.

Hull status transitions: [{"reference": "input_not_reconstructed", "method": "input_not_reconstructed", "requests": 1}, {"reference": "known", "method": "input_not_reconstructed", "requests": 3}, {"reference": "known", "method": "known", "requests": 244}, {"reference": "official_cache_unresolved", "method": "official_cache_unresolved", "requests": 8}].

## A/B tail geometry

The cases below are selected by paired A/eR tails in the two stated numerical intersections. They are not a new benchmark subset.

| Sample / group / composition | ΔA | ΔeR=ΔB | VPA ref → method | Distinct-pair distance ref → method | Self-image ref → method | Lattice condition ref → method |
|---|---:|---:|---|---|---|---|
| 11 / eval:11 / O2P2As2 | 0.6934 | 0.17272 | 7.47328 → 4.66515 | 1.32448 → 1.06743 | 3.1 → 3.1 | 6.63077 → 8.73395 |
| 15 / eval:15 / Si6Cr2Y | -3.57719 | 5.60086 | 11.6779 → 23.9254 | 0.888 → 0.888 | 4 → 4 | 2.58498 → 2.58839 |
| 16 / eval:16 / Sb6Yb8 | 6.70229 | -8.29233 | 23.827 → 23.827 | 0.795911 → 1.02 | 7.4 → 7.4 | 2.10484 → 2.10484 |
| 22 / eval:22 / LiF2Te | -9.82441 | 0.0492659 | 15.3501 → 15.3501 | 0.975411 → 1.95082 | 3.90164 → 3.90164 | 7.27882 → 7.27882 |
| 39 / eval:39 / Si2AsBaTl | -6.50693 | 9.70633 | 13.2203 → 11.5932 | 1.08944 → 1.08944 | 4.2 → 3.82318 | 3.74603 → 3.53998 |
| 41 / eval:41 / In3Pr3Tl3 | 4.79786 | -7.53929 | 28.5067 → 10.9648 | 0.553 → 0.516 | 5 → 4.3 | 2.00247 → 1.73792 |
| 49 / eval:49 / C2K2Lu2 | -6.15566 | 0.137548 | 24.2667 → 24.2667 | 0.91 → 2 | 4 → 4 | 2.275 → 2.275 |
| 56 / eval:56 / O6Cl2Ba4Nd2 | 5.0398 | -0.0821166 | 17.4909 → 17.4909 | 1.72121 → 0.582 | 5.3 → 5.3 | 2.54252 → 2.54252 |
| 58 / eval:58 / Sn5U | 8.83253 | -0.282259 | 32.5799 → 17.1473 | 2.75 → 1.54388 | 5.5 → 4 | 1.81972 → 1.73263 |
| 60 / eval:60 / N2I2 | -7.74175 | 0.237998 | 29.0552 → 29.0552 | 1.525 → 2.2 | 4.4 → 4.4 | 1.85409 → 1.85409 |
| 65 / eval:65 / Mg5Er | -5.25988 | 0.278227 | 19.6241 → 19.6241 | 1.22328 → 2.4 | 4.8 → 4.8 | 2.89729 → 2.89729 |
| 72 / eval:72 / F6MgSn | 113.225 | -26.1398 | 20.6852 → 1.70348 | 0.726 → 0.53 | 4.03393 → 1.86725 | 3.77435 → 16.8351 |
| 75 / eval:75 / Cl3RbAu | -0.178632 | 0.442919 | 44.352 → 34.496 | 2.75 → 2.75 | 5.5 → 5.5 | 1.30909 → 1.01818 |
| 84 / eval:84 / F6KSbCs2 | 4.18172 | -0.0330567 | 22.8084 → 13.1149 | 1.0646 → 1.08398 | 6.3 → 4.6 | 2.20782 → 2.37407 |
| 85 / eval:85 / YPtBi | -7.4061 | 0.0582423 | 21.9123 → 22.8447 | 1.59197 → 2.15 | 4.3 → 4.3 | 2.06327 → 2.09175 |
| 89 / eval:89 / Li2O8F2P2In4 | -75.4385 | 2.91478 | 1.63807 → 13.9662 | 0.552798 → 0.725662 | 2.40406 → 6 | 15.4953 → 1.85851 |
| 105 / eval:105 / Sb6Dy2 | 0.663756 | -0.320935 | 46.6731 → 16.9378 | 1.425 → 1.425 | 5.7 → 4.5 | 2.51819 → 1.8872 |
| 112 / eval:112 / CeTl3 | 4.80214 | -0.599526 | 34.83 → 21.4785 | 2.15 → 1.85 | 4.3 → 3.7 | 1.39535 → 1.45946 |
| 138 / eval:138 / F2I2Ba2 | -5.28666 | -0.401621 | 16.5333 → 28.896 | 1.55 → 2.4 | 4 → 4.2 | 1.55 → 2.28571 |
| 143 / eval:143 / Na4Au4 | -1.33584 | 6.3407 | 15.092 → 7.84 | 0.882 → 1 | 3.2 → 3.2 | 2.40625 → 1.53125 |
| 155 / eval:155 / ScPtAu2 | -9.86419 | -0.0451818 | 15.0274 → 15.0274 | 1.52233 → 2.57536 | 4.2 → 4.2 | 2.06035 → 2.06035 |
| 163 / eval:163 / Cu3In3Yb3 | 2.55652 | -0.154325 | 26.6736 → 26.6736 | 1.81041 → 1.44061 | 5 → 5 | 2.27469 → 2.27469 |
| 166 / eval:166 / S4Sc2W | -10.758 | 10.2948 | 13.3382 → 13.3382 | 1.122 → 1.122 | 3.3 → 3.3 | 4.24264 → 4.24264 |
| 195 / eval:195 / Co2Sn2Ce2 | 2.70608 | -7.40115 | 35.1691 → 35.1691 | 0.869276 → 1.28559 | 5.7 → 5.7 | 1.81818 → 1.81818 |
| 212 / eval:212 / O8Nb4 | 7.54581 | -5.12132 | 14.5417 → 10.9693 | 0.535836 → 0.513164 | 3.58832 → 4.6 | 3.91335 → 1.17464 |
| 213 / eval:213 / Ca4As2Sb2 | -2.09414 | 0.126881 | 13.5889 → 14.5627 | 1.4558 → 1.61068 | 4.39356 → 4.77441 | 6.20176 → 3.33205 |
| 217 / eval:217 / Ge2Ru2Dy | -4.74911 | 8.51625 | 16.4424 → 16.4424 | 0.642818 → 0.865203 | 4.15274 → 4.15274 | 2.44953 → 2.44953 |
| 236 / eval:236 / Na2Br6Th | -7.66133 | -0.0535321 | 42.6806 → 42.6806 | 1.37961 → 2.075 | 7.7 → 7.7 | 2.07153 → 2.07153 |
| 245 / eval:245 / In6Yb2 | 9.47579 | 0.00630522 | 46.4117 → 12.7358 | 2.12421 → 1.425 | 6.7 → 4.3 | 1.73313 → 1.78498 |
| 250 / eval:250 / Ba6Pt2 | 7.72705 | -7.76894 | 43.8425 → 16.4328 | 1.22496 → 1.5 | 6 → 4.6 | 1.78888 → 1.78667 |

Distances use the stored full row-vector lattice and a centered 125-image shell; this does not certify exact MIC for an arbitrary unreduced basis. For tau800, input geometry means the stored refined structure; upstream native token geometry is separate in the JSON.

The JSON also contains top-five increases/decreases for every physical metric, signed tail contributions, coverage counts and per-case existing evidence.
