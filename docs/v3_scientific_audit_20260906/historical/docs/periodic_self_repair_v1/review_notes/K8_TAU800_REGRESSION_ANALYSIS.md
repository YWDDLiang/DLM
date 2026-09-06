# Programmed-path regression analysis: tau800

256 complete paired requests; method minus reference. This is a descriptive diagnostic, not a replacement evaluation or a causal attribution.

| Outcome | Reference | Method | Lost | Gained | Neither | McNemar descriptive p |
|---|---:|---:|---:|---:|---:|---:|
| reconstructed | 255 | 252 | 3 | 0 | 1 | 0.25 |
| native_execution_success | 255 | 252 | 3 | 0 | 1 | 0.25 |
| endpoint_execution_success | 255 | 252 | 3 | 0 | 1 | 0.25 |
| novel | 221 | 214 | 8 | 1 | 34 | 0.0390625 |
| unique_representative | 253 | 250 | 3 | 0 | 3 | 0.25 |
| novel_unique | 221 | 214 | 8 | 1 | 34 | 0.0390625 |
| terminal_verified | 99 | 82 | 40 | 23 | 134 | 0.0429565 |
| strict_stable | 26 | 24 | 5 | 3 | 227 | 0.726562 |
| meta_stable | 152 | 157 | 12 | 17 | 87 | 0.458258 |
| strict_sun | 19 | 17 | 4 | 2 | 235 | 0.6875 |
| meta_sun | 121 | 123 | 15 | 17 | 118 | 0.86005 |
| verified_strict_sun | 10 | 8 | 4 | 2 | 244 | 0.6875 |
| verified_meta_sun | 49 | 47 | 15 | 13 | 194 | 0.850554 |

## finite_physical_intersections

| Metric | n / excluded | Mean delta | Median | p10 | p90 | 10% each-tail trimmed | Without abs top 1 / 3 / 5 |
|---|---:|---:|---:|---:|---:|---:|---|
| A_gap_eV_atom | 252 / 4 | -0.0126426 | -1.19209e-06 | -0.0534981 | 0.0321343 | -0.00247615 | -0.00941248 / -0.00531688 / -0.00516159 |
| eR_eV_atom_delta_equals_B_delta | 252 / 4 | -0.00990393 | -2.58684e-05 | -0.0648036 | 0.0352258 | -0.00646837 | -0.00527071 / -0.00691931 / -0.00715212 |
| raw_energy_eV_atom | 252 / 4 | -0.0225465 | -0.000114143 | -0.115816 | 0.0473261 | -0.0110537 | -0.0154212 / -0.014386 / -0.0123439 |
| raw_force_max_eV_A | 252 / 4 | -0.155673 | 3.10152e-05 | -0.661515 | 0.767622 | 0.02549 | 0.0643663 / -0.0143018 / -0.0132924 |
| raw_stress_max_GPa | 252 / 4 | -0.056699 | -0.000907898 | -2.28145 | 2.30146 | -0.063147 | 0.00486045 / 0.0084616 / -0.094299 |
| terminal_force_max_eV_A | 252 / 4 | 0.00288521 | 0.000113709 | -0.023115 | 0.031787 | 0.00269675 | 0.00251147 / 0.0018896 / 0.0019418 |
| terminal_stress_max_GPa | 252 / 4 | 0.019287 | 0.000100906 | -0.229992 | 0.313312 | 0.0208918 | 0.0157482 / 0.0224043 / 0.0168998 |
| relaxation_steps | 252 / 4 | -2.89683 | 0 | -19 | 14.9 | -0.564356 | -1.40239 / -0.807229 / -0.769231 |

## common_verified_intersections

| Metric | n / excluded | Mean delta | Median | p10 | p90 | 10% each-tail trimmed | Without abs top 1 / 3 / 5 |
|---|---:|---:|---:|---:|---:|---:|---|
| A_gap_eV_atom | 59 / 197 | 0.00293147 | 0 | -0.0132248 | 0.028465 | 0.00145464 | 0.00157205 / 0.0015267 / -0.000145343 |
| eR_eV_atom_delta_equals_B_delta | 59 / 197 | -0.00648938 | -1.90735e-06 | -0.038994 | 0.0101483 | -0.00377533 | -0.00358434 / -0.00426553 / -0.0015301 |
| raw_energy_eV_atom | 59 / 197 | -0.00355791 | -6.07967e-05 | -0.0395721 | 0.00830908 | -0.00440958 | -0.00102617 / -0.00442623 / -0.00473173 |
| raw_force_max_eV_A | 59 / 197 | 0.0833115 | 0.000735737 | -0.0659859 | 0.537693 | 0.0679149 | 0.147283 / 0.0979931 / 0.0631574 |
| raw_stress_max_GPa | 59 / 197 | 0.117893 | -0.00808011 | -0.332983 | 1.13618 | 0.0720605 | 0.0689813 / 0.0724363 / 0.0721501 |
| terminal_force_max_eV_A | 59 / 197 | 0.00156801 | -6.81398e-06 | -0.0155312 | 0.0242031 | 0.000847947 | 0.000258626 / 0.000201185 / 0.00157756 |
| terminal_stress_max_GPa | 59 / 197 | 0.00982055 | -3.26593e-05 | -0.0968713 | 0.152704 | 0.0105391 | 0.0148761 / 0.0155522 / 0.00714682 |
| relaxation_steps | 59 / 197 | 2.27119 | 0 | -5 | 17.2 | 2.10204 | 2.84483 / 2.08929 / 1.37037 |

Trimming and absolute-tail deletion are sensitivity descriptions. Full-request counts and original finite-pair means remain unchanged.

## Same-pair energy decomposition

- finite_complete_energy_pairs: n=252; mean ΔA=-0.0126426, mean Δraw=-0.0225465, mean ΔeR=ΔB=-0.00990393 eV/atom; maximum identity residual=0.
- common_verified_complete_energy_pairs: n=59; mean ΔA=0.00293147, mean Δraw=-0.00355791, mean ΔeR=ΔB=-0.00648938 eV/atom; maximum identity residual=0.

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
- terminal_verified: lost 40, gained 23; method statuses on lost requests: {"not_converged": 39, "generation_failure": 1}.
  - sample 5 / group eval:5 / O8Cr4Ag4: not_converged; recorded details {}.
  - sample 9 / group eval:9 / O8KBa3Bi3: not_converged; recorded details {}.
  - sample 15 / group eval:15 / Si6Cr2Y: not_converged; recorded details {}.
  - sample 17 / group eval:17 / P4Cu4Sr4: not_converged; recorded details {}.
  - sample 19 / group eval:19 / O3F9Mn6: not_converged; recorded details {}.
  - sample 32 / group eval:32 / O8NaV2Fe: not_converged; recorded details {}.
  - sample 34 / group eval:34 / MgGa3: not_converged; recorded details {}.
  - sample 36 / group eval:36 / O8AlKMo2: not_converged; recorded details {}.
  - sample 45 / group eval:45 / S4Nb2Ag2: not_converged; recorded details {}.
  - sample 53 / group eval:53 / Ge12Hf4: not_converged; recorded details {}.
  - sample 55 / group eval:55 / Ni4Sn4Ce4: not_converged; recorded details {}.
  - sample 56 / group eval:56 / O6Cl2Ba4Nd2: not_converged; recorded details {}.
  - sample 59 / group eval:59 / Ni4Sn2Au12: not_converged; recorded details {}.
  - sample 63 / group eval:63 / Ge4Ag6Sb8: not_converged; recorded details {}.
  - sample 65 / group eval:65 / Mg5Er: not_converged; recorded details {}.
  - sample 70 / group eval:70 / ZnGe: not_converged; recorded details {}.
  - sample 79 / group eval:79 / O12Ge4Ba4: not_converged; recorded details {}.
  - sample 83 / group eval:83 / Co7Ge2Sm2: not_converged; recorded details {}.
  - sample 104 / group eval:104 / Cu2Zr12Sn4: not_converged; recorded details {}.
  - sample 108 / group eval:108 / ZnSn2Pb: not_converged; recorded details {}.
  - sample 119 / group eval:119 / Li2F8Mn2Te2: not_converged; recorded details {}.
  - sample 131 / group eval:131 / K4As12Eu4: not_converged; recorded details {}.
  - sample 136 / group eval:136 / Cu3In3Pr3: generation_failure; recorded details {"parser_error": "ValueError: FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|>'", "path_artifact_error": "FixedSlotError: Strict dynamic answer contains non-schema text '<|mdm_mask|>'", "path_trace_failure": "construct_empty_support"}.
  - sample 143 / group eval:143 / Na4Au4: not_converged; recorded details {}.
  - sample 145 / group eval:145 / F6K2AgTl: not_converged; recorded details {}.
  - sample 149 / group eval:149 / O5F8Cu2Sr2Os: not_converged; recorded details {}.
  - sample 155 / group eval:155 / ScPtAu2: not_converged; recorded details {}.
  - sample 158 / group eval:158 / N2Br2: not_converged; recorded details {}.
  - sample 164 / group eval:164 / K2Ag6Sn2: not_converged; recorded details {}.
  - sample 166 / group eval:166 / S4Sc2W: not_converged; recorded details {}.
  - sample 186 / group eval:186 / Co2Ge2Tb2: not_converged; recorded details {}.
  - sample 207 / group eval:207 / O7NaK3Nb2: not_converged; recorded details {}.
  - sample 220 / group eval:220 / NaSe2Sm: not_converged; recorded details {}.
  - sample 222 / group eval:222 / O8Na4V4: not_converged; recorded details {}.
  - sample 230 / group eval:230 / Li6AlAs2In: not_converged; recorded details {}.
  - sample 238 / group eval:238 / Mg3In3Nd3: not_converged; recorded details {}.
  - sample 246 / group eval:246 / Co6Ge6Tm: not_converged; recorded details {}.
  - sample 249 / group eval:249 / Li2C10N4: not_converged; recorded details {}.
  - sample 251 / group eval:251 / C2Si6La8: not_converged; recorded details {}.
  - sample 255 / group eval:255 / Pd2Sb6Nd2: not_converged; recorded details {}.

Only stored terminal statuses, force/stress values, parser/artifact errors and path failures are used. If optimizer-stop or representation-consistency details were omitted from attempt results, this script cannot reconstruct them.

Hull status transitions: [{"reference": "input_not_reconstructed", "method": "input_not_reconstructed", "requests": 1}, {"reference": "known", "method": "input_not_reconstructed", "requests": 3}, {"reference": "known", "method": "known", "requests": 244}, {"reference": "official_cache_unresolved", "method": "official_cache_unresolved", "requests": 8}].

## A/B tail geometry

The cases below are selected by paired A/eR tails in the two stated numerical intersections. They are not a new benchmark subset.

| Sample / group / composition | ΔA | ΔeR=ΔB | VPA ref → method | Distinct-pair distance ref → method | Self-image ref → method | Lattice condition ref → method |
|---|---:|---:|---|---|---|---|
| 2 / eval:2 / O4MgZnSr | 0.0493202 | -0.171721 | 12.5288 → 13.8747 | 2.17587 → 2.0153 | 3.61013 → 3.98545 | 1.65565 → 1.73105 |
| 4 / eval:4 / P4Se8Ag4 | -0.0317421 | 0.0144086 | 26.8452 → 27.9338 | 2.20743 → 2.15976 | 5.76762 → 5.80905 | 2.20218 → 2.24176 |
| 11 / eval:11 / O2P2As2 | -0.329905 | 0.143818 | 19.2856 → 21.6169 | 1.69417 → 1.65281 | 4.48161 → 4.7398 | 1.29566 → 1.36095 |
| 16 / eval:16 / Sb6Yb8 | -0.0266252 | 0.137763 | 34.0263 → 35.6301 | 3.36253 → 3.12039 | 7.75545 → 6.95985 | 1.01731 → 1.72724 |
| 17 / eval:17 / P4Cu4Sr4 | -0.00771999 | -0.251202 | 23.0666 → 22.4526 | 2.08658 → 2.41489 | 4.26805 → 4.20491 | 5.65412 → 5.77772 |
| 23 / eval:23 / H8Li8C4 | 0.191174 | -0.0179801 | 9.09406 → 9.02198 | 1.11804 → 0.96204 | 3.47501 → 4.00784 | 3.05025 → 4.4698 |
| 27 / eval:27 / Li4Si8Cs4 | -0.0271969 | -0.0407248 | 28.6647 → 28.3558 | 2.23848 → 2.24386 | 4.19321 → 4.72232 | 3.12787 → 3.19136 |
| 39 / eval:39 / Si2AsBaTl | 0.0592327 | -0.106785 | 28.3465 → 26.3421 | 2.42158 → 2.62853 | 4.27693 → 4.57947 | 2.09733 → 1.92082 |
| 41 / eval:41 / In3Pr3Tl3 | 0.0817776 | -0.004565 | 31.057 → 31.4427 | 3.00053 → 2.7724 | 4.82608 → 5.03741 | 2.45085 → 1.94541 |
| 50 / eval:50 / N2Mg2S | -0.400467 | 0.65116 | 14.5561 → 14.4585 | 2.07184 → 1.8728 | 3.7467 → 3.60195 | 1.38229 → 2.00938 |
| 54 / eval:54 / Zn2Sn8Tm2 | 0.00579548 | -0.0490177 | 26.2207 → 24.6987 | 2.76776 → 3.04722 | 4.41211 → 4.4312 | 3.62671 → 3.36856 |
| 60 / eval:60 / N2I2 | -0.638162 | -1.17284 | 31.3461 → 38.804 | 3.12617 → 1.30937 | 4.23263 → 4.60693 | 2.64342 → 2.54225 |
| 62 / eval:62 / S8Zn2In4 | 0.0354991 | 0.0400672 | 26.1079 → 26.2246 | 2.35416 → 2.2501 | 5.72261 → 4.11473 | 1.9466 → 3.23888 |
| 78 / eval:78 / C8Si6Ba4 | 0.280914 | 0.0352392 | 18.424 → 19.1261 | 1.7061 → 1.51334 | 3.9131 → 5.64084 | 3.45908 → 1.50535 |
| 79 / eval:79 / O12Ge4Ba4 | -0.00418854 | -0.161455 | 13.9671 → 16.2649 | 2.04285 → 1.69019 | 5.81767 → 4.5854 | 1.41417 → 2.69263 |
| 81 / eval:81 / Ge2Pr10Pb6 | 0.0473552 | -0.0860147 | 31.1257 → 32.6505 | 3.22922 → 3.02372 | 7.32382 → 7.38687 | 1.72284 → 1.72006 |
| 94 / eval:94 / NaSc2As6Tl | -0.0371184 | -0.00966263 | 22.6373 → 24.0685 | 2.48784 → 2.43872 | 4.28005 → 4.0886 | 2.23046 → 2.25799 |
| 105 / eval:105 / Sb6Dy2 | 0.130848 | -0.13174 | 28.2536 → 27.913 | 3.26671 → 3.29466 | 4.60533 → 4.64086 | 2.2876 → 2.20825 |
| 122 / eval:122 / S4K2Mn2 | 0.0435567 | -0.00517845 | 26.8113 → 25.0956 | 2.41905 → 2.56854 | 3.99674 → 3.98104 | 3.32158 → 2.10856 |
| 133 / eval:133 / Co4Ge4Pd4 | 0.128278 | -0.0746531 | 15.8642 → 15.1426 | 2.3675 → 2.47837 | 4.03411 → 3.45266 | 4.64696 → 4.36165 |
| 135 / eval:135 / As8Cd2Tl4 | 0.0459888 | -0.0385613 | 29.7736 → 29.5862 | 2.552 → 2.56969 | 4.18603 → 4.08448 | 3.27335 → 3.505 |
| 140 / eval:140 / O10Cu2Sr4Cs2 | 0.115264 | 0.071527 | 17.663 → 17.9492 | 1.73209 → 1.70684 | 4.01619 → 4.60265 | 3.08213 → 3.19019 |
| 147 / eval:147 / TiCoTm2 | -0.00701523 | 0.159356 | 20.838 → 20.3413 | 2.8876 → 2.9645 | 3.34892 → 4.84493 | 2.21469 → 1.99445 |
| 158 / eval:158 / N2Br2 | -0.272916 | -0.00133896 | 30.7213 → 30.6356 | 1.31034 → 1.20937 | 4.31317 → 4.30943 | 2.46096 → 2.46545 |
| 165 / eval:165 / Sr4In2Th2 | 0.0267065 | -0.0702293 | 38.373 → 38.7659 | 3.66771 → 3.58854 | 5.60943 → 5.6087 | 1.73249 → 1.75215 |
| 178 / eval:178 / Pr12Pt4 | 0.00865173 | 0.0513825 | 31.5891 → 32.7943 | 2.93034 → 2.85821 | 6.69194 → 6.62813 | 1.51504 → 1.63039 |
| 185 / eval:185 / NiGa3Lu | 0.0245829 | -0.174982 | 18.8028 → 18.1296 | 2.24886 → 2.19231 | 4.49874 → 4.1216 | 1.01988 → 1.29423 |
| 197 / eval:197 / O4Pd2Ba2 | 0.00750399 | 0.160602 | 17.4997 → 17.191 | 2.05937 → 2.06273 | 3.91747 → 4.09244 | 2.2137 → 2.00186 |
| 214 / eval:214 / NaSb4Tm | -0.0535493 | 0.0455251 | 30.0597 → 30.2112 | 3.11575 → 3.11629 | 4.41722 → 4.41349 | 2.09215 → 2.10598 |
| 249 / eval:249 / Li2C10N4 | -0.823394 | 0.218649 | 13.2178 → 15.9153 | 0.914182 → 1.21576 | 5.27658 → 4.88417 | 2.2864 → 1.94128 |

Distances use the stored full row-vector lattice and a centered 125-image shell; this does not certify exact MIC for an arbitrary unreduced basis. For tau800, input geometry means the stored refined structure; upstream native token geometry is separate in the JSON.

The JSON also contains top-five increases/decreases for every physical metric, signed tail contributions, coverage counts and per-case existing evidence.
