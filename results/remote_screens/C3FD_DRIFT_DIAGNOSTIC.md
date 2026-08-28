# C³FD-v2 drift diagnostic

| Group | Rows | Benchmark | All-metal | Unary | Unique formulas | Duplicate excess | Pair-score mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_full | 27136 | 90.50% | 34.91% | 0.83% | 24830 | 2306 | 1.0284596415317935 |
| train_benchmark | 24558 | 100.00% | 38.57% | 0.92% | 22367 | 2191 | 1.0277359958277987 |
| p0 | 1989 | 86.68% | 28.91% | 0.25% | 1928 | 61 | 0.9001818295925244 |
| c3fd_v2 | 1972 | 100.00% | 36.71% | 0.00% | 1890 | 82 | 0.7703095211990538 |

Distances to training: `{'p0': {'train_full': {'N': 0.10173111977313906, 'arity': 0.031881921173767064, 'family': 0.09475577127621732, 'all_metal': 0.059966608801237}, 'train_benchmark': {'N': 0.11268708903120596, 'arity': 0.04375660316937391, 'family': 0.08161993333232612, 'all_metal': 0.09660916619712842}}, 'c3fd_v2': {'train_full': {'N': 0.0865724207298404, 'arity': 0.1818210698601171, 'family': 0.15820820795380608, 'all_metal': 0.01808335565846378}, 'train_benchmark': {'N': 0.07030295626679696, 'arity': 0.170527337113266, 'family': 0.14364305753304632, 'all_metal': 0.018559201737427644}}}`
N correlations: `{'c3_N_vs_total_legal': -0.04394065488077825, 'c3_N_vs_zero_legal': -0.04394065488077825, 'c3_N_vs_benchmark_train_N': 0.9894223323154552}`
Top C3 formulas: `[('O4F4V4', 8), ('O4F4Mn4', 5), ('O16Os4', 5), ('Li12N4', 5), ('S6Cr2', 4), ('O2F10Mn6', 4), ('Mg4Cd2', 3), ('Se4Mo4Te4', 3), ('Li2O4Fe2', 3), ('O4F8Mn6', 3), ('O10F2V6', 3), ('K6CuSe2Te', 2), ('LiAg2In', 2), ('Cu2Sn2Er2', 2), ('S12Zn4Sn4', 2), ('Li6NAs', 2), ('Al2Ge2Eu', 2), ('O4F12Mn2Os2', 2), ('Mg2Ag4', 2), ('Li2F8V2', 2)]`
Top P0 formulas: `[('O8F4Mn6', 3), ('O2F10Cu6', 3), ('F6V2', 3), ('Li2F8V2', 3), ('O2F10V4', 3), ('O2F10Co6', 3), ('Li4O8Mn3Co', 3), ('Li4O8Mn3Cu', 3), ('YRh2Ho', 2), ('Li4O8V4', 2), ('O8Na4Bi6', 2), ('Li2O8Mn3Cu', 2), ('S8K4Sn4', 2), ('Si2Fe2La', 2), ('Mg2Si2Ba', 2), ('H4O8P2K2', 2), ('Cl6Rb2SbCs', 2), ('O4Cu3In', 2), ('Mg2Cd4', 2), ('F6SbCs2', 2)]`
