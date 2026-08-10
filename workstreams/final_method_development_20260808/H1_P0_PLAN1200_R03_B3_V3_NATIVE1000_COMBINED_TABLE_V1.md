# P0 Plan1200 × R03/B3：V3 与 native1000 合并表 V1

Updated: 2026-08-11 (Asia/Shanghai)

## 口径标识

| `protocol_type` | 中文名称 | 输入/选择规则 | 主分母 | stage |
|---|---|---|---|---|
| `V3_ALL_ATTEMPT_1000` | V3 固定 1000 全尝试口径 | 每批 raw1200 中按 planner ordinal 冻结前 1000 个 parse-success plan；R03/B3 同批共享 | 全部 1000 次 body/DLM generation attempts，失败仍占分母 | `pre_model494`、`post_model494` |
| `CRYSLLMGEN_NATIVE_SUCCESS1000` | CrysLLMGen-native 成功累计口径 | 按冻结 planner 顺序单次遍历候选，累计前 1000 个 body/`process_one` success，再将这 1000 个全部 refine | 1000 个成功 body 输出；post-refine 必须仍为完整 1000 | `post_model494` |

两种类型必须放在同一张主表中，但不能合并分母或直接把 count
相减。跨 arm 的正式配对推断只对相同候选集合成立；native1000 若
R03/B3 的成功选择集合不同，只作描述性比较。

## 合并主表：当前全部可用数据

`N/A-EF` 表示该字段不是 0，而是因 generation 前工程失败没有观测。
V3/native V1 的冻结 cohort 缺少 body runtime 要求的顶层 `parsed` 字段，
因此实际 terminal crystal-generation attempts 为 0。

| protocol_type | Arm | Repeat | Stage | Planner raw | Parse-success / candidate pool | Target denominator | Actual terminal generation attempts | Selected body successes | Refined structures | Reconstructed | Comp valid | Struct valid | Joint valid | COV-P | COV-R | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V3_ALL_ATTEMPT_1000 | R03 | 0 | pre_model494 | 1200 | 1189 | 1000 attempts | 0 | N/A-EF | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | body schema failure at ordinal 0 |
| V3_ALL_ATTEMPT_1000 | R03 | 0 | post_model494 | 1200 | 1189 | 1000 attempts | 0 | N/A-EF | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine not started |
| V3_ALL_ATTEMPT_1000 | R03 | 1 | pre_model494 | 1200 | 1193 | 1000 attempts | 0 | N/A-EF | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | body schema failure at ordinal 0 |
| V3_ALL_ATTEMPT_1000 | R03 | 1 | post_model494 | 1200 | 1193 | 1000 attempts | 0 | N/A-EF | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine not started |
| V3_ALL_ATTEMPT_1000 | R03 | 2 | pre_model494 | 1200 | 1194 | 1000 attempts | 0 | N/A-EF | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | body schema failure at ordinal 0 |
| V3_ALL_ATTEMPT_1000 | R03 | 2 | post_model494 | 1200 | 1194 | 1000 attempts | 0 | N/A-EF | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine not started |
| V3_ALL_ATTEMPT_1000 | B3 | 0 | pre_model494 | 1200 | 1189 | 1000 attempts | 0 | N/A-EF | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | body schema failure at ordinal 0 |
| V3_ALL_ATTEMPT_1000 | B3 | 0 | post_model494 | 1200 | 1189 | 1000 attempts | 0 | N/A-EF | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine not started |
| V3_ALL_ATTEMPT_1000 | B3 | 1 | pre_model494 | 1200 | 1193 | 1000 attempts | 0 | N/A-EF | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | body schema failure at ordinal 0 |
| V3_ALL_ATTEMPT_1000 | B3 | 1 | post_model494 | 1200 | 1193 | 1000 attempts | 0 | N/A-EF | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine not started |
| V3_ALL_ATTEMPT_1000 | B3 | 2 | pre_model494 | 1200 | 1194 | 1000 attempts | 0 | N/A-EF | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | body schema failure at ordinal 0 |
| V3_ALL_ATTEMPT_1000 | B3 | 2 | post_model494 | 1200 | 1194 | 1000 attempts | 0 | N/A-EF | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine not started |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | R03 | 0 | post_model494 | 1200 | 1189 | 1000 body successes | 0 | 0 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native reserve generation not entered |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | R03 | 1 | post_model494 | 1200 | 1193 | 1000 body successes | 0 | 0 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native reserve generation not entered |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | R03 | 2 | post_model494 | 1200 | 1194 | 1000 body successes | 0 | 0 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native reserve generation not entered |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | B3 | 0 | post_model494 | 1200 | 1189 | 1000 body successes | 0 | 0 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native reserve generation not entered |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | B3 | 1 | post_model494 | 1200 | 1193 | 1000 body successes | 0 | 0 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native reserve generation not entered |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | B3 | 2 | post_model494 | 1200 | 1194 | 1000 body successes | 0 | 0 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native reserve generation not entered |

## Strict S.U.N. 独立表

Headline 必须使用复现脚本的
`reconstructed_structures_exact_legacy` 分母；同时报告固定全尝试/全 refine
分母的 secondary，二者不能混写。

| protocol_type | Arm | Repeat | Stage | Target fixed denominator | Actual evaluated attempts | Reconstructed denominator | Strict numerator | Strict / reconstructed headline | Strict / fixed secondary | Hull evaluated | Hull unknown | Hull failure | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V3_ALL_ATTEMPT_1000 | R03 | 0 | pre_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | generation 前失败 |
| V3_ALL_ATTEMPT_1000 | R03 | 0 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine 未启动 |
| V3_ALL_ATTEMPT_1000 | R03 | 1 | pre_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | generation 前失败 |
| V3_ALL_ATTEMPT_1000 | R03 | 1 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine 未启动 |
| V3_ALL_ATTEMPT_1000 | R03 | 2 | pre_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | generation 前失败 |
| V3_ALL_ATTEMPT_1000 | R03 | 2 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine 未启动 |
| V3_ALL_ATTEMPT_1000 | B3 | 0 | pre_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | generation 前失败 |
| V3_ALL_ATTEMPT_1000 | B3 | 0 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine 未启动 |
| V3_ALL_ATTEMPT_1000 | B3 | 1 | pre_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | generation 前失败 |
| V3_ALL_ATTEMPT_1000 | B3 | 1 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine 未启动 |
| V3_ALL_ATTEMPT_1000 | B3 | 2 | pre_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | generation 前失败 |
| V3_ALL_ATTEMPT_1000 | B3 | 2 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine 未启动 |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | R03 | 0 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native 未进入 generation/refine |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | R03 | 1 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native 未进入 generation/refine |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | R03 | 2 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native 未进入 generation/refine |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | B3 | 0 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native 未进入 generation/refine |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | B3 | 1 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native 未进入 generation/refine |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | B3 | 2 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native 未进入 generation/refine |

## Meta S.U.N. 独立表

Meta S.U.N. 与 strict S.U.N. 使用相同的双分母展示规则，但必须作为
独立 endpoint，不与 strict 合并为一个“总 S.U.N.”数值。

| protocol_type | Arm | Repeat | Stage | Target fixed denominator | Actual evaluated attempts | Reconstructed denominator | Meta numerator | Meta / reconstructed headline | Meta / fixed secondary | Novel | Unique representative | Novel∩Unique | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V3_ALL_ATTEMPT_1000 | R03 | 0 | pre_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | generation 前失败 |
| V3_ALL_ATTEMPT_1000 | R03 | 0 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine 未启动 |
| V3_ALL_ATTEMPT_1000 | R03 | 1 | pre_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | generation 前失败 |
| V3_ALL_ATTEMPT_1000 | R03 | 1 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine 未启动 |
| V3_ALL_ATTEMPT_1000 | R03 | 2 | pre_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | generation 前失败 |
| V3_ALL_ATTEMPT_1000 | R03 | 2 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine 未启动 |
| V3_ALL_ATTEMPT_1000 | B3 | 0 | pre_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | generation 前失败 |
| V3_ALL_ATTEMPT_1000 | B3 | 0 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine 未启动 |
| V3_ALL_ATTEMPT_1000 | B3 | 1 | pre_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | generation 前失败 |
| V3_ALL_ATTEMPT_1000 | B3 | 1 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine 未启动 |
| V3_ALL_ATTEMPT_1000 | B3 | 2 | pre_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | generation 前失败 |
| V3_ALL_ATTEMPT_1000 | B3 | 2 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | refine 未启动 |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | R03 | 0 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native 未进入 generation/refine |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | R03 | 1 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native 未进入 generation/refine |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | R03 | 2 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native 未进入 generation/refine |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | B3 | 0 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native 未进入 generation/refine |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | B3 | 1 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native 未进入 generation/refine |
| CRYSLLMGEN_NATIVE_SUCCESS1000 | B3 | 2 | post_model494 | 1000 | 0 | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | N/A-EF | native 未进入 generation/refine |

## “所有数据”的强制附录

未来成功结果不能只保留上述 headline。合并 Markdown 必须逐
`protocol_type × arm × repeat × stage` 无删减展开以下四类对象，并保留
attempt ledger 的路径、bytes 与 SHA256：

| 附录 | 必须完整展开的对象 | 展示方式 |
|---|---|---|
| CrysLLMGen complete | `direct_native_report_complete`，包括 `metrics_unchanged_upstream` 的全部字段 | 扁平化 `field → value` 表，不删列 |
| S.U.N. exact legacy | `sun_exact_legacy_reconstructed_denominator` 的全部字段 | 单独扁平化表，明确 numerator 与 reconstructed denominator |
| S.U.N. native complete | `sun_native_summary_complete` 的全部字段 | 单独扁平化表，不以 headline 替代 |
| failure / hull / energy | generation/body failure classes、`sun_diagnostics`、hull evaluated/unknown/failure、`e_above_hull` 与 energy summaries | 单独诊断表 |

三重复的逐 repeat exact McNemar、50,000-draw hierarchical paired
bootstrap、条件 reconstructed-denominator bootstrap，以及 pooled3000
描述值也必须全部保留。Pooled3000 只能标作 descriptive，不能伪装成
3,000 个独立样本。

## 当前结论

当前表中的 metric 单元均为 `N/A-EF`，不是 0。只有 planner 数量、
冻结目标、cache 与 Slurm 工程证据可用。要填入真正的 V3/native1000
指标，仍需新的不可变修复运行；本表只冻结合并展示与分母规则，
不授权重投。
