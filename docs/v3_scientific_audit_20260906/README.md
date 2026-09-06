# 全链路科学审计与 V3 研究

2026-09-06。**分组件与一般交叉审计、V2完整评测及冻结模型诊断已完成；具体V3正在第二轮独立审查，尚未训练。**

本目录覆盖完整C³FD/Llama/soft/P→DLM→repair/refiner→物理/N-U/hull/统计、近期变更、历史成功失败、外部晶体生成机制及文档清理。先读 [整合报告](FINAL_AUDIT_REPORT.md) 和 [当前状态](../CURRENT_STATE.md)。

## 报告与裁决

| 报告 | 内容 |
|---|---|
| [整合报告](FINAL_AUDIT_REPORT.md) | SUN、原因排序、证据边界与接续 |
| [实验/变更台账](EXPERIMENT_CHANGE_CASEBOOK.md) | 739个commit清单、实验族与证据等级 |
| [主张审计](CLAIM_AUDIT.md) | 既往收益承诺、代理指标与归因 |
| [晶体diffusion机制](CRYSTAL_DIFFUSION_MECHANISMS.md) | 原论文/作者代码和任务边界 |
| [Transformer/GNN](TRANSFORMER_AND_GNN_SUPPLEMENT.md) | 额外工作、已有消息路径与增量候选 |
| [C³FD/Typed Llama](C3FD_LLM_UPSTREAM_AUDIT.md) | 实际支持、PoE、M/S/P与程序 |
| [DLM概率/kernel](DLM_PROBABILITY_AND_KERNEL_AUDIT.md) | 条件观测、目标、alias、支持/rollback |
| [物理与评测](PHYSICS_PIPELINE_AND_STORY_AUDIT.md) | 几何、R/refiner、force/stress及SUN事件 |
| [一般交叉审稿：DLM](ROUND2_DLM_ON_CRYSTAL_AND_CLAIMS.md) | 已完成，不作为具体V3第二轮 |
| [一般交叉审稿：物理](ROUND2_PHYSICS_ON_DLM_AND_CLAIMS.md) | 已完成，不作为具体V3第二轮 |
| [V2冻结probe解释](V2_FROZEN_PROBE_INTERPRETATION.md) | 40004已完成，实际风险/响应/exp，不是SUN提点 |

## 具体V3与攻击性复核

[H-P33规格](V3_H_P33_SPECIFICATION.md)为当前唯一第二轮对象：共享V2/LLaDA、T construction＋G联合去噪、新增两epoch、明确33NFE及float/Q终点。原CIF身份及实际计算图/DDP验收仍须完成；它是混合表示，P在G中仅条件，尚无SUN保证。

- [A/B/C候选及攻击](V3_DLM_CANDIDATES_AND_ATTACKS.md)
- [共享几何头独立反提案](V3_GEOMETRIC_HEAD_COUNTERPROPOSAL.md)
- [V3第一轮数学](V3_REVIEW_R1_MATH.md)
- [V3第一轮物理](V3_REVIEW_R1_PHYSICS.md)

第二轮独立数学/物理报告正在写，完成后增列裁决。已承认的边界不重复当新发现，新实质问题修订后复核，不无限扩张路线。

## 直接证据

- [历史评测与hash](evidence/EVALUATION_SUMMARIES_20260906.json)：含旧K8官方hull覆盖修正。
- [V2最终训练/M签名](evidence/V2_FINAL_TRAIN_AND_METADATA.json)：39993完成，完整27136/9047。
- [V2两端](evidence/V2_EVALUATION_SUMMARIES_39998.json)：39998 raw8/50、tau16/113，各256。
- [完整构造/修订配对](evidence/V2_CONSTRUCTION_AND_REPAIR_39998.json)：construction12/44、repair8/50。
- [冻结probe40004](evidence/V2_CONDITIONING_PROBE_40004.json)：100来源/200状态/五变体，模型不变。
- [全量条件](evidence/FROZEN_CONDITIONING_AUDIT.json)、[N/U函数](evidence/FROZEN_NU_FUNCTIONS.json)、[终态/SUN交叉表](evidence/TERMINAL_STATUS_SUN_CROSSTAB.json)。
- [连续CIF输入](evidence/CONTINUOUS_SOURCE_INPUT_PATHS.json)、[实际LLaDA core](evidence/LLADA_ACTUAL_CORE_FORWARD.json)。
- [早期V2快照](EVIDENCE_RUNTIME_SNAPSHOT.json)仅表示当时状态，终态以上述产物为准。

## 文档整理

142份历史/组件文件迁移、6份完全重复去重已执行，唯一证据及hash保留；新稿/入口继续增量检查。
[方案](DOCUMENT_CLEANUP_PLAN.md)、[迁移报告](DOCUMENT_MIGRATION_REPORT.md)、[映射](DOCUMENT_MIGRATION_MAP.json)、[父引用](PARENT_REFERENCE_MIGRATION.json)、[当前链接检查](CURRENT_DOCUMENT_LINK_CHECK.json)、[旧根入口字节快照](historical/root_entrypoints/MIGRATION_MANIFEST.json)。

历史current/approved只表示写作当时；当前说明以本目录和CURRENT_STATE为准。原2e运行归档、权重、数据和其他工作树不随整理改变。

## 执行边界

用户已授权审查后1–2epoch V3及有依据的小方法，当前先完成具体复核/必要验收；使用4–6A800，项目总限6A800/24CPU/2jobs。旧新K4/K8、自生成能量teacher和旧连续联合路线保持暂停。

全请求SUN决定采用。开发256同一端Strict≥26、Meta≥128才触发固定源序1000 raw/refined，并另报1200全部请求。失败不补样，不按能量/SUN/verified筛索引，official cache匹配cohort。数学自洽、梯度、代理指标、来源完整分别只支持对应层，不是SUN保证。
