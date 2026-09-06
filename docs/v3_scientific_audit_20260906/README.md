# 全链路科学审计与 V3 研究

2026-09-07（上海）更新。**H-P33 G正式评测40066结束：raw9/61、tau80013/116，均256，未达目标；CPU诊断40069输出链全通过但坐标场风险未实质优于零预测。** 当前有界测试同权重T构造，并补匹配V2 constructor→tau800；用户指定若V2/V3最终无效则最终使用K4或K8，重点分析两者稳定/失败结构并实际尝试有据的改动。上海07:00交付最新与最好SUN。[实际跨策略失败诊断](V3_FAILURE_CROSS_STRATEGY_PHYSICS.md)、[原始G最终报告](evidence_v3_failure_20260907/EVAL_FINAL_40066.json)。

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

[H-P33规格](V3_H_P33_SPECIFICATION.md)为当前唯一第二轮对象：共享V2/LLaDA、T construction＋G联合去噪、新增两epoch、明确33NFE及float/Q终点。原连续CIF身份已全量通过，实际计算图/DDP验收仍须完成；它是混合表示，P在G中仅条件，尚无SUN保证。

- [A/B/C候选及攻击](V3_DLM_CANDIDATES_AND_ATTACKS.md)
- [共享几何头独立反提案](V3_GEOMETRIC_HEAD_COUNTERPROPOSAL.md)
- [V3第一轮数学](V3_REVIEW_R1_MATH.md)
- [V3第一轮物理](V3_REVIEW_R1_PHYSICS.md)

- [V3第二轮数学](V3_REVIEW_R2_MATH.md)
- [V3第二轮物理](V3_REVIEW_R2_PHYSICS.md)
- [模型CPU实现验收](MIXED_GEOMETRY_MODEL_CPU_ACCEPTANCE.md)
- [独立来源/调度实现审查](IMPLEMENTATION_DATA_SCHEDULE_REVIEW.md)
- [V3故事契合边界](V3_STORY_FIT.md)：H-G与程序驱动离散生成的区别、采用所需证据。
- [同权重T部署可行性](V3_TOKEN_DEPLOYMENT_FEASIBILITY.md)：仅接口与对照分析，尚未注册新增采样。
- [科学问题与贡献框架](PAPER_PROBLEM_AND_CONTRIBUTION_FRAMEWORK.md)：统一what/why/how、两项互补候选贡献及结果分支的证据边界。
- [顶会de novo写法](TOPCONF_DE_NOVO_WRITING_REVIEW.md)、[顶会语言路线细读](TOPCONF_LANGUAGE_STORY_PATTERNS.md)：仅已核ICLR/ICML/NeurIPS主会，主任务为组成与结构联合生成。
- [当前de novo cohort记账](DE_NOVO_COHORT_ACCOUNTING.md)：自主生成条件的缓存与外部给定组成CSP的区别，256全请求来源证据。
- [V3不足时的稳定性候选](V3_FAILURE_FALLBACK_SHORTLIST.md)：有界准备，等待原G/Q结果后条件触发。

两份R2均允许进入有界实现验收。模型核心冻结`1111739`，后续`e1439cb`加入六卡微批与图回环检查；40045、40060和40058均已实际PASS，见[实施验收](MIXED_IMPLEMENTATION_ACCEPTANCE.md)。CPU数学/模型/数据41项和采样器11项通过；它们支持启动完整实证，不是SUN收益保证。

## 直接证据

- [历史评测与hash](evidence/EVALUATION_SUMMARIES_20260906.json)：含旧K8官方hull覆盖修正。
- [V2最终训练/M签名](evidence/V2_FINAL_TRAIN_AND_METADATA.json)：39993完成，完整27136/9047。
- [V2两端](evidence/V2_EVALUATION_SUMMARIES_39998.json)：39998 raw8/50、tau16/113，各256。
- [完整构造/修订配对](evidence/V2_CONSTRUCTION_AND_REPAIR_39998.json)：construction12/44、repair8/50。
- [冻结probe40004](evidence/V2_CONDITIONING_PROBE_40004.json)：100来源/200状态/五变体，模型不变。
- [全量条件](evidence/FROZEN_CONDITIONING_AUDIT.json)、[N/U函数](evidence/FROZEN_NU_FUNCTIONS.json)、[终态/SUN交叉表](evidence/TERMINAL_STATUS_SUN_CROSSTAB.json)。
- [连续CIF输入](evidence/CONTINUOUS_SOURCE_INPUT_PATHS.json)、[实际LLaDA core](evidence/LLADA_ACTUAL_CORE_FORWARD.json)。
- [连续CIF全量身份40009](evidence/CONTINUOUS_SOURCE_IDENTITY_40009.json)：27136/9047全通过，0错误/歧义/丢行；[核验工具说明](audit_continuous_source_identity.README.md)。
- [数值与normalizer40013](evidence/GEOMETRY_NUMERICS_40013.json)：19项CPU检查、完整train-only标准化。
- [实际V2评测种子](evidence/V2_ACTUAL_EVALUATION_SEEDS.json)：保留64位整数及logical batch配对语义。
- [真实4卡预检40045提交](evidence/MIXED_PREFLIGHT_40045_SUBMISSION.json)：工程检查，不能用于SUN或当作正式policy。
- [4卡实际结果](evidence/MIXED_PREFLIGHT_40045.json)、[6卡微批2实际结果](evidence/MIXED_BATCH_PREFLIGHT_40060.json)、[实际连续图回环40058](evidence/CONTINUOUS_GRAPHS_40058.json)。
- [早期V2快照](EVIDENCE_RUNTIME_SNAPSHOT.json)仅表示当时状态，终态以上述产物为准。

## 文档整理

142份历史/组件文件迁移、6份完全重复去重已执行，唯一证据及hash保留；新稿/入口继续增量检查。
[方案](DOCUMENT_CLEANUP_PLAN.md)、[迁移报告](DOCUMENT_MIGRATION_REPORT.md)、[映射](DOCUMENT_MIGRATION_MAP.json)、[父引用](PARENT_REFERENCE_MIGRATION.json)、[当前链接检查](CURRENT_DOCUMENT_LINK_CHECK.json)、[旧根入口字节快照](historical/root_entrypoints/MIGRATION_MANIFEST.json)。

历史current/approved只表示写作当时；当前说明以本目录和CURRENT_STATE为准。原2e运行归档、权重、数据和其他工作树不随整理改变。

## 执行边界

用户已授权有依据的小方法并要求6小时内推进实际SUN实验。当前V3正式训练/评测已完，T部署有界验证，K4/K8实际产物分析与稳定性修正为后续重点；项目总限6A800/24CPU/2jobs。旧自生成能量teacher和其他暂停路线不自动恢复。

全请求SUN决定采用。开发256同一端Strict≥26、Meta≥128才触发固定源序1000 raw/refined，并另报1200全部请求。失败不补样，不按能量/SUN/verified筛索引，official cache匹配cohort。数学自洽、梯度、代理指标、来源完整分别只支持对应层，不是SUN保证。
