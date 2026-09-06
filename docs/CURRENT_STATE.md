# 当前项目状态

更新日期：2026-09-07（上海）。此文件统一维护当前状态；历史材料中的旧状态不覆盖本页。

## 当前阶段

**V3 H-P33 G 的40064训练和40066评测均完整结束，raw为9/61、tau800为13/116，分母均256，未达目标。** 实际产物诊断40069已完成：全256输出/图/refiner输入链无错配；100来源六噪声band的坐标风险基本未优于零预测。T construction-only及匹配V2 constructor→tau800评测作业40071已启动；其最后两个冻结G轨迹核验也已PASS，没有观测到大幅往返或末读出抹除。用户要求若V2/V3最终无效就使用K4或K8，当前分析与后续修正以这两个版本的稳定/失败结构为重点；Sep7上海07:00交付最新及最好实测SUN。

统一目录：[全链路审计](v3_scientific_audit_20260906/README.md)。

## 已完成的正式对照

V2 训练作业 **39993** 已完成，Slurm `COMPLETED 0:0`，用时 01:11:28；两 epoch、4524 更新、108544 有效 states，source/view 覆盖 min=max=2。真实零增量 delta=0，前 64 更新各模块梯度非零。正式采样只使用这个完整终点。

固定 256 评测作业 **39998** 已完整成功，raw/refined SUN、对照与 construction→repair 诊断均完成。`V2_EVAL_JOB` 和提交记录已登记，不重复提交 249/250。两端未达到同端 26/128 的补评门槛，本版本不追加 1000。

执行代码固定为 `2e904c260bafb6750c9a5dbb0d920c4ff8c3a868`。输入为原始 LLaDA 和完整 MP20 27136/9047，冻结 C³FD/Llama/pointer；训练直接预测条件 24558 条，显式 teacher-soft fallback 2578 条。验证为 8147 预测与 900 fallback。这不是全量预测条件，也不代表预测软标签与目标几何严格一致。

正式外部 `LAUNCH_MANIFEST.json` 已部署在远端 V2 实验目录；执行 commit 内的同名早期文档不是启动依据。训练、manifest 与唯一提交 claim 保持不变，不重复提交 249。

## 已完成的最近 SUN

以下均为 Strict / Meta SUN；不同 cohort 或样本数之间不作直接配对结论。

| 方法 / cohort | raw | 固定 tau800 refined |
|---|---:|---:|
| 参考 / 开发 256 | 6 / 55 | 19 / 121 |
| 旧 K4 / 同开发 256 | 7 / 57 | 19 / 126 |
| 旧 K8 / 同开发 256 | 11 / 66 | 17 / 123 |
| raw-v1 / 同开发 256 | 3 / 45 | 16 / 113 |
| V2 / 同开发 256 | 8 / 50 | 16 / 113 |
| V3 H-P33 G / 同开发 256 | 9 / 61 | 13 / 116 |
| 旧 K8 / 独立条件 1000 | 39 / 244 | 65 / 484 |
| 旧 K8 / 对应全部 1200 请求 | 47 / 285 | 79 / 570 |

旧 K8 独立评测此前误用了开发小 cohort 的 hull 缓存。官方覆盖已补齐，CPU 作业 39990 重评成功，生成、弛豫、能量和选择索引均未改变。更正后 1200 请求中 hull 已知 1172、官方明确 unresolved 20、不可重建 8。旧覆盖不足的 0/9、1/17 不再作为完整结果。补统计不是模型提升。

原始评测文件与 hash：[评测证据](v3_scientific_audit_20260906/evidence/EVALUATION_SUMMARIES_20260906.json)。历史小幅差异、共同 verified 子集及外部论文分数须按实际口径解释，不能混用来声称提升。

V2 构造终点原生 SUN 为 12/44，一次 full-cell repair 后为 8/50；Strict −4、Meta +6，是这组配对请求的权衡，不能称修订普遍改善。V2 refined 与 raw-v1 同计数，未实现目标提点。[V2两端](v3_scientific_audit_20260906/evidence/V2_EVALUATION_SUMMARIES_39998.json)、[同次构造与修订](v3_scientific_audit_20260906/evidence/V2_CONSTRUCTION_AND_REPAIR_39998.json)。

冻结诊断 **40004** 已 `COMPLETED 0:0`，4A800/16CPU、00:02:24。100来源/200状态、5变体，0训练、新晶体与MLIP；原目标风险复算最大差5.25e−7。旧几何遮蔽主要损害repair风险，真实噪声相对unknown没有清晰平均收益，真实soft降低部分风险但不是可用oracle或SUN证据。真实prefix logits的float64 exp未出现溢出。[完整证据](v3_scientific_audit_20260906/evidence/V2_CONDITIONING_PROBE_40004.json)。

## V3 实施与验收

[H-P33 规格](v3_scientific_audit_20260906/V3_H_P33_SPECIFICATION.md) 复用 V2 final 与共享 LLaDA，保留 T construction、以 G 全几何 v/torus 去噪替代旧 repair CE；拟再训练两epoch。主采样为32个Euler区间加一次明确末端读出，float CIF为主终点，同样样本Q raw为精度诊断。P在G中仅作条件/rank，不新增rank噪声时钟；这是混合表示模型。[第二轮数学](v3_scientific_audit_20260906/V3_REVIEW_R2_MATH.md)与[第二轮物理](v3_scientific_audit_20260906/V3_REVIEW_R2_PHYSICS.md)均允许进入有界实现验收，没有SUN收益保证。

实施冻结提交`1111739bc5a590734b02a61f395faa59be605531`包含共享hidden-only/G头、原V2加载兼容、来源与T/G调度、正式trainer、float导出与真实preflight。模型/数学/数据共40项本机CPU测试通过，40045中Torch2.4再次40项通过；旧V2 loader另2项、export/regression新旧29项通过。[模型CPU验收](v3_scientific_audit_20260906/MIXED_GEOMETRY_MODEL_CPU_ACCEPTANCE.md)、[独立数据调度审查](v3_scientific_audit_20260906/IMPLEMENTATION_DATA_SCHEDULE_REVIEW.md)。

真实预检 **40045**（4卡micro1/acc6）与 **40060**（6卡micro2/acc2）均`COMPLETED 0:0`且全PASS，分别4分48秒、4分21秒。真实共享G梯度非零、T-only增量0、各rank权重一致、33NFE与末端解码有限、T/G保存恢复差0。4卡人口梯度相对误差2.45e−8；6卡整体0.476%、最大片组1.901%，在事前2% BF16门内，不称逐bit相同。[实现验收汇总](v3_scientific_audit_20260906/MIXED_IMPLEMENTATION_ACCEPTANCE.md)。

实际CIF→CrysLLMGen process_one→FP32 refiner输入回环 **40058** 已PASS，8CPU/2分30秒、无模型/MLIP调用。4个例子含N=1/10/20和非Q网格fixture，显式Niggli后Gram/按元素坐标双射检查通过；FP32坐标误差≤2.92e−8，单原子等体积量化反例被拒绝。[实际回环](v3_scientific_audit_20260906/evidence/CONTINUOUS_GRAPHS_40058.json)。

正式配置6A800/24CPU、micro2/acc2/global24，完整27136来源各1T+1G/epoch，新增2epoch共4524更新/108544真实状态。新训练保持fresh optimizer和事先种子。冻结采样器11项CPU测试通过；正式raw/refined及secondary Q raw评测已完成。

**40064**已`COMPLETED 0:0`，实际用时01:29:21；2新增epoch、4524更新、108544有效状态，所有source的T/G覆盖min=max=2，两轮验证均完成，最终step-4524取得采样资格。[完整训练终点](v3_scientific_audit_20260906/evidence/MIXED_TRAIN_FINAL_40064.json)。训练与评测代码均固定`0bf8e8870f96c54e86cbd080e64c527f55fb6943`，运行归档保持不变。[冻结训练清单](v3_scientific_audit_20260906/evidence/MIXED_TRAIN_LAUNCH_MANIFEST.json)、[训练提交](v3_scientific_audit_20260906/evidence/MIXED_TRAIN_SUBMISSION.json)。

评测 **40066** 已`COMPLETED 0:0`，实际00:23:50，完整256 primary float raw/refined与secondary Q raw。G raw为9/61、tau800为13/116，Q raw为6/58；各端对应verified SUN为6/28、6/47、2/26。无主端达到同端26/128，不触发G独立1000。[原始最终报告](v3_scientific_audit_20260906/evidence_v3_failure_20260907/EVAL_FINAL_40066.json)、[raw](v3_scientific_audit_20260906/evidence_v3_failure_20260907/NATIVE_EVALUATION_40066.json)、[tau800](v3_scientific_audit_20260906/evidence_v3_failure_20260907/TAU800_EVALUATION_40066.json)、[Q](v3_scientific_audit_20260906/evidence_v3_failure_20260907/QUANTIZED_EVALUATION_40066.json)。原权重、代码和全请求产物保持冻结。

CPU诊断 **40069** 已`COMPLETED 0:0`，8CPU、12秒、无模型/能量调用。实际256个输出到原生label输入、CIF、Niggli图与ProposalDataset FP32/refiner seed全链无错配；G末态相对坐标移动中位0.0606 Å，仍保留32条初始<0.5 Å碰撞并新增8条。100来源六band准确零预测对照显示低/中噪声u无实质改善；v在最高两band改善约25.5%/32.4%。[实际链](v3_scientific_audit_20260906/evidence_v3_failure_20260907/OUTPUT_CHAIN_40069.json)、[零场对照](v3_scientific_audit_20260906/evidence_v3_failure_20260907/ZERO_VALIDATION_BASELINE_40069.json)、[跨策略产物解释](v3_scientific_audit_20260906/V3_FAILURE_CROSS_STRATEGY_PHYSICS.md)。这支持坐标场欠学习，不单独证明原因或不可修复。

**40071**当前运行，6A800/24CPU、2小时上限，执行`b3efa789fab1a70529251c9c364d5e9ae027e193`，新训练0。先原4请求采样和全新进程同logical batch逐动作重放，通过后全256 T raw/tau800，另补原V2 constructor→tau800；13项CPU行为测试已PASS，不能替代真实权重验收。[唯一提交](v3_scientific_audit_20260906/evidence_v3_failure_20260907/TOKEN_EVAL_SUBMISSION.json)、[冻结清单](v3_scientific_audit_20260906/evidence_v3_failure_20260907/TOKEN_EVAL_LAUNCH_MANIFEST.json)。其开头冻结G探针已PASS：按输入布局选N2与N16两条原完整batch，各33NN；原final坐标/Gram误差0，累计/净内部位移比1.30/1.58，末端内部fractional RMS仅约1.6e−5。此小样本排查不估计全cohort SUN，G诊断到此关闭。[真实轨迹证据](v3_scientific_audit_20260906/evidence_v3_failure_20260907/G_TRACE_40071.json)。

原CIF全量身份 **40009** 已 `COMPLETED 0:0`，8CPU、1分40秒、无GPU。27136/9047全部通过唯一完整Q结构匹配，与原CSV一一对应；0解析/编码错误、0重复Q歧义、0未核验或丢行。精确site permutation后原连续几何重新编码全部等于source_answer，primary original_cif的来源阻断已关闭。[全量身份与parser证据](v3_scientific_audit_20260906/evidence/CONTINUOUS_SOURCE_IDENTITY_40009.json)。

数值 **40013** 已`COMPLETED 0:0`，19项CPU检查通过，train-only normalizer使用全部27136来源且无std floor命中；val只应用这个normalizer。[数值与normalizer证据](v3_scientific_audit_20260906/evidence/GEOMETRY_NUMERICS_40013.json)。

## 已授权的接续

1. V2 完整终点的固定256 raw/refined及construction→repair评测已完成，保留为对照，不重复提交。
2. 任一端同时达到 Strict SUN ≥10% 和 Meta SUN ≥50%（256 中至少 26/128），追加同索引 1000 raw/refined，并另报所有 1200 请求。
3. G有界原因定位已结束；40071正执行同40064权重的T construction-only全256实验及匹配V2 constructor→tau800基线，真实4请求重放通过才进入正式采样。此部署是另一个方法终点，不能称已修复G。
4. 若V2/V3最终无效，用户指定最终采用K4或K8。重点分析两者共同稳定、互相救回/破坏和持续失败的实际产物，再登记统一适用的小改动并实测；不按病例ID路由，不动共同R、hull、N/U或失败分母。总上限仍为6 A800、24 CPU、两个作业，遵守实际QoS。旧自生成能量teacher等其他暂停路线不自动恢复。
5. 根目录入口与142份历史/组件文件迁移、6份完全重复文档去重已完成；唯一历史证据与hash/迁移映射保留。当前收尾新稿引用、状态入口和提交，不再把历史计划当成运行状态。

自动任务`llm-dlm-sun-24h`每十分钟检查，同一任务包含上海07:00最终交付。6小时实证窗口截止Sep7 06:31，07:00报告最新完整及最好实测SUN；异cohort/端点不拼接两个最优值。常规无变化保持安静，必要进展及时告知。详细诊断保存在审计目录。

论文安排以用户最新方向为准：效果优先且保持LLM+DLM主线。V3若有效，围绕一个科学问题的what/why/how组织2–3项互补贡献；若不足，回到既有有效路径做稳定性改进。P、语言预训练、离散扩散的独立价值没有对应证据时缩小主张，不硬凑贡献。

主任务明确为**de novo组成与结构联合生成**。外部给定组成CSP只用于条件诊断；缓存同一批上游自主生成C/S/P的完整输出仍是de novo系统样本，配对差值用于后端归因。当前256来源已核：全请求保留、无结果筛选/替换、上游失败0、重复组成2，仍属于开发样本。[cohort记账](v3_scientific_audit_20260906/DE_NOVO_COHORT_ACCOUNTING.md)。本次科学问题与写法研究只用已核ICLR/ICML/NeurIPS主会论文：[顶会de novo写法](v3_scientific_audit_20260906/TOPCONF_DE_NOVO_WRITING_REVIEW.md)。
