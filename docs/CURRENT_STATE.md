# 当前项目状态

更新日期：2026-09-07（上海）。此文件统一维护当前状态；历史材料中的旧状态不覆盖本页。

## 当前阶段

**40088合作修复终点对照正在运行；此前V2/V3、contact、等权混合与构造终点都未达到同端10%/50%。** 最新完整40083为K4 raw5/48→tau15/126、K8 raw5/54→tau16/119，均256且无1000资格。40088复用原full paths，所有成功构造请求统一停在coop END，原构造失败保留；不按病例挑阶段。上海Sep7 07:00交付最新和最好实测SUN，06:31为6小时实验窗口末端。

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
| V3 T构造 / 同开发 256 | 13 / 53 | 17 / 111 |
| V2构造 / 同开发 256 | 12 / 44 | 17 / 120 |
| K4短接触软约束 / 同开发 256 | 6 / 56 | 20 / 122 |
| K8短接触软约束 / 同开发 256 | 11 / 68 | 15 / 123 |
| K4/K8等权概率混合 / 同开发 256 | 10 / 51 | 18 / 118 |
| K4构造终点 / 同开发 256 | 5 / 48 | 15 / 126 |
| K8构造终点 / 同开发 256 | 5 / 54 | 16 / 119 |
| 旧 K8 / 独立条件 1000 | 39 / 244 | 65 / 484 |
| 旧 K8 / 对应全部 1200 请求 | 47 / 285 | 79 / 570 |

旧 K8 独立评测此前误用了开发小 cohort 的 hull 缓存。官方覆盖已补齐，CPU 作业 39990 重评成功，生成、弛豫、能量和选择索引均未改变。更正后 1200 请求中 hull 已知 1172、官方明确 unresolved 20、不可重建 8。旧覆盖不足的 0/9、1/17 不再作为完整结果。补统计不是模型提升。

原始评测文件与 hash：[评测证据](v3_scientific_audit_20260906/evidence/EVALUATION_SUMMARIES_20260906.json)。历史小幅差异、共同 verified 子集及外部论文分数须按实际口径解释，不能混用来声称提升。

V2 构造终点原生 SUN 为 12/44，一次 full-cell repair 后为 8/50；Strict −4、Meta +6，是这组配对请求的权衡，不能称修订普遍改善。V2 refined 与 raw-v1 同计数，未实现目标提点。[V2两端](v3_scientific_audit_20260906/evidence/V2_EVALUATION_SUMMARIES_39998.json)、[同次构造与修订](v3_scientific_audit_20260906/evidence/V2_CONSTRUCTION_AND_REPAIR_39998.json)。

冻结诊断 **40004** 已 `COMPLETED 0:0`，4A800/16CPU、00:02:24。100来源/200状态、5变体，0训练、新晶体与MLIP；原目标风险复算最大差5.25e−7。旧几何遮蔽主要损害repair风险，真实噪声相对unknown没有清晰平均收益，真实soft降低部分风险但不是可用oracle或SUN证据。真实prefix logits的float64 exp未出现溢出。[完整证据](v3_scientific_audit_20260906/evidence/V2_CONDITIONING_PROBE_40004.json)。

## V3 实施与验收

[H-P33 规格](v3_scientific_audit_20260906/V3_H_P33_SPECIFICATION.md) 复用 V2 final 与共享 LLaDA，保留 T construction、以 G 全几何 v/torus 去噪替代旧 repair CE；已完成新增两epoch。主采样为32个Euler区间加一次明确末端读出，float CIF为主终点，同样样本Q raw为精度诊断。P在G中仅作条件/rank，不新增rank噪声时钟；这是混合表示模型。[第二轮数学](v3_scientific_audit_20260906/V3_REVIEW_R2_MATH.md)与[第二轮物理](v3_scientific_audit_20260906/V3_REVIEW_R2_PHYSICS.md)均允许进入有界实现验收，没有SUN收益保证。

实施冻结提交`1111739bc5a590734b02a61f395faa59be605531`包含共享hidden-only/G头、原V2加载兼容、来源与T/G调度、正式trainer、float导出与真实preflight。模型/数学/数据共40项本机CPU测试通过，40045中Torch2.4再次40项通过；旧V2 loader另2项、export/regression新旧29项通过。[模型CPU验收](v3_scientific_audit_20260906/MIXED_GEOMETRY_MODEL_CPU_ACCEPTANCE.md)、[独立数据调度审查](v3_scientific_audit_20260906/IMPLEMENTATION_DATA_SCHEDULE_REVIEW.md)。

真实预检 **40045**（4卡micro1/acc6）与 **40060**（6卡micro2/acc2）均`COMPLETED 0:0`且全PASS，分别4分48秒、4分21秒。真实共享G梯度非零、T-only增量0、各rank权重一致、33NFE与末端解码有限、T/G保存恢复差0。4卡人口梯度相对误差2.45e−8；6卡整体0.476%、最大片组1.901%，在事前2% BF16门内，不称逐bit相同。[实现验收汇总](v3_scientific_audit_20260906/MIXED_IMPLEMENTATION_ACCEPTANCE.md)。

实际CIF→CrysLLMGen process_one→FP32 refiner输入回环 **40058** 已PASS，8CPU/2分30秒、无模型/MLIP调用。4个例子含N=1/10/20和非Q网格fixture，显式Niggli后Gram/按元素坐标双射检查通过；FP32坐标误差≤2.92e−8，单原子等体积量化反例被拒绝。[实际回环](v3_scientific_audit_20260906/evidence/CONTINUOUS_GRAPHS_40058.json)。

正式配置6A800/24CPU、micro2/acc2/global24，完整27136来源各1T+1G/epoch，新增2epoch共4524更新/108544真实状态。新训练保持fresh optimizer和事先种子。冻结采样器11项CPU测试通过；正式raw/refined及secondary Q raw评测已完成。

**40064**已`COMPLETED 0:0`，实际用时01:29:21；2新增epoch、4524更新、108544有效状态，所有source的T/G覆盖min=max=2，两轮验证均完成，最终step-4524取得采样资格。[完整训练终点](v3_scientific_audit_20260906/evidence/MIXED_TRAIN_FINAL_40064.json)。训练与评测代码均固定`0bf8e8870f96c54e86cbd080e64c527f55fb6943`，运行归档保持不变。[冻结训练清单](v3_scientific_audit_20260906/evidence/MIXED_TRAIN_LAUNCH_MANIFEST.json)、[训练提交](v3_scientific_audit_20260906/evidence/MIXED_TRAIN_SUBMISSION.json)。

评测 **40066** 已`COMPLETED 0:0`，实际00:23:50，完整256 primary float raw/refined与secondary Q raw。G raw为9/61、tau800为13/116，Q raw为6/58；各端对应verified SUN为6/28、6/47、2/26。无主端达到同端26/128，不触发G独立1000。[原始最终报告](v3_scientific_audit_20260906/evidence_v3_failure_20260907/EVAL_FINAL_40066.json)、[raw](v3_scientific_audit_20260906/evidence_v3_failure_20260907/NATIVE_EVALUATION_40066.json)、[tau800](v3_scientific_audit_20260906/evidence_v3_failure_20260907/TAU800_EVALUATION_40066.json)、[Q](v3_scientific_audit_20260906/evidence_v3_failure_20260907/QUANTIZED_EVALUATION_40066.json)。原权重、代码和全请求产物保持冻结。

CPU诊断 **40069** 已`COMPLETED 0:0`，8CPU、12秒、无模型/能量调用。实际256个输出到原生label输入、CIF、Niggli图与ProposalDataset FP32/refiner seed全链无错配；G末态相对坐标移动中位0.0606 Å，仍保留32条初始<0.5 Å碰撞并新增8条。100来源六band准确零预测对照显示低/中噪声u无实质改善；v在最高两band改善约25.5%/32.4%。[实际链](v3_scientific_audit_20260906/evidence_v3_failure_20260907/OUTPUT_CHAIN_40069.json)、[零场对照](v3_scientific_audit_20260906/evidence_v3_failure_20260907/ZERO_VALIDATION_BASELINE_40069.json)、[跨策略产物解释](v3_scientific_audit_20260906/V3_FAILURE_CROSS_STRATEGY_PHYSICS.md)。这支持坐标场欠学习，不单独证明原因或不可修复。

**40071**已`COMPLETED 0:0`，用时00:30:37，6A800/24CPU，执行`b3efa789fab1a70529251c9c364d5e9ae027e193`，新训练0。真实4请求/150动作fresh replay最大logp误差0，正式256全部生成；T raw13/53→tau80017/111，verified6/25→9/45。匹配V2构造raw12/44→tau80017/120，τ verified5/41；原构造重导出与旧标签精确绑定PASS，未误用repair基线。无端达到同端26/128，不触发补1000。[完整结果](v3_scientific_audit_20260906/evidence_v3_failure_20260907/TOKEN_EVAL_FINAL_40071.json)、[真实重放](v3_scientific_audit_20260906/evidence_v3_failure_20260907/TOKEN_PILOT_REPLAY_40071.json)、[V2构造绑定](v3_scientific_audit_20260906/evidence_v3_failure_20260907/V2_CONSTRUCTION_BINDING_40071.json)。

40071开头冻结G探针已PASS：按输入布局选N2与N16两条原完整batch，各33NN；原final坐标/Gram误差0，累计/净内部位移比1.30/1.58，末端内部fractional RMS约1.6e−5。此小样本不估计全cohort SUN，G诊断到此关闭。[真实轨迹](v3_scientific_audit_20260906/evidence_v3_failure_20260907/G_TRACE_40071.json)、[收束结论](v3_scientific_audit_20260906/evidence_v3_failure_20260907/G_DIAGNOSIS_CLOSURE.md)。

**40074**已COMPLETED 0:0，用时00:52:26，6A800/24CPU，执行fa4bd808b0fb9e379875b80635ed0a314bd52964。固定同一短接触软偏好、各256完整raw/tau：K4为6/56→20/122（verified2/29→9/49），K8为11/68→15/123（verified4/34→10/50）。旧K4为7/57→19/126、旧K8为11/66→17/123；新结果均有权衡，K4tau StrictStable仍28。原prefix重放与4条biased fresh replay均PASS，全部失败保留。无端达到26/128，不触发contact1000，不再扫系数。[总结果](v3_scientific_audit_20260906/evidence_v3_failure_20260907/CONTACT_EVAL_FINAL_40074.json)、[四端及配对原始hash](v3_scientific_audit_20260906/evidence_v3_failure_20260907/CONTACT_FINAL_RECEIPTS_40074.json)、[物理诊断](v3_scientific_audit_20260906/K4_K8_STABILITY_AND_CONTACT_DIAGNOSIS.md)。

**40076**已COMPLETED 0:0，用时00:36:46，6A800/24CPU，执行5b95512b21acc7d83775400fa0c07943816b053a。固定K4/K8同态等权概率混合，T=.7、每动作2NN、无contact/MLIP/权重搜索。真实8前缀16NN和4轨迹369动作fresh双模型重放均PASS，正式256请求255生成成功，两模型各13563次forward；原生SUN10/51、tau800为18/118，verified3/29→8/50。均未达26/128，不触发混合1000，不继续扫权重。[完整结果](v3_scientific_audit_20260906/evidence_v3_failure_20260907/MIXTURE_EVAL_FINAL_40076.json)、[原始报告及hash](v3_scientific_audit_20260906/evidence_v3_failure_20260907/MIXTURE_FINAL_RECEIPTS_40076.json)、[方法审查](v3_scientific_audit_20260906/K4_K8_EQUAL_MIXTURE_BOUNDED_REVIEW.md)。

**40083**已COMPLETED0:0，用时31:42，6A800/24CPU，执行f2f57d44c5221172bfabd866b2fa573244e93859。原K4/K8全256轨迹统一截到首个construct END，保留原动作、C/S/P、64bit seed和所有失败，0新DLM采样。K4有254构造成功/2失败，raw5/48→tau15/126（verified2/20→6/49）；K8有252/4，raw5/54→tau16/119（verified2/24→9/48）。均未达26/128，不触发构造终点1000。K4 tau与原完整循环Meta均126，但共同112、各独有14；Strict共同13、原完整循环独有6、构造独有2，不能把同计数称逐例稳定。[总原始终点](v3_scientific_audit_20260906/evidence_v3_failure_20260907/CONSTRUCTION_EVAL_FINAL_40083.json)、[K4原始hash](v3_scientific_audit_20260906/evidence_v3_failure_20260907/CONSTRUCTION_K4_FINAL_RECEIPTS_40083.json)、[K8与后续登记原始hash](v3_scientific_audit_20260906/evidence_v3_failure_20260907/CONSTRUCTION_PHASE_COOPERATIVE_RECEIPTS_20260907.json)。

**40086**只读CPU阶段审计已COMPLETED0:0，1:30，0GPU/2CPU/16G，执行2d0dfd96d7f9c9e7f4cef67cfd1742c0348625e6。两方法各256、metadata_errors0、三END与原完整末态tokens精确核对PASS；真实构造失败后阶段记not_executed。有效结构的完整周期最短距均满足原0.5Å硬约束。构造/coop/closure的最短距中位：K4 .926/.9705/.972Å，K8 .9496/1.0337/1.135Å；closure体积逐例不变但改253/254、246/252条末态tokens。这只描述几何，不能证明SUN收益。[真实全量摘要](v3_scientific_audit_20260906/evidence_v3_failure_20260907/PHASE_GEOMETRY_SUMMARY_40086.json)、[提交](v3_scientific_audit_20260906/evidence_v3_failure_20260907/PHASE_GEOMETRY_SUBMISSION.json)。

**40088**已提交并确认RUNNING，6A800/24CPU/440G/2h，执行5d0904d842aa6ca5886632bb534a38b569792b49。48项CPU检查PASS4.37s，源码LF绑定与两batch接口/语法检查通过。固定k4_cooperative_endpoint_v1/k8_cooperative_endpoint_v1：原模型原全256轨迹，统一保留构造+唯一coop的全部draw/rollback，到coop END停止；构造失败不删行，无新DLM/MLIP选择，原R/tau/hull/N-U不变。K4 raw已完成8/48（verified3/27），tau与K8尚待完成；不据partial申请1000。若完整结果有真实26/128，冻结对应方法补评固定独立条件1000及全部1200；若两者都达，K4优先。[固定规格](v3_scientific_audit_20260906/evidence_v3_failure_20260907/k4k8_cooperative/candidate_spec.json)、[冻结清单](v3_scientific_audit_20260906/evidence_v3_failure_20260907/COOPERATIVE_EVAL_LAUNCH_MANIFEST.json)、[唯一提交](v3_scientific_audit_20260906/evidence_v3_failure_20260907/COOPERATIVE_EVAL_SUBMISSION.json)、[源码与48项检查](v3_scientific_audit_20260906/evidence_v3_failure_20260907/k4k8_cooperative/EXECUTION_LF_BINDING.json)。

重复性诊断已收束：各171个相同有效native结构，实际refiner输入张量与逐图seed全部相同；4次运行共1011有效tau结果也通过rank合并/样本/原子映射，坐标误差0。N=2各5例精确复现，N≥3各166例不精确；归约顺序是有形状证据支持的候选原因，未确认首个分叉算子。有限平移/同元素对应残差只是达到的上界。原生R标签自己也有小幅重复差异，因此小幅SUN变化不能全部归因contact或refiner。[收束结论](v3_scientific_audit_20260906/REFINER_REPEATABILITY_DIAGNOSIS.md)。

原CIF全量身份 **40009** 已 `COMPLETED 0:0`，8CPU、1分40秒、无GPU。27136/9047全部通过唯一完整Q结构匹配，与原CSV一一对应；0解析/编码错误、0重复Q歧义、0未核验或丢行。精确site permutation后原连续几何重新编码全部等于source_answer，primary original_cif的来源阻断已关闭。[全量身份与parser证据](v3_scientific_audit_20260906/evidence/CONTINUOUS_SOURCE_IDENTITY_40009.json)。

数值 **40013** 已`COMPLETED 0:0`，19项CPU检查通过，train-only normalizer使用全部27136来源且无std floor命中；val只应用这个normalizer。[数值与normalizer证据](v3_scientific_audit_20260906/evidence/GEOMETRY_NUMERICS_40013.json)。

## 已授权的接续

1. V2 完整终点的固定256 raw/refined及construction→repair评测已完成，保留为对照，不重复提交。
2. 任一端同时达到 Strict SUN ≥10% 和 Meta SUN ≥50%（256 中至少 26/128），追加同索引 1000 raw/refined，并另报所有 1200 请求。
3. G有界定位和40071的T/V2构造实验已完整结束，保留其负面与权衡结果，不再扩大G积分器或T参数尝试。
4. 用户指定V2/V3最终无效后采用K4或K8。实际稳定/失败产物分析已完成，40074固定软惩罚已完整评估且未达标，40076等权混合也已失败，40083构造终点也未达标，最后的40088合作修复终点正在评估；不按ID路由，不根据SUN改系数，不改共同R、hull、N/U或失败分母。必要1000入口已准备，只有真实26/128才登记和启动。总上限仍为6 A800、24 CPU、两个作业；旧自生成能量teacher等暂停路线不自动恢复。
5. 根目录入口与142份历史/组件文件迁移、6份完全重复文档去重已完成；唯一历史证据与hash/迁移映射保留。当前收尾新稿引用、状态入口和提交，不再把历史计划当成运行状态。

自动任务`llm-dlm-sun-24h`每十分钟检查，同一任务包含上海07:00最终交付。6小时实证窗口截止Sep7 06:31，07:00报告最新完整及最好实测SUN；异cohort/端点不拼接两个最优值。常规无变化保持安静，必要进展及时告知。详细诊断保存在审计目录。

论文安排以用户最新方向为准：效果优先且保持LLM+DLM主线。V3若有效，围绕一个科学问题的what/why/how组织2–3项互补贡献；若不足，回到既有有效路径做稳定性改进。P、语言预训练、离散扩散的独立价值没有对应证据时缩小主张，不硬凑贡献。

主任务明确为**de novo组成与结构联合生成**。外部给定组成CSP只用于条件诊断；缓存同一批上游自主生成C/S/P的完整输出仍是de novo系统样本，配对差值用于后端归因。当前256来源已核：全请求保留、无结果筛选/替换、上游失败0、重复组成2，仍属于开发样本。[cohort记账](v3_scientific_audit_20260906/DE_NOVO_COHORT_ACCOUNTING.md)。本次科学问题与写法研究只用已核ICLR/ICML/NeurIPS主会论文：[顶会de novo写法](v3_scientific_audit_20260906/TOPCONF_DE_NOVO_WRITING_REVIEW.md)。
