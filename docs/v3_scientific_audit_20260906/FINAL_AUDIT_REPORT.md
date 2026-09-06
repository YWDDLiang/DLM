# 全链路科学审计整合报告

2026-09-06。全链路与V3两轮独立审查、V2完整终点及冻结诊断已完成；H-P33正式两轮训练40064运行中，连续来源、真实4/6卡及图回环门均已关闭。本文随新证据更新，**不表示用户授权的后续实验已经完成**。

## 1. 裁决与实际SUN

**V2组合改动没有实现目标提点，也没有证据证明一个“修好就必涨SUN”的单一根因。** 此前把K增加、log-SPD/GEM、非零梯度或理论去噪合理性说成收益保证，应撤回。这些证据分别支持候选覆盖、表示/输入、计算图和统计对象，不能代替最终收益。

每格为Strict／Meta SUN命中数，保留原冻结headline口径。

| 方法 | 分母 | raw | 固定tau800 refined |
|---|---:|---:|---:|
| 参考 | 256 | 6／55 | 19／121 |
| 旧K4 | 256 | 7／57 | 19／126 |
| 旧K8 | 256 | 11／66 | 17／123 |
| raw-v1 | 256 | 3／45 | 16／113 |
| V2 final | 256 | 8／50 | 16／113 |
| 旧K8独立固定源序子集 | 1000 | 39／244 | 65／484 |
| 旧K8同cohort全部请求 | 1200 | 47／285 | 79／570 |

开发256为同请求口径；独立1000/1200不与开发集作逐例配对。
V2构造终点raw为12／44，一次full-cell repair后8／50；构造→tau800未测。V2未达到同端26／128，不追加1000。
来源：[历史原始汇总与hash](evidence/EVALUATION_SUMMARIES_20260906.json)、[V2两端](evidence/V2_EVALUATION_SUMMARIES_39998.json)、[完整构造／修订配对](evidence/V2_CONSTRUCTION_AND_REPAIR_39998.json)。

## 2. 覆盖与证据缺口

清点自2026-08-26起739个first-parent commit和209份审计前Markdown，按实验族去重，并保留更早18族台账。559条失败文本是发现线索，不能称559个独立失败。缺原始逐行产物、checkpoint或完整协议的条目维持“历史汇总证据”，没有通过再次引用升级为独立复算。[完整实验与变更台账](EXPERIMENT_CHANGE_CASEBOOK.md)

| 链路 | 核验范围与报告 |
|---|---|
| C³FD／Typed Llama／PoE／S／P | [上游审计](C3FD_LLM_UPSTREAM_AUDIT.md)：支持、witness、typed输入、独立soft heads、归一化、M签名及contact程序 |
| DLM数据／目标／状态／采样 | [概率与kernel](DLM_PROBABILITY_AND_KERNEL_AUDIT.md)：fixed-old、prefix/dense、alias、支持、rollback及数值 |
| 晶体diffusion／flow／DLM | [机制矩阵](CRYSTAL_DIFFUSION_MECHANISMS.md)：原论文、作者代码、条件、forward/reverse、对称性及评测边界 |
| Transformer／GNN | [补充审计](TRANSFORMER_AND_GNN_SUPPLEMENT.md)：额外晶体工作、已有周期消息和融合增量 |
| 物理／R／refiner／SUN | [物理报告](PHYSICS_PIPELINE_AND_STORY_AUDIT.md)：cell/force/stress、停机、N/U/hull及失败事件 |
| 既往结论 | [主张审计](CLAIM_AUDIT.md)：过度保证、代理指标和因果归因 |
| 项目清理 | [迁移报告](DOCUMENT_MIGRATION_REPORT.md)：入口收敛、逐文件hash/映射与引用 |

一般交叉审查已完成：[DLM审晶体/主张](ROUND2_DLM_ON_CRYSTAL_AND_CLAIMS.md)、[物理审DLM/主张](ROUND2_PHYSICS_ON_DLM_AND_CLAIMS.md)。这些不是具体V3的第二轮审稿。

## 3. 成功原因已经支持到哪一层

**化学支持有真实执行收益。** C³FD v2.5两seed合计2000请求通过其composition benchmark，精确count、电荷及声明witness支持解决了文本自由发射下的实际守恒/可达问题。这不证明完整化学空间覆盖，也不保证输出组成在当前几何预算达到低hull。

**站点身份和事务匹配修复过真实接口问题。** species-coordinate记录一起canonicalize，解决早期约85%物种槽位次序不匹配；后续program/schedule匹配改善执行有效性。contact程序非平凡且能执行，并未建立learned P优于canonical的净SUN证据。

**固定refiner具有系统贡献。** 旧K8独立1000从raw39/244变成refined65/484，同时N∩U和方法差异也改变。不能只称refiner“洗掉收益”，也不能把其提升全部记作原生DLM能力。

**官方hull覆盖修复纠正了虚假退化。** 旧K8独立cohort曾用开发小cache，补齐后保持生成、能量、relaxation与索引不变而重算；1200中1172已知、20官方unresolved、8不可重建。这是评测完整性收益，不是生成能力提升。

## 4. 失败原因排序：事实、解释和待测

以下按行动价值排序，不是估计好的因果效应大小。

| 优先级 | 已核事实/机制差距 | 可以解释什么 | 不能据此断言 |
|---|---|---|---|
| 1 | V2为typed scalar CE与事务construction/repair；借log-SPD扰动/GEM没有建立外部晶体diffusion的完整全时间逆过程 | 借表示不能继承整套生成能力，状态/目标/更新必须一起定义 | 连续头是唯一根因或必然更稳定 |
| 2 | V2 repair失去8个Strict、新增4；Meta保留18、失去26、新增32 | 修订没有单调收益，construction是便宜竞争方案 | 所有repair/corrupt→Bayes均无效 |
| 3 | prepared S/P主要由冻结上游给定C独立采样；已测exact-C下M唯一 | 训练可能不识别requested S的几何控制语义；全MP20不补齐冻结上游覆盖 | soft完全无用/一定有害；target-soft oracle可正常部署 |
| 4 | 多次CE/条件margin/候选teacher/energy偏好改善代理量，却未稳定改善SUN | 状态、目标、终点与泛化须分别验证 | 只有能量负样本才可能成功或似然模型不能学稳定性 |
| 5 | codec精度、有限shell和非等变路径有明确限制 | 表示和归纳偏置值得对照 | 一般反例已经证明本批失败由量化或MIC造成 |
| 6 | R优化器停机与外部force/stress联合验收不同；headline含部分无效终态 | 验证与teacher筛选解释需要修正 | 改R/阈值就是模型提点，或加最大步数必解决 |

V2真实两epoch、108544状态、完整source/view覆盖、0corruption fallback、0support conflict、初始化delta0及非零模块梯度排除了若干简单工程解释；82个empty-supervision不直接改名为prefix冲突。[最终训练证据](evidence/V2_FINAL_TRAIN_AND_METADATA.json)

## 5. 冻结模型probe：真实forward收口

40004使用4A800/16CPU、2分24秒；100个validation来源、200状态、五变体，实际425 batches/850行（相同输入复用）。模型不变，无训练、新晶体、MLIP或结果筛样；原目标复算最大差5.25e−7。

| 处理减对照 | source宏平均风险差 | 有限解释 |
|---|---:|---|
| true-noise减unknown | +0.000559 | 输出通路存在，但无清晰平均预测优势 |
| target-soft减predicted-soft、固定P | −0.035770 | 部分目标条件信号存在，oracle/分布外边界保留 |
| masked-old减available-old | +0.062878 | 旧几何确实被使用，主要在repair，不能定位单模块因果 |

soft有30来源输入完全相同、70变化；38个文字改变但长度不变的来源均值约−.04435，因此不能把所有soft效应归于长度。它仍不是SG遵从或SUN证据。
old遮蔽repair均值约+.11990、construction约+.00585，排除“模型完全不看旧几何”的简单故障。

147真实baseline teacher-prefix向量、30645合法logits，raw BF16后按实际采样路径转FP64再exp，范围−19.25至17.125，无overflow/下溢/全零。理论极端logit反例不解释本probe；未来生成状态也未由此全量证明。
source分位数不是CI，token计权TV与source宏均值不能混算。[独立解释](V2_FROZEN_PROBE_INTERPRETATION.md)、[完整证据](evidence/V2_CONDITIONING_PROBE_40004.json)。

## 6. 为何不能直接继承DiffCSP++等工作

机制报告核对DiffCSP/++、CDVAE、FlowMM、FlowLLM、MatterGen及其他晶体diffusion/flow/Transformer和离散diffusion。每项都保留任务、条件、分母、训练和作者实现边界：

- DiffCSP++的SG/Wyckoff约束与oracle条件CSP，不等同三个独立soft hints的de novo；CSP匹配率不是SUN。
- 周期表示、晶格chart、几何特征、forward和reverse是不同层；采用其中几项没有实现同一学习对象。
- 正确似然/score/flow可以靠数据学习稳定结构，不要求每种成功方法都加能量损失；经验记忆却可令novelty趋零，稳定与新颖必须共同看。
- 外部几何网络的对称性、参数/训练量和forward成本不同，其100—1000步不能免费换成8B LLaDA调用。
- 本模型已有全局Transformer与1-hop周期消息；多跳/角度/等变GNN是额外增量。局部等变特征不会让带绝对位置/rank的整个LM自动等变。

这些提供可迁移机制，不为本项目SUN背书。[原论文/作者代码矩阵](CRYSTAL_DIFFUSION_MECHANISMS.md)、[额外Transformer/GNN](TRANSFORMER_AND_GNN_SUPPLEMENT.md)。

## 7. 物理与统计限制

headline H、终态几何通过G、完整verified V不同。旧K8 raw H=47/285中7/12来自已知invalid_terminal，G=40/273、V=19/136；raw-v1 H=3/45中2/3来自invalid，G=1/42、V=0/20。各基线必须同口径；不能先改某个方法再宣称超越。[交叉表](evidence/TERMINAL_STATUS_SUN_CROSSTAB.json)

V2 construction/repair的invalid总数79/83，不是SUN中invalid命中数，不能从12/44或8/50直接相减。

R/FIRE的generalized-gradient停止不等于外部Cartesian force≤0.1和stress≤0.5GPa联合通过。K8 tau763个not-converged中762个已optimizer-stop且未到500步，raw-v1 tau164个全部如此；单增最大步数不能修复已经主动停机的过程。评测频率不能直接外推为旧teacher入池排除率。

radius2有数学反例，但本轮5788个实际可用几何中没有发现漏掉0.5Å以下近邻；有限经验核验不是任意triclinic胞的全局保证。

N/U保持冻结StructureMatcher和源序规则；official unknown及全请求失败保留；paired1000不按能量/SUN/verified筛取。用于设计的开发与旧1000不再称未见确认。

## 8. V3具体选择与审查

[A/B/C候选](V3_DLM_CANDIDATES_AND_ATTACKS.md)、[共享几何头反提案](V3_GEOMETRIC_HEAD_COUNTERPROPOSAL.md)各有独立作者；[首轮数学](V3_REVIEW_R1_MATH.md)和[首轮物理](V3_REVIEW_R1_PHYSICS.md)给出实质反例。

C的forward/score可定义生成器，但64步β_tilde在oracle Gaussian数据上可使方差约降至.852；rank clock混入末端量化精度。该DDPM反例不适用于H的v/ODE，H自身有限ε却仍是blur端点。B不能以trace.success判断inner提交，A需面对联合条件支持与oracle不可得。

主审收敛为 [H-P33具体规格](V3_H_P33_SPECIFICATION.md)：

1. 完整V2 final暖启动，冻结base但G直接训练共享LoRA/conditioner；T construction与G各一view/source/epoch，新增两epoch。
2. train-only log-SPD chart、统一site时钟、joint lattice-v/wrapped torus-u；不加rank clock/H-C/多轮repair/新GNN。
3. 原CIF全量身份、精确site permutation和Q相等通过后才用original_cif；不静默混decoded标签。
4. 32Euler区间加一次ε=.002实际forward/readout，共33NFE；确定性readout不冒充精确posterior sample。float native主终点，同样样本Q raw为secondary。
5. 明示混合表示、G中P仅条件、新增曝光和活参数/NFE；共同评估/refiner/失败分母不改。

尚未证明它是最佳架构或会提高SUN。它必须区别于旧PMTR的冻结head-only路径，以真实共享梯度证明实现。[第二轮数学](V3_REVIEW_R2_MATH.md)与[第二轮物理](V3_REVIEW_R2_PHYSICS.md)已经完成，均允许推进有界实现。模型接口、来源调度、trainer与float导出已冻结为1111739；40项mixed CPU检查通过，[独立模型验收](MIXED_GEOMETRY_MODEL_CPU_ACCEPTANCE.md)与[数据调度审查](IMPLEMENTATION_DATA_SCHEDULE_REVIEW.md)注明小模型/CPU边界。真实8B、DDP、低σ敏感度、33次前向和保存恢复由40045执行，尚不能据此宣称SUN收益。

来源门已由实际CPU40009关闭：train27136/val9047全verified、与CSV一一对应，唯一完整量化answer匹配；0解析错误、0完整Q重复歧义、0未核验或丢行。原连续值经精确site permutation后Q全部等于现有source_answer，未用formula匹配、cif.conv或decoded fallback。它支持original_cif监督，不证明模型梯度、稳定性或SUN。[完整证据](evidence/CONTINUOUS_SOURCE_IDENTITY_40009.json)

## 9. 清理与剩余执行

根目录旧入口保留完整字节快照；当前入口收敛为README/CURRENT_STATE/CURRENT_ARCHITECTURE/WORKFLOW。142份历史/组件文件迁入本审计historical/docs或reference/component_specs；6份完全重复文档经hash相同且有immutable来源后删除重复副本。[迁移映射](DOCUMENT_MIGRATION_MAP.json)、[旧根入口快照](historical/root_entrypoints/MIGRATION_MANIFEST.json)

旧公式曾被误识别为Markdown链接，已恢复原字节并修正检查器；历史旧prompt原先缺5目标不伪造补齐，当前入口单独增量检查。原执行归档、权重、数据及其他工作树不随清理变动。

真实实现验收已关闭：[40045/40060及40058汇总](MIXED_IMPLEMENTATION_ACCEPTANCE.md)。正式选择6卡micro2/acc2/global24，数据窗口与108544状态不变；实际checkpoint重复输出差0，不把BF16分组梯度容差或非零敏感度当作精确score/良好条件数证明。

剩余：完成40064新增2epoch；固定256 primary raw/refined与Q诊断；满足门槛则1000/1200。按用户最新授权，若V3两端缺乏可信改善，回到K4/K8或之前有效路径选择稳定性改动并实际应用；若有效，围绕统一科学问题的what/why/how收敛2–3项互补贡献。[故事契合边界](V3_STORY_FIT.md)与性能分开核对。不能在综述或单次负结果后提前结束，也不以没有事先SUN保证为由无限审计。
