# 2026-09-07 限时实验结果

在约定六小时实验窗口内，没有做出达标方法。本轮已完成的14组主要开发对照，均未在同一端点达到 **Strict SUN ≥10% 且 Meta SUN ≥50%**。最后一个实验40088已COMPLETED0:0，用时30:05；K4/K8各自raw、固定tau800、与原完整循环配对及总完成标记齐全。原始结果已经冻结，按约定上海07:00交付。

以下均为同一开发cohort，**每个方法保留全部256请求**。表中命中数按Strict/Meta列出；达到目标至少需要同端26/128。

| 方法 | raw命中数 | tau800命中数 | tau800百分比 |
|---|---:|---:|---:|
| 原K4完整循环（保留主线） | 7/57 | 19/126 | 7.42% / 49.22% |
| K4短接触软约束（Strict最高） | 6/56 | 20/122 | 7.81% / 47.66% |
| 最新K4合作修复终点 | 8/48 | 18/114 | 7.03% / 44.53% |
| 最新K8合作修复终点 | 10/57 | 15/122 | 5.86% / 47.66% |

原K4与短接触K4各有取舍，**不能拼成20/126**。短接触版比原K4多1个headline Strict，但二者verified Strict SUN同为9，verified Meta为49对53，且存在已核实的重复运行差异；当前不足以据此替换原K4。推荐保留原K4完整循环，保留软约束结果作为对照。[基线checkpoint、权重hash与原采样阶段](evidence_v3_failure_20260907/FINAL_BASELINE_DECISION_20260907.json)已明确登记。这个选择基于开发集，尚不能声称K4在独立大样本上优于K8。

原生端的两个最高值也来自不同方法：V3 T构造Strict最高13/256（5.08%），其Meta为53/256（20.70%）；K8短接触Meta最高68/256（26.56%），其Strict为11/256（4.30%）。[完整14方法快照](evidence_v3_failure_20260907/SUN_SNAPSHOT_AFTER_40088.md)及[原始报告路径、hash和verified计数](evidence_v3_failure_20260907/SUN_SNAPSHOT_AFTER_40088.json)保留全部结果。

三个固定阶段的实际计数如下。构造与合作终点都统一截断全部原轨迹，没有按病例或能量挑阶段；原模型、C/S/P、seed、失败记账、共同R及固定tau800保持同一口径。

| 模型／端点 | 构造结束 | 合作修复结束 | 完整循环 |
|---|---:|---:|---:|
| K4 raw | 5/48 | 8/48 | 7/57 |
| K4 tau800 | 15/126 | 18/114 | 19/126 |
| K8 raw | 5/54 | 10/57 | 11/66 |
| K8 tau800 | 16/119 | 15/122 | 17/123 |

完整循环的tau800优于各自两个截断终点。保留closure后，两模型raw Meta均增加9，但K4 raw Strict少1；不能称每一项都单调改善。headline与verified还可能反向：K8从合作终点到完整循环，tau800 headline由15/122变为17/123，verified却由9/49变为8/47。

逐请求核对覆盖12组原始attempt/EVAL文件、3072条阶段结果和2048条相邻配对，SHA、身份、原bool计数以及共同R/N-U协议全部通过。每个相邻对只有27–63条双方verified+known，56–73条验证状态发生变化。因此headline净增不能直接解释为已验证的能量净改善；verified标记本身也不是headline公式的输入。[逐请求短结论](evidence_v3_failure_20260907/phase_outcomes_40088/CONCLUSIONS.md)、[完整配对表](evidence_v3_failure_20260907/phase_outcomes_40088/adjacent_changes.csv)。

对于“哪些一直稳定、哪些反复失败”，将K4/K8各三个阶段合成六个已观察端点后：

- tau800六端都为verified Strict SUN的只有 **88／AlAgLu2、250／Ba6Pt2**；六端都为verified Meta SUN的有18条。raw六端共同verified Strict SUN为0，共同verified Meta SUN只有96／CaCd2Nd。
- tau800六端都verified且hull已知、但原Meta标记均未通过的有7条：1、39、74、139、165、169、201，包括PrPu和CaCl2。它们是已观察到的可靠失败末态，不证明这些组成不存在其它稳定结构。raw对应共同失败有4条：1、112、138、144。
- 原K4 tau800有28/160条满足Strict/Meta能量阈值，经过N/U后成为19/126；原K8由24/157变为17/123。多样性损失与真正高能、未verified和生成失败分别记账。
- 每格有8条官方明确unresolved的hull，official_cache_not_covered为0；K4另有2条、K8另有4条输入未重建，未混入unknown，也未当高能失败。

这些交集只用于诊断，没有拿来选择或拼接生成结果。[全部共同成功／失败ID与组成](evidence_v3_failure_20260907/phase_outcomes_40088/CROSS_MODEL_COMMON_OBSERVATIONS.json)、[逐阶段真实能量与验证](evidence_v3_failure_20260907/phase_outcomes_40088/COMMON_VERIFIED_IDS.json)。

阶段几何重建的512条原轨迹全通过：有效结构的三个阶段均满足原0.5Å硬约束；closure体积逐例不变，但K4有253/254、K8有246/252条token末态改变。相同上游token仍可能有S/verified变化，整批其它输出也可能改变U代表。脚本将这些情况与仅N-U变化分开，保留[R/refiner重复性边界](REFINER_REPEATABILITY_DIAGNOSIS.md)；几何代理没有代替稳定性证据。

V3 G本轮的主要失败证据是坐标场欠学习：100来源、六个噪声band的零预测对照显示坐标u场无实质改善，全256产物也显示相对坐标移动很弱；冻结小轨迹探针没有发现足以解释问题的大振荡或末端读出抵消。[G诊断收束](evidence_v3_failure_20260907/G_DIAGNOSIS_CLOSURE.md)。后续T-only、固定接触软惩罚、等权概率混合和两种阶段截断都已实测，均未达到目标。本轮不能据此声称新的性能贡献。

已完成的旧K8独立条件结果单独保留，不能与开发256作直接配对比较：

| 旧K8独立条件口径 | raw Strict/Meta | tau800 Strict/Meta |
|---|---:|---:|
| 固定解析顺序1000 | 39/244（3.90%/24.40%） | 65/484（6.50%/48.40%） |
| 对应全部1200请求 | 47/285（3.92%/23.75%） | 79/570（6.58%/47.50%） |

该独立cohort此前已用于K8，不能称未见数据上的新确认。由于新候选均没有真实26/128资格，准备好的新1000入口没有执行；没有用更多样本替换失败请求。

实现和评估均可追溯：合作终点采样／提取默认开关关闭，48项CPU检查通过；所有实际GPU作业已结束，最后逐请求分析只有CPU读取，未新增NN、MLIP或优化。[40088冻结清单](evidence_v3_failure_20260907/COOPERATIVE_EVAL_LAUNCH_MANIFEST.json)、[40088完整结果](evidence_v3_failure_20260907/COOPERATIVE_EVAL_FINAL_40088.json)、[24原始文件与实际CPU核对](evidence_v3_failure_20260907/phase_outcomes_40088/REAL_CPU_EXECUTION.json)、[最终SUN快照hash](evidence_v3_failure_20260907/SUN_SNAPSHOT_FINAL_RECEIPTS_40088.json)。

文件原字节、分析源码及当前文档链接已统一核验：[最终交付核验记录](evidence_v3_failure_20260907/FINAL_DELIVERY_VERIFICATION_20260907.json)。
