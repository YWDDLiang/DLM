# 40064／40066 实际失败定位：最小证据映射

2026-09-07。响应最新优先级：先定位 G 失败，T 实现暂停。本次仅检查当前代码与已同步记录；不改40066、不更新权重、不创建 optimizer、不提交 GPU。根代理已报告 G raw 9/61、τ800 13/116；本文不把这些结果移到未测 T。硬截止为上海 2026-09-07 06:31。

**先做实际全256输出链核对，再用固定少量冻结 forward 区分学习与积分。** 当前记录足以发现输出／图路径问题，却不足以从最终晶体反推最后一步的 v/u。不要先换 solver、clock、readout 或训练目标。

## 1. 已有真实记录能回答什么

| 文件／字段 | 直接可用的检查 | 不能据此下的结论 |
|---|---|---|
| [40064 TRAIN_FINAL](evidence/MIXED_TRAIN_FINAL_40064.json) | 实际4524 updates、2新增epoch、T/G各54272真实状态、coverage min=max=2；确认读的最终policy及完整预算。 | 文件没有loss、prediction norm或时段误差，完成训练不等于学会生成。 |
| `train/training_log.jsonl` | 各区间 `geometry_lattice_sum/geometry_weight_sum`、`geometry_coordinates_sum/geometry_weight_sum`、T对应指标及step；只比较相同加权口径。 | training的log-uniform随机t总体均值不能与验证六点等权平均直接比较。 |
| `train/validation_log.jsonl` | 同一固定100来源、epoch=0和相同band噪声，比较step2262/4524的 `t_band_b_geometry_{lattice,coordinates}_sum / t_band_b_geometry_weight_sum`。 | 只有两个epoch端点；没有零头基准或逐来源误差。高t的u目标本来趋零，小u-loss不证明学会去噪。 |
| `sample/CONFIG.rank*.json` | 最终policy、normalizer全部μ/std、torch/device/dtype、conditions SHA、max_length、seed/布局；结合launch核对实际版本0bf。 | normalizer正确、数值finite不说明输出分布合理。 |
| `sample/SAMPLE_FINAL.json` 与 `SAMPLING.rank*.json` | 全256成功／失败及reason，实际forward/row/padding数量、原logical batch的sample_indices。 | 33NFE全部完成只排除显式执行失败，不能证明场正确或积分稳定。 |
| `sample/native/paths.jsonl` | 每条精确初始CPU FP64 `initial_geometry_prior.z/fractional/sha256`、最终float `structure`、time grid、NFE、失败stage、条件/seed。可由normalizer重新编码最终晶格并比较初始/最终分布。 | 成功行未保存最终z、pre-ε状态或任何v/u；`trace`只是summary，不是ODE逐步trace。 |
| `sample/quantized/paths.jsonl` | 同draw的Q structure/body、clip counts、alias处理及失败预览；与float按trajectory逐条比较。 | Q较好或较坏不能定位continuous solver；也不能按Q结果替换G主端。 |
| `native/paths.jsonl`、`ARTIFACT_FINAL.json`、`proposal_graphs.pt` | native_structure/structure、artifact_error、refiner_graph_error、graph数量/原sample_idx/refiner_seed；可核验实际图几何。 | graph生成成功只验证N/元素数量；exporter现有代码没有对每个实际graph验证Gram或坐标等价。 |
| `native-labels/labels.jsonl`、`tau800-labels/labels.jsonl`、各 `*-evaluation/attempt_results.jsonl` | 全请求status，raw_min_distance、raw energy/force/stress，终态几何、SUN/N/U及verified；refiner上游与下游可逐条连接。 | verified子集均值改善不能替代全256；共同R的失败不能未经证据归为新sampler错误。 |

实际验证时点是六个区间几何中心：`sqrt(.002*.01)`、`sqrt(.01*.05)`、`sqrt(.05*.2)`、`sqrt(.2*.5)`、`sqrt(.5*.9)`、`sqrt(.9*1)`，**没有t=1或ε=.002的直接验证值**。[当前实现](../../src/scripts/train_mixed_geometry_dlm.py)、[验证源／noise stream](../../src/crystal_dlm/mixed_geometry_training_data.py)。

## 2. 先做0 GPU的实际输出链核对

1. 按原 `sample_idx/group_id/trajectory_id` 连接sample float、Q、native导出、τ800、labels和attempt_results，核验256全集、seed/条件/原子身份一致；列出每一层新增缺失、generation_failure、invalid_raw、graph failure。失败保留原分母，不从链中消失。
2. 对每个实际成功请求，比对sample.structure与native.native_structure／native.structure的lattice Gram、固定species次序及fractional；正常应无几何变化。再独立读实际导出CIF，核验Gram、composition、逐species周期坐标。不能仅看体积或N，单原子同体积的错误晶格也会漏检。
3. 从每个真实graph的 `length/angle/a_type/x_coord` 重建结构。`process_one(..., True, False, "crystalnn", False, .01)` 采用Niggli处理，因此expected使用**该请求的实际CIF**经 `get_reduced_structure()` 后的等价几何；逐species匹配周期坐标，检查reduced Gram、volume与距离。可复用 [graph_geometry_checks／ProposalDataset检查](../../operations/mixed_geometry/verify_continuous_graphs.py) 的比较函数，但输入换为实际40066全256，不能以40058的四个工程源通过替代。
4. 原graph→ProposalDataset FP32转换须检查实际N/元素/seed、length/angle/F舍入量和off-0.01-grid坐标是否仍保留。缺图时看具体异常：坏生成几何使图处理失败与正常几何被实现错误改变，是不同问题。τ800的 `native_structure` 必须仍等于该请求的原float上游，不能从Q重建。
5. 全256统计原始float的V/N、最短晶格奇异值、condition number、共同0.5Å规则下最近距离、最终归一化z各轴和与初始z的变化；关联已有raw status/force/stress，保留逐请求表。它能定位是否集中在缩胞、扁胞或近邻冲突，**不能仅凭长尾就断言由某一步产生**。

**直接判别：** [raw label入口](../../scripts/label_programmed_paths.py) 优先读 `record.structure`，G raw评估不经proposal_graphs。若sample→native结构一致，则graph/refiner独有问题不能解释raw弱结果；若这一层已经变化，应先修输出实现并重跑同端评估。CIF/graph结果应分别报告，不能用graph正确替代raw结构核对。

## 3. 最小冻结模型诊断：只补现有记录缺少的量

与机制审计代理分工：对方主审H1/H2解释；此处给出最小数据合同，供根代理统一安排一次0 optimizer诊断，避免重复跑。

| 检查 | 固定输入及需额外保存的量 | 判别用途 |
|---|---|---|
| H1：实际条件下的去噪学习 | 先用相同100验证源、epoch0、`noise_stream="validation_band_b"` **在CPU重建同一targets**，得到每band zero-v/zero-u的target平方均值，与已存最终MSE比较。目标能量接近0的band不报不稳定比值。 | 原log足以与零头基准比较，不必先重跑100×6模型forward；低t无改善与高t u≈0须分开。单一高t不可预测误差不能叫“没学会”。 |
| H1补充冻结forward | 事前固定验证source keys（建议沿已有validation selection前16个），保留原C/S/P；六中心点加ε与1，保存逐源逐t的v/u prediction、target、mask/count、各轴SSE、norm与内积；v第6轴单列，报告零头readout与模型readout相对clean原CIF的误差。 | 区分未产生可用预测、符号／幅度失准、volume轴问题；直接验证部署两端时点。只用forward/no_grad，不能借此更新head或挑checkpoint。 |
| H2：实际已生成轨迹 | 事前取sample_idx0…7并补齐它们在40066中的**完整原logical batches**，使用原receipt/seed/FP64 prior、batch成员、BF16网络输入与同一最终权重重放原33NFE；先核对最终结构与原40066是否一致。保存每次的FP64 z/F/t、网络v/u、更新量，尤其pre-ε与第33次readout前后结构。 | 确定坏几何在先验、高／中／低t还是末端出现；记录每步V/N、最短距离和field/step范数，不用最终结构倒推未知中间态。 |
| H2有条件单一数值对照 | 只有原轨迹复现且出现显著步长／末步异常时，用同一prior、相同权重和readout做**一个**更细Euler网格对照，保留原33NFE结果与全请求，不同时改clock、epsilon或参数化。 | 细网格差异是局部数值敏感性证据，不是SUN收益。readout前/后结构是同轨迹诊断，不自动注册新生成端点。 |

原轨迹重放使用所有原batch成员很重要：只拿单行跑会改变神经batch形状，BF16差异与积分差异混在一起。额外检查的forward预算单列，既有40066不变；CPU目标重建与字段提取不需要任何模型或优化器。

## 4. 快速收口顺序

最先同步：两训练日志、sample CONFIG/SAMPLE_FINAL、sample float/Q paths、native/τ800 paths和ARTIFACT_FINAL、真实native CIF与proposal_graphs、各端labels/attempt_results。先输出实际数据表与身份差异；分时段基准和少量原轨迹trace再决定是否存在一个明确、可测试的实现或数值修复。若输出链完整、去噪／field在实际条件上已显著不足，或没有可归因的局部数值异常，就不以无证据的solver扫描消耗六小时；由根代理按最新授权决定后续路线。

T暂停状态：只留下未测试的 `src/crystal_dlm/mixed_token_sampling.py` 草稿，SHA256 `af492ff3f54dad0128f3a9b565452565d44a3377b4b594c3a755410f715d1807`；没有CLI、测试、采样、注册或提交，不能部署。此状态不影响既有G/Q产物。
