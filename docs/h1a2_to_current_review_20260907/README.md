# H1-A2 到当前：为什么改了很多，SUN没有持续上升

2026-09-07。此次复盘查了旧项目和当前仓库的实际终态、历史源码及训练记录，建立了[57行、15份来源胶囊的结果台账](HISTORICAL_RESULT_LEDGER.md)。没有重新训练、生成或运行MLIP；R03那组达到10/50的结果还重新逐行核对了512条原始attempt及配对计数。

结论是：**历史上确实出现过10/50，但H1-A2到现在并不是同一条强基线上的连续增量实验。** 这其中既有真实的学习瓶颈，也有权重重置、条件信息变化、精修后的收益消失，以及结果视图混用。仅围绕最近K4/K8再试几个采样小改动，覆盖不了这些问题。07:00报告中“保留K4”的判断适用于最近的固定开发cohort，不能扩展成全项目历史冠军。

下面几组数最重要；每行的分母和来源都单独保留，不能按时间直接排名。

| 结果身份 | Strict / Meta SUN | 能说明什么 |
|---|---:|---|
| H1-A2历史frozen1000，旧协议重算 | 94/474，即9.40%/47.40% | 有实际可追踪的旧强基线；这个1000来自历史成功前缀 |
| 同一历史1000，fresh official视图 | 96/478，即9.60%/47.80% | 只是同结构的评价视图变化，没有重新生成 |
| R03归档256、官方hull完成 | **28/128，即10.94%/50.00%** | 确有达到目标的单cohort点；同组H1-A2为24/118 |
| 同一exact Plan1200配对 | H1-A2 **103/553**；R03 **101/559** | 较大匹配实验没有保持R03的Strict优势；known-both两项McNemar均p=1 |
| R03同一Plan256的4次CUDA实现，clean cache | 120/500，分母1024，即11.72%/48.83% | Strict较高，但Meta仍低于50%；不是4个独立Planner样本 |
| 当前原K4完整循环，开发256 | 19/126，即7.42%/49.22% | 当前分支的保留基线，未达目标 |
| 当前K8独立条件全部1200 | 79/570，即6.58%/47.50% | 与旧cohort不是同条件配对；不能把差额全归因某个模块 |

R03归档点的原始terminal SHA为`63128c86…f85b0`；逐行复算得到Strict控制独有2、R03独有6，Meta控制独有14、R03独有24，与原terminal完全一致。相应p为0.289/0.143。它是真实的历史点，但不是一个已在独立大样本上证明的普遍提升。[原terminal胶囊](evidence/h1a2_r03_archive256_terminal.json)、[完整账本及JSON字段指针](HISTORICAL_RESULT_LEDGER.json)。

## 1. 并非“同一个DLM从H1A2一直训练到现在”

- H1-A2的epoch2是**Planner两轮**，body仍是May29 R5-C/B0；不能把它算作DLM的第二轮。
- 38703 native SFT、39282 canonical、raw-v1、periodic V2分别有fresh初始化分支。K4/K8属于39282之后的SPAD链；V3则续V2。旧R5-C、grounding和K4/K8的权重能力不会自动叠加。
- 这些重构有具体动机。39282前的C3FD-native/Compact数据，teacher site顺序与该代hard-prefill一致的只有4069/27136；重排修复是真实需要。这个比例不属于本次对May29 R5-C的逐行诊断，不能外推成H1A2本身一定有同一错误。
- 缺少同数据、同条件、同预算的fresh-versus-warm-start实验。因此可以确认发生了重置，**不能把reset直接定为全部退化的原因**。grounding和SGTC自己没有已证实的稳定性增益，也不能称这些成果后来被“丢失”。

[完整训练、条件与父checkpoint谱系](TRAINING_AND_CONDITION_LINEAGE.md)区分实际终态证明、合同/代码支持与尚缺的原始记录。

## 2. 学习目标经常改善了代理量，没有改善最终稳定性

| 同实验内的改动 | 实际改善 | 最终结果 |
|---|---|---|
| ordinary CE：DLM总epoch2→3 | body985→992，Direct871→878 | SUN **81/489→79/477**，各1000 |
| counterfactual rich-field grounding | true/counterfactual Plan的可辨别性变强 | SUN **89/487→86/467**，各1000 |
| SGTC strict-positive几何训练 | 正例训练/格式与几何支持成立 | base **60/412**，strict arm **53/417**，各1000 |
| C3FD-v2.5组成生成 | 两seed合计composition-valid **1724/2000→2000/2000** | 这是组成端真实收益，该实验没有测结构SUN，不能当作稳定性提升 |

所以，不能继续用“再训一轮”“loss更低”“格式合法”“条件模块有梯度”推导SUN会增加。另一方面，K4/K8确实已经超出positive-only CE：它们使用经终态验证的同组成物理路径teacher，并有CE anchor，不能再笼统说“从来没训练过物理价值”。

但其有效对比信号有限：K4的972条verified路径只覆盖526/1024个组成，其中273组有多候选，经验teacher相对均匀分布的平均总变差只有2.5613%；K8扩至1843条verified/662组。这里的“两passes”不是两轮完整MP20能量训练，teacher端双目标改善也不保证学生实际rollout改善。V2/V3随后回到原MP20 teacher，又没有继承这批路径偏好。

## 3. 同一model494既大幅救回结构，也会压缩DLM之间的差距

老L6同512条件的记录很有判别力：

| 条件 | refiner前 Strict/Meta | model494后 Strict/Meta |
|---|---:|---:|
| full-axis rich条件 | 10/66 | 48/230 |
| hard-axis条件 | 2/7 | 47/230 |

这是**净SUN差距**显著缩小的实测，不代表两臂逐个结构都相同，也不证明所有DLM改进都会被抹掉。它说明最终tau800不能作为DLM自身学习是否有效的唯一读数。另一个同panel校准中，tau0/200/500/800分别是10/66、29/171、39/222、48/230；简单缩短精修保留了更多新颖性，却损失了稳定性，不能作为已知解法。[原始L6转换](evidence/refiner_conversion_l6.json)、[tau校准](evidence/tau_calibration_l6.json)。

当前K4/K8的A/B价值对应完整DLM路径的raw→R及R终态；固定model494又位于另一个最终评价分支。若下一轮主要追tau800结果，应明确研究包含tau800的系统终点价值。若主要追native，则直接承认并优化native的瓶颈，不能把原生收益当作精修终点的保证。

## 4. 有些高分和失败标签需要恢复正确身份

- Shared-Plan/S2的10.55%/67.19%来自held-out **gold/oracle** 条件256诊断，是wrapper-null与matched的post-hoc比较；不是H1A2 de novo，也不是直接对untouched parent的SUN比较。修复null路径后的H1A2 same-draft F10/F11仍都是15/256 Strict，没有证明可直接迁移的de novo优势。
- 原S2/gold执行源码没有`force_null -> base(...)`快捷分支；它在7月27日的`16e6401…`才加入。旧代码已有代数mask，但调用路径不同，独立CUDA运行本身也不逐位确定。因此，原完整轨迹byte门不能单独证明adapter污染；也不能把旧失败完全解释为parent数值波动。[源码与机制核对](REFINER_AND_GEOMETRY_LINEAGE.md)。
- R03本身是**DLM reveal-order改动**，不是Shared-Plan residual。它正式比较已用refiner batch1；没有找到batch128↔1纯因子同输入对照，不能拿batch变化解释其下降。
- SI-LWA目前只确认数据准备；D3PO启动过训练，但关键训练/生成/正式评价终态仍未核实。它们不是可直接归类的完整科学失败。
- Potential-Closure stream17记录了**17/115→20/125**的tau800方向。原生能量差均值−0.236 eV/atom，但区间[−0.599,+0.131]跨零，故没有扩大。应保留为有方向但未确认的物理训练分支，而非直接当成无效或冠军。[同期执行工作记录](evidence/potential_closure_report.md)。

## 5. 数字与口径造成的假象也确实存在

H1-A2旧报告的“raw SUN”常指未经coverage调整的比例，结构本身已经经过tau800；它不是现在的refiner前raw端点。同一exact1200的H1-A2首1000成功前缀视图为94/479，而全1200为103/553，这是同次生成的不同统计对象。[原始终态](evidence/h1a2_exact1200_terminal.json)。

资料里反复保留的105/1000、488/1000被其自己的审计文件称为aggregate headline。本次没有找到将这两个整数绑定到同一1000条原始记录的来源，因此没有把它加入“实际1000实验”表，也没有构造对应microdata。[该历史声明](evidence/aggregate_headline_boundary.md)。

Strict≤0、Meta≤0.1以及核心N/U源码没有随本轮任意改变：本次从远端取回的旧`eval_sun.py` SHA仍是当前N/U冻结源的`564b4490…9852b`。旧R使用库默认`StructOptimizer.relax`与缓存；当前R显式固定参数、记录终态和周期表示能量一致性。当前headline仍不以verified标记为必要条件，**不能用“增加verified检查”解释headline下降**。评价差异的数值贡献尚须同一结构的R桥接实验，不能从跨cohort曲线倒推。[旧评价源码](evidence/legacy_eval_sun.py.txt:186)、[旧可恢复R源码](evidence/legacy_eval_sun_resumable.py.txt:121)。

## 下一轮应怎样收拢

[基线登记](BASELINE_REGISTRY.json)已区分四个角色。项目级保护锚点应同时保留**H1-A2/B0、R03/D2**，当前K4继续作当前分支incumbent；Potential-Closure列为待确认分支。旧的目标点存在，但不能据此直接替换当前方法并声称解决了新cohort。

最先补的是[已具体化的同结构评价桥接与基线对照](NEXT_REPLAY_SPEC.md)：先确认旧强输出在当前R/记账下的行为，再恢复同台控制。后续只改变一个学习因素，验收学生实际rollout的物理价值和最终SUN；同时记录多候选verified覆盖和精修保留率。不要再把未完成路线当失败、把代理量当稳定性，或把新架构名字当作旧能力的延续。

这次复盘没有给出“某一个原因已解释全部退化”的虚假结论。已确认的是谱系断点、多个同实验负结果和精修后的差距缩小；reset、条件分布、R实现各自解释多少，仍缺严格交叉对照。
