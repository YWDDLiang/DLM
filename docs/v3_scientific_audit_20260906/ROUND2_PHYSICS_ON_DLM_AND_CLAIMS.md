# 第二轮独立交叉审稿：概率主张、物理终态与 SUN

日期：2026-09-06。范围：交叉审查三份首轮报告的高风险结论，并用已保存的四端点产物复核物理含义。本轮未改变模型、冻结评测、标签或作业；未执行新生成、MLIP 推断或弛豫。

**裁决：三份报告关于条件链、固定 old、CE 能力边界和现有几何 conditioner 的主要论证成立。最需要补强的是“分布不变”与“物理近平衡”不是同一个 stationarity；且原 headline SUN 实际包含已被判定几何无效的弛豫终态。** 后者应单独补充统一的终态几何口径，不能只写为“verified 较少”，也不能据此宣称修 R 就能提高模型能力。

## 1. 阅读对象与可确认结论

本轮阅读快照如下；SHA 用于明确交叉审查对象，不锁定其他作者后续修订。

| 文件 | 阅读快照 SHA256 |
|---|---|
| [DLM_PROBABILITY_AND_KERNEL_AUDIT.md](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/DLM_PROBABILITY_AND_KERNEL_AUDIT.md) | `9a508d6e8c8fed4271dc45318644fcc2da32925a697d913e92309c361b46d407` |
| [CLAIM_AUDIT.md](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/CLAIM_AUDIT.md) | `f9758f56092357bbf9459b36fff99fc9d799ca25019dd704c85c37f543a21441` |
| [CRYSTAL_DIFFUSION_MECHANISMS.md](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/CRYSTAL_DIFFUSION_MECHANISMS.md) | `d94a30e209bfb650c8f2115b07d1d37ca0716f3bf1c6a7bc9c3c830ba84e1f35` |

| 受攻击主张 | 本轮裁决 | 不能扩大的结论 |
|---|---|---|
| 正确 corruption 与同一 joint 的 Bayes denoiser 配对可保持目标分布 | 确认。量化、拒绝和 fallback 本身不否定此证明，但必须包含在真实 C 中。 | 没有因此得到任意起点的有限步混合、能量单调性或 SUN 保证。 |
| 正样本 CE 能学会稳定结构分布 | 原则上确认，不能把能量负样本写成必要条件。 | 当前 field CE、合法率和有限 validation 尚非目标 joint KL 或新材料发现概率的上界。 |
| 固定 old 不符合连续 diffusion，所以必须在每个 scalar 后更新 | 否定这一必要性推断；三份报告现有修正正确。 | 保留 U 的后验分解不等于网络已高效利用 U；辅助几何特征是否有用仍待测。 |
| 当前只有几何 attention bias，未有几何进入 hidden/value | 被完整调用链反驳。已有 periodic pair aggregation、cell/site embeddings 和 global 广播。 | 已有功能不等于容量充分，也不证明再加多层 GNN 必定无用或有用。 |
| 补缓存、改 R 或放宽验收会让模型更好 | 不能成立。三者改变观察覆盖、后处理算子或判据。 | 系统产出可能改变，但不能把同一生成分布的重评涨分归给参数学习。 |

概率 lane 对真实 prefix 支持、噪声 metadata、温度与 rollback 的保留条件是必要的，不宜为了提出新小方法而删掉。机制 lane 已独立给出训练经验分布导致 novelty 为零的反例；本报告确认该反例，并进一步检查批量 U、弛豫算子和真实无效终态。[概率核前提](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/DLM_PROBABILITY_AND_KERNEL_AUDIT.md:186)、[机制 lane 交叉审查](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/CRYSTAL_DIFFUSION_MECHANISMS.md:568)

## 2. 最高优先级的新证据：原 SUN 包含已知无效终态

冻结 evaluator 以生成或 refiner 输出 y 的 N/U 与公共 R(y) 的能量阈值求交；headline S 不要求 `verified=true`。冻结 label 仍会保存几何检查失败终态的有限能量，并将其标为 `invalid_terminal`。因此这里是**已声明评价对象的物理解释不足，不是发现 evaluator 违反了代码合同**。[冻结 label 的检查与状态顺序](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/frozen/scripts/label_programmed_paths.py.txt:127)、[首轮实际链路](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/PHYSICS_PIPELINE_AND_STORY_AUDIT.md:32)

必须分开三个嵌套事件。下列定义保持原 hull、原 N/U 和全请求分母不变：

\[
H_i= N(y_i)U_i(y_{1:n})\,\mathbf1\{e(R(y_i))-h(c_i)\le t\},\qquad
G_i=H_i\,\mathbf1\{R(y_i)\text{通过终态几何检查}\},\qquad
V_i=H_i\,\mathbf1\{\texttt{terminal\_verified}_i\}.
\]

H 是原 headline SUN；G 是补充的终态几何通过 SUN；V 是冻结协议下完整 verified SUN。严格阈值为 `t=0`，Meta 阈值为 `t=0.1 eV/atom`；不可重构、能量/hull 缺失或非有限的请求不能命中，仍留在分母。三者都不是 DFT 稳定性证明，也没有将 N/U 改为 R 后计算。对当前四端点有 `V⊆G⊆H`。[冻结分类器](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/frozen/scripts/evaluate_programmed_paths.py.txt:30)

| 端点 | 全请求分母 | H：原 Strict / Meta SUN | H 中已知 invalid-terminal | G：终态几何通过 | V：完整 verified SUN |
|---|---:|---:|---:|---:|---:|
| K8 independent native | 1200 | 47 / 285 | 7 / 12 | 40 / 273 | 19 / 136 |
| K8 independent tau800 | 1200 | 79 / 570 | 0 / 0 | 79 / 570 | 36 / 224 |
| raw-v1 development native | 256 | 3 / 45 | 2 / 3 | 1 / 42 | 0 / 20 |
| raw-v1 development tau800 | 256 | 16 / 113 | 0 / 0 | 16 / 113 | 8 / 40 |

以上为本轮从[四端点状态交叉表](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence/TERMINAL_STATUS_SUN_CROSSTAB.json)独立求和；该文件保留各远端 `attempt_results.jsonl` 的路径和原字节 SHA。K8 表是完整 1200 请求，不能把这些数当作独立 conditional1000 数；raw-v1 为另一个 development256 cohort，不能横跨两种分母直接作配对因果比较。

这个分解改变了应当怎样描述失败：raw-v1 native 的 3 个原 Strict SUN 中，2 个终态已经几何无效；剩下 1 个仍未通过完整 verified。K8 native 也有 7 个这样的 Strict 命中。它们不是“已经获得稳定材料，只缺更严格力学证明”；低 surrogate 能量不能为已知不合格几何背书。tau800 消除了大多数终态几何失败，但仍有很多 F/S 未通过，两个问题不能合并。

**建议保留 H 的历史定义，增加所有方法共同的 G 列，并继续并列 V。** G 只需对现有终态作一致检查，不需要重生成或重弛豫。本轮四组保存的几何记录和状态一致，故表中 G 可由 H 减去 `invalid_terminal` 命中得到；一般工具不能仅用 `status!='invalid_terminal'` 代替显式几何通过标志，因为状态优先级可能把几何错误藏在能量错误之后，缺失几何也不能判为通过。reference 的同口径 G 未在这张表内，因此不能据此重新宣布胜负。

这里的 G 与将 N/U 全部迁移到 R 后是两项不同变更。若将来改成共同 R 后 N/U，应另登记、对全部方法重算；不要在加入几何门槛时默默一并更换 matcher、novelty reference、cohort 或分母。

## 3. FIRE 停止与物理验收：纠正本 lane 首轮措辞

首轮报告用过“停止/验证语义不一致”和“修 R 的合同”。更准确的结论是：**FIRE 的广义坐标收敛与公开的外部 Cartesian 力/应力验收是两个不同、已经明确实现的判据。** `optimizer_converged=true` 而外部 F/S 超阈值时，label 将其判为 `not_converged`，正是在履行现有合同，不能称为 FIRE 算错或代码违约。应该纠正的是论文把 optimizer 停止等同于所需物理近平衡、以及把该状态一概解释为步数耗尽的推断。[冻结验收谓词](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/frozen/scripts/label_programmed_paths.py.txt:153)

| 端点 | `not_converged` 数 | 其中 optimizer=true 且未到 500 步 | 其中仅应力失败、原子力通过 | 仅应力失败且 0 步停止 |
|---|---:|---:|---:|---:|
| K8 native | 504 | 399 | 295 | 0 |
| K8 tau800 | 763 | 762 | 712 | 19 |
| raw-v1 native | 105 | 77 | 59 | 0 |
| raw-v1 tau800 | 164 | 164 | 140 | 1 |

后两列限定在 optimizer=true 的记录内。K8 tau800 只有 1 个 `not_converged` 真正耗尽 500 步；另 1 个耗尽步数的 `invalid_terminal` 未混入这个分母。完整数据和逐例审计分别见[实际汇总](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/ACTUAL_SAVED_PHYSICS_AUDIT.json)、[独立解释复算](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/ACTUAL_AUDIT_INTERPRETATION.json)。

ASE Frechet filter 的 cell 梯度经过体积与 `exp_cell_factor` 缩放，原子广义力也涉及 deformation gradient，不能把单个广义 `fmax` 当作原始 Cartesian 力和 GPa 应力两个阈值。参考构型下默认因子 N 给出 cell 梯度量级 `(V/N)σ`；例如 `V/N=18 Å³`、`σ=.6 GPa` 的广义梯度数值约 `.067408`，可以低于 `.1`，而 `.6 GPa` 仍不通过外部 `.5 GPa`。这是不同坐标尺度下的优化范数，不能把这个数直接当作原子力的物理单位换算。[ASE 原始 filter 实现](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/authors/ase-3.28.0/ase/filters.py.txt:599)、[解析反例](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/CPU_PHYSICS_PROBES.json)

“增加迭代步数”必须明确增加哪一种迭代：

| 所指迭代 | 当前证据允许的结论 |
|---|---|
| 只提高相同 R 的 `max_steps`，停止谓词及起点均不变 | 不能解决已经主动停止的这些记录。对真正耗尽步数者仍未排除作用。 |
| 改变 R 的广义范数、外部联合停止谓词或 cell 缩放 | 会定义新的弛豫操作；可能改变 F/S、终态、能量、失败率和成本，收益未知。 |
| 增加 DLM 完整 repair 次数 | 上表没有测试它；概率 lane 的反复 D 反例否定普遍保证，不是否定所有实际增益。 |
| 增加训练更新或同一源的监督暴露 | 与 FIRE 记录无直接对应；仍需固定目标/状态及终点证据。 |

概率 stationarity 是 `pK=p`；物理近平衡是对所选势、应力条件和容差检查 F/S；严格局部极小还需排除负曲率方向，DFT/动态稳定又是其他命题。二者只是共享了一个词。C→D 的分布不变性不能补足 R 的 F/S 证据，F/S 通过也不能证明生成 kernel 采样了正确分布。

## 4. 把验收筛选连到旧 teacher，但不连错到 V2 CE

旧 K4/K8 的 A/B teacher 从同组成候选里选择 `verified=true` 且标签有限的条目。记候选路径为 \(x_{cj}\)，公共算子为 R，则可用集合为

\[
\mathcal I_c(R)=\{j:V_R(x_{cj})=1,\ e_0(x_{cj}),e_R(x_{cj})\text{有限}\},\qquad
u_{cj}(R)=\frac{\mathbf1\{j\in\mathcal I_c(R)\}}{|\mathcal I_c(R)|}.
\]

上式 u 仅在非空池上定义；空池不产生有效 teacher 参考。实现中 \(q_c\) 的两列标签实际为 \((A=e_0-e_R,\widetilde B=e_R)\)，没有以 hull 覆盖筛选 teacher。固定组成时减去 h(c) 不改变偏好差值，但不能把评价用 B 的 hull 需求偷偷加到 teacher 入池规则。[双目标实际入池代码](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/basin_path_objective.py:241)

\(q_c\) 在这个有限参考支持上优化双目标偏好。真实停止记录说明：外部验收可以排除 optimizer 已停止的候选。因此 verified pool 的大小同时依赖生成路径和 R/验收条件，不能把每条排除记录直接解释为生成分布中没有低能盆地；也不能将 FIRE=true 的条目绕过公开 F/S 条件直接恢复为可靠 teacher。[teacher 入口与协议检查](D:/codex_work/ai4s/DLM_periodic_self_repair/src/scripts/build_programmed_path_teacher.py:97)

但四端点数据是已保存的评测产物，**不是旧训练 teacher pool 的逐例停止交叉表**。本轮证明的是筛选机制与这些评测端点的真实发生频率，尚不能把“762/763”等比例外推到全部旧 teacher，或精确归因 498 个无 verified 条件中多少能由新 R 恢复。旧的 972 条 verified、526 个条件和有限池 A 下界，仍是原 R 下的真实有限池结论；不是全模型支持的下界。[旧 teacher 诊断的明确范围](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/27_SELF_IMPROVEMENT_REPAIR_PLAN.md:9)

若未来改变 R，\(\mathcal I_c,u_c,A_R,B_R,q_c\) 都可能改变；不只是把缺失标签补齐。新增 verified 候选可以提供有用目标，也可能处于较高能盆地或具有很大的原生 A，且统一参考权重和 KL 约束会一起变化。“更多 verified”不蕴含 teacher 的双目标更好，更不蕴含 student SUN 更好。必须重新冻结可比标签池、重算 teacher，再分别验证 student 对条件路径偏好的学习与完整生成效用。

**V2 当前 MP20 CE 不读这套 CHGNet verified pool。** V2 的 `legal_eligible` 来自 clean teacher prefix 的可达性与 repair old 的支持准入；这不是 R 后物理验收。FIRE 早停不能解释 V2 的 prefix 零 loss、某个 target 未计入监督或几何分支梯度问题。概率 lane 的 prefix 过滤反例攻击的是后验目标/支持定义，不能拿它与上述物理标签筛选互为因果证据。[V2 实际 eligibility](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_v2_training_data.py:268)

## 5. stationarity、CE 与 SUN：最强正反边界

### 5.1 接受正确的 Bayes 核证明，不接受发现保证

固定完整条件 c，若 \(J(x,u)=p_*(x\mid c)C(u\mid x,c)\)，D 为同一 J 的 Bayes posterior，则

\[
p_*(x\mid c)K(x'\mid x,c)
=\sum_u\frac{J(x,u)J(x',u)}{J(u)}
=p_*(x'\mid c)K(x\mid x',c).
\]

求和只在 \(J(u)>0\) 的可定义观测上进行。这证明 detailed balance 和不变性，既不要求高斯 C，也不要求多档 C 构成 semigroup。相反，仅凭 CE 的 properness，不能跳过真实噪声/标签、支持截断、温度、完整 posterior 因子和 rollback 是否形成这个 K 的检查。

即使前提全部满足，以下反例仍成立：

1. **经验支持与 novelty。** 若 \(p_*\) 仅支持 novelty reference 内的训练晶体，正确 D 仍只输出这些晶体。它们可以全部低能、通过物理验收，CE 拟合和 detailed balance 都可以精确，但 novelty 与 SUN 为零。这确认机制 lane 已给出的反例；要谈泛化，必须另外论证 population 目标和未见条件，不可把经验不变性代替发现能力。
2. **稳定且新颖的边缘分布也不保证 U。** 设固定组成只有 K 个互不匹配、对 reference 新颖且稳定的模式，模型已精确均匀采样它们。n 个 IID 样本的期望唯一代表数为 \(K[1-(1-1/K)^n]\)，所以期望 SUN 比率为该数除以 n。取 K=20、n=1200，仅约 1.667%，尽管每个边缘样本都稳定且新颖。若样本之间相关，仅有边缘 stationarity 对 U 的约束更弱。这是 toy，不是对实际候选模式数的估计。
3. **不变性不提供有限步混合。** C 为 identity 时，其正确 D 与 K 也为 identity；任意 p 都可不变，但坏起点不会改善。将 identity 换成极小正概率移动，可得到不可约却任意慢的链。一个明确的概率对象值得诊断，不因此获得在给定调用预算下优于一次 D 的理论排名。

因此 C→D 可以作为核语义清楚的对照候选；现有证据不支持把它叫作提高 SUN 的最小充分修复。本轮不执行该候选。

### 5.2 CE 不是原则障碍，但现有风险不是所需的 joint 保证

若能得到同一可生成支持上真正的 \(\mathrm{KL}(p_*\|q_\theta)\le\epsilon\)，则对固定的单样本稳定事件 S，有

\[
q_\theta(S)\ge p_*(S)-\sqrt{\epsilon/2}.
\]

所以“正样本 CE 永远不能保证稳定性”错误；负样本或能量梯度不是必要条件。但当前平均 field CE、不同任务混合风险、部分位置 eligibility 和 teacher 的经验 KL 都不是上述部署 joint KL 的证明。若需约束批量 U，必须对相应联合采样分布建立界；IID 时产品分布 KL 还会随样本数累加，不能原样沿用单样本界。[概率 lane 已保留的正面边界](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/DLM_PROBABILITY_AND_KERNEL_AUDIT.md:327)

prefix admission 的反例是原则上有效的：局部合法前缀的计分不自动等于完整合法 joint 的条件化；但实际严重性仍取决于 clean 全路径冲突率、old 准入与部署可达状态。不能用一个 toy 宣布真实 V2 的主要失败原因已经锁定。反过来，保证语法/距离底线也没有把支持收缩到低能域，更不构成 SUN 保证。

## 6. fixed old、soft Plan 和 GNN：应保留的纠偏

对修复终点 Y 和完整 noisy old U，\(p(Y\mid U,c,P)=\prod_jp(Y_{o_j}\mid U,c,P,Y_{o_{<j}})\) 是正常条件分解。U 是观测，prefix 是逐渐揭示的终点；每个 scalar 后覆盖 U 会改变条件对象。保留 U、另加 \(H_j=h(U,Y_{o_{<j}})\) 的 working geometry，可以改善有限网络计算，但 H 不增加给定原观测的信息，也没有保证 hybrid 几何更自然、能量更低或保持某一噪声时刻。[机制 lane 的三种时钟](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/CRYSTAL_DIFFUSION_MECHANISMS.md:264)

已有 state conditioner 的 periodic pair features→neighbor pooling→global cell/site update→hidden 路径，能把 known-site 环境广播给 active site；新增 bias 的某行是零，不能推出 active hidden 不含几何。初始 lattice scalar 尚无完整几何可见，这一局限成立；网络是否将后期几何用在正确预测上则没有由非零梯度证明。新增 GNN 必须回答相对于已有一跳聚合、多层 LM 和全局通路增加了什么可学习对象，并以效果作结；“缺 GNN”不是当前已经证实的单一根因。[实际已有功能](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/CRYSTAL_DIFFUSION_MECHANISMS.md:404)

CLAIM_AUDIT 对 soft Plan 的条件独立论证也有必要边界：理论 C 必须包含 Planner 实际使用的全部相关 metadata，且要与 DLM 真正可见字段区分。若 Planner 见到而 DLM 未见到某个 M，S/P 可成为信息渠道；确定性或信息冗余的特征也可能帮助有限模型计算。反过来，soft annotation agreement 不是生成后的 plan adherence，更不是 SUN 的因果贡献。新加 GNN、重排 P 或增加 repair 次数都不能自动修复训练中 requested S 与目标几何的语义关系。[软条件论证](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/CLAIM_AUDIT.md:25)

## 7. 最直接的因果反例：R 改善可使 SUN 上升、A 变坏，而模型完全不变

固定同一生成结构 y、其原生能量 \(e_0=-4\) 和同组成 hull \(h=-5\)，并固定原 N/U=true。以下数值均为 eV/atom 的代数 toy：

| 公共后处理 | eR | A=e0−eR | B=eR−h | 当前能量阈值 SUN |
|---|---:|---:|---:|---|
| 原 R | −4.85 | 0.85 | 0.15 | Strict/Meta 均不命中 |
| 另一 R′ | −5.05 | 1.05 | −0.05 | Strict/Meta 均命中 |

同一个 y 的 B 降低 .2，A 升高 .2，headline SUN 可由否变是，\(p_\theta\) 和原生结构却没有发生任何变化。只要现行 pre-R N/U 固定、能量有限且 \(e_{R'}\le e_R\)，这种 hybrid 稳定事件可以弱单调增加；这只是指定后处理和计分定义的结论，不是对实际更严格 R 的单调能量保证。

若 R′ 使很多 y 落到同一已知相，R 后计算的 N/U 还可能下降；若新 R 失败或换了可用分母，连上述条件化关系也不能直接套用。故“先修停止，再宣称原生生成稳定性提升”属于因果归属错误。若将来新 R 的终态进入训练，只有随后的参数更新与冻结公共评测下的完整生成验证，才可能建立模型学习收益；不能从标签更好直接跨过这一层。

G 门槛与修改 R 的方向也不同：固定既有能量和几何，加入 G 只会保持或降低原 H 计数；改变 R 则可能同时改变能量、几何、验证和后处理 N/U。不要将两者都称为一次“评测修复涨分”。

## 8. 仍未决的问题与本轮可交付结论

| 优先级 | 已确认 | 尚未确认 | 最小下一证据 |
|---|---|---|---|
| 高 | 原 headline 含已知无效终态，四端点数量已复算 | 所有 reference/历史 arm 的同口径 G 排名 | 从已保存终态和 attempt records 统一补显式几何通过与 H/G/V；保持分母、cache 和 matcher。 |
| 高 | 当前评测中很多 R 已主动停止但外部 F/S 未通过 | 旧训练 teacher 中同类筛选频率；改变停止是否改善实际终态 | 先对旧训练 labels 作相同只读停止交叉表；新 R 属于另行定义的后续候选。 |
| 高 | C→D 的理想证明有明确前提；CE 没有原则性失败定理 | 冻结 V2 的实际 posterior 误差、支持冲突及完整生成收益 | 汇总既定训练/评测记录，按实际 eligibility、corruption、任务和 source 对齐。 |
| 中 | 当前已有几何 hidden/global 传播与真实程序执行 | 几何特征、S 语义、P 路由各自的 SUN 因果贡献 | 先核实已有影响/遵从性记录；缺失时保留未知，不用梯度或 annotation agreement 代替。 |
| 已降级 | 有限 radius-2 MIC 的普遍数学保证不成立 | 它是否解释本批失败 | 本批 5,788 个可用原/终态记录已完成阈值 MIC 复核，无漏掉 `<.5 Å` 的记录；不能继续列为本批已知主因。 |

另外，873 条原 verified 标签各有三种已保存 fresh representation 分数，只有 raw-v1 native 一条平移表示的力从阈值内轻微跨到 `.100036 eV/Å`，能量跨度约 `9.54×10⁻⁷ eV/atom`。这支持记录边界敏感性，不支持“广泛数值记账错误导致低 SUN”。本轮只读取这些已存分数，没有重新执行 CHGNet。[实际数值与 MIC 审计](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/ACTUAL_AUDIT_INTERPRETATION.json)

本轮交付的新增结论是 H/G/V 真实计数、停止与验收合同的措辞纠正、旧物理 teacher 与 V2 support CE 的明确分界，以及不变性/批量唯一性/后处理归因反例。没有证据要求立刻再加 GNN、替换 fixed old、盲目增加 DLM 或 R 迭代，也没有证据允许宣布这些候选必定无效。V3 选择应等待既定证据并对应具体被证实的失败对象。

### 关键输入收据

| 存量证据文件 | SHA256 |
|---|---|
| [TERMINAL_STATUS_SUN_CROSSTAB.json](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence/TERMINAL_STATUS_SUN_CROSSTAB.json) | `c1d3221505226622219888afa6d9beb9af3d36dc9f1499875f21e3d8df304c7a` |
| [ACTUAL_SAVED_PHYSICS_AUDIT.json](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/ACTUAL_SAVED_PHYSICS_AUDIT.json) | `de94c8744cc92d31b6ce68f83a1f389412b73152cdb2908e1ec67753125b034f` |
| [ACTUAL_AUDIT_INTERPRETATION.json](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/ACTUAL_AUDIT_INTERPRETATION.json) | `ca23834a1f1a885a2a6850dfd6989ce18e86ee00f2ce38513668c3a221d4eb97` |
