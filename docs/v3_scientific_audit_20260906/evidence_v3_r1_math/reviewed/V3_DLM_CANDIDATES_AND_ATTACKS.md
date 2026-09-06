# V3 的三个候选科学对象及其最强反驳

2026-09-06。状态：**设计与反驳；没有采用、实现或执行任何候选。** 本文只新增本文件及 [独立 CPU 小证明](v3_candidate_cpu_checks.py)、[结果](v3_candidate_cpu_checks.json)，不改 V2、评估、模型、训练数据或其他报告。本文的建议不是新 GPU 作业的启动指令。

## 1. 先用已完成的证据约束设计

当前 V2 从原始预训练 LLaDA 新建适配器，冻结约 8.035B 原参数，训练 36,993,232 个适配参数。2 epoch 共 108,544 个真实状态、4,524 次更新，耗时记录为 4,096.53 秒。所有 source 每个 view 覆盖两次；corruption fallback 为 **0**，support-conflict 文件为 **0**，empty-supervision 状态为 **82**。因此，首轮的选择性 prefix 截断反例仍然推翻无条件 posterior 保证，但**没有证据表明本次训练触发了该问题**；不把它排在 V3 首位。来源：[最终训练与 metadata 证据](evidence/V2_FINAL_TRAIN_AND_METADATA.json)。

同一份证据中，prepared 数据 pooled 的 29,249 个 exact-composition 分组有 2,144 个多行组，但没有多个实际 chemical-M signature 的组。当前有限表没有提供“相同硬组成下，隐藏 M 传入不同几何信息”的证据。不能把 reduced formula 跨不同 N 的变化混进此结论，也不能据有限表证明所有未来 composition/witness 的 M 都唯一。冻结 LLM 的特征和程序仍可能有有限容量的计算价值。

固定开发 256 的原协议结果已出：

| 模型 | Native Strict / Meta SUN | 固定 tau800 Strict / Meta SUN |
|---|---:|---:|
| 开发参考 | 6 / 55 | 19 / 121 |
| raw-v1 | 3 / 45 | 16 / 113 |
| V2 final | **8 / 50** | **16 / 113** |

V2 native verified 为 70；tau verified 为 109。tau 的 strict/meta stable 为 25/148，较 raw-v1 各多 2，但 novel-unique 从 219 降到 217，SUN 同计数。**预测 Plan、exact CE、log-SPD corruption 与 GEM 的组合没有提高 refined SUN。** Native strict 的小幅点估计不提供某一模块因果作用，也没有达到开发门槛 26/128。较早 [39998 结果快照](evidence/V2_EVALUATION_SUMMARIES_39998.json) 的顶层 `success=false` 是当时后续诊断未结束；随后完整 eval 已 `_SUCCESS`、无 `_FAILED`，不是两端评估失败。

追加完整 construction/repair 证据：[V2_CONSTRUCTION_AND_REPAIR_39998.json](evidence/V2_CONSTRUCTION_AND_REPAIR_39998.json)。同一 256 个请求，construction 的 Strict/Meta SUN 是 **12/44**，一次 full-cell repair 后是 **8/50**；NU 均为 251，verified 从 68 到 70，strict/meta stable 从 14/48 到 11/54。这是 **Strict −4、Meta +6 的权衡**，不是统一改善。paired Strict SUN 为保留 4、失去 8、新增 4；Meta 为保留 18、失去 26、新增 32。没有运行 construction→tau800 支路，不能由 repair→tau 结果倒推它。

物理配对进一步限制了“repair 改善”的解读：全部 256 的 raw E 平均变化为 **−0.17806 eV/atom**，但中位变化 **+0.19127**，117 个下降、137 个上升、2 个相等；raw 最大应力均值下降 143.49 GPa，中位变化却为 +0.09256 GPa。共同 verified 仅 **33** 个，另有 35 个失去、37 个新获 verified；不能拿 68→70 的净数代表原样本普遍变好。verified Strict/Meta SUN 更从 **4/21 降到 2/18**。这些统计表明作用异质，不应把均值下降直接外推成多数盆地改善。

headline SUN 使用冻结的 terminal-energy 标准，和 verified SUN 不同；已知终态无效但能量通过的个例使其物理解释受限。候选评估必须同时呈现原 headline、终态几何有效子集、完整 verified 子集；修评估准则不能算模型提点。详见 [物理交叉审计](ROUND2_PHYSICS_ON_DLM_AND_CLAIMS.md)。

## 2. 三个候选，不叠成一个大模块包

| 候选 | 要学习或改变的对象 | 最小变化 | 主要科学风险 | 当前判断 |
|---|---|---|---|---|
| A：条件与目标一致的几何 Plan | 可识别的条件联合分布 `p(G,S|c)` | 目标属性定义、Plan 训练关系及其程序 | 原有 soft 标签在 codec 下不保真；部署先验请求了无数据支持的条件 | 若要主张 LLM 几何控制，这是最直接的候选；先核属性可识别性 |
| B：同核 corruption→denoising | 已有 V2 的执行 kernel | 同三档 C 后接一次现 D，最多讨论两次 outer | 生成错误不服从训练 posterior；stationarity 不代表新颖或低能 | 最小、可立即形成对照的候选；等待 construction/repair 净作用 |
| C：共享主干的完整几何反向过程 | 连续 lattice + torus 的噪声分布和反向生成 | 完整 noisy geometry 输入、dense score/epsilon head、真实 reverse update | 改变原生 token 生成主张；连续目标、先验、量化和小预算可学性 | 比为 ELBO 引入全新离散 chart 更直接；是有条件的 mixed-space 研究路线 |

A、B、C 是互斥的首个实验选择。不能把 A 的标签校正、B 的多轮、C 的 head 和评估修订一次叠上，然后宣称找到了根因。本文在第 6 节完整比较未入选的全离散 Q 路线；拒绝它作为默认第三候选的理由是表示迁移与有限预算，不是“离散一定学不好”。

记号：`c` 是 DLM 的实际硬上下文（N、精确 species/counts、anion family）；`m` 是冻结 Planner 的实际 typed chemical 输入，可能含 goal。`S` 是三 soft 字段，`P` 是冻结 LLM/pointer 编译的程序。`G` 指现有量化晶体，`d=6+3N` 是可生成几何 scalar 数。所有候选保留 C³FD 的可达组成；不把 model494 当作原生能力，不使用 K4/K8 labels。

## 3. 候选 A：把 requested Plan 与预测信念分开

### 3.1 概率对象与为何不同于 V2

定义一个冻结的、可重算的属性函数

\[
a_\delta(G)=(\text{lattice-system},\text{SG-bucket},\text{VPA-bin}) .
\]

这里的输入必须是 **DLM 实际生成并交给评估的量化结构**。`delta` 包括固定的对称性分析容差、cell setting、原点/规范化规则、bucket 映射和 VPA 边界。全部从已有训练定义确定并在结果前冻结，不用开发 SUN 调容差。SG 未知/数值分析失败应有明确的 unknown 语义；不能丢弃这些 source。如果现有词表无法表达这种情况，需先修改属性契约，不能默默归到 P1。

把训练联合分布改为

\[
q_A(G,S,P\mid c,m)
=q_{\rm data}(G\mid c,m)\,
\mathbf1\{S=a_\delta(G)\}\,
\rho_{\rm frozen}(P\mid c,m,S).
\]

`rho` 表示冻结 pointer 的**实际**解码规则及 seed；若它是 greedy，退化为 delta。缺少可用 typed m 的原 source 必须保留：A 可将该层 `rho` 明确设为硬 species 的 canonical order，并单列覆盖率，不能重用从 target contact geometry 得到的顺序冒充 frozen prediction。这个 fallback 与当前 V2 的实现差别须进入实验合同。部署联合模型为

\[
p_A(G,S,P\mid c,m)
=\pi_{\rm LLM}(S\mid c,m)\,
\rho_{\rm frozen}(P\mid c,m,S)\,
p_{\theta,A}(G\mid c,S,P).
\]

LLM 提供请求 S 的先验；DLM 学习给定请求的结构条件分布。当前 V2 则抽预测 S 后仍配原样 G；在独立抽样的理想模型下会有 `G independent of S | (c,m)`。A 改变的是可识别的联合分布，而非又多一个 conditioner。CPU 二状态示例中，相同边缘下独立标签的 MI 为 0，目标一致标签为 `log(2)`；这只说明可识别性差异，不是实际 MI 测量。

**不是直接恢复旧 teacher Plan 就已经完成 A。** 原连续 CIF 的 SG 注释经过 0.01 fractional / 0.1 Å / 1° codec 后可能改变；先检查原注释与 `a_delta(G)` 的混淆表、unknown 数和可用类别。若大多数高对称目标在冻结容差下变成 P1，A 的三字段可控性设想就缺少表示基础。精确 SG 编译还需 Wyckoff/free-coordinate 表示、群与量化可交换性，本候选没有这项保证。

### 3.2 表示、corruption、posterior、采样与 loss

- 表示继承 V2 的 `7+4N`、硬 N/E 和现有数值词表。周期 100/0 仍按现有 `logaddexp` 合并后除 T=0.7，规范输出为物理类别 0；不重定义 alias 测度。
- Corruption 继承 V2 的三档 `(sigma_v,sigma_shape,sigma_F[A])`，先完整几何扰动、量化、现支持验收，至多 8 次后明确 fallback；construction/repair 两 view、prefix/dense 采样和 noise-metadata dropout 都保留。
- DLM 的 repair 目标是 `q_A(G | U,c,S,P)` 的逐 scalar teacher conditionals；输入仍有真正整数 old U 和当前可见 prefix，绝不输入未量化目标几何。known 三分量或 unknown 混合标签保持训练/采样一致。
- 损失继承 exact CE 与对象权重；prefix 只对实际合法支持归一化，dense 按真实 inclusion probability 加权。不会把这个风险宣称为最终 native marginal 的 ELBO。
- 采样仍为程序 construction 后一次 full-cell repair。包含支持拒绝、回滚及失败状态的 `p_theta,A` 才是实际 native 分布，单 trace 的 action log-prob 不等于积分后的结构概率。

合法支持仍只保证已有结构底线；S 是可学习请求，**不**作为硬 SG/能量筛选。`a_delta(G_out) != S` 是一次不服从的输出，照样保留在全请求分母，不重抽 Plan 或生成候选补位。

### 3.3 学什么、费用及最强反例

为保持可归因性，A 若采用，应从同一原始 LLaDA、同一新适配初始化协议开始，只改变训练 Plan 与实际 P；不偷偷 warm-start V2 后称为同一 2-epoch 对照。可训练规模约为 V2 的 36.99M，完整 27,136 source、两个 view、1/2 epoch 对应 2,262/4,524 global-24 更新。新增的是离线属性重算与一次 frozen pointer 编译；DLM 推理 nominal NFE 仍 `2d`。

**最强反例：** 即使 `p_theta(G|c,S)` 完美，LLM 的 `pi(S|c,m)` 也可能集中在不稳定、novelty 低或当前数据没有的 S。若 pi 给互相矛盾的字段组合正概率，而 `a_delta` 根本没有这样的原像，该条件没有可学习的真实结构后验；必须把不满足/失败计入请求分母，不能用拒绝重抽隐藏 prior 的问题。A 修复控制语义，不保证边缘 SUN。相同 c 下稀有 polymorph 太少时，条件模型可记忆，prompt 服从性也可能以减少多样性为代价。

如果必须主要使用 original annotation 而无法在实际输出上定义 `a_delta`，则论文只能说“按原注释类型条件化”，不能承诺量化后的 SG 服从。这个限定本身可能使 A 不值得实施。

**最小证伪顺序：** 先用现有 V2 做固定 source 的 S/P 分离干预，分别记录 target CE、实际输出属性与 raw/tau SUN。交换 S 时必须报告是保持 P 还是重编 P，避免把程序顺序效应当成语义控制。若 target-aligned 标签无可靠定义，或现有预测 Plan 已对真实属性呈现稳健控制，A 的优先级下降。真正采用后只比较 A 与 V2 固定终点；按 direct/fallback、重复组成/近唯一组成分层，保留全部失败。

## 4. 候选 B：训练同核 C_mix→D 的最小执行校准

### 4.1 它改变哪一个 kernel

使用**同一个 V2 final checkpoint**，冻结 Plan、P、T=0.7 和采样支持。`B_theta` 为 construction 分布，`C_mix(U|G)` 为 V2 实际三档均匀混合、量化、支持重试和 clean fallback 组成的完整 forward kernel，`D_theta` 为 unknown-noise 通道的程序 full-cell repair。

理想形式为

\[
K_B(G'\mid G)=\sum_U C_{\rm mix}(U\mid G)D_\theta(G'\mid U),
\qquad p_B=B_\theta K_B^H .
\]

第一个候选固定 **H=1**，替换现有“一次 D”；H=2 只作为成本与理论比较，不能见到结果后追加到满意为止。`G_0` 是本次实际 construction，**不是**原始训练 G 或一个新 broad-noise prior。这是局部执行校准，不是从简单先验开始的 diffusion generator。

执行上需要显式外层事务：如果 inner repair 失败，返回加噪前 G，而不是把 U 留作“成功修复”。若 `D_ok(.|U)` 是成功返回的 subprobability，`r(U)=1-sum D_ok`，实际核为

\[
\widetilde K_B(G'\mid G)
=\sum_U C_{\rm mix}(U\mid G)
\left[D_{\rm ok}(G'\mid U)+r(U)\mathbf1\{G'=G\}\right].
\]

合法但 unchanged 的成功路径不能被误认成 failure；腐蚀 exhaustion 的 clean fallback 也须保留原标签与路径。这个外层回滚核规范化、保留原有效起点，但一般不再是理想 Gibbs kernel。forward 的完整量化/重试密度未积分时，仅保存实际 noise、seed、提议、接受和 rollback trace，不能伪造其精确 log-likelihood。

### 4.2 噪声、条件与实际支持

使用现代码三档 `(0.03,0.01,0.05 Å)`、`(0.10,0.03,0.15 Å)`、`(0.20,0.06,0.30 Å)`，前两项分别是 log-volume 与正交无迹 half-log-metric coefficient 标准差，第三项相对仿射变形参考的 Cartesian displacement 标准差。不是未变形 clean Cartesian 总位移。用任意 Gaussian 代替这条有量化及重试的核，都不是“matched C”。

现 runtime 调 unknown 通道，正好对应训练三档 mixture 的边缘 posterior；**不能只选最低档却仍以同一 stationarity 证明作担保**。想用 known low sigma，需要增加真实三分量接口并核对训练条件，不能用 mask ratio 假冒几何误差。三档是固定设计先验，从来不是已拟合的真实生成错误分布。

旧几何在每次 inner transaction 内仍固定，下一 outer 才重取 G。current prefix 和 legal masks 用新提交值，old 条件仍用该次 U；这个语义虽然合法，但不等于每个 scalar 后重算当前 score。N/E、原 codec、alias、几何支持、联合验收全部继承；没有新 head、loss 或可训练参数。

### 4.3 理论动机的极限，以及为何不称 broad matched-noise

若存在目标分布 `p_*` 且 `D(G|U)=p_*(G)C(U|G)/sum_H p_*(H)C(U|H)`，则 `p_* C D = p_*`。现损失的正对象权重和共同 T 本身不破坏无限容量 Bayes 最优；本次 0 support conflicts 也不支持以截断作为真实根因。

但有四个更强反例：

1. 若 `p_*` 是被 novelty 库覆盖的有限训练经验分布，完美 D 仅输出训练 support，一步后 Novelty/SUN 可为 **0**。stationarity 甚至可能与发现新结构的目标冲突。
2. construction 分布 `B_theta` 不是 `p_*`；平稳性不保证从这个起点 H=1/2 变好，更不保证能量或 SUN 单调。
3. 有限神经 posterior、概率温度误校准和错误 old 分布都会改变 K。首轮两状态反例已显示直接 D 会压低少数低能态；更换 C 而不换 D 也会改变其平稳分布。
4. 更强 noise 可把不错的 generated G 推离已有盆地，已学修复不足时净作用就是加噪。合法 rollback 只能处理显式失败，不能识别“合法但物理更差”。

因此，B 不声称三档 synthetic corruption 已覆盖真实模型误差。若要把它升级为“broad matched-noise”，必须先得到 train-only、未按好坏筛选的实际生成误差及可定义的参考关系，重定噪声/监督与起点；那已是新的数据与训练方案，不能在本候选中悄悄使用 K4/K8 或自生成 teacher。

### 4.4 NFE 与最小物理反驳

无失败重试时，construction 为 `d` 次 DLM forward，每次 full-cell repair 再 `d` 次。现 V2 与 B(H=1) 都约 `2d`；C 的扰动本身无 DLM forward。H=2 为 `3d`，N=20 时从 132 增到 198 次，不是“多加一个 diffusion step”的成本。真实 batching、支持重试、Transformer 长度、几何 bias 和 LM-head 成本仍须用 trace/wall time 计量。

最小对照使用固定 construction G：已有 `G`、已有 `D(G)`、noise-only `U` 与新 `D(U)`。noise-only 是诊断，不作为另一份可替换输出。完整拆解已显示 construction **12/44**、一次 D 后 **8/50**，所以 **保留 construction** 是必须击败的零训练竞争方案，Strict/Meta 需要共同判断。不能仅凭 Meta +6 决定追加 D；B 还要证明其额外 noise 能改善这种权衡。constructor→tau 的结果尚不存在，不在此比较中虚构。

采用时只登记 H=1，一份相同条件/seed 的全请求配对；报告 native raw E、F/stress、terminal validity、原 SUN 与 verified SUN、tau 固定终点和 novelty 流失。发生 parser/support/inner failure 的请求不删、不补。可用配对二元变化与区间描述证据强度，开发 256 不承担已经独立确认论文增益的角色。

**失败条件：** 只有 synthetic reconstruction 改善；noise-only 明显变坏且 D 未恢复；A/raw/eR 分解显示损害被 denominator 隐藏；或 tau basin entry 改善被 novelty 损失抵消。出现这些现象时，不靠增加 H 继续试到成功。

## 5. 候选 C：同一个 LLaDA 主干上的 mixed-space geometry diffusion

### 5.1 明确承认它改变了什么

这是**共享 DLM Transformer 的连续几何 diffusion**，最终可渲染晶体 token，但坐标/晶格不是从原 LM-head 的 categorical probabilities 逐项采样。不能仅因它使用 LLaDA 权重，便把结果记作原协议“DLM 原生 token 生成”成功。若论文核心必须保留后者，C 当前不满足该契约；应保留 A/B，而非改名掩盖。这不等于 C 的数学对象或物理潜力被否定。

C 解决一个真实接口差异：从第一步起六个 lattice 自由度与全部 sites 都有实际 noisy geometry；一次 forward 为所有几何自由度提供监督和 reverse drift。它不预设 V2 的失败来自 GEM，因为 V2 已有全局 conditioner 与一跳 hidden 消息；新增的关键是**训练/采样对象与监督密度**，不是把旧 GNN 改名。

### 5.2 连续目标与表示

需要按现有 27,136 个 MP20-train source 的身份、精确 N/E、site permutation 和 cell setting 对齐**原始连续 CIF/结构**。当前 token answer 不能恢复 sub-bin target，给量化 token 随机加小数也不是原始连续目标。若不能对所有 source 建立可核查对应，C 不能直接进入同数据量的正式对照；缺失项必须报告，不能暗中筛出较易的训练子集。

对规范 row-lattice L，令

\[
S_L=\tfrac12\log(LL^\top),\quad
v=\operatorname{tr}(S_L)-\log N=\log(V/N),\quad
H=S_L-\tfrac{\operatorname{tr}(S_L)}3I.
\]

`v` 的体积单位以 Å³ 为参考。取 V2 已有五个 Frobenius 正交无迹 basis `B_r`，`h_r=<H,B_r>`，连续 lattice chart 为 `w=(v,h_1,...,h_5)`。用**仅 train** 的均值和正尺度作 `z=D^{-1}(w-mu)`；常量字段必须使用明确正尺度下限而不是除零。该全局仿射标准化是噪声单位选择，不是物理的 Riemannian 坐标。

逆变换为

\[
S_L=\sum_r h_r B_r+\tfrac{v+\log N}{3}I,\quad
L=\operatorname{chol}(\exp(2S_L)),\quad F\in\mathbb T^{3N}.
\]

有限 chart 值给出 SPD cell；不要把论文未归一 basis 的 `v=3 k_6` 混到这里。fractional 坐标保持 `[0,1)`，同一 source 的连续 lattice、F 和 species 始终同步规范化。该 chart 对固定 basis 下的旋转冗余有明确处理，但不承诺 GL(3,Z) 重参数化或严格空间群等变。

### 5.3 Forward corruption：要能写出真正的条件密度

固定 `T=64` 作为一个待登记的工程起点，不做 32/64/128 SUN 搜索。令 `bar_alpha_0=1`，`bar_alpha_k=exp[-lambda_z(k/T)^2]`，定义逐步 `beta_k=1-bar_alpha_k/bar_alpha_(k-1)`。给定原连续 `z_0`：

\[
z_k=\sqrt{\bar\alpha_k}z_0+
\sqrt{1-\bar\alpha_k}\,\epsilon_z,
\qquad\epsilon_z\sim N(0,I_6).
\]

`lambda_z` 通过 train chart 的终端 prior-mismatch 上限决定，不能靠开发 SUN 选择。与 `N(0,I)` 的条件 KL 可直接计算为

\[
\tfrac12\left[\bar\alpha_T\|z_0\|^2+
6\{-\bar\alpha_T-\log(1-\bar\alpha_T)\}\right].
\]

例如把 train 最大条件 KL 预先限制到 `1e-6`；有限 `bar_alpha_T>0` 仍是近似先验，不能说完全消除了数据依赖。val 只检查是否出现更大失配，不重新拟合标准化或 prior。

坐标采用已定义的**fractional isotropic torus Brownian**，而非冒称 Cartesian isotropic。先取 `sigma_base(0)=0`；`k>=1` 时从 `1e-3` 到 `1` 对数等距。若 P 有 m 个 species，site 的 species rank 为 r，定义 `gamma_i=1+0.5(1-r/(m-1))`；m=1 时 `gamma_i=1`。设

\[
\sigma_i(k)=\sigma_{\rm base}(k)^{\gamma_i},\qquad
F_{i,k}=\operatorname{wrap}(F_{i,0}+\sigma_i(k)\epsilon_i).
\]

各增量方差是 `Delta v_i(k)=sigma_i(k)^2-sigma_i(k-1)^2 >= 0`。前序 species 在中间时刻噪声较少，相应在 reverse 中更早恢复结构；这是一个明确的 **program-controlled noise clock**，不是旧逐 species repair 顺序的等价实现。sigma 的单位是 fractional，不是 Å；在 cell L 下 Cartesian covariance 为 `sigma_i^2 L^T L`（列向量约定），不能照搬 V2 的 0.05/0.15/0.30 Å。

这里 lattice 与 F 的 forward 密度在给定原结构时分解，但联合数据的相关性进入模型所学的 marginal score。独立 forward noise 不意味着 reverse 网络独立预测物理结构。选择这种可计算 kernel 是为了避免在 lattice 同时随机变化时，写下一个没有正确联合密度的“Cartesian matched score”。它对细长/三斜晶胞的噪声尺度不均匀是实质归纳偏置，应报告，不能称为错误自动消失。

终端先验为 `z_T~N(0,I_6)`、所有 `F_i,T~Uniform(T^3)`，固定 N/E。因为所有 species 在终端 `sigma_i=1`，一维 wrapped Gaussian 与均匀密度之差满足 `2 sum_(n>=1) exp(-2 pi^2 n^2)` 的 Fourier 上界；最多 60 个坐标分量的 TV 上界约 **1.61e-7**。这只控制条件 torus prior 的近似，不控制模型误差、表示规范化或 final SUN。独立 CPU 已复算。

### 5.4 正确的 wrapped score，不用最短位移冒充

给定一颗原 site f0，条件 torus density 与 score 为

\[
q(f\mid f_0,\sigma)=
\sum_{n\in\mathbb Z^3}\varphi_{\sigma^2I}(f-f_0+n),
\]

\[
s_F^*(f,f_0,\sigma)
=-\frac{\sum_n(f-f_0+n)\exp[-\|f-f_0+n\|^2/(2\sigma^2)]}
{\sigma^2\sum_n\exp[-\|f-f_0+n\|^2/(2\sigma^2)]}.
\]

此处不 fold ASU、不求 SG orbit-center density。实际实现可利用各轴乘积分解、小 sigma 的 image sum 和大 sigma 的 Fourier series，但必须有数值尾项界与导数检查。不能全噪声范围都使用 `-MIC(f-f0)/sigma^2`。

CPU 反例：一维 `f-f0=0.49, sigma=0.2` 时，正确 score 为 **−1.30441**，nearest-image 近似为 **−12.25**；差值 0.5 时正确 score 为 0，而单 image 给 ±12.5。`sigma=1` 时真 score 接近 0，单 image 仍会强行把点拉向某一标签。这足以否定错误 score 的直接迁移，不证明正确版本会提高 SUN。

### 5.5 模型必须实际通过共享主干计算

保留同一预训练 LLaDA 双向 Transformer；新 LoRA、V2 可复用的 conditioner/几何交互按新训练协议适配。canvas 仍有硬 N/E 和原 slot 排布，但数值状态来自明确的连续 `(z_k,F_k)`，以 family embedding、非周期 lattice features、周期 Fourier features 和 time/noise features 注入相应 slot。连续 current 直接构造完整几何 bias；不把 rounded token、未知值或概率均值作为该连续状态的替代。

输出只从 Transformer hidden 得到：六个 cell-slot hidden 的固定 pooling 接一个 `6-output epsilon head`；每 site 的 XYZ hidden pooling 接共享 `3-output torus-score head`。最简单的线性 readout 在 hidden=4096 时共 **36,873** 个参数（含 bias），另计连续输入/time adapter。heads 不另接绕过主干的独立几何预测网络。已有几何消息仍经主干 hidden；不 detach H，梯度应回到 LoRA/输入与几何适配器。

head/adapter 梯度、old/current 几何敏感度、把 H 置换后的响应只能证明实际通路，不能证明预训练 LLaDA 比一个小几何模型更好。后者需要资源匹配的比较；如果 head 仅凭化学和局部 side input 就能得到同样效果，论文的 DLM 必要性主张须降低。

为隔离第三候选，S/P 沿用 V2 缓存与冻结 LLM，不同时采用 A 的重标注。LLM 一次给出条件与噪声 clock；不观察在线 geometry，也不通过 energy 选下一步。仅有 executable clock 仍不证明 learned P 优于固定 rank；这个收益是待检验命题。

### 5.6 Dense denoising score loss

每个 source/view 只采一个 k 和一份完整 noisy geometry；全部六个 lattice 与 3N 个坐标分量都产生监督。定义 `sigma_z(k)=sqrt(1-bar_alpha_k)`，模型输出 `epsilon_theta` 和 `u_theta,i=sigma_i s_theta,i`，目标为

\[
\mathcal L_C=\mathbb E_{k\sim U\{1,...,T\}}
\left[\frac{1/2}{6}\|\epsilon_\theta-\epsilon_z\|^2+
\frac{1/2}{3N}\sum_i\|u_{\theta,i}-\sigma_i s_{F,i}^*\|^2\right].
\]

在无限容量及对应条件分布下，denoising-score matching 的条件期望给出 noisy **marginal** score；不能把单个 paired target 的 clean 位移直接当成数据 marginal score。对象权重和 sigma 缩放用于单位与方差管理，固定时间条件下不改变最优目标。该设计未声称这个简单加权 MSE 就是所选有限 64 步 mixed sampler 的精确 ELBO。

无 CHGNet/force/stress/reward 训练项，无模型自生成 positive，无连续 head 再叠很多未经分离的 CE/transport/energy loss。若需要 token anchor，则是另一个有影响的训练选择，不能临时加入并继续称只改了 score 对象。

### 5.7 实际反向 kernel 与最终 token 测度

给定当前完整 `(z_k,F_k)`，一次共享主干 forward 得到两类输出。lattice 用已定义 VP chain 的 x0 参数化 Gaussian reverse：

\[
\hat z_0=\frac{z_k-\sqrt{1-\bar\alpha_k}\epsilon_\theta}
{\sqrt{\bar\alpha_k}},\quad
\mu_k=\frac{\sqrt{\bar\alpha_{k-1}}\beta_k}{1-\bar\alpha_k}\hat z_0+
\frac{\sqrt{1-\beta_k}(1-\bar\alpha_{k-1})}{1-\bar\alpha_k}z_k,
\]

\[
z_{k-1}\sim N(\mu_k,\widetilde\beta_k I),\quad
\widetilde\beta_k=
\frac{\beta_k(1-\bar\alpha_{k-1})}{1-\bar\alpha_k}.
\]

k=1 的 variance 为 0，取相应确定性 mean。这个公式给定 true z0 时是正确 forward posterior；换成神经 `hat z0` 后并非一般 joint reverse posterior 的精确积分。高噪声除以很小的 `sqrt(bar_alpha)` 必须检查数值稳定性，不能悄悄 clip clean prediction 而不登记分布变化。

坐标用对应 variance increment 的一次 reverse Euler 步：

\[
F_{i,k-1}=\operatorname{wrap}\{F_{i,k}+
\Delta v_i(k)\,u_{\theta,i}/\sigma_i(k)+
\sqrt{\Delta v_i(k)}\xi_i\},\quad \xi_i\sim N(0,I_3).
\]

末步也按这条已声明 kernel 执行；它在有限步数下有残余噪声和离散化误差。不能因为 score 公式正确，便宣布 64 步 Euler 精确。lattice 与 F 在每一步同步更新，下次 attention 读取**真正更新后的** geometry。

整体归一化生成模型为

\[
p_{\theta,C}(X_0\mid c,S,P)
=\int p_T(X_T)\prod_{k=T}^{1}K_{\theta,k}(X_{k-1}\mid X_k,c,S,P)\,dX_{1:T}.
\]

每个 Gaussian/wrapped-Gaussian（及末端 delta）是明确 kernel；它没有把正确的单变量 clean marginals 一次独立填成晶体。有限步条件因子化仍会近似真实 joint reverse，dense forward 并不提供其精确性保证。

连续反向中只硬固定 N/E、确保 chart 有限/SPD 和 torus wrap；**不施加旧逐 scalar PBC hard mask**。加入拒绝/投影会改变 forward/reverse 关系，不能说免费继承。最后一次按固定原 codec 渲染并调用同一 complete-geometry 门；超范围、量化 clipping、碰撞或 nonfinite 进入明确失败状态，不偷偷重采/替换。这样仍有全请求概率空间，但不再有 V2 每一步整胞有效性的保证。

若最终渲染为 token y，其概率是连续模型对量化原像的质量

\[
p_C(y)=\int_{Q^{-1}(y)}p_{\theta,C}(x)\,dx,
\]

不是原 DLM logits 的 categorical probability。0/100 alias 必须折叠到同一个 physical bin，最后只输出 canonical 0；没有理由另加两个 token logits 的 logaddexp。需要分别报告连续 `X_0`、一次原 codec 后的 `Q(X_0)`；作为与 A/B 的正式对照时原 codec 结果是主对象，连续结果仅用于定位量化损失。否则“continuous 好而 token 坏”会被藏起来。

### 5.8 与旧 PMTR 的真实差别

旧 [PMTR 方法](historical/docs/teacher_feedback_unified_v1/15_PMTR_SCIENTIFIC_METHOD_AND_EXECUTION.md) 已提出联合 SPD/PBC corruption、repair-vector head 和 manifold-to-token transport；[代码审计](historical/docs/teacher_feedback_unified_v1/16_PMTR_CODE_GROUNDED_ARCHITECTURE_AUDIT.md) 已记录相关模块实现。当前本地 [pmtr_training.py:302](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/pmtr_training.py:302) 明确 `no_grad`，到 335 行 detach hidden；不能把这些已有构想换名当作 V3 创新，也不能把未形成正式 SUN/checkpoint 证据的 PMTR 称为已失败实验。

| 维度 | 旧 PMTR 提案/代码路径 | 候选 C |
|---|---|---|
| 学习对象 | 单 source 的局部 manifold retraction，transport 到 active token logits | 指定多时刻 forward 的 joint noisy score/epsilon，显式 reverse kernel |
| 起点 | 已有 SPAD construction | Gaussian lattice-chart 与 uniform torus prior |
| 监督 | active cell/XYZ transaction，附连续 retraction loss | 每 noisy state 全部 6+3N scalar；真 wrapped-score target |
| 主干 | 初始集成版冻结 retained DLM/LoRA、detached H、head-only | 同一原始 LLaDA 主干；新 LoRA/输入/geometry/head 梯度连通 |
| 物理证书 | 离线 CHGNet corruption certificate，未作为 loss | 本候选不依赖新证书，只有原始连续训练结构 |
| 输出 | residual logits，仍通过 token probability 生成 | 连续 mixed-space reverse；最后 codec push-forward |
| program | cell+reverse-species 的事务序列 | 冻结语义条件和明确的 species noise clock |

差别并非仅“把 detached 改成可训练”：概率对象、数据、起点、noise units、损失、解码与论文声明同时变化，所以 C 不应作为小修补塞进 V2。

### 5.9 小预算下的机会、成本与失败条件

主干适配仍可限定新 LoRA 与小连续输入/geometry/head；不需要全参更新 8B。原先为 full-vocabulary output 学的新 token-output rows 对 C 的 score loss 没有用途，应从可训练参数账中单列，不能用“仍 36.99M”掩盖死参数。若 wrapper 能直接得到 H，可省去无用的 full-vocabulary logits 投影；这是待实现的效率选项，不能预报未经测量的吞吐。

保持每 source 两个独立 k/noise view，1–2 epoch 的真实 state 数与 V2 相同。V2 每 view 的期望有监督 scalar 数（未计 admission）是

\[
0.75\times1+0.25\times E[p_{mask}](6+3N)
=0.75+0.1375(6+3N).
\]

| N | V2 期望 scalar/view | C dense scalar/view | 密度比 | V2 nominal NFE | C 固定 NFE |
|---|---:|---:|---:|---:|---:|
| 1 | 1.9875 | 9 | 4.53 | 18 | 64 |
| 10 | 5.70 | 36 | 6.32 | 72 | 64 |
| 20 | 9.825 | 66 | 6.72 | 132 | 64 |

这是目标分量计数，**不是**独立信息、梯度质量、训练速度或 SUN 的倍数。C 给 N=20 更多每状态几何监督，且推理可能少 forward；N=1 反而明显更贵。每个 8B forward 的 geometry、sequence、precision、batching 成本也可能不同。

V2 记录的 4,096.53 秒仅是已有运行尺度。4–6 A800 可维持 global batch=24，通过 microbatch/accumulation 调整匹配内存，但不能把“卡够放模型”写成“2 epoch 足够学会 noisy continuous geometry”。高噪声 prior 无有效邻近结构、lattice 长尾、6.7 倍相关 labels、有限 Transformer 适配容量，都可能使它仍无真实生成收益。

**拒绝 C 的实质条件：** 连续 target 无法全量对齐；输入仍只喂量化 token 导致低噪声信息消失；wrapped score/噪声单位错误；head 绕过/不使用主干；新 prior 的大部分 geometry 不在模型可读范围；结果仅连续端变好而原 codec 端损坏；有限模型或粗 reverse 步产生平均构型/重复结构而缺少 basin 多样性；或 fixed reverse 的物理结果不能击败更便宜的 V2 construction/B。正确的 marginal score 本身可以表达多峰分布，不能把最后一种失败无条件归因于使用 MSE。

**最小 SUN 相关检验：** 先在 held-out original source 上按 k 分层核对 lattice/torus denoising与主干作用；然后一份固定完整先验的 free generation，分别报告 continuous→codec 的失效转移、raw物理量、全部分母的原 SUN/终态几何有效 SUN/verified SUN，以及固定 tau novelty/盆地结果。只修复 original perturbations 不构成生成有效性；只提高 verified 比例不等于 SUN 增长。

## 6. 为什么不为“拿到 ELBO”默认再造一套全离散 chart

全离散数值 diffusion 是成立的竞争路线，不能以“token 无连续梯度”排除。以下给出足够明确的对象，说明它哪里可行、为什么未列作默认第三候选。明确 Q、累计 Q 和 reverse/变分项的框架参照 [D3PM 原文 §2–3](https://arxiv.org/html/2107.03006v3#S3)；下面的晶体 chart、clock 和取舍是本文的构造，并非该论文已验证的晶体方法。

### 6.1 可以构成合法的 categorical diffusion

沿用 C 的六维 lattice chart w，但量化成六类非周期有序 token，各 `K=129`；坐标使用 `Z_100` 三个周期字段。chart 范围从全部 train extrema 加冻结的正 margin 得到，不能用 val/SUN 调范围；超界不 clip 成隐形成功。解码 chart 后天然 SPD，但这是一套**新的 lattice 词义和 codec**，不是原 length/angle token 的等价重命名。要先测全部 train roundtrip 的物理误差和有效性，不能假设覆盖 extrema 即足够精细。

每一字段给严格正的 prior pi。lattice 可用平滑后的 train 边缘频率，坐标用物理 100-bin 均匀 prior，彻底移除 0/100 双 alias。以相邻 bins 的无向权重 `w_ab=w_ba` 定义 rates

\[
R[a,b]=w_{ab}\sqrt{\pi_b/\pi_a}\;(a\ne b),\qquad
R[a,a]=-\sum_{b\ne a}R[a,b].
\]

有序 lattice 是反射端点的 path，坐标是 cycle；`pi_a R[a,b]=pi_b R[b,a]`。对每字段的已知递增 clock lambda 定义

\[
Q_k=\exp\{(\lambda_k-\lambda_{k-1})R\},\qquad
\bar Q_k=\exp(\lambda_kR).
\]

P 可以按 species rank 改变对应 coordinate clock 的幂，所有终端达到相同预先规定的 prior-mismatch 上限。这里真正有 semigroup；不是把量化 Gaussian 的三档方差相减伪装成 reverse schedule。用 prior-weighted 相似变换可做实对称特征分解。CPU 的 9-state 非均匀 prior path 检查 row-sum、stationarity、semigroup 误差均约 `1e-15`，posterior 归一化误差小于 `6e-11`。

条件 forward 在字段上分解，posterior 为

\[
q(x_{k-1}^j=a\mid x_k^j=b,x_0^j=c)
=\frac{\bar Q_{k-1}^j[c,a]Q_k^j[a,b]}
{\bar Q_k^j[c,b]}.
\]

同一 DLM 每次读取完整 noisy integer geometry，输出各字段的 `r_theta,j(c|x_k,k,c_hard,S,P)`。定义

\[
p_{\theta,k}(x_{k-1}\mid x_k)=
\prod_j\sum_c r_{\theta,j}(c\mid x_k,k)\,
q(x_{k-1}^j\mid x_k^j,x_0^j=c).
\]

所有因子规范化；仍然是有限步 factorized reverse 的模型限制，不能因每个 oracle marginal 正确而宣称完整 joint posterior 精确。主训练目标可明确包含 `KL(q(x_T|x0)||pi)`、k>=2 的 `KL(q(x_(k-1)|xk,x0)||p_theta)`、k=1 reconstruction，再加已固定系数的 exact x0-CE；只有前面对应项才构成负 log-likelihood 的变分上界。T 个 outer 的采样从 product pi 开始，每次所有字段通过一次共享 DLM 得到下一步，不需执行 d 个 scalar forward。

### 6.2 支持与资源：两个方向都不能夸大

**优点是真实的：** 原生离散输出概率明确，最终量化和生成测度一致；ordinal/cyclic locality 可精确表达；全部字段同时可见，dense 时间监督和 T 次 NFE 都能定义。不要因为 mixed-space head 更小，就断言离散 head 必然学得差。

**Q 本身也不是天文成本。** 不显式存储高维 Kronecker joint；每个 Q 只有 100×100 或 129×129，lattice 的 6 字段、64 时刻、Q/累计 Q 两份 FP32 约 51 MB，coordinate cycle 可只存 circulant 首行。可重复使用 eigensystem 或按需计算 clocks。把这条路线否定为“巨大 full-joint 转移矩阵放不下”是错误反驳。

实际成本在新数值表示与建模契约：

1. 六字段若追加 129-bin、untied input/output rows，在 hidden=4096 时是 **6,340,608** 个新参数；替换旧 lattice rows 可回收参数，净增量必须实际计数。这些新词义没有现成 token 语义，1–2 epoch 必须重新学习晶格 chart 与几何的对应。
2. train 各字段分别有限范围并不使笛卡尔积中的组合合理。独立极端 shape bins 可产生很偏斜的 cell，均匀 F 可产生碰撞；SPD 是较弱的结构条件。
3. 旧逐 scalar physical-support mask 不能在每次 Q/reverse update 后直接投影并继续使用原 posterior/ELBO。若 forward 一并限制在合法结构图上，归一化常数、可达性和 tractable posterior 都变成高维问题。若只保留最后验收，则和 C 一样放弃了 V2 的中间整胞有效性保证。
4. 新 chart→旧 length/angle codec 若再渲染一次，会出现第二次量化与 many-to-one push-forward；新 chart token likelihood 不自动是旧 body likelihood。把新 chart CIF 直接当 native 又改变了基线的 lattice 量化精度，需要明确对照。
5. 图上的相邻 bins 是新 chart 的数值邻近，不等于 Cartesian/能量邻近。长尾 cell 和 atom coupling 不会因为 Q 有 ELBO 就被解决。

因此，如果必须保持 token-native 论文契约且愿意重新定义 lattice codec，这条路线在数学上可行；但没有证据说明它比 A/B 或 mixed-space C 更可能在短预算中提升 SUN。**不推荐仅为了拥有 ELBO 而首先承担这套表示迁移。** 这也没有把 C 的连续输出主张冲突洗掉：两条第三族路线各有必须支付的代价。

## 7. 反驳之后的执行优先级和论文边界

当前不启动新实验。既有 V2 完整结果使最小下一步更清楚：先以 construction **12/44** 和一次 repair **8/50** 的配对路径为起点审查实际改变量；不从 `C→D` 的平稳性或“更多监督”直接跳到新架构必然提点。

1. **B 是最便宜的可证伪比较，H=1 已足够提出清楚问题。** 它的额外 noise 不增加 nominal DLM NFE，但要同时击败不 repair 与现 repair 的物理/Strict/Meta 权衡。没有证据支持现在直接追加 H=2 或 broad-noise training。
2. **A 取决于论文是否真的主张 LLM 的几何条件控制。** 如果 S 实际是计算特征而非请求，就应证明其路由/样本效率作用，不一定要改成 controllable latent。若要保留几何控制主张，标签与输出属性的一致性比新堆 GNN 更直接。
3. **C 是第三族中较直接的物理生成研究对象，但有前置事实条件。** 原连续目标可全量对齐、共享主干真实参与、论文允许 mixed-space、最终 codec 后效果仍成立，才值得讨论 1–2 epoch。若 token-native 是不可变核心，就保留 C 为未采用的反例/竞争方案，不以改名绕过；全离散 Q 路线需要独立 codec 审计后才能替代。

这三个候选都不修复“只有高质量训练结构才足够覆盖未知 composition 下稳定 polymorph”这一数据支持限制。有限经验分布的完美重建可以损害 novelty；能量、verified、SG 服从、原生 token 概率、固定 tau basin entry 是不同观察量。没有哪个候选靠规范化 kernel、正确 score 或 CPU 通过就保证开发 26/128，更不能据 256 上的单次选择直接确认 paper1000 成功。

## 8. 独立检查与未决事项

本次 CPU 脚本只验证 wrapped score 的有限差分/周期性、uniform-torus prior 上界、reversible discrete Q/posterior、语义联合 toy、监督密度和 nominal NFE 算术。它没有加载真实 checkpoint、调用 SSH/GPU、运行 CHGNet、读取原连续 CIF，或宣称候选效果。

仍待实际证据：`a_delta` 在完整 native codec targets 上的可识别性；固定 S/P 干预的结果；construction/repair 的逐 request 物理与 terminal 转移；C 的原连续 target 全量映射和 chart 范围；所需 adapter/hidden path 的真实吞吐。文档中提供完整的数学对象和失败条件，**不是已经完成这些数据核验或给任何新路线签发通过结论**。
