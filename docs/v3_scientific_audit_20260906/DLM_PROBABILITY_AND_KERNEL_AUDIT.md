# 离散周期 DLM 的概率、表示与 kernel：独立攻击性首轮审计

2026-09-06。范围是整个近期 DLM 路径的概率对象及其与 LLM 条件的衔接，重点核对 V2 执行代码 `2e904c260bafb6750c9a5dbb0d920c4ff8c3a868`。本轮读取时，相关 `src/crystal_dlm` / `src/scripts` 相对该执行版本没有差异；后续 operations 提交不作为训练实现变更。没有修改训练代码、加载真实模型权重、访问 SSH、调度 GPU、运行 MLIP 或生成新训练样本。

独立 CPU 证据：[脚本](dlm_probability_cpu_checks.py)、[数值结果与源码 SHA256](dlm_probability_cpu_checks.json)。脚本只运行有限状态矩阵及小型随机参数网络，不能当成 V2 实际效果测量。

## 1. 首轮结论：值得保留的机制与最危险的叙事跳跃

1. **V2 的离散条件概率有清楚定义，程序前缀采样能表示晶格—坐标的联合分布。** 不能因为输出是类别、或因为没有显式 mask-time，就宣布它不可能生成好晶体。相反，把多个正确的单变量后验直接并行采样，仍可能破坏联合结构；V2 的逐 scalar 条件链在这方面有明确价值。
2. **V2 的实际生成过程是 program-conditioned construction 加一次 old-conditioned full-cell repair。** 当前没有定义逐级几何 Markov 转移及其反向后验，也没有优化已证明对应当前最终输出分布的 diffusion ELBO。它可以诚实称为“离散周期条件去噪 DLM”，不能因采用 LLaDA 主干或 log-SPD 扰动而自动获得连续 score-flow / 反向 diffusion 的性质。
3. **几何 attention 对 construction 早期有真实的可见性瓶颈。** 在六个 construction lattice scalar 的预测时，旧晶胞与所有站点坐标都不完整，实际几何 bias 为零。后续 active coordinate site 也尚不完整，因此其直接几何 query 边仍为零；已知站点/晶格之间的 bias 可以通过 Transformer 多跳影响预测，不能把“直接边为零”扩大为“模型完全没用几何”。
4. **直接重复 denoiser 没有数据分布不变性保证；同一 forward corruption 后接完美 Bayes denoiser 有。** 这给一个有界诊断提供理论依据，但本实现的 prefix admission、未知噪声混合、有限容量和执行回滚都使完美 posterior 前提需要逐项检查。即使 stationary 成立，也不保证从低质 construction 分布出发，一两步就提高 SUN。
5. **随机预测 soft Plan 配原样 clean target，可能不能识别 soft 意图的服从性。** 在充分条件化的 population 独立模型下，最优 DLM 可以忽略随机 soft；但这不证明 LLM 的计算路由无用，也不证明当前有限数据真的独立。metadata、teacher fallback、同源预训练和近唯一组成都必须进入反驳。
6. **历史失败不能一概归为“离散 token 不懂物理”。** canonical slot 对齐、可达化学支持和概率回放修正解决了真实问题；K4 的候选支持与 teacher 投影只提供有限偏好，不能保证 student 漂移或稀有低能尾部；raw-v1 又同时改变初始化、可训练行、状态和损失。V2 相比这些版本仍是一个组合干预。

证据分级：**A** 为真实代码路径或数学恒等式；**B** 为独立有限状态/小 CPU 反例；**C** 为与本项目吻合但尚无隔离实验的假设。A/B 证明“某个保证不能直接成立”，不证明其就是实际 SUN 不足的原因。

### 已纠正的实验背景

以 [CURRENT_STATE](../CURRENT_STATE.md) 和主审核实口径为准：旧 K8 独立条件 1000 的 native Strict/Meta SUN 为 **39/244**，tau800 为 **65/484**。此前 hull cache 缺覆盖的中间输出不作为完整模型退化证据。raw-v1 的开发 256 为 native **3/45**、tau800 **16/113**，对应开发参考为 **6/55**、**19/121**。不同 cohort 不能直接作成模型排名。

主审已核实 39993 的 V2 初始化 `zero_logit_delta=0.0`、36,993,232 个可训练参数，其中新 token rows 20,324,352、LoRA 14,680,064；约 8.035B 原预训练参数冻结。因此本轮是**从原始预训练 LLaDA 新建晶体适配器**，不是从随机初始化训练整个 8B。不能把“两 epoch”直接与论文全参数、大数据预训练 epoch 等价比较。

主审还提供了首 64 步各模块累积梯度非零的检查。这排除了“一直完全断梯度”这一简单解释；梯度总和受维度、参数化和频率影响，不能据此比较模块贡献或推出 SUN 改善。

## 2. 原论文与作者代码：可以借用哪些数学对象

| 工作与已核对原文版本 | 原对象及本审计采用的边界 | 作者代码快照 |
|---|---|---|
| [D3PM，arXiv 2107.03006v3](https://arxiv.org/html/2107.03006v3)，2023-02-22 修订；NeurIPS 2021 | 通过明确的 categorical 转移矩阵定义 forward/reverse chain；ordinal Gaussian、absorbing 和邻接图是不同的 (Q_t)。辅助 (x_0)-CE 与变分项有区分。它直接反驳“离散扩散只能 mask、不能有数值邻近性”。 | google-research `20c6bc70bfcedc40465942e8815f3de57d1658ce`：[Gaussian 转移](https://github.com/google-research/google-research/blob/20c6bc70bfcedc40465942e8815f3de57d1658ce/d3pm/images/diffusion_categorical.py#L198)、[posterior / reverse 参数化](https://github.com/google-research/google-research/blob/20c6bc70bfcedc40465942e8815f3de57d1658ce/d3pm/images/diffusion_categorical.py#L399)、[loss 类型](https://github.com/google-research/google-research/blob/20c6bc70bfcedc40465942e8815f3de57d1658ce/d3pm/images/diffusion_categorical.py#L625)。 |
| [SEDD，arXiv 2310.16834v3](https://arxiv.org/html/2310.16834v3)，2024-06-06；ICML 2024 | 估计离散状态概率比，乘 forward rate 构造 reverse rate。score entropy 的一致性和似然界有明确支撑/权重/转移前提；不能将一组普通 CE logits 直接称为该 score。 | `0605786da5ccb5747545e26d66fdf477187598b6`：[reverse rate](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/graph_lib.py#L77)、[absorbing score entropy](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/graph_lib.py#L244)、[采样器](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/sampling.py#L60)。 |
| [MDLM，arXiv 2406.07524v2](https://arxiv.org/html/2406.07524v2)，2024-11-10；NeurIPS 2024 | SUBS 参数化把 absorbing diffusion 的目标化简；可见 token 的 carry-over 和由噪声日程确定的转移系数是过程的一部分。其 time-independent 网络讨论对排除“没 t 就不是 diffusion”的误诊很重要。 | `c112c526d193436838c98d81455ee51f90309470`：[SUBS](https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/diffusion.py#L261)、[reverse update](https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/diffusion.py#L612)、[加权目标](https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/diffusion.py#L847)。 |
| [LLaDA，arXiv 2502.09992v3](https://arxiv.org/html/2502.09992v3)，2025-10-18 | 独立 mask forward、masked conditional predictor 与近似 reverse generation；论文也允许 AR/block sampling，low-confidence remasking 是实际采样选择，不应与任意重复修订混为一谈。 | `9182493720ed723ef8031210d85959364e51cbe0`：[训练指南 (1/p)](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/GUIDELINES.md#L25)、[实际逐步提交](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py#L142)。该 snapshot 含论文之后的扩展；此处只核对保留的 mask/transfer 路径。 |

这些是作者代码，不使用第三方复现来证明原方法。以上 URL / SHA 在本次联网只读检索中核对；没有安装依赖或下载模型。下面大部分反例是独立推导，不是论文的实验结论。

### 三种容易混淆的数学量

采用行随机矩阵记号，D3PM 的一阶转移为 (Q_t[a,b])，累计转移为 $\bar Q_t=Q_1\cdots Q_t$。在 (x_0=c,x_t=b) 下：

\[
q(x_{t-1}=a\mid x_t=b,x_0=c)
=\frac{\bar Q_{t-1}[c,a]Q_t[a,b]}{\bar Q_t[c,b]}.
\]

它的 reverse 参数化、先验项、重建项和逐级 KL 必须与指定 forward chain 对应；仅换一个“更平滑”的 CE target 没有自动给出这些对象。[D3PM 原文 §2–3](https://arxiv.org/html/2107.03006v3#S3)。

对连续时间、行向量约定的离散生成率 (R_t)，相反方向的率包含

\[
\overleftarrow R_t[x,y]=R_t[y,x]\,p_t(y)/p_t(x),\qquad x\ne y.
\]

SEDD 学的是这些局部概率比。坐标 token 的 CE 概率、一个物理 energy score、以及这个 ratio 是三个不同量。[SEDD 原文 §3](https://arxiv.org/html/2310.16834v3#S3)。

V2 实际学的则是下面的条件分类风险；没有将 logits 解释为 (p_t(y)/p_t(x))，这一点应保留。

## 3. V2 的真实概率对象：什么成立，什么尚未成立

主要源码：

- [training data:205](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_v2_training_data.py:205)：两个 view，prefix/dense 分支与 teacher prefix。
- [objective:22](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_v2_objective.py:22)：T=0.7、exact CE、无 ordinal smoothing。
- [runtime:255](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/programmed_path_runtime.py:255)、[runtime:321](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/programmed_path_runtime.py:321)：construction 和 full-cell repair。

设条件 (c) 包含硬组成、prompt 和冻结 program。对实际状态 (s_j)，先合并周期 alias 再在合法支持 (S_j(s_j)) 上定义

\[
p_{\theta,j}(a\mid s_j,c)
=\operatorname{softmax}_{a\in S_j(s_j)}\{g_{\theta,j}(a,s_j,c)/0.7\}.
\]

对一次没有拒绝的 full-cell repair 提议，程序 $\pi$ 给出的联合条件分布是

\[
D_\theta(G'\mid U,c)
=\prod_{j=1}^{6+3N}p_{\theta,\pi_j}
 (G'_{\pi_j}\mid G'_{\pi_{<j}},\mathrm{MASK}_{\pi_{\ge j}},U,c).
\]

这不是“每个数值独立预测”的模型。无限容量且学到正确 conditional 时，任何联合离散分布都能这样分解；程序只是改变 factorization 与有限容量的学习难度。最终支持验收和 rollback 会把多个提议/attempt 合并到同一个返回结构，因此真实执行 kernel 还包括返回 old 的概率质量。

独立反例：数据只有 `00`、`11`，各占一半；全 MASK 时两个完全正确的单变量 marginals 都是 `(0.5,0.5)`。一次并行独立抽样却给 `01/10` 共 **50%** 的质量，oracle sequential conditional 的这项错误为 **0**。所以把 dense logits 一次并行填完，不等于采到了 joint posterior；小时间步的 reverse 近似与一次大步提交也不能混为一谈。这反而是应保留 V2 条件链、而非先把它改成并行独立坐标采样的理由。

由 construction 输出 $G_0$ 再 repair，一般有

\[
p_{\mathrm{native}}(G\mid c)=\sum_{G_0}p_{\mathrm{construct}}(G_0\mid c)
K_{\mathrm{repair}}(G\mid G_0,c),
\]

还需保留失败状态。单条 trace 的 $\sum_j\log p(a_j|s_j)$ 不是这个 marginal 的 log probability。旧 K4/K8 拟合 attempted paths、且包括被拒绝动作，是正确区分；不能用这个事实推得对最终结构概率已精确积分。

### 实际 loss

每源每 epoch 有 construction/repair 两个 view；各自 prefix 概率 0.75、dense 概率 0.25，整体分支概率是 $3/8,1/8,3/8,1/8$。数值对象风险为

\[
R=\tfrac12\tfrac1{6}\sum_{j\in L}\ell_j+
\tfrac12\tfrac1{3N}\sum_{j\in F}\ell_j.
\]

Prefix 抽一个 position，用累计 teacher reachability 与 old admission 的 indicator (Z_j) 乘该状态 CE。Dense 取 (p\sim U(0.1,1))，用独立 mask 的 (M_j/p) 系数；其 task 是 `structured_denoise`，只在 canonical typed 类别中归一化。相关代码明确没有再乘一次分支概率，也没有把空 mask 或非法 prefix 从源分母删除。[objective:179](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_v2_objective.py:179)。

**最强反驳：** 正的对象/位置权重和共同的温度参数化，本身不改变无限容量、固定条件下的 Bayes conditional 最优；不能称“0.5/0.5”或“T=0.7”在数学上让 posterior 不成立。实际问题是有限容量、选择性支持、不同任务状态以及是否已经学到那些 conditional。

**不能扩张的结论：** `M/p` 与 masked diffusion 中的一个权重形式相似，不会自动使“截断 p 范围＋对象权重＋prefix mixture＋几何噪声 old＋执行验收”的总目标变成当前 native marginal 的 ELBO。V2 现有文档也没有作该宣称。

对线性 absorbing mask 日程，标准形式包含

\[
L_{\rm mask}=\int_0^1\frac{dt}{t}\,
\mathbb E_{x_0,x_t}\left[\sum_i\mathbf 1[x_t^i=M]
\{-\log p_\theta(x_0^i\mid x_t)\}\right].
\]

对应的先验/边界、SUBS/carry-over 和 reverse 参数化也是似然论证的一部分。LLaDA 的作者指南在 masked CE 上使用 `1/p_mask`，MDLM 的实现使用噪声日程导数系数；V2 的相似局部形式不是把这些边界与过程一起继承过来了。[作者训练指南](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/GUIDELINES.md#L50)、[MDLM 实现](https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/diffusion.py#L893)。

## 4. Mask 的局部性与 time：先反驳两个错误诊断

### 4.1 Absorbing mask 不定义数值邻近，但不是表达能力的硬上限

若每个数值类别以概率 (t) 跳到 MASK，跳转机制不会区分 0.01 和 0.02 比 0.01 和 0.51 更接近。它丢掉该变量的数值锚点；D3PM 的 ordinal transition 则可以让相邻值更容易互换。

但 V2 repair 的完整整数 old (U) 仍提供锚点，Transformer 可学习“old 37 附近应如何改”。因此“MASK 把数值抹掉，所以 V2 修订完全不知道原数字”是错误反驳。真正需要检测的是：old 是否影响 active logits，以及模型学到的是局部修复还是条件重采样。

**最小证伪：** 在同一个固定 validation source 上，只改变 old 数值、保持 current/prefix 与 Plan，比较 active logits 的周期位移响应、目标 CE 和生成改变量。若 old 有强烈、方向正确且可泛化的作用，单纯 locality 缺失解释就应降级。

### 4.2 Mask-time 不必显式输入；几何噪声强度是另一回事

对独立 absorbing mask，给定 mask pattern 后，所有相容 clean (x_0) 的 forward 因子都含同一个 $t^k(1-t)^{n-k}$。所以

\[
p(x_{0,M}\mid x_{t,\bar M},M,t)=p(x_{0,M}\mid x_{t,\bar M},M).
\]

这解释了 time-free mask predictor 的合法性；LLaDA Appendix A Eq.11 明确给出该形式，MDLM 也讨论 time-independent denoiser。[LLaDA 原文](https://arxiv.org/html/2502.09992v3#A1.SS1)、[MDLM §4](https://arxiv.org/html/2406.07524v2#S4)。reverse 的 mask/unmask 转移比例仍需要日程，不能把“predictor 可不读 t”误写为“任何重复调用都是同一个 reverse process”。

对几何 Gaussian corruption，

\[
p(G\mid U,\sigma)\propto p(G)C_\sigma(U\mid G)
\]

通常随 $\sigma$ 改变。V2 实际有三分量 noise channel；None 和 $(-1,-1,-1)$ 都编码为全零的 unknown 特征，[model:68](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_v2_model.py:68)。CPU 检查证实两者相同。50% metadata dropout 覆盖部署 unknown 接口，不是“visible 默认等于 clean”的实现。

**尚未解决：** unknown 分支学到训练 corruption 混合下的 posterior，不意味着生成错误也来自该混合。mask ratio 不是数值误差幅度，不能拿它伪造 $\sigma$。

## 5. 真实几何路由瓶颈：早期晶胞、active site 与固定 old

证据 A/B：`geometry_inputs` 从 old 六个晶格 token 和每 site 三个坐标解码；缺一个就相应 unknown。[state_conditioned_model:95](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/state_conditioned_model.py:95)。V2 site/site 与 cell/site 边都通过 known mask，[V2 attention:213](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_v2_model.py:213)、[324](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_v2_model.py:324)。

独立 CPU 中，即使把所有相关投影设置为非零，construction 位置 1–6 的全 bias 仍为 0；首个 site XYZ 时也为 0。后来已有完整 site 时，全局 bias 非零，但当前未完整 site 的 query 行仍为 0。repair old 完整时，同一模块的 bias 明显非零。

这意味着当前 architecture 的早期 lattice 决策主要依赖 Plan、program、基础 Transformer 和数值行，而不是已知原子邻接。一个以所有原子/晶胞都带噪但可见为起点的几何 diffusion，从一开始就能把几何状态放进共同更新；V2 必须依靠后面的 full-cell repair 补偿。此处是任务接口差异，不是仅比较网络名字。

**最强反证：** Transformer 的多层 token 交互能把已知 node 的几何效应传到没有直接 bias 的 query；数值 conditioner 也有化学/program 信息。early geometry 为零没有证明这些 lattice 决策一定差，也没有证明另一个模型一定好。

**竞争解释：** 新数字行/LoRA 尚未充分拟合晶胞先验；随机 soft 条件并不提供实际几何意图；四类训练状态共享小适配器；粗 codec/组成长尾。不能只归因 GEM。

**最小证伪实验：** 最终 V2 的 construction 与 repair 分开做固定配对统计；再在同一 checkpoint / 同一 teacher states 上做 old-geometry 与 learned geometry-bias 的开关诊断，分别记录 active-logit 敏感度、cell/coord CE、实际 trace 和 SUN。先前梯度非零检查不能代替这个实验。

固定 old 在一个 repair transaction 内不更新，[runtime:325](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/programmed_path_runtime.py:325)。新 lattice 已提交之后，attention 的距离仍由旧 L/F 描述；当前 token embedding 与合法 mask 则使用新值。这是合法的条件模型，但不是对“当前结构”持续计算局部 score 的积分器。

一个 V3 候选是区分 old 与确实可见的 current 几何两条信息通道，再训练相同可见性状态。不能在推理时把未知坐标的概率均值当真，也不能偷偷替换 old snapshot 来获得表面上的连续更新。

## 6. 重复修订与 C→D：完整假设攻击

### 6.1 直接反复 denoise 的两状态反例

令目标分布 $p=(0.9,0.1)$，forward noise 为 flip 概率 0.2 的对称矩阵 (C)。完美 Bayes denoiser 是

\[
D=\begin{pmatrix}.972973&.027027\\.692308&.307692\end{pmatrix}.
\]

直接把已经完整的状态当 noisy observation，再用 (D)，一次便有

\[
pD=(.944906,.055094)\ne p;
\qquad \pi_D=(.962428,.037572).
\]

即使模型是完美 posterior，重复它也会改变目标分布。若 toy energy 为 $(0,-10)$，平均能量从 $-1$ 变到约 $-0.3757$，更低能的少数状态反而减少。这是有限状态反例，不是对真实 V2 能量的拟合。

### 6.2 正确配对的 forward corruption 与 Bayes denoiser

在固定条件 (c) 下，若

\[
J(G,U)=p_*(G\mid c)C(U\mid G,c),\quad
D_*(G'\mid U,c)=J(G',U)/J(U),
\]

则

\[
K(G'\mid G,c)=\sum_U C(U\mid G,c)D_*(G'\mid U,c)
\]

满足 $p_*K=p_*$，并有 detailed balance。证明只需把 (J(U)) 消去并交换求和。这是 data-augmentation / Gibbs 构造；(C) 可以包含离散量化、拒绝或 fallback，**不要求单次 (C) 是 Gaussian，也不要求三档噪声构成 semigroup**。

CPU 中，同一个 (C,D) 恢复 $(.9,.1)$。但若 forward 改为 flip 0.05，仍用原混合噪声的 (D)，则得到 $(.933680,.066320)$。因此“只加一点低噪声”与“使用训练 unknown-mixture denoiser”不自动满足同一个 Bayes 构造。

### 6.3 V2 各因素是否破坏此前提

| 因素 | 最强支持/反对结论 | 需要实际核对 |
|---|---|---|
| 对象权重 0.5/0.5、各 position 正概率 | 固定 c/N 下是正权；无限容量时不改变每个条件分布的 Bayes optimum。不是原则性反例。 | 有限容量时 cell/coord 风险权衡、每轴 coverage。 |
| T=0.7 | train/sample 同 T，logits 可重参数化为 T·log posterior。不是原则性反例。 | 推理改温度就不再是该 posterior；不要用“低温更像 diffusion”跳过检查。 |
| 25% dense | 使用不同 task、条件状态；无限容量可同时拟合 marginals 与 prefix conditionals。 | 有限共享参数的梯度竞争与实际 joint conditional 误差。 |
| 50% noise metadata dropout | 独立隐藏时可学正确 unknown mixture posterior。 | 现 runtime 没传三分量 σ；fixed-low C 配 unknown D 可能不匹配。 |
| 八次 proposal / rejection | 仍定义合法概率核；它不是未截断 Gaussian。 | 使用实际完整规则，含因 G 而异的 acceptance 与 fallback。 |
| fallback 标签 | exhaustion 给 old=clean、σ=0；accepted-but-unchanged 仍标原 requested σ。可把 (U,applied-tag) 视为增广观测。 | known 路径要传真实 emission tag；unknown 要混合完整训练核。 |
| prefix 支持过滤 | **真实原则障碍。** 局部可达条件不等于完整合法结构 posterior。 | 全 teacher path reachability、目标/前序冲突、old admission 的分母。 |
| final support / rollback | 完美、全合法 posterior 下失败概率为零；有限模型时额外改变执行 kernel。 | 失败返回的是已加噪 U；外层是否需要退回原 G，并保存该新事务语义。 |
| 从 construct 输出出发 | stationary 只描述目标分布，不保证任意起点有限步的能量或 SUN 单调性。 | 一次修订、noise-only、C→D 的固定全请求配对。 |

runtime [processed_logits:175](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/programmed_path_runtime.py:175) 仅构造 unknown noise context；所以现成模型接口最直接对应 **训练原三档 mixture 的 C，再调用 unknown D**。若要 fixed-low σ，则必须显式支持三分量输入，不能把标量 mask ratio 当 σ。

### 6.4 累计 prefix admission 仍不是全局合法 posterior

取 clean 数据 $p(00)=.1,p(01)=.4,p(11)=.5$，完整合法集合 $S=\{00,11\}$。第一步两个首值都合法，CE 学 $p(a=0)=p(a=1)=.5$。第二步对 `01` 的非法 target 置零，于是合法 continuation 分别变成 `0→0`、`1→1`。

实际学得合法 joint 为 $(.5,.5)$，而全局目标 $p(\cdot|S)=(1/6,5/6)$。累计前序检查能防止“前序非法但后一步又偷偷算 loss”，却不会删除早期 target 中通向未来非法结构的质量。

在 repair 中，非法 clean target 经扰动后可能得到合法 old U，所以 old admission 不一定自动移除这个反例。若实际所有 clean targets 全路径可达，反例影响就消失；因此**冲突率是关键证据，不应先宣判此问题在 V2 中严重**。

### 小 trick 的优先级建议

值得做一次有界 paired 诊断，优先于盲目追加多次 full-cell repair：保持同一最终 V2 checkpoint、Plan/program、T=0.7、请求分母；对照 `原始G`、`一次D`、`noise-only U`、`训练同核 Cmix→Dunknown`。若另测 low-noise，应明确 σ 接口与 posterior 假设的差异。

失败路径不能让 noise-only 结构悄悄成为成功修复。可以另登记外层事务：inner repair 失败时回原 G；它是保守执行规则，不是假装已证明保持 Gibbs kernel。forward corruption 的离散量化/重试概率尚未积分时，只记录其 RNG/参数/提议，不宣称整个新路径 likelihood 已知。

当前没有执行这个 trick，也不建议据理论自动启动多轮搜索或修改 39993。

## 7. 数值表示、周期 alias 与离散局部 kernel

### 7.1 不是十进制数位模型，也不是仅 17 维 Fourier 的瓶颈

当前 `<X_037>` 是一个完整类别 token，模型不会自动把字符 0/3/7 当十进制数位做算术。长度 0.1 Å、角度 1°、fractional 0.01 的 codec 定义了最终离散测度。[fixed_slot config:153](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/fixed_slot.py:153)。

但“8 个 Fourier mode 限制模型只能输出平滑坐标分布”也不成立：[numeric adapter:178](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_repair_model.py:178) 之外，还有每个新 token 的 **full-rank 输入/输出 row residual**，[initialization:44](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_repair_initialization.py:44)。它可表示更尖锐的类别偏好。数值特征是归纳偏置，不是唯一输出通道。

固定 4 Å 立方晶胞中，独立均匀 sub-bin rounding 的单原子 Cartesian RMS 为约 **0.02 Å**；三角不等式给出的保守最坏误差上界是 0.06 Å，实际欧氏最坏界更紧。这个尺度可能影响陡峭能量面，但不足以独立证明 codec 是主要瓶颈。应先对原 clean 与同一 codec 的 roundtrip 做同协议上限诊断，而不是先换连续坐标头。

### 7.2 Alias 与温度顺序正确，但初始类别测度并非均匀

V2 保持 $g_0=\log(e^{z_0}+e^{z_{100}})$ 后再除 T；runtime 与 CE 对此一致。[objective:36](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_v2_objective.py:36)。它和先分别温度化两个 token 再相加不同，不能随意交换。

CPU：101 个 coordinate token 等 logits、T=0.7 时，物理类别 0 的概率是 **0.0264702**，其余类别各约 **0.00983363**；并非 100 个物理类别各 0.01。给两个 alias 的未温度化质量各分一半，可恢复物理均匀。这是表示冗余导致的 prior/measure 效应，**不是 logaddexp 写错**。

最强反驳：真实晶体坐标常有特殊位置和固定 origin，训练目标的 0 频率可能本就很高；输出行可学会补偿。只有出现 early zero-plane/alias pile-up 且持续影响有效性时，才值得把物理类别 prior 校准作为独立候选。不能仅凭这个 toy 就修改已验证的 alias 规则。

### 7.3 Quantize-Gaussian 不自动满足离散 semigroup

若每次先在连续轴加 Gaussian、再 round 回离散格点，一般有

\[
Q_{\sigma_1}Q_{\sigma_2}\ne Q_{\sqrt{\sigma_1^2+\sigma_2^2}}.
\]

8 格周期 toy 的 $\sigma_1=.2,\sigma_2=.4$ 最大矩阵差为 **0.0437688**。中间 round 改变了过程；不能把三档几何扰动按方差相减，就声称得到了正确离散 reverse steps。单次 corruption 仍是合法核，前节 C→D 的论证并不受此反例否定。

如果 V3 要有真正逐级 numerical diffusion，可定义周期 graph generator 或显式 (Q_t)，用矩阵乘积得到累计转移；coordinate 必须是 torus/circulant 语义，length/angle 不能照抄周期环。D3PM 的 ordinal Gaussian 只是参考，不是现成的晶格物理核。

此外，fractional 坐标独立等方差 noise 不等于 Cartesian 等方差：行向量晶格 L 下，所需 fractional covariance 是 $\sigma_F^2L^{-T}L^{-1}$。cell=(1,10,10) 的轴方差比为 100；三斜晶胞还含 off-diagonal coupling。若 lattice 也变，坐标噪声与 cell 要共同定义。局部逐 token walk 还可能在拥挤合法集合中不遍历，不能用它替代全局生成而不检查可达性。

## 8. 当前可达的 Gumbel 数值风险：工程修正，不算 SUN 假设

实际 V2 路径是 `250 → sample_state_programmed_paths.make_sampler → ProgrammedPathSampler._draw → _transaction_candidate_tokens`，不是只在 legacy closure 才调用。

- [250 wrapper:44](D:/codex_work/ai4s/DLM_periodic_self_repair/slurm/250_evaluate_periodic_dlm_v2.sbatch:44)
- [make_sampler:107](D:/codex_work/ai4s/DLM_periodic_self_repair/src/scripts/sample_state_programmed_paths.py:107)
- [runtime draw:204](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/programmed_path_runtime.py:204)
- [candidate sampler:499](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/spad_generation.py:499)

当前计算 $e^z/(-\log U)^T$ 后 argmax。实数算术上等价于 $z-T\log(-\log U)$，但 FP64 exp 有范围限制。

CPU 同一个 seed、相同类别相对 logits (0,2)，整体平移 0/+1000/−1000 后，现实现分别选 2/1/0；−1000 情况甚至选中预先被 mask 的低索引类别。log-space 形式三次都选 2，稳定 softmax 概率也一致。日志却使用稳定 log_softmax，所以这种极端情况下记录的概率与实际采样不相符。

**没有观测到真实 active logits 接近该范围。** 这是数值健壮性修正候选，不能解释现有 SUN，也不应当成为“新物理方法”的贡献。若主审采纳，需证明实际范围内固定 RNG 的选择一致、非法支持从不被选、极端平移下不溢出；不要把它混入声称科学改进的 ablation。

## 9. Frozen 随机 soft Plan：条件独立命题及最强反驳

设 (c) 是完整 chemical/metadata 条件、冻结权重视为常量，$G$ 为 clean target。如果 soft $S\sim\pi(S|c)$ 的抽样独立于 G，而输出 target 原样保留，则

\[
p_{train}(G,S\mid c)=p_{data}(G\mid c)\pi(S\mid c),
\qquad p_{train}(G\mid c,S)=p_{data}(G\mid c).
\]

若 program (P=h(c,S))，同样有 $p(G|c,S,P)=p(G|c)$。CPU toy 的 mutual information 约为数值零；这证明“匹配部署随机 soft＋不改 clean target”本身不能识别“按该 soft 实现几何意图”。

真实代码确实只替换 Plan 三 soft 字段及 program，不改 answer，[plan_data:294](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_v2_plan_data.py:294)、[finish_source:366](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_v2_plan_data.py:366)。三字段在 [native prompt](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/c3fd_native_plan.py:139) 中明确叫 soft hints；运行合法支持不强制 SG、lattice_system 或 VPA-bin 服从性。

### 不能直接套用到本项目的地方

1. c 不能只写组成后，忽略输入的 `stability_condition`、原化学 transcript、oxidation-state/action history、ledger 或可能与 G 相关的 metadata。若它们包含结构相关信息，条件独立需要在更完整 c 下重新陈述。
2. 同一 frozen Planner 曾用同源数据训练，有限数据的模型记忆和 source-seeded 一次抽样可能留下经验相关性；随机机制独立不等于一份有限训练表的经验 MI 恰为零。
3. 原 MP20 近唯一组成很多，同 c 的可比多晶型不足时，“S 可忽略”与“模型借 S 记忆/简化映射”无法仅从训练 loss 识别。
4. 约 9.5% teacher-soft fallback 不是随机预测；CPU 仅作为示意的混合模型可产生非零 MI。真实 fallback 所在 c/goal 分布不同，需要分层实测，不能把同一 toy 比例当成本项目 MI。
5. 即使 S 不增加 Shannon 信息，冻结 LLM 提供的确定性特征和 P 的 factorization 仍可能降低有限模型的学习/计算难度。信息冗余不等于路由无用。

还有一个更强的条件化边界：如果真实 Planner 使用 $\pi(S\mid c,M)$，其中 M 是与 G 相关的 goal/chemical transcript，则

\[
p(G,S\mid c)=\sum_M p(G,M\mid c)\pi(S\mid c,M)
\]

一般不能分解。即使 $G\perp S\mid(c,M)$ 成立，M 也未必全部显式进入 DLM native prompt；此时 S/P 可以成为向 DLM 传递 M 的渠道。不能先把 M 加入理论 c 消掉其信息量，再从实际模型输入省掉 M，却据此宣布 S 无用。当前 native prompt 的明确字段并不包含完整 typed ledger 与 stability goal。**没有直接读取 teacher soft/geometry，只能排除该直接输入路径；不能证明所有间接相关性不存在。**

主审提供的 V2 direct 预测覆盖为 train 24558、fallback 2578；val direct 8147、fallback 900。direct soft 对原注释的 agreement 为 lattice 14103/24558、SG 13987/24558、volume 16217/24558。这些数目显示提示并非随机猜测，但**annotation agreement 不是 DLM 生成后对提示的服从性，也不是提示对 SUN 的因果贡献**。

### 最小证伪与自然 LLM+DLM 候选

- 同一 c/source 下换 S，并重新由 pointer 得到实际 P；比较 target conditional CE、实际几何标签 adherence 和 SUN，另分 P 固定/S 换、S 固定/P 换。只有控制了 program 的变化，才能区分语义提示和访问顺序作用。
- 分 direct / teacher fallback、重复组成 / 近唯一组成、goal 层报告。不能从混合平均推出全源控制机制。
- 若目标是**可控 latent plan**，更清楚的故事是 $p(G,S|c)=\pi_{LLM}(S|c)p_{DLM}(G|c,S)$，其中训练的 requested S 与 target G 有可核验语义关系；部署再从 LLM 选择 S。teacher 真实 Plan、预测不确定性特征、条件 dropout 各是待比较方案，而不是预定答案。
- requested S 与预测先验 $\pi(S|c)$ 可以分开：前者定义目标，后者表达 LLM 的信念。不能把“抽了哪个不一定正确的预测标签”自动等同于用户请求。
- 原连续 CIF 的 SG 注释不必在粗量化 body 上精确保留。若把 SG/几何标签升级为可控目标，先核查标签、codec 和评估容差，不能把 soft SG 当作已知 Wyckoff 硬约束。

这部分可能需要改变论文主张：如果当前 P 的作用主要是计算路由，就应证明路由收益，而不是先宣称 LLM 的 soft Plan 在控制精细几何。

## 10. 为什么 CE、合法性、teacher 改善都不等于稳定性保证

### CE 可以有保证，但当前没有测得所需前提

一个离散 toy：数据只有好状态；模型给坏状态 ε 概率，则数据 CE 为 $-\log(1-\epsilon)$，可非常小；若坏状态能量代价 M 很大，期望代价是 εM。固定有限晶体网格上理论上存在能量上界，但我们没有一个足够小且可用的 M，平均 CE 也不是实际未知条件下的 joint KL 上界。

最强反驳是：若能证明目标 joint 与模型 joint 的 KL 很小，并且目标稳定集合质量高，TV/Pinsker 的确能约束稳定事件概率。不能说 CE 在任何假设下都不可能带来稳定性保证；应说本项目的训练风险、经验 validation 与 batch-level uniqueness 尚未建立这个保证。

### 合法域远大于低能域

0.5 Å 门槛对 (0.6 Å,1.2 Å) 两种距离都放行，但纯排斥 proxy $r^{-12}$ 相差 4096 倍。这不是 CHGNet 结果，而是“满足底线”不控制能量尺度的反例。N/U、SG、force、A 与 B 也不能互相替代。

### 旧 K4/K8 的概率层不能混用

[既有只读 teacher 诊断](historical/docs/teacher_feedback_unified_v1/27_SELF_IMPROVEMENT_REPAIR_PLAN.md) 记录：972 条 verified 路径覆盖 526 个条件，等组成 A 从 6.538571 到 6.516099 eV/atom，只有约 0.3437% teacher 相对改善；另外 498 个条件无 verified 路径。这个有限候选池不能靠重加权产生池里不存在的几何。

但其中存在 Pareto 改善信号，而且修正后的 K8 独立 1000 结果明显不是“几乎全失败”。因此反驳应该是“偏好有限、迁移与 student 漂移未被保证”，不是“teacher 没有任何信号”或“扩大 K 必定无用”。

teacher 的经验 KL 预算不是 student 相对参考策略的 KL 约束；组间平均 A/B 改善也不是每组成/每起点单调改善。原生 A 增大可以同时包含 raw E 上升与 eR 降低，必须在同一 finite/verified 配对集合中分解，不能拿另一分母的 raw 平均来解释。

## 11. 跨版本失败与成功：本 lane 的可证伪根因表

| 变化/现象 | 有证据的机制 | 最强竞争解释或反证 | 最小证伪实验 |
|---|---|---|---|
| canonical slot 对齐曾提高 Direct | 训练 target 与推理硬预填 slot 的身份错配被修正；该约束不需要模型学会化学计数 | Direct 增益并未保证 raw energy/SUN；历史组合版本不是单模块因果实验 | 固定相同 prompt/program，统计强制 N/E 与 target slot 一致性及 parser/geometry 分解 |
| 旧 K4/K8 reweight 支持有限 | 低 A 候选稀少、teacher 改善幅度小；student 只近似吸收路径偏好 | 有 Pareto 信号，独立 SUN 也并非零；还可能是 optimizer/支持概率学习不足 | 在同池上分别核对候选下界、teacher 改善、student 条件 log-prob 与真实 endpoint，四层分开 |
| raw-v1 synthetic CE 改善未带来目标 SUN | smoothing 熵底随 epoch 改变；单 repair scalar 覆盖稀少；teacher prefix/synthetic old 与部署不同 | 也改变了 raw 初始化和可训练表行；不能把失败单归为 synthetic noise | 固定 exact-target CE、按 source/axis/任务分层和完整生成对照 |
| V2 修复上述 loss/condition/geometry 接口 | exact CE、inverse inclusion、累计 prefix audit、预测 program、head-specific bias 均有真实代码 | 很多同时改动；梯度存在≠使用有效；early cell 没有几何 | final construction vs repair、geometry/logit influence、匹配 ablation；先看已注册结果 |
| 两 epoch 某些 source 没有直接 coord-repair 标签 | 按现分支概率，未获得任何 coord repair target 的概率在 N=1 约17.7%、N=20约14.1%，还未计 admission；无coord-prefix概率39.1% | 参数跨 source 共享，dense 一次可教多个位置；没有“每个源每个 token 必须覆盖”的定理 | 实际 target_coverage 与 val误差按缺监督源/轴分层；等算力改sampling需另外登记 |
| DLM 不像逐级几何 diffusion | 当前是全局类别恢复+程序访问，而非数值邻近 Markov 逆步；old-view 固定 | 完美离散 conditional 仍可表示正确 joint；机制名称不决定SUN | 先测 once C→D 和反复D的对照；再考虑有明确定义的局部Q_t而非盲加步数 |
| 预测 Plan 可能被忽略 | 在充分条件独立模型下随机 S 不提供目标可识别性 | 同源训练、metadata、fallback、有限容量和程序路由会产生作用 | S/P正交干预、adherence与source分层，不能只看soft agreement |

这里的 source 覆盖概率是生成机制的期望，不是本次实际计数；CPU JSON 给出公式。实际训练还有 padding、support admission、共享参数与固定随机表，必须以记录核对。

## 12. 首轮优先建议与暂不接受的主张

**优先诊断，不先改故事：**

1. 读取完整 V2 final 的分任务/分轴 CE、prefix conflicts、old admission、corruption rejection/fallback、几何 influence 与 construction/repair 配对终点。保留全请求失败；对 missing hull 使用已补完整的评估，不用过期 cache 得出模型结论。
2. 对 Cmix→D 的候选进行一次有界、预先固定的配对检查，保留 noise-only 和 unchanged/rollback 样本，保持同一 T；若支持冲突明显，stationarity 只作理想动机，不能作当前实现结论。
3. 将 Planner 语义控制与程序路由分开验证。目标若是可控结构 latent，训练关系必须能识别该 latent；若实际仅作为计算路由，就证明路由收益。
4. 数值 Gumbel log-space 改写作为健壮性修正单独处理，不计入 SUN 科学候选。

**需要 V3 设计而不是一个形容词的小改动：**

- 真正数值局部、周期一致的 discrete transition / noise-time / reverse kernel；晶格与坐标的耦合、合法图遍历性及低噪边界必须明确。
- 同时可读真实 old 和真实可见 current 的几何交互，训练与推理同接口；不从概率平均造“当前坐标”。
- 若采用多尺度 token 或 sub-bin 表示，先声明离散测度与 alias；连续 offset 不能默默接管原生坐标，再把它称为 DLM 本体成功。

**本轮拒绝直接作出的推论：**

- 没有 t embedding ⇒ 不是 diffusion / 模型一定坏。
- 8-mode Fourier ⇒ 输出不能尖锐（full-rank 新行反驳）。
- 多头几何梯度非零 ⇒ 几何已用于关键动作并提高 SUN。
- 更低 CE / teacher KL / 多次 repair ⇒ 低能盆地概率单调改善。
- 把独立连续生成器接到后面 ⇒ LLM+DLM 已自然成为晶体 diffusion。
- 完成同一项目的所有工程检查 ⇒ 已证明哪个模块必要或哪个失败原因唯一。

首轮已经得到可复算的反例与可实施的证伪方案，但没有产生新模型性能结论。下一轮交叉反驳应优先攻击本报告的三项假设：**C→D 的 posterior 前提是否实际近似成立；soft 条件是否真的在充分 c 下独立；early-geometry 路由瓶颈是否被多层 Transformer 有效绕过。**
