# V3 新规格第一轮独立数学审稿

2026-09-06。审稿对象：另一作者的 [V3_DLM_CANDIDATES_AND_ATTACKS.md](V3_DLM_CANDIDATES_AND_ATTACKS.md)。本轮读取版本 SHA256 为 `bd98573df716b73c867e5624721e2babf982dcaa58b247e868eec9b260a3d366`，完整 [审阅快照](evidence_v3_r1_math/reviewed/V3_DLM_CANDIDATES_AND_ATTACKS.md) 与 [来源清单](evidence_v3_r1_math/source_manifest.json) 已保存。源码依据固定执行 SHA `2e904c260bafb6750c9a5dbb0d920c4ff8c3a868`。

这是 V3 新规格的第一轮，不是旧报告的 round2。本审稿人是另一份几何头反提案的作者；本文只把 H-P/H-C 当作说明取舍的比较对象，**不为自己写的候选作通过裁决**。本轮只写本报告及其独立 NumPy/stdlib 证明，没有改受审文档或模型，没有 SSH、GPU、子代理、模型调用、MLIP 或新训练。

## 1. 裁决：三个候选已有可讨论的数学对象，但尚不能选架构

**B 的最小执行对象最接近现有代码；A 的语义联合分布在其支持内成立；C 的 forward、normalized-score 和有限反向核可以定义一个生成器。C 不能凭这些事实升级为“64 步已近似正确”或“程序 clock 有物理贡献”。** 本轮给出两个量化反例：

- C 的末步仍重新加入坐标噪声。`γ=1` 与 `γ=1.5` 的末步标准差分别为 `0.001` 与 `0.00003162`。对紧邻原 codec 边界的同一个精确点目标，即使 score 无误，离开原量化 bin 的概率也可由 **46.017%** 变为 **0.07827%**。这足以制造与 P 相关的输出差异，却不包含程序改善物理结构的证据。
- C 的 lattice reverse 使用固定 `β̃_k`。即使训练数据与先验都是标准 Gaussian、模型输出精确 marginal ε，64 步后的方差仍可从 1 降至 **0.8631 或 0.8530**。这是一个规范化但带偏的有限反向模型，不是其概率对象非法，也不是 SUN 失败预测。

应分开四层结论：

| 层次 | 本轮能够判断 | 本轮不能替代的证据 |
|---|---|---|
| 生成器定义 | kernel、初始分布、rollback、失败状态和最终测度是否明确 | 实际实现是否完全遵守定义 |
| 反向近似 | 正确 score 放入有限算法后是否仍有系统偏差 | 实际网络/实际结构上的误差幅度 |
| 学习与计算 | live 参数路径、targets/forward、NFE、初始化是否自洽 | 1–2 epoch 的真实可学性、显存/吞吐 |
| 物理目的 | 哪些指标与理论没有推出关系 | Strict/Meta SUN、新颖性、终态有效性与共同 refiner 的真实配对结果 |

受审文档已经明确承认许多边界，这应保留。下列反例不是为了把“承认近似”改判为“数学错误”，而是防止在方案选择时低估近似大小、实现缺口和归因成本。

## 2. C 的 forward 与先验：主要成立，prior 精度不能免费取得

### 2.1 Lattice chart 与 VP chain 是一致的

row-lattice 下 `G=LLᵀ`，`S_L=½log G`，`v=tr S_L−log N`，五个正交归一 traceless basis 提供 shape 通道。逆变换的体积项 `(v+log N)I/3` 正确。这里 v 是 log volume per atom，**不同于**把归一化第六 basis 的 coefficient 当 v；受审文档没有再犯这个混用错误。

`z=D⁻¹(w−μ)` 中 train-only 正尺度给出六维 Euclidean chart，噪声与 score 都可以在这个 chart 定义；不必额外把它包装为某种 intrinsic SPD Brownian motion。建议仅补上 `½log(LLᵀ/ℓ₀²)` 与 `ℓ₀=1 Å`，并明确 `chol_lower`，消除矩阵 log 单位和库函数约定的歧义。`GL(3,Z)`、origin 和 site permutation 不变性仍没有由该 chart 解决，原文对此的限制正确。

`ᾱ_k=exp[-λ_z(k/T)²]` 与 `β_k=1−ᾱ_k/ᾱ_(k−1)` 对 λ>0 给出合法 VP Markov chain。文中的条件 KL

\[
\mathrm{KL}\{q(z_T|z_0)\|N(0,I_6)\}
={1\over2}\left[\bar\alpha_T\|z_0\|^2+6\{-\bar\alpha_T-\log(1-\bar\alpha_T)\}\right]
\]

正确。训练 source 上逐点上界也能控制其经验混合的 prior mismatch；它不自动控制未见组成/超出训练 chart 范围的结构。最大条件 KL≤`10⁻⁶` 只是一个拟议数值标准，不是物理必要阈值。

这个标准有代价。以 **示例而非实测** 的最大 `||z₀||²=6` 或 `100` 求 λ，得到：

| 示例最大平方范数 | 达到 KL≤10⁻⁶ 的 λ | ᾱ_T | T=64 的最大 β_k |
|---:|---:|---:|---:|
| 6 | 14.914123 | 3.33333×10⁻⁷ | 0.37025 |
| 100 | 17.727534 | 2.00000×10⁻⁸ | 0.42285 |

这不是很细的每步 VP 变化。把 terminal prior 匹配设得很精确，与固定 64 步的反向近似误差可能构成取舍。不能只展示很小的 prior KL，再把有限反向 kernel 当成误差也同样小。应先固定 λ 的规则，再对实际 chart 与算法的误差作诊断；不要求为了这个问题新增复杂 teacher。

### 2.2 Species clock 是合法异质 torus 时间表

受审定义 `σ_i(k)=σ_base(k)^{γ_i(P)}`，`γ_i∈[1,1.5]`，`σ_base` 递增，故 `Δv_i≥0`，能构成各站不同速度的 torus Brownian marginal/增量。m=1 单独设 γ=1 避开了除零。所有站在 k=T 都有 σ=1，uniform terminal approximation 的 Fourier 界与作者 CPU 结果一致；最多 60 个坐标分量约 `1.61×10⁻⁷`。这部分没有必要推翻。

fractional isotropic covariance 转成 Cartesian 列坐标是 `σ_i²LᵀL`，原文说明了它不是 isotropic Å noise。对训练的完整 `(z_k,F_k)` 采用联合网络，可以学习 joint noisy marginal score；forward 条件独立不等于 reverse 物理独立。原文这些区分正确。

但“前序 species 在 reverse 中更早恢复结构”应改成更窄的陈述：**该 schedule 使其目标边缘在同一 k 处于较低噪声；有限模型是否更早恢复有用结构尚无证明。** 正确连续反向在正确先验下最终仍针对同一数据分布；P-clock 主要改变 finite-step approximation、有限网络的任务分配和优化难度，不自动创造新的结构约束或能量偏好。

## 3. C 的 normalized score 与反向式：符号正确，端点是实质近似

### 3.1 `u=σs` 的缩放没有新增奇异性

wrapped-normal 像和 score 公式正确；用 axis factorization 计算它合法。作者已给出的 `δ=0.49,σ=0.2` 反例正确否定了全时刻 MIC 替代。补充一点：sampled `−ε` 本身作为随机 target 仍满足 `E[−ε|F_k,F₀]=σs*`，所以原始 ε regression 不是仅因 wrap 就数学错误；显式像和选择的是低方差条件均值 target。

在 k≥1、σ>0，文中的反向坐标均值项

\[
\Delta v_i(k)\,\hat u_i/\sigma_i(k)
\]

符号与 reverse Brownian Euler 一致，也可以直接计算为 `(Δv_i/σ_i)u_i`。小 σ 不意味着该 **整体更新增益** 爆炸，因为 `0≤Δv_i/σ_i≤σ_i`。不应把单独的 `u/σ` 数值很大误判为采样 drift 必然不稳定。

但它只是有限 reverse Euler 核。它没有积分掉 clean latent，也未等于有限时间真实 backward heat kernel；把正确 marginal score 插入正确方向的 Euler 步仍有离散化误差。这一点尤其发生在最后一步。

### 3.2 最强端点反例：完美点目标仍被重新加噪

在低噪声且唯一局部像支配时，考虑固定 clean coordinate `f₀`：

\[
s^*(f_1|f_0)=-(f_1-f_0)/\sigma_1^2,
\quad \Delta v(1)=\sigma_1^2.
\]

作者最后一步给出

\[
f_{\rm out}=\operatorname{wrap}\{f_1+\sigma_1^2s^*(f_1|f_0)+\sigma_1\xi\}
=\operatorname{wrap}(f_0+\sigma_1\xi).
\]

也就是即使均值完全恢复，输出仍不是该点目标。torus 多像在这些 σ 下的修正是指数小量，不改变该反例的实质。残余噪声不是编码错误，而是已声明采样器的输出分布。

现在使用原 0.01 fractional nearest-bin codec，取 `f₀=0.0049`，离 0.005 边界只有 0.0001。针对同一个原目标、同一个完美局部 score，独立 CPU 得到：

| γ | σ_i(1) | 输出离开原 bin 的概率 | k=2…64 的相对方差步 `Δv/σ_k²` | 64 时刻中 σ<0.005 的个数 |
|---:|---:|---:|---:|---:|
| 1 | 0.001 | 0.460172 | 0.196914 | 15 |
| 1.25 | 0.00017783 | 0.286942 | 0.239757 | 25 |
| 1.5 | 0.00003162 | 0.00078270 | 0.280314 | 31 |

因此当前 clock 同时改变了至少三个对象：谁先进入低噪声区、哪些 species 得到更多 sub-bin denoising 时间、哪些 species 的最终随机量化误差更小。γ 越大，中间每步相对方差跳跃反而更大。learned P 相对 canonical P 的优势如果存在，可能来自这些数值分配，而非更符合化学传播顺序。

这不证明 clock 必败，也不证明所有实际坐标都靠近 bin 边界。它证明了**仅比较最终 P 的 SUN，不能把差别直接归因为规划结构恢复顺序**。

最小修订有两个层次，不必马上增机制：

1. 若程序物理性没有独立证据，首个 mixed-space 定义可以用所有 species 相同的 clock，明确 P 只作条件。这是可接受的研究主线变化，不应因论文故事需要而排除。
2. 若保留 rank clock 为明确待证假说，必须单列其不同 `σ_i(1)`、有限步误差与输出量化作用。若作者选择让终端低噪声尺度一致，需给出新的时间表及核；不能一边按现公式实现、一边宣称已消除端点混杂。取消末步噪声或添加 terminal denoise 同样是新采样核，也必须登记，不能称“免费修正”。

## 4. C 的 lattice reverse：合法 Gaussian kernel 不等于正确边缘

### 4.1 一步 μ 的数值形式可以直接简化

原文经 `hat z₀` 计算 posterior mean 的代数正确，但可以等价写成

\[
\mu_k={1\over\sqrt{1-\beta_k}}
\left[z_k-{\beta_k\over\sqrt{1-\bar\alpha_k}}\hat\epsilon_\theta\right].
\]

这消除了中间显式除以 `sqrt(ᾱ_k)` 的巨大 x₀ 值。独立 CPU 与原式之差仅约 `10⁻¹⁵`。所以不能声称“ᾱ_T 很小必然令单步 reverse μ 爆炸”；可直接使用稳定形式而不改变候选的概率分布。若 x₀ 仅用于诊断，仍需报告其条件数，不在该中间量上悄悄 clip。

另一方面，若新 ε head 零初始化，则 μ=`z_k/sqrt(1−β_k)`；全部步的确定性增益为 `1/sqrt(ᾱ_T)`，带噪声后更大。上述两个示例 rate 下，整个未训练模型最终 z 标准差分别约 **2,216.75** 与 **8,900.28**。这只是 **零输出参考模型的数值风险**，不是已经训练 C 的结果；任意新生成器在训练前都可能很差，不能用此数值直接否决它。但它说明“head 很小，任意初始化都差不多”不成立。

可以选择明确的 Gaussian-reference 输出、v/preconditioning 或其他已有数值参数化；选择若同时改变 loss 的时间权重，就不是纯 algebra refactor。本轮只要求把实际初始化与训练对象写清，不据此批准本审稿人另一候选中的参数化。

### 4.2 精确 marginal ε 的 Gaussian 反例

令真实数据 `z₀∼N(0,I)`。VP 的所有边缘和 terminal 都恰好是 `N(0,I)`，没有 terminal-prior mismatch。精确回归输出是

\[
E[\epsilon\mid z_k]=\sqrt{1-\bar\alpha_k}\,z_k.
\]

代回作者 μ，得到 `μ_k=√(1−β_k)z_k`。然而真实 Gaussian 反向条件的方差为 **β_k**；作者选择的是给定 clean z₀ 的 posterior 方差 `β̃_k≤β_k`。积分掉未知 clean 的条件方差贡献没有被这个固定方差保留。

于是从 unit variance 开始，实际 variance 递推为

\[
V_{k-1}=(1-\beta_k)V_k+\widetilde\beta_k.
\]

64 步的确定性 CPU 递推给出：

| λ | true final variance | 候选 final variance，oracle ε | 方差缺口 |
|---:|---:|---:|---:|
| 14.914123 | 1 | 0.8631366 | 13.69% |
| 17.727534 | 1 | 0.8529514 | 14.70% |

这里 λ 来自上一节的示例率；Gaussian population 并不具有有限最大范数，**没有把这两个例子的 max-norm KL 假设偷渡到 Gaussian population**。任何固定 λ 都可以单独用这份 Gaussian population 检查 reverse；该 population 的 prior 本来就精确。

这是比“模型可能没学好”更强的算法反例：score 完美时仍有误差。它也比“factorized reverse 可能丢联合相关”更简单，因为一维就能出现。原文已承认 finite posterior approximation，因此应把它写成可量化的取舍，不能说 generator 未规范化。对某些任务这类固定 variance 近似可能仍有效；是否需要改 variance、步数或 sampler，必须基于明确的候选核和实际误差，不自动叠加 learned-variance 网络。

## 5. C 的最终测度、full-float 接口与 live 模块

### 5.1 采用概率测度可同时覆盖 delta 与失败

原文明确 k=1 lattice variance=0，允许 deterministic terminal map。这样的 normalized kernel 可以产生奇异分布，例如模型让全部 `hat z₀=0`，最终 lattice chart 就是 δ₀。不能无条件假设最终 `p_C(x)` 一定有相对于六维 Lebesgue 的普通 density。

建议把 token 概率统一写为

\[
P_C(y)=\int \mathbf1\{Q(x)=y\}\,P_{\theta,C}(dx),
\qquad Q:\mathcal X\to\mathcal Y\cup\{\bot\}.
\]

其中 `⊥` 包含 nonfinite、codec 越界/裁剪、量化后 geometry gate 失败等预先定义状态。这个修订不改变采样器；它让 source chart、terminal delta 和失败质量处于同一概率空间。只有在证明存在 density 时，才换写 `∫_(Q⁻¹y)p(x)dx`。不能把三乘三矩阵 L 的九维 ambient Lebesgue 测度冒充六维 chart 测度。

原文对 0/100 alias 的处理是正确的：连续物理点的量化原像只有一份质量，最终 canonical 0 不应再次凭空做两个 token logits 的 `logaddexp`。也正确区分了连续 CIF 与原 codec pushforward：本候选若与 A/B 作正式旧接口对照，codec 后对象是主终点；continuous-only 改善不能替代这个终点。若研究任务改成 mixed native CIF，必须另行改写主线和对照。

### 5.2 原始连续 CIF 是精度证据的必要条件，不是所有机制比较的必要条件

受审文档要求原始 CIF 按 source identity、cell convention 和 exact site permutation 全量对齐，正确地排除了随机补小数和筛小子集。还可区分一个更便宜、但主张更窄的备选：用当前 codec clean body 解码作为连续噪声的目标，仍可检验 full-noise state、dense target 和 reverse algorithm；只是不能声称学到了原标签没有的 sub-bin 精度。

所以“没有恢复全部原始 CIF”可否决 **原始连续监督的正式候选 C**，但不数学否决共享连续 heads 这一机制族。如果作者希望使用 decoded labels 作同数据机制比较，应明确另立这个精度合同，不在结果后混选 endpoint。任何 source 对齐失败都不能靠删除请求或训练来源掩盖。

### 5.3 当前输入与初始化规格还不够执行

数值槽采用 MASK、固定 placeholders 还是其他 token ids，连续 state 经什么显式 API 输入，legacy old-token 解码如何 bypass，time/σ 的单位如何与已有 task/noise features 隔离，仍应写明。原则“真实 continuous current”是对的，但调用现有 wrapper 并传一份外部 float 字典不会自动满足它；原代码的 geometry_inputs 仍从整数 old token 解析。

当最低 σ 达 `3.16×10⁻⁵`、31/64 时刻低于半个旧 coord bin 时，还应检查 float state 的微小差异经过 FP32 feature→projection→BF16 embeddings 后是否可见。没有理由断言它必然丢失，投影可能放大差异；也不能因为入口 dtype 是 float 就保证全部 terminal 噪声信息已到达 hidden。此处真正需要的是按所选 σ 的输入敏感度/梯度验收，而不是另写许多防御层。

初始化也需单一表述。§5.5/5.8 的“新 LoRA、同一原始 LLaDA”较像 fresh adaptation，§5.9 又称 V2 可复用模块；它没有明确承诺完整加载 V2 final。两个合法选择不同：

- 从相同 raw base 和 fresh adapters 开始：与原 V2 两 epoch 的曝光较公平，但不能借用“已有 V2 适配成功”支持可学性。
- 完整载入 V2 final 后继续：更可能复用已学几何输入与 LoRA，但总训练历史多了 V2 两 epoch；与 fresh A/C 或原 V2 作成本/效果归因时必须列出累计曝光。

把 fresh LoRA 与未说明来源的训练后 conditioner 混搭，也应视为实际新初始化，不能仅用“同一个 LLaDA”概括。两种初始化都不等于随机 8B 从零，但它们的短预算难度并不相同。

### 5.4 score-only 分支不能继承 36.99M 的整个训练清单

静态源码确认 [periodic_repair_model.py](evidence_v3_r1_math/frozen/src/crystal_dlm/periodic_repair_model.py.txt) 的 numeric adapter 只修改 logits；[periodic_repair_initialization.py](evidence_v3_r1_math/frozen/src/crystal_dlm/periodic_repair_initialization.py.txt) 的 `output_delta` 也在 hidden 下游。G loss 只读 hidden 时，它们没有 loss 路径。

| 旧模块 | 参数库存 | score-only 路径 |
|---|---:|---|
| `new_token_rows.output_delta` | 10,162,176 | 不用，应冻结并跳过其计算 |
| `numeric_adapter` | 651,264 | 不用，应冻结并跳过其计算 |
| `new_token_rows.input_delta` | 10,162,176 | 读到的 N/E/prompt rows 可用；若 numeric 全 MASK，相应 body rows 无直接 score 梯度 |
| `repair_task_projection` | 36,864 | 取决于新 task API 是否使用它；不能默认为活跃 |
| LoRA | 14,680,064 | 必须从新 heads 经 hidden 反传；不是仅 head 在变 |

因此，至少 `10,813,440` 个旧 output-only 参数不能算进 C 的 active trainable inventory。若同时 bypass 旧 task projection，旧模块的整 tensor 库存上界为 `26,142,928`；还含 batch 未出现的 input rows/常数特征列。新连续 input/time 模块在受审规格尚未定形，**此时没有证据给出精确的新总参数或吞吐**。作者提到了 output rows，应再补 numeric adapter、task projection、实际输入 rows 的区分。

原 [V2 trainer](evidence_v3_r1_math/frozen/src/scripts/train_periodic_dlm_v2.py.txt) 126 行已使用 `find_unused_parameters=True`；无须为了处理这件事发明新的训练框架。C 若纯 G，直接冻结不参与的模块，forward 只返回参与该 loss 的几何输出；若保留 token branch，则按两任务实际图处理 unused parameters，另算 CE 曝光。不要计算并返回大量不用的 token logits，然后把名义全参数参与当作共享学习证据。

零 head 第一步可能只有 output head 获得非零梯度，后续主干梯度才打开，这不自动是 detach bug；验收应覆盖不止首步。反过来，zero-loss 乘上所有参数让 DDP 满足形式上的“被用到”，也没有科学必要。

## 6. A：target alignment 改变联合分布，但固定 C 下的控制仍可能不可识别

### 6.1 对齐标签支持理论动机，不能拿全局 MI 替代条件 MI

定义 `S=a_δ(G)` 与 frozen `ρ(P|c,m,S)` 得到合法的训练联合关系；相较独立抽 S 再配同一 G，它确实改变了训练对象。`a_δ` 应从实际 codec 输出而非原始 CIF SG 直接定义，原文对此的区分正确。A 的关键风险不是“监督标签能否与几何相关”这个一般问题，而是实际数据是否支持固定条件下可识别的请求语义。

最强小反例：两条训练记录分别 `(c,G,S)=(0,0,0)` 与 `(1,1,1)`，各占一半。全局 `I(G;S)=log 2`，但 `I(G;S|c)=0`。`G=c`（完全忽略 S）与 `G=S`（使用 S）都完美拟合训练数据；对 `c=0,S=1` 给出相反输出。对齐标签没有在这些 off-support 干预中识别哪种语义正确。

原文的 fixed-c 二状态 toy 本身成立；它只是不能替代真实 MP20 条件覆盖测量。现有 exact-C 多行组和 chemical-M 唯一性统计，也没有证明这些多行组具有 **不同的 `a_δ(G)`**。应优先检查重复 exact-C 内目标属性的真实变异及联合组合，而不是将无条件 SG/VPA 熵当固定 C 的 controllability 证据。

多数 C 近唯一不使条件模型完全无用：跨组成的共享表征可以学到可泛化关系。但这种泛化需要证据，不由经验联合分布唯一地识别。A 可以是较有意义的语义监督改动，尚不能称已解决可控几何。

### 6.2 冻结独立 prior 的联合支持比单字段校准更重要

原文已给出 unsupported request 的反例；独立 CPU 将其量化：假设实际属性映射只可能产生 `000` 和 `111` 两种三字段值，边缘各是均匀 Bernoulli。三个独立 prior heads 即使每个 marginal 完全校准，仍会给八种组合各 1/8，仅有 **1/4** 的请求拥有该属性函数的非空原像。完美条件模型也不能让其余 3/4 同时服从三个属性。

这只是 semantic-support 反例，**不是**说实际项目 75% Plan 物理不可能，也不是 SUN≤25% 的推导。需要区分：

- 在有限训练集中没有出现的组合；
- 实际属性函数在给定 c 下数学上没有原像的组合；
- 有原像但稳定/新颖性不佳的组合。

这三者的物理含义不同。神经模型在没有训练 posterior 的条件下仍能定义规范化输出，因此 A 的 generator 不会仅因 unsupported S 就失去归一化；失去的是被监督识别的控制含义。

冻结 LLM prior 在新 `a_δ` 语义下是否仍匹配，需要重新检查。不能为了补这个缺口临时重训 prior、改 SG hard mask、筛掉不符 S 的来源，同时继续称 A 只改变标签。unknown 的表示若超出 frozen vocabulary，也必须进入属性/先验接口说明。原文选择失败和不服从保留在全请求分母是正确的。

### 6.3 P 和 m 的条件化边界

如果 `P=ρ(c,m,S)` 是确定性 greedy 编译，换 S 同时重编 P 会同时改变条件与执行顺序；固定 P 换 S 则可能落到训练联合支持之外。前者不能单独归因于 S，后者只是模型函数敏感度诊断，不能无条件当成真实可实现的语义 intervention。作者已要求区分两种操作，应继续把这个支持边界写清。

另外，训练形式写了 `q_data(G|c,m)`，DLM 却只输入 `c,S,P`。当前有限表中 m 对 exact-C 唯一，且 goal 固定，这让眼前问题较小；一般若 m 仍有不能被 `(c,S,P)` 表达的几何信息，DLM 只能拟合将 m 积分后的条件。不要把未观察 m 的条件模型自动称为每个 m 下完全匹配的 joint。无需为这个抽象边界立即增加一个新 m encoder；先保持实际输入定义和证据范围一致。

## 7. B：正确的最小核还需要正确的事务事件判定

### 7.1 原式 C_mix 和 unknown posterior 的对应基本成立

源码 [periodic_v2_corruption.py](evidence_v3_r1_math/frozen/src/crystal_dlm/periodic_v2_corruption.py.txt) 明确在重试前一次性均匀选择三档之一，同一档最多八次 proposal；不是每次 rejection 再抽 level。其 kernel 包含 log-SPD/affine Cartesian 污染、量化、alias canonicalization、原 complete support 检查、重试，以及 exhaustion 时回 clean。

accepted 但量化未改变的 proposal 仍保留 requested σ；exhaustion 才把 applied σ 置零。若 B 在 unknown metadata 模式运行，保留完整层级混合与 fallback 的边缘语义即可；不能只保留接受的样本、每次 attempt 重抽 σ、或只选 low-noise，再继续称训练同核。训练 metadata dropout 与 corruption 使用独立随机流，支持作者在理想条件下讨论 unknown mixture posterior。

复用 helper 时，其 source arrays 必须来自本次 constructor 的 G，不能错误地回读原训练 source；`target_source=original_MP20_clean_native` 这类 helper provenance 文本也不能原样充当生成 G 的来源证据。这个接口改变不需要新权重，但要明确。

文中的理想 stationarity 证明成立，且作者已主动限制到精确 posterior/匹配 C；加上外层 rollback 后一般没有相同 Gibbs 结论。保持 fixed U 而逐 scalar 读取当前 prefix 是合法 posterior 因子分解，不能再次用“没有逐 scalar 重算几何 score”判 B 错。

### 7.2 实际 rollback 不等于 `trace.success=False`

[programmed_path_runtime.py](evidence_v3_r1_math/frozen/src/crystal_dlm/programmed_path_runtime.py.txt) 的 `_restore`（234–237 行）只恢复 old tokens 并追加 `op=rollback` event。full-cell 路径在 334/338 行调用它，却没有把全局 `trace.success` 设 False；后者主要表示 construction/最终起点可用性。因此内层失败回 U 后，全局 `success=True` 是可能的。

B 文档的实际核

\[
\widetilde K_B(G'|G)=\sum_U C_{mix}(U|G)
\{D_{ok}(G'|U)+r(U)\mathbf1[G'=G]\}
\]

本身正确。但如果实现只检查 `success`，就会把已经 inner rollback 的 U 接受下来，实现另一条核。

CPU 两状态例让 C 总是翻转 G、inner 总是 rollback U 且 global success=True。事件正确识别后，外层核是 identity；只看 success 的实现则是 flip。**两个核都归一化，但算法完全不同。** 这说明必须使用 full-cell transaction 内 `rollback` event 或显式 transaction outcome，不能用 output unchanged / 全局 success 代替。

最小执行定义应明确：外层保存 G；生成 U；以 `construct=False, cooperative=False, closure=False, full_cell_repair=True` 运行一次 inner；若该 full-cell transaction 出现 rollback，外层返回 G。原 runtime 的 cooperative/closure 默认是 True，若漏掉这些 flags，还会误加未登记阶段与 NFE。合法 accepted-unchanged 不算 failure；`U=G` 的 clean fallback 与真正 full-cell failure 也分别记账。

这些是可直接修订的 API/trace 合同，不需要重新理论化 rollback，也不意味着要在本轮运行新 GPU 作业。

### 7.3 Stationarity 仍不决定 SUN 或有限步修复收益

作者列出的完美经验 posterior 导致训练 support/novelty 为零，是对平稳性叙事有效的反例；但它也不意味着所有神经 denoiser 都只会记忆。construction 的 `12/44` 与 full-cell repair 的 `8/50` 已经构成非单调权衡，所以保留 G 是实质竞争者，B(H=1) 不能仅靠比 D(G) 好就宣称合理，还要考虑零额外修复的 G。

一次 C 本身不调用 DLM，故 B(H=1) 与现 V2 nominal `2d` NFE 的算式成立；完整 trace/wall-time 仍可能因 corruption retries、批处理和失败 early exit 不同。增加 H=2 是再加 d 次 8B forward，不是免费多一小步。原文没有直接授权 H=2，应该保持。

## 8. 密集监督、4–6 A800 与不同初始化的公平性

受审密度表 `0.75+0.1375(6+3N)` 正确；C 两个完整 G views/source/epoch 相较 V2 稀疏/dense 混合能明显增加回归 targets/forward。按已有 train histogram，两 epoch C 是 `108,544` forward examples、`4,045,428` 几何 scalar targets。V2 对应期望 categorical numeric labels 为 `637,654.35`。

这不能变成“六倍有效训练数据”或“同样卡时必然收敛”。coordinates 强相关，较低 σ 的 target 包含可能被编码精度淹没的噪声，较高 σ 的 ε target 大量接近 Gaussian-reference 恒等映射；优化中得到大量 labels，不等于得到同量的结构辨别信号。dense token control 可以以同样的 source/forward 分配取得更多监督，所以任何连续优越性主张都需要分离监督分配。

C 是两个纯 G views，H-P/H-C 反提案采用一个 T 加一个 G，其几何监督只有上述一半，但保留 token construction 的训练锚点；它们不是相同 loss 的两种 sampler。C 使用 fresh adaptation 还是 V2 warm-start 也影响累计训练预算。比较时至少列出 checkpoint 起点、累计 source-view 曝光、T/G 数量、active trainable inventory、实际 NFE 和主 endpoint，不能只写“2 epoch、4–6 A800”。

64 次共享 8B forward 的 NFE 在平均 N≈10.42 时略少于 V2 `2d≈74.54`，对 N=1 却多于三倍；score-only 若省 vocabulary projection 是潜在优势，但当前 wrapper 会计算 logits，未经实现/profile 不能预报实际节省。4/6 卡 global batch24 的 accumulation 配置是算术可行，不构成显存/吞吐或新连续任务可学性的证据。

## 9. 与 H-P/H-C 的取舍，以及不应添加的机制

这里只比较已写出的逻辑关系，不给本审稿人自有报告签发通过。

- **纯 prior mixed-space（H-P 类）**可以诚实保留 C/S/P 条件，但不保留旧程序的数值执行顺序。它说明没有必要为了“P 一定要执行”而将 γ clock 设为连续路线的前提。它自身的 finite ODE、端点、dense loss、V2 warm-start 和 mixed native CIF 都仍需别人审查。
- **token constructor 后连续输运（H-C 类）**保留真实 P construction 路径，但起点是 `p_construct q_τ`，通常不是 `p_data q_τ`。它不是用理论把 proposal distribution shift 消掉了；额外 d 次 forward 和 τ 选择也可能抵消低 NFE 的理由。应允许纯 prior comparator 否定其必要性。
- **当前 C**有完整 simple-prior 起点与 64 次全胞更新，比只把 local head transport 成 logits 更直接；其 γ clock 提供实际可执行操作，但该操作的终端误差和学习分配必须进入贡献解释。
- **离散 Q**在受审文档已正确定义为低维 factorized kernels，不需要高维巨表；其新 lattice codec、支持和第二次量化是真实成本。不能为了获得 ELBO 即默认它更好，也不能为了采用 continuous heads 而歪曲它的内存需求。

没有一条比较要求立即增加 multi-hop GNN、角度 teacher、force reward、joint SG/Wyckoff hard variables 或程序价值网络。若关键诉求只是正确的几何状态和密集监督，应先比较能实际承载该诉求的最小定义。若核心论文对象必须是 token-native DLM，则 mixed-space 路线确实改变了研究问题，应正面决定是否改写，而不是保留 tokenizer 来掩盖。

## 10. 作者应修订的最小清单与本轮边界

| 项目 | 处置性质 | 本轮要求 |
|---|---|---|
| C clock 被解释为更早恢复结构 | 结论收窄/方案选择前 | 改成目标边缘噪声差异；显式承认 terminal noise 与量化混杂；决定 clock 是否真有必要 |
| C 64 步 reverse | 算法近似证据 | 加 Gaussian oracle 方差反例，不能只报 prior mismatch；是否换 sampler 不由本轮自动决定 |
| C μ 的 x₀ 中间巨大值 | 可简单数值修订 | 用代数等价稳定 μ 形式；初始化参考和 loss 参数化另写清 |
| C final probability | 可简单数学修订 | 用 probability measure/pushforward，包含 singular terminal map 和失败 `⊥` |
| C live modules/full-float API/init | 实现前必需 | 冻结 output-only 参数；明确 numeric token ids、time/old接口、checkpoint来源与累计曝光 |
| A 的可识别性 | 数据事实待核 | 重复 exact-C 内 `a_δ` 联合变异和 frozen prior 覆盖；全局 MI 不足 |
| A prior/unknown与P干预 | 条件与支持说明 | 不把 empirical absence 当物理不可能；区分 S/P 同变与 off-support 敏感度 |
| B outer rollback | 可简单接口修订 | 用 full-cell rollback event；明确仅运行一次 inner repair 的 flags 和实际 G 来源 |

独立 [review_checks.py](evidence_v3_r1_math/review_checks.py) 与 [结果](evidence_v3_r1_math/review_checks.json) 完成了 clock/endpoint、Gaussian oracle reverse、μ 代数、conditional-identifiability、frozen prior support 和 rollback status 反例。它没有测量本项目连续 score、实际 prior chart、运行时间或 SUN。

**本轮建议保留 B 为最小零训练执行对照的科学对象，A 等待实际属性支持证据，C 保留为认真比较的 mixed-space 路线但先完成上述规格修订。没有选择 V3、批准新训练，或宣称数学自检预示双 SUN 门槛可达。**
