# H-P33 第二轮独立数学与概率审稿

2026-09-06。对象是 [V3_H_P33_SPECIFICATION.md](V3_H_P33_SPECIFICATION.md)，最终所读 SHA256 为 `294bfc604a8ec7477e16dbea0124b66bbf3cad7c222b21d55e6863a9c4c3dd80`；[审阅快照](evidence_v3_r2_math/reviewed_spec.md)、[来源清单](evidence_v3_r2_math/source_manifest.json)已保存。独立比较确认，第3–7节数学与训练内容与 `6f572af` 相同；本轮确认主审随后修正了来源状态、开发种子、float exporter 和触发规则。

我不是 H 候选作者。本轮仅审当前 H-P33，不重新列 A/B/C 选择题；只新增本报告和独立 CPU 算例，没有修改其他报告、模型或数据，没有 SSH、GPU 或实际模型训练。

## 1. 裁决：可以进入有界实现，不等于可以跳过实现验收

**没有发现必须否决 H-P33 的概率定义或反向符号错误。允许按当前单一候选进入有界实现。** 全量 original-CIF 身份门已经由40009关闭；不能继续把它写成未决。33次映射、固定 float 主端点、同源 Q 次端点和失败质量也有清楚定义。

新增读出有效解决了第一轮局部点目标的主要残余模糊，但不等于精确恢复一般 clean 分布。独立 oracle 算例显示：相关 Gaussian 的两个特征方差在规定32区间后偏约 **+1.86% / −1.96%**；一个平滑的周期相对坐标分布，其一阶 Fourier 矩从 **0.4 变为0.382804**。这是明确的有限算法代价，不是“生成器不规范”，也不是实际 SUN 失败预测。

| 判断层 | 本轮状态 |
|---|---|
| joint 目标、chart、噪声单位、反向符号 | 通过定义审查 |
| original-CIF 全量来源身份 | **已有实际证据通过** |
| 32 Euler + 第33次读出 | 可实现的已登记近似；不是精确逆过程 |
| hidden-only adapter、低噪声可读性、真实 T/G 梯度累计、显存/吞吐 | 尚待规格第8节的实际验收；不能写成已通过 |
| SUN、稳定性、程序或预训练的净贡献 | 未验证，留给固定正式结果 |

下文给出两个必须写清的实现口径：每窗口的损失归一化，以及用**梯度增量**判断一个模式是否更新某参数。这些是可直接落实的说明，不是新增架构或额外科学臂。

## 2. Joint v/u 风险确实对应同一个联合几何场

固定实际条件 `c=(C,S,P)`，令 clean 联合分布为 `p0(z0,F0|c)`。虽然 forward 在给定 clean 结构后分解成 Gaussian lattice 和 wrapped-normal coordinates，训练器/网络必须条件于**完整** `Xt=(zt,Ft,c,t)`。两个输出的最优值是

\[
\bar v(X_t)=E[\alpha\xi_L-\sigma_Lz_0\mid X_t],\qquad
\bar u(X_t)=E[t\nabla_{F_t}\log q_t(F_t\mid F_0)\mid X_t].
\]

坐标噪声在给定 clean F0 后独立于晶格噪声，因此后一条件期望等于联合 noisy marginal 的 `t∇F log p_t(zt,Ft|c)`；它不是忽略 lattice 的单独坐标模型。每个 t/N 条件下，正的类别权重和已知缩放不改变该最优条件期望。log-uniform t 改变有限模型的任务分配，不会把上述对象改成另一种 score。

对 `α=cos(πt/2), σL=sin(πt/2)`，同噪声耦合的路径导数为

\[
\dot z_t=(\pi/2)(\alpha\xi_L-\sigma_L z_0).
\]

对 `Ft=wrap(F0+t ξF)`，条件路径速度为 `E[ξF|Xt]=−ubar(Xt)`。所以 `dz/dt=(π/2)vbar`、`dF/dt=−ubar` 与规格一致，向较小 t 更新时两个符号也正确。也可以由 probability-flow 方程推导；无需将它说成物理力或额外加入 Jacobian/能量损失。

T 的合法 token CE 与 G 的回归是共享参数的两个风险；**不推出** T 的 categorical 概率等于 G 经 Q 后的概率，更不是一个已推导的联合 likelihood。规格已经放弃该主张，应保持。T 初始化等价只承诺更新前；T/G 联合优化之后不存在原 T 行为必然不退化的定理。

## 3. Chart、端点先验与 wrapped 像窗可接受

`S=½log(LLᵀ/ℓ0²)`、五个 traceless 正交基和 `E6=I/√3`、VPA 的 `log N` 修正与 `chol_lower` 逆变换彼此一致。train-only `ddof=0` 和正 floor 明确了标准化测度；无需声称它是 intrinsic SPD Brownian motion。坐标 sigma=t 是 fractional 单位，不能解释成 V2 的 Å 单位噪声。

t=1 时 lattice 是精确 `N(0,I6)`，不依赖 clean 结构。坐标 uniform 与其真实终端边缘的 TV 上界约1.606×10⁻⁷（N≤20）。还有一个可以保留的正面保证：**同一个已固定的可测采样/失败映射不会放大这个初始 TV 差**。这不控制网络误差、Euler 误差或 SUN，但无需因 finite-shell 的分段性就否定先验近似的这个界。

独立检查 `n=-8…8` 与 `-32…32`：在 `|δ|≤1,t≤1`，最近保留像距离≤0.5，遗漏像距离至少8。由此可得相对遗漏质量上界 **2.871×10⁻¹⁴**、normalized u 的保守绝对误差界 **5.455×10⁻¹³**。选定测试点的 u 差最多约4.05×10⁻¹⁴，log-density 差约5.00×10⁻¹⁵。该像窗足够支撑所声明的 FP64 target 计算；不需要为此再扩大方法。

normalizer 数值及 floor 命中项仍要作为实现产物保存。小 sigma 的 z/F 必须在解码、FP32特征和BF16 hidden链路上可区分；**原CIF身份通过不自动证明低噪声输入信息仍可读**。规格已有一次具体敏感度验收，按它完成即可，不先添加另一套编码器。

## 4. 第33次读出：成立的等式与不能成立的推论

由旋转式插值的代数恒等式

\[
z_0=\alpha z_t-\sigma_Lv^*
\]

可得在 oracle 输出下 `z_out=E[z0|Xt]`。因此它是正确的条件均值表达，不含原 C 的小 `sqrt(alpha_bar)` 除法问题；也不是从 clean posterior 随机抽样。

对 coordinate，写 `δ+n=Ft−F0+n=t ξF`。则

\[
F_t+t\bar u(X_t)=E[F_0-n\mid X_t].
\]

最终再 wrap，得到以当前观测选择 lift 后的均值。它不是无条件的欧氏 `E[F0|Xt]`，也不是全局 circular mean 或 torus posterior sample。规格的“local-lift wrapped Tweedie readout”描述可接受。

**局部点目标的修复成立。** 对 `f0=.0049, ε=.002`，独立CPU在±6ε邻域使用正确像和 target，33rd读出最大周期误差约1.11×10⁻¹⁶，测试网格的 Q-bin 变化数为0。第一轮只输出pε时靠近0.005边界的主要残余模糊已被移除。这是 FP64 oracle 算例，不是实际BF16网络精度；远离唯一局部像的情形不在这个点目标结论内。

**一般数据分布仍改变。** 若 `z0~N(0,I)`，oracle v=0，ODE可以完全保持unit Gaussian，但读出使 variance成为

\[
\alpha(\epsilon)^2=0.9999901304.
\]

约1×10⁻⁵的缩小很小，却足以说明“读出等于精确逆过程”不成立。一般欧氏条件均值满足 `Cov(E[z0|Xt])=Cov(z0)−E Cov(z0|Xt)`；不能把 posterior uncertainty 当作已经随机重建。

**碰撞没有被数学排除。** 取两个已标号、同物种的source modes `(.2,.3)` 和 `(.3,.2)`，cell轴长6 Å，各自间距0.6 Å。在对称 noisy observation `(.25,.25)` 上，两mode后验各半，oracle readout产生`(.25,.25)`，间距0。该观察在ε=.002下的每mode log-density约 **−614.41**，在理想低噪声分布中极其罕见；本文不把它当实际碰撞率。它只确认规格必须保留 mean-readout 风险和失败分母，不能升级为普遍合法性保证。

## 5. 32区间的误差不会被末端读出全部清除

独立 [NumPy 算例](evidence_v3_r2_math/h_p33_math_checks.py)／[结果](evidence_v3_r2_math/h_p33_math_checks.json)直接使用规格的 ε=.002、32个平方时间区间和第33次读出，没有拟合神经网络。

对 Gaussian chart 的一个特征方差λ，oracle v 系数为

\[
v(z_t,t)=\frac{\alpha\sigma_L(1-\lambda)}{\alpha^2\lambda+\sigma_L^2}z_t.
\]

选择二维 covariance `[[1,.8],[.8,1]]`，其特征值为0.2和1.8，**每个原通道的方差仍为1**，符合逐通道标准化。其余四维可取独立unit Gaussian。

| clean 特征方差 | exact flow到ε再读出 | H-P33输出方差 | H-P33相对误差 |
|---:|---:|---:|---:|
| 0.2 | 0.19999013 | **0.20371985** | **+1.86%** |
| 1.0 | 0.99999013 | **0.99999013** | −0.000987% |
| 1.8 | 1.79999013 | **1.76465432** | **−1.96%** |

这来自当前Euler离散化及readout，不能把第一轮针对另一个候选 `beta_tilde` 的约15%方差反例直接贴给 H。

再取平滑的相对坐标联合密度

\[
p_0(f_1,f_2)=1+0.8\cos[2\pi(f_2-f_1)].
\]

两个坐标都以std=t独立扩散，所以相对坐标的forward方差为2t²，Fourier amplitude为 `0.8 exp(−4π²t²)`，oracle joint场可解析写出。固定H-P33得到：

| 一阶相对cos矩 | 数值 |
|---|---:|
| clean目标 | 0.40000000 |
| exact pε，再做33rd读出 | 0.40006316 |
| 32 Euler后的pε候选 | 0.38267706 |
| **完整H-P33** | **0.38280431** |

最终矩相对偏约 **−4.30%**，主要来自有限积分而非readout；32,768/65,536点确定性积分的该结果变化小于1e−10。这里初始uniform误差远小于此差异，field本身平滑，不是坏score或NaN造成的反例。

这些例子量化的是表示分布误差，没有能量函数或晶体SUN。**它们限制精确性宣称，不阻止采用一个固定的33-NFE候选做实证。** 没有必要现在依据toy追加Heun、更多区间或新臂；先执行已登记对象，失败或实测不足再按明确证据处置。

## 6. T/G 累计可实现：需要清楚的窗口数学

设world=W、累计K、每microbatch一条样本。每个窗口应实现

\[
\widehat L_{window}=\frac1{WK}\sum_{k=1}^K\sum_{r=1}^W w_{rk}\ell_{m_k}(x_{rk}).
\]

真实view权重1、padding权重0；满窗口中T/G各半，相当于 **½(mean T + mean G)**。这符合通常的view平均解释；若意图是两种均值直接相加，梯度会差2倍，需在manifest明确，而不能仍把LR/配方称相同。

可按“12个source×两个view”组成共同global窗口。6卡用1×4累计、4卡用1×6累计，两者各含12 T+12 G，每epoch都为2262次更新；每模式27,144个padded位置，对27,136个真实source，即 **各8个零权重padding**。末窗口各模式只有4个真实样本，共8个真实view，不是24个真实view。保留原V2的epoch级padded/real校正，或明确记录另一归一化；不要按各rank局部非padding数量随意缩放。

每个micro同步不会把既有梯度重复放大：此前累积梯度已在各rank相同，再次平均保持该部分，只加本micro的新平均。每个loss应除以K；`zero_grad`和`clip_norm`在整个窗口边界，而不是每切换一次T/G便调用。反例梯度`(2,−1)`：先加再clip得到1，逐micro各自clip再相加得到0，已改变更新。

**mode验收应看新增梯度。** T backward之后，T-only参数已有`.grad`；随后的G backward不应该把它删除。在独立21参数toy里，T-head梯度范数在T后和G后均为0.080626，但G增加的梯度为0。这是正确累计，不能用“G后`.grad`非None”误判G梯度泄漏。可分别zero后测各模式，或比较前后梯度增量。

[单进程4/6布局算例](evidence_v3_r2_math/tg_window_algebra_check.json)与一次global-window梯度的最大差为2.78×10⁻¹⁷。尝试真实两进程CPU Gloo时，本机Torch2.8在初始化阶段报`unsupported gloo device`，尚未到toy forward；[记录](evidence_v3_r2_math/tg_ddp_cpu_attempt.json)明确不是DDP通过，也不是H失败。实际Torch2.4/CUDA、4/6卡与LLaDA checkpoint路径仍由规格第8节验收。

[PyTorch 2.4官方文档](https://docs.pytorch.org/docs/2.4/generated/torch.nn.parallel.DistributedDataParallel.html)区分了checkpoint的reentrant行为：`use_reentrant=False`支持通常DDP用法，reentrant与unused参数组合有明确限制。hidden adapter应核实并保留实际非reentrant路径，不在改写forward时落回默认；不使用`static_graph=True`推定交替T/G图相同。现有core证据显示了checkpoint调用位置，不能仅据此声称已核实具体运行flag。

## 7. 来源与评测协议：本轮已关闭的事项

[40009完整身份审计](evidence/CONTINUOUS_SOURCE_IDENTITY_40009.json)已核实train **27,136**、val **9,047** 全部一一对应指定CSV的`cif`列；完整量化answer唯一匹配，无解析/编码错误、重复Q歧义、未匹配或丢行，精确species-stable permutation后Q全部对齐。**原连续来源门已关闭，不再要求退回decoded标签或追加一次同样的身份审计。** 这项证据不包含能量证书、normalizer、模型梯度或SUN，三者仍要分开。

主审已把以下具体修订写入最终受审版，应按新文本实现：

- 开发比较用实际base seed **20260905**及原`path_seed`键、batch cap4/layout2；refiner默认`20260905+sample_idx`。不把独立cohort的20260906替换进来。[原始种子证据](evidence/V2_ACTUAL_EVALUATION_SEEDS.json)保留大于2⁵³的整数原文，不经JavaScript数值重序列化。
- primary float输出明确走continuous CIF／native_structure和显式structure源；不让旧exporter无条件parse body而悄悄量化，也不套0.01-bin duplicate guard。label/evaluate的structure优先路径可以复用；真实输入到N/U、共同R和refiner的结构hash须证明precision没有换层。
- 正式触发直接按同一端counts≥26/128。旧评估的`strictly_exceeds_10_and_50`是严格大于50%，Meta=128时为false；新launch不复用这个布尔，也不用改其原有含义。

这些是实际修复，不是H数学定义的否证。float主端点、同样256的Q raw次端点已在结果前确定；不能只报更好的一端。N/U与R的代码版本相同有助于比较，但warm-start、额外两epoch、连续监督、loss与输出精度都变了，首个结果只支持完整系统的增减，不识别单因素贡献。

## 8. 有界实现的最低交付与最终意见

当前不需要再改概率对象。实现仍需完成已有第8节的具体工作：

1. 从已通过的完整来源产物计算并保存normalizer；G输入不走旧codec，连续状态保留独立于BF16 hidden的精度。
2. hidden-only adapter复用同一LoRA/block/attention-bias/padding/checkpoint/ln_f；T step0输出等价，G不detach，G不虚算旧词表分支。
3. 按上面的窗口公式验证T/G和零权重padding，head打开后检验共享参数的有限梯度增量；每个optimizer窗口只clip/step一次。
4. 按已登记小sigma检验输入可读性和实际显存/吞吐；生成失败按请求进入⊥，不能因同batch一条失败删除其他请求或重抽。
5. 证明float原结构真正进入既定评估，Q另列；完成固定两个新增epoch及唯一final后再判定SUN。

**最终意见：通过当前数学/协议定义审查，支持进入上述有界实现与真实验收。** 本轮不签署“实现已通过”“33步足够精确”或“必有SUN提升”。只有验收中出现具体来源、计算图、数值或接口失败才需要修复；“还没有新SUN”本身不是继续阻断一个已定义清楚模型的理由。
