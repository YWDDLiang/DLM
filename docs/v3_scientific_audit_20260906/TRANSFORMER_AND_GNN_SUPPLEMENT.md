# Transformer、离散几何与 GNN 接入补充审计

2026-09-06，主审补充。状态：原论文与本项目代码的机制比较，**不是 V3 采用决定，也没有新模型实验**。以下论文分数均保留各自任务定义，不与本项目 SUN 横向排名。

## 1. 四个有用但不能照搬的对照

| 工作 | 本次核到的有效线索 | 最强边界或反证 |
|---|---|---|
| [ADiT v2](https://arxiv.org/html/2503.03965v2)，§2、Appendix C/D | autoencoder 将混合几何/物种转为 latent，DiT 在 latent 生成；作者给出 Equiformer-V2 与普通 Transformer 消融 | Table 5 中 MP20-only S 的 DFT SUN 6.5%，B 为 4.7%；大模型更高稳定率不等于更高 SUN。latent8 的等变 encoder 重建改善，生成有效率却弱于普通 Transformer；重建质量不能代替生成结果 |
| [CrystalDiT v1](https://arxiv.org/html/2508.16614v1)，Method、Tables 1–3、Appendix C | 共享 attention 处理晶格与原子；普通 Gaussian diffusion、元素的周期表表示；SUN 与多样性一起考察 | simple/complex 同时改变层数和信息组织，不是“越简单越好”的普遍定理。大量 checkpoint 生成与选择属于预算的一部分；其 9.62%/25.94% 不对应本项目 CHGNet 全请求协议 |
| [Crystalite v1](https://arxiv.org/html/2604.02270v1)，§3.3–3.4、§5、Appendix F | 连续完整 noisy geometry，periodic bias 与 EDM；GEM 更明显改善局部精度，其图示消融支持特定环境下的稳定性收益 | 同名 SUN 在不同评估管线下差异很大；其降低物种 loss 的策略不直接适用于本项目已固定元素计数的 DLM。GEM 成功不证明移植一个 bias 就足够 |
| [MaskGXT v1](https://arxiv.org/html/2606.22866v1)，§4–5、Appendix A/B | 离散坐标、SG/Wyckoff tokens、同站点特征聚合、ordinal smoothing 与 bounded sub-bin head；证明离散 masked 模型可以用于高质量 CSP | 论文自己说明 SG/几何一致性由训练鼓励而非硬保证；主指标是 MR/METRe，de novo 是未来方向。空间群分层多候选、argmax 与最多 3000 epoch 不能直接搬作本项目 one-request/one-trajectory SUN 的收益证据 |

ADiT v2 中关于“规模改善 SUN”的一处概述与其 Appendix C/Table 5 的实际数值方向不一致。本审计按明确表格与对应任务解释，不从摘要抽取单向成功叙事。晶体机制主报告中的其他 16 项研究见 [完整对照](CRYSTAL_DIFFUSION_MECHANISMS.md)。

## 2. 现有模型已经是什么样的 GNN

代码证据：`periodic_state_conditioning.py` 的 pair MLP 读取周期相对几何与物种，聚合邻居，再经 cell/site MLP 形成每站点与全局向量；`state_conditioned_model.py` 将这些向量注入 cell 与 atom/XYZ 隐藏状态。V2 另增加几何 attention bias。

因此“现有几何只改 attention、不进 hidden”是已撤回的错误描述。这个结构已经包含一跳周期消息传递；将它改名为 GNN 不能形成新贡献。

以一般记号，可写已有路径为

\[
m_i=\operatorname{Agg}_{j}\phi(e_i,e_j,\gamma(r_{ij}),P),\quad
g_L=\rho_L(\{m_i\},L),\quad
g_i=\rho_s(m_i,e_i,g_L),\quad
H_{I(i)}\leftarrow H_{I(i)}+Wg_i.
\]

这里的 `Agg`、作用域和 known-mask 由实际实现决定。全局 cell 向量还能将已知几何送至没有直接 attention 几何边的 active site。早期 construction 的完整几何尚不可见，是另一个问题，不能用模块参数名混淆。

## 3. GNN 的真实增量应在哪里验证

| 候选增量 | 要回答的具体问题 | 主要反例与成本 |
|---|---|---|
| 二至三跳周期消息传递 | 是否比现有一跳聚合更好表达配位环境与相互作用 | LM 自身已有多层全局交互；更深可能冗余、过平滑或增加拟合负担 |
| 角度 / 高阶环境特征 | 是否区分相近 pair-distance 分布下的不同局部几何 | 均匀更强的几何表达不保证 novel low-hull 结构；先验证实际数据存在该混淆 |
| old 与 visible-prefix 双通道 | 是否让有限模型更容易协调新 cell 与已写坐标 | 不增加条件信息；替换掉 old 可能丢失 noisy observation。训练与部署需同义 |
| 分层注入 hidden/value | 是否解决单次输入残差被深层 LM 淹没 | 不可由梯度总量推断，应检验真实 logits/表示响应；新增参数与训练量需受控 |
| 新连续几何头或 sub-bin head | 是否减少有实际影响的离散精度/联合更新误差 | 只改半格内精度不能解决错误晶格或盆地；要先测 codec 本身的可达误差 |
| 预训练 atomistic encoder | 是否提供少量数据难学到的几何先验 | 改变依赖、训练来源和原生推理口径；不能把 MLIP 预训练知识称为免费、无额外物理模型 |

[MACE 原论文](https://arxiv.org/abs/2206.07697) 支持高阶消息传递作为原子模型表达方式；[Equiformer-V2 原论文](https://arxiv.org/abs/2306.12059) 支持相应等变特征设计。两者的预测任务成功不直接证明接到 DLM 后会提高晶体 SUN。

## 4. 几何保证的接口条件

采用行向量 `R=FL`。同一个几何对象的更新与梯度必须区分：

\[
\delta F=\delta R\,L^{-1},\qquad s_F=s_R\,L^T.
\]

前者是位移/速度，后者是标量函数的 covector 梯度；不能用同一矩阵转换。GNN 给出 Cartesian 向量、DLM 却消费 fractional scalar 时，先定义输出究竟是哪一种量。

独立等变 GNN 加入非等变 LM，不会自动使整体模型等变。另一方面，六长度/角度已消去全局朝向，规范化表示也可减少冗余；缺少显式等变性并不是架构不可能成功的证明。应检查表示、噪声和目标需要哪种对称性，而不是最多地增加约束。

若 U 是固定 noisy observation，单次修订可合法分解为

\[
p(Y\mid U,c)=\prod_jp(Y_j\mid U,Y_{<j},c).
\]

`h(U,Y_<j)` 的 working-canvas 特征只是已知信息的重参数化。它可能帮助有限模型，但不是缺失的额外信息；它也不是连续 diffusion 外层时间更新的同义操作。

## 5. 不能为了数学名义重复一次失败

- Proper CE 与 ordinal smoothing 解决不同统计目标。移除 smoothing 可恢复声明类别下的精确监督含义，却不保证有限预算下的几何误差或 SUN 更好。反过来，MaskGXT 的 smoothing 对 CSP 有用，也不自动证明本项目该恢复同一权重。
- C→D 的 stationary 证明只说维持所指定分布。若它被解释为有限训练经验分布，完美重采样将落回训练支持，novelty 可为零。若解释为未知可泛化 population，则泛化能力与 SUN 仍需实验。
- raw SUN 仍经过共同物理评估；codec 精度、原生力误差、修订范围与最后 SUN 之间存在中间过程。任何一个代理改善都不能代替目标结果。
- V2 约 36.99M 新可训练适配参数附着在冻结预训练主干上，不能简单比成从零训练 8B，也不能据外部几千 epoch 直接判定它一定欠训练。

## 6. V3 采用前的科学问题

需要由完整审计确定最早有证据的缺口：条件语义、状态可见性、几何表示/量化、概率 kernel、有限拟合、周期图表达、物理后处理或统计口径。随后选可直接针对该缺口的小改动或架构；比较更小替代和最强反例，记录实际初始化与 1–2 epoch 可学性。

代码应把表示、corruption、conditioner/图模块、条件解码器与实验配置分开复用。保留核心数据身份、数值和评测分母检查即可，不将当前 job id、路径、epoch 或资源数量硬编码为通用模型语义。当前仍在审计阶段，没有因为新增 GNN 话题预先决定采用 GNN，也没有因论文名称提前确定 V3。
