# 晶体 diffusion / flow 的机制与 DLM V3 迁移边界审计

审计日期：2026-09-06。作者：独立晶体机制审计 lane。任务是研究、否证与提出可检验候选；没有 SSH、GPU 调度、依赖安装或模型/训练文件修改。

本地执行基线固定为 `2e904c260bafb6750c9a5dbb0d920c4ff8c3a868`。已读 V2 execution package、loss/geometry proposal、implementation review，并对 corruption 直接读取冻结 Git 对象；核对 `periodic_v2_{model,corruption,training_data,objective,plan_data}.py` 在该 SHA 与本次读取 HEAD 之间没有差异。运行中 V2 的进度、Planner 实测统计由主审提供，本文没有自行访问远端或独立重算那些训练统计。

## 1. 当前可成立的结论，以及不能提前下的结论

1. **V2 确实引入了有科学意义的周期几何信息和修复监督，但它没有变成 DiffCSP++。** 其 log-SPD 用于合成扰动和特征；监督仍是离散条件 CE，解码仍是程序驱动的 scalar 构造/修复。不存在由该扰动推导出的 continuous score 反向积分器，也没有 SG/Wyckoff 的硬生成变量。
2. **DiffCSP++ 最值得借鉴的是“条件、自由变量、噪声、预测及更新共同满足一个约束对象”，不是单独的 log-SPD。** 原论文自己的无 SG `DiffCSP-k` 消融没有提高 MP20 匹配率。CSPML 模板已经贡献了大部分匹配率；不能把模板检索收益归因给 diffusion。
3. **连续模型提供了不同的训练对象和迭代状态，而非“连续”二字自动带来优势。** 跨噪声时刻的完整几何更新、联合预测与噪声状态覆盖值得研究；但它的外层t时钟不能直接类比DLM单个posterior内部的scalar顺序。固定old完全可以是正确的条件去噪。SGEquiDiff也反对“所有变量必须同时连续更新”的普遍命题。
4. **对称性不是 SUN 保证。** 合法 SG/Wyckoff 可以约束在错误高对称鞍点；精修可以增加稳定性却减少新颖性；数据分布的 score 也不一般等于量子能量的负梯度。文献已有支持这些边界的消融和失败实例。
5. **当前软计划是提示/程序变量，不能直接升级为 DiffCSP++ 的真实空间群条件。** 需要研究软条件与目标几何的统计耦合；低 annotation agreement 只是诊断线索。若新条件只从组成独立采样、目标几何不变，population 最优模型可能合理地忽略其几何含义。
6. **GNN 可以接入 DLM，而且当前 state conditioner 已经实现一跳周期消息汇总并注入 hidden。** 新模块的真实边际应是多跳/方向/逐层交互或 working-state 更新，不能把已有功能改名为新 GNN。其自身的等变性也不会自动传给整个 LM，新权重不保证在一至两 epoch 内学成。
7. 本文没有依据断言 V2 成功或失败；没有用任何论文分数预测本项目能否同时达到同端 Strict SUN ≥10%、Meta SUN ≥50%。以下候选均是待隔离验证的研究方向。

## 2. 证据范围与任务对齐

证据等级：**A**＝原论文相关方法/实验正文及作者关键实现相互核对；**B**＝原论文方法/实验已读，只有部分实现或尚无可核代码；**C**＝仅发现线索，不用于结论。这里“已读”指列出的实质章节，不把全文下载等同于逐条复现，也不把 README 当成训练/评测复现。

### 2.1 核心工作各自在解决什么问题

| 工作 | 生成条件与变量 | 主要已核评测 | 与本项目 SUN 的边界 |
|---|---|---|---|
| [CDVAE, arXiv 2110.06197v3](https://arxiv.org/html/2110.06197v3) | VAE 潜变量先预测组成比例、N、晶格；条件 decoder 生成物种与坐标 | 重建、validity/COV/属性分布、预测器指导的属性优化 | 原文主要是代理指标；100% 距离有效不等于热力学稳定 |
| [DiffCSP, 2309.04475v2](https://arxiv.org/html/2309.04475v2) | 主要是给定每原子元素与 N 的 CSP；联合晶格/坐标；另有生成扩展 | 单/多候选结构匹配、匹配样本 RMSE；生成代理指标 | 多候选中对 GT 匹配的 oracle 与独立新组成 SUN 不是同一任务 |
| [DiffCSP++, 2402.03992v2](https://arxiv.org/html/2402.03992v2) | 必须有 SG 和 Wyckoff 分配；CSP 用 GT 或 CSPML 模板；生成也抽训练模板 | CSP MR/RMSE；生成 validity/COV/Wasserstein | 不报告本项目意义的完整 SUN；模板与额外条件改变问题难度 |
| [MatterGen, Nature 2025](https://www.nature.com/articles/s41586-025-08628-5) | 物种、分数坐标、晶格联合扩散；属性 adapter | DFT 后 ≤0.1 eV/atom、SUN、DFT 前后 RMSD；有实验案例 | “stable”主要对应 0.1 门槛；Alex-MP-ICSD hull/新颖性库、ordered-disordered matcher 与我方不同 |
| [FlowMM, 2406.04713v1](https://arxiv.org/html/2406.04713v1) | CSP 固定 A；DNG 还生成 analog-bit 物种 | CSP 匹配及 CHGNet→DFT 的 DNG 稳定/SUN | MP-2023 hull、负 hull 与 arity 条件；不能直接比较 CHGNet 共同 R 的双门槛 |
| [FlowLLM, 2410.23405v1](https://arxiv.org/html/2410.23405v1) | LLM 给完整结构 base；专门训练 RFM 输运 LLM 几何→数据几何，固定物种 | CHGNet→DFT 稳定/SUN；raw→CHGNet 的几何/能差另评 | 预拒绝无效 LLM 输出；主表 SUN 与几何接近度分开；不是现成通用 refiner 无条件接入 |
| [CrysLLMGen, 2510.23040v1](https://arxiv.org/html/2510.23040v1) | 独立训练 LLM 与 CSP diffusion；把 LLM 几何直接注入中间 τ | CHGNet hull、strict/meta 与 SUN；CSP/代理指标 | 原文主文 10,000 与附录 1,000 分母表述不一致；发布实现及我方 model494 谱系须分开 |

本项目 native SUN 本身仍包含共同物理评估中的 CHGNet 松弛；它的“native”是没有额外 model494，而不是未经任何评估松弛的晶体稳定性。旧文档的“完整 verified 交集”、全请求分母和历史未解决 hull 不能混在一张表中。主审提供的已纠正独立旧 K8：native 39/244、tau800 65/484（1000 请求）；raw-v1 开发 native 3/45、refined 16/113；参考开发 6/55、19/121（256 请求）。这些是背景约束，不与以上论文数字作排名。旧 `0/9`、`1/17` 不作完整结果。

### 2.2 原始来源与代码冻结表

全部访问日期为 2026-09-06。下表的 Git SHA 是本次公开仓库 HEAD 的冻结读数，**不自动等于论文正式实验 checkpoint 的源码版本**。下载内容的 URL、时间和 SHA256 记录在 `evidence_crystal_diffusion/source_receipts*.json`；新仓库 HEAD 元数据在 `extended_repo_receipts.json`。关键代码均只作静态阅读，没有运行。

| 原始工作/所读版本 | 所读范围 | 作者仓库 SHA | 强度 |
|---|---|---|---|
| DiffCSP++ v2 | §4–5、App A/B；corruption、sampler、crystal-family、score helper、config | [jiaor17/DiffCSP-PP](https://github.com/jiaor17/DiffCSP-PP/tree/e82e7ffa7cb2a383bde69b067a343ca137d73e47) `e82e7ffa7cb2a383bde69b067a343ca137d73e47` | A |
| DiffCSP v2 | §4–5、App B/C/D；joint denoiser/sampler/config | [jiaor17/DiffCSP](https://github.com/jiaor17/DiffCSP/tree/7121d159826efa2ba9500bf299250d96da37f146) `7121d159826efa2ba9500bf299250d96da37f146` | A |
| CDVAE v3 | §3–5、App A/D；model/noising/decoder/config | [txie-93/cdvae](https://github.com/txie-93/cdvae/tree/f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea) `f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea` | A |
| MatterGen, Nature 2025 及最终补充 | 主文；SI A.4–A.11、C、D.1/D.3；SDE、lattice-score、D3PM/config | [microsoft/mattergen](https://github.com/microsoft/mattergen/tree/92423660a8bd70e83679086e88f88596d484dc16) `92423660a8bd70e83679086e88f88596d484dc16` | A |
| FlowMM v1；FlowLLM v1/NeurIPS 2024 | 几何/flow、base、实验、训练预算；`model_pl`, manifolds, config | [facebookresearch/flowmm](https://github.com/facebookresearch/flowmm/tree/6a96aec3b6eba89f6fa07436f0c8837979abb285) `6a96aec3b6eba89f6fa07436f0c8837979abb285` | A；具体 checkpoint 预设待核 |
| CrysLLMGen v1 | §4–5、App C；直接 τ 初始化代码 | [kdmsit/crysllmgen](https://github.com/kdmsit/crysllmgen/tree/94bb287751cd20a882c7c1df7ca736633d78e5e1) `94bb287751cd20a882c7c1df7ca736633d78e5e1` | A；论文评测分母未决 |
| [SymmCD v1](https://arxiv.org/html/2502.03638v1) | §4–5、App D/E；discrete site-symmetry diffusion | [sibasmarak/SymmCD](https://github.com/sibasmarak/SymmCD/tree/a50bc003e8c5fa2348336dc77477d2439f165fd4) `a50bc003e8c5fa2348336dc77477d2439f165fd4` | A/B；未复核最终会议版全部变化 |
| [WyckoffDiff v4](https://arxiv.org/html/2502.06485v4) | §3–5、App A/F/G；GNN/D3PM/config | [httk/wyckoffdiff](https://github.com/httk/wyckoffdiff/tree/9c4694291be1c20e4f06cc33f02baed7daaf0b7a) `9c4694291be1c20e4f06cc33f02baed7daaf0b7a` | A |
| [WyckoffDiff-Adaptor v1](https://arxiv.org/html/2601.08115v1) | 全部方法/结果；官方配置/工作流 | [pfnet-research/wyckoffdiff_adapter](https://github.com/pfnet-research/wyckoffdiff_adapter/tree/9cb01bbc0fcf17e8d16968cf2d76cda9761ca070) `9cb01bbc0fcf17e8d16968cf2d76cda9761ca070` | B；正文 adaptor 链接拼写不同 |
| [SGEquiDiff v3](https://arxiv.org/html/2505.10994v3) | §3–5、App A.1/A.2/A.7；lattice/Transformer/score/sampler | [rees-c/sgequidiff](https://github.com/rees-c/sgequidiff/tree/2231031fdabbf6b372031f3f233f8638922fa91e) `2231031fdabbf6b372031f3f233f8638922fa91e` | A |
| [CrystalFlow, Nature 2025](https://www.nature.com/articles/s41467-025-64364-4) | 模型、CSP/DNG、formation-energy 条件、Methods；flow/config | [ixsluo/CrystalFlow](https://github.com/ixsluo/CrystalFlow/tree/9c25ff0245d787efd87c9d1d797a8968443608cc) `9c25ff0245d787efd87c9d1d797a8968443608cc` | A/B |
| [SymmBFN, npj 2026](https://www.nature.com/articles/s41524-026-02140-8) | 方法/评测/消融；BFN DNG/config | [aimat-lab/symmbfn](https://github.com/aimat-lab/symmbfn/tree/697e99ae910a1336aa51e2be838d7c6e7f3fdc11) `697e99ae910a1336aa51e2be838d7c6e7f3fdc11` | B；补充协议未逐项复现 |
| [DiffCrysGen, npj 2026](https://www.nature.com/articles/s41524-026-02147-1) | 生成/比较、IRCR/EDM/训练/数据；UNet/EDM loss | [SouravMal/DiffCrysGen](https://github.com/SouravMal/DiffCrysGen/tree/88562cb8ec89c67272b191b362b340104fbd7f56) `88562cb8ec89c67272b191b362b340104fbd7f56` | A/B；磁性案例不作为通用 SUN 证据 |
| [ChargeDIFF, Nature 2026](https://www.nature.com/articles/s41467-026-73985-2) | A/X/L/C、VQ-VAE、消融、数据/Methods；model/network/config | [parkjunkil/ChargeDIFF](https://github.com/parkjunkil/ChargeDIFF/tree/32acf56b88883108621eb044451b9220df17b609) `32acf56b88883108621eb044451b9220df17b609` | B；未逐条复核全部评测脚本 |
| [DynaCrys v1, 2026-08-07](https://arxiv.org/html/2608.07401v1) | §4–5 与 symbolic legality/geometry 附录 | 正文及 arXiv 页面未找到作者代码入口；不声称不存在代码 | B；非常新的预印本 |

CrystalDiT、ADiT、MaskGXT、Crystalite 由其他 lane 深审，本报告不重复做其实现裁决。它们不是本报告“只发现却冒充全文已核”的附加条目。

## 3. 晶体几何究竟在哪个空间上学习

### 3.1 统一记号与五种容易混淆的对称性

采用本项目行向量约定：`L∈R^{3×3}` 的行是晶格基，`F∈[0,1)^{N×3}` 是分数坐标，Cartesian `R=FL`。完整物理对象是

\[
\mathcal C(A,F,L)=\{(A_i,(F_i+n)L):i=1,\ldots,N,\ n\in\mathbb Z^3\}.
\]

| 操作 | 表示变化 | 现有 DLM 能声称什么 |
|---|---|---|
| 全局空间旋转 | `L→LQ, R→RQ, F` 不变 | 六长度/角度已经消掉晶胞朝向。缺少 Cartesian 等变输出头不等于完全没有旋转处理 |
| 坐标周期 alias | `F_i→F_i+n_i` | `mod 1`、000/100 canonical alias 与 Fourier 支持周期表达；不等于整体平移不变 |
| 整体原点平移 | `F→wrap(F+t)` | 相对边特征不变，但绝对数值 token/位置/程序路径可变；没有整模型分布不变的证明 |
| 站点置换 | `(A,F)→(PA,PF)` | 物理对象不变；native slots、species program、LM 位置需要一致变换。部分图分支等变不推出整 LM 等变 |
| 等价整数基 | `L→UL, F→FU^{-1}`, `U∈GL(3,Z)` | log-Gram 不是此操作的不变量；有限 image shell、codec 和 canonicalization 都须检查 |
| 内部空间群 | 某些 `(R_g,t_g)` 把整晶体映回自身 | 是样本本身的结构约束，需要元素、轨道、晶格同时相容；它不是以上任意一项的别名 |

模型可以在明确的规范代表上建模，而不必在冗余表示空间严格不变；但要区分“规范代表的分布”和“对全部等价表示的概率密度”。Niggli/标准胞、原点选择在退化边界还可能不唯一或不连续，不能把预处理口号当成全空间唯一性定理。

### 3.2 联合晶格/坐标的物理耦合

\[
dR=dF\,L+F\,dL.
\]

固定 F 改 L 会仿射移动所有原子。对相对分数位移 `u=F_i−F_j+n`，

\[
r=uL,\qquad d^2=u(LL^\top)u^\top=uGu^\top,
\quad d(d^2)=2(uG)\,du^\top+u(dG)u^\top.
\]

因此，最合适的坐标动作依赖晶格，最合适的晶格动作依赖所有站点环境。把两者各列一项 loss 并不表示模型忽略耦合：DiffCSP/MatterGen 的前向噪声可以分解，但每个反向预测头都读完整 `(A,F_t,L_t,t)`，通过共同网络学习联合依赖。

也不能反向推论“分阶段必然失败”：任何联合分布均可分解为 `p(L,F|A)=p(L|A)p(F|L,A)`。无限容量、正确条件和精确采样时，先晶格后坐标可以精确采样联合分布。需要实测的是有限数据/模型的误差传播与修复能力，不能把架构顺序当成不可能性证明。

## 4. DiffCSP++：值得转移的完整机制

### 4.1 log-polar 表示，及它不提供的保证

列向量论文写 `L=Q exp(S)`；本项目行向量相应写

\[
G=LL^\top,\quad S=\tfrac12\log(G/\ell_0^2),\quad
v=\operatorname{tr}S=\log(V/\ell_0^3),\quad S_{dev}=S-\tfrac v3I.
\]

给任意有限对称 S，`G=ℓ₀² exp(2S)` 数学上正定。这个事实保证非奇异 metric，不保证合理密度、键长、能量、SG、化学稳定性或生成分布正确。`log(UL Lᵀ Uᵀ)` 也一般不等于对原 `log(G)` 的不变表示。

必须区分论文记号和作者代码：论文展示的六个基未单位化，体积基为I，因此该记号下`v=3k₆`，等方k-noise不是Frobenius等方。但本审计所引固定作者代码`get_basis():33–35`随后除以每个基的Frobenius norm，实际采用六个正交单位基；其体积基为`I/√3`，故实际代码`v=√3 k₆`。此时五个shape基与V2正交无迹基仅顺序不同，不能再声称代码的shape noise非等方。V2仍独立指定log-volume与shape的σ；该差别不等同于论文未归一基的差别。

论文/作者实现用空间群映射到晶族，对k作affine约束`P(k)=Mk+b`。例如cubic只留体积自由度；hexagonal不是把某些系数都置零，论文对应系数为`−log(3)/4`，**代码因基归一化改为`−√2 log(3)/4`**。状态的affine投影与噪声/向量的linear投影应区分，但不能只看到噪声也调用P就判bug：作者forward末端和reverse末端再次调用P，且`Mb=0,M²=M`，于是`P(aP(x)+cP(ε))=aMx+cMε+b`，自由坐标过程并未被固定offset污染；loss中同样的固定offset也会抵消。[原论文 §4.1](https://arxiv.org/html/2402.03992v2#S4.SS1)、[作者 CrystalFamily](https://github.com/jiaor17/DiffCSP-PP/blob/e82e7ffa7cb2a383bde69b067a343ca137d73e47/diffcsp/pl_modules/lattice/crystal_family.py)。

### 4.2 轨道自由变量、相关噪声和更新必须一起约束

用 `u_s∈R^{d_s}` 表示第 s 个 Wyckoff 轨道的自由参数，可以写

\[
F_{si}=\operatorname{wrap}(A_{si}u_s+b_{si}),\qquad
A_{si}\in\mathbb R^{3\times d_s},\quad d_s\in\{0,1,2,3\}.
\]

一个轨道的所有 atoms 同物种，且 `Σ_s m_s=N`。扰动应由一个共享轨道噪声生成，而不是每个等价原子独立加噪声：

\[
u_{s,t}=u_{s,0}+\sigma_t\xi_s,\qquad
\Delta F_{si}=A_{si}\sigma_t\xi_s.
\]

单独对 F 做逐原子 Gaussian 再宣称 SG preserved 不成立。对应反向网络的 atom-level 预测还要 pull back 到自由坐标，在同一轨道汇总，再 push forward 并按操作重新展开。作者代码确实读取 `ops_inv`、`anchor_index`，对 orbit 平均后重建各原子；每次 predictor/corrector 后都会展开完整轨道，而不只是训练加一个 SG embedding。[关键 sampler](https://github.com/jiaor17/DiffCSP-PP/blob/e82e7ffa7cb2a383bde69b067a343ca137d73e47/diffcsp/pl_modules/diffusion.py#L226)。

这也说明：复制某一个空间群 embedding、加 log-SPD corruption、事后把角度取整，均不是相同机制。完整约束保证的是结构对给定群操作不变；偶然更高对称、轨道碰撞、数值容差及 conventional/primitive 转换仍需考虑，不能保证检测出的**最大**空间群恰好等于输入 G。

### 4.3 周期 score 与符号约定

对单位 torus 上某一坐标，真实 wrapped-normal score 为

\[
q_\sigma(f|f_0)\propto\sum_{n\in\mathbb Z^3}
e^{-\|f-f_0+n\|^2/(2\sigma^2)},\quad
\nabla_f\log q_\sigma=-\frac1{\sigma^2}\sum_n w_n(f-f_0+n).
\]

最近像是 small-noise、单个 image 主导时的近似，不是全部 σ 的公式。反向坐标更新由这种向量目标训练，并配合 wrap、predictor/corrector、噪声 schedule 与 normalization。作者 `d_log_p_wrapped_normal` 名字有误导性：实现返回的是上式负号相反的 denoising direction，sampler 使用减法；不能根据函数名抄正负号。[helper](https://github.com/jiaor17/DiffCSP-PP/blob/e82e7ffa7cb2a383bde69b067a343ca137d73e47/diffcsp/pl_modules/diff_utils.py#L38)。

独立数学例子：`f−f₀=.5` 时，左右 image 对称抵消，真实 score 为 0；最近像任取一边在 `σ=.5` 时得到 ±2。`σ=.5, δ=.49` 时，完整近似和给出 `−0.00576`，最近像给 `−1.96`。这不是微小权重差，而是目标方向/尺度的结构差异。

### 4.4 损失的一个有用而有限的借鉴

若同轨道 pull-back 后的预测为 `v_i`、目标 u，令 `v̄=m^{-1}Σ_iv_i`，则

\[
\frac1m\sum_i\|v_i-u\|^2
=\|\bar v-u\|^2+\frac1m\sum_i\|v_i-\bar v\|^2.
\]

post-average loss 比只监督轨道均值多了预测一致性的方差惩罚。可迁移的是“等价站点不能靠互相抵消预测误差过关”；若没有真实 orbit 标签，不能在任意同元素站点之间施加此约束。同元素不等于相同晶体环境。

### 4.5 直接否证，不替论文放大结论

- `DiffCSP-k` 无 SG 消融：MP20 MR `51.49→50.76`，RMSE `.0631→.0608`。原作者把它解释为可用表示，而非证明 log-k 本身提高匹配率。
- CSPML 模板 MR 已为 `70.51%`；DiffCSP++ 精修后 `70.58%`，RMSE `.0338→.0272`。因此把 `51.49→70.58` 整体算作 diffusion dynamics 增益是不成立的归因。GT SG/Wyckoff 的 `80.27%` 是另一种信息条件。
- 生成实验从训练集有放回抽 10,000 模板；指标是有效性、覆盖与属性分布，不是 SUN。

这些数字均来自[原论文 Table 4、Table 5、§5.3](https://arxiv.org/html/2402.03992v2)，只在论文自身条件下解释。它们既不否定约束精修，也不支持我方采用表面相似部件就会达到物理目标。

## 5. 其他核心模型分别补上了什么

### 5.1 CDVAE：近数据去噪和低能结构有联系，但不是量子力场

其 decoder 在固定预测晶格上做周期图去噪。物种噪声由 clean one-hot 与预测组成比例混合后取 categorical sample；坐标目标是噪声结构到 clean 的最小周期位移，使用等变 GemNet decoder，配合多个噪声尺度的 Langevin sampling。[作者 model](https://github.com/txie-93/cdvae/blob/f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea/cdvae/pl_modules/model.py)。

原文的 harmonic-force 解释有明确前提：小噪声、训练误差近零、目标是同一个局部平衡结构。若 `E(R)≈E(R*)+½δRᵀHδR`，真实力是 `−HδR`，而 isotropic denoising restoring field 是 `−cδR`。二者只有特殊 Hessian/预条件关系才相同。多个多晶型的混合 posterior、软模、错误晶格及较大噪声都会破坏“就是物理力”的说法。[原文 App A](https://arxiv.org/html/2110.06197v3#A1)。

对我方最直接的教训不是把 force 数值塞进 token，而是保持同源噪声状态到 clean 的可靠几何监督，并检查部署坏晶体是否还处于这种局部邻域。

### 5.2 DiffCSP：factorized corruption 并不意味着独立晶格/坐标模型

晶格 DDPM 与 fractional wrapped noise 分开定义，但共享网络读完整几何。其 Fourier 相对位移使周期性易处理；晶格内积进入消息，晶格头根据全图特征输出。这比“六个晶格 token 一类，3N 坐标另一类”的 loss 名称更接近真正联合建模。[作者 decoder](https://github.com/jiaor17/DiffCSP/blob/7121d159826efa2ba9500bf299250d96da37f146/diffcsp/pl_modules/cspnet.py)、[joint sampler](https://github.com/jiaor17/DiffCSP/blob/7121d159826efa2ba9500bf299250d96da37f146/diffcsp/pl_modules/diffusion.py)。

限制也实际存在：普通 lattice Gaussian prior 可落在不合理几何区域；CSP 默认给定组成及 N；多候选 oracle MR 不能当成实际可用 ranker 的效果。它证明一种有效路径，不证明分阶段分解不可用。

### 5.3 MatterGen：有因果辨识力的消融比摘要更有价值

MatterGen 的完整机制包括：fractional wrapped noise 按 `N^{-1/3}` 缩放；对称 lattice 的 VP 噪声靠近具有数据平均密度的 cubic mean；物种 D3PM；Cartesian 周期边/方向和 stress-like lattice head。**对称 Gaussian 矩阵不保证全路径 SPD**，不能和 log-SPD 的数学保证混为一谈。[corruption 实现](https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/diffusion/corruption.py)、[lattice score](https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/common/utils/lattice_score.py)。

SI Table A2 在同一 MatterGen-MP 框架内给出：完整模型 SUN 22.56；移除 N 方差调整 20.59、移除对称晶格 17.70、把 wrapped score 换成 CDVAE 近似 5.18、去掉 lattice-angle augmentation 2.16、物种 D3PM 换成 one-hot VP 4.32。这里的数字是该论文自己的 ≤0.1/DFT 协议。[最终补充 A.10](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-08628-5/MediaObjects/41586_2025_8628_MOESM1_ESM.pdf)。

最反直觉的一项：作者发现周期 GNN 对等价晶胞选择过于不变，无法表达其基相关 Gaussian 的 score，于是**主动加入 lattice-angle 信息打破该不变性**，再用 Niggli 数据规范化。这个例子反对“加入越多 invariance 就越好”。应让网络的对称性匹配所学的表示和噪声对象。

另有反证：去除 N 方差调整后 RMSD 更低但 SUN 更低；几何接近度和 SUN 并非同一目标。物种 D3PM 的大收益也不能直接搬到当前固定硬组成的 DLM 数值阶段；该阶段根本不预测物种。

### 5.4 FlowMM / CrystalFlow：可选择 base，但不是换个 noise 名字

以简单 flat torus 为例，令

\[
\delta_i=\operatorname{wrap}_{[-1/2,1/2)}(F_{1i}-F_{0i}),\quad
\bar\delta=N^{-1}\sum_i\delta_i,\quad
F_{t,i}=\operatorname{wrap}(F_{0i}+t(\delta_i-\bar\delta)).
\]

其训练对象是路径速度 `u_i=δ_i−δ̄`，终点可以是 target 的整体平移等价类；神经场通过 `E||vθ(x_t,t)−u_t||²` 学习输运，部署真正积分该场。晶格可以在线性或 log-polar 参数空间有另一条一致路径。**conditional path 不是原子真实运动轨迹，velocity 也不是力。**

FlowMM 使用数据拟合的正长度 base、角度变换及可选 anti-annealing；CrystalFlow 使用 log-polar k 和显式 property/pressure 条件。CrystalFlow 形成能条件的原生分布仍有明显偏移，优化后更接近目标；形成能低也不等于相对竞争相 hull 低。[FlowMM 作者 manifolds](https://github.com/facebookresearch/flowmm/tree/6a96aec3b6eba89f6fa07436f0c8837979abb285/src/flowmm/rfm/manifolds)、[CrystalFlow flow](https://github.com/ixsluo/CrystalFlow/blob/9c25ff0245d787efd87c9d1d797a8968443608cc/diffcsp/pl_modules/flow.py)。

技术核对发现 FlowMM/FlowLLM 的 HTML 角度公式写成 `(η−60)/120`，与其声称的 `[60,120]` 不对应；作者代码实际用 `biject_to(Uniform(59.9,120.1))`。正确一般公式是 `logit((η−a)/(b−a))`。这个小例子说明必须核代码与边界，不能把论文排版公式逐字当作可执行规格。

### 5.5 FlowLLM 和 CrysLLMGen 是两种不同的衔接

FlowLLM 用冻结 LLM 在训练组成上生成 base 结构，与数据结构形成 composition-matched pairs，再训练 RFM。其目标可写作

\[
\mathbb E_{A,\ G_0\sim p_{LLM}(\cdot|A),\ G_1\sim p_{data}(\cdot|A),\ t}
\|v_\theta(G_t,A,t)-u(G_0,G_1,t)\|^2.
\]

这使部署初始分布有直接训练对应。任意同组成 pair 可以用于**分布输运 coupling**，但不能改称“G₀ 的正确修复就是 G₁”；不同多晶型不提供同一盆地的真值关系。把这种 pair 直接作为我方逐原子 full-cell repair CE teacher，需要额外的原子对应、周期/晶胞规范及语义论证。

其 Types-only 消融表明 LLM 化学分布本身有价值，完整 geometry base 又有增益；但没有测试“Llama→额外 token DLM→refiner”的必要性。它还明确承认 LLM base 不严格 translation invariant；等变 flow 不会自动修复一个非不变 base 的全部密度性质。[FlowLLM §4.3/5.3](https://arxiv.org/html/2410.23405v1)。

CrysLLMGen 则独立训练 CSP diffusion，直接在 τ 注入生成几何。发布代码 `sample()` 明确把 `batch.frac_coords` 和输入 lattice 放到起点，未调用 `q_τ(·|G)` 添加匹配噪声。[作者实现](https://github.com/kdmsit/crysllmgen/blob/94bb287751cd20a882c7c1df7ca736633d78e5e1/models_ddpm/diffusion.py#L90)。这个方法可以经验有效，但一般

\[
p_{DLM}(G|A)\ne q_\tau(G|A).
\]

因此 τ 是经验校准的 sampler 设置，而不是“该 DLM 误差必然对应第 τ 步”的定理。本项目已有 tau800 增益不能被否定；也不能把它描述为经训练证明的精确后验或固定盆地精修。

## 6. V2 CE 为什么不会自动成为几何动力学

### 6.1 CE 的正确性与 score 的正确性是不同命题

若 continuous corruption 为 `Y=X+σε`，MSE denoiser 的最优解是 `D*(Y)=E[X|Y]`，在正则条件下有 Tweedie 关系

\[
\nabla_Y\log p_\sigma(Y)=\frac{D^*(Y)-Y}{\sigma^2}.
\]

这需要明确的连续测度、noise kernel、同一条件后验与几何变量。V2 学的是 canonical/legal 类别上的 `pθ(a|s)`；proper CE 可使其匹配声明状态下的目标类别 posterior。它没有因此自动获得上述导数、正确的 score normalization、可积分向量场或满足详细平衡的 transition。

V2 还先量化，再做完整支持拒绝，最多8次后回退clean。一个noise-grid level在这8次尝试中保持固定；条件于源G及该次请求的噪声级σ，若单次proposal admission probability为`α_{G,σ}`，则old-state分布具有

\[
q_{old}(z|G,\sigma)=\big[1-(1-\alpha_{G,\sigma})^8\big]q_{prop}(z|G,\sigma,accepted)
+(1-\alpha_{G,\sigma})^8\delta_{Q(G)}(z).
\]

整体分布再对所选grid level混合，不能先平均α再套8次幂。fallback的model-visible applied σ还会置零。这不是未经截断的Gaussian family。V2不宣称score/ELBO是正确的边界，审计不能把“不具备它从未承诺的score机制”误报为实现bug。

### 6.2 离散模型可以逼近动力学，所以不能判“token 永远学不会”

反例性的正面推导：在间距 h 的规则网格上，若有一致正密度 π，可定义最近邻 jump rate

\[
q(z,z\pm he_j)=\frac{D}{h^2}
\exp\{\tfrac12[\log\pi(z\pm he_j)-\log\pi(z)]\}.
\]

Taylor 展开得 drift `h(q+−q−)→D∂_j logπ`、方差率 `h²(q++q−)→2D`。所以离散 Markov 模型在适当极限能恢复 score-like diffusion。

但当前 scalar CE、不同 masked views、程序前缀和事务回滚没有定义成这套一致跳跃率；合法支持还随状态变化。证明“理论上可以”不能代替“本实现已经训练/使用了相同机制”。另一方面，按正确链式条件精确采样全部 token，本来就可以表示复杂联合结构，无需物理轨迹解释。

### 6.3 完整几何、固定 old、训练分布与量化需要分别分析

**构造早期。** 晶格/目标站点未完整时，没有真实目标坐标供几何图计算。V2 正确屏蔽未知几何；因此新增 geometry bias 不直接改善全 MASK 晶格起步。C³FD、composition prior 和 scalar token 模型承担这部分。

**修复事务内部。** old 是完整整数快照，state conditioner与bias读取其几何；已提交prefix则由current tokens可见。重算的是特征/gate，没有把每个新scalar变成新的old。[V2设计](../periodic_self_repair_v1/LOSS_GEOMETRY_AND_V2_PROPOSAL_20260906.md)、冻结 `periodic_v2_model.py` 的 `geometry_features()`。

这不是先验缺陷。若U为受扰old、Y为要生成的修复终点，则精确Bayes后验本来就分解为

\[
p(Y|U,C)=\prod_j p(Y_j|U,C,Y_{<j}).
\]

U作为观测应保持固定；scalar prefix是逐步揭示Y，不是下一噪声时刻。额外构造 `H_j=h(U,Y_{<j})` 的working canvas是已有观测的确定性函数，满足 `I(Y_j;H_j|U,Y_{<j},C)=0`，没有增加统计信息。它可能帮助有限网络计算新晶格与部分新坐标的关系，也可能因为hybrid几何不自然而使表示更难；若用H替掉U，还可能丢失原观测。

必须分清三件事：连续模型的外层噪声t更新；多个**完整**repair kernel之间把新晶体设为下一old；单个posterior内用prefix重算辅助几何。这三者不等价。没有证据要求V2每个scalar强制更新old；working-canvas只能是经隔离验证的特征候选，不是物理正确性的必要条件。

**合成 old 与生成 old。** 旧失败样本 In6Yb2 的 VPA 从 46.4117 到 12.7358 Å³，对数差为 −1.293，相当于 V2 最大 `σ_v=.2` 的 6.47 个标准差。这不证明 V2 一定失败，但说明“训练扰动形式合理”不等于覆盖已知过密错误。强压缩、错误配位拓扑、错晶胞形状、错误多晶型，不能统一归约为目标周围的小扰动。[旧逐样本诊断](historical/docs/teacher_feedback_unified_v1/26_ROUND0_GAP_INCREASE_DIAGNOSIS.md)。

**量化。** fractional 分辨率 h=.01 对 Cartesian 的尺度依赖 L；边长 20 Å 时单轴一个 bin 是 .2 Å。在正交晶胞、clean 位于 bin 中心的简化情况下，最低 `.05 Å` 扰动不改变该 scalar bin 的概率为 95.45%；5 Å 和 10 Å 时分别为 38.29%、68.27%。这些是示例，不是本项目实际 unchanged 比率，后者必须读 corruption audit。

若要精确编译 Wyckoff，100-bin 网格甚至不能表示 1/3：`|1/3−.33|=.003333`。对应对称操作不再在网格上闭合；某个宽松 symprec 仍可能识别 SG，但那是容差结果。不能声称“保留 7+4N 与所有原 bin 就能严格继承全 230 SG 的 orbit decoder”。

### 6.4 score 不是能量；对称点也可能是鞍点

只有在指定 ensemble 满足 `p(G|A)∝exp(−βE(G,A))`、采用一致参考测度且相关项处理正确时，才有 `∇logp=−β∇E`。数据库采样频率受发现/收录偏好、重复结构、计算筛选及条件频率影响，一般不是这个 Boltzmann distribution；有限 σ score 更是平滑密度的梯度。

简单反例：`E(x)=(x²−1)²` 对镜面对称，但强制对称点 x=0 会把系统固定在 `E=1, E''=−4` 的鞍点，真正低能结构在 ±1。约束切空间的优化不能排除破坏对称的负曲率模式。对真实晶体，对称性保持、力小、低形成能、低 hull、动态稳定、可合成性是不同性质。

SUN 还包含 N/U。即便每个样本优化后的 E 都更低，多样结构可能一起落入已知相，使 SUN 减少。旧结果中精修的新颖性下降和文献中训练更久 novelty 减少，均与这个机制相容。

## 7. Planner 软条件：何时是提示，何时是约束，何时是潜变量

主审提供的正式 V2 数据：train 24,558 直接预测、2,578 teacher-soft fallback；val 8,147/900。直接预测 train 上单字段相符数：lattice 14,103/24,558、SG bucket 13,987/24,558、volume 16,217/24,558。clean answer 字节保持原样。以下是条件语义分析，不由这些相符数直接认定失败原因。

### 7.1 随机新 soft 和固定 clean 的统计问题

令 C 包含**所有**推理允许的化学输入、N、anion framework、稳定性目标及元数据；Y 是 clean 几何；S 是 sampled soft plan。若生成新 S 的分布只依赖 C，则训练 population 联合为

\[
p_{train}(Y,S,C)=p_{data}(Y,C)\pi_{planner}(S|C),
\quad Y\perp S\mid C.
\]

在这个假设下，最优完整条件模型满足

\[
p^*(Y|C,S)=p_{data}(Y|C).
\]

所以对 S 的几何语义不敏感可能是风险最优解，不一定是 LoRA 没学会。若 π 还决定 program、观察顺序和 mask 集合，S 对有限模型仍可有**计算顺序**用途；给定真实可见字段及其索引后，它并不凭空提供目标 SG 的额外信息。最终结构分布在无限精确条件下可以与顺序无关，而有限误差下次序有作用。

适用前提必须检查：C 是否遗漏了源相关 metadata；Planner 是否通过训练记住了源几何关联；重复组成是否有多个多晶型；每源固定一次采样与 population 独立采样是否可区别；fallback 是否从 Y 导出真实 soft。单次 frozen annotation agreement 不能回答这些问题。

fallback 分支还会形成混合：一部分 `(S,Y)` 来自真实注释，一部分 S 条件独立于 Y。于是学习到的 S 语义可能是弱关联、局部关联或 provenance 相关，而不是纯忽略。独立 Bernoulli 数学例子已显示 9.5% 目标相关 fallback 能产生条件差异；这只是逻辑反例，不是对真实 Planner 的拟合。

### 7.2 三种语义要分别训练

| 对 S 的声明 | 所需训练联合/接口 | 可检验内容 |
|---|---|---|
| 化学提示、可忽略的先验 | `p_data(Y|C)` 与 sampled S；允许忽略错误提示 | 是否在稀疏组成上降低实际几何误差；不用“严格 obey SG”描述 |
| 用户可实现的几何约束 | 训练 Y 必须满足 `Y∈Ω(C,S)`；compiler/decoder 在同一 Ω 上更新 | realized lattice/SG/Wyckoff 与输入对应，兼容性与拒绝率；之后再看 SUN |
| 选择多晶型的 latent plan | 学习 `p(S|C)` 和 `p(Y|C,S)`，训练后验/标签必须把 Y 与 S 耦合 | 相同 C 下 plan 控制的多样性、实现率、稳定性；警惕 latent collapse |

DiffCSP++ 用已知模板/GT SG 和 orbit assignment 属于第二种；FlowLLM 把实际几何 base 输入输运路径，构成另一种明确耦合；DynaCrys 先输出有语义的 symbolic P，再以同一个 P 条件化几何。它们都不是“随机补上条件、clean 不变”便获得控制。

若未来让 Typed Llama 输出 Wyckoff program，需满足元素计数约束

\[
n_e=\sum_s m_s\,\mathbf1[A_s=e],\qquad\sum_s m_s=N
\]

以及 0D 位点不可重复占据、目标胞设置、基变换和 codec 可表示性。例如在标准 primitive SG 19 的通用轨道 multiplicity 为 4 的设置下，逐轨道同元素的计数必须由这些 multiplicity 拼出；任意 N/组成不能直接硬塞进去。conventional/primitive 转换若未一起处理，会把“计数不相容”误当几何网络失败。

本轮不建议在 V2 运行中强行硬化 sampled SG，也不建议因为 agreement 不足就改动 clean targets。候选首先应明确自己要上述哪一种语义。

## 8. 2025–2026 扩展：机制类别，不做论文名称堆砌

### 8.1 从模板约束到学习 symbolic occupations

**SymmCD。** 在 asymmetric unit 上联合生成 site-symmetry 离散变量、元素、坐标及 k；G 与 orbit count 先从数据分布采样。表示可减少图规模、学习未见 Wyckoff 模板。实质限制是独立轴预测的 site symmetry 未必组成合法点群，作者最后找最近合法 subgroup，再把坐标投到最近 Wyckoff 位置；这与全程精确 quotient likelihood 不同。其 main stability 实验用 CHGNet，不能当作 DFT SUN；“10 常见 SG”子设定改变了待生成分布。[方法及 App A](https://arxiv.org/html/2502.03638v1)。

**WyckoffDiff。** D3PM 的对象是**protostructure**：固定点位选元素或 vacancy，有自由度的 Wyckoff row 对每元素生成 occupancy count。`Q_t=(1−β_t)I+β_t1mᵀ` 可选择 uniform、empirical marginal 或 zero base；稀疏 occupancy 下后两者更合适。结构质量的主体指标是 protostructure FWD/N/U；示范发现先 PyXtal 多次赋自由坐标，再约束 MACE relaxation，再筛最低形成能的 200 项做 DFT，不能改写为未经筛选的 200/全请求 SUN。[原文 §3–5](https://arxiv.org/html/2502.06485v4)、[D3PM 实现](https://github.com/httk/wyckoffdiff/blob/9c4694291be1c20e4f06cc33f02baed7daaf0b7a/wyckoff_generation/models/d3pm/d3pm.py)。

**WyckoffDiff-Adaptor（2026）。** 增加 chemical-system、hull 等条件 adapter，先离散 protostructure 后 PyXtal 实现。其自身展示 Ba-Ta-In-O 的不合理 O 配置，以及相同 MgAl₂O₄ protostructure 因氧自由参数不同得到不同几何。这个失败对 V3 极有价值：**压缩语法不等于键合/自由坐标学习已经解决**。其约 6 倍时间优势也含 prototype→CIF 后处理并允许最多 100 次 PyXtal 尝试；没有本项目单请求单轨迹 SUN 保证。[原文 §2–3](https://arxiv.org/html/2601.08115v1)。

这三类可自然接入 Typed Llama 的化学/占位程序，但都不允许省掉实际 geometry generator 后把 SG 语法当作稳定性。

### 8.2 SGEquiDiff：对称向量场的更强理论，及其反例

其主分解为

\[
p(G)\,p(L|G)\,p(W,A|L,G)\,p(F|W,A,L,G).
\]

晶格使用 telescoping categorical sampler（粗 bin→细 bin），Wyckoff/element 用无位置编码、带语法 mask 的 Transformer，坐标用 space-group-wrapped-normal score 和群平均 GNN。因此它直接证明**离散晶格+符号 Transformer+连续坐标**可以是一套完整方法；不要求所有变量放进同一个连续 diffusion。

关键推导：若 `g(x)=R_gx+t_g=x` 且 `v(gx)=R_gv(x)`，则 `(R_g−I)v(x)=0`，于是 `g(x+hv(x))=x+hv(x)`。向量自然位于 stabilizer 的切空间。噪声也必须投到相同自由空间；作者实现确实投影 Gaussian noise，并对数值 score 作小的投影修正。这不是“用了 SE(3) GNN 就任意 SG 自动成立”。[定理与实现](https://arxiv.org/html/2505.10994v3)、[diffusion model](https://github.com/rees-c/sgequidiff/blob/2231031fdabbf6b372031f3f233f8638922fa91e/src/model/diffusion/diffusion_model.py)。

它还有一个重要动力学边界：确定性等变 field 无法穿过某些更高对称的 0D 点连接两段 1D Wyckoff 区域，故使用 projected stochastic motion。这说明“更强硬约束”也可能限制可达路径。

**独立的metric攻击：不能原样继承其SGWN全部理论主张。** 上述stabilizer→tangent的代数结论成立，但“等方Euclidean Gaussian的群平均密度是G-invariant”要求群操作在该Euclidean metric下是isometry。fractional基的三方/六方矩阵一般不正交。此段改用列fractional坐标，取

\[
R=\begin{pmatrix}0&-1\\1&-1\end{pmatrix},\qquad
H=\begin{pmatrix}1&-1/2\\-1/2&1\end{pmatrix}.
\]

它满足 `R³=I, RᵀHR=H`，却不满足 `RᵀR=I`。这可嵌入3D三方晶格。对 `x₀=(.13,.24), x=(.18,.26), σ=.04`，把三个orbit中心和充分多integer images作等方Euclidean Gaussian求和，得到 `q(x)=.40404`、`q(Rx)=.66614`；改成metric平方距离 `(x−gx₀−n)ᵀH(x−gx₀−n)` 后两值均为 `.55225`。差异不是数值舍入。

公开代码 [`p_asu_wrapped_normal` / `d_log_p_asu_wrapped_normal`](https://github.com/rees-c/sgequidiff/blob/2231031fdabbf6b372031f3f233f8638922fa91e/src/model/diffusion/diffusion_utils.py#L452) 明确对fractional differences逐轴平方求和，未接收晶格metric；训练调用也直接传fractional coordinates。因此公开helper与全SG等变score之间存在需要核清的metric/坐标假设。另即使标量密度满足 `p(Rx)=p(x)`，普通坐标梯度也按 `s(Rx)=R^{-T}s(x)` 变换；只有metric-raised vector `v=H^{-1}s` 才按R变换。

本审计已用独立NumPy例子证实这个数学反例，但未运行该作者checkpoint。一个重要竞争解释是：实际训练/采样先wrap到canonical ASU，可能只拟合helper在该域上的限制，再通过group averaging定义域外extension。因此反例严格针对全空间fractional isotropic-Gaussian公式的SG不变性，不由此宣判ASU sampler或其实证全部无效。完整预处理/密度extension仍需核对。可安全借鉴的是条件分解、symbolic约束、tangent构造与正确metric的群化方法，不能不经核验地复用“所有fractional SGWN已严格证明”等字样。

原文 `+LDiff` 联合晶格扩散消融没有一致胜过 AR 晶格；原主模型 validity/Wyckoff/SUN 更好。其 DFT SUN 是从 1,000 **双 valid** 结构中评估 U.N. 再做 DFT，门槛 `<.1`，不能把约 24% 当全请求 Strict SUN。对称表征的强结论值得借鉴，排行榜条件必须保留。

### 8.3 DynaCrys：动态符号选择，比随机 soft 提示强在哪

该 2026-08 预印本在符号扩散中让 G、orbit count、Wyckoff row 和物种一起演化；SG corruption 利用 group-subgroup 图，row vocabulary 随 G 变动，共享 crystallographic codebook 连接 symbolic 和 geometry 两阶段。最终 geometry stage 在合法 P 的自由度上生成。其代价是全新符号空间、codebook、set matching 与约束 decoder，远多于给当前 soft plan 添一个字段。[原文 §4](https://arxiv.org/html/2608.07401v1)。

其“exact”结论应精确理解：拒绝采样在接受条件下采到 `p_fac(x₀|x_t)1[x₀ legal]`；不证明 factorized heads 就是真实 clean posterior，不证明 predict-clean→renoise 的整个链等于任意原始 Markov posterior，更不证明 SUN。主表用 CHGNet 与 MACE 两套评估；同一表中其显式对称模型在 MetaSUN 仍低于非显式对称 MatterGen。动态符号的某些收益也不是全目标共同改进。现阶段无可得作者代码，作为 V3 概念候选而非优先移植实现。

### 8.4 SymmBFN：更新分布参数的另一种迭代，不等于 DLM CE

BFN 迭代的是 Gaussian/categorical 的参数（均值、精度、概率），网络预测 clean 分布，Bayesian update 累积信息。SymmBFN 用 asymmetric unit、k mask、离散 atom/site symmetry，最后匹配合法点群/最近 Wyckoff 并展开。其几何 carrier、sender distribution、accuracy schedule 和 loss 均需一起定义；“softmax 概率逐步变尖”并不足以把 DLM 称为 BFN。其论文将全部稳定性统一到 CHGNet，而属性实验只对 metastable 子集画分布，必须保持这一选择条件。[Methods/评测](https://www.nature.com/articles/s41524-026-02140-8)、[BFN 实现](https://github.com/aimat-lab/symmbfn/blob/697e99ae910a1336aa51e2be838d7c6e7f3fdc11/symmbfn/model/bfn/bfn_dng.py)。

### 8.5 DiffCrysGen：没有显式等变 GNN 的成功也是反证

其 IRCR 把物种、晶格、fractional 坐标、occupancy 等编成统一矩阵，用 1D UNet/EDM denoiser 在连续矩阵上拟合 clean prediction，再通过 sampler 更新。它不保证 SG，而试图从数据学习规律。[原文 Methods](https://www.nature.com/articles/s41524-026-02147-1)、[EDM/UNet 实现](https://github.com/SouravMal/DiffCrysGen/blob/88562cb8ec89c67272b191b362b340104fbd7f56/diffcrysgen/model.py)。

其成功不构成“任何 token LM 都应一样”的证明：最终 Alex-1 只有 unary/binary/ternary、Z≤94、≤20 atoms、非零磁性、≤.1 hull、非正形成能筛选的 151,294 结构，约 94% ternary；1.32M 模型最多训练 1000 epochs。这与全 MP20/生成化学支持不同。它支持“显式 GNN 不是必要条件”，同时支持“数据支持、表示和正确 denoising/update 比模型参数总量更有辨识力”。

### 8.6 ChargeDIFF：额外物理信息有价值，但不是无需数据的 trick

增加 charge-density modality C，先用 VQ-VAE 把 `32³→8³` 压缩，再把 charge probes 与 atoms 作为异构图节点，联合预测 A/X/L/C。AXL-only 与完整版本的 mSUN 消融是 21.3→24.8；粗采样低分辨率 C 而不做 latent compression 则下降到 19.6。说明附加物理表示必须有足够信息量，不能只是加一个辅助头。[原文结构/消融](https://www.nature.com/articles/s41467-026-73985-2)。

其训练用新筛选的 40,516 个有 charge density 的结构；评估有 MatterSim→DFT。当前 V2 无 charge-density 标签，也无此图/decoder，因此不是一至两 epoch 的低代价候选。可迁移的抽象是可靠物理中间表示通过消息传递互相约束，不是凭空从 Llama hidden 声称获得电子结构。

## 9. GNN 如何真正接入 DLM

### 9.1 当前已经有 GNN-like state conditioner，不能误报缺少几何 hidden

这是本审计沿调用栈检查后修正的关键点：`state_conditioned_model.py:158–180` 调用 `state_conditioner(**geometry_inputs(...))`，把 `cell_embedding` 加到六个 lattice slots，把 `site_embeddings` 加到每个 native site 的 E/XYZ slots。冻结版本的 `periodic_state_conditioning.py:222–245` 具体执行：周期 image 的 RBF 与相对 Fourier/元素特征→`pair_mlp`→邻居平均→global cell与site MLP→hidden projection。它已经是一跳的 learned periodic message aggregation，而不是只有距离 bias。两个文件与冻结SHA一致。

还有一条直接通路：known-site的邻居环境汇总进入cell embedding，再经`site_update`广播到所有present site，包括坐标尚未完整的active site。故“active query的直接attention几何边为零”也不能扩大为“active hidden必须经过Transformer多跳才有物理几何”；conditioner本身已提供global几何。初始六lattice scalar时完整几何仍不存在。

因此，“geometry 尚未进入 value/hidden”这个初步猜想不成立。已有注入自然会进入LM的Q/K/V。必须以完整调用链为对象，而不是只看新增 `periodic_v2_model.py`。

V2 现有 attention 大体为

\[
\mathrm{Attn}_{ij}^{(h)}\propto
\exp\{q_i^{(h)\top}k_j^{(h)}/\sqrt d+b_{ij}^{(h)}(G_{old})\},
\qquad o_i^{(h)}=\sum_j\mathrm{Attn}_{ij}^{(h)}v_j^{(h)}.
\]

这条**新增边 bias 分支**改变路由，而已有 state conditioner 已经提供几何内容；多层 LM 还能进一步混合它们。若新增 GNN 仍只是重复一跳 radial aggregation 或相同距离 bias，容易与已有分支重复。

可讨论的增量是多轮站点级 nonlinear environment：

\[
u_i^{(l+1)}=u_i^{(l)}+
\phi_h\!\left(u_i^{(l)},\sum_{j,n}\phi_m(u_i^{(l)},u_j^{(l)},e_{ij,n})\right),
\]

其中 `e` 含 Å 距离、相对方向/周期 image、元素、晶胞信息。已有conditioner约对应一个edge→node聚合及全局池；多层专用GNN可进一步组合邻居的邻居，显式方向/三体项可区别仅两体径向统计难以区分的环境。但当前LM也有多层非线性，所以专用GNN更有效是待测假说。是否必须用严格等变tensor network取决于输出对象，而不是GNN名字。

### 9.2 四种接口，各自不同的保证

| 接法 | 数学/代码接口 | 原输出 ABI | 能保证什么；不能保证什么 |
|---|---|---|---|
| **升级已有站点 hidden 残差** | 现有 `h←h+W u_i` 改为多层/方向性MPNN，或实验逐层value融合 | 保留离散 7+4N | 增加环境计算能力；不是首次接入几何hidden，不保证整LM等变、降能或SUN |
| **跨 attention 的 site memory** | 几何图作为 N 个 memory keys/values，lattice 另给 pooled/global state | 保留 | 可让所有 scalar query 读完整图；代价是 cross-attention 权重和状态对齐，仍非几何更新器 |
| **几何 auxiliary head** | DLM/GNN hidden 另回归同源 corruption 的位移/score；训练后不用连续输出 | 保留推理 ABI | 能检验 latent 是否含方向/尺度；辅助头准确不保证 categorical actions 使用这些知识 |
| **等变 denoising output head / refiner** | GNN 输出当前完整几何上的 joint vector field，LM 给 invariant chemical/global conditions | 若输出连续结构则改变 native 定义 | 可按设计保持 field 的等变性/周期性；需新 forward kernel、目标、solver 和单独物理评测 |

若所有位置未知，任何一种图接口都没有真实几何；只能使用 chemistry graph/soft prior，不能伪造 coordinates 当作已知事实。因此这些接口最先应该在 full-cell repair 中建立作用，而非承诺自动解决全 MASK 起步。

### 9.3 等变性不会穿过任意 LM 路径自动保留

设 GNN 为 `Φ(PG)=PΦ(G)`，将它复制/投影到每个站点的 XYZ token 可以使**该分支**随站点置换正确变化；但固定位置 embedding、species program rank、非等变的 token path 若没有同时变换，整网仍可违背这个性质。空间群等变更严格：只有 SE(3)-equivariant GNN 加周期邻接，并不足以保证某个给定 SG 的 Wyckoff tangent 约束；需 orbit expansion、group averaging 或等价构造。

对全局旋转，六 lattice 参数+F 已经选择了 orientation-free 表示。若 GNN 只输出 rotationally invariant scalars 注入 LM，不必额外要求输出旋转向量；如果输出 Cartesian vector，则所有合并路径必须按相同规则变换，scalar gates 也要 invariant。

尤其不能混淆速度与 score（仍采用行向量）：

\[
\Delta F=\Delta R\,L^{-1},
\qquad s_F=s_RL^\top.
\]

前者是向量位移的坐标变换，后者是对数密度梯度的 pull-back（固定 L，Jacobian 对 F 为常数）。把 `L^{-1}` 同时用在二者上会产生错误量纲/方向。联合 lattice score 还要考虑 `R=FL` 的晶格依赖，不能只是把坐标 force 平均成六 lattice token。

### 9.4 四至六 A800 与一至两 epoch：容量可承载不等于已经可学

- N≤20 的浅层 `d≈128–256` MPNN 及共享投影，新增参数可在百万量级；其计算通常远小于 8B LM，但 OTF 周期图在过密坏晶体上仍可能昂贵，需测实际节点/边数和 step 时间。只用 node count 估 FLOPs 不可靠。
- 新增 2–4 层 MPNN 仍可能需要从零学习几何，文献常用大量重新采样的 noisy states。两 epoch 的非零梯度证明梯度可达，不能证明环境表示或几何方向已学成。
- 可冻结已有、与表示相容的预训练 crystal encoder，再训小 projection；但预训练数据/任务/计算必须计入证据。若用 CHGNet/MACE embedding，即使不读取 energy output，也在 native inference 引入了 MLIP-trained 网络，不能继续宣称原有“native 无 MLIP”。
- zero-initialized output projection 可以保持初始 logits 不变并首先训练 projection；不要同时把 gate 与该 projection 都设为零，否则可能形成乘积死梯度。是否保存/重建、dtype/shape 和实际几何输入一致才是必需工程检查；无需为每个 job/epoch 写一套硬编码 guard。
- 同预算应比较“已有state conditioner+多头bias”、加深/升级其消息传递、以及替代式方向性MPNN；保留相同working-state定义。若现有一跳聚合已够，增加GNN的复杂度没有证据。

主审的 [Transformer/GNN 补审](TRANSFORMER_AND_GNN_SUPPLEMENT.md) 进一步核对 ADiT：更好的 EquiformerV2 VAE 重建不必带来更好的后续生成，增大 DiT 的参数也没有单调提高 SUN。这是本文不预先支持新 GNN 的外部反证；该文件为另一 lane 证据，不在本 lane 冒称独立复现。

## 10. 训练量、数据支持与旧失败：成功原因不能只归为 architecture

| 工作 | 可核训练/数据量 | 正确解释 |
|---|---|---|
| V2 | 原始 27,136 train、9,047 val；2 epochs×2 views；4,524 updates，原 LLaDA 初始化 | 每个 source 的 clean label 没变；两 view 不等于两个独立结构；prefix 多为一个 scalar 目标 |
| DiffCSP / DiffCSP++ | MP20 最大 1000 epochs；DiffCSP++ 6 layers/512 hidden、1000 diffusion steps | 同 source 可反复经历大量随机 noisy states；不是每个 epoch 都产生新材料 |
| CDVAE 发布 MP20 config | max 1000 epochs、batch256、50 noise levels、sampling 50×100 | 不能把 5000 sampling steps 误称 5000 train epochs |
| FlowLLM | LLaMA-2 70B LoRA 10 epochs；3.3M base-target pairs，RFM 最大20 epochs；base sampling约250 A100 GPU-days | input-distribution 对齐用了大量生成计算；不是零成本两阶段组合 |
| MatterGen final SI | 607,683 Alex-MP-20；base 1.74M steps×8 GPUs×64/GPU | 数据与 noisy-state exposure 远大于原 MP20；不能只拿44.6M vs8B比较能力 |
| SGEquiDiff | 最大1000 epochs、各模块独立 early stop；约10h/A40（MP20） | 小专业模型可高效训练；其训练状态/目标与 LM CE 不等价 |
| WyckoffDiff | WBM约257k、留10k+10k；1000epochs/batch256，约38h/A100 | 是符号 protostructure 任务，geometry realization 成本另外存在 |
| DiffCrysGen | Alex-1 151,294、80/10/10；最多1000epochs/batch64 | 只覆盖筛选的≤ternary磁性化学；不等于完整 MP20 迁移 |

这些量是来源预算/配置，不能按 epoch 直接做算力或收敛比较；early stopping、dense vs scalar targets、自然语言预训练与材料预训练均不同。`mechanism_checks.json` 只给出透明的 exposure 乘法，不据此得出“V2必然欠训”。

旧项目已经提供反例：`13_STREAM19_DIAGNOSIS_AND_FINAL_ITERATION.md` 中初判 UNDERTRAINED，追加一轮后最终条目撤回了这一单因果判断；新 prospective stream raw 回退、refined Strict 不改善，被改判 LOCAL_CONTROL_INSUFFICIENT。因为评价 stream 不同，它也不构成纯 epoch 因果实验。

其他旧证据应按机制串联：

1. [K4支持诊断](historical/docs/teacher_feedback_unified_v1/27_SELF_IMPROVEMENT_REPAIR_PLAN.md)：972 verified 路径、526 有监督组成中，只有13组有 A≤.1 的候选；同池任挑最小A的均值仍5.48 eV/atom。说明该候选池再加权不能制造不存在的近平衡支持，而非整个模型空间不可能。
2. [过密样本诊断](historical/docs/teacher_feedback_unified_v1/26_ROUND0_GAP_INCREASE_DIAGNOSIS.md)：多个大幅A变坏样本同时缩胞/缩短接触距离，而且差异已经在construction终点出现；不能把所有失败归咎某一次closure。
3. [连续扩展审计](historical/docs/teacher_feedback_unified_v1/24_CONTINUOUS_DIFFUSION_TRAINING_AND_SDE_AUDIT.md)：model494 没有独立长期 Plan/program 通道；它能改变盆地概率，不能预设“只作局部精修”。训练教师与 continuous sampler teacher 也不是同一个对象。
4. V2 已把 sparse repair、部署 prefix mismatch、old 几何不可见等明确问题修正。这是针对实质弱点的设计进展，但把多项一起改变后，正/负结果都不能自动归因到其中一个模块。

独立的成功机制还可能来自训练数据中的化学组成先验，而非几何网络：如果某些 C 根本没有本协议下的低 hull 结构，固定组成 decoder 无法补救其不可达性。对 C 的选择质量与给定 C 的几何实现质量必须分开检验。

## 11. 最小候选与必要时的 V3 分支

所有候选均为**研究建议，未实现、未训练、未更改 V2**。不恢复已暂停的新 K4/K8、能量 teacher 或连续联合训练。优先顺序应由正式 V2 的 construction/repair 诊断决定，而非论文热度。

### 11.1 小改动候选

**候选 A：评估prefix-derived working geometry作为辅助特征。** 用“已提交current值+尚未提交old值”构造H，同时完整保留U=old；保持同一scalar输出、composition、程序和支持。它不提供新统计信息，也不要求改变old，仅测试是否能让有限网络更容易计算prefix与旧观测的关系。训练必须出现这种混合前缀，hybrid几何可能暂不相容；不能未经训练替换输入或把无效hybrid当真实稳定晶体。它若有效，增量可能集中于prefix已经明显改变晶格的状态；若现有hidden已学会这种关系，则应取消该候选。

**候选 B：在已有图式 conditioner 上验证多跳/方向性增量。** 复用其site/cell→hidden接口与相同canvas，比较已有一跳聚合和浅层多跳或显式角度/方向网络；不要再并列添加一个几乎相同的一跳GNN。可以先检验跨站点的局部环境消融，确认现有通路已被学会后再决定是否加深。它只改变表示容量；SUN无保证。

**候选 C：有限网格闭合的 nuisance augmentation。** 对 clean/current/old 与对应程序索引一起做合法、可逆的平移/站点变换，控制 origin/order 任意性；在相同 source-view 预算内采样。先验证在真实 codec/support 下确实闭合，再谈一致性 loss。几何已规范化且部署同规范时可能无益，不能视为必需正则。

**候选 D：按真实错误类型校准 corruption，而非照搬某篇 σ。** 用训练来源、无终态选择的生成样本诊断 log-volume、形状、接触、配位和 token-change 分布；再预注册后续 synthetic corruption family。可以优先保持物种和可用 atom correspondence，加入 collective distortion，避免只覆盖独立小扰动。调大σ本身也会破坏可恢复性、增加拒绝/fallback；它不是单调改进。

### 11.2 需要新机制时，三个边界清楚的 V3

**V3-S：Typed Llama 产生可实现的 symmetry/occupancy program。** C³FD 硬化学→Llama 选择兼容 G/W/元素轨道→compiler 保证计数/容量→DLM 实现剩余自由参数→独立 refiner。DiffCSP++、SGEquiDiff、WyckoffDiff/DynaCrys 提供不同分解。优点是低维可实现条件；代价是新的监督/codec/量化闭合、symmetry metadata 与 wrong-plan fallback。若维持原100-bin full-cell输出，严格SG承诺必须缩小到可表示子集或取消；不能两者都宣称全覆盖。SG不是当前SUN目标的必要附加限制。

**V3-G：DLM hidden 与周期 GNN 构成联合可见状态的离散 denoiser。** 仍输出7+4N；以完整 old/working canvas、chemical/program 为输入；GNN提供环境信息，DLM学习条件类别。若另加 vector auxiliary loss，它是训练信号而非 native连续decoder。必须证明离散 action 消费了几何特征，且更好 hidden reconstruction能转化到同请求SUN。

**V3-R：适配实际 DLM base 的独立几何 refiner。** 借鉴 FlowLLM 的 distribution matching，把冻结最终 DLM 的训练分布作为base；与正确条件/周期规范建立输运或同起点可靠修复数据。优点是初始分布对齐；代价是额外生成、标签/配对、refiner训练，以及收益归属于系统端。不能据此声称DLM原生稳定性提高。这条路径目前仍暂停，只在另一轮明确解冻时实施。

### 11.3 必须保留的架构否证对照

如果论文主张 token DLM 是实现化学计划到几何的关键，应容许比较同一 C³FD/Llama 化学条件直接送入专门 crystal generator/refiner。SGEquiDiff 和 FlowLLM-Types 已经表明这类分工可行；现有自然语言模型不能仅靠“8B、更通用”排除它。

DLM 的独特价值可能是离散可回放、程序控制、统一语法、部分结构 infilling 和 suffix-visible修订，但这些性质必须对应测得的功能或SUN收益。若简化架构在同条件/同预算下更好，应如实报告，而不是增加更多耦合模块以维护既定故事。

## 12. 可检验预测与最强反证清单

| 假说 | 预注册可观察量 | 支持它的结果 | 否证/降级结果 |
|---|---|---|---|
| prefix-derived几何特征帮助有限条件模型 | 固定完整old与训练条件，只加确定性H特征；按实际ΔL/ΔF分层 | H增量集中于特定困难状态并提高最终同端SUN | 无增量/变坏；fixed-old本来正确，不能再宣称必须实时更新 |
| 已有一跳geometry conditioner不足 | 相同canvas的现有模块vs多跳/方向性MPNN，计入参数和计算 | 多跳/方向信息有最终同端SUN增量 | 已有模块解释全部效果，或辅助任务改善而SUN无增量 |
| 两epoch noisy-state覆盖不足 | fixed clean sources上的同noise/同state验证曲线与实际self-states | held-out self-state conditional风险持续下降，并迁移到SUN | clean loss下降，self-state风险/同端SUN不变或回退 |
| soft S 真正控制结构 | 固定C和初始随机变量，改变S；同时核实现出的字段/SG、计数 | 对应几何响应且在合法/物理评测上成立 | 仅token差异/程序次序改变；或最佳模型忽略S而SUN更高 |
| symmetry是主要瓶颈 | 相同C下真实兼容G/W约束与无约束的匹配对照 | 对称实现、几何质量与SUN一起增加 | 更对称但坠入错误盆地/新颖性下降；即不能上升为核心目标 |
| GNN表征缺失而非decoder更新缺失 | GNN辅助预测、离散action变化与最后SUN三层验证 | 三层均有对应增量 | auxiliary误差改善但token/SUN不改善（已有ADiT相关反例） |
| 量化限制近平衡 | 连续clean→native codec→共同R，同源配对并保留全体 | codec本身造成显著可解释物理误差 | clean量化基本无损，不能把生成错误归因分辨率 |
| 合成perturbation覆盖生成错误 | 训练域内 synthetic/self 误差统计与model难度 | 难度/错误模式相近，repair收益可迁移 | 密度/配位/collective错误明显失配；调σ仍不覆盖 |
| 系统端改善来自初始分布匹配 | 同一DLM bodies、固定评价器下适配refiner vs旧494 | 匹配base的refiner提高系统SUN | 仅均值能量/速度改善；SUN未改善，或新颖性被压缩 |

全部最终胜负以完整请求分母和同一端点的 Strict/Meta SUN 判定；诊断 A/B、force/stress、geometry/conditional loss 可以解释机制，不替代 SUN，不在 final 用户摘要里堆叠代理指标。

**最强反证汇总：** (i) DiffCSP++ 的 log-k-only 没有提高MR；(ii) CSPML模板解释大部分高MR；(iii) MatterGen打破不适配的cell-choice不变性反而有收益；(iv) FlowLLM不完全translation invariant仍能有效；(v) SGEquiDiff分阶段AR lattice不逊于joint diffusion；(vi) WyckoffDiff-Adaptor明确失败于自由坐标/不合理键合；(vii) DiffCrysGen不用显式GNN但其成功依赖不同数据域；(viii) 旧项目额外一轮并未解决Strict目标；(ix) 更好几何重建与更高SUN不等价；(x) 合法/对称支持不能制造训练候选池中没有的低能几何。

另有一个数学层面的反证：fractional basis下不正交的SG操作不能直接与Euclidean isotropic score混用；8.2给出对公开SGWN helper的具体攻击。这比笼统认为“知名等变论文已有严格保证”更严格。

## 13. 未决点与审计产物

1. 公开仓库 HEAD 与论文/下载 checkpoint 的完全对应未普遍给出；FlowLLM 的 paper `λ_F=200,batch256` 与公开 generic `null_params`/data config 的 `400,128` 有差异，不能擅自选一套冒称复现。
2. SymmCD v1 与 SymmBFN 主文出现“MP20=40,476”表述，与原始 MP20=45,231 不一致；本审计没有下载各方法完整数据重数，训练量比较保留这个来源不一致。
3. CrysLLMGen 主文/附录的评测数量、SUN分母用词以及 score 的排版表达存在不一致；以本项目实际 evaluator 与 frozen artifacts 为准，不能沿用文献口号。
4. SGEquiDiff、WyckoffDiff 等方法的筛选、valid-only、重复生成/物理实现成本与我方单请求分母不同。本报告不据公开分数计算“领先我方多少”。
5. DynaCrys 为最新预印本；本文已读方法/核心实验，但未找到作者代码，无法独立核 sampler exactness 的工程实现及全部评测边界。
6. 新论文的一些 Nature HTML 数学公式未完整进入 text extraction；凡本文给出自行推导的公式都已明确作为分析使用，没有把缺失公式猜成原文。
7. 没有通过本地几何/MLIP运行重新标注训练数据，没有复现任何外部模型。因此论文消融是作者证据、CPU例子是数学证据、项目结果是已保存/主审提供证据，三者不互相替代。
8. SGEquiDiff公开fractional SGWN helper的metric前提存疑，见8.2；须用真实三方/六方样本和完整预处理进一步核查。tangent定理与SGWN密度/score定理要分别评价。

### 审查中已撤回/修订的初步表述

| 被撤回的猜想 | 导致撤回的直接证据 | 最终表述 |
|---|---|---|
| 当前geometry只有attention bias，尚未进入hidden/value | 冻结`state_embeddings`与`PeriodicStateConditioner`已经有periodic pair messages、neighbor pooling、site/cell projections | 已有一跳图式conditioner；新GNN必须证明多跳/方向/融合方式的边际价值 |
| continuous每步更新几何意味着DLM每个scalar都必须更新old | `p(Y|U)=∏p(Yj|U,Y<j)`允许且需要保持完整观测U；hybrid H是确定性特征不增加信息 | 区分outer noise/kernel时钟和inner posterior factor；working feature只是有限模型的候选 |
| DiffCSP++作者代码也使用论文展示的未单位化基，因此其shape noise与V2正交basis不同 | peer交叉核对`get_basis():33–35`实际归一化，hex bias同步乘√2；本lane重新读冻结源码确认 | 论文未归一记号`v=3k₆`，代码normalized basis为`v=√3k₆`；V2差别在独立volume/shape尺度，不能混用两套σ换算 |
| 对noise调用affine P可直接判定offset错误 | 需复合forward/reverse最终P；`Mb=0`使固定offset在自由分量消除 | 不机械判断单个helper调用，分析整个更新器 |

独立可复核产物：

- 本报告；原始来源收据 `evidence_crystal_diffusion/source_receipts*.json`、`extended_repo_receipts.json`。
- `evidence_crystal_diffusion/code/`：带明确版本来源的作者代码片段，未执行。
- `evidence_crystal_diffusion/papers/`：所读原论文HTML/提取文本及MatterGen最终SI，用于回查；不作为原创内容发布。
- `evidence_crystal_diffusion/mechanism_checks.py/.json`：CPU数值例子，验证log-SPD旋转/基变换差异、周期score抵消、量化尺度、软条件混合与训练exposure计算。已执行，数值检查通过；这些不是模型性能测试。

V3 的优先决策应先回答“具体缺失了哪种可学习对象/状态”，然后选择最小改动。没有一条证据要求立即扩大架构；也没有一条证据允许因为当前几何机制尚不完整，就断言离散DLM路线必然无效。

## 14. 对概率 lane 第一轮的交叉攻击

已只读审查 [DLM probability/kernel审计](DLM_PROBABILITY_AND_KERNEL_AUDIT.md)。其逐scalar联合分解、C→D数据增强kernel的详细平衡证明、mask-time与geometry noise区分均与本文一致。没有修改该lane文件，以下已发送主审与作者。

### 14.1 C→D的stationarity不是SUN的理论优先级

概率lane正确证明：若 `J(Y,U)=p_*(Y)C(U|Y)`、D为J的Bayes posterior，则 `K=CD` 保持p_*。但应加一个更强的目标反例：取p_*为有限训练结构经验分布，novelty reference包含这些结构，C有正支持；完美D在任何U下只输出training-support。于是对任意起点，一次D后Novelty=0，SUN=0，尽管详细平衡精确成立、每个训练结构都可以物理稳定。

这个问题不是C→D独有，也不是否定生成模型的泛化；它说明理论中的p_*必须被解释成并另行论证为可泛化population，而不是把经验训练stationarity自动升级为新材料发现保证。C→D比反复D更有明确概率对象，可作为有界诊断，但这不是它必然更接近SUN的定理。

### 14.2 后期active geometry还有conditioner直达通路

该报告对attention bias的CPU测试可成立；但后期active query还可能直接收到known-site→global-cell→active-site的conditioner几何，见本文9.1。因此几何瓶颈应具体区分“无完整几何的初始晶格”“没有active direct bias edge”和“整个active hidden中没有几何”三个命题。只有第一个在construction早期有明确支持，第三个不能从bias矩阵零行推出。

### 14.3 支持内的表达能力与全局合法posterior仍需区分

所有“任意离散分布可由CE/条件链表示”的命题都应限定在声明可达支持内。概率lane已提供prefix admission不等于全局合法posterior的反例；这与本文corruption拒绝/fallback law相容。仅规范化logits无法把被支持规则排除的clean结构重新变为可学习终点。
