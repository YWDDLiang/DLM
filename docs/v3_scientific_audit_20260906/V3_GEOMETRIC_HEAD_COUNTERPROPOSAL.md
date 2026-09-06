# V3 独立反提案：共享 DLM 主干的连续晶体几何头

日期：2026-09-06。状态：**供两轮独立攻击的机制草案；未选择为 V3，未授权实施或训练。** 本文只读取冻结源码与已核原始文献，运行 NumPy 数学与预算检查，没有 SSH、GPU、模型调用、安装或模型代码修改。执行基线为 `2e904c260bafb6750c9a5dbb0d920c4ff8c3a868`；本地冻结文本与 SHA256 见 [source_manifest.json](evidence_geometric_head/source_manifest.json)。

## 1. 独立裁决

**共享预训练 LLaDA/DLM 主干、让现有 LoRA 与几何条件模块直接接受连续去噪损失，是值得比较的中等改动候选。它比把几何增量重新渲染成 scalar logits 更直接，也能避免为保持纯 token 身份而先构造一套新的晶格离散转移系统。但目前没有证据证明它比同预算的密集 token 去噪提高 SUN，更不能据此启动新训练。**

它最具体的潜在收益有三个：每个几何训练 forward 同时监督六个晶格和全部 `3N` 坐标通道；迭代状态保留全精度而不在每一步量化；每个反向时刻用同一个 Transformer 联合预测整个晶胞。它们分别是监督分配、表示精度和反向计算图的改变。三者都不是“continuous 天然更物理”的证明；密集监督也可以加入 token 候选，Transformer 的全局交互在 V2 已存在。

本反提案明确接受两项可能改变论文主线的代价。

1. 几何主输出是浮点场及其反向积分，**模型已经是共享 DLM 主干的 mixed-space 生成器，不再是纯 token DLM 采样器**。保留 tokenizer、token 辅助损失或预训练权重不能消除这一事实。
2. 并行几何反向过程中，现有元素程序 `P` 只能作为条件或 rank 特征。先计算全部更新、再按 `P` 的顺序写回，结果与写入顺序无关。要让 `P` 继续实际控制数值生成顺序，必须保留程序驱动的 token construction，或增加逐块重算；它们各有额外代价，不能藏进“共享几何头”这个名称。

因此，本文给同一组权重定义两个可比较的推理模式：**H-P** 从明确先验直接生成几何，方法最完整但减弱程序故事；**H-C** 先做现有程序构造，再显式加噪与连续输运，保留程序的实际路径但增加初始分布偏移和 NFE。本文不先指定 H-C 必要，也不为了保存故事给 `P` 人为增加未经验证的 species noise clock。

## 2. 证据边界：哪些结论已经排除，哪些仍未知

本报告建立在 [晶体 diffusion 机制审计](CRYSTAL_DIFFUSION_MECHANISMS.md)、[DLM 概率与 kernel 审计](DLM_PROBABILITY_AND_KERNEL_AUDIT.md)、[上游独立审计](C3FD_LLM_UPSTREAM_AUDIT.md)、[Transformer/GNN 补审](TRANSFORMER_AND_GNN_SUPPLEMENT.md) 之上，不重新把已被交叉审稿纠正的说法当动机。

- V2 的固定 old 观测可以对应合法 posterior `p(Y|U)=∏j p(Yj|U,Y<j)`；不存在“旧条件不随 scalar 更新，因此不是 Bayes 去噪”的普遍错误。working canvas 是这些已知量的确定性派生特征，可能改善有限网络，但不增加 Shannon 信息。
- V2 已有一跳周期 pair message、neighbor pooling、cell broadcast 和 site hidden 注入。不能把这些已有计算改名为首次接入 GNN；多跳或角度消息是额外的新假设。
- 无 SG 的 log-SPD 参数化本身没有从 DiffCSP++ 的消融得到匹配率提升证明；模板和 SG/Wyckoff 条件改变了其任务。SPD、周期性、合法组成都不保证热力学稳定或新颖性。
- 当前 Planner 的 `lattice_system`、`spacegroup_bucket`、`volume_per_atom_bin` 是三个独立 soft heads 的输出；species pointer 由 clean-CIF 接触启发式监督，未获 SUN 最优程序证据。全 MP20 的 DLM 训练不等于扩大了冻结上游的支持。
- 主审提供的 V2 完成结果是固定 256 请求：native Strict/Meta SUN `8/50`，共同额外 refiner 后 `16/113`；未达到同端 `26/128` 开发触发数。它不识别唯一败因，也不证明任何本候选可以补齐差额。本文没有独立调用评估模型。

外部依据是“这种训练对象已经有原始先例”，不是项目效果背书。[DiffCSP](https://arxiv.org/html/2309.04475v2) 给出晶格/torus 联合去噪；[MatterGen](https://www.nature.com/articles/s41586-025-08628-5) 给出混合变量生成与明确物理评测；[FlowMM](https://arxiv.org/html/2406.04713v1) 与 [FlowLLM](https://arxiv.org/html/2410.23405v1) 区分几何流与 LLM proposal 的角色。它们的网络、数据和评测均不同于本项目。原文及作者实现的具体版本、已读范围和边界已列入晶体机制审计，不引用二手摘要推断性能。

## 3. 与旧 PMTR15/16 的区别必须落实到计算图

已读冻结 [PMTR15](evidence_geometric_head/frozen/docs/teacher_feedback_unified_v1/15_PMTR_SCIENTIFIC_METHOD_AND_EXECUTION.md.txt) 与 [PMTR16](evidence_geometric_head/frozen/docs/teacher_feedback_unified_v1/16_PMTR_CODE_GROUNDED_ARCHITECTURE_AUDIT.md.txt)，并以实际训练代码裁决。

| 对象 | 已存在 PMTR 路径 | 本候选必须具备的实际区别 |
|---|---|---|
| 主干训练 | `freeze_spad_model` 冻结包括适配参数在内的整个模型；`frozen_spad_forward` 在 `no_grad` 下执行并 detach hidden/logits | frozen 8B base 继续冻结；V2 LoRA、state conditioner、相关适配器由几何损失获得梯度；hidden 不 detach |
| 几何输出 | 局部 manifold 增量经 token transport 变成 scalar categorical logit 修正 | 在明确 `z∈R⁶,F∈T^(3N)` 状态上预测连续 `v/u`；反向积分直接更新此状态 |
| 训练对象 | 局部修复/transport 与其辅助目标 | 已定义全时间 forward corruption 对应的密集 denoising regression |
| 可微路径 | 几何 head 依赖 frozen 表征；scalar renderer 仍是输出瓶颈 | 几何损失→linear heads→共享 Transformer/LoRA→共享条件模块；不经过 scalar codec |
| 谱分解风险 | 历史中出现更新前 nonfinite gradient，未完成正式科学终点 | 在已计算的 log-chart 中做 MSE；不把预测后的 exp/log/eigendecomposition 再放入主训练损失反传 |

源码锚点：[pmtr_training.py](evidence_geometric_head/frozen/src/crystal_dlm/pmtr_training.py.txt) 的 `freeze_spad_model` 在 150 行、冻结赋值在 156 行、`@torch.no_grad()` 在 302 行、hidden detach 在 335 行；[train_pmtr.py](evidence_geometric_head/frozen/src/scripts/train_pmtr.py.txt) 的冻结调用在 199 行。这里不是把所有 `.detach()` 判错：监督目标和随机污染后的输入是数据，完全可以不对其构造过程反传。不可断开的是 **head 所读 Transformer hidden 到 LoRA/conditioning 的训练路径**。

PMTR 的历史数值失败不能当成“连续几何方法已被 SUN 否证”；没有完成训练就没有这种科学结论。反过来，避免旧计算图错误也不等于证明新模型可学成。重复特征值下谱函数的数值梯度是应检查的风险，不将它武断宣判为历史唯一原因。

## 4. 联合分布、上游职责和程序的真实语义

令 `C=(N,A,chemical metadata)` 为冻结上游给出的 hard 化学条件，`A` 是按当前 Plan 分块排列的逐原子元素。`S` 是三个 soft 字段，`P` 是 frozen pointer 编译的元素/站点程序。新几何模型只学习

\[
p_\theta(z_0,F_0\mid C,S,P),\qquad
p(C,S,P,z_0,F_0)=q_{\rm up}(C,S,P)\,p_\theta(z_0,F_0\mid C,S,P).
\]

它不联合反传到 C³FD/Llama/Pointer，不生成新的元素或 N，不引入 SG/Wyckoff 硬约束，也不把 volume bin 变成密度拒绝阈值。训练仍使用全 MP20 的每个来源和冻结准备的 `S/P`，不筛到 soft 恰好匹配的子集。若 `S` 在给定 `C` 后与 source geometry 独立，最优几何模型可以忽略它的控制语义；增加 continuous head 不修复这一统计问题。

H-P 中 `P` 可进入 prompt/现有 program-rank 通道，影响有限模型表征；但所有几何通道在同一个时间步联合更新，不能称为“指针规划了几何的自回归生成顺序”。一个 forward 先产生全部 `ΔFi`、再按不同排列赋值是交换操作；[CPU 检查](evidence_geometric_head/geometric_head_checks.json) 的最大差异为零。

H-C 中先用现有六个晶格 scalar、species anchors、remaining sites、每站 X→Y→Z 的程序 construction 获得 `Y_D`。此时 `P` 的执行路径真实存在；连续阶段仍是全胞联合场。是否有必要保留这个 proposal 必须由同权重 H-P 对照决定。不能用“paper 需要程序”替代其净作用证据。

本报告不把 `P` 强行转换为额外的异质噪声时钟。若另一个候选使用 `σ_i(t;P)=σ_max t^{γ_i(P)}`，必须同时改写各站 score target、反向 drift `-σ'_i(t)u_i`、所有终端方差和有限停止点，并承认它改变了噪声分配。`γ_i` 是 rank 的某个单调函数并不自动得到接触传播或 SUN 优势。尤其 `σ_i(ε)` 在不同 rank 间差很多，codec 边界穿越概率也变了；程序优势可能只是某些站获得了更低终端噪声。仅仅让 `P` 进入公式，不构成程序物理性的证据。

## 5. 输入、共享架构与已有 V2 初始化

### 5.1 两种显式模式，避免在 old-token 契约中暗塞浮点状态

保留当前 `7+4N` body 的位置结构：一个 N、六个 lattice scalar 槽，以及每站 E/X/Y/Z 四槽。定义显式 task mode：

- **T 模式**：现有 token construction 的整数输入、legal support、CE 与输出接口。用于保留生成 proposal 的能力及 DLM 训练锚点。
- **G 模式**：N/E 槽照常填入，全部 lattice 与 X/Y/Z 数值槽为 MASK；真实输入另有明确的 `geometry_source=continuous_noisy_state` 和浮点 `(z_t,F_t,t)`。这些 MASK 是索引 scaffold，不表示几何未观测。G 模式已完整观测当前噪声几何，不能让旧 integer `old_*` 校验器把它认作缺失。

每一步由当前 `z_t` 解码 `L_t`，把 `L_t,F_t,A` 送入共享 state conditioner 和适用的几何 attention bias。现有 conditioner 已接受浮点 lattice/fractional 和 known/active/rank 字段，因此可复用其运算与权重；但必须新增显式模型 API 和 source 语义，不能悄悄覆盖原 V2 的离散 old 流程。G 模式所有真实站点与晶格均 known，所有几何槽均 active，padding 仍不参与消息。

**中间态不量化。** 若每一步先 `Q(z_t,F_t)` 再喂 old token 或再解码，网络读取的分布已经变成另一个 kernel，本文 score 目标与 ODE 不再成立。可以在 token 输出端做一次明示 roundtrip，不能在状态更新中隐藏它。

旧 V2 噪声元数据的三通道含义不能直接复用成新 VP/torus 时钟。新增 mode/time embedding，至少可由 `t` 的固定 Fourier 特征加训练 MLP 表示；新噪声尺度有不同单位。旧噪声字段在 G 模式按新 API 标为不适用，T 模式保持原语义。最小 G 路由将 legacy 九维 task features 和 V2 的 old-noise features 置零，省略 `repair_task_projection`，用新 time/task embedding 提供时钟；几何 known/active 状态仍据完整浮点几何计算。这个零向量是 G 新 API 的明确缺省，不伪装成现有 task id 或旧噪声尺度。旧 attention MLP 继续从其非时钟的几何输入工作，其中接收零特征的列没有直接 G 梯度。

### 5.2 最小共享 head 定义

令 `h∈R^H` 为最后一层共享 Transformer hidden，`H=4096`。

\[
\bar h_L={1\over6}\sum_{j\in\text{six lattice slots}}h_j,
\quad \hat v_L=W_L\bar h_L+b_L\in\mathbb R^6,
\quad \hat u_i=W_Fh_{E_i}+b_F\in\mathbb R^3.
\]

同一个 `W_F` 用于全部站；site hidden 通过全局 attention 与周期特征感知其他站、晶格和条件。一次 forward 输出全晶格和全部坐标，但不是六个晶格独立模型或每站独立网络。均值 lattice pooling 是一个可运行的最小读出选择，可能损失槽位特异性；若实际难学，槽位读出/attention pooling 属于待比较改变，不预先叠加复杂度。

示例 time 模块采用 64 维固定 Fourier 特征→128→4096，加一个 G-task 向量；`t` 已决定两种噪声尺度，另有 logSNR/logσ 也应作为显式派生量而非偷用旧单位。两个带 bias 的 linear heads 共 `6(H+1)+3(H+1)=36,873` 参数；示例 time MLP `64×128+128+128×4096+4096=536,704`，task 向量 4,096，总新增约 **577,673**。这是建议的规模核算，不是已实现模块。现有 shared conditioner 无需另外安装或先训练独立几何 backbone。

最后线性 heads 零初始化，G 路由保持可训练通路而不再乘一个也为零的门。此时首个 G backward 中输出层权重有梯度、其前 hidden 梯度可能为零，是零初始化的代数结果；之后应实测 LoRA/conditioner 获得非零且有限的梯度。不能把“head 参数在变”当成共享主干已学习。随机 time embedding 只用于 G 模式，T 模式的 step-0 输出应与载入 V2 一致。

### 5.3 不是随机 8B 从零训练，也不是 head-only 再包装

[PeriodicV2DLM 的真实类与加载代码](evidence_geometric_head/frozen/src/crystal_dlm/periodic_v2_initialization.py.txt) 继承现有 state-conditioned 模型；载入最终 V2 要恢复相同 base、LoRA、追加 token rows、state conditioner、attention/numeric adapters。不能误用 `initialize_periodic_v2_model` 的 fresh 初始化接口，它的 docstring 明确拒绝 adapted starting path；必须走完成 checkpoint 的加载与 provenance 校验。

V2 已记录可训练参数为 **36,993,232**：追加 token rows 20,324,352、LoRA 14,680,064，其余适配组件 1,988,816；约 8.035B base 参数冻结。因此候选是在已完成两 epoch 的晶体适配权重上改变任务与输入分布，新增约 0.58M 参数。冻结 base 不消除经过整套 Transformer 反传到 LoRA 的激活与时间开销。

T/G 共用 LoRA、conditioner 和 attention 通路。T 的 CE 训练 token rows；G 的 MSE 从 hidden 训练共同表征。上述 36.99M 是 **V2 的可训练模块库存**，不能原样声称它们都被 G loss 优化；两模式的精确路径见下一节。若新 API 实现最终仍把 hidden detach、仅训练新 heads，或者绕过 LLaDA 用独立几何网络承担全部生成，应认定没有实现本候选。

现有一跳消息的源码见 [periodic_state_conditioning.py](evidence_geometric_head/frozen/src/crystal_dlm/periodic_state_conditioning.py.txt)：118 行周期径向 pair、217 行 site 编码、232–234 行 pair message 和归一化邻居汇总、237–249 行 cell/site 更新。当前候选先复用它，不把 multi-hop/角度 GNN 当必选。邻居 image 范围、极端 cell 的距离/径向尺度和全噪声输入分布仍是实际风险；周期输入特征不自动给予整个带绝对位置与程序 rank 的 Transformer 平移/置换等变性。

### 5.4 活跃参数、输出分支与 DDP：不能把 dead logits 算作 G 学习

额外核对 [periodic_repair_model.py](evidence_geometric_head/frozen/src/crystal_dlm/periodic_repair_model.py.txt) 和 [periodic_repair_initialization.py](evidence_geometric_head/frozen/src/crystal_dlm/periodic_repair_initialization.py.txt)：`CrystalNumericAdapter` 在前者 177 行定义，248 行只给 `output.logits` 加值；`new_token_rows.output_delta` 在后者 63 行定义，72 行投影 hidden、100 行同样只加到 logits。它们位于 hidden 的下游，不会因为 G 读取同一个 hidden 而得到 G 梯度。`input_delta` 则在 92 行进入输入 embeddings，属于上游。

| 模块/参数 | T construction CE | 本文 G loss | 处理 |
|---|---|---|---|
| 14,680,064 LoRA 参数 | 活跃 | 活跃，head 打开后应检验 | 两模式共享训练，base 继续冻结 |
| state conditioner / geometry attention | 依 T 可见状态而异 | 从完整 noisy geometry 进入 hidden | 共享训练；并非每个 mask/常数输入列每步都有非零梯度 |
| `input_delta`，10,162,176 | 读到的 prompt/已填 body rows 活跃 | N/E/条件中实际出现的 rows 活跃；body numeric 永远 MASK | 整个 tensor 可在优化器中；记录按 token 家族的直接梯度覆盖 |
| `output_delta`，10,162,176 | logits→CE 路径活跃 | 不活跃 | T+G 保留为 T 专用；纯 G 配方应冻结并跳过其 forward |
| `numeric_adapter`，651,264 | logits→CE 路径活跃 | 不活跃 | 同上；不是新的 score head |
| `repair_task_projection`，36,864 | 活跃 | 按本方案 bypass | T 专用；G 用新增时钟/模式模块 |
| 新 v/u heads、time MLP、G-task 向量，577,673 | 不走该分支 | 活跃 | G 专用；T 输出初始化等价 |

`numeric_adapter` 的计数来自六个长度/角度投影各 18 个 feature、三个坐标投影各 17 个 feature，乘 H=4096；task projection 为 `9×4096`。因此本最小 G 分支能够处于损失图中的 **现有整 tensor 库存上界** 是 `36,993,232−10,162,176−651,264−36,864=26,142,928`，再加新参数为 `26,720,601`。它仍包含很多本 batch 未出现的 input rows/常数 feature 列；不是实测有效自由度。T+G 优化器的候选并集库存是 `37,570,905`，不是把两个分支重复相加。数值见 CPU accounting。

G 模式应在共享 Transformer final normalized hidden 后直接读出 v/u，并只返回该模式实际参与 loss 的输出；跳过 numeric adapter/output-row 路径。base 内的 vocabulary projection 是否能在保持 checkpoint 与 hidden 语义的情况下省去，需要核对实际 LLaDA forward 接口；当前 V2 wrapper 总会计算 logits，所以“调用旧 full forward 后丢弃 logits”没有省下这部分成本。不得 detach 共享 hidden 来省显存，也不能用 `0×unused_parameters` 的伪损失把 dead 模块伪装成几何优化。

全 numeric MASK 不意味着整张 `input_delta` 都死掉，N/E 和 prompt 新 tokens 仍可参与；未出现的 numeric rows 没有直接 G 梯度，也不意味着它们严格不变，既有 optimizer momentum/weight decay 可继续作用于整个 tensor。若仅训练纯 G，应按实际分支冻结 output-only/legacy-task 模块；不必为了零梯度行拆散全部 embedding table。

当前 [V2 trainer](evidence_geometric_head/frozen/src/scripts/train_periodic_dlm_v2.py.txt) 126 行已经使用 `find_unused_parameters=True`。T/G 交替也必须保留能处理实际未使用参数的 DDP 路径，不能未经检查改为静态所有参数必用；所有 rank 的任务调度、梯度累计和现有 activation checkpoint 路径要做真实 forward/backward 验收。G 的 returned outputs 不应包含只在 token 分支使用、却被本次 loss 忽略的 logits。最小验收包括 T/G 各自输出和 loss、预期 active groups 的有限梯度、零 head 首步与后续步区别、T 专用参数在 G 的预期无梯度、两模式连续更新后 DDP 无遗漏归约。这里要求的是一次明确计算图验收，不是新增大型防御框架或在当前审计阶段运行模型。

## 6. 晶格状态：明确 chart、尺度和适用边界

采用行晶格约定：`R=FL`，`G=LLᵀ`，`V=det L>0`。选单位 `ℓ₀=1 Å`，

\[
S={1\over2}\log(G/\ell_0^2)\in\operatorname{Sym}(3).
\]

使用 Frobenius 正交归一基 `E₁,…,E₅`（traceless）和 `E₆=I/√3`。例如两个 traceless 对角基及三个对称非对角基各按 Frobenius 范数归一。此处特意写明归一：不能混用 DiffCSP++ 论文的未归一符号与作者实现的归一化 basis，导致 `tr S`/体积通道差 `√3`。

\[
y_j=\langle S,E_j\rangle_F\ (j\le5),\qquad
y_6={\operatorname{tr}S-\log N\over\sqrt3}
={\log[V/(N\ell_0^3)]\over\sqrt3}.
\]

只在训练 split 上计算各通道均值 `μ_j` 与正标准差 `d_j`，保存到 checkpoint；极小标准差的 numerical floor 必须记录，不能用验证数据估计。令 `z_j=(y_j-μ_j)/d_j`。第六通道标准化 volume per atom，是 N 条件下的体积尺度处理，不是对 soft volume bin 的硬拟合。

解码为

\[
k_j=\mu_j+d_jz_j\ (j\le5),\quad
k_6=\mu_6+d_6z_6+{\log N\over\sqrt3},\quad
G=\ell_0^2\exp\left(2\sum_{j=1}^6k_jE_j\right),\quad
L=\operatorname{chol}_{\rm lower}(G).
\]

数学上每个有限 z 对应 SPD metric 与正体积；实际极端 z 仍可导致 exp 溢出、病态 Cholesky 或异常密度。不得因为“SPD 对所有 z 成立”而声称浮点实现不需数值检查。若采样失败，应计入全请求分母，不通过隐藏 clipping、投影、重抽或 hard density rejection 修改结果。

这个 chart 消除了晶格在 Cartesian 空间的全局转动表示冗余，但 **没有**解决 `GL(3,Z)` 等价基、origin shift、相同物种置换或空间群商空间；需与来源晶胞约定一致。score 是标准化 z 的 Euclidean density score，不是仿射不变 SPD Brownian score，也不是能量梯度。把输出 Cholesky cell 与原 F 配对保留 fractional displacement 的度量 `ΔF G ΔFᵀ`；任意重新约简晶胞则必须同步变换 F。

## 7. Forward corruption、正确监督与联合 score

给定已对齐 source `(z₀,F₀)`，取 `t∈[0,1]`，两类噪声条件独立：

\[
\alpha(t)=\cos(\pi t/2),\qquad
\sigma_L(t)=\sin(\pi t/2),\qquad
z_t=\alpha(t)z_0+\sigma_L(t)\epsilon_L,\quad \epsilon_L\sim N(0,I_6),
\]

\[
\sigma_F(t)=t\sigma_{\max},\quad\sigma_{\max}=1,
\qquad F_t=\operatorname{wrap}(F_0+\sigma_F(t)\epsilon_F),\quad
\epsilon_F\sim N(0,I_{3N}).
\]

`σ_F` 单位是 fractional coordinate；它在 Cartesian 的协方差是 `σ_F² L_tᵀL_t`，不是各向同性 Å 扰动。这里没有 SGWyckoff 投影，没有未经推导的最小像 score 替换，也没有额外 COM 零和投影。

### 7.1 晶格 v target

\[
v_L^*=\alpha\epsilon_L-\sigma_Lz_0,
\qquad \mathcal L_L={1\over6}\|\hat v_L-v_L^*\|^2.
\]

对 `0<t<1`，可换算为

\[
\widehat z_0=\alpha z_t-\sigma_L\hat v_L,
\quad \widehat\epsilon_L=\sigma_Lz_t+\alpha\hat v_L,
\quad \widehat s_z=-z_t-{\alpha\over\sigma_L}\hat v_L.
\]

模型最优回归量是给定 **全体** `(z_t,F_t,C,S,P,t)` 的条件均值。此时 `ŝ_z` 对应联合边缘 `p_t(z_t,F_t|C,S,P)` 的 z 分量 score。不是把每个训练样本的 `z₀` 在部署时当已知，也不是六维单独建模。MSE 与适当加权的 ε/score/x₀ 学习具有同样 population 条件最优；并不意味着它们在有限样本、有限网络、不同时间权重下有相同优化速度。

选择 v 参数化是为了端点正规性：`t=1` 时 `α=0,σ_L=1`，先验恰好 `z₁=ε_L`；从 ε 公式除以 α 求 x₀ 会失效，但 v 输出和下述概率流 drift 仍可有有限极限。零 v head 的参考 drift 为零，不会在未训练时产生 `1/α` 爆炸；它也当然不是合格晶体生成器。

### 7.2 坐标必须使用 wrapped-normal score

对单坐标 `δ=f_t-f₀∈(-1,1)`，

\[
q_\sigma(f_t|f_0)=\sum_{n\in\mathbb Z}{1\over\sqrt{2\pi}\sigma}
\exp[-(\delta+n)^2/(2\sigma^2)],\quad
w_n={e^{-(\delta+n)^2/(2\sigma^2)}\over\sum_m e^{-(\delta+m)^2/(2\sigma^2)}},
\]

\[
u_F^*=\sigma_F\nabla_{f_t}\log q_{\sigma_F}(f_t|f_0)
=-\sum_nw_n{\delta+n\over\sigma_F},\qquad
\mathcal L_F={1\over3N}\|\hat u_F-u_F^*\|^2.
\]

三轴和不同站的 forward density 乘积使目标按轴计算即可；共享网络仍读取全体几何。训练目标可在 float64 用 logsumexp 计算；`σ≤1` 时例如 `n=-8…8` 的截断尾项极小，但实施时应以明确 tail bound 验证，而非把任意固定 window 称为 exact。导数方向和整数像符号必须按同一 δ 约定。

最初 sampled `-ε_F` 与该 wrapped 观测的条件 score 不是逐样本相等，但 **它仍可作为无偏的随机回归 target**：`E[-ε_F|F_t,F₀]=u_F*`。因此不能因 wrap 就断言 ε 回归数学错误。此处选择显式求像和的条件均值 target，减少多像潜变量带来的方差；接近均匀先验时 wrapped score 趋近零，unwrapped ε 仍可能很大。另一个常见替代——最短 fractional difference 除以方差——则一般不等于完整像和 score，不能不作小噪声近似声明就使用。

### 7.3 Joint learning 和时间分配

\[
\mathcal L_G={1\over2}\mathcal L_L+{1\over2}\mathcal L_F.
\]

每例将晶格/坐标两类各占一半，避免大 N 仅因通道更多而取得更大样本权重。这是明确选择的 denoising regression risk；没有声称等于未加权数据 log-likelihood/ELBO，也没有附加 force、能量、SG 或 SUN 奖励。

拟议训练 `t` 在 `[ε,1]` 上 log-uniform，`ε=0.002`，每个几何 view 一次时间和一次全胞噪声。它具有全区间正密度并增加低噪声覆盖；有限训练可能仍不足，尤其最高时间附近权重较低。全时间目标的 population 最优保持相同，但这不是在一至两 epoch 内学好所有噪声层的保证。`t=1` 端点由连续极限定义，有限采样通常不恰好命中，应专门检查端点邻域预测。

forward 在给定 source 后分解，不代表反向场在晶格与坐标、原子之间独立。head 的输入来自同一组全局 hidden；所有损失共同训练该联合条件场。反之，给多个独立模型分别建目标再称为“joint diffusion”不实现本定义。

## 8. Reverse process、先验和计数明确的采样器

### 8.1 概率流 ODE 的具体形式

与上述 VP 边缘相配的 `β(t)=π tan(πt/2)` 在 `t=1` 发散，所以不能把标准 VP SDE 的 β 表达式直接在该端点数值求值。采用有正规 v 极限的概率流：

\[
{dz\over dt}={\pi\over2}\hat v_L(z,F,C,S,P,t),\qquad
{dF\over dt}=-\sigma'_{F}(t)\hat u_F(z,F,C,S,P,t)
=-\sigma_{\max}\hat u_F.
\]

前式来自 `-βz/2-βs_z/2` 的代入与消项；后式来自 torus VE forward 的 `g_F²=2σσ'` 与 probability-flow drift `-g_F²s_F/2`。当 score 正确、初始分布正确且连续积分满足条件时，它保持所需边缘演化；本报告实际提出有限步近似，没有收敛或 SUN 保证。

对反向区间 `t→t-Δt`，Euler 更新是

\[
z\leftarrow z-{\pi\over2}\Delta t\hat v_L,\qquad
F\leftarrow\operatorname{wrap}(F+\sigma_{\max}\Delta t\hat u_F).
\]

坐标正号来自向较小 t 积分，不能因为叫“去噪”凭直觉另加负号。可选 Heun 必须在预测的新几何重新 forward，不是对同一旧输出做两次代数操作。

### 8.2 H-P 的初始先验

取 `z₁∼N(0,I₆)`，`F₁∼Uniform(T^(3N))`，相互独立，N/A 固定。晶格 prior 在本 VP 定义下精确；坐标 uniform 是 `σ_max=1` 的 wrapped-normal terminal mixture 的显式近似。

利用 torus Fourier 系数，任一 1D wrapped normal 相对 uniform 的 TV 上界可取

\[
\sum_{k\ge1}\exp(-2\pi^2k^2\sigma_{\max}^2)=2.675287991\times10^{-9}.
\]

产品耦合与混合的凸性给出 N≤20 时至多 `60` 倍，即 `1.605172795×10⁻⁷`。这是理想 prior 的近似界，不是网络误差、采样误差、最终 SUN 差或总输出 TV 的实测结论。网络误差远可能支配它。

### 8.3 H-C 的初始分布不能冒充数据正向 marginal

先用 T 模式 construction 得 `Y_D`，经相同 cell chart 解码 `z_D,F_D`。显式选一个预先声明的 `τ∈(ε,1)`，按 **同一个** `q_τ` 真正加噪：

\[
z_\tau=\alpha(\tau)z_D+\sigma_L(\tau)\epsilon_L,\quad
F_\tau=\operatorname{wrap}(F_D+\sigma_F(\tau)\epsilon_F),
\]

然后上述 ODE 从 τ 反向到 ε。其起点分布是 `p_construct q_τ`，通常不等于 `p_data q_τ`；它不是理所当然的 Bayes 修复 posterior。ODE 是确定性输运，不能直接套用“精确 corrupt→Bayes posterior sample”产生的 Gibbs/detailed-balance 证明。若 constructor 恰有数据分布且 score 精确，起点边缘才能与训练一致；现实没有这个前提。

`τ=0.1` 可作为一个完整写下公式的初始设计例，但目前没有证据选定它。不能把外部 model494 的 `tau800`、mask ratio 或旧 V2 噪声编号直接代入连续时间。若必须事后反复用开发 SUN 搜 τ 才能使 H-C 优于 H-P，应把选择成本和选择偏差计入，不能宣称是现成的理论默认。

### 8.4 本草案的具体计算预算

为了可核算，指定首个数学候选用 `K=32` 个 Euler 区间，时间网格在低 t 附近更密：`t_j=ε+(t_start-ε)(j/K)²`，按 `j=K…0` 反向。输出 `t=ε` 的状态，**没有额外 terminal denoise forward**。H-P 为 32 NFE；同样 32 区间 Heun 为 64 NFE。H-C 的 `t_start=τ`，加上 token construction 的 NFE。

`ε=0.002` 意味着输出仍是有限噪声端点。它不是精确 t=0 样本；对靠近旧 codec 边界的坐标，哪怕很小的残余噪声也会影响量化结果。若加 x₀ 修正、减小 ε、增加步数、加入 predictor-corrector 或随机反向 SDE，应另列定义和实际 NFE，不能继续沿用 32。

32 NFE 是预算提案，未被该 backbone 的质量证据支持。若 32/64 步不够而需要 100–1000 次 8B forward，本候选可能失去近程计算理由；不能引用小型几何网络的低成本掩盖这个风险。

## 9. 一至两 epoch 内的监督与硬件可行性

### 9.1 训练 views 和比较对象

从完成 V2 checkpoint 继续：每个 source 每 epoch 一个 T construction view（现有 75% prefix / 25% dense）加一个 G full-noise view，替换旧 repair view。G 不先增加第二次 LLaDA forward 来生成 teacher input；噪声目标都从 source 算出。这样两 epoch 仍是每 source 四个 forward examples，训练 T/G 各占一半。每例使用对应 task risk；几何 view 内再按上一节平衡晶格和坐标。

这不是“所有原任务完全保留”的方案：旧 repair CE 被连续去噪替换，T 只锚定 construction。若需要另留 repair CE，也应重新分配 view 或承认增加预算。全部四个 views 改成 G 可以再提高几何曝光，但会丢失 token construction 训练锚点，不能算作同一 primary design。

### 9.2 每个 forward 的实际 target 数量

令 `d=6+3N`。当前 V2 每 view 以 0.75 概率只监督一个 prefix scalar；dense 以独立随机 mask 监督，mask 概率均值为 0.55。因此期望数值 labels 是

\[
m_{V2}=0.75+0.25(0.55d)=0.75+0.1375d.
\]

两 epoch 两 view/source 是 `4m_V2=3+0.55d`。新 T/G 配方两 epoch 为 `2m_V2` 个 categorical labels，加 **`2d` 个几何标量回归 targets**。公式是在记录为无支持冲突的理想实际路径上做期望 accounting；随机 draw、padding、空 dense mask 的实现计数应保留。逆概率 mask 权重使 V2 dense risk 的期望正确，不让一次前向自动拥有未被监督位置的梯度样本。

| N | d | V2 labels / forward | V2 labels / source / 两 epoch | 新 T categorical labels / source | 新 G targets / source |
|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 1.9875 | 7.95 | 3.975 | 18 |
| 10 | 36 | 5.7 | 22.8 | 11.4 | 72 |
| 20 | 66 | 9.825 | 39.3 | 19.65 | 132 |

同一个被替换 repair view 的 dense 连续 target 数量，分别为原期望 scalar 数量的 `4.53/6.32/6.72` 倍。这不是信息、Fisher 信息、梯度有效样本或学习速度的倍数；同胞多通道高度相关，MSE 的难度/权重也不同。T 的原 CE 不能与 G 的 MSE 简单相加后称为 likelihood 总量。

主审已落盘的 MP20 train atom histogram 包含 `27,136` sources、`282,847` atoms，平均 `N=10.4233`；全 source 一次完整几何 view 有 `1,011,357` 通道。由此两 epoch：

| 计数 | V2 | 本候选 T+G |
|---|---:|---:|
| forward examples | 108,544 | 108,544 |
| 原式 categorical numeric targets，期望 | 637,654.35 | 318,827.175 |
| 连续几何 targets | 0 | 2,022,714 |
| 几何完整视图 | 0（该意义下） | 54,272 |
| global batch 24 的更新量 | 4,524（已执行） | 约 4,524，须沿用尾 batch/padding 计数 |

“几何完整视图为 0”只描述本节的 full continuous target，不抹去 V2 已存在的 dense branch 和几何输入。若四个 views 都为 G，通道总数为 `4,045,428`；这是另一个任务分配，不能凭此夸大 primary 配方。

数据来源和所有算式见 [geometric_head_checks.py](evidence_geometric_head/geometric_head_checks.py) 与 [数值结果](evidence_geometric_head/geometric_head_checks.json)。脚本只检查代数和预算，没有实例化模型。

### 9.3 4–6 A800 的边界

在 global batch 24 下，4 卡×microbatch 1×accumulation 6，或 6 卡×microbatch 1×accumulation 4，是与总样本计数相容的候选配置，不是已经 profile 通过的显存结论。应该保留现有能够运行 V2 的冻结 base、精度、LoRA 和激活 checkpoint 策略；新增不足 0.6M head/time 参数不是主要容量成本。

G 模式可能避免完整 vocabulary output projection/CE 的一部分计算，但它仍需同一 8B Transformer 前向和对 LoRA 的反向；保存 hidden、continuous feature 构造、极端格子下几何计算也可能增加负担。相同 forward example 数不等于相同秒数、显存或 FLOPs，不能据此虚报 wall-clock 加速。全噪声状态也比 V2 局部 old-state 输入分布更陌生。

一至两 epoch 的有利条件是已适配 V2 表征与更密集 G targets；不利条件是全时间 joint field、新模式、先验端点和连续读出均要学，且 task interference 可能损伤 T proposal。没有相应实测就只能说“参数和 forward 预算有可比性”，不能说“必然学得会”。

### 9.4 推理 NFE 不能只报连续阶段

沿用当前 construction 和单轮 full-cell scalar repair，每轮需 `d=6+3N` 个 Transformer forwards；V2 合计 `2d`。H-P 不先运行 token construction；H-C 加上 d。

| N | V2 construction+repair | H-P Euler32 | H-P Heun32区间 | H-C construction+Euler32 | H-C construction+Heun32区间 |
|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 32 | 64 | 41 | 73 |
| 10 | 72 | 32 | 64 | 68 | 100 |
| 20 | 132 | 32 | 64 | 98 | 130 |

这是假定输出 complete、每 scalar 一次 forward、无拒绝/重试的可比较计数；真实实现若多轮或早停须按实际 trace。上游 Planner 和共同 model494 的调用不在几何 NFE 列内，必须在整条流水线预算中同样单列。H-P 对小 N 可能更贵；H-C 在平均 N 下只有很有限的 Euler NFE 优势，Heun 可能更贵。减少输出 logits 不免除 Transformer NFE。

## 10. 连续 CIF 标签、站点对齐和最终 codec

### 10.1 现有 SFT body 不足以宣称 sub-bin 监督

已读 builder 路径仍携带原 quantized `clean_answer/source_answer`，没有在当前 SFT 记录中凭空恢复原始 CIF 连续参数。`canonicalize_dynamic_answer_to_plan` 只将完整 species-coordinate records 按 Plan 的元素顺序移动、同元素内部保持 source order；它显式校验 lattice 和 record multiset 不变。**这不是连续 CIF label 的来源，也不是重新求物理等价晶胞的算法。**

实施前有两个诚实选项，不能混用其精度主张。

1. 按 `source_row_idx + split` 回连原始 MP20 CIF，复现当初用于生成 body 的 cell convention、origin、站点次序及 tokenizer 量化；保存 exact site permutation。对齐后的连续 CIF 经旧 codec `Q` 应复现当前 canonical clean body；否则定位 source mapping/convention 差异，不按相同组成随便找一份 CIF。原坐标只能在该对应关系确定后进入 G loss。
2. 暂时从原 body 解码到 float，作为连续噪声的 clean target。这保持当前 source 监督与全数据覆盖，但标签本身落在旧网格上；可以检验 state/noise/update 机制，不能声称得到了原子位置的新增 sub-bin 信息。

按 source split join 不等于以 formula 匹配：同一组成可能有多个结构，cell reduction 也会改变 F。拒绝丢掉难对齐样本来创造更干净的训练集合；无法对齐时应保留来源并显式降级为 decoded label、或在方案选择中承认任务尚未完成。混合 label 精度还需要标记和分层诊断，不能当统一原始 CIF 监督。

本候选没有提议新 force/energy teacher 或额外 DFT；只需重新取得已有训练来源的连续数据。canonical 原子顺序不能从 noisy geometry 每步重排，否则噪声 target 与 head 的站点索引不再对应；相同元素也不能在 loss 中无说明地做 Hungarian 最优匹配后继续声称原 density score 目标。

### 10.2 Native endpoint 必须事先定清

连续 decoder 自然输出 `L,F,A` 并写 CIF。若将其定义为新模型 native 输出，应以这些 float CIF 做共同独立物理评估，再对同一组样本调用相同额外 refiner；不能先 quantize 后与 float 结果择优。

仍可保留完整 `7+4N` 兼容导出 `Q(L,F,A)`。这是连续分布的 pushforward，输出精度和合法区间由 codec 决定，**不等于保留了原 categorical 生成概率、逐 scalar log-prob 或 token diffusion 数学对象**。某些 SPD metric 会超出 codec 的 length/angle 有限支持；转换失败必须可见，不能隐藏裁剪。

若项目强制 native 必须是原 codec roundtrip，则主评估应使用 `decode(Q(L,F,A))`，连续 CIF 只能另列诊断。此时连续模型可能仍因更新轨迹而获益，但最终 sub-bin 优势被量化丢掉；不能将另一 endpoint 的 SUN 写到 native 名下。建议论证阶段明确把“mixed native CIF”和“quantized roundtrip 诊断”分开，最终由论文任务定义选择一个主终点。

## 11. 与 scalar full-cell 和离散 Q 候选的公平比较

| 维度 | 现有 scalar posterior / full-cell 修复 | 明确离散几何 Q 候选 | 本候选共享连续 heads |
|---|---|---|---|
| 主状态 | 原 finite codec，支持与输出语义已成熟 | finite state，需给坐标及晶格一致 kernel/时间表 | `z∈R⁶,F∈T^(3N)`，中间态 full float |
| 学习目标 | exact conditional CE；稀疏/密集配比可改 | categorical reverse/clean posterior，可全槽密集 | 全几何 v/torus score MSE |
| 先验与反向 | 固定 old posterior 可成立；多步状态分布仍需定义 | 若 Q 与 reverse 正确可以完整定义 | 本文先验明确；ODE/有限步误差与 H-C 起点偏移明确 |
| 主干复用 | 最大 | 取决于新 tokenization/参数化 | V2 LoRA/conditioner 复用，新增 G API/time/head |
| 程序 P | 实际 scalar 顺序 | 取决于 AR reverse 或 block schedule | H-P 为条件/rank；H-C 的前置 construction 为真实顺序 |
| 精度瓶颈 | 每步 codec | 由选定 finite state 决定 | native float；兼容导出仍量化 |
| 新增难点 | 密集训练与部署顺序差异、支持/状态覆盖 | SPD 晶格的有限表示、支持、joint reverse与终端分布 | 原 CIF 对齐、joint field可学性、全噪声输入、finite-step积分 |

不能用“离散 Q 必须存 `100^(3N)` 大表”攻击另一候选。逐坐标 circulant wrapped Gaussian Q 可以因子化、用一维矩阵或 Fourier 结构实现，绝非必然指数存储；Transformer 可让 reverse 条件依赖全晶体。困难主要是六维晶格的合法表示/边界、与数据 codec 的关系、训练 posterior 和实际解码的衔接，而不是一个虚构的必然大联合矩阵。

连续候选也没有消灭这些统计问题：log-chart 和 noise scale 是表示选择，站点/cell 规范是数据选择，ODE 数值方法是输出分布选择。它只是以已有成熟几何扩散对象避免先创建全新晶格 Q。若纯 token 是论文不可改变的核心研究对象，经过严格定义的 dense token/Q 候选可能更符合问题，即使连续方案算起来更方便。

**最低公平归因条件：** 若以后比较的核心声明是“连续几何优于 scalar”，必须纳入同一 checkpoint、同一 source 曝光与尽量匹配的 full-geometry dense token control；否则“更多 targets/forward”与“连续表示/反向 kernel”完全混在一起。它不要求本审计现在增加训练，而是规定将来可写的因果主张。P 的必要性还需 H-P/H-C 或适当 canonical/learned P 比较，S 的语义需单独条件检验；不能把同一次总体 SUN 改善拆成三个无依据的贡献。

## 12. 主要反对理由与可否决条件

1. **一至两 epoch 可能不够。** 新头虽然小，主干见到完整高噪声浮点状态的模式仍新；密集 labels 不能弥补先验端点和联动结构学不会。如果只能靠大量额外训练或新几何 backbone 才可用，就不再满足本项目的近程约束。
2. **DLM 的额外价值未必被证明。** 共享 LoRA 有梯度只是实现条件，不证明语言/掩码预训练相对适当几何 Transformer 的优势。如果最终主要是 conditioner+小头起作用，论文应降低“语言先验驱动几何”的表述。
3. **程序故事可能不成立。** H-P 更直接、更省 NFE，且 `P` 条件可被忽略；H-C 增加 proposal 偏差。如果删去实际程序并不损失 SUN，应接受方法简化，不增 clock、gate、teacher 来人为保留贡献。
4. **原始 CIF 对齐可能超过看似小改动的成本。** 旧 codec labels 容易接入，但不能证明精度收益；原始 cell/site 对齐需要逐源证据。若只有少数可对齐样本，不能换成小而优质的数据集后继续称同预算全 MP20 比较。
5. **几何度量和对称性仍有限。** fractional isotropic noise 是合法 torus kernel，却不是 Cartesian isotropic perturbation；log-chart 不是唯一晶格商空间；全模型没有自动 SE(3)/permutation/SG theorem。
6. **数值成本与尾部样本可能主导。** SPD 不保证良好条件数，32 ODE 步不保证晶体。若网络误差导致极端 lattice、wrap高频场或需要大量 corrector，算力优势会消失。失败样本不能通过额外过滤补成好看 SUN。
7. **共同 refiner 可能抹去差异。** native 获益可以是真实内部几何改善，但若共同 refiner 后完全相同，贡献叙述需限于 native；不能把共同 refiner 的收益归给新 head。所有 SUN 都按原请求分母和相同新颖性库、hull 门槛报告。

## 13. 两轮独立审查所需的具体攻击面

本报告不请求启动训练，也不把数学自检当成模型试验。主审可将以下内容分别交给不同 reviewer，先形成书面修改记录再选择候选。

**第一轮：概率、表示与接口。** 检查 v↔ε/x₀/score 换算、torus 像和目标、VP 端点 β 发散但 v drift 正规的论证、ODE 反向符号、uniform prior TV 界、H-C 非数据 marginal/非 Bayes posterior 的区分；检查 row/column lattice 约定、basis 归一、`log N` 体积项、CIF site/cell 对齐、G full-float 输入有无被中间 codec 改写；确认实际梯度路径没有 PMTR 式断开。

**第二轮：物理、预算与故事归因。** 检查 1–2 epoch 密集目标是否足以支持任何可学性判断、8B NFE 与硬件边界、time sampling 和有限 ε 的量化影响、极端 cell/邻居特征的输入分布；攻击 H-P 中 P 的必要性以及 H-C 的 proposal/τ 假设；判断和 dense token/Q 对照是否公平，native endpoint 是否变更了研究任务。它应允许否决整个候选，而不是只修公式后默认为可训练。

若两轮后仍成立，结论最多是“可以在明确预算与冻结指标下实现一个可否证的 comparator”。它不能直接升级为“V3 已定”或“必能达到双 SUN 门槛”。若否决原因在纯 token 研究对象、P 的真实必要性、原 CIF 接口或计算收益不足，应把这个否决保留在最终方法选择记录。

## 14. 已完成检查与仍未做的事

[CPU 数学结果](evidence_geometric_head/geometric_head_checks.json) 核对了 v 反变换、score 代数、零 head 端点、先验界、预计算更新的写入顺序交换性和上述监督/NFE 表。最大 v 反变换误差为浮点舍入量级；它只证明脚本里的等式，没有证明连续神经场可学、ODE 32 步足够或 SUN 提升。

源码冻结包含旧 PMTR、当前 PeriodicV2DLM 初始化/加载、state conditioner、V2 model/training/objective。没有做模型 forward、梯度 smoke、CIF 全量回连、真实 runtime/profile、样本生成、CHGNet/model494 调用或 SUN 评估。不存在可交付 checkpoint 或已通过训练方案。

本文最终立场是：**可保留共享连续几何头作为独立候选，但应以完整 mixed-space 定义、真实共享梯度和可比较预算来审查；不能为了维护 DLM/程序名称压缩其代价，也不能为了反对 scalar 而歪曲 fixed-old posterior 或离散 Q 的可行性。**
