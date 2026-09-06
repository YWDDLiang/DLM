# V3 第一轮独立物理审稿

2026-09-06。审查对象为三候选稿与独立几何头反提案，**不是此前 V2 报告的第二轮审查**。本轮只读代码、已保存评测及原始资料，运行小型 NumPy/解析反例；没有模型 forward、SSH、GPU、MLIP、弛豫、安装或训练。

**结论：建议主审以 H-P 的统一噪声时钟、v/torus 联合头和明确 ODE 起草一份最小 mixed-space 规格，再进入第二轮；不把 C 的 species clock 或 H-C proposal 设为必需。** A 是条件语义候选，B 是零训练的执行比较，都尚未给出 SUN 收益。C/H 的数学对象大体可成立，真正需要先解决的是数据精度/身份、采样终点、共享梯度和计费对象；小预算可学性与 SUN 是之后的实证问题，不应伪装成已解决，也不应变成永远不能开始的抽象门槛。

## 1. 被审版本与已确认的部分

| 对象 | 阅读版本 SHA256 | 本轮范围 |
|---|---|---|
| [V3_DLM_CANDIDATES_AND_ATTACKS.md](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/V3_DLM_CANDIDATES_AND_ATTACKS.md) | `bd98573df716b73c867e5624721e2babf982dcaa58b247e868eec9b260a3d366` | A、B、带 program clock 的 C |
| [V3_GEOMETRIC_HEAD_COUNTERPROPOSAL.md](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/V3_GEOMETRIC_HEAD_COUNTERPROPOSAL.md) | `d229a661585005a484387ec8263db0c88c6396b5f37b862f8273ce616cecb7ac` | H-P/H-C、T/G、v/ODE、活参数补充 |

下文简称 C 为第一稿的 `epsilon + DDPM/wrapped Euler + rank clock`，H 为第二稿的 `v + probability-flow ODE + uniform clock`。不能把针对 C 的反例自动套给 H。第二稿现有 §5.4 已列活参数、原 `find_unused_parameters=True` 和模式差别，本审稿接受这些修订，不再把已经澄清的问题当作新发现。

确认：row-lattice 的 `G=LLᵀ`、half-log metric 与 Cholesky 逆变换可构成明确 chart；fractional isotropic torus noise 是合法选择；正确 wrapped score 不等于 nearest-image 位移除方差；H 的 `dz/dt=(π/2)v`、`dF/dt=−σ_max u` 及向较小 t 积分的符号成立。固定 old 对离散 posterior 合法，现有 conditioner 已有几何 hidden/global 传播；本轮不重启“没有 GNN”或“old 每个 scalar 必须覆盖”的旧推断。

## 2. 完成的 V2 结果约束下一步，但没有识别唯一根因

| 同一 development256 请求 | Strict / Meta SUN | 完整 verified SUN | verified 终态数 | invalid-terminal 数 |
|---|---:|---:|---:|---:|
| construction | 12 / 44 | 4 / 21 | 68 | 79 |
| 一次 full-cell repair 后 | 8 / 50 | 2 / 18 | 70 | 83 |
| repair 后固定 tau800 | 16 / 113 | 另按该端原汇总读取 | 109 | 不由前两行推断 |

construction→repair 的 Strict 命中保留4、失去8、新增4；Meta 保留18、失去26、新增32。共同 verified 只有33个，35个失去、37个新增。raw E 均值下降 `.17806 eV/atom`，中位变化却上升 `.19127`；117个下降、137个上升。**保留 construction 是一个现成、少 d 次 forward 的真实竞争方案；但这些开发点估计也不足以宣布所有 repair 都无效。** 来源：[完整 construction/repair 配对](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence/V2_CONSTRUCTION_AND_REPAIR_39998.json)、[端点评测汇总](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence/V2_EVALUATION_SUMMARIES_39998.json)。

最终训练108,544个真实状态、4,524次更新，corruption fallback为0，支持冲突文件为空。不能继续把“选择性 prefix 截断大量发生”当作本次低 SUN 的已证根因；它只是以前的原则反例。另一方面，0冲突、非零梯度、准确计数也没有证明模型学成了稳定几何。[最终训练证据](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence/V2_FINAL_TRAIN_AND_METADATA.json)

79/83个 invalid-terminal 是全体终态计数，不能直接从 SUN 减掉这些数来计算几何通过 SUN，仍需逐例交叉事件。当前 headline、终态几何通过、完整 verified 三层必须保留。更换共同 R 或验收阈值改变标签/后处理，不能作为任何候选的模型收益。[物理口径边界](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/ROUND2_PHYSICS_ON_DLM_AND_CLAIMS.md)

## 3. C 与 H 是不同的完整候选，不能只比较“头很小”

| 项目 | C | H 主方案 |
|---|---|---|
| 初始化 | 原始 LLaDA，新适配初始化 | 完成两 epoch 的 V2 checkpoint 续训 |
| 再训练对象 | 两个 view 均为全几何 score/epsilon | 一个 T construction CE、一个 G 全几何 v/u |
| 晶格反向 | `hat z0` 代入 DDPM mean，方差取 `beta_tilde` | v 参数化 probability-flow ODE |
| 坐标反向 | rank相关 σ，有限 wrapped Euler，末步仍加噪声 | 统一 `σ_F=t`，ODE 到 `ε=.002` |
| 程序 P | 新增幅度0.5的 species noise clock | H-P为条件/rank；H-C另有真实 token construction |
| 几何 NFE | 64 | H-P Euler32；H-C额外加d；Heun另计 |
| 输出主对象 | 原 codec 结果为正式对照对象 | 建议 float native，并要求事先区分 Q roundtrip |

H 的“一至两 epoch”是 **额外**预算，总训练曝光已经有 V2 的两 epoch；C 的一至两 epoch 是从原始适配初始化开始。H 的暖启动、T锚点、时间采样、solver、终点精度同时不同，不能把未来差异全归给 v 参数化或连续头。若只比较一个实用系统，完整登记这些差别即可；若要声称哪一因素起作用，则需要相应匹配对照。

## 4. 连续源和 chart：可消除的定义阻断

### 4.1 `source_row_idx` 不是未经核验的原 CSV 行号

现有 R5 builder 用 `Structure.from_str(cif)` 后立即 `structure_to_dynamic_answer` 再 parse，已经量化；失败行会被跳过并写另一个文件。后续 C³FD builder 还存在 `allow_source_ordinal_index` 分支，能按中间 SFT 文件 ordinal 补编号。stable site canonicalization 保留同元素 source 顺序，返回 permutation，但当前转换记录主要保存 `site_order`/是否变化，不等于已经保存了原 CIF→最终站点的完整证明。[原 CIF 读取与量化](D:/codex_work/ai4s/DLM_periodic_self_repair/src/scripts/build_r5_exact_length_sft_data.py:58)、[ordinal 分支](D:/codex_work/ai4s/DLM_periodic_self_repair/scripts/build_c3fd_native_sft_data.py:935)、[站点排列](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/canonical_site_order.py:27)

反例很小：CSV为A/B/C，B解析失败，中间表为A/C；若之后编号0/1，把1直接连接CSV行1就取到了B。这证明仅凭编号不足，**不是声称实际 MP20 已错配**；若当时失败确为0且文件顺序/哈希一致，这个疑点即可关闭。

原连续监督需要一份逐源记录：原 split/文件 SHA、原行或稳定 material id、解析版本、连续 cell/F、固定 site permutation，以及 `Q(对齐连续结构)=当前 canonical answer`。不按 formula 随意匹配，也不把 N 改变后的 primitive/conventional cell 当作同一来源。所有 train27136/val9047来源应有明确处置。

这只阻断“已获得原连续 CIF/sub-bin监督”的主张，不必无限期阻断几何头接口。H明确允许**完整 decoded-token targets**作为另一种标签精度；这是可行的机制比较，只需放弃原连续精度主张。两种标签不能在不标记的情况下混合，也不能筛出容易对齐的子集后继续称同数据量。

### 4.2 SPD、旋转规范与等价晶胞是三个问题

固定 row basis 时 `S=.5 log(LLᵀ)` 消除 Cartesian 全局旋转冗余，`chol(exp(2S))` 与原 F 配对保留 `ΔF G ΔFᵀ`（周期距离再对 images 取最小）。它不保证所有等价整数基在 chart 中邻近。

CPU例取 `L=3I`，`M=[[1,3,0],[0,1,0],[0,0,1]]`，`L′=ML`。两者体积同为27 Å³，描述同一 simple-cubic translation lattice，但无迹 half-log-metric 的 Frobenius 距离为 **1.68965**，γ角从90°变到18.435°。按现有 metric-cell 分类可变成 monoclinic，conventional cubic symmetry却不变。这同时反驳“log-SPD已解决GL(3,Z)”和“A可硬套metric-lattice→SG的一对一表”。

C的 `v=log(V/N)` 与H的第六通道 `v/√3` 可以只是尺度/排列约定；若均值、标准差和 numerical floor 同步变换，标准化后可以等价，不应算成两个物理机制。两者都应固定 train-only normalizer。Gaussian chart 无界，数学 SPD 不等于浮点 exp/Cholesky 总是可用；有限数值/极端 cell 的失败应明示，不偷偷以 clipping 重定分布。

chart score也不是力。例如物理体积密度 `p(V)=exp(−V)`、`v=logV`，则 `p(v)=exp(v−exp(v))`，score为`1−exp(v)`，而负能量梯度为`−exp(v)`。这个 Jacobian 反例只是限制“score即物理力”的解释，**不要求给已明确的经验 `dz dF` DSM 目标擅加一个新能量/Jacobian项**。

## 5. σ、N、Å与 translation：不能通过名称继承物理尺度

两稿的 fractional isotropic noise 都是合法但有偏置的选择。在 row-lattice 约定下，Cartesian covariance 是 `σ_F² LᵀL`。固定 VPA=20 Å³、fractional σ=.05，N=1/4/10/20的立方胞单轴 Cartesian 标准差分别为 **.13572/.21544/.29240/.36840 Å**；N从1到20增加2.714倍。对 `L=diag(2,20,5) Å`，同一σ对应单轴 `.1/1/.25 Å`。它们不能借用 V2 的 `.05/.15/.30 Å` 解释成同等局部修复。

N/体积归一和 `1/(3N)` loss平均解决的是不同问题：前者组织晶格尺度，后者避免大N仅凭通道数取得更大样本权重；都不自动将 fractional noise 变成 Å 各向同性。网络看见L可学习补偿，不等于一至两epoch已经学会。

还必须固定 translation 与 canonicalization 的合同：

- 在完整 torus 上噪声独立并只 wrap，是当前公式最直接的实现。需要全模型平移等变时，绝对位置/程序特征和训练原点分布也要一致；“有周期特征”不能代替证明。
- 若把 N 个独立噪声减去共同平均，covariance从I变为`P=I−11ᵀ/N`，秩是N−1，单站方差为`1−1/N`；N=1完全为0。它不再是原文的独立全维 torus conditional density。可以另定义 quotient/pushforward过程，但不能加一次居中后继续照搬原目标。
- `F=(.99,.01)` 与共同平移`.02`后的`(.01,.03)`物理等价；简单算术均值居中却分别给`(.49,.51)`和`(.99,.01)`，代表元跳半个cell。物理相对几何未变，绝对特征/标签却变了。
- 原 source 的 cell setting/site permutation 可一次固定并沿噪声携带；不能每个 noisy 时刻重新排序、Niggli/primitive化或按 clean 坐标匹配，再声称还在训练原始密度。若确实采用这种变换，需明确其共同作用与新测度，而不是一概禁止规范化。

## 6. 正确 score 与有限 solver：两个真实数值反例

### 6.1 C 即使完美 epsilon 也可因选定方差变窄

取一个标准化晶格通道的真实 clean 分布为 `z0~N(0,1)`。所有 VP forward marginals 都恰好N(0,1)，终端 prior完全正确，最优epsilon回归也是已知的。此时真正反向条件是

\[
z_{k-1}\mid z_k\sim\mathcal N(\sqrt{\alpha_k}z_k,\beta_k),\qquad
\epsilon^*(z_k)=\sqrt{1-\bar\alpha_k}\,z_k.
\]

C 的 mean代入后正确，但`beta_tilde`小于`beta`，丢掉了对未知z0积分后的条件方差。由`v_{k-1}=alpha_k v_k+beta_tilde_k`可直接递推。对本设计允许的示例`lambda_z=18`，T=32/64/128的最终方差为 **.74557/.85202/.91843**，目标为1；64步窄约14.8%。这不是实测 train chart 的λ，也不是任何晶体模型成绩，而是其**有限kernel误差可在零模型误差下存在**的严格反例。[CPU结果](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_v3_physics_r1/counterexamples.json)

DDPM原文也区分了Gaussian数据适合β、单点数据适合`beta_tilde`这两个方差选择；不能把已知clean条件下的posterior方差自动当成一般marginal reverse的精确方差。[Ho等，§3.2](https://arxiv.org/pdf/2006.11239)

这不否定 C 是归一化生成模型，也不证明换β后真实SUN就上升；它要求把solver/方差误差与学习误差分开。**该反例不适用于H的v/ODE：同一Gaussian例的理想v drift为0，概率流可保持N(0,1)。** H仍有自己的有限时间网格和神经场误差。

数值实现可直接用代数等价形式

\[
\mu_k=\alpha_k^{-1/2}\left(z_k-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\epsilon_\theta\right)
\]

避免先除极小`√bar_alpha`形成巨大`hat z0`再乘回；这是稳定改写，不改变kernel，也不消除上述方差问题。不能用“修了数值表达”宣称已经改进物理分布。

### 6.2 C的末步注噪与H的有限ε，都不能保证codec几乎不变

设一维clean fractional为`.0049`，原`.01`网格的canonical输出是0，上边界为`.005`。对C末步、近clean的精确单点wrapped score，漂移消去当前噪声后又加`σ_1 ξ`。直接算wrapped Gaussian落入量化bin的质量：

| 终点 | 残余fractional尺度 | 变成不同codec bin的概率 |
|---|---:|---:|
| C，γ=1 | .001 | **46.0172%** |
| C，γ=1.5 | .0000316228 | **0.07827%** |
| H，理想ODE输出`p_ε`，ε=.002 | .002 | **48.0061%** |

C行忽略的是可忽略的远像项；H行是理想边缘在有限ε仍有的blur，**不是说H的ODE末步又注入了随机噪声**。三行都不是总体codec失败率估计；它们证明“σ小于bin一个数量级，所以可忽略”没有普遍保证，并给出了rank相关的精度冲突。

这也不自动否定headline SUN：共同R可能把附近点带入同一盆地。应分别测量raw几何/F/应力、连续→Q的转移以及固定R/tau后的SUN。选择float native、Q主终点或额外terminal readout，应在结果前明确；增加readout/corrector需要重新计NFE，不能输出不同端点后择优。

### 6.3 密集target的数量不是有效物理信息

对单个wrapped-normal条件目标，CPU积分的`E_q[(σs*)²]`在σ=.001/.05/.2/.5/1时约为 **1/1/.7631/.001021/5.65×10⁻¹⁶**。高噪声torus target接近零；lattice高噪声epsilon目标也可能主要学输入噪声的简单关系。6.7倍通道数既不是6.7倍独立结构，也不是同等梯度信息或SUN倍数，两稿对此的限定应保留。

H的log-uniform `t∈[.002,1]`在`t>.9`的质量只有约**1.695%**；每源两次G view后约**96.64%**的源不会在这个高时间段出现，但全训练仍约有920个这样的G状态。这是拟议分布的期望，不是实际训练计数，也不存在每个source必须遍历每个t的定理。它说明应按时间段看held-out误差和先验生成，不能只用低噪声平均loss背书H-P。

## 7. P的实际作用：允许简化，不为故事规定新clock

C的`γ=1+.5(1−rank/(m−1))`确实改变过程；0.5的幅度和这个clock本身却没有SUN依据。对任意P，若有其正确forward、正确终端marginal和精确reverse，最终仍回到该clean条件分布。**当clean数据在给定条件后与P独立时，不同clock不会凭空创造P的目标语义。** 它可以改变有限模型的计算分配、数值误差与样本效率，这些才是可检验贡献。

上节的46%对0.078%说明clock不仅“让某类更早恢复”，还使不同rank获得不同终点精度。m=1时更没有程序排列自由度。这不是禁止clock，而是反对将它纳入最小C的必要部分。

建议第二轮先以`γ=1`、P仅作现有条件/rank为简单规格。若以后仍要声称learned P优于canonical，应在**相同clock形状/幅度和相同NFE**下比较排列，不能把“加入异质噪声”与“学到更好程序”合并。H-P的同步全胞更新更诚实地放弃了旧scalar顺序主张；H-C通过前置construction保留真实P路径，但必须为额外d次forward和proposal偏移付费。

## 8. 共享DLM计算图和活参数：具体工程验收，已有问题不重复指控

原`numeric_adapter`只修改`output.logits`，`output_delta`也是hidden下游的logit投影；纯G loss从H读出时不会反向训练它们。`input_delta`则在embeddings上游，全numeric MASK时仍可能有N/E/条件行参与。[实际forward](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_repair_model.py:226)、[追加输入/输出行](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_repair_initialization.py:60)

因此C不能原样报36.99M活参数；应从optimizer中冻结不被score-only路径使用的模块。H最新§5.4已给出G模式可入图的整tensor库存上界26,720,601、T+G并集37,570,905，并承认它们不是实测有效自由度。这一处理方向正确：T+G中output-only参数由T训练，纯G中应冻结；同一embedding tensor的部分行未被索引，不等于整个参数未被DDP使用。

新模式的最低验收很具体：G的浮点状态确实进入共享H；head打开后LoRA/conditioner收到有限、有作用的梯度；T-only模块在G里预期无梯度；两种task及梯度累计在实际DDP/checkpoint配置下完成更新。现V2已`find_unused_parameters=True`，不应再提出“先修原来False”的过时指控，也不能未经核查换成假定图不变的设置。[官方DDP参数语义](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)

零初始化head首步可有head梯度而没有hidden梯度，随后应再检查；这与持续`detach(H)`不同。旧PMTR的`no_grad`/hidden detach不能复制到新共享训练路径。监督/随机输入的构造可不反传，推理可`no_grad`；**不需要为证明共享主干而把整条32/64步采样链放进训练反传**。直接用旧wrapper计算全部vocabulary logits后丢弃，也没有省下那部分成本；只返回并计算G需要的H/heads，是一个可复用API工作。

## 9. A与B的特定修订

### 9.1 A：属性可识别，不等于prior请求可实现

A应该冻结`a_δ`，但metric-cell class、conventional SG bucket和VPA-bin仍是三个不同定义。现有`lattice_system`由lengths/angles判别，SG由`spacegroup.number.conv`等metadata独立判别；`sg_195_230`并不要求当前primitive或非约简cell看起来是cubic metric。[源定义](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/r5_plan_state.py:280)、[已核组件语义](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/reference/component_specs/C3FD_RICH_FIELD_SEMANTICS_AUDIT_V2.md)

`.01`网格不能精确闭合1/3平移：`{0,1/3,2/3}`变为`{0,.33,.67}`后，再施加1/3平移，最大匹配残差为`.0066667` fractional；轴长6 Å时是`.04 Å`。这是群闭合的几何反例，未运行spglib，也不估计真实SG失真比例。δ要用物理单位/明确cell规则，并保留unknown；不能把分析失败默默映射成P1，或把旧SG注释直接当成新的Q端SG。

此外，冻结π仍在旧标签关系上训练。即使三个边缘head分别完全校准，联合请求也可能落在没有监督的组合：同一c仅有`000/111`两个等概率tuple，三个独立公平bit给这两个tuple的总质量只有25%，其余75%在未观察组合。**未观察不等于物理不可能**，但条件likelihood不再约束这些请求。需核已保存soft vocabulary、unknown可表达性和joint prior支持，不能只给DLM换aδ就宣布上游联合分布也被修复。

这是一组可以离线结束的检查：输出端属性定义、旧注释/Q重算混淆、unknown和每源标签处置、π对新标签/组合的覆盖。如果用户只需要软特征的计算作用，而不声称几何请求控制，A不是必须加入的修订。

### 9.2 B：外层rollback需要真正的事务结果

B使用实际三档mixture、量化/拒绝/fallback并匹配unknown通道的方向正确；不能只取最低σ再沿用同一stationarity证明。

但当前`_restore`只记录rollback事件，不将top-level trace.success设false；full-cell事务最后对所有alive样本都写`end`。所以B若仅看`success=true`或是否出现`end`，会把已回到U的inner failure当成功，不能保证回到加噪前G。[`_restore`](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/programmed_path_runtime.py:234)、[full-cell结束逻辑](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/programmed_path_runtime.py:321)

最低实现应明确区分inner commit、inner rollback、合法unchanged，以及forward exhaustion。它可以直接从已分相位的事务事件构造结果，不需要另造复杂系统。不能把“unchanged”等同failure，也不能用整体生成成功掩盖内层回滚。这个阻断属于B新外层的接口定义，不是说当前V2既有合同违约。

即便回滚正确，它也会改变模式权重。toy中目标是`(.5,.5)`、成功输出质量为`(.5,0)`、失败率`.5`；回滚到输入产生`K=[[1,0],[.5,.5]]`，一次将目标变为`(.75,.25)`并最终集中到第一模式。它规范化且保留有效输入，却不保证多样性或SUN。对合法但物理更差的输出更无保护；不能增加R、放宽验收或反复H直到成功来掩盖净损害。

## 10. H-P/H-C与资源：哪些是选择，哪些才是阻断

H的v端点正规、ODE符号及uniform-torus prior界成立；`ε=.002`、Euler32和log-uniform时间都是**明确但未经质量验证的设计选择**。不需要先证明一个全局SUN定理才可做候选；也不能因为这些公式正确就签发采用。

H-C的起点是`p_construct q_τ`，不是`p_data q_τ`，ODE也不是Bayes posterior随机抽样。τ=.1在平均N附近并非V2的`.1 Å`小扰动；它给fractional σ=.1。τ接近1时会丢掉proposal信息，却仍支付construction费用。当前没有constructor→tau800的对照，更不能由repair→tau800推导H-C优劣。

| N | V2 `2d` | C64 | H-P Euler32 | H-C `d+32` | H-C Heun32区间 `d+64` |
|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 64 | 32 | 41 | 73 |
| 10 | 72 | 64 | 32 | 68 | 100 |
| 20 | 132 | 64 | 32 | 98 | 130 |

六卡/global24只是已有可工作的预算锚点，不是新G API显存或吞吐的证明。37M左右适配参数、0.58M新模块和相同forward examples都不能直接换算成秒数；8B主干激活、词表投影是否跳过、全噪声几何特征和batching才决定实际成本。上游、model494和共同R的调用另列；不要把tau800当成这里的DLM NFE。

只需在实际实现后做一次有边界的T/G前后向与峰值显存/吞吐核验，再决定是否满足预算；这是可结束的工程检查。真实free generation与SUN必须之后测，不能把一两epoch“或许不够”设成无限期阻断，也不能预先保证足够。

## 11. 给第二轮规格的具体建议

我的最小选择建议是 **H-P为连续候选主路径，uniform clock，v/torus heads，共享V2主干，保留已提议的一T一G配方；H-C只作为同权重的可选诊断，不先锁定τ，也不加species clock/GNN/能量teacher。** 理由是H-P删掉了proposal起点和rank终端精度这两个额外假设，T仍提供可核的现有construction路径。代价是每源G曝光减半；若主审改选纯G，就明确放弃T锚点，并按纯G冻结dead模块，不能沿用T+G参数或损失账。

按用户当前SUN实效优先的目标，我倾向明确命名 **mixed-native float CIF** 为该新模型的主输出，同时固定报告同样本的一次Q roundtrip；共同R、hull、N/U阶段、失败分母和refiner保持一致。若论文问题明确要求token-native，则在规格阶段把Q端改为主终点；不能等两边SUN出来再选择。改变输出表示可以是方法差别，不能继续称为原categorical采样的收益。Q后的概率是其原像质量，0/100只对应一个canonical物理bin，G模式没有两个LM logits需要再做logaddexp；T模式的原alias处理仍保留。

连续标签方面，先核已有CIF回连；若尚未完成，规格可以诚实选择全量decoded-token targets做首个机制比较，保留同源完整分母，明确没有sub-bin监督。不能混合精度而不标记，也不必为获得“连续”名称先下载新训练库。保留现有half-log chart和一次固定site permutation，不新增运行中canonicalization。

| 必须在规格/实现时关闭的事项 | 可行动处置 |
|---|---|
| 原CIF标签身份/精度未声明 | 逐源回连证明，或明确选择完整decoded labels；不以同formula代替 |
| float与Q哪个是主终点未定 | 在看候选结果前确定，并固定另一端诊断 |
| G读不到真正float geometry或H被detach | 显式G API；真实一小段前后向检查共享梯度 |
| dead模块/任务DDP账不清 | 按T/G或纯G活路径列optimizer组，沿用已有unused策略并核验 |
| B没有可识别inner事务结果 | 区分commit/rollback/unchanged，外层失败回原G |
| A新属性/unknown/π支持不清 | 冻结aδ与词表、报告joint支持，不加错误的metric→SG硬表 |

其余如Euler32是否足够、log-uniform是否最好、H-P能否学会先验、T是否退化、P是否提高SUN，是**待测的候选效用**，不是已解决的事实，也不是逻辑上永远无法开工的阻断。若实际SUN不优于更便宜的construction/现有基线，就应接受简化或否决；不靠追加clock、R或多轮修复维护故事。

## 12. 本轮可复核交付

[counterexamples.py](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_v3_physics_r1/counterexamples.py) 与[结果JSON](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_v3_physics_r1/counterexamples.json)包含：C末步与H有限ε的codec概率、Gaussian数据的DDPM方差递推、wrapped target二阶矩、N/Å/NFE、等价基chart距离、平移投影covariance、1/3群闭合、rollback模式偏移、A联合支持、chart score/能量区别、编号错配toy、time覆盖期望与零head首步梯度例。数值输出为有限值并记录所读源码SHA。

本轮没有验证实际CIF全量映射、任何新模型梯度、显存/吞吐或SUN。以上裁决足以起草一份更小且对象明确的规格；第二轮应攻击那份具体规格，而不是再把所有路线泛泛叠加。
