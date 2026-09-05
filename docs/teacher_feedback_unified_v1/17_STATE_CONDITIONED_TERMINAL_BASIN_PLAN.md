# 保留科学 LLM 的状态条件化、双目标盆地监督 SPAD

日期：2026-09-05。版本：双目标审计修订稿。最新状态：**用户要求停止本地推进并交接，不干预A800任务；自动化已暂停。39853 warmup与39857条件准备已自然完成，详见[当前交接](../HANDOFF_STATE_PROGRAMMED_SPAD_20260905.md)。下文计划等待用户明确恢复。**此前24小时努力目标为2026-09-06 19:19 Asia/Shanghai；最新额度6A800、每卡4CPU，不自动续期。

当前工作树：`D:\codex_work\ai4s\DLM_llama_programmed_basin_closure`，分支 `codex/llama-programmed-basin-closure`。

**恢复记录：用户于2026-09-05 21:58 Asia/Shanghai明确恢复同一主线实施；复用39853/39857，以上暂停段落为历史快照。当前状态以[04清单](04_EXECUTION_CHECKLIST.md)顶部为准，截止仍为2026-09-06 19:19 Asia/Shanghai。**

本稿完整参考用户提供的 GPT-6 网页意见，并重新检查当前代码。此前迭代记录见 [GPT-6 审计交接](../GPT6_AUDIT_HANDOFF_20260905.md)；这里不再复制整套历史。

## 1. 决定：采用新的稳定性方向，但保留现有 LLM 贡献

建议下一条路线只做一个统一改动：

> **C3FD 支持的 Llama 继续决定合法组成与构造程序；DLM 在该程序下读取实际周期状态、协同修改晶格和多原子，并用完整路径的“低能盆地 + 低弛豫负担”共同训练原生 token 概率。**

不是继续修 PMTR 的连续位移头，也不是把 K10 loss 改名。要一起修正三个相互依赖的对象：

1. **状态信息**：被重新 mask 的变量，其旧数值和真实周期环境不能从条件中消失。
2. **可执行动作**：允许晶格与相互耦合的原子一起改变，避免分步验收挡住协同移动。
3. **监督对象**：同时降低完整部署终点的弛豫负担与最终盆地能量，不再把局部动作的瞬时降能当作最终稳定性。

保留现有 C3FD/Llama 权重、species program、`7+4N`、SPAD 事务与非因果修订。新增一个轻量状态输入接口，不新建连续生成器、不加载推理 MLIP、不恢复 G2/BTRD。model494 仍保留为单列的系统保底。

这比继续扩大局部头更值得投入，但不是已经证明能过 `10% / 50%`。本稿把预期优势、可验证条件和剩余风险分开，避免再次把数学上理想的性质当成实际训练保证。

## 2. 当前基准及这次真正要解决的问题

### 2.1 已经有效的部分不重做

| 已有部分 | 保留方式 | 本轮不作的推断 |
|---|---|---|
| C3FD 对 Llama 的合法化学动作支持 | 原组成支持与 typed Planner 原样保留 | 化学合法不等于该组成的任意结构稳定 |
| 同一 Planner Llama 的 scaffold/species pointer | 继续输出构造顺序，并参与协同修改区域的选择 | 静态程序不是实时 Llama controller，也未证明已有 pointer 本身提高 SUN |
| SPAD 完整 cell/XYZ 事务、合法支持和 suffix-visible 修订 | 保留为解码基础 | 近 100% 几何有效不等于近 100% 低能 |
| 已有显式 `L|X` 与反向物种修订训练 | 使用真正受过该训练的 closure 基座 | 不能把早期未受训的 DLM 当作已能闭环预测晶格 |
| 固定 model494 | 留在独立 refined 结果栏 | 不代替 DLM 原生稳定性证据 |

### 2.2 必须按同一评估端点理解已有结果

| 实验/条件 | 已观察结果 | 支持的结论 |
|---|---|---|
| BS 机制开发队列 | Raw Direct `511/512 = 99.80%` | 执行与基本几何问题基本解决；不是后面所有队列的相同计数 |
| 固定 512 的 BS raw SUN 队列 | Strict `16/512 = 3.12%`，Meta `107/512 = 20.90%` | **无 model494** 的低能盆地命中率仍低 |
| 同一 BS + tau800 | Strict `35/512 = 6.84%`，Meta `234/512 = 45.70%` | 连续细化仍有实际系统收益 |
| 完整 MP20 closure CE，固定 256 | comp/fast-struct `256/256`；Strict/Meta `7/56`，对应 BS `7/54` | 状态/顺序训练能保持有效性，稳定尾部未明显改变 |
| K10 大池 | 4104 状态、15348 候选；局部 best-vs-no-op median headroom 约 `143.93 meV/atom` | 有局部候选降能空间；不是收敛终态的路径价值 |
| 大池训练后的独立流 | stream19 raw `12/61`、tau800 `14/107`；下一遍 stream21 raw `5/49`、tau800 `14/123`，各分母 256 | 未形成稳定过线证据；**不同 stream 不能直接证明第二遍训练退化** |
| 最近 PMTR | 39825/39826 类型错误、39827 首次更新前非有限梯度；无训练后模型结果 | 是工程失败与设计风险，不是 PMTR 已被实测证明无效 |

来源：[当前 README](../../README.md)、[最新结果工作日志](11_LATEST_RESULTS_WORKLOG_AND_NEXT_STEPS.md)、[stream19 诊断](13_STREAM19_DIAGNOSIS_AND_FINAL_ITERATION.md)、[PMTR 代码审计](16_PMTR_CODE_GROUNDED_ARCHITECTURE_AUDIT.md)。

特别注意：仓库的 **raw SUN 也包含公共 CHGNet 评价弛豫**。它表示“未先经过 model494 的 DLM 提议”，不表示未经任何弛豫的 token 几何已经稳定。

### 2.3 本轮确认的代码问题

| 当前实现 | 具体问题 | 新设计的对应修正 |
|---|---|---|
| [manifold_repair_head.py](../../src/crystal_dlm/manifold_repair_head.py) | 晶格分支读取 cell hidden 与 pair 消息前的 pooled site hidden；不直接接收数值 metric，也不接收周期距离消息汇总 | 数值晶格、体积与周期邻域同时进入 cell/site 状态条件 |
| [pmtr_training.py](../../src/crystal_dlm/pmtr_training.py) | `frozen_spad_forward` 完全冻结 DLM，只训练外部 head；一次 transaction-start forward 评分多个分量 | 状态接口与现有 DLM LoRA 一起训练；按实际分量提交后的状态评分 |
| [potential_closure.py](../../src/crystal_dlm/potential_closure.py) 与 [train_spad_basin_posterior.py](../../src/scripts/train_spad_basin_posterior.py) | 即便先得到 deployed action score，loss 内仍对 K 个 score 做 `log_softmax`，默认标签仍为 K10；不是已经实现的绝对全路径 likelihood | 替换训练目标及 trace 数据对象，不把此前文档中的计划当成已实现功能 |
| [manifold_token_transport.py](../../src/crystal_dlm/manifold_token_transport.py) | 连续目标通过有限 token transport 影响输出，不等于充分学习完整离散动作分布 | 原 LM head 直接输出完整合法 token 家族上的概率 |
| [spad_generation.py](../../src/crystal_dlm/spad_generation.py) | cell 改完就对旧坐标进行整胞支持检查，可能拒绝“新晶格 + 随后移动原子”才合法的组合 | 新增一个 cell+sites staging transaction，完整联合动作结束后才验收 |
| [transaction_logits.py](../../src/crystal_dlm/transaction_logits.py) | 已保存 `complete_pre_remask_tokens`，但主要交给外部 logit transform | 复用这个快照，把旧几何真正送进 DLM，而不是再造一套隐藏缓存 |
| [periodic_geometry_ops.py](../../src/crystal_dlm/periodic_geometry_ops.py) | 当前是有界 27/125-image 搜索 | 不再写成对任意极端三斜基底都有全局精确保证；在现有支持域内复用并检查斜晶胞边界 |

这次没有发现“必须换词表”的依据。部署坐标实际是 **101 个编码值、0.01 fractional 步长，000/100 为周期别名**，不是 1000-bin。晶格长度步长 0.1 Å，角度步长 1°；动态特殊 token 2457 个、底座完整词表 128830。全 MP20 27136/9047 已能准确序列化。

此前 Phase 0 的连续→量化单点能量 median 差仅 `+2.846 meV/atom`，q95 `+19.031 meV/atom`，Meta proxy retention `98.63%`。保留词表；它没有证明 Strict 零阈值完全无量化问题，也没有完成所有多晶型的盆地保持证明。参见 [token 审计](06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md) 与 [Phase 0 结果](09_EFFICIENCY_FIRST_POTENTIAL_CLOSURE_PLAN.md)。

## 3. 科学对象：局部不平衡与高能盆地不是同一件事

固定组成 `c`，晶体 `x=(L,U,A)`。令 `e(x)` 为同一势的每原子能量，`R(x)` 为明确收敛规则下的变胞弛豫算子，`h(c)` 为同一参考体系的组成 hull 能量：

\[
e(x)-h(c)
=\underbrace{e(x)-e(R(x))}_{\text{局部非平衡/弛豫负担}}
+\underbrace{e(R(x))-h(c)}_{\text{终态盆地热力学劣势}}.
\]

这是代数分解，不自动保证某个有限步优化器单调降能，也不能混用未校正 CHGNet 总能与不兼容的 DFT 参考而称为精确物理定律。

同一组成、同一能量与校正规则下，`h(c)` 是常数，因此训练可以在组内比较 `e(R(x))`，不用反复联网查 hull。跨组成绝对能量排序不成立。

当前已高概率进入几何可行域 `V_c`，但尚未高概率进入低能盆地的吸引域。更具体的研究问题是：

> 在给定组成的多盆地周期势能面上，能否使单次结构提议更常产生共同评价协议下的低 hull 终点，同时减小原生—弛豫终态能差，并维持既定输入结构的 N/U？

这不是“让 force 更小”就能回答：高能亚稳极小值也可以有零 force；也不是“局部动作更容易弛豫”就能回答：最终落入的盆地可能没变。

本轮主目标是提高

\[
\Pr_{x\sim p_\theta(\cdot\mid c,P)}
\big[e(R(x))-h(c)\le\delta\big],\quad \delta\in\{0,0.1\}\;\mathrm{eV/atom},
\]

并保持有效性、Novel/Unique 与组成覆盖。原生 `e/F/stress` 和 `e(x)-e(R(x))` 单独衡量几何精度。低 hull、动力学稳定、可合成性不能混为一谈。

本次修订将两项都纳入显式训练对象，记

\[
A(\xi)=e(x_T)-e(R(x_T)),\qquad
B(\xi)=e(R(x_T))-h(c).
\]

只降低 A 可通过抬高弛豫终点制造假改善；直接等权最小化 `A+B` 又退化为只优化原生能量。我们因此采用**固定条件分布下，两项平均收益分别为正的经验路径教师**，而非固定混合 loss。

这是对整个生成程序的终点要求，不要求每个中间 token 动作降能，也不要求每个组成的四条候选都存在同时改善。细节见第 5.3 节。

## 4. 一个统一的模型和执行接口

### 4.1 保留 Llama，冻结本轮 Planner，不重新采 Plan

```text
C3FD 的可达化学支持
       ↓
现有 Planner-Llama：composition / compact Plan / species program P
       ↓                           ↘ 同一 P 选择协同修改区域与修订顺序
SPAD DLM：非连续构造 → 读取已生成完整周期状态 → 协同修订 → 原有闭合
       ↑                                                    │
       └──── 离线终态盆地价值训练这些实际 token 决策 ──────────┘
                                                            ↓
                                              单条原生晶体输出
                                                            ↓ 可选、另列
                                                固定 model494
```

Llama 不是装饰性 prompt：现有 pointer 输出的是**物种排列**，编译器再把它映射为 canonical body 的原子槽位、各物种首槽 anchor、构造次序与反向闭合次序。它并不直接预测任意原子级图或数值晶格。新增区域选择器把同一排列与当前周期邻域结合，形成可执行修改区域。DLM 决定具体晶格、坐标，并能保留未来原子重写早期部分。**本轮不声称端到端联合训练 Llama，也不声称在线 Llama 回路。**

现有 Plan、pointer checkpoint 与参考生成流均复用。新稳定性训练不改变 composition 概率，也不增加“更容易生成的组成”偏好。

三种表示通过语义对象而非 token ID 相接：Llama typed chemical actions → `N/elements/counts + soft Plan + species permutation` → DLM canonical `7+4N` 槽位；数值状态编码只读取已提交晶体，不接收 CHGNet 能量、力或 test 标签。C3FD 不承担几何 logits 的预测。

### 4.2 状态输入：给 DLM 看真实旧几何，不再只给被 mask 的数字

对每次修订，保存两个不同对象：

- `old_state`：事务开始前完整晶体的数值快照；
- `canvas`：DLM 此刻看到的旧/已改 token 与 MASK，及 active/committed/unknown 标志。

一个轻量 `PeriodicStateConditioner` 从旧晶体计算：

- cell：`G = L L^T` 的归一化六分量、`log(V/N)`、尺度/形状特征；
- site：物种、program rank、旧坐标的周期编码、已知/活动标志；
- environment：物种相关的周期距离 RBF、邻域方向/相对坐标、program-rank 关系；
- cell pooling：明确汇总 site 的周期环境，防止晶格分支再次看不到 packing。

最小实现采用小型共享 pair MLP + pooling，把得到的 cell/site 向量投影为**现有对应 token 位置的 embedding 增量**，送入 DLM 的双向 Transformer。最后仍由原 LM head 输出 token logits。

这样不增加词表、不延长 `7+4N`、不加连续回归头、不使用 SPD 特征分解反传、不另建大型 GNN。DLM 的 attention 负责跨区域的信息交换，而不是在 LM head 后用一个向量单独决定修复。

状态数值运算用 FP32；投影输出与 embedding dtype 对齐。输出投影零初始化以继承**同一输入状态下的 logits**；这不保证改用联合 mask/schedule 后整条轨迹仍与旧基座相同。已有 LoRA 与 conditioner 联合训练，不再 `frozen_spad_forward`。

最小实现的明确边界：

1. 构造阶段只编码实际已知的几何；没有晶格或坐标时使用 unknown 标志，绝不能从 ground truth 或 q0 坐标均值补齐。
2. 修订阶段旧坐标在状态侧可见是设计本身；新目标坐标不可见。这用于修正旧结构，不是把答案泄漏给模型。
3. 周期邻域不能只保留 `i != j` 的单个 MIC：单原子/小胞也有自身周期像邻居。保留有限范围周期像、边界掩码，固定支持域测试；不宣称任意晶格基变换严格等变。
4. MP20 warmup 采用一致的周期平移、同元素置换与 program-to-site 重映射，不能只改输入不改 target。带能量的已采集 trace 不额外作会改变几何/状态概率的随机扰动；若做严格等价变换，必须同步重建整条状态与评分，不能沿用失配的路径概率。
5. 部署底座提供 `get_input_embeddings()`；还需小型接口测试确认实际 LLaDA/PEFT 的注入、重计算和 reload 行为。不能把另一模型的 `inputs_embeds` 支持当成现成结论。

实现限定为一个显式、逐次调用的状态包装器：优先通过模型支持的 embedding 输入传递；否则使用局部 embedding 包装，不能依赖一次 forward 后就清掉、而 backward 重计算仍会读取的可变全局 hook 状态。不得把 conditioner 输出 `detach`；旧几何本身可以作为非可微输入。零投影初始步上游 encoder 梯度为零是正常现象，检查后续短步是否能训练，而不是据此误报失败。

### 4.3 动作：一个联合 cell–multisite 事务，保留原生离散生成

保留原 SPAD 构造，然后加入一次联合修改：

\[
a_{\rm coop}=\big(L',\{U_i':i\in S(P,x)\}\big).
\]

`S` 用固定、无能量的规则选择，默认规模 `M=min(N,max(2,ceil(N/2)))`。根为 program 最早物种的首槽；优先按 program 纳入尚未覆盖物种的近邻，再按距已选区域的周期距离扩展，距离相同按 program rank/native slot 打破平局。仅在组成允许时形成跨物种区域，不要求每次覆盖全部物种。

N≥3 时该规模保留部分未改坐标作为可见上下文；N=1/2 的联合步骤可能没有剩余原子，单列为小胞退化情况，不宣称每个联合步骤都存在未改 suffix。后续闭合仍使用已生成的完整上下文。

执行方法：

1. 保存完整旧状态；同时 mask 六晶格 token 与所选 XYZ。
2. 在 staging canvas 依原生分量条件分布逐分量生成；每次看到已生成的新分量及未改区域，不将 XYZ 当作三个独立的边缘概率相乘。
3. 全程保留 exact composition、字段类型、非退化晶格等已能确定的支持检查。
4. 对将要重写的原子，不再用其旧坐标提前拒绝新晶格；对已确定的留存/新原子仍作可确定的几何检查。
5. 完整联合动作结束后作整胞支持检查；合法一次性提交，失败整体回滚，不额外候选重试。
6. 用同一 DLM 执行现有 reverse-species closure，输出一次。新联合 cell+sites 步骤**替换**旧 closure 的单独 cell 步骤，不再额外先跑一次旧 cell closure。不是循环到能量满意，也不调用在线 MLIP。

协同更新允许跨越逐原子/逐晶格更新难以通过的中间状态；**它不是一条模拟真实原子动力学的轨迹**。目标是可行终点提议，不要求 token 提交中途处处物理低能。

回滚范围包括六个 cell token、全部活动 XYZ、lattice version 和派生几何缓存；非活动 token、组成和 program 不变。只有旧状态本来合法时，回滚才保合法，不能借回滚抹掉原构造失败。现有 cell validator 的 N=1 分支未覆盖自身周期像；需要共同 validator 对参考与新输出检查这类小胞，作为一致性修正单列，不归功于稳定性学习，也不静默改写旧 99.8% 的定义。

保留现有物理最低碰撞支持；本轮不顺手把 0.5 Å 改成普适共价半径硬下限。不同键型、配位与压力下并不存在这样简单的普适合法边界。距离信息应作为模型条件，不冒充热力学判据。

## 5. 训练：先适配状态和动作，再学习低能盆地与低弛豫负担

### 5.1 基座与 MP20 几何训练

参考基座选已有完整 MP20 closure 训练的 checkpoint，保留受过训练的 `L|X`、非因果坐标回填能力；不从尚未产生 checkpoint 的 PMTR 恢复。

第一阶段覆盖**完整 MP20 train 27136 条，一轮 source epoch**，不再用 384/512 条接口样本充当正式训练集。

每次由同一个 MP20 晶体构造一个一致的 source→target：包含 cell 应变、相干多原子扰动、普通 mask 与联合事务 mask；训练目标仍是该晶体的原生表示。其目的只是状态识别、几何恢复和新事务适配，不把这些局部扰动称为跨盆地教师。

本轮不再训练 Planner。基础 SFT 的 teacher-rich 数据保持不变；接口 warmup 复用完整 MP20 closure 数据的 teacher Plan 与已记录 program 来源，明确区分 learned pointer/contact-tree teacher/canonical fallback，不能把 teacher order 冒充 Llama 预测。新 on-policy 阶段只使用部署 Planner 的冻结 predicted Plan/program；不把两阶段同 schema 说成完全同分布。

联合 warmup 只扰动本事务可修改的坐标及晶格，非活动分数坐标保持同一源晶体的一致值。这样目标联合动作结束时确实对应该 MP20 晶体，不拼出“新 cell + 无关旧坐标”的教师。旧状态始终是 source，不可偷换成 clean target。

训练必须使用与部署相同的旧状态侧条件和分量顺序。目标原子只在其被监督之后进入后继 canvas。不同 MP20 多晶型绝不拼接成“前半来自一个、后半来自另一个”的伪教师。

固定初始设计：LoRA 延续已有结构，conditioner 小规模新参数；有效 source batch 16，一轮约 1696 updates。warmup 初始 LR `1e-5`，warmup 100 updates、cosine；盆地阶段初始 LR `5e-6`，不按 SUN 改学习率。首个短工程测试只判断数值、梯度和部署评分一致性，不用 SUN 搜参。最早 checkpoint 只作恢复，不按 dev SUN 挑选。

为使预算真实，warmup 每条 source 抽取一个部署子状态及实际 active 分量，跨 source 均衡覆盖 cell、多原子和后续闭合；不是每条 source 的全部子步骤都反传后仍声称只有一次前向成本。普通 MP20 masked CE 与这种修订条件监督分别记账。这里的一轮表示 source 覆盖一遍，不保证所有可能修订状态已经学完。

### 5.2 终态教师必须来自完整部署路径

固定 1024 个 MP20-train 条件，使用已有 Planner 的同一 Plan/program；每个条件生成 K=4 条**完整独立路径**，每条都包含构造、协同事务与闭合，合计 4096 个终点。

这仍只使用 MP20-train 条件，但多了模型自己生成的训练输入：不是“4096 个新的真实 MP20 教师”，也不是仅凭真实晶体做监督。它正是弥补真实生成状态偏移所需的 on-policy 训练数据。之前“只用 MP20”的数据边界应明确为不读 dev/test 的能量或结构作教师，而非禁止模型在 train composition 上生成。

对每个终点记录：原生单点 e/F/stress、公共弛豫的结构/能量轨迹、终点能量、力和变胞收敛状态。由同一次计算得到 A 和 B 所需的两项能量：**终态盆地与原生弛豫负担共同做 label**，不是 K10、不是 model494 tau800 终点、不是每一步 force 的符号。e0 与 eR 必须来自同一 CHGNet 势及相同能量口径；A 显著为负先排查弛豫或记账，不能通过截零让两项自动变好。

同组重复路径保留其采样质量，不强制重采直到多样；相同终点可共享昂贵评价缓存，但不能把重复频次悄悄抹掉。另记收敛后 basin cluster 与能量跨度，用来判断实际探索覆盖。

**已有数据能复用到哪一步：**128/4104 的旧池是“同一局部状态 + no-op/DLM/force/stress 候选 + 旧 closure”，并非新动作接口下四条独立完整路径。可以复用 train composition、Plan/program、几何工具和同协议缓存；不能直接把旧候选标签改名为本轮终态轨迹监督。

### 5.2.1 不能沿用旧缓存的“有限能量 = 已收敛”判断

[relax_spad_basin_closure_shard.py](../../scripts/relax_spad_basin_closure_shard.py) 当前仅调用 `optimizer.relax(structure, verbose=False)`，输出缓存只有能量和组成，没有 final structure 或 convergence 证明。旧 K10 的 `converged_fmax` 也只检查原子力。新教师需要明确写出：

- CHGNet 权重与软件包版本分别记录；固定 optimizer、cell filter、原子和晶胞自由度、压力目标及最大步数，不能依赖库默认值。
- 拟用零外压、FIRE 变胞优化、原子 `Fmax<=0.1 eV/Å`、残余应力最大分量 `<=0.5 GPa`、最多 500 updates，并要求 cell-filter 的优化终止状态通过。实际包中受支持的 cell filter 必须在第一个接口测试中确认；不能不支持时静默换成固定晶胞。该参数是一份待实施的教师协议，不是旧缓存已经满足的事实。
- 保存 relaxed structure、终点 e/F/stress、实际步数，以及 cell-filter 优化器终止状态。原子力阈值小而残余晶胞应力大，不标为变胞收敛。
- 力与应力同时用于判定/诊断教师终点质量；不作为推理输入，也不把 force norm 当 basin energy。
- 预算内完成且验证通过记 `V=1`。完成但非法/未收敛记 `V=0`，其含义是“未验证”，不是已证明高能；未运行、作业丢失、计算异常单记 unknown，补算或明确留下缺失，不能填零能量。
- 全组未验证不硬造 winner；保留计数并跳过该组盆地更新。由此产生的训练覆盖损失必须报告。

终态训练与新模型/参考的共同评价使用同一个明确协议。若补齐收敛标准导致它不同于旧冻结评估器，应在新协议下成对重算参考；保留旧值作历史列，不跨协议直接计算提升。

### 5.3 双目标经验路径教师，不是 K4 分类器

令 `ξ` 为包含实际采样动作及确定性提交/回滚的完整部署路径，`x_T` 为它的最终原生晶体。理想条件分布目标是：

\[
\min_{q}\mathbb E_{c,P}
D_{\rm KL}(q(\xi\mid c,P)\Vert p_0(\xi\mid c,P))
\]
\[
\text{s.t.}\quad
\mathbb E_q A\le\mathbb E_0 A-\Delta_A,\qquad
\mathbb E_q B\le\mathbb E_0 B-\Delta_B.
\]

合法支持、组成与程序的边际分布固定。在可行性与通常的对偶条件下，解具有

\[
q(\xi\mid c,P)\propto p_0(\xi\mid c,P)
\exp[-\lambda_A A(\xi)-\lambda_B B(\xi)]
\]

的形式，但**两个系数非负本身不保证两项都降**，交叉响应受 `Cov(A,B)` 影响。双降来自显式可行约束。也不能从两个均值下降推出 Strict/Meta 阈值概率或 SUN 必然上升。

#### 实际求解：只用一次标注池，两个低维凸问题

对第 c 个有验证候选的条件，`u_c` 是**该组验证路径的经验参考**：保存每次抽样 occurrence 时为均匀分布，若合并完全相同路径则保留重数。不同 trace 即使终点相同也不合并。未验证组保留在完整 coverage 母表，但不伪造能量参与求解。

每组分别中心化：

\[
r_{A,cj}=A_{cj}-\mathbb E_{u_c}A_c,\qquad
r_{B,cj}=e_{R,cj}-\mathbb E_{u_c}e_{R,c}.
\]

第二式在同组成内恰好消掉 `h(c)`；**没有对不同组成的绝对能量作优先级排序**。令

\[
D_k(w)=\frac1C\sum_c\sum_j w_{cj}r_{k,cj},\qquad
I(w)=\frac1C\sum_c KL(w_c\Vert u_c).
\]

1. 求经验可行的最大共同改善 `ρ_max`：

\[
\max_{w,\rho}\rho,\qquad
D_A(w)\le-s\rho,\quad D_B(w)\le-s\rho,\quad I(w)\le0.2,
\]

其中每组 w 为概率分布，`s=0.1 eV/atom` 只是两个同单位轴的共同尺度，KL 单位为 nat/经验组。该问题具有二维凸对偶，使用数值优化而不是 β/权重网格。

2. 若 `ρ_max>0`，只要求一半可行收益 `d=sρ_max/2`，求

\[
w^*=\arg\min_w I(w),\qquad D_A(w)\le-d,\quad D_B(w)\le-d.
\]

对应两维非负对偶变量：

\[
\Phi(\lambda)=\frac1C\sum_c\log\sum_j u_{cj}
e^{-\lambda_A r_{A,cj}-\lambda_B r_{B,cj}}
-d(\lambda_A+\lambda_B),
\quad \min_{\lambda_A,\lambda_B\ge0}\Phi.
\]

得到 `w* ∝ u exp(-λ_A r_A-λ_B r_B)`。有
`∇Φ=-D-d·1`、`∇²Φ=mean Cov_w(r) >= 0`，可直接检查求解残差。

因为参考 u 与第一步可行解的 1:1 混合已满足半程目标，且其 KL 不超过 0.1 nat/组，第二步最小 KL 解也不超过此值；**不再另叠一套 KL 硬门**。这些默认值是本轮单一设计参数，不是宣称其最优。

如果 `ρ_max=0`，返回经验参考并记录“当前采集池未发现双降空间”。不删除难组成、不重抽凑赢家、不把零收益教师当成功，更不据此断言真实势能面不存在更好结构。首 128 组只作协议/覆盖检查，不按收益挑选余下 896 组；完整池是否值得做有监督更新由这项可学信号决定。

#### 学生如何学习

\[
\mathcal L_{\rm path}
=-\frac1C\sum_c\sum_j\operatorname{stopgrad}(w^*_{cj})
\sum_t\log p_\theta(a_{cjt}\mid s_{cjt},P_c).
\]

教师 w 在 K4 内归一化；**学生每一项 probability 的分母仍是完整的实际合法 token 支持，不能再次对 K4 路径 score 做 log-softmax。**

每条 trace 每遍分层抽六个真实 scalar 决策点（构造/协同/闭合各两个，空阶段按实际支持重新分配），用 inclusion probability 的倒数校正。均匀抽样时等价于 `T/m`；短轨迹有放回则保留重数。所有真实决策都有正包含概率，不再除以轨迹或 3/6-token 事务长度。零 teacher 权重不能经过旧 collator 的 `value or 1.0` 被变成一。

**理论保证止于当前经验教师。**验证筛选使 u 不再是原始四抽的无条件 p0；无验证组成的缺失会改变可监督覆盖。K4 仍存在有限采样与收敛选择偏差。经验池上双降是求解构造性质，不是独立性能发现；学生拟合误差、CE anchor 与未见组成泛化均不继承它。独立固定评估才检验是否真正改善。

若某低能事件在参考中的概率为 p，四抽最多提供“至少见到一次”的覆盖 `1-(1-p)^4`；扩大条件组数不等于增加每组探索。一次模型刷新用于更新这项覆盖，不是对测试结果进行无限重抽。

### 5.4 实际部署概率必须处理的两件事

**分量顺序。** 当前 `add_gumbel_noise` 的采样并非简单对未经温度处理的 logits 求 softmax。训练要复现真实温度、几何 mask 与 token 000/100 合并，逐分量重算；不能在 transaction-start 的同一 forward 下评分全部 XYZ/cell 后宣称精确部署 likelihood。

具体地，温度 `T>0` 时，代码的 Gumbel 变换对应 `softmax(processed_logits / T)`；T=0 是确定性 argmax。alias 合并与温度缩放的先后顺序不能擅自交换。当前 SPAD 单位置程序避免了多位置置信度竞争的边缘化问题；不把该结论推广到任意 LLaDA sampler。

**回滚。** 多个被拒绝的尝试都可能落到同一个 committed no-op。只取旧 token 的 log-prob 不等于 no-op 的总概率。本稿采用记录实际 attempted trace 的方式，把确定性回滚视作状态映射；需要保留 rejected attempt 和失败分支记录，不重新猜测合并概率。

例如 XYZ 的 Z 支持为空：保留已经采样的 X/Y 两项概率；没有采样的 Z 不补概率，恢复旧 XYZ 也不额外乘旧 token 的概率。当前日志没有完整保留这些前缀，scorer 还可能把被回滚的 cell 提议置为负无穷，因此 recorder/replay 需要实际补上，不能直接复用旧日志宣称路径完整。

终点好而协同事务一直回滚的路径不能被描述为“新动作成功”；单列接受率、修改幅度和最终 basin 变化。否则模型可能只学会走旧路径，loss 仍看似合理。

### 5.5 训练强度与刷新

双目标阶段更新 **conditioner + DLM LoRA**，不是只训一个小 head。完整加载 checkpoint 的 token embeddings/output head，但将这两张大表冻结，包括 PEFT `modules_to_save` 的副本；优化器只纳入 LoRA A/B 与 conditioner。先完整遍历 1024 组两轮，抽取每条轨迹不同阶段的决策点；保留少量真实 MP20 CE 更新。CE 不在自己生成的高能端点上做自我模仿。

不采用未经量级检查的固定 `0.5:0.5` 混合。初始安排每四次路径更新插入一次真实 MP20 CE 更新，分开微批次；短梯度检查只排查量纲/数值错误，不用主结果调权重。路径评分关闭 dropout，但保留训练所需 autograd/checkpoint，不直接复用临时切 `eval()` 的旧 scorer；采样和 replay 需做同状态 logits 数值比较。

预先允许**一次**数据刷新：由更新后的同一个模型在同一批 train 条件再生成 K4，重标终态并继续完整两轮。刷新用于更新状态分布，不是测试集没过线后的无限重抽。统计应分别写清 source epoch、trajectory group pass、optimizer updates，不把这些混称 MP20 epochs。

刷新后将参考明确定义为本轮采集模型，形成逐轮经验策略改进。旧轮数据若保留作 CE anchor，只作有标注的回放，不声称未经重要性校正仍是新轮参考的 on-policy 目标。`T/m` 仅对已给定轨迹的状态抽样无偏，不修复 K4 偏差或离策略漂移。

旧稿的单 β 终态加权由第 5.3 节替代，不再作为并行方法。第一轮与刷新轮分别在自己的完整 train 池求一次 teacher 权重，保存实际双收益、ESS、验证覆盖与求解残差；不做按组 MAD 缩放、不根据测试 SUN 搜系数。经验 KL 不是整个模型部署分布的 KL 保证。

## 6. 最少但决定性的验证，不再展开多臂消融

### 6.1 四个短检查，直接复用正式代码和训练数据

1. **状态可辨识**：同样 masked canvas、不同旧晶格/packing，conditioner 输出必须不同；训练后对压缩与膨胀状态的修改分布应有对应差异。输入可辨识不等于已经学会压力方向。
2. **协同可执行**：构造单独改 cell/单独移原子会失败、联合改变合法的例子；最终事务必须能提交，同时非法终点仍回滚。
3. **采样—评分同分布**：小词表穷举或频率测试覆盖温度、周期别名、顺序分量、空支持和回滚。不需要重跑晶体性能评测。
4. **监督有终态区分度**：正式 1024 组的首 128 组先完成收敛标注，检查低能变化是否在公共充分弛豫后仍存在，以及同组是否实际探索到不同盆地。它们属于 train，不拿来报 SUN，也不替换成更好看的组成。

首 128 组不是新方法筛选赛：若协议未收敛、trace 错位或全回滚，先修这些明确的工程/标签问题；不按能量收益重抽或换组成。若只是收益弱，完成原定同一 1024 组后再判断全池的经验双降可行性。全池 `ρ_max=0` 时不自动做无物理区分度的 post-training；需要报告缺的是探索、可执行动作还是教师可信度，而不是用更多 epoch 掩盖。

### 6.2 主结果只保留一个参考和一个新方法

用固定、同组成、同 program、同随机流的参考与新模型做一次 256 配对生成；保存全部请求与失败。主要输出：

- comp_valid / fast struct_valid，不先算昂贵的全库 Direct 描述符；
- 原生单点 e/F/stress 和尾部分布；
- 无 model494、经过共同评价弛豫后的 Stable/N/U/SUN；
- 同一原生输出经过固定 tau800 的系统结果，单列保底。

如果复用的是之前已看过结果的 256 队列，它仍叫固定 development 队列，不改名为全新 prospective。训练不能使用它的结构或能量；它也不用于重新求 λ、选 checkpoint/seed。正式 1000 队列应在方法冻结后使用尚未用于调整方法的 Plan 来源，优先复用已有未用库存，若不足再明确登记一次新采样；不把 256 的反复开发结果包装成独立主测试。

每个能量比较给成对差值和 wins/losses；Strict 小计数不靠一两个样本宣称显著。无需先完成旧建议的 2×2、两训练 seed、多 τ、额外 controller 消融。

另一个已核实的口径：现有 [full evaluator](../../eval_runtime/run_full_reconstructed_eval.py) 先在输入结构上算 N/U，再取公共弛豫能量判定稳定。因此历史 SUN 是“输入结构 N/U ∩ 弛豫能量稳定”。保留这个可比列，同时在训练 K4 内按 relaxed structure 看 basin 重复；不能把 native N/U 高误写成低能终点彼此不重复。正式论文若改用 relaxed N/U，参考与新模型必须一起重算，不能静默更换定义。

通过首轮后才扩大到独立的 1000 有效 CIF 主采样。若采用有效 CIF 分母，必须按源顺序、只依可解析性继续取样，同时报告总请求和拒绝数；不能根据稳定性、N/U 或能量从后面挑补。

建议论文主表同时给固定请求产率和有效 CIF 条件产率，以免 `10/50` 是分母变化而不是模型变化。已完成的旧结果保留原始来源，不替换成新方法结果。

## 7. 6×A800 上限下的实施顺序、真实训练量与时间预期

资源上限已由用户增至6 A800、每卡4 CPU、最多两个job。训练优先四卡保持全局batch16，另外两卡用于独立数据/评价；生成和物理标注按可用卡数分片，可用全部六卡，不强制偶数成对等待。下表原四卡训练吞吐预算不因为额度提高而自动声称缩短。

| 阶段 | 实施内容 | 初步时间预算，不含排队 |
|---|---|---|
| 工程与 CPU 审计 | state conditioner、联合事务、部署 trace/replay 共用接口；保留旧路径 | 6–12 小时 |
| 接口训练 | full MP20 一轮、新状态条件与联合 mask | 1–3 小时 |
| 首轮终态数据 | 1024×4 完整生成 + 收敛 CHGNet 标注；首 128 组嵌入其中 | 3–8 小时，收敛尾部可能更长 |
| 全路径训练 | 两轮 group pass，包含构造/联合/闭合决策和 CE anchor | 约 2.5–5 小时，仅在实测四卡吞吐满足下述假设时成立 |
| 一次刷新与训练 | 同 train 条件重新生成、终态标注、再完整两轮 | 约 5.5–13 小时，依赖实测 |
| 首轮固定 256 配对结算 | 原生物理量、共同评价 SUN；tau800 可分片并行 | 2–4 小时 |
| 独立 1000 有效采样 | 只在主模型确定后启动 | 另约 4–8 小时，视收敛和 MP 查询缓存 |

完整覆盖的工作量如下，不能再把它写成一次 348-step 小训练：

| 训练部分 | scalar 状态数 | 有效 batch16 对应 optimizer updates |
|---|---:|---:|
| full MP20 warmup，一 source 一状态 | 27136 | 1696 |
| 一采集轮的两遍路径训练：1024×4×6×2 | 49152 | 3072 |
| 两采集轮的路径训练合计 | 98304 | 6144 |
| 每四个路径更新一次 CE anchor | 24576 | 1536 |
| 全部训练合计 | **150016** | **9376** |

表中路径计数为所有四条均验证、有六个抽样决策的完整预算；缺失监督只减少实际计数，不能改变覆盖报告。若 batch 定义改变，按 scalar state 而非 source/组数重算 updates。warmup 与 post-training 的 optimizer/schedule 分开；两个 post-training 采集轮延续同一优化设置，刷新的是数据参考，不从测试选择重新开始。

一采集轮连 CE 约 61440 状态。若四卡合计实测达到 `3.5–7 状态/秒`，该轮约需 `2.4–4.9 小时`，否则按实际吞吐顺延。conditioner 的输入梯度需要经过 DLM，LoRA 参数少并不意味着只做小头反传。

工程与首批 CPU/data 工作可重叠，生成/标注可用两卡+两卡流水，训练则集中四卡，但有依赖的阶段不能虚报全并行。**批准后首轮方向按约 14–26 小时规划；包含一次刷新和 256 结算按约 24–48 小时预留。**此前 20–36 小时只能视为乐观条件估计，首个真实短测试后必须更新，不能承诺硬截止内必过线。

效率措施只改变调度、不改变科学分布：按 N/程序长度批处理；4 卡独立 shard 生成；CHGNet 优先 batch graph inference，共享模型；每卡有限多结构并行，避免每进程重复占满模型或 4 CPU 被 16 个重优化器反复争用；缓存相同输入/同一评价协议的终态。先测实际吞吐再调 worker，不把 worker 数当利用率。

最贵的是收敛终态与真实路径 replay。只保存可重建的 token 动作/提交事件，不保存每步全词表 logits；训练按 inclusion-corrected 状态子采样，不对全部生成步反传，也不穿过 CHGNet 或弛豫器反传。

当前前向会产生全序列全词表 logits。先以冻结 embedding/head、混长分桶、microbatch/accumulation 测真实吞吐；若输出头成为明确瓶颈，再用等价的 active-position projection 优化训练评分，并数值对照完整 head，不改变采样分布。该优化是工程选项，不是先验声称已有或免费。

## 8. 预期：这次为什么更有希望，又不能靠什么保证

| 希望改善的量 | 本方案的直接作用 | 仍可能失败的原因 |
|---|---|---|
| 弛豫能量差 A | 数值状态、full MP20 几何训练与显式 A 平均收益约束 | 教师约束不等于学生约束，A 也不直接等于力/应力 |
| 终态盆地能量 B | 完整路径按 A/B 双目标重加权，且晶格/多原子可以协同改变 | K4 没探索到更好的盆地，有限样本/LoRA 不能吸收目标 |
| raw-path Strict/Meta SUN | 同组成下更低终态能量提升低 hull 的机会 | 阈值附近教师误差、novelty/diversity 损失、组成泛化 |
| 几何有效性 | 继承支持检查、整体回滚、MP20 CE | 新事务高拒绝率虽能守 valid，却可能使稳定性改进为零 |

数学与代码支持“这次优化对象更准确、状态与动作更充分”，不支持事前给出可信的过线概率。

审计分开裁定：**双目标数学规格可接受；新 embedding/whole-transaction/trace 接口尚未验收，工程风险较高；保留旧资产与默认关闭新路径能限制回退成本，但不能把未测试实现宣称为低风险。**目标是用第 6 节的合并短检查把风险尽早暴露，不让完整训练替代接口验收。

量级上，BS raw 从 `3.12%/20.90%` 到 `10%/50%` 不是微小补点，而是 Strict 约 3.2 倍、Meta 约 2.4 倍的产率变化。因此必须让完整路径重新分配盆地概率；只靠微小局部降能或多扫一遍旧数据，理由不够。

最有价值的小成功是：共同充分弛豫后的终态分布变好，且低能路径绝对概率与实际采用率一起上升。只看到 teacher loss 下降、K4 排名改善、原生 force 下降，都不能代替这条证据。

只有共同评价下的终态能量或 Stable 改善支持“低能盆地提议更好”；SUN 单独上升可能只是 N/U 变化。A 下降称“原生—弛豫终态能差减小”，不自动等于 force 更小或步数更少。若原生 e/F/stress 与实际弛豫步数也改善，才进一步支持 DLM 自身逼近平衡结构。完全去掉 model494 是可检验的最终目标，不是本轮提前宣布的事实。

## 9. 论文：保留 LLM，收敛成一个核心贡献与三层解释

核心表述：

> **A scientifically supported, LLM-programmed crystal DLM that learns to reach low-energy basins with a small relaxation burden through state-aware cooperative decoding.**

中文方法目标：**科学约束支持的 LLM 程序化晶体 DLM：以周期状态驱动协同解码，学习低能盆地和低弛豫负担的同一生成程序。**在结果完成前，这是方法目标，不是已取得的结果句。

三层贡献不是并列外挂：

1. **化学支持到执行程序**：保留 C3FD–Llama 学习型决策与 pointer，使合法组成和构造次序成为同一生成程序的输入。现有 comp_valid 与 program 学习证据继续使用，清楚标注对应模型版本。
2. **程序到可修改的周期状态**：DLM 以双向上下文保留未来原子，同时读取旧晶格/packing，协同修改相互依赖的 cell 与多原子。解释为什么不是纯 AR 文本补全，也不是“已全局可见所以不需要数值物理状态”。
3. **物理终点到程序执行概率**：同一次弛豫给出“盆地好不好”和“原生结构离它多远”；经验双目标教师训练同一个 DLM 的构造与修订，不在推理时改结果。连续物理知识由此进入执行模型的参数。

本轮固定的 C3FD 约束 Llama 只输出 composition/soft Plan/species program；选区由编译器与冻结规则完成。一个参考加一个最终方法的主比较支持**整法收益**，不拆分宣称 Llama、conditioner、联合事务分别带来多少 SUN，也不证明 Llama 独立不可替代。Llama 的既有学习证据与新执行记录作为机制证据，版本分别标明。

不要把“能量加权”“小图层”“LLM+diffusion”本身声称为首次。真正需证明的连接是：**同一程序下的状态可辨识 → 协同动作可执行 → 低能完整路径更常被模型直接生成。**

若最终只有前两项证据成立，就保留其方法价值，不把尚未成立的终态收益写成贡献。无需为了论文整齐而把三个不同实验的数值拼成一个模型。

## 10. 已核对的相关工作与适用边界

- [Diffusion Controller](https://arxiv.org/html/2603.06981v1)：提供正则化控制与终态奖励训练的框架；不能据此保证有限 K4 的晶体 DLM 会过线。这里借鉴路径分布目标，不搬用其图像实验结论。
- [CrystalREPA](https://arxiv.org/abs/2605.08960)：研究 MLIP 与晶体生成器表征的差距及训练期物理表征迁移。它支持“物理状态信息值得进入生成器”的方向，但不证明我们需要复制 MLIP 特征或另加一项对齐 loss。本轮不引入该额外支路。
- [SCOPE/D3IM](https://arxiv.org/abs/2606.01026)：指出已提交 token 的 preservation bias，并研究 sampler-matched 自修订训练。支持真实修订状态匹配；数学/代码结果不能直接当作晶体稳定性证据。
- [PackFlow](https://arxiv.org/abs/2602.20140)：在分子晶体中用物理后训练改善提议与低能盆地分布。支持“把昂贵筛选知识摊销进生成概率”的思路，但与无机、离散晶体 token 体系不同。
- [CrysLLMGen](https://arxiv.org/abs/2510.23040)：明确采用 LLM 离散生成与连续 diffusion 精修的互补架构。我们的差异目标是让 LLM 编程的 DLM 自身学到稳定性，不把现成 hybrid 组合重新命名为新贡献。

上述 2026 工作已核实 arXiv 标题与摘要；不把它们未经核实的 venue 说成已发表顶会。附件提到的 CrystalReasoner 未由其给出的 CrysLLMGen 链接证实，本文不以它作为已核验依据。

## 11. 批准后才启用的简明 checklist

- [x] 用户确认采用本修订路线；on-policy 数据仅来自 MP20-train 条件，实测后更新预算，资源增至6A800/24CPU。
- [ ] 固定现有 Planner/program、完整 closure 参考与唯一最终比较协议。
- [ ] 实现状态 conditioner 与联合事务；完成四个短检查的接口部分。
- [ ] full MP20 一轮状态/协同训练，不用 PMTR head-only checkpoint。
- [ ] 1024×K4 全路径采集；首 128 组先结算真实终态区分度，随后完成同一数据集。
- [ ] 对完整经验池求 A/B 双降权重并记录实际可行收益；无可学信号不伪造标签。
- [ ] 全路径双目标训练两轮；一次 train-only 刷新，再完整两轮；正确记 source/pass/scalar/update。
- [ ] 固定 256：原生物理量、公共弛豫 SUN、tau800 保底分别结算。
- [ ] 模型确定后独立 1000 有效采样；保存源顺序、请求分母及失败。
- [ ] 按同一模型证据更新论文主表和三层方法解释。

审计阶段未运行实验；用户现已授权实施和10分钟续行，实际进展以 [04 执行清单](04_EXECUTION_CHECKLIST.md) 为准。此前未跟踪的 PMTR 工作文件原样保留。历史审计发现与处理见 [18 审计决策记录](18_DUAL_OBJECTIVE_REVIEW_AND_DECISIONS.md)，其中等待批准状态为当时快照。
