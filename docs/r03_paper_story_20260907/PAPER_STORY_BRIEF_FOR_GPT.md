# Plan-conditioned crystal generation with physical feedback

这份 brief 用来帮助 GPT 按照当前真实的模型、数据和实验设置撰写论文。写作时应区分已经有直接证据的事实、正在验证的方法假设和暂时不能声称的结果。不要把训练 loss、局部 token 偏好或教师结构的质量写成学生模型已经获得的物理性能。

## 一句话定位

本文研究一个更严格的目标：在给定同一个材料 Plan／prompt 的条件下，如何让离散晶体生成模型的**完整生成—精修—物理评价链路**更倾向于产生可靠的 SUN（严格稳定且满足 N/U 条件）结构，而不是只提高 token 合法率或局部结构似然。

## 一、科学问题

MP20 监督训练可以让 DLM 学会从 rich Plan 生成固定长度的晶体结构 token，但“结构语法正确”并不等于“结构经过下游精修后具有物理稳定性”。因此核心问题不是模型能否生成一个合法的 `7+4N` body，而是：

> 对同一个 Plan／prompt，在保持组成、样本身份和采样条件一致的情况下，物理反馈能否被转化为 DLM 的生成偏好，使完整链路产生更多可靠 SUN 结构？

这个问题包含三个可检验的子问题：

1. 生成模型是否能够从同一 Plan 下的物理优劣比较中学习，而不破坏原有的几何可行性和组成约束？
2. 固定的连续精修器和 CHGNet 物理评价能否提供足够可靠、可绑定到具体 prompt 的训练信号？
3. 性能变化来自真正的物理偏好学习，还是来自 Plan 分布、随机种子、几何支持、后端精修或样本曝光量的变化？

## 二、背景和基线

原始 H1A2 有两个相邻但不同的训练阶段：

- Planner 从 MP20 学习生成七行 rich Plan；
- Body DLM（B0/R5-C）在对应 Plan prompt 下学习 MP20 的真实 `7+4N` 晶体 body，采用 answer-only masked CE。

这条基线确实建立了“Plan → structure”的条件生成能力，但它没有使用同一 Plan 下的 raw 候选比较，也没有把 F800、CHGNet、凸包稳定性或 SUN 标签回流到 DLM 的训练目标中。固定的 model494 F800 在本工作中作为下游精修器保留，不重新训练。

因此本文不是声称重新发明 Plan-conditioned crystal generation，而是把问题从“复现 MP20 teacher structure”推进到“在固定 Plan 条件下优化完整物理端点”。

## 三、科学方法

### 1. 固定条件的 Plan cohort

使用固定的 256 条 TRAIN Plan。每条请求保存完整 Plan、对应 prompt、组成、原子数、G 生成 seed 和 F 精修 seed。所有轮次沿用同一请求顺序和同一 seed ledger，确保更新前后是 paired comparison。

### 2. 完整的 G→F→physical evaluation 链路

每个请求按以下流程执行：

```text
Plan / prompt
    ↓
G: discrete body generation (7+4N tokens)
    ↓
hard geometry and composition checks
    ↓
F: frozen model494, fixed 800-step refinement
    ↓
tokenized F endpoint
    ↓
CHGNet / hull / N-U protocol
    ↓
SUN, strict Stable, Meta Stable, ordinary improvement, unstable, unknown
```

G 的生成顺序遵守原生 lattice → grouped X → grouped Y → grouped Z 语义分组，并保留周期几何支持。非法结构、几何构造失败、物理未收敛和 hull unknown 分开记录，不能把 unknown 自动标为负例。

### 3. 物理等级和同条件偏好

训练排序固定为：

```text
SUN > strict Stable > Meta Stable > ordinary improvement > unstable
```

只有同一个 Plan／相同条件下的两个端点才形成 G 或 E 的偏好关系。普通 improvement 需要超过固定的 `0.01 eV/atom` 工程分辨阈值；未知、未完成和不可比较的端点保留为 unknown，不能强行制造 DPO 胜负。

G 使用同一 prompt 下的 raw 候选及其后续 F/物理评价构造监督。E 使用本轮 token-F current state 学习 KEEP、局部或更大范围 EDIT，以及对具体候选的接受／拒绝。CHGNet 是冻结的物理教师和评价器，不替学生从多个候选中直接挑最终输出。

### 4. 受控更新和资源设置

第一次 G1 试验暴露了小数据高重复和过强更新的风险：185 条训练记录中只有 3 条 strict Stable，却因按等级抽样而被过度重复；695 步更新也没有由完整数据遍历次数约束。修复设置包括：

- 按打乱后的完整数据 pass 限制样本曝光，取消按等级独立过采样；
- G 显式携带完整 Plan，并按实际生成语义分组形成条件视图；
- 对更新前策略施加条件 KL 约束，并降低 G 学习率；
- 使用真实多样本 batch 和独立 GPU 并行，提高吞吐但不偷偷增加总样本曝光；
- 用同一 Plan／seed 的完整 G→F→物理链路验收更新，而不是用局部 loss 作为最终指标。

## 四、实验设计

### 基线与更新

- `S0`：固定 B0／E0 的初始端点和完整物理评价。
- `G1`：由 S0 同条件物理反馈构造的 G 更新。
- `E1`：在 G1 生成的 current token-F 状态上收集编辑反馈并更新 E。
- 后续轮次继续使用同一 256 Plan 和同一评测协议，直到预定轮次或出现明确的严重退化。

### 主要指标

报告必须同时给出完整 256 分母上的：

- generation success / geometry construction failure；
- `Struct_valid` 和 `comp_valid`；
- reliable SUN；
- strict Stable、Meta Stable；
- known physical failure；
- generation failure、not-converged、hull unknown 等 unknown 分解；
- S0→Sr 的逐请求 SUN／稳定等级转移；
- GPU 时间、实际 batch、吞吐量和峰值显存。

只报告已完成且绑定当前输入的 stage。不能用成功结构子集的百分比替代固定 256 分母，也不能把教师候选的质量当作学生输出质量。

## 五、论文贡献点的推荐写法

以下三点是最稳妥的主贡献表述：

### 贡献一：重新定义评价目标

提出以“同一 Plan 条件下的完整生成—精修—物理端点质量”为评价对象，把 token 合法性、几何可行性和最终稳定性分开测量。这样可以识别出生成模型在语法正确但物理端点失败的具体位置。

### 贡献二：Plan-preserving physical-feedback framework

建立一个保持 Plan、组成、请求身份和随机条件不变的闭环框架：DLM 生成 raw，固定 F 产生可比的 refined/token-F 端点，冻结 CHGNet 提供分级物理反馈，再将同 Plan 的可靠偏好用于 G/E 更新。该框架将物理反馈绑定到正确的 prompt，而不是把不同材料的好结构混合作为通用正例。

### 贡献三：可复现的失败分解和更新控制协议

提出完整分母、paired seed、unknown 分解、几何支持审计和参数更新凭据的评测协议，并记录生成失败、物理失败和稳定性退化的不同来源。该协议可以区分“模型确实学到物理偏好”和“更新破坏了生成可行性”这两类现象。

如果修复后的正式重跑确实改善可靠 SUN，可以再加入条件性贡献：

> 在固定 Plan、固定 seed 和固定下游物理协议下，受控的物理反馈更新提高了 reliable SUN 或严格稳定率。

如果重跑仍然没有改善，应把论文重点放在“为什么局部偏好学习不能直接转化为完整物理端点提升”以及失败机制诊断上，不要声称提出了有效的稳定性提升方法。

## 六、目前已知的负面证据必须保留

第一次 G1 更新不是成功结果，而是重要的诊断实验：

- generation success 从 `229/256` 降到 `157/256`；
- 99 条失败全部发生在 Z 补全阶段的 `pbc_no_legal_completion`；
- G+F reliable SUN 从 `27/256` 降到 `18/256`；
- 原有 27 条 SUN 中保留 15 条、丢失 12 条，并新增 3 条；
- X/Y 同列原子聚集增加，与 Z 补全失败相吻合，但这仍是机制证据，不应写成未经干预验证的唯一因果解释。

这组结果支持“物理偏好反馈不能只靠局部 token loss 和长时间重复更新实现”的判断，也说明完整链路验收是必要的。

## 七、建议论文结构

1. Introduction：从 Plan-conditioned generation 的语法目标转向物理端点目标。
2. Problem formulation：定义 Plan、raw、F、token-F、物理等级和 paired comparison。
3. Method：G 生成、固定 F、CHGNet 标注、G/E 偏好和 unknown 处理。
4. Experimental protocol：固定 256 Plan、seed ledger、完整分母、资源和复现协议。
5. Results：S0 与各轮完整指标、逐请求转移、生成与物理失败分解。
6. Failure analysis：G1 的 Z 补全失败、XY 聚集、样本曝光和 surrogate mismatch。
7. Limitations：MP20 cohort 覆盖有限、CHGNet／hull 不是 DFT 真值、局部条件目标不是完整扩散似然。
8. Conclusion：物理反馈闭环的可行性、边界和下一步。

## 八、可直接交给 GPT 的写作提示词

```text
请根据下面的研究 brief 撰写论文的 Introduction、Problem Formulation、Method Overview 和 Contributions。语气应像机器学习与计算材料交叉领域论文，避免宣传性语言。

核心科学问题是：在固定材料 Plan/prompt、组成、请求身份和随机种子的条件下，能否通过 frozen refinement 与 CHGNet 的物理反馈，提升离散晶体 DLM 产生可靠 SUN 结构的概率？

请明确区分：
1. 原始 H1A2 MP20 teacher SFT（Plan→7+4N structure token）；
2. 当前的物理反馈闭环（G→F→CHGNet→G/E preference update）；
3. 生成语法合法、几何可行和物理稳定三个不同层次。

方法描述必须包括：固定 256 Plan cohort、同一 prompt/seed paired evaluation、G raw generation、frozen model494 F800、token-F、CHGNet/hull 分级标签、SUN > strict Stable > Meta Stable > ordinary improvement > unstable、unknown 不作为负例、G/E 更新和完整分母审计。

请把第一次 G1 退化作为诊断证据而不是隐藏掉：generation success 229/256→157/256，G+F reliable SUN 27/256→18/256，99 个失败集中在 Z 轴周期几何补全。解释这说明局部 token 偏好和完整物理端点之间存在 gap，但不要把样本过采样或 XY 聚集写成已经完成因果证明的结论。

贡献点请写成三类：
（1）物理端点导向的问题定义；
（2）保持 Plan 条件一致的 G→F→物理反馈框架；
（3）可复现的 paired、unknown-aware、failure-decomposed 评测协议。

如果没有正式重跑证据，不要声称 reliable SUN 已经提升；把性能提升写成待验证假设。不要编造 DFT 结论，也不要把 CHGNet 教师结构质量写成学生模型质量。
```
