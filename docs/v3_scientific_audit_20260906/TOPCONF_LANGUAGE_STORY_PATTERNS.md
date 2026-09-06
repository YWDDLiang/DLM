# 三篇 NeurIPS 主会语言路线的论文问题写法

2026-09-06。范围仅 FlowLLM、GenMS、CrysLLMGen；三篇官方页面均明确列为 Main Conference Track。阅读官方主论文的引言、问题定义、方法组织与贡献段，不将未核主会的预印本纳入本轮故事依据，也不在此扩展技术综述。

| 主论文与阅读位置 | 问题 → 缺口 → 机制 | 贡献怎样组织；我们不能重复认领什么 |
|---|---|---|
| **FlowLLM，NeurIPS 2024**。[官方页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/51d317df78eded9eb3c9d3fb1091c279-Abstract-Conference.html)；[主论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/51d317df78eded9eb3c9d3fb1091c279-Paper-Conference.pdf)，§1 pp.1–2、§4 pp.3–6、§5.3 pp.8–9。 | **What：**联合生成原子类型、晶格和坐标。**Why：**离散类型与连续精度难以兼顾。**How：**LLM 产生学习到的初始分布，RFM 学习其到数据分布的输运；训练用组成配对，完整采样由无条件 LLM 请求起始。 | 贡献段是**一个混合方法＋主任务结果＋简化组合消融**，不是三个互不相干的算法。Types-only 另检验组成先验。不能再认领“LLM 初始分布＋连续修正”或把训练中的给定组成误读为主任务 CSP。 |
| **GenMS，NeurIPS 2024**。[官方页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/447d012bd95b6767a4bfdebf96cdfcc9-Abstract-Conference.html)；[主论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/447d012bd95b6767a4bfdebf96cdfcc9-Paper-Conference.pdf)，§1 pp.1–2、§2.1–2.3 pp.3–5。 | **What：**从高层语言自动提出化学式并生成结构。**Why：**用户难以预先给出正确化学式；语义知识与结构数据分散，需求又可能不完整。**How：**分层生成、按高低层目标搜索/筛选，紧凑表示服务采样效率。 | 引言按“观察→对应设计→实验”组织；形式化目标连接语言、结构与性质预测。不能再把“语言→化学式→结构”或“增加性质验证器”单独叫新意。其主任务是语言条件搜索，不能借其叙事替我们证明尚未具备的指令遵从能力；搜索候选成本也要区分。 |
| **CrysLLMGen，NeurIPS 2025**。[官方页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f789a628fca473e922c806657512a20f-Abstract-Conference.html)；[主论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/f789a628fca473e922c806657512a20f-Paper-Conference.pdf)，§1 pp.1–2、§2 p.3、§4.1–4.2 p.4、§4.5 p.7。 | **What：**从数据学习完整晶体分布，生成新的组成与结构，条件生成另列。**Why：**用引言 Table 1 展示两类模型不同的组成/结构有效性短板。**How：**独立训练 LLM 与 diffusion，保留生成的原子类型，将生成几何从中间 τ 接入 denoiser。 | 一个统一混合方法，配主任务结果与条件能力；对 FlowLLM 的差异明确到训练关系和衔接位置。不能再认领“LLM＋diffusion 首次组合”“保持组成后精修”或把未改的 model494 算新方法。 |

以上是原论文内容的概括。下面是对本项目写作的推论。

**共同写法不是“先找三个模块”。** 先用一句话规定生成什么、输入什么和成功标准；再给一个能观察到的缺口；每个机制负责缺口的一部分；最后以完整任务结果和必要消融闭合论证。FlowLLM 与 CrysLLMGen 都说明，一个方法框架可以用互补机制与实证支撑，不必包装成三项新算法。

**我们的 What 必须是 de novo 联合生成。** 输入不提供目标化学式或参考结构，系统自主选择 N、元素/计数，再生成晶格与坐标。几何子模块可以使用组成条件，但这只是联合模型的内部因子。论文主表评估完整请求；固定 C、缓存同一批上游抽样或给定组成的重建，只负责隔离机制，不能替代主任务。

**可保留的具体切入点：** 在明确可完成的化学动作支持内学习提案，并让 DLM 承担实际周期几何生成。H-G 有效时，以共享语言骨干的混合表示实现为候选增量；回到离散路线时，以真正有效的离散条件更新为增量。新意要落在本项目的支持、学习与生成接口，并有 de novo SUN/预算证据；三篇论文已覆盖的总体分工不能再认领。

**证据次序：**完整 de novo 效果与成本 → 组成/几何/精修分解 → 固定组成的配对诊断。参数共享、梯度存在和格式验收只能支撑实现；语言预训练、P 或软条件的因果作用仍须相应对照。三篇各自的指标与筛选协议不相同，本页不搬用其分数给本项目排位。
