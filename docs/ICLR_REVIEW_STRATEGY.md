# ICLR内部红队：问题定义、贡献与故事线

本文件允许使用内部组成实验数据。目标不是包装现有结果，而是判断什么论点能承受
严格reviewer追问。

**故事锁定：**论文方法默认H1-A2。R03只作为personal历史数据审计，不进入主贡献、
主方法或concept-only叙事；最新故事与文献定位分别见`PAPER_STORY_INTERNAL.md`和
`RELATED_WORK_INTERNAL.md`。

**De novo锁定：**主路线推理时从learned Planner采样Plan。训练时从MP-20提取Plan
label是监督；R5C式MP-20 Plan、frozen Plans和user Plans只作为conditional controls，
不能单独支撑fully de novo headline。

## 一、先判断“Novel/Unique一骑绝尘”是否成立

同一套本地CrysLLMGen-compatible指标下：

| 结果 | 分母 | Novel | Unique | N∩U | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|---:|---:|---:|---:|
| CrysLLMGen baseline | 1,000 | 88.70% | 98.90% | 88.10% | 9.00% | 46.10% |
| H1-A2 frozen1000 | 1,000 | 89.20% | 99.70% | 89.00% | 9.40% | 47.40% |
| R03 D2 process-pool raw | 1,024 | 88.28% | 96.88% | 88.28% | 11.43% | 48.44% |
| exact replay H1 all-raw | 1,200 | 87.42% | 96.50% | 87.00% | 8.58% | 46.08% |
| exact replay R03 all-raw | 1,200 | 86.83% | 96.25% | 86.67% | 8.42% | 46.58% |
| continuous1280 H1 pooled raw | 3,840 | 86.88% | 97.16% | 86.69% | 7.63% | 45.47% |
| continuous1280 R03 pooled raw | 3,840 | 86.30% | 96.88% | 86.12% | 7.47% | 45.26% |

结论：

- H1-A2相对本地baseline的N/U/N∩U仅提升约`0.5/0.8/0.9 pp`，不能写
  “一骑绝尘”。
- R03 D2的高Strict点没有对应更高的raw N/U；其4次运行也不是4个独立
  Planner样本。
- exact replay与continuous1280中，N∩U和Strict均低于历史frozen1000点。
- 可守住的说法是“在历史最佳点上保持约89%的高N∩U，同时取得非退化的
  Strict/Meta”，不是“DLM已经证明显著提高N/U”。

## 二、与公开论文的Pareto位置

CrystalDiT使用统一DFT协议报告以下结果；我们的评价器、稳定性来源、样本规模和
分母不同，因此只能用于定位故事，不能直接进行SOTA排名。

| 方法 | UN rate | Strict conversion within UN | SUN | Meta conversion within UN | MSUN |
|---|---:|---:|---:|---:|---:|
| DiffCSP | 87.17% | 4.00% | 3.49% | 23.80% | 20.75% |
| FlowMM | 87.66% | 4.80% | 4.21% | 23.69% | 20.77% |
| MatterGen | 89.89% | 4.07% | 3.66% | 26.90% | 24.18% |
| CrystalDiT | 63.28% | 13.87% | 8.78% | 40.93% | 25.90% |
| H1-A2内部历史点* | 89.00% | 10.56% | 9.40% | 53.26% | 47.40% |
| R03 D2内部描述点* | 88.28% | 12.94% | 11.43% | 54.87% | 48.44% |

`*` H1-A2/R03不是CrystalDiT的DFT协议；内部conversion按`SUN/UN`计算，
hull unknown仍按lower-bound失败处理。这些行不能用于论文中的直接排名。

这个表支持的故事是：H1-A2/R03位于“高UN候选供给、稳定性转化仍是瓶颈”的
区域。它不支持“其他方法都牺牲Novel/Unique”，因为MatterGen的UN为89.89%，
高于H1-A2历史点；新近方法也在专门优化该Pareto前沿。

## 三、推荐的问题定义

> **当不同晶体合法性检查只有在生成出不同前提信息后才能判断时，在前提信息具备时
> 限制违规候选，并决定每个阶段哪些几何变量有资格竞争下一次提交，是否会影响模型
> 提出的composition被可靠实现为周期晶体？**

Scope：composition和atom count固定，研究域为learned source采样的eligible完整
Plans。Main RQ只研究selected support timing与commitment policy。Learned Plan source
定义fully de novo scope；fixed model494只作downstream consequence。

## 四、推荐的三个贡献点

### 贡献1：Constraint-prerequisite问题形式化

不同selected checks需要不同前提信息，因此可判断时机不同；每个阶段哪些几何变量
有资格竞争下一次提交也是一个独立、可证伪的inference决策。

### 贡献2：Core masked executor

Composition-anchored、exact-cardinality typed masked executor，在当前partial state上
施加三项selected support，并支持group-restricted confidence-adaptive与fixed
positional commitment policies的严格对照。

### 贡献3：Plan-level paired empirical analysis

以完整Plan为统计单位，在相同checkpoint、NFE和call-indexed随机流下成对估计
selected support、commitment policy及其interaction，并追踪fixed refiner的pre/post
conversion。严格wiring和结果完成前只能写“we evaluate”。

## 五、为什么这样设计

| 失败机制 | 设计选择 | 证据边界 |
|---|---|---|
| model-sampled condition必须在realization中保持 | composition/N/elements anchors | anchors是任务合同，不是性能贡献 |
| selected checks需要不同前提变量 | partial-state selected support | 只覆盖zero length、opportunistic gamma和discrete PBC duplicate |
| commitment trajectory可能改变上下文和mask机会 | 同checkpoint grouped vs positional policy | 不比较DLM与AR，不声称当前order最优 |
| token坐标有限精度 | fixed model494 continuous refiner | inherited组件，只作downstream conversion |
| 稳定性优化可能造成模式收缩 | 同报UN、stability与SUN | 需要统一评价器Pareto图 |

## 六、严格reviewer会攻击什么

1. **主表与cohort audit必须分开。** `105/1000`与`488/1000`是冻结的未来论文
   S.U.N.主表合同；`103/1200`与`94/1000`属于不同cohort/evaluation views，不能
   被用来替换或重新解释主表。
2. **组成实验被从public repo隐藏。** reviewer无法从公开artifact追溯headline。
3. **H1 frozen1000存在survivor-prefix性质。** 与requested-1000 baseline直接比较可能
   带来分母偏差。
4. **R03 repeats不是独立seed。** 不能作为一般化证据。
5. **exact/continuous复现不支持R03稳定增益。** exact McNemar为`p=1`，连续重复
   pooled差接近0。
6. **没有matched AR body。** 无法把高UN或SUN归因给DLM。
7. **训练级复现不完整。** B0全局训练seed未记录，model494训练seed也未确认。
8. **跨论文评价器不同。** CHGNet/official-MP、DFT、Matbench hull不可直接混排。

## 七、达到ICLR可接受线所需实验

1. 同Plans、同token representation、同refiner、同评价器的AR body vs DLM body；
2. 默认H1-A2至少3个独立Planner seeds，每个requested attempts≥1,000；
3. exact-length、Plan anchors、typed support、duplicate mask、refiner逐项消融；
4. raw-attempt口径下画`UN rate × stability-within-UN` Pareto图；
5. 将Plan seed作为独立统计单元，报告bootstrap CI与paired tests；
6. 报告采样速度、显存和DLM steps，避免只讲质量；
7. 使用与至少一个强基线完全一致的稳定性评价合同；
8. 保留`105/488`主表，同时公开其组成来源与所有cohort audit，禁止把任一审计口径
   静默替换成headline。

## 八、扬长避短的叙事

不要写：

> DLM比所有方法更Novel/Unique，因此必须使用DLM。

建议写：

> Our masked crystal generator operates in a high unique-and-novel regime
> without relying on post-hoc search or reward optimization. Its exact-length
> representation and constraint-aware denoising order make structured body
> generation reliable; the remaining bottleneck is the conversion of this
> diverse candidate supply into strict thermodynamic stability.

中文：

> 我们不是通过筛选或奖励优化把生成分布压向少数稳定模板，而是先用exact-length、
> 约束感知的masked DLM维持高UN候选供给，再用连续refiner完成局部几何修复。
> 当前剩余瓶颈不是候选多样性，而是多样候选到严格热力学稳定性的转化率。

这条故事能突出长处，同时诚实承认stable conversion仍弱；但要把“DLM导致高UN”
升级为因果贡献，必须补matched AR实验。

## 十、两位严格reviewer的收口结论

Public-only reviewer评分：

| 维度 | 1–4分 |
|---|---:|
| Soundness | 1 |
| Significance | 2 |
| Novelty | 2 |
| Clarity | 3 |
| Reproducibility | 1 |

Personal evidence reviewer评分：Novelty `3/4`、Technical quality `2/4`、
Empirical support `1/4`、Reproducibility `1/4`、Clarity/significance `3/4`。
两者均给出当前Reject。

共同结论：

- 高absolute UN、弱strict conversion是可观察现象；
- UN领先、DLM因果优势和R03稳定提升均未被证明；
- Plan-to-realization interface是最强机制点，但仍需独立Plan seeds和matched body对照；
- exact-cardinality、typed completion与factorization有方法价值，但缺决定性消融；
- `105/1000`与`488/1000`继续作为未来主表；其组成来源与exact/historical审计必须
  明确分列。

本项目已决定保留`105/1000`与`488/1000`作为未来S.U.N.主表数据，不再将其降为
deprecated descriptive aggregate。统计检验仍必须使用其对应的原始实验单位，不能
借用exact replay或historical artifact的逐样本记录替代。

## 九、相关工作

- [LLaDA](https://arxiv.org/abs/2502.09992)
- [FlowLLM](https://proceedings.neurips.cc/paper_files/paper/2024/file/51d317df78eded9eb3c9d3fb1091c279-Paper-Conference.pdf)
- [CrystalDiT](https://ojs.aaai.org/index.php/AAAI/article/download/37121/41083)
- [Crys-JEPA](https://arxiv.org/abs/2605.14759)
- [CrysLLMGen](https://arxiv.org/abs/2510.23040)
