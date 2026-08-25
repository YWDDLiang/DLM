# ICLR内部红队：问题定义、贡献与故事线

本文件允许使用内部组成实验数据。目标不是包装现有结果，而是判断什么论点能承受
严格reviewer追问。

**故事锁定：**论文方法默认H1-A2。R03只作为personal历史数据审计，不进入主贡献、
主方法或concept-only叙事；最新故事与文献定位分别见`PAPER_STORY_INTERNAL.md`和
`RELATED_WORK_INTERNAL.md`。故事到实验的完整映射见
`EXPERIMENT_PRIORITIES_INTERNAL.md`。

**De novo锁定：**主路线推理时从learned Planner采样Plan。训练时从MP-20提取Plan
label是监督；R5C式MP-20 Plan、frozen Plans和user Plans只作为conditional controls，
不能单独支撑fully de novo headline。

## 一、先判断“Novel/Unique一骑绝尘”是否成立

同一套本地CrysLLMGen-compatible指标下：

| 结果 | 分母 | Novel | Unique | N∩U | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|---:|---:|---:|---:|
| CrysLLMGen baseline | 1,000 | 88.70% | 98.90% | 88.10% | 9.00% | 46.10% |
| H1-A2 historical frozen1000 compatibility view | 1,000 | 89.20% | 99.70% | 89.00% | 9.40% | 47.40% |
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

> **在生成式材料发现中，发现产率的提升，在多大程度上来自所探索材料规格分布的
> 改变，又在多大程度上来自给定已探索规格后的结构实现能力提升？**

英文冻结版：

> **In generative materials discovery, to what extent do gains in discovery
> yield arise from changing the distribution of material specifications being
> explored, versus improving structural realization conditional on an
> explored specification?**

Main RQ不包含H1-A2，也不把晶体证据外推到其他领域。晶体实例问题是：

> **For de novo crystal generation, can composition-anchored masked completion
> improve structural realization across model-sampled chemistries beyond gains
> explained by measured changes in the proposed-chemistry distribution,
> without collapsing cohort-level diversity?**

H1-A2的method hypothesis是：Planner采样chemistry，anchors固定
composition/N，DLM生成`6+3N` geometry tokens，fixed model494只修geometry。旧的
selected support timing与commitment policy问题降为Mechanism RQ。

## 四、推荐的三个贡献点

### 贡献1：Proposal-versus-realization问题形式化

Aggregate discovery yield混合了explored-specification distribution与
specification-conditioned structural realization。论文在晶体中用composition/N实例化
该问题，并将uniqueness作为cohort-level outcome。

### 贡献2：Core masked executor

Composition-anchored、exact-cardinality typed masked executor，在当前partial state上
施加三项selected support，并支持group-restricted confidence-adaptive与fixed
positional commitment policies的严格对照。

### 贡献3：Distribution and mechanism attribution

完整报告化学分布与stagewise conversion，进行common-mix standardization；再以
fixed-condition paired mechanism和fixed-refiner pre/post conversion区分系统广泛性、
DLM execution与continuous refinement。结果完成前只能写“we evaluate”。

## 五、为什么这样设计

| 失败机制 | 设计选择 | 证据边界 |
|---|---|---|
| aggregate gain混合selection与realization | 全化学分布、common-mix标准化、accounting decomposition | 只在预注册measured strata与overlapping support内解释，不称causal mediation |
| model-sampled condition必须在realization中保持 | composition/N/elements anchors | anchors是任务合同，不是性能贡献 |
| selected checks需要不同前提变量 | partial-state selected support | 只覆盖zero length、opportunistic gamma和discrete PBC duplicate |
| commitment trajectory可能改变上下文和mask机会 | 同checkpoint grouped vs positional policy | 不比较DLM与AR，不声称当前order最优 |
| token坐标有限精度 | fixed model494 continuous refiner | inherited组件，只作downstream conversion |
| 稳定性优化可能造成模式收缩 | 同报UN、stability与SUN | 需要统一评价器Pareto图 |

## 六、严格reviewer会攻击什么

首要识别攻击：即使oxide、halide、arity和每个N-bin都提升，Planner仍可能在每个
粗类别内部选择更容易的exact formulas；最终结果也可能主要由model494修复。完整
类别表只能证明广泛性，必须结合common-mix standardization、fixed-condition mechanism
和pre/post-refiner conversion。

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

## 七、达到ICLR可接受线所需证据

当前优先级不要求新增训练：

1. 冻结一组可逐attempt追踪formula/Plan/outcome的标准H1-A2 analysis cohort；若
   `105/488`只是aggregate，保留headline但不制造伪microdata；
2. 完整展示预先固定的compound family、arity、N-bin和element distribution；
3. 对每类报告attempts、Direct comp/struct/joint、reconstructed、hull known/unknown、
   Strict/Meta stability、novel及stable-and-novel conversion；
4. 报告common-support coverage、composition-standardized difference和对称
   selection/within-stratum accounting；
5. 用现有或最小fixed-condition paired evidence连接到selected support与commitment
   mechanism；
6. 报告pre/post-refiner conversion，排除最终收益完全来自model494；
7. 在固定cohort size上重新计算Uniqueness和S.U.N.，不能将U当作per-body Bernoulli；
8. 保留`105/488`主表，同时公开所有负向或insufficient-support strata。

当前crystal RQ点名masked completion，因此matched AR executor是强DLM论文的必要实验；
更多独立Planner seeds和其他training-level ablation仍是增强项。已有cohort可先完成
proposal distribution与coarse-strata residual accounting，但不能单独完成
exact-specification或masked-architecture attribution。

## 八、扬长避短的叙事

不要写：

> DLM比所有方法更Novel/Unique，因此必须使用DLM。

建议写：

> Aggregate generative-materials yield conflates which material specifications
> are explored with how reliably an explored specification is structurally
> realized. In crystals, H1-A2 exposes this boundary by proposing chemistry
> upstream and anchoring composition/cardinality during body generation.

中文：

> 我们不把proposal reweighting视为错误，而是要求区分高分来自探索了什么
> specification，还是来自给定specification后的结构实现。H1-A2让Planner保留de novo
> chemistry proposal，同时在body阶段固定composition/N，因此可以分别审计proposal
> distribution、coarse-strata residual和fixed-condition mechanism。

这条故事能突出长处而不攻击reward、symmetry或search方法。Fixed-condition evidence
只能归因selected support/policy；要把系统级差异归于masked architecture，必须有matched
executor实验。

## 九、Reviewer收口结论

以下是旧public-artifact readiness review，不等于最新concept-only故事评分：

| 维度 | 1–4分 |
|---|---:|
| Soundness | 1 |
| Significance | 2 |
| Novelty | 2 |
| Clarity | 3 |
| Reproducibility | 1 |

旧personal evidence reviewer评分：Novelty `3/4`、Technical quality `2/4`、
Empirical support `1/4`、Reproducibility `1/4`、Clarity/significance `3/4`。
两者均对当时的evidence/artifact readiness给出Reject。

最新Proposal-versus-Realization Proposer–Reviewer–Arbiter裁决为：

> **Concept-only APPROVED，约7/10。**

最新共同结论：

- Main RQ应研究aggregate gain来自attempted-chemistry selection还是conditional
  realization；
- 全化合物分布与条件稳定率是广泛性证据，common-mix standardization才是主要
  anti-shortcut证据；
- fixed-condition mechanism与pre/post-refiner负责进一步归因，不能被粗类别表替代；
- 旧support×commitment问题保留为Mechanism RQ；
- `105/1000`与`488/1000`继续作为未来主表；其组成来源与exact/historical审计必须
  明确分列。

本项目已决定保留`105/1000`与`488/1000`作为未来S.U.N.主表数据，不再将其降为
deprecated descriptive aggregate。统计检验仍必须使用其对应的原始实验单位，不能
借用exact replay或historical artifact的逐样本记录替代。

## 十、相关工作

- [LLaDA](https://arxiv.org/abs/2502.09992)
- [FlowLLM](https://proceedings.neurips.cc/paper_files/paper/2024/file/51d317df78eded9eb3c9d3fb1091c279-Paper-Conference.pdf)
- [CrystalDiT](https://ojs.aaai.org/index.php/AAAI/article/download/37121/41083)
- [Crys-JEPA](https://arxiv.org/abs/2605.14759)
- [CrysLLMGen](https://arxiv.org/abs/2510.23040)
