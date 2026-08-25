# H1-A2严格Proposer–Reviewer最终裁决

## Disposition

> **APPROVED，concept约7/10。**

最终批准的论文主线是**Proposal versus Realization**。原constraint timing与
commitment问题保留为Mechanism RQ，不再承担论文级科学动机。批准不等于预先批准
正向结果。

## 冻结Main Scientific RQ

> **In generative materials discovery, to what extent do gains in discovery
> yield arise from changing the distribution of material specifications being
> explored, versus improving structural realization conditional on an
> explored specification?**

中文：

> **在生成式材料发现中，发现产率的提升，在多大程度上来自所探索材料规格分布的
> 改变，又在多大程度上来自给定已探索规格后的结构实现能力提升？**

该领域级问题当前只在de novo inorganic crystals上实证。晶体实例是：

> **For de novo crystal generation, can composition-anchored masked completion
> improve structural realization across model-sampled chemistries beyond gains
> explained by measured changes in the proposed-chemistry distribution,
> without collapsing cohort-level diversity?**

在H1-A2中，explored specification是body生成前确定的composition、`N`和element
multiset。改变proposal distribution不是错误；科学问题是聚合提升究竟由哪部分解释。

## 冻结H1-A2 Method Hypothesis

> **Given a model-proposed composition and atom count, anchoring chemical
> identity and cardinality while using masked discrete completion for periodic
> geometry, followed by identity-preserving continuous refinement, defines a
> system in which we test for a positive standardized within-stratum outcome
> difference and retained cohort-level uniqueness. Causal attribution to the
> masked architecture requires a matched executor.**

真实实现链：

```text
Planner采composition/N
→ anchors固定N和element slots
→ Crystal DLM生成6+3N个geometry tokens
→ model494保持N/atom types，只修continuous geometry
```

## 冻结Mechanism RQ

> **At fixed composition and atom count, do prerequisite-aware restrictions on
> selected invalid token choices and a dependency-aware commitment policy
> improve discrete periodic-body realization and downstream conversion under
> an unchanged identity-preserving refiner?**

必须保持边界：selected checks、每次提交一个位置、soft rich-Plan fields不硬执行、
不声称DLM普遍优于AR。

## 为什么通过严格review

- 研究的是generative discovery收益来源，不是一个解码技巧；
- Main RQ不包含H1-A2或正向答案；
- Planner/anchors/DLM/refiner自然对应selection与realization的可分接口；
- 可以由完整化学分布、标准化转化、fixed-condition机制和pre/post-refiner证伪；
- 将uniqueness正确处理为cohort-level而非per-body标签；
- 对reward、symmetry或search方法不作“作弊”指控。

## Concept评分

| 维度 | 分数 |
|---|---:|
| 科学问题重要性 | 8/10 |
| 可证伪性 | 8/10 |
| 与H1-A2匹配度 | 8/10 |
| 方法新颖性 | 6.5/10 |
| 综合concept | **7/10** |

只有粗类别点估计表而无标准化与归因时，故事约`6/10`。

## 冻结贡献层级

1. **科学与评价贡献**：区分explored material-specification distribution与
   specification-conditioned structural realization，并在晶体中以composition/N实例化。
2. **方法贡献**：composition-anchored exact-cardinality typed masked executor，
   配合selected partial-state support、explicit commitment policy和
   identity-preserving continuous refinement。
3. **证据与归因贡献**：完整化学分布和stagewise conversion、common-mix
   standardization、fixed-condition mechanism及pre/post-refiner conversion。

Contribution 3在结果完成前只能写成framework或“we evaluate”。

## “几乎每类都提升”的证据身份

它首先是主文核心的：

> **broadness evidence across preregistered chemical strata**

五种证据不能互相替代：

```text
全化合物分布＋每类条件稳定率
→ broadness evidence

common-mix标准化＋proposal/within-stratum residual accounting
→ measured-strata anti-shortcut evidence；不是exact-specification attribution

matched AR-versus-DLM executor
→ scoped masked-architecture attribution

fixed-condition paired mechanism
→ selected-support/policy attribution under one masked checkpoint

pre/post-refiner＋all-request funnel
→ conditional refiner conversion and aggregate lower-bound context
```

最终必须写成：在有充分共同支持的`K`个预注册层中，有`x/K`个层为正，覆盖参考
分布`y%`；另报supported、inconclusive和negative层。不能先写“几乎全部提升”再找分类。

## 最强拒稿理由

> 即使oxide、halide、arity和各N-bin都提升，Planner仍可能在每个粗类别内部选择更
> 容易的exact formulas；最终收益也可能主要由model494产生。Broad tables alone do
> not identify composition-conditioned realization or the DLM mechanism.

防线不是写作，而是common-support standardization、explicit scope、matched executor、
fixed-condition mechanism和pre/post-refiner attribution。

## Claims lock

允许在证据满足后写：

- aggregate S.U.N.混合了attempted chemistry与conditional realization；
- 在预注册、共同支持的化学层上，H1-A2保留正的standardized within-stratum residual；
- 提升分布在多数充分支持的层，而非集中于单一高成功率family；
- measured chemical-distribution reweighting不能解释大部分observed gain；
- scoped matched executor支持masked completion comparison；
- fixed-condition selected support和commitment policy影响同refiner下的conversion。

禁止：

- “其他方法投机取巧”或只会选easy chemistry；
- 所有compound types或全部chemical space统一提升；
- accounting decomposition是causal mediation；
- 全部提升由DLM造成；
- 超出matched scope的DLM普遍优于AR；
- rich Plan soft fields被硬执行；
- selected checks保证全局合法；
- refiner读取Plan或改变composition；
- Planner backbone或model494属于算法创新。

## Evidence readiness

保留未来主表`105/1000` Strict与`488/1000` Meta。它们是aggregate headline，
在comparator/evaluator合同冻结前不单独证明competitiveness，也不回答Main RQ。当前需
补齐：

- 预先固定的compound family、arity、N-bin和元素分布；
- 每层完整漏斗、denominator、risk difference和uncertainty；
- common-support coverage及composition-standardized accounting；
- fixed-condition mechanism evidence；
- pre/post-refiner conversion；
- cohort-level uniqueness/S.U.N.重算；
- 可逐attempt追踪的raw analysis cohort和冻结comparator/evaluator合同；
- full-Plan versus anchors-only conditioning-scope ablation；
- 当前crystal RQ所需的matched executor。

这些主要是现有结果整理与最小机制归因，不要求重新训练checkpoints。

## 与旧故事的关系

旧的“Serialization Is Not Commitment Order”保留为Methods insight。旧Main RQ降为：

```text
selected support timing × commitment policy
```

新Main RQ研究：

```text
which material specifications are explored
vs
how well an explored specification is structurally realized
```
