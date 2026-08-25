# Proposal–Realization故事与实验优先级

## 冻结问题层级

领域级Main RQ：

> **In generative materials discovery, to what extent do gains in discovery
> yield arise from changing the distribution of material specifications being
> explored, versus improving structural realization conditional on an
> explored specification?**

晶体实例：

> **For de novo crystal generation, can composition-anchored masked completion
> improve structural realization across model-sampled chemistries beyond gains
> explained by measured changes in the proposed-chemistry distribution,
> without collapsing cohort-level diversity?**

Mechanism RQ：固定composition/N后，selected prerequisite-aware support和explicit
commitment policy是否提高body realization及同refiner下的conversion。

`105/1000` Strict与`488/1000` Meta继续是未来主表headline，不被本文件的任何审计
视图替换。

## 故事如何落到实验

```text
领域问题
proposal distribution vs conditional realization
        ↓
晶体实例
model-sampled composition/N vs periodic structure
        ↓
H1-A2接口
Planner → anchors → masked DLM → fixed refiner
        ↓
证据链
breadth → standardization → matched executor → mechanism → refiner attribution
```

五类证据身份固定为：

| 证据 | 回答什么 | 不能替代什么 |
|---|---|---|
| 全化合物分布＋within-stratum funnel | 提升是否覆盖广泛chemistry | common-mix anti-shortcut、DLM因果 |
| common-mix standardization＋accounting | measured proposal reweighting解释多少aggregate gap | exact-formula conditioning、causal mediation |
| matched AR-versus-DLM executor | scoped masked-architecture attribution | 普遍DLM superiority、support/policy拆分 |
| fixed-condition paired mechanism | support/policy在同checkpoint下是否有效 | DLM>AR |
| pre/post-refiner conversion＋all-request funnel | proposal差异是否被refiner保留、修复或抹除 | 未进入refiner的反事实结果 |

## 需要同步调整的论文内容

1. **术语层级：**领域级使用`material specification`与`structural realization`；晶体
   实例再落到composition、`N`和element multiset。不要把soft SG/volume/lattice fields
   写成被严格实现的specification。
2. **Planner身份：**Planner是proposal-distribution model和fully de novo来源，不是
   主要算法贡献；其分布、记忆和attrition必须成为可观察对象。
3. **DLM身份：**DLM是composition-anchored realization mechanism，不联合生成
   chemistry，不预设优于AR。
4. **Refiner身份：**model494是固定、继承的conversion stage；主文必须报告它保留、
   修复和破坏了什么。
5. **结果组织：**从单一leaderboard改成`headline funnel + proposal distribution +
   standardized realization + mechanism/refiner attribution`。
6. **比较语言：**不写竞争方法“投机取巧”；只写aggregate metric无法识别proposal
   reweighting与conditional realization。
7. **实证外推：**Main RQ可以面向generative materials discovery，但结果和结论只验证
   de novo crystals。
8. **主方法锁定：**public默认标准H1-A2；不把R03或Safe-axis写入主故事。R5-C只作
   Gold-Plan conditional reference。

## 第一优先级：只分析现有输出，不新增生成

### 0. 冻结protocol与estimands

在看结果前固定：

- primary comparator及其逐attempt数据可用性；
- requested denominator、retry/replacement/refiner合同；
- Strict/Meta阈值、novelty reference和StructureMatcher设置；
- 一个primary additive per-request endpoint；
- 一个primary reference mix、common-support与trimming规则；
- pooled mix采用equal-method还是sample-size weighting，并传播reference-mix估计不确定性；
- hull unknown/ineligible/non-overlap的处理；
- fixed-condition panel的risk-difference estimands、SESOI、power和equivalence rule；
- `without diversity collapse`的cohort-level non-inferiority margin与CI判据。

`Gain`在这些合同冻结前只是待估计量，不是已定义结果。

### 1. 先冻结可逐attempt追踪的analysis cohort

`105/1000`与`488/1000`继续保留为headline，但如果它们是normalized/rounded aggregate
而不是一组具有1:1原始记录的cohort，就不能给每个“headline sample”事后分配formula、
family或outcome。

Proposal–realization分析必须明确使用哪一个raw per-attempt cohort。优先使用现有标准
H1-A2原始cohort并单独标注；若没有与论文主方法一致且完整的raw cohort，则最值得补的
大样本实验不是新的ablation，而是一组fresh standard-H1-A2 requested-1000 analysis
cohort。禁止用不同cohort的逐样本记录拼出105/488的伪microdata。

### 2. 全attempt proposal distribution

对H1-A2主cohort和冻结baseline，从requested attempt开始统计，而不是只看最终成功样本：

- mutually exclusive compound family；
- unary/binary/ternary/quaternary+；
- `N` bins；
- element presence和atom-weighted frequency；
- parse/ineligible作为显式failure mass；
- requested→body→reconstructed→stable各stage的TVD/JSD与survival drift。

Baseline也必须具有逐attempt composition和failure records；只有最终成功结构时，只能做
survivor-distribution context，不能完成proposal-distribution decomposition。

Primary strata在看结果前固定为：

```text
compound family × arity × N-bin
```

Exact element set、training frequency、nearest-training-composition distance、
charge plausibility和pretreatment Plan fields只作敏感性分析。Final symmetry、energy或
最终geometry属于post-treatment，不用于主标准化。

### 3. 每类完整realization funnel

每个充分支持的stratum同时报告：

- attempts与eligible；
- body success；
- Direct composition valid、structure valid、joint valid；
- refined/reconstructed；
- hull known、unknown；
- Strict/Meta stable yield与stable among known；
- novel与stable-and-novel；
- absolute risk difference和uncertainty。

“几乎每类提升”最终必须写成`x/K`个充分支持层为正、覆盖参考分布`y%`，并完整列出
positive、inconclusive和negative strata。

### 4. Common-mix标准化与accounting

只设一个primary reference mix，避免多目标选择：

1. primary：pooled overlapping mix；
2. sensitivity：frozen baseline mix；
3. sensitivity：MP-20 held-out mix。

每个mix报告common-support coverage、effective sample size、最大权重、trimming及排除
的概率质量。在同一个目标总体上计算standardized gap和对称proposal-distribution/
within-stratum residual accounting。Full-population raw gap不能直接减去trimmed-overlap
standardized gap；ineligible、unknown与non-overlap mass必须单独列出。只称descriptive
accounting，不称causal mediation。

粗strata结果只能支持`within measured coarse chemical support`；同一exact formula的
结论只能由fixed composition/Plan配对给出。

线性accounting只用于可加的per-request endpoints，例如body success、Direct joint、
stable all-request yield、novel或stable-and-novel。`Stable among known`受method-dependent
missingness影响，只与hull unknown/coverage并列报告，不单独承担stability claim。
Uniqueness与完整S.U.N.禁止进入该线性分解。

“大部分gap不由proposal reweighting解释”必须事前定义判据并报告joint uncertainty；
若两个components异号，不使用简单百分比归因。

### 5. Cohort-level diversity

Uniqueness不是per-body Bernoulli。固定cohort size并在目标mix下进行stratified
subsampling，基于StructureMatcher equivalence clusters重新计算：

- Unique representatives；
- Novel；
- `N∩U`；
- Strict/Meta S.U.N.。

避免普通有放回record bootstrap制造人工重复；优先Plan-cluster jackknife、paired
Plan-label permutation或基于equivalence clusters的加权估计。

该步骤只比较固定size/mix下的非线性cohort function，不将S.U.N.拆成线性proposal与
realization components。

### 6. DLM→refiner attribution

优先复用已有pre/post tensors，报告：

- `N`与ordered atom types invariant rate；
- lattice/coordinate displacement；
- Direct validity pre/post；
- valid→valid、invalid→valid、valid→invalid；
- known-both energy/hull变化；
- proposal-stage gap在refinement后保留、衰减或反转。

必须与all-request funnel并列，避免successful-body survivor conditioning被误写成final
aggregate refiner effect。

### 7. Planner/Plan记忆与难度审计

无需训练，使用现有parsed Plans报告：

- formula collision与完整Plan collision against MP-20 train；
- exact element-set与nearest-composition距离；
- family/arity/N以及soft field边际；
- Plan entropy、重复率与parse attrition；
- 由训练稀疏度或冻结baseline事先定义的difficulty bins中的realization。

Difficulty不得由H1-A2最终稳定结果反向定义。

### 8. R5-C的最小融合

R5-C只作为**Gold/oracle-Plan-conditioned same-executor reference**：

- 使用同DLM、同refiner、同evaluator；
- 回答可靠specification给定时executor/refiner能达到什么；
- 与learned Plans并列展示proposal difficulty和realization conversion；
- 不进入fully de novo headline；
- 不把历史adjusted指标与现代H1-A2裸差解释成Planner causal effect。

若其soft SG/volume/lattice fields由held-out target structure导出，正文名称进一步写成
`oracle-rich-Plan-conditioned reference`。Learned/Gold chemistry mix不同时只能分别报告
条件结果或先标准化，不能裸比yield。

## 第二优先级：最小新增推理，不新增训练

### 推荐的fixed-condition mechanism panel

使用public quick cohort规模即可形成一个最小、清楚的机制实验：

```text
256 frozen eligible learned Plans
× 1 paired attempt seed per Plan
× 4 cells
= 1,024 body attempts
```

四个cells为：

```text
selected support on/off
×
grouped confidence-adaptive / fixed positional policy
```

标准H1-A2是`support on + grouped policy`。禁止retry、replacement、reranking或按结果
筛选。固定checkpoint、Plan、anchors、temperature、NFE/model-call budget、call-indexed
randomness、parser和post-hoc evaluator。统计上这是256个paired Plan–seed blocks，不是
1,024个独立样本；Plan是主要聚类单位，重复formula/Plan另作聚类敏感性分析。

Primary endpoint事前固定为：在全部eligible-Plan body requests中，body可重构且通过
同一独立post-hoc selected-check bundle的二元结果。Risk-difference scale上预先定义：

- support main effect；
- policy main effect；
- 两个simple effects；
- support×policy interaction；
- SESOI、power和equivalence/inconclusive rule。

一个seed/Plan可以估计该frozen Plan population上的平均干预效果，但不能分离Plan
heterogeneity与generation-seed variance，不能声称seed robustness，也不能估计原始
Planner all-request effect。

第一阶段只跑discrete body。只有equivalence interval落入事前SESOI时才称“无实际
effect”并停止；CI同时跨越有意义正负效果时称inconclusive，可按事前规则增加一个
seed-sensitivity subset或停止并诚实报告。

第二阶段control禁止根据结果挑选。事前固定标准H1-A2与两个single-factor controls：

```text
support on + grouped      (standard)
support off + grouped     (support simple effect)
support on + positional   (policy simple effect)
```

若相应body simple effect达到继续条件，对这三个cells的全部body successes运行同一个
refiner，比较pre/post conversion。`support off + positional`保留用于discrete factorial
interaction，不作为结果后选择的refiner comparator。

另在事前选择的64–128 Plans上增加第二个independent paired body seed，作为
mechanism seed-sensitivity audit。它检验方向是否对单个body seed脆弱，但不替代独立
Planner seeds。

### 必要的conditioning-scope ablation

真实executor读取完整Plan，而论文的最小material specification只定义composition/N。
因此复用2×2中的standard cell，并增加约256个paired anchors-only bodies：

```text
同一Plan、formula/N/element anchors、seed
standard support + grouped policy
full Plan context  vs  neutralized soft SG/volume/lattice context
```

Neutralization必须使用训练中出现过且schema支持的null/missing representation。若没有
in-distribution null表示，则primary control改为在同family/arity/N stratum内置换soft
fields，neutralized/OOD版本只作敏感性分析。

该对照识别soft rich-Plan context是否显著影响body realization。若有效，system-level
estimand必须明确写成`rich-condition realization`或把soft fields加入conditioning scope；
不能把其贡献静默计入composition-conditioned realization。它不证明Planner backbone
本身有效。

### 高价值低成本：conditional multiplicity

如果现有quick repeats不能直接复用，额外选择64个事前分层的learned Plans，每个Plan
只运行标准H1-A2的4个paired sampling seeds，共256 bodies。用StructureMatcher clusters
和local-environment fingerprints报告：

- 每个Plan的有效cluster数；
- effective multiplicity；
- 同Plan不同实现的lattice/coordinate差异；
- diversity经过refiner是否保留。

该实验只回答“一个model-proposed specification是否仍对应多个结构实现”，可反驳
“DLM只是在填一个几乎确定的template”。它不回答support/policy因果，也不进入
S.U.N. headline。

### 不应与上述面板混在一起的强实验

Matched AR-versus-DLM需要同composition/Plan、representation、anchors、parameter/
training budget、refiner和evaluator。当前冻结的crystal RQ明确点名masked completion，
因此严格ICLR版本将matched executor列为必做，不能用support/policy ablation替代。
若最终不做matched executor，主结论只能写
`the H1-A2 system exhibits a positive standardized within-stratum difference`，不能把该
差异因果归于masked architecture，并且必须同步改写crystal RQ、title和contributions。

当前主计划与fallback为：

1. **冻结主计划——强DLM方法论文：**训练/接入严格matched AR executor并做同
   specification paired comparison，正面回答“为什么使用DLM”；
2. **资源不足fallback——系统论文：**不新增matched AR训练，DLM只作为实现接口的
   设计选择；启用该fallback时必须正式降级RQ和claim，不能与当前顶层措辞并存。

鉴于当前标题和贡献仍突出masked completion，默认执行方案1。

独立Planner seeds用于证明proposal distribution和standardized gain可泛化。Body
repeats不能替代Planner seeds。若当前只使用一个Planner cohort，主文结论必须限定于该
frozen proposal population。

### 最小执行顺序

1. 冻结comparator/evaluator/endpoint/reference mix并确认raw cohorts；
2. 完成all-request funnel、common support和一个primary pooled-mix accounting；
3. 单独完成cohort U/S.U.N.与pre/post-refiner analysis；
4. 只有standardized per-request gap仍成立时，再跑matched executor、2×2 mechanism和
   soft-Plan ablation；
5. 其他reference mixes、memory/difficulty、multiplicity与R5-C作为sensitivity/appendix。

## 第三优先级：增强可信度但不承担主故事

- 对跨strata挑选且事前定义的小型候选集做更高保真relaxation/DFT验证；
- 使用第二个结构/能量模型检查稳定性排序是否对单一MLIP敏感；
- 报告sampling calls、wall time和显存，但不预设DLM更快；
- 只有在artifact可直接接入时统一重评一个强公开baseline，不按published score挑选。

这些是可信度增强项，不应阻塞先完成proposal–realization主分析。

## 实验停止与改写规则

| 结果 | 论文调整 |
|---|---|
| common-mix后gap消失 | 将aggregate提升解释为proposal-distribution associated，不写realization gain |
| 正向结果只在少数family | 删除broad-across-chemistry claim，报告集中性 |
| common support不足 | 只写overlap-restricted estimand |
| support/policy无效 | support/policy降为interface，不写其性能机制贡献 |
| soft Plan context有效 | 扩大conditioning scope，不再把所有residual差异叫composition-only realization |
| refiner抹除proposal gap | claim限定在discrete proposal stage |
| diversity明显下降 | 不写“without collapse”；把stability–diversity trade-off作为结果 |
| Gold Plan显著优于learned Plan | condition-source compatibility列为主要瓶颈 |

## 建议主文结构

1. Figure 1：Proposal–Realization问题与H1-A2接口；
2. Table 1：`105/488` headline＋完整all-request funnel；
3. Figure 2：proposal distributions＋各family条件转化差；
4. Figure 3：raw/standardized gap＋proposal/within-stratum residual accounting；
5. Table 2：matched AR-versus-DLM executor comparison；
6. Table 3：fixed-condition support/policy mechanism＋pre/post-refiner conversion；
7. Appendix：完整joint strata、element maps、common-support、R5-C、negative strata。

## 最小结论边界

不做matched executor时，最强安全结论是：

> Within preregistered measured and overlapping chemical support, H1-A2's
> additive per-request outcomes retain a positive standardized within-stratum
> difference that is not explained by measured proposal-distribution
> reweighting at the chosen stratum resolution; equal-size standardized cohorts
> retain diversity.

只有在exact/matched specifications、soft-Plan scope和fixed-condition evidence也
支持时，才进一步写`composition-conditioned structural realization`。完整S.U.N.只作
nonlinear standardized cohort comparison，不进入线性accounting。

只有matched AR成立后才能将其升级为“DLM相对AR的因果优势”。
