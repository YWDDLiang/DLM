# H1-A2 Proposal-versus-Realization论文内部冻结稿

## 最终裁决

经过逐级Proposer–Reviewer、Constraint Guardian、User Advocate和Arbiter审查：

> **APPROVED，当前concept约7/10。**

批准的是Proposal-versus-Realization科学问题、H1-A2方法假设和机制层级，不是尚未
得到的正向结果。每个机制问题都允许零结果和负结果；若出现，必须删除对应的性能
claim。

## 最终Main Research Question

> **In generative materials discovery, to what extent do gains in discovery
> yield arise from changing the distribution of material specifications being
> explored, versus improving structural realization conditional on an
> explored specification?**

中文：

> **在生成式材料发现中，发现产率的提升，在多大程度上来自所探索材料规格分布的
> 改变，又在多大程度上来自给定已探索规格后的结构实现能力提升？**

Scope：

> **Main RQ在generative materials discovery层面提出，但当前只在de novo inorganic
> crystals上验证，不外推到molecule、protein、真实合成或其他科学领域。**

晶体实例化：

> **For de novo crystal generation, can composition-anchored masked completion
> improve structural realization across model-sampled chemistries beyond gains
> explained by measured changes in the proposed-chemistry distribution,
> without collapsing cohort-level diversity?**

这里的material specification操作化为body生成前确定的composition、原子数`N`和
element multiset。Main RQ区分“提出/探索了什么specification”和“提出后实现得怎样”。

最通俗的一句话：

> 一个生成模型产率更高，究竟是因为它改变了探索什么材料，还是因为它真正更擅长
> 把已经提出的材料规格实现成结构？

H1-A2方法假设：

> 给定模型提出的composition和原子数，H1-A2固定化学身份与基数，以masked
> discrete completion生成周期几何，再进行identity-preserving continuous refinement；
> 我们检验该系统是否在预注册化学层上保留正的standardized within-stratum difference，
> 并在等规模标准化cohort中保持uniqueness。将差异因果归于masked architecture需要
> matched executor comparison。

机制RQ降为：

> 在composition和原子数固定后，根据当前前置信息限制部分违规token，并按信息依赖
> 决定哪些几何变量有资格竞争下一次提交，能否在同一个identity-preserving refiner下
> 提高离散周期body的实现率及最终转化率？

## 为什么先前问题被否决或降级

### Constraint timing and commitment

这是合格且可证伪的机制问题，但只描述H1-A2内部的inference intervention，不足以
承担整篇论文的材料发现动机。它保留为解释realization提升来源的Mechanism RQ。

### Composition-to-structure

“如何把composition变成structure”是所有晶体生成工作的共同任务，无法区分H1-A2。

### Pipeline bottleneck

“Planner、DLM还是refiner是瓶颈”是正经问题，但只自然引出漏斗分析，不能自然引出
masked DLM，因此降为secondary analysis。

### Serialization mismatch

直接主张serialization不应决定commitment过于solution-first，并与DDPD、ADLM、
DINGO相邻。最终将其改成可证伪的policy intervention，而不是先验真理。

## 问题演化图

```text
Main Scientific RQ
proposal-distribution gain  vs  specification-conditioned realization gain
                    ↓
Crystal Instantiation
model-sampled composition/N  vs  periodic-structure realization
                    ↓
H1-A2 Method Hypothesis
learned chemistry → anchored composition/N → masked geometry → fixed refiner
                    ↓
Mechanism RQ
selected support timing × commitment policy
                    ↓
Q1 duplicate-Z intervention
→ Q2 selected-support bundle
→ Q3 eligible learned-Plan scope与异质性
→ Q4 commitment-policy intervention
→ Q5 fixed-refiner downstream conversion
```

## Q1：最小可证伪问题

> 在冻结的同一组Plan–attempt配对及全部非干预条件下，仅在每个Z token提交前屏蔽
> 会形成离散PBC重复位点的候选token，相比不施加该屏蔽，是否提高全部请求中得到
> “可重构且duplicate-free”body的概率？

### 冻结终点

```text
Y1(B) = reconstructable(B) AND duplicate_free(B)
```

`reconstructable`要求：

- Plan-matching typed parse；
- pymatgen periodic Structure构建；
- CrysLLMGen graph conversion；
- finite字段与一致的`n_atom`。

Post-hoc duplicate detector独立于generation mask：对全部active sites按100-bin PBC模
等价检查坐标，忽略species。两个arms使用同一detector，mask状态不能作为label。

### Reviewer修订

- 删除“early rejection/backtrack”说法：当前只是在Z提交前屏蔽token；
- 主终点必须同时包含reconstructability，否则duplicate-free提升接近机械结论；
- 固定all-request denominator、NFE和attempt-local随机流；
- 允许零或负效果。

最终状态：`APPROVED`。

## Q2：Selected support bundle

> 在schedule不变时，若某项selected check的前提变量已经存在于当前partial state，
> 对违反它的候选token施加mask，相比生成期间不施加mask、只做统一post-hoc检查，
> 是否改变联合有效body产率？

真实范围仅包括：

1. length zero-token restriction；
2. alpha/beta恰已提交时的opportunistic gamma-degeneracy restriction；
3. X/Y及相关site信息可见时的discrete PBC duplicate-Z restriction。

```text
Y2(B) = R(B) AND L(B) AND G(B) AND NOT D(B)
```

### 关键Reviewer发现

六个lattice fields处于同一confidence group，不能保证alpha/beta早于gamma。因此：

- 不写“每项约束都在最早时刻激活”；
- 只写“prerequisites在当前partial state已可见时激活”；
- 不声称三项检查推出完整`lattice→X→Y→Z`顺序；
- Q2只识别三个masks作为bundle的净效应。

最终状态：`APPROVED`。

## Q3：Fully de novo Plan scope

最初尝试把anchors当作causal treatment，但被Reviewer否决：

- 固定`7+4N`时让count真正变化缺少一致语义；
- Plan-matching parser本身已经包含identity条件；
- hard element prefill还固定了任意slot顺序；
- anchors是任务合同，不应硬包装成性能贡献。

最终Q3改成：

> Q2的paired effect在冻结learned-source sampling contract得到的eligible完整Plan-record
> 分布上是否非零，并在只用pretreatment Plan字段定义的strata中表现出怎样的异质性？

完整Plan `P`包含formula、`N`、elements/counts和soft coarse fields。DLM读取完整P；
只有formula-derived composition、`N`和elements/counts属于hard anchors。

Plan-level effect：

```text
delta(P) = average_repeat [Y2(mask_on) - Y2(mask_off)]
```

Finite-cohort effect以Plan为统计单位；Plan内repeat不是独立Planner样本。Planner
ineligible outputs只作upstream attrition，不进入downstream mechanism effect。
当最小面板每Plan仅一个body seed时，`average_repeat`退化为单一paired block，只估计
frozen Plan population的平均干预效果；不能估计Plan-specific seed-averaged effect、
seed variance或seed robustness。

Q3只解释learned condition source为何用于闭合fully de novo scope，不把其backbone
作为算法贡献。

最终状态：`APPROVED`。

## Q4：Commitment policy

> 在相同masked checkpoint、Plans、anchors、typed schema、selected support、NFE和
> call-indexed随机流下，group-restricted confidence-adaptive policy与fixed positional
> policy是否产生不同的联合realization yield？Support效果是否依赖policy？

Grouped policy：

```text
[six lattice fields] → [all X] → [all Y] → [all Z]
```

Group内部按confidence提交一个position。因此正式treatment是：

```text
group restriction + confidence-adaptive position selection
```

Fixed positional policy：

```text
LA, LB, LC, alpha, beta, gamma,
X1, Y1, Z1, ..., XN, YN, ZN
```

count和element anchors跳过。两臂严格执行`6+3N`denoising forwards，并读取相同的
call-indexed full suffix×vocab noise ledger。

### Q4能够支持

> 对同一masked checkpoint，commitment policy是可操纵、可能影响crystal-body
> realization的inference变量。

### Q4不能支持

- DLM优于AR；
- grouped policy是唯一或最优order；
- dependency grouping单独产生收益；
- DDPD式learned planning；
- committed token revision。

当前代码已有grouped接口，但strict positional skip-anchor control、paired noise ledger
和实际model-call验证仍需最小wiring；历史default/exact结果不能重解释成Q4证据。

最终状态：`APPROVED`。

## Q5：Discrete-to-continuous consequence

> 对Q4每个policy产生的全部`R(B)=1` discrete bodies，固定model494相较于原body，
> 是否改善同一body的统一评价；proposal-stage policy gap在refinement后是否保留、
> 衰减或反转？

所有且仅有`R=1` bodies进入，不按Q2、energy、stability、N/U筛选。

```text
B = (A, quantized lattice, quantized coordinates)
M(B) = (A, continuous lattice, continuous coordinates)
```

Refiner必须逐位置保持`N`和atom types，只修改lattice与fractional coordinates，不读
Plan。异常、identity violation、NaN或shape mismatch作为post failure保留在分母中。

严格区分：

- successful-body paired conversion；
- hull known-both及known/unknown coverage；
- all-request lower-bound yield；
- cohort-level N/U/S.U.N.重算。

Uniqueness不是逐body独立标签。Policy gap缩小只能称observed attenuation，不称causal
mediation。

model494是继承组件；贡献在于identity-preserving interface和conversion evidence，
不是refiner算法。

最终状态：`APPROVED`。

## 最终论文层级

### Main scientific problem

```text
explored material-specification distribution
vs
specification-conditioned structural realization
```

当前实证实例严格限定为composition/N → periodic crystal structure。

### Method hypothesis

```text
learned composition/N
→ anchored exact-cardinality state
→ masked periodic-geometry completion
→ identity-preserving continuous refinement
```

### Primary mechanism

```text
state-conditional selected support × commitment policy
```

### Evidence hierarchy

```text
完整化学分布与条件稳定率：广泛性
→ common-mix标准化与accounting decomposition：反shortcut
→ matched AR-versus-DLM：masked architecture attribution
→ fixed-condition paired analysis：selected-support/policy mechanism
→ pre/post refiner conversion：refiner attribution
```

## 当前真实方法合同

```text
P ~ p_phi(P)
A(P) = formula-derived N + element multiset/counts
G ~ p_theta(G | P, A(P), support, policy)
B = (A(P), G)
M ~ p_psi(M | B)
```

完整state为`7+4N`，实际自由生成`6+3N`个geometry tokens。Refiner只读B，不能写成
`p_psi(M|B,P)`。

主RQ的统计对象按预注册化学层`h`定义：

```text
p_m(h) = 方法m尝试化学层h的概率
r_m(h) = 方法m在h内的additive per-request outcome rate
theta_m = sum_h p_m(h) * r_m(h)
```

`p_m`差异是proposal-distribution associated；shared measured support中的`r_m`差异
首先称within-stratum residual。只有层内specification difficulty充分平衡时，才能解释为
realization-associated。对称分解只称accounting decomposition，不称causal mediation。
Primary strata为formula-derived family × arity × N-bin；exact element set、训练集稀疏度
和独立baseline-difficulty只作预先定义的敏感性分析。

因此primary standardization严格识别的是`within measured coarse chemical strata`的
residual difference，不是固定同一个exact formula的效果。Exact-condition结论只能由
同composition/Plan的paired mechanism evidence给出。

该线性identity只用于body success、Direct validity、stable all-request yield、novel或
stable-and-novel等可加per-request endpoints。Uniqueness和完整S.U.N.是非线性
cohort-level functions，必须在固定size/mix下单独重算，不能进入上述分解。

## Exactly three contributions

1. **科学与评价问题形式化**：区分explored material-specification分布变化与给定
   specification后的structural realization提升；在晶体实例中将specification操作化为
   composition/N，并将uniqueness正确处理为cohort-level outcome。
2. **Core executor**：composition-anchored、exact-cardinality typed masked executor，
   结合state-conditional selected support与显式commitment-policy bundle。
3. **Attribution framework**：完整报告化学分布和stagewise conversion，进行common-mix
   标准化，再以matched executor、fixed-condition mechanism与pre/post-refiner analysis
   区分系统广泛性、masked architecture、execution policy和continuous refinement。

Contribution 3在distribution、standardized accounting和fixed-condition结果完成前
只能写“we evaluate”，不能写“we demonstrate”。

## 最通俗故事

> Aggregate discovery yield同时混合了“系统提出/探索什么材料规格”和“它能否把这些
> 规格真正实现成结构”。在晶体实例中，S.U.N.具有同样混杂。H1-A2让Planner保留
> de novo化学探索，但在body阶段固定composition和N，
> 迫使masked DLM直面给定chemistry的周期几何实现；固定refiner随后只修几何。完整
> 化合物分布、每类稳定转化率和common-mix标准化用来判断增益是否主要来自选择容易
> chemistry，fixed-condition机制与pre/post-refiner则解释realization增益来自哪里。

## Results context

未来论文主表继续保留：

- H1-A2 Strict S.U.N.：`105/1000 = 10.50%`；
- H1-A2 Meta S.U.N.：`488/1000 = 48.80%`。

若该headline不是具有1:1逐attempt记录的raw cohort，则proposal–realization分析必须使用
单独命名的raw standard-H1-A2 cohort；不得为105/488构造伪microdata。

用户当前给出的本地CrysLLMGen参考约为：

- Strict约`9%`；
- Meta约`44%`。

外部数值只说明最终系统处于有竞争力区间。协议冻结前不能证明strict superiority，也
不能替代support、policy或refiner的paired mechanism evidence。public表中当前精确
CrysLLMGen记录与该内部约数仍需最终核对。

## 2026-08-26双候选实证裁决

### Candidate A：Composition-Matched Counterfactual Plan Grounding

Candidate A保留为有边界但有用的技术改进。Matched validation factual CE由对照的
`1.9153`降至`1.6240`；四次fixed-256 screen汇总显示body completion约`+0.29pp`、
Direct joint约`+0.29pp`、N∩U增加5个，Strict hull-known约`+0.17pp`。同时Meta
hull-known约`-1.36pp`，因此不能包装成全面提升，也不能替换现有public `105/488`。
当前最诚实的身份是：counterfactual grounding改善了部分realization与Strict行为，
但存在Meta trade-off，值得作为最终技术贡献候选继续保留。

### Candidate B：Difficulty-Decomposed Self-Improving Planner V2→strong20 V3

未归一化V1（`34697/34704`）冻结为负结果。修正版V2采用
`proposal_shift × within-stratum-normalized advantage`，历史buffer为1219条，
ESS为832.475（ratio 0.6829），并用两个Planner seed进行真实下游评价。V2确实改变了
proposal mix：oxide在seed17由45增至58、seed18由42增至47；但这种变化没有稳定转化
为discovery yield。

512-attempt pooled结果为：

- Direct joint：`449/512 → 455/512`，`+1.17pp`；
- Strict：`37/512 → 36/512`，`-0.20pp`；
- Meta：`229/512 → 224/512`，`-0.98pp`；
- novel rate：`453/509 → 446/508`，`-1.20pp`；
- body completion：`509/512 → 508/512`，`-0.20pp`。

seed17的Strict/Meta分别为`+2.34pp/+1.56pp`，seed18则为
`-2.73pp/-3.52pp`。Pooled known-both exact McNemar中，Strict discordance为
`29 vs 29`（p=1），Meta为`105 vs 108`（p=0.891）。因此normalized V2未通过
Strict方向和novelty non-inferiority两项预设判据，不保留为正方法贡献。

但post-hoc训练审计进一步发现：V2使用`batch_size=1`，原loss又除以当前batch的
sample-weight总和，因此每个scalar difficulty weight在单样本microbatch中完全约掉。
V2实际测试的是加入1219条self-improvement rows后的近似uniform buffer mixture，
而不是预期的difficulty-decomposed weighting。故V2负结果不能否定正确加权方法。
修正版strong20 V3采用独立sampling-weight字段与replacement weighted sampling，
将self-improvement真实抽样概率设为20%，并将control/candidate统一为800 updates。

strong20 V3实际抽样率为`19.875%/20.063%`。在两个Planner seed、每cell 256 attempts
的冻结下游中，pooled结果为：

- body：`504 → 506`，`+0.39pp`；
- Direct joint：`437 → 445`，`+1.56pp`；
- novel：`437 → 443`，N∩U：`437 → 442`；
- Strict：`34/512 → 37/512`，`+0.59pp`，且两个seed方向均为正；
- Meta all-attempt：`216/512 → 213/512`，`-0.59pp`；
- hull known/unknown：`478/26 → 499/7`。

这说明正确加权后出现了比V2更一致的realization、novelty和Strict正信号，支持“Plan
优化有效但效应较小”的解释。不过，新增21个hull-known结构扩大了known denominator，
Meta hull-known rate由`45.19%`降至`42.69%`（`-2.50pp`），因此未通过事前要求的
all-attempt与known-rate双重Meta non-inferiority gate。该结果应称promising scoped
Planner improvement，不能称完整通过或替换public headline。Pooled exact McNemar的
Strict/Meta p值分别为`1.0/0.5459`，两个Planner seed仍不足以作强显著性主张。

这项负结果仍直接服务主RQ：Planner改变proposal distribution、甚至提高Direct joint，
并不自动意味着S.U.N.提升。它是proposal-mix与downstream conversion必须分开报告的
内部证据，但两个Planner seed不足以支持面向全领域的普遍结论。不同arms的composition
不同，ordinal pairing仅是common-random-number的端到端比较，不能解释为固定composition
的realization effect。

完整记录见
[`PLANNER_DIFFICULTY_V2_FINAL.md`](../results/remote_screens/PLANNER_DIFFICULTY_V2_FINAL.md)、
同名JSON和CSV。论文主线继续以冻结H1-A2为fallback，并优先保留Candidate A；
public `105/1000 Strict、488/1000 Meta`保持不变。V3在独立run中评价，不覆盖V2。
完整证据见
[`PLANNER_DIFFICULTY_V3_STRONG20_FINAL.md`](../results/remote_screens/PLANNER_DIFFICULTY_V3_STRONG20_FINAL.md)。

## 最强剩余拒稿风险

> 即使oxide、halide、arity和各N-bin都提升，Planner仍可能在每个粗类别内部选择更
> 容易的exact formulas；最终增益也可能主要由继承的model494产生。粗类别表不能单独
> 识别composition-conditioned realization，更不能单独证明DLM因果有效。

防线是完整预注册分层、common-support diagnostics、composition standardization、
fixed-condition mechanism和pre/post-refiner conversion，而不是继续加强措辞。

## Decision Log

| 阶段 | 新增概念 | Reviewer保留 | Reviewer删除 |
|---|---|---|---|
| Q1 | duplicate-Z intervention | 最小paired因果问题 | duplicate-rate机械终点、计算节省claim |
| Q2 | constraint prerequisites | 当前三项selected-support bundle | “最早激活”、完整schedule由约束推出 |
| Q3 | self-sampled Plan domain | eligible Plan finite cohort与异质性 | anchor causal effect、revisable count、easy-condition事后定义 |
| Q4 | commitment policy | 同checkpoint grouped vs positional policy | DLM>AR、最优order、grouping单独归因 |
| Q5 | continuous conversion | fixed-refiner paired consequence | refiner算法创新、最终不稳定全归refiner、causal mediation |
| 旧Paper压缩 | 机制主次层级 | support×policy保留为Mechanism RQ | 将其继续当作论文级科学问题 |
| 新Main RQ | proposal vs realization | explored specification与conditional realization分离 | 向非晶体领域过度外推、指控其他方法作弊、把accounting写成causal mediation |

## 当前大致缺口

- 完整且预先固定的compound-family、arity、N-bin和元素分布；
- common-support coverage、proposal-distribution/within-stratum residual accounting；
- fixed-condition mechanism evidence与相同pre/post评价合同；
- full-Plan versus anchors-only conditioning-scope ablation；
- 与当前masked-completion crystal RQ匹配的AR executor comparison；
- DLM→refiner conversion和cohort-level uniqueness重算；
- 最终核对CrysLLMGen约`9%/44%`和public精确记录的口径；
- 对所有负向或证据不足类别完整披露。

除matched executor外，这些主要是现有结果整理、标准化和最小机制归因，不要求重训
Planner、DLM或model494。若不做matched executor，必须把crystal RQ降为system-level。
