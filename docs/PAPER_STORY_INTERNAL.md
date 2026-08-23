# H1-A2论文问题演化与内部冻结稿

## 最终裁决

经过逐级Proposer–Reviewer、Constraint Guardian、User Advocate和Arbiter审查：

> **APPROVED，当前concept约7/10。**

批准的是问题、方法和贡献的逻辑，不是尚未得到的正向结果。每个机制问题都允许零
结果和负结果；若出现，必须删除对应的性能claim。

## 最终Main Research Question

> **When different crystal-validity checks can only be evaluated after
> different information has been generated, do restricting invalid choices
> whenever the prerequisite information is available and choosing which
> geometric variables are eligible for commitment at each stage affect how
> reliably a model-proposed composition is realized as a periodic crystal?**

中文：

> **当不同晶体合法性检查只有在生成出不同前提信息后才能判断时，在前提信息具备时
> 限制违规候选，并决定每个阶段哪些几何变量有资格竞争下一次提交，是否会影响模型
> 提出的composition被可靠实现为周期晶体？**

Scope：

> **Composition与原子数保持固定，研究域限定为learned source采样得到的eligible
> Plans。**

最通俗的一句话：

> 有些晶体错误只有生成到特定步骤才能发现；我们研究在错误已经能判断时限制生成，
> 并改变相关几何字段的提交策略，是否真的能提高可用周期晶体的实现产率。

## 为什么先前问题被否决

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
Q1  单一duplicate-Z restriction的介入
 ↓
Q2  当前partial state上可计算的selected-support bundle
 ↓
Q3  effect在self-sampled eligible Plan分布上的scope与异质性
 ↓
Q4  同masked checkpoint下的commitment-policy intervention
 ↓
Q5  fixed continuous refiner的downstream conversion
 ↓
Main RQ只保留support timing × commitment policy
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

## 最终三层论文结构

### Primary mechanism

```text
state-conditional selected support × commitment policy
```

### Fully de novo scope

```text
eligible complete Plans sampled by the learned source
```

### Downstream consequence

```text
fixed identity-preserving continuous refinement
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

## Exactly three contributions

1. **问题形式化**：selected validity checks需要不同前提信息，检查何时介入以及每个
   阶段哪些geometry变量有资格竞争下一次提交成为可检验问题。
2. **Core executor**：composition-anchored、exact-cardinality typed masked executor，
   结合state-conditional selected support与显式commitment-policy bundle。
3. **Plan-level paired empirical analysis**：以Plan为统计单位隔离support、policy及其
   interaction，分析pretreatment strata异质性和fixed-refiner downstream conversion。

Contribution 3在paired wiring和结果完成前只能写“we evaluate”，不能写“we
demonstrate”。

## 最通俗故事

> 系统先自己提出一个材料Plan，formula固定有哪些原子以及数量。随后生成晶格和
> 坐标，但有些错误只有相关字段出现后才能判断，而且先决定哪些字段也可能影响后续
> 结构。我们用同一个masked模型严格比较：信息够用时是否应限制明显违规候选，以及
> 按字段组限制下一次可竞争的位置是否比机械按位置提交更可靠。每次model call仍只
> 提交一个选中的字段。最后再看这些离散阶段差异经过固定连续
> 精修后是否仍然存在。

## Results context

未来论文主表继续保留：

- H1-A2 Strict S.U.N.：`105/1000 = 10.50%`；
- H1-A2 Meta S.U.N.：`488/1000 = 48.80%`。

用户当前给出的本地CrysLLMGen参考约为：

- Strict约`9%`；
- Meta约`44%`。

外部数值只说明最终系统处于有竞争力区间。协议冻结前不能证明strict superiority，也
不能替代support、policy或refiner的paired mechanism evidence。public表中当前精确
CrysLLMGen记录与该内部约数仍需最终核对。

## 最强剩余拒稿风险

> H1-A2仍可能被视为预训练masked模型外加手工`lattice→X→Y→Z`policy、三个局部
> masks和继承refiner。若support、policy及interaction没有清晰Plan-level效果，或效果
> 在refinement后失去意义，constraint-prerequisite framing会退化成constrained-decoding
> 工程。

该风险不能继续靠改写故事解决，只能由冻结的paired evidence回答。

## Decision Log

| 阶段 | 新增概念 | Reviewer保留 | Reviewer删除 |
|---|---|---|---|
| Q1 | duplicate-Z intervention | 最小paired因果问题 | duplicate-rate机械终点、计算节省claim |
| Q2 | constraint prerequisites | 当前三项selected-support bundle | “最早激活”、完整schedule由约束推出 |
| Q3 | self-sampled Plan domain | eligible Plan finite cohort与异质性 | anchor causal effect、revisable count、easy-condition事后定义 |
| Q4 | commitment policy | 同checkpoint grouped vs positional policy | DLM>AR、最优order、grouping单独归因 |
| Q5 | continuous conversion | fixed-refiner paired consequence | refiner算法创新、最终不稳定全归refiner、causal mediation |
| Paper压缩 | 主次层级 | support×policy为primary | 将Plan source、refiner和瓶颈分析并列进Main RQ |

## 当前大致缺口

- strict positional skip-anchor control；
- call-indexed paired random stream和稳定Plan/attempt metadata；
- 相同pre/post评价合同与Plan-level统计；
- 最终核对CrysLLMGen约`9%/44%`和public精确记录的口径；
- paired结果本身。

这些属于最小实验与评价wiring，不需要重新训练Planner、DLM或model494。
