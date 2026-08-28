# H1-A2 Proposal-versus-Realization论文内部冻结稿

## 最终裁决

经过逐级Proposer–Reviewer、Constraint Guardian、User Advocate和Arbiter审查：

> **APPROVED，当前concept约7/10。**

批准的是Proposal-versus-Realization科学问题、H1-A2方法假设和机制层级，不是尚未
得到的正向结果。每个机制问题都允许零结果和负结果；若出现，必须删除对应的性能
claim。

## 2026-08-26 Candidate终态更新

双候选实验已经给出负结论：

- Candidate B difficulty-decomposed Planner在两个Plan-256 seed中均使projected
  Strict/Meta chemistry mix下降，按规则停止；
- Candidate A counterfactual grounding完成4次独立fixed-256、逐sample_idx配对的
  control/candidate比较；
- Candidate A在body、Direct joint、novelty和Strict方向上没有退化，pooled Strict
  known为`103/985 → 105/988`（`+0.171 pp`）；
- 但pooled Meta known为`472/985 → 460/988`（`-1.360 pp`），未通过`-1.0 pp`
  非劣门；Strict exact McNemar `p=0.8506`，也没有显著增益。

因此：

> **Counterfactual grounding保留为有用的Strict/realization改进：它提高body、Direct、
> N∩U和Strict tail，但伴随Meta trade-off。标准H1-A2继续作为headline fallback。**

固定同一Plan cohort后，proposal mix严格相同，仍能观察到realization结果变化。
因此可以准确声称grounding带来小幅realization与Strict-oriented improvement；不能声称
它全面提高thermodynamic yield，Meta下降必须作为trade-off并列报告。

完整内部证据见
[`GROUNDING_FINAL_REPEAT4.md`](../results/remote_screens/GROUNDING_FINAL_REPEAT4.md)。
Public headline继续冻结为`105/1000 Strict、488/1000 Meta`，不得用本次Candidate的
`105/1024`替换或混称。

## 2026-08-27 Grounding稳健性修正

后续同Plan训练时长扫描和独立固定requested-1000复核推翻了“Candidate A可作为稳定
正向训练贡献”的较强表述：

- raw-256扫描中，约`0.295/0.590/1.000 epoch`的Strict/Meta candidate-control差依次为
  `+3/-1`、`+3/+4`、`-5/-2`（每点256 attempts）；
- `0.590 epoch`是唯一Strict与Meta同时为正且全部downstream门通过的窗口，但其
  candidate validation CE比control差`+0.07209`，所有paired McNemar也不显著，故不
  通过冻结的mechanism screen，不能事后挑为成功checkpoint；
- 在first-1000 parsed Plans、无survivor过滤的独立cohort上，full-epoch control/candidate
  为Strict `89→86`、Meta `487→467`，因此Strict和Meta方向门均失败；
- body `994→990`、Direct joint `877→874`、novelty和两种stable→S.U.N. retention都在
  `-1 pp`非劣界内。主要问题不是schema、Direct或novelty塌缩，而是candidate生成的
  stable结构减少：Strict-stable `109→106`、Meta-stable `589→569`。

因此当前只能声称：DLM训练存在非单调的novelty–stability优化窗口，既有
counterfactual-grounding目标没有形成跨规模稳健的S.U.N.增益。它不进入贡献列表。
下一候选必须直接作用于结构稳定性，例如固定Plan、低`E_hull` novel body为正样本、
高`E_hull` body为负样本的连续energy-contrastive supervised margin；不用policy
gradient，也不在推理阶段rerank。完整证据见
[`GROUNDING_CHECKPOINT_SWEEP_FINAL.md`](../results/remote_screens/GROUNDING_CHECKPOINT_SWEEP_FINAL.md)
和
[`GROUNDING_FIXED1000_FINAL.md`](../results/remote_screens/GROUNDING_FIXED1000_FINAL.md)。

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

若最终稿只保留两个主贡献，采用以下合并，不再寻找或包装失败的训练trick：

1. **Plan-conditioned exact-cardinality masked executor**：rich specification接口、
   composition/N硬锚和typed masked realization构成一个方法贡献；
2. **Proposal--realization attribution protocol**：raw-attempt ledger、stagewise
   conversion、proposal-mix/common-mix与fixed-condition诊断构成科学评价贡献。

`105/488`是上述系统与协议的结果证据，不单列为第三个算法。counterfactual grounding、
count-valence text Planner和energy-pair preference feasibility均为完整披露的负向开发证据，
不能进入贡献列表。若投稿目标强制要求第二个独立性能模块，则当前证据不足，必须另立
新方法与新预注册，而不能改门槛复活这些候选。

Candidate A稳健性复核后的限制是：Specification-compiled exact-cardinality executor仍是
核心技术贡献；counterfactual grounding不能作为第二个训练侧贡献。四重复Strict小信号和
raw-256中间窗口只作为机制诊断，独立requested-1000的Strict/Meta均下降是当前更强的
稳健性证据。若需要新增训练贡献，必须由新的stability-targeted目标重新取得正向、跨seed
且跨规模的Strict/Meta结果，不能沿用Candidate A命名或挑选step1000。

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

Candidate A四重复只作为内部方法筛选：Control/Candidate pooled Strict known为
`10.46%/10.63%`，Meta known为`47.92%/46.56%`。由于Meta非劣失败，它不进入
未来论文主表的“proposed method”行，也不改变105/488。

固定requested-1000稳健性复核进一步得到Control/Candidate Strict S.U.N.
`89/86`、Meta S.U.N. `487/467`。该cohort与public headline口径严格分开，但足以说明
full-epoch grounding的小幅Strict信号不能稳健复现。raw-256的step1000虽为`+3/+4`，
仍因冻结mechanism gate失败和统计不显著而不进入论文正向结果。

随后在冻结raw1000 rich-Plan cohort上进行普通DLM训练时长复核。总2 epoch与总3 epoch
分别得到Strict S.U.N. `81/1000`与`79/1000`、Meta S.U.N. `489/1000`与
`477/1000`；对应stable本身为Strict `102/100`、Meta `587/578`。更长CE虽把body
从`985`提高到`992`，却同时降低stable和stable∩novel，因此冻结Pareto规则选择总2
epoch，但其绝对`8.1%/48.9%`仍未过`10%/50%`门。这一结果否定“把基础DLM继续多训
一个epoch即可解决稳定性”的简单故事，并把后续训练贡献收缩为same-Plan
energy-contrastive geometry preference。

CrysVCD启发的count-valence Planner也只保留为负向机制证据。虽然离线价态分配覆盖
train/val/raw1000达到`96.66%/96.32%/94.10%`，普通text LLaMA输出的电中性率只有
`50.31%`，parse下降且all-metal shortcut升至`45.82%`。因此不能声称“物理价态标签
改善Planner”；真正的后续版本需要显式species/count表示、charge约束或专用head，且
必须重新做matched attribution。

same-Plan energy-contrastive路线同样在训练前的数据门停止。冻结的train-only 256 Plan
cohort在4 streams时得到`67/22` train/validation pair，扩至预注册上限8 streams后为
`95/27`；虽然energy-gap中位数为`0.1185 eV/atom`，train pair仍比冻结最低门`96`
少1。我们没有把“只差一对”解释为近似通过，也没有降低`0.06 eV/atom` gap、改split
或继续抽样。它只支持“同一Plan下存在明显结构能量差”这一数据诊断，不支持
preference-trained DLM贡献。

若该headline不是具有1:1逐attempt记录的raw cohort，则proposal–realization分析必须使用
单独命名的raw standard-H1-A2 cohort；不得为105/488构造伪microdata。

用户当前给出的本地CrysLLMGen参考约为：

- Strict约`9%`；
- Meta约`44%`。

外部数值只说明最终系统处于有竞争力区间。协议冻结前不能证明strict superiority，也
不能替代support、policy或refiner的paired mechanism evidence。public表中当前精确
CrysLLMGen记录与该内部约数仍需最终核对。

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
| Grounding epoch sweep | 非单调中间训练窗口 | 三个checkpoint全部披露 | 事后只挑step1000 |
| Grounding fixed1000 | 无survivor过滤的规模复核 | full-epoch负结果与stable瓶颈 | 将四重复小Strict信号包装为贡献 |
| DLM sufficient raw1000 | body与stable的非单调epoch Pareto | 总2/3 epoch全披露并冻结选择规则 | 只按CE或body挑更长checkpoint |
| Count-valence Planner | 物理标签覆盖与生成执行分离 | 96% teacher覆盖但50% emitted-neutral的负证据 | 将标签覆盖包装成生成化学正确性 |
| Same-Plan energy pairs | outcome-blind固定Plan与能量跨度 | 8-stream `95/27` pair-yield完整披露 | 因train只差1对而放宽96/24门 |
| Formula-only复审 | H1-B已完成的弱条件反例 | 执行成功与结构欠定/模板化分离 | 把formula-only重跑包装成新贡献 |
| Coordinate schedule复审 | D1/safe-axis的X/Y→Z合法性 | mixed-axis重复坐标负证据 | 将XYZ联合提交误称safe-axis/H1-A2 |
| CCFD候选 | 在线atom/charge守恒compiler | 只主张composition correctness | 暗示稳定性、合成性或首个价态tokenizer |
| dLLM RL复审 | diffu-GRPO/AGRPO/DiSPO必选先例 | RL只作最后工程手段 | 旧TraceRL或segment reward直接当新贡献 |

## 当前大致缺口

- 完整且预先固定的compound-family、arity、N-bin和元素分布；
- common-support coverage、proposal-distribution/within-stratum residual accounting；
- fixed-condition mechanism evidence与相同pre/post评价合同；
- full-Plan versus anchors-only conditioning-scope ablation；
- 与当前masked-completion crystal RQ匹配的AR executor comparison；
- DLM→refiner conversion和cohort-level uniqueness重算；
- 最终核对CrysLLMGen约`9%/44%`和public精确记录的口径；
- 对所有负向或证据不足类别完整披露；
- Candidate A不进入贡献列表；后续只允许预先冻结的stability-targeted非RL目标，且必须
  同时报告Strict/Meta、stable本身、stable→S.U.N. retention和全部checkpoint；Candidate B
  只按新授权做一次真实下游验证，不根据结果追加大规模搜索。

新的第二贡献候选不再与上述stability训练候选混同。CCFD Phase0 false-rejection/
coverage合同与冻结tokenizer接口审计均已通过；下一步是F0/F1同checkpoint
requested1000双seed试验。若仅在线FSM有效，论文只称conservation compiler。只有
独立tokenizer主效应通过时才讨论tokenizer贡献。外部方法比较已按用户决定移出本路线；
matched因果比较只使用内部同checkpoint控制。
双轨执行合同见
[`DUAL_TRACK_COMPOSITION_STABILITY_PLAN_V1.md`](DUAL_TRACK_COMPOSITION_STABILITY_PLAN_V1.md)：
Track A只主张composition correctness，Track B只主张fixed-composition stable conversion；
二者可组成系统故事但不得共享因果措辞。

CCFD formal Phase1现已终态失败：虽然内部assignment达到`1983/2000=99.15%`，独立
legacy comp-valid仅`1724→1725`，两seed方向不一致且N TVD为`0.064`。因此不把
conservation compiler或新tokenizer加入论文贡献；该结果只保留为“保证内部可赋价并不等于
提升独立composition validity”的机制负证据。稳定性Track B继续独立推进。

Track B的固定tau扫描同样给出明确负结论：tau0/200/500/800 pooled512 Strict
S.U.N.依次`10/29/39/48`，Meta依次`66/171/222/230`；短tau虽恢复novelty和
stable→S.U.N. retention，却单调损失稳定/S.U.N.，没有候选通过预注册gate。model494仍
保留tau800；该结果支持“refinement strength沿novelty–stability Pareto移动”，不能包装成
稳定性改进贡献。下一候选只允许冻结的noisy-state energy-critic独立评价审计。

除matched executor外，这些主要是现有结果整理、标准化和最小机制归因，不要求重训
Planner、DLM或model494。若不做matched executor，必须把crystal RQ降为system-level。

C³FD语义重构证明composition compiler具有强信号，但尚未形成可发表的完整第二贡献。
正式v2在requested2000上实现独立comp-valid `86.2%→98.6%`，两个seed同向，且
ionic-only增益不是all-metal shortcut；然而proposal-mix门失败。以完整训练分布重新定义并
预先冻结的v2.1成功修复N/arity/family与NU，但requested256 pilot出现`52/512`
semantic dead ends，parse降至`460/512`，all-metal也比full-train高`4.66pp`。因此：

- 可以作为机制/负结果称“semantic conservation显著提高composition correctness”；
- 不能把C³FD-v2或v2.1列为正式贡献、headline或下游稳定性原因；
- 不允许事后删除dead ends、放宽all-metal门或扩到requested1000；
- 若未来重新开启，必须预注册family-aware joint reachability，而不是继续调
  temperature/top-k/pair weight。

阶段证据见
[`C3FD_PLANNER_FINAL.md`](../results/remote_screens/C3FD_PLANNER_FINAL.md)、
[`C3FD_DRIFT_DIAGNOSTIC.md`](../results/remote_screens/C3FD_DRIFT_DIAGNOSTIC.md)和
[`C3FD_V21_PILOT_FINAL.md`](../results/remote_screens/C3FD_V21_PILOT_FINAL.md)。
