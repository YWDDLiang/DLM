# H1-A2论文故事冻结版

本文件冻结当前已经实现的方法，不把未来算法写成现有贡献。工程与复现问题另见
`ICLR_REVIEW_STRATEGY.md`，不参与本文件的concept-only评分。

## 一句话

> Formula告诉模型晶体里有哪些原子、共有多少个；H1-A2不按文本顺序逐个写坐标，
> 而是在一个大小刚好的typed state中补全量化周期几何，最后由连续扩散在不改变化学
> 身份的前提下精修几何。

## 30秒故事

现有语言模型把晶体序列化后生成，但晶格和所有原子坐标彼此关联，文件顺序并不等于
合理的科学承诺顺序。H1-A2先由learned Planner采样formula与粗粒度条件；formula派生
并锚定composition、原子数和element multiset。Crystal DLM随后在exact-cardinality
typed state中补全六个lattice/angle字段与全部坐标，并按照
`lattice -> X -> Y -> Z`的依赖顺序提交字段。CrysLLMGen式continuous refiner最后只接收
离散proposal并精修连续几何。评价时把condition-source、discrete realization和
continuous refinement分开归因。

## Main research question

> **How should a masked discrete language model realize a model-sampled
> composition as an exact-cardinality periodic crystal body when different
> crystallographic legality checks become evaluable at different partial
> states, without letting serialization order dictate commitment order?**

这个问题没有声称DLM是唯一可能方案，也没有声称AR无法表达相同联合分布。主张是：
在这个接口中，masked completion允许我们显式安排信息何时可见、字段何时提交。

## 当前真实概率合同

令：

- `P`：learned Planner输出的formula与coarse fields；
- `A(P)`：由formula派生并预填的`N`与ordered element multiset；
- `G`：DLM自由生成的六个lattice/angle字段与`3N`个坐标字段；
- `B=(A(P),G)`：完整离散proposal；
- `M`：continuous refiner输出。

则当前实现是：

```text
P ~ p_phi(P)
G ~ p_theta(G | P, A(P))
M ~ p_psi(M | B)
```

refiner不读取Plan，因此不能写`p_psi(M|B,P)`。

## `7+4N`究竟表示什么

完整state包含：

```text
1 count + 6 lattice/angle + N * (1 element + X + Y + Z)
```

总长度为`7+4N`。但默认H1-A2中count和全部element slots已由formula预填并冻结，
DLM实际自由生成的是`6+3N`个geometry tokens。论文必须写成
**composition-anchored geometry completion**，不能写成species与geometry联合生成。

## 当前support边界

已实现：

- typed token schema；
- 非零lattice lengths；
- alpha/beta已知后对degenerate gamma的条件检查；
- X/Y已知后对PBC-equivalent duplicate Z的排除；
- `lattice -> X -> Y -> Z`提交顺序。

未实现：

- Plan volume-bin硬约束；
- exact space-group执行；
- minimum-distance或全局可满足性保证；
- revealed token重新打开；
- violation-guided revision。

训练仍是random masking加masked-token cross entropy。support-consistent training、
legal-mass objective、stoichiometric assignment都属于未来工作。

## Exactly three contributions

1. **Problem/interface contribution.** 将fully de novo generation写成model-sampled
   global chemistry之后的composition-anchored、exact-cardinality typed realization，
   并明确区分learned-Plan inference、gold-Plan reference与frozen replay。
2. **Crystal DLM executor.** 在`7+4N`state上结合non-prefix context、field-specific
   token support和`lattice -> X -> Y -> Z`提交顺序，并使用当前确实可计算的局部
   lattice/PBC legality checks。
3. **Attribution/evaluation contribution.** 通过Gold Plan（R5-C）、refiner前后分析和
   chemistry-standardized decomposition，把condition-source、discrete realization、
   continuous refinement以及composition-mix effect分开。

## R5-C的论文身份

| 名称 | Plan来源 | 科学身份 | Fully de novo |
|---|---|---|---:|
| `A_learned` | learned H1-A2 Planner | 主系统 | 是 |
| `C_gold / R5-C` | held-out MP-20派生gold Plan | conditional executor reference | 否 |
| `C_replay` | frozen generated Plans | downstream/replay control | 否 |

R5-C不是Planner-free，也不是数学upper bound。它的作用是拆出learned Plan gap、
given-Plan DLM realization和continuous refinement conversion。历史adjusted R5-C只作
legacy context；新的matched comparison必须统一schema、DLM、schedule、refiner、
evaluator和raw denominator。

## S.U.N.结果合同

`105/1000` Strict与`488/1000` Meta继续作为未来论文S.U.N.主表值，不降为deprecated
descriptive aggregate。`103/1200`、`553/1200`是exact all-requested-attempt audit；
`94/1000`、`474/1000`是historical frozen compatibility。三个视图必须分别命名，
不能互相替换。

## 化学mix与structure conversion

总体稳定性可写成：

```text
P_m(Y=1) = sum_h p_m(h) * mu_m(h)
```

其中`p_m(h)`是方法采到的化学分层分布，`mu_m(h)`是固定化学分层下的结构转化率。
这允许我们问提升来自更容易稳定的composition mix、相同chemistry下更好的结构实现，
还是两者兼有。不能把该分析预先写成“RL只会选容易稳定的composition”。

## 禁止进入当前摘要/贡献的表述

- DLM联合生成species-site assignment与geometry；
- exact length本身是新的variable-length DLM；
- support-consistent training或legal-mass objective；
- violation-guided reopening/revision；
- Plan volume、lattice family或space group被硬执行；
- refiner读取Plan；
- non-prefix等价于permutation invariance；
- DLM天然更快、更多样或更稳定。

## 标题

首选：

> **Serialization Is Not Commitment Order: Composition-Anchored Masked
> Completion for Crystal Generation**

保守版：

> **Composition-Anchored Crystal Completion with Masked Language Models**

## 当前评分

当前诚实故事约`5.5–6/10`。补充分析可让论证更完整，但不能单靠写作变成稳健
Weak Accept。要达到可信`7/10`，优先缺少matched constrained AR与经过验证的
support-consistent DLM training；它们是未来方法升级，不冒充本轮已完成工作。
