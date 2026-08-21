# Fully de novo边界、Planner角色与R5-C控制

## 冻结结论

训练时从MP-20确定性提取Plan label仍是监督学习；推理时直接从MP-20结构派生Plan，
则不再是Plan层面的fully de novo generation。主路线必须从learned Planner采样Plan。

当前实现：

```text
P ~ p_phi(P)             learned condition source
A(P)                     formula-derived N and element anchors
G ~ p_theta(G|P,A(P))    six lattice/angle + 3N coordinate tokens
B = (A(P), G)
M ~ p_psi(M|B)           refiner does not read P
```

## 三种Plan来源

| 名称 | 来源 | 用途 | Plan层面fully de novo | 主文位置 |
|---|---|---|---:|---|
| `A_learned` | learned H1-A2 Planner | fully de novo主系统 | 是 | headline |
| `C_gold / R5-C` | held-out MP-20结构派生gold Plan | conditional executor reference | 否 | 第二任务/诊断 |
| `C_replay` | frozen H1-A2/R03 generated Plans | paired downstream control | 否 | appendix |

R5-C不能叫Planner-free、oracle upper bound或fully de novo。它回答的是：给定可靠的
global specification时，同一个DLM executor与refiner能做到什么。

## 为什么Planner仍有用

body state长度为`7+4N`，因此`N`必须在state实例化前存在；composition也必须先确定，
才能预填count和element slots。learned Planner的作用是闭合de novo loop，而不是证明
特定LLM backbone本身新颖。

不使用独立Planner仍有两条不同路线：

1. DLM先生成header，再实例化body。这仍可de novo，但需要新的two-pass训练，当前未
   实现。
2. 使用MP-20/frozen/user Plan。这是specification-conditioned realization，不是
   Plan-level de novo。

## R5-C如何进入论文

- `A_learned`承担fully de novo headline；
- `C_gold`分解condition-source gap与executor/refiner capability；
- `C_replay`解释固定Plan下的历史执行差异；
- historical adjusted R5-C不与现代H1-A2作裸差或因果解释；
- matched A-vs-C必须统一rich schema、DLM、schedule、refiner、evaluator、raw
  denominator，并refine全部body successes。

## 决策解释

- gold明显优于learned：condition-source compatibility是主要瓶颈；
- gold与learned接近：不能继续把端到端损失主要归因于Planner；
- rich fields相对formula/shuffle没有adherence差异：将条件收缩为formula/N/composition；
- 一个Plan的有效结构几乎只有一个cluster：删除multiple realizations主张。

这些判断后续需要由事先确定、且不按最终energy挑样本的对照证据支持。
