# H1-A2严格Proposer–Reviewer最终裁决

## Disposition

> **APPROVED，concept约7/10。**

这是对科学问题、方法演绎和贡献层级的批准。它不批准尚未产生的正向经验结论。若
selected support或commitment policy得到零/负结果，Research Question仍成立，但对应
方法claim必须删除。

## 冻结Main RQ

> **When different crystal-validity checks can only be evaluated after
> different information has been generated, do restricting invalid choices
> whenever the prerequisite information is available and choosing which
> geometric variables are eligible for commitment at each stage affect how
> reliably a model-proposed composition is realized as a periodic crystal?**

Scope：composition和atom count固定；研究域为learned source采样的eligible Plans。

## 为什么通过严格review

- 问题不包含H1-A2、DLM或正向答案；
- support和policy均有明确反事实；
- 允许正、零、负和Plan-stratum异质性；
- 方法事实与代码吻合；
- masked executor由partial-state和policy问题自然引出；
- Planner和refiner被正确降为scope与downstream consequence；
- claim仅覆盖当前三个selected checks。

## Concept评分

| 维度 | 分数 |
|---|---:|
| 科学问题清晰度 | 8/10 |
| 可证伪性 | 8/10 |
| 与H1-A2匹配度 | 8/10 |
| 方法新颖性 | 6.5/10 |
| 综合concept | **7/10** |

## 最强方法贡献

> A composition-anchored typed masked executor in which selected validity
> support is activated from the current partial state and commitment policy is
> an explicit, testable inference variable.

它比“用DLM生成晶体”更具体，但新颖性仍是conditional：只有support、policy及其
interaction产生清晰Plan-level结果，且proposal差异在refinement后仍有意义，方法贡献
才达到ICLR强度。

## 最强拒稿理由

> 这可能只是一个预训练masked checkpoint，加上手工grouped policy、三个局部logit
> masks和继承的model494。若paired结果不支持support与policy机制，最终高S.U.N.不能
> 挽救核心方法claim。

这个风险不能靠进一步改写解决。

## 冻结贡献层级

1. **问题贡献**：selected crystal checks的prerequisite information和geometry
   commitment成为可检验变量。
2. **方法贡献**：composition-anchored typed masked executor＋state-conditional
   selected support＋grouped confidence-adaptive policy。
3. **证据贡献**：Plan-level paired mechanism、pretreatment heterogeneity及fixed-refiner
   downstream conversion。

Contribution 3在严格wiring和结果完成前只能写成“we evaluate”。

## Claims lock

禁止：

- DLM普遍优于AR；
- serialization普遍有害；
- 当前policy唯一或最优；
- 三项checks构成通用constraint system；
- support-consistent training；
- rich Plan soft fields被硬执行；
- committed token revision；
- Planner backbone或model494属于算法创新；
- 用`10.5/48.8`对`9/44`替代机制证据。

允许：

- 同一masked checkpoint支持显式commitment-policy intervention；
- 当前partial state决定哪些selected checks可计算；
- Plan-level paired对照可估计support、policy及interaction；
- fixed model494可用于identity-preserving pre/post conversion；
- end-to-end数字在协议注明后作外部context。

## Evidence readiness

现有checkpoint无需重训。概念落地仍需：

- strict positional skip-anchor control；
- equal model-call budgets；
- call-indexed paired randomness；
- Plan/attempt/body ID贯穿；
- 所有successful bodies进入fixed refiner；
- 相同pre/post Direct、CHGNet与hull评价；
- 以Plan为统计单位；
- 最终确认CrysLLMGen内部约数与public精确记录口径。

## 与旧故事的关系

旧的“Serialization Is Not Commitment Order”保留为Methods insight，不再作为Main RQ。
旧的“哪个模块是瓶颈”保留为secondary stage analysis。当前Main RQ只研究：

```text
selected support timing × commitment policy
```

其fully de novo scope由learned Plans定义，其end-to-end后果由fixed refiner观察。
