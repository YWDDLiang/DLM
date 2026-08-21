# 严格reviewer的concept-only冻结结论

本文件只评问题、方法与故事。代码、checkpoint、seed和统计审计单列，不因工程缺口
直接扣concept分；但代码事实决定哪些主张可以成立。

## Verdict

当前诚实故事：`5.5–6/10`，Borderline/Weak Reject区间。

最强中心主张是：

> Serialization order need not determine commitment order. Given a
> model-sampled composition, H1-A2 anchors cardinality and element identity,
> then uses masked completion to realize the remaining periodic geometry
> before identity-preserving continuous refinement.

它比“把CrysLLMGen的AR换成DLM”更清楚，但当前还不是新的通用DLM算法。

## Reviewer认可的三点

1. composition/N先于geometry realization，任务接口清楚；
2. `7+4N`state与实际`6+3N`自由geometry tokens区分明确；
3. non-prefix context、typed schema、依赖顺序和局部legality checks形成可解释executor。

## 最强拒稿理由

> 现有训练仍是vanilla random-mask CE；主要新意可能是领域接口与inference policy，
> 而不是学习算法。没有matched AR时，也不能把结果优势归因于masked factorization。

## 当前可守主张

- composition-anchored exact-cardinality geometry completion；
- serialization order与commitment order可以解耦；
- selected checks在依赖信息出现后才可计算；
- learned/gold/replay Plan来源可以分解端到端失败；
- chemistry mix与within-chemistry conversion应分别报告。

## 当前不可守主张

- DLM联合生成species-site assignment；
- legal support被训练目标学习；
- Plan volume/SG得到硬执行；
- revealed tokens会被revision；
- refiner由Plan条件化；
- non-prefix意味着atom-permutation invariance；
- DLM普遍优于AR、更快、更多样或更稳定。

## 当前大致缺什么

- 更公平的matched baseline和少量关键消融；
- rich Plan与DLM各自贡献的直接证据；
- 更多独立seed和统计支持；
- 完整公开资产、评价协议与端到端复现。

这些补齐前，不写“DLM普遍优于AR”或“提出新的通用DLM算法”。具体实验矩阵暂不锁定。
