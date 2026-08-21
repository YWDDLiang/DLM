# 无新增训练实施状态

## 已在public branch实现

分支：`codex/h1a2-paper-de-novo`。

- E0 attempt-level funnel、family/arity/N/shortcut分层、TVD/JSD；
- symmetric chemistry-mix / within-chemistry decomposition；
- target-mix standardization与overlap、ESS、maximum-weight诊断；
- all-attempt与hull-known口径、known-both exact paired McNemar；
- element presence、atom-weighted frequency、stage survival与U&N conversion；
- Plan collision、tuple entropy、nearest-train tuple与taxonomy conversion审计；
- E1 learned/gold exact matched Plan选择；
- `Full K=8`、`Formula K=4`、`Shuffle K=4`，共768个body tasks；
- 前四seed跨arm配对；
- E2前8对Plan、两种source、三种arm、四个seed，共192 requested attempts；
- body failure保留在E2 denominator；
- DLM与refiner逐task deterministic seed；
- E1 lattice/SG/volume adherence、StructureMatcher clusters、CrystalNN
  fingerprint与multiplicity gate；
- E2 proposal/refined tensor对齐、identity/displacement/SG/P1分析和paired
  CHGNet single-point energy；
- 一键Slurm依赖提交，不触发任何训练。

## 等待资产后才能运行

- `data/plans/h1a2_learned_rich.jsonl`；
- `data/plans/r5c_gold_rich.jsonl`；
- DLM base/adapter；
- `checkpoints/diffusion/model_494.pt`；
- attempt-level E0输入表及pre/post tensors。

## E1/E2决策门

- rich adherence无差异：删去rich Plan realization主张；
- 至少75%的Plan只有一个有效cluster：删去multiple realizations主张；
- gold显著优于learned：condition source为主要瓶颈；
- refiner擦除E1差异：DLM贡献限制为proposal reliability。

## 明确不运行

- 新Planner训练；
- 新DLM/refiner训练；
- R03/safe-axis大扫表；
- duplicate-mask full1000重跑；
- 新full1000/1200 S.U.N.，除非E1/E2明确支持主claim。
