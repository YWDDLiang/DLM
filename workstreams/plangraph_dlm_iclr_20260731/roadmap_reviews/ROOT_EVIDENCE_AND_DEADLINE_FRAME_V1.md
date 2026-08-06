# H1 / PlanGraph-DLM ICLR 2027 证据与时间框架 V1

状态：`decision_only_no_execution_authorization`

日期：2026-08-04

用途：为 Planner、Body-token/support 与 mask-aware RL 三份提案及其交叉评审
提供共同事实、投稿时间和判定规则。本文不授权训练、生成、refinement、S.U.N.
或任何自动下游。

## 1. 官方投稿边界

ICLR 2027 官方页面给出的时间为：

- 摘要：2026-09-18 AOE；
- 全文：2026-09-25 AOE；
- 主文最多 9 页；
- 必须在论文与 submission form 中披露生成式 AI 的使用。

来源：

- https://iclr.cc/Conferences/2027/index.html
- https://iclr.cc/Conferences/2027/AuthorGuidelines
- https://iclr.cc/Conferences/2027/AIPolicyForAuthors

从 2026-08-04 起分别只有 45 和 52 个自然日。倒排硬边界：

| 日期 | 硬交付 |
|---|---|
| 08-07 | 所有只读审计、候选设计和论文一句话主张冻结 |
| 08-10 | 不再引入新的架构方向；P-control、token Gate−1、RL R0 有结论 |
| 08-15 | 唯一主候选必须已有 paired-64 机制信号；否则退出主线 |
| 08-22 | 唯一主候选 paired-256 完成；不得再调方法或阈值 |
| 08-31 | 独立 seed confirmation、共同评测和核心消融完成 |
| 09-05 | 科学结果冻结；之后原则上不启动新的 principal factor |
| 09-12 | 9 页主文、附录、图表、匿名复现包进入只修不扩状态 |
| 09-18 | 提交真实摘要，锁定全部作者 |
| 09-25 | 提交全文与 supplement |

## 2. 冻结事实

### 2.1 现有系统锚点

论文系统不是从零开始。现有锚点为：

```text
P0 Planner
  -> rich Plan / PlanGraph
  -> B0 / R5-C exact-length masked DLM, D1 decoding
  -> frozen CrysLLMGen model_494, exact 800 reverse steps
  -> Direct evaluator + strict/meta S.U.N.
```

现有可支持的核心：

- Plan 与 Body 的分层生成；
- exact `7+4N`、Plan-prefilled count/element、离散 masked DLM Body；
- 冻结连续 refiner；
- conditional post-refiner structure validity 约 `99.7–99.9%`；
- all-attempt、无 retry/replacement/repair/filter/rerank 的证据合同。

### 2.2 当前瓶颈

| 层 | 冻结证据 | 判断 |
|---|---|---|
| Planner composition | H1-A2 `87.8%`；R03 successful-refined `~86%` | 最大明确性能空间 |
| Body completion | R03 D1 pooled `984/1024` | 仍有小空间，但不是主要 validity 瓶颈 |
| Post-refiner structure | D1 `982/984=99.80%`；safe-axis `989/992=99.70%` | 已近饱和 |
| strict S.U.N. | H1 历史 `9.4%`；coverage-adjusted diagnostic `9.71%` | 已强，但不同评测口径不可混表 |
| meta S.U.N. | H1 历史 `47.4%`；coverage-adjusted diagnostic `48.94%` | 仍需提升 |
| safe-axis | completed-snapshot lower bound strict `+18/1024`、meta `-27/1024` | 机制有效但 scientific stop；不作为新 anchor |

Planner invalid taxonomy：

- safe-axis 248 个 successful formula 中 35 个 composition-invalid；
- 其中 charge 24、Pauling 4、其他/标签不一致 7；
- 更早的 1,186 parsed Plans 中 invalid 142：charge 98、Pauling 37、
  oxidation missing 7。

因此：

```text
composition -> Planner 主责
structure validity -> 当前 Body/refiner 已接近上限
meta -> 几何/能量分布与 refiner basin，不能仅归因于 Planner
```

### 2.3 P-control 信号边界

同一 discovery 512 ordinals：

| Planner | Composition valid |
|---|---:|
| P0 | `434/512 = 84.77%` |
| P-control | `456/512 = 89.06%` |
| P* | `442/512 = 86.33%` |

P-control 的 `+4.30 pp` 是 post-selected discovery signal，不是确认结果。
它只可充当低成本 anchor confirmation，不能作为论文核心算法创新。

### 2.4 特殊 token 证据

- tokenizer 定义 2,481 个晶体特殊 token；
- held-out 9,046 rows 中只出现 1,437 个；
- schema-defined numeric stochastic action 2,343 个，held-out unseen 1,013 个；
- 其中 953 个是 LA/LB/LC length token；
- full train coverage 与 legal probability mass 尚未恢复，因此目前不能断言
  `training unseen`，也不能直接删 token；
- 该问题不是当前 composition 或 conditional structure validity 的已证实解释，
  但它是 RL reward exploitation 的明确前置风险。

### 2.5 RL 证据边界

- 旧 TraceRL 缺少 legal-support renormalization、joint token-position
  likelihood、temperature 和 exact replay/resume，正式实验为 NO-GO；
- safe-axis 证明 strict 与 meta 可朝相反方向变化；
- fixed Plan 下 composition 对 Body rollout 无 advantage variance，不能进入
  Body RL reward；
- 新 RL 至少需要 token + reveal-position 的 mask-aware policy likelihood、
  reward/final evaluator isolation 和 meta noninferiority；
- 两个完整 RL 模型不是截止日前的合理默认方案；优先评估一个
  multi-fidelity LoRA，两个小 LoRA 只可作因果诊断。

## 3. 新颖性近邻

必须正面区分以下工作，不能只比较数字：

- CrysVCD：valence-balanced composition transformer + conditional diffusion；
  https://arxiv.org/abs/2507.19799
- PLaID++：Wyckoff text representation + iterative preference alignment；
  https://arxiv.org/abs/2509.07150
- CRYSTAL：autoregressive crystal LLM 的 coordinated multi-objective RL；
  https://openreview.net/pdf/94d95333b625bc19463eca098ff60038d639d590.pdf
- Mask-Aware Policy Gradients for DLMs：token 与 masking/reveal decision 的
  two-stage action；
  https://arxiv.org/abs/2607.15200
- CrysTune：Wyckoff representation、auxiliary tasks 与 RL；
  https://openreview.net/forum?id=Oe5iihLiiV

由此产生的 claim 限制：

1. “先生成组成再扩散结构”不是本项目独有；
2. “保证电荷平衡”不是单独足够的新颖性；
3. “对 crystal LM 做 RL”也不是独有；
4. 可争取的组合差异是 rich PlanGraph、prefix-level chemical reachability、
   exact-length masked DLM、可验证 structured-action likelihood、refiner-aware
   constrained optimization，以及严格 all-attempt 证据；
5. 在完成系统性检索前不得使用 “first”。

## 4. 统一提案评分

每个方向按 1–5 评分：

| 维度 | 问题 |
|---|---|
| Innovation | 相对上述近邻是否有清楚的新方法，而非换领域复用？ |
| Expected effect | 能否直接作用于 comp/meta/strict 中的真实瓶颈？ |
| Evidence | 当前冻结证据是否支持机制，而非只支持愿望？ |
| Feasibility | 代码、模型、评测和数据是否已具备？ |
| Deadline fit | 能否在 08-15 前得到 64 信号、08-22 前完成 256？ |
| Paper coherence | 能否服务一句话主张，而不是增加一个孤立 trick？ |

硬门优先于加权总分。以下任一成立即退出 ICLR 主线：

- 需要同时改变两个以上 principal factors 才可能工作；
- 08-15 前没有 paired-64 机制信号；
- 依赖未冻结或与 final evaluator 相同的 reward judge；
- 需要 retry/replacement/repair/filter/rerank 才得到改善；
- raw all-attempt 改善来自 unary/all-metal、support collapse 或 denominator；
- meta、novelty、uniqueness 或 structure 非劣门失败；
- 无法在共同 evaluator 下与 H1-A2/CrysLLMGen 对齐。

## 5. 投稿必须项与模型候选分离

无论哪个候选胜出，以下均为 MUST：

1. H1-A2、CrysLLMGen 与最终模型同口径 all-attempt 主表；
2. 原论文报告值与本项目重算值分表，逐值记录页码、阈值、分母、refiner、
   novelty database 与 hull snapshot；
3. proposal-before-refine、after-refine 两套结果，避免把 refiner 增益误归于
   Planner/Body；
4. 至少一个真正独立 seed confirmation；
5. 失败 taxonomy、paired discordance、McNemar、paired/hierarchical bootstrap；
6. 方法图、algorithm/pseudocode、token/support 表和 compute 表；
7. 匿名代码/配置/seed/SHA 复现包；
8. 论文骨架从本周开始并行写，不等待最终数字；
9. 按 ICLR 2027 政策如实披露生成式 AI 用于方案、代码、分析和写作，并由
   作者人工核验。

## 6. 第二轮评审必须回答

每位交叉 reviewer 必须给出：

- `KEEP_MAIN / KEEP_BACKUP / APPENDIX_ONLY / CUT`；
- 最早 64 与 256 日期；
- 最大可接受 A800 GPUh；
- 一个最可能成功的机制和一个最可能失败的机制；
- 会不会污染已有 H1 因果归因；
- 相对 CrysVCD/PLaID++/CRYSTAL/mask-aware PG 的真实新颖性；
- 如果 08-15 仍无信号，论文如何退回现有 H1 主张；
- 一句可以放进 9 页主文的贡献，以及一句绝不能写的过度主张。

