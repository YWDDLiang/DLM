# H1 Planner、化学约束与 Body-DLM RL 三方向可行性报告 V1

状态：`decision_only_no_execution_authorization`

日期：2026-08-03

适用基线：H1-A2 `P0 + B0 + CrysLLMGen model_494 exact-800`

本文只完成方案调研、冻结证据复核和实验设计，不启动训练、生成、
refinement、S.U.N.、promotion 或任何自动下游。

## 1. 结论摘要

三个想法都具有研究价值，但成熟度和优先级不同。

| 方向 | 判定 | 建议的第一步 | 现在不应做什么 |
|---|---|---|---|
| 1. 重新考虑 P-control Planner | **CONDITIONAL GO，优先级 1** | 先补齐旧 P-control shortcut/drift 审计；通过后，用全新 1,024 ordinal ledger 独立复核冻结 checkpoint | 不能把旧 `456/512` 直接写成已确认提升；不能重新调 P* 权重 |
| 2. 化学约束进入 Planner | **CONDITIONAL GO，优先级 2** | 实现 formula-line charge-reachability 约束解码；先 32 工程、64 机制、256 科学筛选 | 不做 last-count mask、事后 repair/filter/retry/rerank；第一阶段不同时加入 Pauling 和多项几何约束 |
| 3. 两个 Body-DLM RL 模型 | **CONDITIONAL GO，优先级 3** | 先做 32-Plan、无训练的 refine 前/后 reward 可预测性与 action-logprob 校准 | 不直接复制 CRYSTAL 的 AR-GRPO；不直接用现有 TraceRL 宣称严格 safe-axis RL；不立刻训练两个完整模型 |

总体路线是：

```text
旧 P-control 只读审计
  -> 冻结 P-control 独立确认
  -> 在胜出的 Planner 上单独测试化学可达性约束
  -> Planner 因子冻结
  -> DLM-RL Gate 0（无训练、前后 reward 可预测性）
  -> 仅在 Gate 0 通过后训练两个小 LoRA 做因果诊断
  -> 只扩展胜者
```

最重要的科学判断是：

1. 当前 `comp_valid` 的主要可改进空间属于 Planner，不属于 Body-DLM。
2. 当前 post-refiner `struct_valid` 已接近饱和，DLM-RL 的主要价值应是改善
   stability distribution、refiner basin compatibility 和 meta S.U.N.，而不是
   宣称它能修复 composition。
3. strict S.U.N. 与 meta S.U.N. 已经观察到相反方向，因此后续任何 RL 都必须
   把 meta 设成非劣约束，而不能只把 strict reward 权重调大。
4. 三条路线都必须从 H1 单变量出发；在各自独立通过前，不允许把
   P-control、化学约束、safe-axis 和 RL 一次性叠加。

## 2. 冻结事实与问题定位

### 2.1 Planner 的探索性信号

同一组 512 raw all-attempt ordinals 上：

| Planner | Composition valid | 相对 P0 |
|---|---:|---:|
| P0 | `434/512 = 84.77%` | — |
| P-control | `456/512 = 89.06%` | `+22/512 = +4.30 pp` |
| P* | `442/512 = 86.33%` | `+8/512 = +1.56 pp` |

P-control 值得重新确认，但旧 `+4.30 pp` 是查看三臂结果后识别出的信号，
属于 discovery evidence，而不是 confirmatory evidence。

而且相对 P0，P-control 同时改变了：

- 从 P0 epoch-2 LoRA 继续训练；
- 使用固定的 3,200-row stratified replay；
- 400 updates、LR `2e-6`；
- 使用 field-balanced loss：
  formula `0.35`、chemistry `0.25`、geometry `0.35`、terminator `0.05`。

因此旧结果不能归因为单独一个 loss 改动。相对 P*，P-control 才是“去掉
look-ahead loss”的干净对照。

### 2.2 P* 为什么不值得原样重开

现有证据更像 auxiliary-task negative transfer：

- 七个 auxiliary heads 中五个是 geometry-related；
- formula 到 lattice/spacegroup/volume 是一对多目标；
- smoke 中 look-ahead loss 的数量级远高于 field loss；
- 未使用 gradient norm/cosine balancing；
- `all_metal`、`charge_fail` 等被当作普通标签学习，而不是被惩罚；
- auxiliary heads 在推理时全部丢弃；
- P* 的 target NLL、field loss 和 composition validity 都劣于 P-control；
- P* 还触发了 all-metal shortcut gate。

所以：

> 原样重开 P* 或根据旧 512 结果调整其 loss 权重，均为 NO-GO。

若未来重新研究辅助规划目标，应先在冻结 validation batches 上测各任务对
LoRA 参数的 gradient norm 和 cosine，再预注册 PCGrad、GradNorm 或其他
冲突处理方法，不能边看生成结果边调权重。

### 2.3 当前 end-to-end 瓶颈

R03 safe-axis candidate 的 pooled refined denominator 为 992：

- composition valid：`852/992 = 85.8871%`
- structure valid：`989/992 = 99.6976%`
- joint valid：`851/992 = 85.7863%`

35 个 composition-invalid successful proposals 中：

- 24 个 charge failure；
- 4 个 Pauling failure；
- 7 个为其他 Direct composition-invalid。

而 1,024 raw candidate attempts 中：

- 8 个 Planner failure；
- 24 个 body failure；
- 只有 3 个真正的 post-refiner structure-invalid。

这说明：

```text
comp_valid 的主问题 = Planner formula/chemistry
struct_valid 的主问题 ≠ 当前 frozen DLM/refiner
meta 的问题 = 完整几何/能量分布，不能只归因于 Planner
```

### 2.4 strict/meta 的关键反例

R03G lower-bound pooled endpoint：

| Endpoint | Control | Candidate | Candidate - Control |
|---|---:|---:|---:|
| strict S.U.N. | `99/1024 = 9.67%` | `117/1024 = 11.43%` | `+18/1024 = +1.76 pp` |
| meta S.U.N. | `523/1024 = 51.07%` | `496/1024 = 48.44%` | `-27/1024 = -2.64 pp` |

R03H 证明 meta 的 `-27` 全部来自 finite `E_hull <= 0.1` threshold
crossings，而不是 residual unknown 或 composition label 变化。

因此后续 reward 不能把 strict 当成 stability distribution 的充分统计量。

完整冻结证据见
[H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md](H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md)。

## 3. 方向一：重新考虑 P-control Planner

### 3.1 判定

**CONDITIONAL GO。**

P-control 是当前最便宜、最有信息量的候选，因为：

- 已有冻结 checkpoint；
- 不需要重新训练；
- 推理 schema、prompt、parser、采样参数和 body interface 都不变；
- 旧结果有 `+4.30 pp` 的有意义探索性信号；
- P-control 同时比 P* 有更好的 composition、NLL 和 field loss。

它不能直接晋级，因为：

- 旧结果是 post-selection；
- 缺少正确的 paired discordance/McNemar 摘要；
- 当前本地证据未完整披露 P-control 的 all-metal、single-element、
  formula uniqueness 和全部 TVD；
- composition evaluator 会将 unary 和某些 all-metal composition 计为 valid，
  存在 shortcut 获利风险。

### 3.2 Stage 0：旧 P-control 只读审计

不调用模型、不改变历史文件，只恢复并核验：

- `Pcontrol-only valid` 与 `P0-only valid` discordant counts；
- exact McNemar；
- all-metal 和 single-element counts；
- unique formula rate 和 top-1 formula frequency；
- mean atom count；
- 已注册的 11 个 marginal TVD；
- invalid reason taxonomy；
- train formula exact/reduced-composition overlap；
- 模型生成的 `anion/charge` 与 formula 重算标签的一致率。

Fail-close：

- all-metal/single-element shortcut 膨胀；
- 任一注册 TVD 相对 P0 恶化超过 `0.02`；
- unique-formula 或元素覆盖出现明显坍缩；
- 原始 attempt/ordinal/seed 证据无法恢复。

任一项成立，当前 P-control checkpoint 直接停止。

### 3.3 Stage 1：独立 Plan-only confirmation

若 Stage 0 通过：

- P0 与冻结 P-control 各 `1,024` raw attempts；
- 使用全新的、预先 SHA 冻结的 base-seed/ordinal ledger；
- 每 ordinal 两臂共享 Planner RNG role；
- effective batch size 1；
- checkpoint、prompt、schema、parser、temperature、top-p、top-k 全冻结；
- 旧 512 不进入 primary test；
- 无 retry、replacement、repair、filter、rerank。

Primary gate：

- raw all-attempt composition gain `>= +3.0 pp`；
- paired 95% CI 下界 `> 0`；
- exact McNemar 完整报告；
- parse/completion drop `<= 0.5 pp`；
- unique-formula rate `>= 95%` of P0；
- `|delta mean N| <= 0.5`；
- 每个注册 TVD worsening `<= 0.02`；
- all-metal、single-element 不膨胀；
- formula novelty 和 formula↔charge/anion coherence 不劣化。

判定：

- 通过：P-control 成为“待下游验证的 Planner 候选”，仍不是新 baseline。
- 未通过：保留 P0，不改 seed，不调 P-control，不做补充筛选。

### 3.4 Stage 2/3：下游但仍保持单变量

只有 Plan-only 独立通过后：

1. `P0/P-control + 同一 H1 D1 + B0` paired 64；
2. 通过后 paired 256；
3. exact model_494 refine800；
4. 相同 Direct 和 common-snapshot S.U.N.。

先使用 D1，而不是把 P-control 与 safe-axis 立刻结合。safe-axis 的 strict
改善和 meta 损失已经构成独立 scientific stop；直接叠加会失去因果归因。

下游 gate 建议：

- exact `7+4N` 100%；
- body completion drop `<=2 pp`；
- duplicate-coordinate 不增加；
- 无新 failure class；
- raw joint validity `>= P0 +2 pp`；
- conditional structure validity `>=99.5%`；
- strict 非劣；
- meta 至少 `+2 pp`，或 paired CI 明确为正；
- unique/novel 不下降超过 `2 pp`。

### 3.5 如果 P-control 确认通过，下一种 Planner 应是什么

优先级：

1. 本文方向二的 formula chemical reachability；
2. formula-first anchored continuation：
   先增大 formula loss，再退火回 P-control 权重，并用 teacher-token KL 锚定
   其他字段；
3. typed two-stage Planner：
   第一个 adapter 只生成一次 formula，第二个 adapter 在冻结 formula prefix 上
   生成其余六行。

第 2、3 项均不应在 P-control 独立确认前训练。

## 4. 方向二：化学约束进入 Planner

### 4.1 判定

**CONDITIONAL GO。**

首选不是“最后一个 count hard mask”，而是：

> Formula-line charge-reachability product automaton  
> 加上独立验证的派生化学字段 compiler。

两部分必须分开测试：

- formula gate 主要针对 `comp_valid`；
- compiler 主要针对 formula/anion/charge 的条件一致性，可能影响下游
  geometry 和 meta；
- 它们对 `struct_valid` 的目标是非劣，而不是承诺直接提升。

### 4.2 当前 Planner 化学接口的问题

当前七行 Plan 中：

- `formula` 由模型生成；
- `N/elements/counts` 由 Python 从 formula 推导；
- `anion` 和 `charge` 却直接信任模型自报；
- `oxidation_candidates` 常为 `unknown`，进入 PlanGraph 后成为空列表；
- PlanGraph 检查类型和长度，但没有验证
  `sum_i n_i q_i = 0`。

因此目前存在“formula 是一种化学，metadata 又声称另一种化学”的风险。

### 4.3 候选 A：formula charge-reachability product automaton

第一阶段只在 `formula:` 行干预。模型权重、prompt、七行 schema、body、
refiner、evaluator 和 seed ledger 均冻结。

解码状态由两部分组成。

#### Formula 语法状态

- 元素符号；
- integer count；
- 当前总原子数；
- 保持现有 parser 接受的语言；
- 保证 `1 <= N <= 20`。

#### 化学可达性状态

- 已完成的 element/count pairs；
- 冻结版 SMACT oxidation-state table；
- 当前可实现的总电荷集合；
- 使用 memoized dynamic programming 判断：
  当前 composition 已经电荷中性，或在剩余原子预算内仍存在中性 completion。

只有存在真实 oxidation assignment 且满足
`sum_i n_i q_i = 0` 时，才允许从 formula 行转到下一行。否则，只允许仍能
到达某个中性 completion 的 token。

第一阶段的边界：

- 只约束 charge neutrality；
- 暂不加入 Pauling；
- unary/all-metal shortcut 不作为可达性证明；
- 只进行一次 renormalized sampling；
- 无 beam、candidate pool、retry、replacement、rerank；
- 无合法 token 时，该 raw attempt 直接失败；
- 不 silent fallback 到 unconstrained generation；
- lattice、spacegroup、volume 暂不约束。

为什么不能只 mask 最后一个 count：

- tokenizer 可能拆分 element 和 integer；
- 中性取决于全部 oxidation-state combination；
- 某个已选元素集合可能根本不存在合法最后 count；
- 只阻止 newline 会把模型推入 dead-end 或低概率乱码区。

输入相关 grammar/constrained decoding 的可行性可参考
[Grammar-Constrained Decoding](https://aclanthology.org/2023.emnlp-main.674/)；
但 grammar 保证不等于化学有效，且 hard mask 会改变原始模型分布，所以必须
同时报告 diversity 和 marginal drift。

### 4.4 候选 B：派生化学字段 compiler

formula 完成后，用冻结函数推导：

- `anion_framework`；
- `charge_bucket`；
- 一个可验证的 `oxidation_candidates` certificate。

模型接下来的 `anion:`、`charge:` 只允许输出与 formula/compiler 一致的
canonical string；lattice、spacegroup、volume 仍由模型生成。

预期作用：

- 不修复已经 invalid 的 formula；
- 消除 self-reported chemistry 与 formula 的不一致；
- 给 body 一个可审计的 chemistry condition；
- 可能改善 meta，但当前没有证据保证这一点。

特别注意：

> R03 的 meta `-27` 来自相同 formula 下的 finite-hull crossings，  
> 所以 compiler 不能被宣传为已经解释或必然修复 R03 meta 损失。

### 4.5 单变量验证顺序

不建议一开始做 combined arm。按以下顺序：

1. `C0`：现有 frozen ledgers 的只读 compiler/parity audit；
2. `C1`：unconstrained formula vs charge-reachability，仅改变 formula decoding；
3. `C2`：在同一 frozen formula source 上，self-reported fields vs compiler fields；
4. 只有 C1、C2 分别通过，才注册 combined candidate；
5. combined candidate 仍需从 paired 64 重新开始。

这比一次性四臂叠加更符合现有 H1 单变量规则。

### 4.6 32 -> 64 -> 256 gate

#### 32：工程正确性

- 逐 ordinal 配对；
- 0 tokenizer/FSM 异常；
- 0 silent fallback；
- 所有 terminal constrained formulas 均有真实 neutrality certificate；
- constraint oracle 与冻结 composition evaluator 逐公式 parity；
- 七行 parser、raw/canonical SHA、attempt denominator 完整；
- 记录 blocked-token、blocked-newline、mask activation、cache hit 和 mask entropy。

#### 64：机制筛选

- raw comp-valid 至少 `+3/64`；
- charge failure 至少下降 50%；
- parse/completion 最多下降 1 个；
- single-element 最多增加 1 个；
- all-metal 最多增加 2 个；
- shortcut 对总 comp gain 的贡献不超过 10%；
- unique formula `>=95%` of control；
- `|delta mean N| <=0.5`；
- compiler arm 的 formula↔anion/charge 一致率接近 100%。

#### 256：科学筛选

- raw comp-valid 至少 `+3 pp`，即至少 `+8/256`；
- charge failure 至少下降 50%；
- raw structure validity 非劣 1 pp；
- strict S.U.N. 不下降；
- meta 至少 `+2 pp`，或 paired CI 明确为正；
- formula/structure unique 和 novel 不下降超过 2 pp；
- top-1 formula、arity、元素边缘分布、anion 分布无坍缩；
- 报告 McNemar、paired bootstrap 和完整 reason taxonomy。

冻结候选后，需要四个真正独立的 256 scientific ledgers：

- comp-valid 至少 3/4 为正，均值 `>=+3 pp`；
- meta 至少 3/4 非负；
- structure、strict 均非劣；
- 每个 ledger 都通过 shortcut/diversity gate。

### 4.7 明确 NO-GO

- 仅做七行语法 FSM；
- 只 mask 最后一个 count 或 newline；
- generation 后 repair formula；
- filter invalid composition；
- retry、replacement、candidate rerank；
- 一开始同时引入 charge、Pauling、space group、volume 和 RL；
- 只看 conditional comp-valid，不看 raw all-attempt 和 shortcut。

## 5. 方向三：Body-DLM 做 RL，训练 refine 前/后两个模型

### 5.1 判定

**两个小 LoRA 作为因果诊断：GO。**  
**立即训练两个完整长期模型：NO-GO。**  
**整体方向：CONDITIONAL GO。**

这个提议的科学问题是合理的：

- 模型 A：学习 proposal 本身是否更接近物理可接受 basin；
- 模型 B：学习 proposal 是否更适合当前 frozen model_494 refiner。

两者的差别能回答：

> RL 的收益来自离散 proposal 自身，还是来自对特定 refiner 的适配？

但是否真的需要长期保留两个模型，必须由 A reward 对 B endpoint 的预测性决定。

### 5.2 Body-DLM 不能优化 composition

同一个 Plan 的 body rollouts 中：

- atom count 已预填；
- element sequence 已预填；
- formula/composition 已冻结。

所以 comp-valid 在同一个 GRPO group 内是常数，advantage 为零。

结论：

- composition 只作为 Plan eligibility 和最终审计项；
- Body-DLM RL 不能宣称提升 comp-valid；
- comp-valid 改进必须来自方向一、二；
- DLM-RL 应优化 completion、pre/post structure validity、refiner compatibility、
  stability、novelty、uniqueness 和 OOD robustness。

### 5.3 当前 masked diffusion 的 RL action

当前 body sampler：

- answer suffix 初始全部为 MASK；
- 每步对剩余 masked positions 预测 token；
- safe-axis 限定 legal position group；
- `low_confidence` 根据候选 token confidence 选择本步真正提交的位置；
- schema、duplicate-coordinate、volume 等约束会改变合法 action support。

严格 state/action 应定义为：

```text
state =
  Plan + prompt + partial sequence + group/step + all legal masks

action =
  selected reveal position + selected token
```

只记录最终 token 的 log-prob 不够，因为“哪个位置先揭示”也是策略的一部分。
[Mask-Aware Policy Gradients](https://arxiv.org/abs/2607.15200) 将 masked
diffusion generation 建模为 token action 与 masking/unmasking action 的两阶段
MDP，正好对应这一问题。

现有 `scripts/llada_trace_rl.py`：

- 能重建部分 masked state；
- 能对 reveal token 做 PPO-like ratio；
- 但忽略 reveal-position policy；
- step map 缺失时存在近似顺序。

所以它可以作为工程原型，但不能直接作为“严格 safe-axis RL”的论文实现。

可选路径：

1. 最小 pilot：冻结一个预声明的 safe-axis legal site order，只训练 token
   policy；先证明这种固定顺序不伤当前 endpoint；
2. 正式方案：在 safe-axis hard constraints 内显式参数化
   Plackett-Luce/unmask policy，并同时记录 token 与 position log-prob；
3. 若采用 likelihood-free 方法，必须单独验证其在结构化晶体 mask 上的稳定性，
   不能因为避免 log-prob 就省略 action-support audit。

### 5.4 模型 A：不经过 model_494 的 reward

“不经过 refine”有两种语义，必须预注册。

#### A0：完全不做任何 relax

可用：

- exact length、parse、Plan match、graph completion；
- pre-structure validity margin；
- minimum distance margin；
- volume/atom；
- frozen MLIP single-point energy/forces；
- local coordination；
- proposal OOD score。

不能直接称为 stability reward，也不能直接计算可信 `E_hull`。

#### A1：不经过 model_494，但允许 cheap proxy relaxation

可用：

- frozen proxy MLIP short-relax success；
- maximum force；
- RMS displacement；
- volume strain；
- short-relax energy/hull proxy；
- 对 B reward 的离线 surrogate。

A1 更有预测力，但必须清楚写成“跳过 model_494”，不能写成“完全未 refine”。

模型 A 的主要风险：

- 离散 proposal 对 MLIP OOD；
- proxy reward 改善但 model_494 后消失；
- 过度优化 pre-valid geometry，损失 diversity 或可 refine 性。

### 5.5 模型 B：经过 frozen model_494 exact800 的 reward

B 与部署目标更直接，但成本和方差更高。

应包含：

- model_494 completion hard gate；
- post-refiner structure validity；
- exact800 后的 stability tier/continuous score；
- proposal→refined RMSD、volume change、local-environment change；
- refiner seed sensitivity；
- 相同 attractor 重复率；
- novelty 和对称 batch uniqueness；
- OOD penalty。

不能取多个 refiner seed 的最大 reward；应使用均值、下置信界或预注册的
robust aggregate，防止 seed cherry-picking。

### 5.6 Reward 不能直接复制 CRYSTAL

[CRYSTAL](https://openreview.net/pdf/94d95333b625bc19463eca098ff60038d639d590.pdf)
提供了有价值的多目标思路：

- validity、stability、novelty、diversity/uniqueness 联合优化；
- 使用 multiplicative aggregation 避免某个目标完全补偿另一个目标；
- GRPO 组内标准化。

但不能直接照搬：

- CRYSTAL 是 autoregressive policy；本项目是 masked diffusion；
- CRYSTAL 的 composition 是模型 action；本项目 body composition 已由 Planner
  冻结；
- 当前 strict 约 10%，多个 binary reward 相乘会产生大量全零 group；
- sequential first-seen uniqueness 有顺序偏差；
- CRYSTAL 表中的 `25.1% S.U.N.` 使用更宽的稳定性口径，不能当作本项目
  strict `E_hull <=0` 比较值。

[PLaID++](https://arxiv.org/html/2509.07150v4) 的两个经验更应直接吸收：

- reward 优化可能提高 stability，却降低 S.U.N./diversity；
- reward evaluator 与 final evaluator 应隔离，并用独立 evaluator/DFT
  检查 reward hacking。

### 5.7 推荐 reward：约束式，而不是一个固定加权和

保留完整向量：

```text
z = (
  completion,
  pre_valid,
  post_valid,
  meta_quality,
  strict_quality,
  novelty,
  symmetric_uniqueness,
  -OOD,
  -refiner_displacement,
  -refiner_seed_sensitivity
)
```

composition validity 单独记录为 `Q_plan`，不进入同 Plan 的 body advantage。

优化目标：

```text
maximize:
  strict_quality + novelty + uniqueness - OOD penalties

subject to:
  meta >= B0_meta - delta_meta
  completion >= B0_completion - delta_completion
  validity >= B0_validity - delta_validity
```

checkpoint selection 必须 lexicographic：

1. completion、validity、meta 未过非劣门槛的 checkpoint 全部拒绝；
2. 只在幸存 checkpoint 中比较 strict；
3. 再用 novelty、uniqueness、OOD 做 tie-break/audit；
4. final evaluator 不参与 reward tuning 或早停。

这直接防止再次出现“strict 上升、meta 明显下降”。

### 5.8 Reward evaluator 与 final evaluator 隔离

新的 RL workstream 必须改写现有 V3 的训练规则。当前 V3 明确禁止
S.U.N.、MP、energy、hull、CHGNet 或 generated-crystal result 用于训练或
checkpoint selection。

因此 DLM-RL：

- 不能偷偷挂接到现有 V3；
- 必须使用新的 protocol amendment/workstream；
- 训练 reward 可用冻结 eSEN/eqV2、其 ensemble 或 B-teacher surrogate；
- final evaluation 保持当前 frozen Direct + CHGNet/MP snapshot，且不用于
  调 reward 权重；
- 若训练必须使用 CHGNet，则 final claim 至少需要独立 eSEN/DFT holdout。

### 5.9 两模型的最小决策实验

#### Gate 0：32 Plans，无训练

- 32 个 frozen、comp-valid Plans；
- 按 N、arity、chemistry family 分层；
- 每 Plan `G=4` body rollouts，共 128 proposals；
- 保存真实 state、legal mask、token action、position action、old log-prob、
  reveal order、RNG seed；
- A 对全部 128 评分；
- B 对同一 128 进行 model_494 exact800；
- 对边界/OOD 子集增加预注册的 refiner-seed repeats。

必须通过：

- old-policy replay ratio 接近 1；
- 所有 log-prob finite；
- action support 完全一致；
- 至少一半 Plan groups 有非零 reward variance；
- A 对 B 的 Spearman/bootstrap correlation 为正；
- A top quartile 对 B-meta/B-strict 有明确 enrichment；
- strict/meta refiner flip rate 和 seed variance 被量化。

决策：

- A→B 预测性强：进入两个小 LoRA 诊断；
- A→B 预测性弱：不训练 A，转 B-only 或 B-teacher surrogate；
- action likelihood 不闭合：RL 工程 NO-GO，先修 trajectory semantics。

#### Gate 1：64 Plans，两个小 LoRA

- A、B 均从同一 B0 初始化；
- 相同 Plans、rollout 数、group size、optimizer、LoRA rank、update budget；
- 唯一变量是 reward 位于 model_494 前或后；
- 只训练 body LoRA，Planner、base、model_494、reward models 全冻结；
- 建议从非常小的 LR 和 1–2 次 online refresh 开始；
- fresh 64 ordinal holdout 同时评估 B0、A、B；
- 每个模型都同时报告 pre-refine 与 exact800 post-refine endpoints。

立即停止：

- meta pooled effect 为负；
- completion/joint-valid 明显下降；
- unique/novel 下降超过预注册 `2 pp`；
- reward proxy 上升而 held-out final evaluator 下降；
- OOD、refiner failure、displacement 或 seed sensitivity 上升；
- KL 超过目标两倍或 importance-ratio ESS 崩溃。

#### Gate 2：只扩展胜者

- fresh 256 all-attempt ordinals；
- 3 套独立 body/refiner scientific seed ledgers；
- failures/unknown 全部计 false；
- 无补样、repair、filter、rerank。

继续条件：

- meta 三套 ledger 均非负；
- strict 至少 2/3 为正；
- pooled meta 非负；
- completion、validity、novelty、uniqueness 和 OOD 全过门槛；
- 正式 promotion 时 meta noninferiority tolerance 收紧到 0。

256 仍只是 pilot confirmation。论文级 claim 建议最终使用至少 1,024 raw
paired attempts，并预先冻结 evaluator、threshold、snapshot 和统计方法。

### 5.10 两个模型之后如何收敛

两个独立 LoRA 的长期决策：

| Gate 0/1 结果 | 最终策略 |
|---|---|
| A 能稳定预测 B，且 A-model 的 post-refine 也改善 | 一个模型做 `A -> B` curriculum |
| A 与 B 相关性弱 | 放弃 A，B-only |
| B 标签昂贵但可预测 | 用 B 标签训练带 uncertainty/OOD rejection 的 surrogate，再周期性真实 B 查询 |
| A、B 优化方向冲突 | 保留两者作为论文 ablation，不作为双生产模型 |
| B 对固定 refiner 过拟合 | 增加 seed robustness，并用独立 refiner/evaluator holdout；未解决前停止 |

另一个高效选项是 mixed-fidelity：

- 所有 rollout 得到 A reward；
- 随机预注册子集得到 B reward；
- 用 A 作为 B advantage 的 control variate。

这通常比长期维护两个完整模型更经济。

## 6. 三条路线的统一优先级

### Phase A：Planner 先行

1. P-control 旧证据只读审计；
2. 冻结 P-control 1,024 Plan-only independent confirmation；
3. 只在通过后进入 D1+B0 下游；
4. 根据结果冻结 P0 或 P-control 为新的 Planner anchor。

原因：Body-DLM 无法改变 formula，当前最大的 raw validity 空间在 Planner。

### Phase B：化学可达性

1. 只读 compiler/parity audit；
2. formula reachability 32；
3. 64；
4. 256；
5. compiler 作为另一单变量轴；
6. 两者各自通过后才组合。

原因：这是最直接处理 24 个 charge failure 的机制，且不需要先训练新 Planner。

### Phase C：DLM-RL

1. 注册新 RL workstream 和 evaluator isolation；
2. Gate 0 reward/action calibration；
3. 两个小 LoRA Gate 1；
4. 只扩胜者；
5. 最终决定 curriculum、B-only 或 surrogate。

原因：DLM mechanics 已经较强，RL 的潜在价值主要在 meta/stability；但它也是
三条路线中算法和 reward-hacking 风险最高的一条。

## 7. 统一实验纪律

所有方向共同遵守：

- raw all-attempt denominator 为主；
- conditional 指标只作诊断；
- ordinal、attempt、seed、raw/canonical Plan SHA 全保留；
- no sample ID；
- no retry、replacement、repair、filter、rerank；
- 一次只改变一个 principal factor；
- 32/64 工程机制通过后才到 256；
- 每次 expansion 使用 fresh、预先 SHA 冻结的 ledger；
- 失败后不换 seed；
- exact-length `7+4N` 保持；
- 所有 body-success proposal 必须通过 frozen model_494 exact800 后才进入正式
  Direct/S.U.N. endpoint；
- single-element/all-metal shortcut 单独披露；
- final reward/evaluator/threshold/snapshot 不根据结果回调；
- no automatic promotion、training feedback 或 downstream。

### 7.1 统计报告

每个科学对比至少报告：

- raw counts 和 rates；
- paired discordance；
- exact McNemar；
- paired bootstrap interval；
- reason taxonomy；
- per-seed sign stability；
- pooled result；
- diversity/shortcut/OOD；
- unknown 处理；
- raw 与 conditional 两种 denominator。

### 7.2 真正独立重复

R03E 四个 CUDA repeats 复用了同一个 scientific seed ledger，所以不能当作
四个独立 scientific samples。未来 confirmatory study 必须使用真正不同的、
预先冻结的 scientific base-seed ledgers。

## 8. 与论文贡献的关系

如果三条路线都产生正结果，论文故事不是“把很多 trick 叠起来”，而是一个
清晰的分层归因：

1. **Planner chemistry** 决定可行 composition；
2. **exact-length masked DLM** 在固定 composition 下生成高完成率 proposal；
3. **chemistry-reachable planning** 消除无效 action support；
4. **refiner-aware RL** 优化 proposal 进入稳定 basin 的方式；
5. **strict/meta constrained optimization** 防止 endpoint polarization；
6. 全流程用 all-attempt、common snapshot 和 independent evaluator 防止
   denominator/reward hacking。

即使最终只有前两条成功，也足以形成有价值的工程与科学结论。DLM-RL 不应被
视为投稿的必需条件；它是高风险、高上限的增强项。

## 9. 最终决策

### 可以尝试

- 冻结 P-control 的独立确认；
- formula-line charge-reachability constrained decoding；
- formula-derived anion/charge/oxidation compiler，但作为单独因子；
- refine 前/后两个小 LoRA 的诊断性 RL；
- 约束式 strict/meta reward；
- reward evaluator 与 final evaluator 隔离；
- Gate 0 先验证 A reward 是否预测 B。

### 暂不尝试

- 原样或调权重重开 P*；
- 一次性组合 P-control + chemistry constraint + safe-axis + RL；
- last-count hard mask；
- post-hoc chemistry repair/filter/retry/rerank；
- 用 Body-DLM RL 声称提升 composition；
- 直接复制 CRYSTAL 的 AR-GRPO 和 reported S.U.N. 阈值；
- 直接用现有 TraceRL 宣称 exact safe-axis policy gradient；
- 永久维护两个完整 RL 模型；
- 用训练 reward evaluator 同时作最终科学评价。

### 最有价值的下一步

> 先完成 P-control 旧 512 的 shortcut/drift/paired 只读审计。  
> 审计通过后，注册一个全新 1,024-ordinal 的 P0 vs frozen P-control
> Plan-only confirmation；此阶段不运行 body、refiner 或 S.U.N.。

这一步成本最低、因果最清楚，并直接决定后续化学约束应以 P0 还是
P-control 为 Planner anchor。

## 10. 主要外部依据

- [Mask-Aware Policy Gradients for Diffusion Language Models](https://arxiv.org/abs/2607.15200)
- [CRYSTAL: Coordinated Multi-Objective Reinforcement Learning for Crystal Generation](https://openreview.net/pdf/94d95333b625bc19463eca098ff60038d639d590.pdf)
- [PLaID++: A Preference Aligned Language Model for Targeted Inorganic Materials Design](https://arxiv.org/html/2509.07150v4)
- [Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning](https://aclanthology.org/2023.emnlp-main.674/)
- [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063)
- [Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://proceedings.mlr.press/v80/chen18a.html)
- [Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)

