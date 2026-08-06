# Planner / Chemistry 论文主线提案 V1

Status: `decision_only_conditional_go`

Date: 2026-08-04

Scope: Planner 与 Planner→Body chemistry interface；不修改 Body-DLM、
continuous refiner、S.U.N. evaluator 或任何冻结实验。

Authorization: 本文只提出方法、资源和门禁，不授权训练、生成、API 查询、
checkpoint promotion 或 automatic downstream。

## 0. 一页决策

### 0.1 推荐路线

在 ICLR 2027 截止前，建议把候选工作分成四个角色，而不是把四件事包装成
一个方法：

| 路线 | 论文角色 | 当前决定 |
|---|---|---|
| 冻结 P-control 独立确认 | **支撑性确认 / anchor 选择** | `CONDITIONAL GO`，但绝不是核心创新 |
| Formula-prefix charge-reachability product automaton | **唯一主贡献候选** | `CONDITIONAL GO`，先过 novelty kill gate，再走 32→64→256→4×256 |
| Formula-derived chemistry compiler | **可选第二贡献 / 机制消融** | `CONDITIONAL GO`，只有现存字段不一致有足够 headroom 且独立改善 meta 才进入主文 |
| 训练型 Planner 改进 | **失败备线** | `DEFERRED CONDITIONAL GO`，仅当 hard constraint 因模型 support mismatch 失败时启动 |

建议的最小论文系统是：

```text
冻结选定 Planner anchor
  + formula-prefix charge-reachability
  + 原 H1 D1 / B0 / model_494 refine800
```

它不依赖 compiler，也不依赖新的 Planner 训练。Compiler 和训练型候选必须
各自独立通过后才能讨论组合；最小投稿不需要二者。

### 0.2 为什么这是当前最合理的主线

1. 当前 R03 safe-axis 的 conditional composition validity 是
   `852/992 = 85.8871%`，35 个 invalid 中有 24 个 charge failure；这是
   最大且有明确机制的可改进桶。
2. conditional structure validity 已达 `989/992 = 99.6976%`，真正
   post-refiner structure-invalid 只有 3 个。Planner 方法不应承诺显著提高
   conditional `struct_valid`，其目标应是非劣以及提高 raw joint-valid。
3. safe-axis 的 strict S.U.N. 为 `+18/1024`，但 meta S.U.N. 为
   `-27/1024`；因此 safe-axis 不能直接成为新 Planner 研究的“稳定正向”
   anchor。
4. P-control 的旧 `456/512 = 89.06%` 比 P0 高 `4.30 pp`，但它是
   post-selection 结果，而且缺 shortcut、drift 和 paired discordance 的
   完整审计。它适合选 anchor，不足以形成论文方法贡献。
5. Body special-token audit 在本地 held-out exact corpus 中发现
   `1,013/2,343` 个 stochastic-action special tokens 未出现，其中 953 个是
   axis-specific length tokens。在完整 train coverage 未审计前，本轮不应把
   Body-RL 或 token representation 改造抢成主线。

### 0.3 两个硬性前置条件

主方法必须同时过两个前置条件：

- **Novelty gate，最迟 2026-08-08：** CrysVCD 已经提出
  “valence-constrained composition generator + diffusion structure
  generator”。不能再宣称“首次将价态约束用于晶体生成”。只有当逐 token
  prefix product automaton、有限原子预算下的 charge reachability、
  proof-carrying rich Plan、单次 renormalized sampling 和 all-attempt
  因果评估构成实质差异时，主线才保留。
- **64 gate，最迟 2026-08-14：** charge failure 至少下降 50%，raw
  comp-valid 至少增加 `3/64`，且不靠 unary/all-metal shortcut。若未通过，
  它不能再占用主线时间。

### 0.4 截止时间

本提案按以下官方截止倒排：

- 摘要：2026-09-18 23:59 AOE，即北京时间 2026-09-19 19:59；
- 全文：2026-09-25 23:59 AOE，即北京时间 2026-09-26 19:59。

为了给统计、作图和写作留下缓冲，内部硬 cutline 是：

- main-method paired-64：**2026-08-14**；
- main-method paired-256：**2026-08-22**，最迟不可晚于 08-24；
- 四个独立 `256` ledger：**2026-09-05**；
- 最终结果、claim 和表格冻结：**2026-09-10**。

## 1. 审计边界与冻结事实

本提案只读核对了：

- [H1 Planner / Chemistry / DLM-RL feasibility report](../H1_PLANNER_CHEMISTRY_DLM_RL_FEASIBILITY_REPORT_V1.md)；
- [R03 safe-axis reproducibility report](../H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md)；
- [Body special-token coverage audit](../H1_BODY_SPECIAL_TOKEN_COVERAGE_AUDIT_V1.md)；
- [Experiment TODO index V3](../EXPERIMENT_TODO_INDEX_V3.md)。

### 1.1 P-control 证据边界

Plan-only 512 的冻结结果为：

| Planner | Raw composition valid | 相对 P0 |
|---|---:|---:|
| P0 | `434/512 = 84.77%` | — |
| P-control | `456/512 = 89.06%` | `+4.30 pp` |
| P* | `442/512 = 86.33%` | `+1.56 pp` |

但 P-control 相对 P0 同时改变了 continued training、3,200-row stream、
400 updates、LR `2e-6` 和 field-balanced loss。因此，即使新确认复现，
论文也只能说“这个冻结 recipe/checkpoint 有效”，不能把提升单独归因给
field-balanced loss。

P* 已因增益不足、低于 P-control、all-metal shortcut 膨胀而停止；本提案
不重新调 P* head 权重，也不把 look-ahead 原样重开。

### 1.2 R03 对指标优先级的约束

R03 safe-axis 的 pooled raw Direct 结果为：

| 指标 | D1 control | safe-axis | 差值 |
|---|---:|---:|---:|
| generation complete | 984 | 992 | +8 |
| composition valid | 848 | 852 | +4 |
| structure valid | 982 | 989 | +7 |
| joint valid | 846 | 851 | +5 |

这些 raw structure 差值主要来自 upstream completion，不是 refiner 几何
质量大幅改变。Conditional 统计中：

- comp-valid：`852/992 = 85.8871%`；
- struct-valid：`989/992 = 99.6976%`；
- 35 个 comp-invalid = 24 charge + 4 Pauling + 7 other；
- 35 个 raw structure-invalid = 8 Planner failure + 24 body failure +
  3 true post-refiner structure failure。

因此本轮优先级应是：

```text
Planner comp_valid / raw joint
  > meta noninferiority or gain
  > strict noninferiority
  > conditional struct_valid 的微小 ceiling gain
```

`struct_valid` 必须同时报告 raw 与 conditional。把 upstream completion
改善包装成 conditional geometry 改善是不正确的。

### 1.3 Safe-axis 不能直接交叉

R03G lower-bound 结果：

- strict：`99 → 117/1024`，`+1.7578 pp`；
- meta：`523 → 496/1024`，`-2.6367 pp`。

R03H 已证明 meta 损失来自相同公式下的 finite-hull crossings，不是 residual
unknown，也不是 formula label 改变。因此：

- 新 Planner 实验先固定 H1 D1、B0、model_494；
- 不把 P-control、charge constraint、compiler 与 safe-axis 一次叠加；
- 只有某个 Planner 因子独立通过后，才可另行注册 interaction study。

### 1.4 独立重复的定义

R03E 的四次 CUDA process repeat 复用了同一 scientific ordinal/seed
ledger，它们不是四个独立科学样本。本文的 `4×256` 指：

- 四个新 base seed；
- 四个互不重叠、预先冻结的 ordinal ledger；
- 每个 ledger 都有 control/candidate 配对；
- checkpoint、代码、阈值、evaluator 和 denominator 完全相同；
- 不能按结果替换某次重复。

## 2. 核心研究问题与因果顺序

### 2.1 主假设

主假设不是“加入更多化学规则总会更好”，而是：

> 在一次、无 repair/retry/rerank 的 Planner 采样中，把 formula token 的
> 合法支持限制为仍可到达真实 charge-neutral witness 的前缀，可以减少
> Planner charge failure，并在不造成分布坍缩的情况下提高 raw
> composition/joint validity。

这是一条可证伪的机制假设。Meta/strict 改善是需要实测的 downstream
结果，不从 charge neutrality 自动推出。

### 2.2 固定 anchor 的顺序

顺序必须是：

```text
A. 只读审计旧 P-control
  -> B. 新 1,024 Plan-only P0 vs frozen P-control
  -> C. 一次性冻结 P0 或 P-control 为 anchor
  -> D. anchor vs anchor + reachability
  -> E. 可选：在相同 frozen formula source 上单测 compiler
```

在 D 之前不允许根据 reachability 的结果重新选择 anchor。P-control 若未
通过，D 直接使用 P0；这不杀死主方法。

### 2.3 单因素定义

每一阶段只允许一个 principal factor：

| 阶段 | Control | Candidate | 唯一变化 |
|---|---|---|---|
| P-control confirmation | P0 | frozen P-control | Planner checkpoint/recipe |
| C1 reachability | frozen anchor | anchor + reachability | formula-line legal support |
| C2 compiler | self-reported fields | compiled fields | anion/charge semantic source |
| T1 training backup | frozen anchor | trained Planner | training objective/checkpoint |

Reachability 与 compiler 不能在第一次实验中合并。训练候选与 hard decoding
也不能在第一次实验中合并。

## 3. 路线 A：冻结 P-control 独立确认

### 3.1 角色与判定

Decision: `CONDITIONAL GO_AS_SUPPORTING_CONFIRMATION`

它的优点是已有 checkpoint、不需训练、预期信息回报快。它的缺点是：

- 创新性低；
- 旧结果来自三臂结果后的探索性识别；
- recipe 不是单一 loss 因子；
- 可能通过 unary/all-metal shortcut 抬高 comp-valid。

即使完整通过，P-control 只能成为 anchor 或强 baseline，不能作为论文标题
方法。

### 3.2 Stage A0：旧 512 只读审计

必须恢复：

- P-control-only valid / P0-only valid discordance 和 exact McNemar；
- all-metal、single-element；
- unique-formula、top-1 formula frequency、element coverage；
- mean atom count；
- 注册的 11 个 marginal TVD；
- invalid-reason taxonomy；
- train exact/reduced-composition overlap；
- formula 与模型自报 anion/charge 的一致性；
- 原始 attempt、ordinal、seed、raw/canonical SHA。

任一条件触发立即停止：

- shortcut 膨胀；
- 任一 TVD 相对 P0 恶化超过 `0.02`；
- unique formula 或 element coverage 明显坍缩；
- paired raw ledger 无法恢复；
- 结果依赖 survivor-prefix、repair、filter 或 replacement。

### 3.3 Stage A1：新 1,024 Plan-only confirmation

若 A0 通过，注册全新 1,024 ordinal ledger：

- P0/P-control 使用相同 prompt、schema、parser、temperature、top-p、
  top-k 和 per-ordinal RNG role；
- effective batch size 1；
- raw all-attempt denominator；
- 旧 512 不进入 primary inference；
- 无 retry、repair、filter、replacement 或 rerank。

Primary pass：

- raw comp-valid `>= +3.0 pp`；
- paired 95% CI lower bound `>0`；
- parse/completion drop `<=0.5 pp`；
- unique formula `>=95%` of P0；
- `|Δ mean N|<=0.5`；
- 任一 TVD worsening `<=0.02`；
- all-metal/single-element 不膨胀；
- formula novelty 与 formula↔chemistry coherence 不劣化。

### 3.4 时间、算力和预期

- 旧 ledger audit：约 0.5–1.5 个工程日，CPU only；
- 新 1,024 Plan-only：约 2–4 个日历日，包括 source freeze、两臂单 A800
  sampling 和 assembly；
- time-to-downstream-64：从授权起约 5–7 日；
- time-to-downstream-256：约 8–12 日；
- GPU：两条单 A800 Planner sampling arm；通过后下游 body/refine 另计；
- API：Plan-only 和 Direct 都不需要；S.U.N. 才可能需要 MP common snapshot。

效果先验而非承诺：

| 指标 | 预期 |
|---|---|
| comp-valid | 若旧信号复现，约 `+3~4 pp` |
| meta S.U.N. | 无可靠方向；只能下游实测 |
| strict S.U.N. | 无可靠方向 |
| conditional struct-valid | 近似不变，目标非劣 |
| raw joint-valid | 可能随 comp 提高，但须 body/refine 证明 |

最迟 2026-08-12 必须完成 anchor 决策。若届时 audit/confirmation 未完成，
直接固定 P0，避免低新颖确认阻塞主贡献。

## 4. 路线 B：Formula-prefix charge reachability

### 4.1 角色、创新性与判定

Decision: `CONDITIONAL GO_AS_MAIN_CONTRIBUTION`

暂定方法名：

> Chemistry-Reachable Planner（CR-Plan）

创新性评级：**中等、条件性**，不能评为高。

[CrysVCD](https://arxiv.org/abs/2507.19799) 已占据广义
“valence-constrained composition generation”叙事。因此 CR-Plan 的可发表
差异只能来自以下组合：

1. 约束发生在 rich seven-line Planner 的 formula prefix，而不是生成后
   检查一个完整 composition；
2. grammar state 与 finite-budget charge-reachability state 组成 product
   automaton；
3. 每个 terminal formula 保存真实 oxidation witness/certificate；
4. 单次 legal-support-renormalized sampling，无 beam、repair、retry、
   replacement、filter 或 rerank；
5. 失败 attempt 原样留在 denominator；
6. 约束 Planner 接入 exact-length Body-DLM 和 refine800，并同时报告
   comp、raw/conditional struct、strict 与 meta。

这六点必须由文献调研逐条核对。若 CrysVCD 或其他工作已有等价
prefix-level DP 与单次 proof-carrying decoding，CR-Plan 新颖性降为低，
不应作为 ICLR 主贡献。

### 4.2 Product automaton

对 formula line 的每个候选 tokenizer token，维护：

```text
state =
  formula grammar state
  × parsed element/count prefix
  × used element set
  × current atom count N
  × remaining atom budget (20 - N)
  × reachable total-charge set
```

Grammar 部分必须按 tokenizer token 的完整字符展开，而不能假设一个 token
只对应一个元素或一个整数。一个 token 中若同时包含 count、下一个元素或
newline，必须依次推进字符级 DFA。

对已完成 composition `C={(e_i,n_i)}`，使用冻结 oxidation table 定义：

```text
Q(C) = { sum_i n_i * q_i : q_i in Ox(e_i) }
```

对未完成 prefix，memoized DP 计算在剩余 atom budget、合法新元素和 count
下可加入的 charge contribution。候选 token 只有在以下条件成立时才合法：

- 推进后 grammar 仍可被现有 parser 接受；
- `1 <= N <= 20` 仍可满足；
- 存在某个合法 suffix，使 `0` 仍属于最终 reachable charge set。

从 formula line 离开的 newline/EOS 只有在：

```text
0 in Q(C)
```

且可恢复一个明确的 `(q_1,...,q_k)` witness 时才合法。

额外边界：

- 第一版只做 charge neutrality，不同时加入 Pauling；
- unary/all-metal 不可作为可达性成功证书；
- lattice、spacegroup、volume 完全不约束；
- 无 legal token 时记作该 raw attempt 的 constrained dead-end；
- 禁止 silent fallback 到 unconstrained generation；
- 每步只对原模型 logits 的 legal support 重归一化并采样一次。

### 4.3 必须保存的机制 telemetry

每 ordinal 保存：

- raw/canonical Plan 和 SHA；
- 每 formula step 的 legal token count；
- masked probability mass；
- blocked token / blocked newline count；
- DP cache hit/miss；
- grammar、atom-budget、charge 三类 mask activation；
- terminal oxidation witness；
- dead-end 的最早 token index 和状态；
- formula log-prob under unconstrained model；
- sampling/retry/filter/rerank flags，均须明确为 false。

这些 telemetry 既用于 debug，也用于论文回答一个关键问题：

> 增益来自删除少量明显非法尾部，还是把模型推入了低概率、低多样性的
> 化学区域？

### 4.4 预期效果与不能承诺的效果

| 指标 | 设计预期 | 解释 |
|---|---|---|
| charge failure | `-50%` 到 `-80%` | 直接机制目标 |
| raw comp-valid | `+2~5 pp` | 历史 headroom 支持，但取决于分布迁移 |
| raw joint-valid | 小幅正向 | 只有 body/refine 成功时才兑现 |
| conditional struct-valid | 约 0，允许最多 `-1 pp` | 已接近 ceiling |
| strict S.U.N. | 无直接因果保证 | 必须非劣 |
| meta S.U.N. | `0~+2 pp` 的弱先验，高不确定 | charge-neutral 不等于低 hull |

R03 safe-axis 中 24 个 charge failure 对固定 992 个 refined sample 的静态
上界约为 `24/992 = 2.42 pp`；但 constrained sampling 会生成不同 formula，
所以这不是严格 treatment-effect 上界。它只提醒我们不要承诺不现实的
两位数 comp 增益。

### 4.5 32 → 64 → 256 → 4×256 门禁

#### Gate 32：工程正确性

配对 32 raw attempts，要求：

- tokenizer transition、formula DFA 和 DP oracle 的已知 fixture 100%；
- 现有 parser 与 constraint oracle 对完整 formula 100% parity；
- 0 silent fallback；
- 0 repair/retry/replacement/filter/rerank；
- 所有 terminal constrained formula 都有可复算的 neutrality certificate；
- raw denominator、ordinal、seed、raw/canonical SHA 完整；
- exact seven-line continuation 可解析；
- constrained dead-end 必须有可解释状态；若 completion 比 control 低超过
  1/32，不扩到 64；
- 任何 certificate 错误、tokenizer/FSM 错误或 unexplained failure 均为
  engineering stop。

Gate 32 不作科学提升声明。

#### Gate 64：机制筛选

要求：

- raw comp-valid 至少 `+3/64`；
- charge failure 至少下降 50%；
- parse/completion 最多下降 1 个；
- single-element 最多增加 1 个；
- all-metal 最多增加 2 个；
- shortcut 对 comp gain 的贡献不超过 10%；
- unique formula `>=95%` of control；
- `|Δ mean N|<=0.5`；
- 无新 failure class；
- masked probability mass、dead-end 和 formula likelihood 不显示
  support collapse。

任一 primary 条件失败，停止 CR-Plan 主线；不调 seed、阈值或 mask 规则后
重跑同一 ledger。

#### Gate 256：科学筛选

使用同一 frozen anchor、H1 D1、B0、model_494 refine800：

- raw comp-valid 至少 `+8/256`，即 `>=+3 pp`；
- charge failure 至少下降 50%；
- raw structure validity 非劣 `1 pp`；
- conditional structure validity `>=99.5%` 或相对 control 非劣；
- raw joint-valid 为正；
- strict S.U.N. 不下降；
- meta S.U.N. 至少 `+2 pp`，或 paired CI 明确为正；
- formula/structure unique 与 novel 各不下降超过 `2 pp`；
- top-1 formula、arity、mean N、element marginal、anion marginal 无坍缩；
- 完整报告 exact McNemar、paired bootstrap、reason taxonomy 和 unknown
  hull coverage。

如果 comp/charge 通过但 meta 未通过，CR-Plan 可保留为机制结果或 appendix，
但不能成为“同时改善 meta 的最终系统”，也不能事后降低 meta gate。

#### Gate 4×256：独立确认

冻结同一个候选后，四个新独立 scientific ledgers，要求：

- comp effect 至少 3/4 为正，四次均值 `>=+3 pp`；
- pooled/hierarchical paired CI 对 comp 的下界 `>0`；
- charge failure reduction 至少 3/4 达到 50%；
- meta effect 至少 3/4 非负，且平均不低于 control；
- structure 与 strict 在每个 ledger 都通过非劣 gate；
- 每个 ledger 单独通过 shortcut、diversity、completion 和 failure-class
  gate；
- residual hull unknown 必须为零，或严格成对且证明对 treatment effect 的
  贡献为零；
- 不能把四个 CUDA process realization 当作这四个 ledger。

### 4.6 工期、GPU 与 API

从执行授权开始的现实估计：

- DP/oracle/tokenizer implementation + fixtures：3–5 工程日；
- time-to-32：4–6 日；
- time-to-64：6–9 日；
- time-to-256：10–15 日；
- time-to-4×256：20–30 日。

资源：

- 32/64 Plan-only：各一个单 A800 paired sampling job，CPU assembly；
- 256：Planner/body generation + frozen `model_494` exact800；refiner 是主要
  GPU 成本；
- 4×256：四个独立 paired jobs，建议最大并发 2，避免把排队时间假定为零；
- Direct comp/struct 不需要外部 API；
- strict/meta 需要 frozen S.U.N. runtime 与 common Materials Project
  thermo snapshot。新 formula 可能产生现有 227-chemsys snapshot 未覆盖的
  系统，不能假设旧 cache 足够。

API policy：

- 仅在 256 Direct/novelty 候选已通过后构造两臂 union common snapshot；
- credential 只允许 runtime-only mode-0600 one-use file；
- snapshot 冻结后所有 arms 共用；
- API/coverage failure 不能通过只删除 unknown candidate 来修复；
- 2026-08-20 前必须验证 API/cache 路径，否则 meta 主张有硬进度风险。

## 5. 路线 C：Formula-derived chemistry compiler

### 5.1 角色与判定

Decision: `CONDITIONAL GO_AS_OPTIONAL_SECONDARY_CONTRIBUTION`

创新性评级：**中低到中等**。单独把 formula 映射成 canonical metadata
更像必要的语义工程。只有它产生可验证的 downstream 因果改善时，才有资格
成为第二贡献。

### 5.2 Compiler 定义

对一次冻结生成的 formula，deterministically 导出：

- canonical `anion_framework`；
- canonical `charge_bucket`；
- lexicographically canonical oxidation witness；
- sidecar `oxidation_candidates` certificate。

visible seven-line Plan 仍只有：

```text
formula
anion
charge
lattice
spacegroup
volume
terminator
```

certificate 是 sidecar，不偷偷增加可见 schema。模型自报的 anion/charge
在 candidate arm 中被 canonical token sequence 取代；lattice、
spacegroup、volume 继续由相同 Planner 自回归生成。

它不允许：

- 修改 formula；
- 对 invalid formula 做 repair；
- 选择多个 candidate 中最好的一个；
- 根据 body、refiner 或 S.U.N. 结果修改 canonical rule。

### 5.3 C0：先证明有 headroom

在任何 GPU sampling 前，对训练/验证和冻结 generated ledgers 做只读审计：

- compiler 与冻结 SMACT/reason oracle 100% parity；
- 多 oxidation witness 时 canonical tie-break 完全确定；
- anion mapping ambiguity 被显式列出，而不是 silent guess；
- formula self-report 与 recomputed anion/charge 的 mismatch rate；
- mismatch 是否集中在 comp-invalid、meta-loss 或某些 chemistry family。

停止条件：

- oracle parity 不是 100%；
- canonical mapping 依赖不可冻结的启发式；
- 可支持 formula 中 self-report mismatch 小于 5%，没有足够 intervention
  headroom；
- mismatch 与 body prompt 实际消费字段无关。

### 5.4 独立 C2 实验

Control 和 candidate 使用同一 formula source：

```text
control:   formula + model-self-reported anion/charge -> sampled geometry
candidate: same formula + compiled anion/charge       -> sampled geometry
```

因此 formula、N、elements 和 counts 必须逐 ordinal 相同。Comp-valid 在
这个实验中理论上不应改变；若改变，说明 denominator 或 formula identity
有错误。

最小门禁：

- 32：formula SHA 32/32 相同、compiler parity/certificate 100%、0 fallback；
- 64：formula 64/64 相同、coherence 接近 100%、completion 最多下降 1、
  无新 failure class；
- 256：comp label 逐 ordinal 完全相同，raw/conditional structure 非劣，
  strict 非劣，meta `>=+2 pp` 或 paired CI 正，unique/novel 各不降
  `2 pp`；
- 4×256：仅当 compiler 要进入主文第二贡献时需要；meta 至少 3/4 非负，
  strict/structure 每次非劣。

R03 meta `-27` 发生在相同 formula 下的 coordinate/hull redistribution。
这说明 compiler **可能**通过改变 geometry conditioning 影响 meta，但绝不
证明它会修复 safe-axis meta loss。

### 5.5 时间与资源

从 C0 通过开始：

- C0 read-only audit：1–2 工程日，CPU only；
- time-to-64：4–7 日；
- time-to-256：9–14 日；
- 4×256：额外 10–16 日；
- GPU/API 依赖与 CR-Plan downstream 相同。

全局 cutline：

- C0 最迟 2026-08-09；
- C2-64 最迟 2026-08-18；
- C2-256 最迟 2026-08-27；
- 08-27 没有独立 meta 信号即降为 appendix/negative result，不阻塞主方法。

## 6. 路线 D：训练型 Planner 改进

### 6.1 角色与启动条件

Decision: `DEFERRED_CONDITIONAL_GO_AS_BACKUP`

训练型 Planner 不应与 CR-Plan 同时竞争主线。只有以下诊断成立才启动：

1. 32 工程 gate 正确；
2. 64 显示 constraint oracle 的确删除 charge-invalid path；
3. 但 masked mass 过大、dead-end 或 diversity collapse 表明模型原分布与
   reachable support 不对齐。

如果 CR-Plan 因“charge 没有 headroom”或“comp 通过但 meta 变差”而失败，
训练 formula likelihood 并不解决原因，训练备线也应停止。

### 6.2 三种训练想法的比较

| 训练想法 | 优点 | 风险 | 建议 |
|---|---|---|---|
| formula-first reweight + continuation | 实现最简单 | 只提高 teacher formula CE，不直接惩罚非法 support；又一次改 exposure | 作为 ablation |
| typed two-stage Planner | formula 与其余字段边界清楚 | 两个 adapter、两段推理、训练与部署变量多 | 本轮延期 |
| reachability-mass SFT | 与 CR-Plan oracle 直接一致；不使用 S.U.N. reward | dynamic legal mask、loss scale 和梯度干扰 | **首选训练备线** |

### 6.3 Reachability-mass SFT

在 teacher-forced formula prefix `s_t` 上，由相同 product automaton 返回
可达 token 集 `A(s_t)`，定义：

```text
L_reach(t) = -log sum_{v in A(s_t)} p_theta(v | s_t)
L_train    = L_field + lambda * mean_t L_reach(t)
```

它不是要求模型预测唯一的下一 token，而是要求概率质量留在仍有中性
completion 的支持内。

公平性约束：

- 从和 P-control 相同的初始 P0 checkpoint 出发；
- 使用相同 3,200 rows、400 updates、LR、validation panel；
- `lambda` 在任何 generated output 前由固定初始 panel 的 loss-scale
  rule 一次冻结，不做 generation sweep；
- checkpoint 只按 likelihood/support metrics 选，不看 comp、S.U.N.、
  energy 或 hull；
- 第一次科学比较使用 **unconstrained inference**，从而只测试 learned
  Planner factor；
- 只有 training 与 hard decoding 分别通过后才可研究组合，本轮最小投稿
  不做该组合。

### 6.4 门禁、工期与依赖

32 training smoke：

- teacher next token 100% 属于 legal set；
- 0 empty legal set、NaN、OOM；
- `L_reach` 和 legal probability mass 方向正确；
- fixed-panel target NLL 相对 anchor 不恶化超过 1%。

64 Plan-only：

- charge failure 至少下降 25%；
- comp-valid 至少 `+2/64`；
- parse/completion 不下降超过 1；
- shortcut/diversity/drift gate 全过。

256 与 4×256：

- 沿用 CR-Plan 的 comp/structure/strict/meta/shortcut/diversity gate；
- 不因为是训练模型而降低门槛；
- hard constraint 不得同时打开。

时间：

- time-to-64：触发后 9–14 日；
- time-to-256：15–22 日；
- time-to-4×256：25–36 日；
- 一次训练需要单 A800、400 updates；sampling/refine/API 另计。

最迟 2026-08-17 必须决定是否启动。晚于该日才启动，很难在 09-10
结果冻结前完成独立确认，应转为 future work。

## 7. 四路线横向比较

数值为 planning prior，不是承诺或 power calculation。

| 路线 | 论文角色 | Novelty | 到 64 | 到 256 | GPU | MP API | comp | meta/strict | struct |
|---|---|---|---:|---:|---|---|---|---|---|
| frozen P-control | 支撑确认 | 低 | 5–7 日（须先 1,024 Plan） | 8–12 日 | 单 A800 sampling；下游另计 | Plan/Direct 不需 | 旧信号约 +3~4 pp | 未知 | conditional 近似不变 |
| CR-Plan reachability | **主贡献候选** | 中等、受 CrysVCD 威胁 | 6–9 日 | 10–15 日 | 32/64 单 A800；256+refine | S.U.N. 条件需要 | +2~5 pp，charge -50%+ | 无直接保证；必须过 gate | 非劣，raw joint 可能升 |
| chemistry compiler | 可选第二贡献 | 中低~中 | 4–7 日（C0 后） | 9–14 日 | 下游同上 | S.U.N. 条件需要 | 同 formula 下应为 0 | 只有实测 +2 pp/CI 正才保留 | 非劣 |
| reachability-mass SFT | 备线 | 中等、待检索 | 9–14 日 | 15–22 日 | 1 A800 train + sampling/refine | 训练不需；final 条件需要 | +2~4 pp 弱先验 | 未知 | 非劣 |

按“创新性 × 成功概率 × 截止可达性”排序：

1. CR-Plan；
2. compiler，但仅在 C0 有 headroom 时；
3. P-control 确认，作为 anchor；
4. training backup。

### 7.1 统一 1–5 评分与评审标签

| 路线 | Reviewer label | Innovation | Expected effect | Evidence | Feasibility | Deadline fit | Paper coherence |
|---|---|---:|---:|---:|---:|---:|---:|
| CR-Plan | `KEEP_MAIN`，但受 novelty/64 hard gate 约束 | 3 | 4 | 4 | 4 | 4 | 5 |
| frozen P-control | `APPENDIX_ONLY` / supporting anchor confirmation | 1 | 4 | 4 | 5 | 5 | 3 |
| chemistry compiler | `KEEP_BACKUP`；无 headroom 即 `CUT` | 2 | 2 | 2 | 4 | 4 | 3 |
| reachability-mass SFT | `KEEP_BACKUP`；未触发 support mismatch 即 `CUT` | 3 | 3 | 2 | 3 | 2 | 4 |

CR-Plan 的 Innovation 只给 3，而不是 4–5，原因是 CrysVCD 已经覆盖宽泛的
valence-constrained generation。只有完整检索证实 prefix-level product
automaton 与 proof-carrying rich Plan 的差异后，评分才保持 3；若已有等价
方法，则降到 1–2 并退出主线。

### 7.2 最大可接受 A800 GPUh

以下是 go/no-go 资源上限，不是必须花满的配额；queue wait 不计入 GPUh，
CPU analysis 和 MP API 不计入：

| 路线 | 到 256 的最大 A800 GPUh | 4×256 追加上限 | 超限处置 |
|---|---:|---:|---|
| P-control supporting confirmation | Plan-only 24；若做 D1/B0/refine256，总计 72 | 不单独为 P-control 做 4×256 | 超限则固定 P0 |
| CR-Plan main | 72 | 192 | 先保 256 与至少一个独立 seed；不得牺牲写作 cutline |
| compiler | 72 | 仅在 256 meta pass 后追加，最多 192 | 无独立 meta 信号立即停止 |
| reachability-mass SFT | 96 | 最多 192 | 只允许触发式备线；不得与 CR-Plan 同时烧满预算 |

项目层面不应同时给 CR-Plan、compiler 和 training backup 各自完整
`4×256`。优先级是 CR-Plan confirmation；compiler/training 只有成为实际
主候选时才获得 repeat 预算。

### 7.3 最可能成功与最可能失败的机制

最可能成功：

> 大部分 charge-invalid trajectory 在 formula newline 前仍有明显、局部的
> 非中性终止状态；只阻止这些终止而不大幅删除内部高概率 token，即可用很小
> 的 distribution shift 换取 charge-failure reduction。

最可能失败：

> 中性 suffix 在 P-control/P0 下承载的原始概率质量过低，prefix mask 把模型
> 推入罕见元素/count 尾部，造成 dead-end、formula diversity collapse，或
> 得到 charge-valid 但高-hull 的新 composition；于是 comp 上升而 meta 不升
> 甚至下降。

### 7.4 因果污染判断

按本提案执行不会污染已有 H1 归因，因为 primary comparison 固定 D1、B0、
model_494、seed role、evaluator 和 denominator，只改变 formula legal
support。以下操作会污染归因，因而禁止：

- 在第一次 CR-Plan test 同时换 P-control、safe-axis 或 compiler；
- 根据 S.U.N. 结果修改 charge table、mask 或 canonical tie-break；
- 把 safe-axis 的 strict gain 与 Planner 的 comp gain 合并成一个未拆分的
  treatment；
- 把 training backup 与 hard constraint 同时打开。

## 8. 倒排执行计划与最迟 cutline

### 8.1 主线日历

| 日期 | 必须完成 | 决策 |
|---|---|---|
| 08-04 ～ 08-08 | CrysVCD/相邻工作 claim-by-claim 对照；P-control A0；compiler C0 | novelty 不足则 CR-Plan 降级；P-control fail 则固定 P0 |
| 08-05 ～ 08-11 | product automaton、tokenizer transition、oracle fixture、telemetry；代码和 gate freeze | 不能看生成结果后改规则 |
| 08-08 ～ 08-12 | 新 1,024 P-control confirmation | 08-12 冻结 anchor；未完成也固定 P0 |
| 08-12 ～ 08-14 | CR-Plan 32 与 64 | 08-14 未过 64，停止主线 |
| 08-15 ～ 08-22 | CR-Plan 256 + Direct + common-snapshot S.U.N. | 08-24 为不可逾越的 late cut |
| 08-23 ～ 09-05 | 四个独立 256 ledgers | 09-05 冻结 confirmatory evidence |
| 08-16 ～ 08-27 | 可选 compiler 64/256；不与 CR-Plan 合并 | 08-27 无 meta 信号则 appendix |
| 09-06 ～ 09-10 | 统计、失败归因、图表、claim freeze | 之后不因写作需要重选模型/seed |
| 09-11 ～ 09-18 AOE | 摘要与主文叙事 | 09-18 23:59 AOE 摘要截止 |
| 09-11 ～ 09-25 AOE | 完整论文、附录、reproducibility checklist | 09-25 23:59 AOE 全文截止 |

### 8.2 备线日历

训练备线只有一个窗口：

- 08-15 前识别为 support mismatch；
- 08-17 前冻结训练协议并启动；
- 08-24 前完成 64；
- 09-01 前完成 256；
- 09-10 前完成所需独立 repeats。

任何节点延迟时，备线不得挤占已通过 CR-Plan 的确认和写作资源。

### 8.3 若 08-15 仍无 Planner 信号

若 2026-08-15 前没有 paired-64 机制信号，停止把新 Planner 写成 ICLR
主贡献。论文退回已冻结的 H1/R03 证据：

1. exact-length Body-DLM 在冻结 H1 contract 下具有高 completion；
2. decoding schedule 与 constraint precondition 存在可复现的因果接口；
3. safe-axis 消除 mixed-axis duplicate collapse；
4. Planner chemistry 是 composition bottleneck；
5. safe-axis 对 strict/meta 产生 polarization，而非 broad stability gain。

这个 fallback 是 systems/diagnostic narrative，不得把 stopped safe-axis
改写成全指标改进。CR-Plan 只作为未完成方法或 future work，不展示选择性
小样本结果。

## 9. 失败树与停止纪律

```text
P-control A0/A1 fail
  -> 固定 P0
  -> CR-Plan 仍继续

Novelty gate fail
  -> 不把 valence constraint 写成主创新
  -> compiler/training 也不能靠改名救回
  -> 转为系统/分析型论文叙事或停止

CR-Plan Gate 32 fail
  -> engineering stop
  -> 新版本只能修明确 bug，并重新从 32 开始

CR-Plan Gate 64: charge 不降 / comp 不升
  -> scientific stop
  -> 不训练、不调 mask、不换 seed

CR-Plan Gate 64: charge 降，但 masked mass/dead-end/diversity collapse
  -> hard decoding stop
  -> 允许触发 reachability-mass SFT 备线

CR-Plan Gate 256: comp 通过，meta/strict/struct fail
  -> 不晋级最终系统
  -> 保留机制结果，不能与 safe-axis/compiler 叠加“抵消”

Compiler C0 无 headroom
  -> compiler stop
  -> 不做 GPU 实验

Compiler 256 无独立 meta 信号
  -> appendix/negative result
  -> 不进入 combined candidate

4×256 sign 不稳定
  -> 不做强主张
  -> 不删除不利 repeat
```

全路线共同禁止：

- post-hoc formula repair；
- invalid formula filtering；
- candidate pool、beam 后按 chemistry/S.U.N. rerank；
- retry/replacement；
- survivor-prefix denominator；
- S.U.N./energy/hull 用于训练或 checkpoint selection；
- P-control + reachability + compiler + safe-axis 一次性组合；
- 把 process repeats 写成 independent scientific repeats。

## 10. 论文叙事

### 10.1 推荐题目

> Chemistry-Reachable Planning for Exact-Length Diffusion Crystal Generation

或者：

> Reachable Before Realizable: Proof-Carrying Formula Planning for Crystal
> Diffusion Language Models

在 novelty audit 完成前，不使用 “first” 或 “novel valence constraint”。

### 10.2 主文逻辑

1. **诊断：** hierarchical crystal generator 的 body/refiner 已达到高
   completion 和约 99.7–99.8% conditional structure validity，但 Planner
   formula chemistry 是 raw joint validity 的主瓶颈。
2. **方法：** 在 rich Plan 的 formula prefix 上，把语法和 finite-budget
   charge reachability 合成 product automaton，并输出可复算 witness。
3. **实验纪律：** 只有一次 legal-support sampling，所有失败留在 raw
   denominator；无 repair/filter/retry/rerank。
4. **结果：** 先证明 charge failure 与 comp-valid 的因果改善，再证明
   body/refiner、strict、meta、diversity 非劣或改善。
5. **可选机制：** 若 compiler 独立通过，说明“formula validity”和
   “rich-Plan semantic consistency”是两个不同的 Planner 问题。

### 10.3 与 CrysVCD 的安全 claim 边界

暂定比较框架，必须由完整论文阅读确认后才能进入主文：

| 维度 | CrysVCD 已知宽泛定位 | 本提案可争取的差异 |
|---|---|---|
| chemistry constraint | valence-balanced constrained composition generation | token-prefix grammar × charge-reachability product automaton |
| generation interface | composition generator → diffusion structure generator | rich seven-line Plan → exact-length Body-DLM → refine800 |
| certificate | 待全文核对 | 每个 terminal formula 的 oxidation witness |
| invalid handling | 待全文核对 | no repair/retry/filter/rerank，raw failure retained |
| semantic fields | 待全文核对 | formula-derived anion/charge sidecar compiler |
| evaluation | 待全文核对 | paired all-attempt comp/joint + strict/meta + independent ledgers |

如果差异只剩“我们用另一个模型实现同样 valence mask”，新颖性不足。真正
可写的贡献必须是 prefix reachability、proof-carrying rich planning 和严格
causal evaluation 三者的组合。

对其他近邻的边界：

- **PLaID++：** preference alignment 与 Wyckoff-text generation 已占据
  “迭代对齐提高晶体质量”的叙事；CR-Plan 不使用 preference reward，差异
  必须落在 symbolic prefix reachability，而不是笼统的 alignment。
- **CRYSTAL：** coordinated multi-objective RL 已占据 crystal LLM RL
  叙事；本主线是 frozen-policy constrained decoding，不使用 energy/S.U.N.
  reward。Reachability-mass SFT 也只能声称 symbolic support alignment，
  不能声称新的通用 RL。
- **Mask-Aware Policy Gradients for DLMs：** token 与 reveal-position 的
  two-stage action 是 Body-DLM RL 必须面对的问题；CR-Plan 位于
  autoregressive Planner formula prefix，不改变 Body reveal action。两者
  不应混成同一算法贡献。
- **CrysTune：** auxiliary-task/RL crystal generation 已使“加 chemistry
  auxiliary task”本身不够新；训练备线只有 reachability-set probability
  mass 这一特定结构仍可能有差异。

### 10.4 可以与不可以声称的结论

若门禁通过，可以声称：

- Planner chemistry 是当前 H1 raw validity 的主要瓶颈；
- prefix reachability 显著减少 charge-invalid formula；
- 在一次采样、无后处理条件下提高 raw comp/joint validity；
- exact-length Body-DLM 与 frozen refiner 保持结构非劣；
- meta/strict 的结果按预注册门禁和 common snapshot 完整报告。

不能声称：

- 首次把 valence constraint 用于 crystal generation；
- charge neutrality 必然改善 hull stability；
- compiler 已解释 R03 safe-axis 的 meta loss；
- P-control 证明 field-balanced loss 单独有效；
- safe-axis 是 broad stability improvement；
- conditional struct-valid 的微小差值是新的主要突破。

一句可放进 9 页主文的贡献：

> We introduce a proof-carrying, formula-prefix reachability decoder that
> constrains a rich crystal Planner to charge-neutral completions in a single
> all-attempt sampling pass, and isolate its effect through the frozen
> exact-length DLM and refiner pipeline.

一句绝不能写的过度主张：

> We are the first to guarantee chemically valid crystals and improve
> composition, structure, strict S.U.N., and meta S.U.N. simultaneously.

## 11. 最终建议

### 可以尝试

CR-Plan **可以尝试**，而且是四条路线中唯一有机会形成 ICLR 主贡献的方向，
但必须同时满足：

1. 08-08 前证明相对 CrysVCD 的实质方法差异；
2. 08-14 前通过 paired-64 机制 gate；
3. 08-22 前通过 paired-256 的 comp、meta、strict、structure 和 diversity
   gate；
4. 09-05 前完成四个真正独立的 256 ledgers。

### 支撑性工作

P-control 值得立即做只读审计和新 1,024 confirmation，但最迟 08-12 必须
结束。它通过则作为 stronger anchor；失败或延迟则固定 P0。无论结果如何，
都不把它包装成论文核心创新。

### 有条件追加

Compiler 先做 C0 headroom audit。只有 self-reported chemistry mismatch
足够多且 C2 独立改善 meta 时，才作为第二贡献。否则留作一致性工程或
appendix。

### 备线

Reachability-mass SFT 只处理“模型概率质量与 reachable support 不匹配”
这一种失败。它不用于挽救没有 chemistry headroom 或 meta adverse 的结果，
也不与 hard decoding 同时首测。

### 总判断

从实现和算力上，这条路线在 45/52 天窗口内可完成；最大的风险不是 GPU，
而是：

- CrysVCD 导致的 novelty 压缩；
- charge-valid 对 meta S.U.N. 缺少直接因果联系；
- 新 chemical systems 的 MP hull coverage；
- 为追求最终全指标正向而过早组合多个因素。

因此最稳妥的论文策略是把 **CR-Plan 的 comp/joint 因果改善**作为主轴，
把 meta/strict/struct 设为严格 downstream gate，而不是在方法名称中提前
承诺它们都会提高。
