# 第二轮交叉审稿：Body-token 作者对 Planner / Body / RL 的强制收敛

状态：`decision_only_no_execution_authorization`

日期：2026-08-04

评审身份：Body-token/support 提案作者兼苛刻交叉 reviewer。

完整审阅材料：

- `ROOT_EVIDENCE_AND_DEADLINE_FRAME_V1.md`；
- `PLANNER_CHEMISTRY_PROPOSAL_V1.md`；
- `BODY_TOKEN_PROPOSAL_V1.md`；
- `MASK_AWARE_RL_PROPOSAL_V1.md`；
- CrysVCD、PLaID++、CRYSTAL、Mask-Aware Policy Gradients 的一手页面/论文。

本文只作路线裁决，不授权训练、生成、refinement、S.U.N.、API 查询、
checkpoint promotion 或自动下游。

## 0. 一页裁决

### 0.1 强制只留一主一备

| 方向 | 第二轮决定 | 真正保留的候选 | 其余内部候选 |
|---|---|---|---|
| Planner chemistry | **`KEEP_MAIN`** | `P0 vs P0 + CR-Plan` formula-prefix charge reachability | P-control 仅 supporting/appendix；compiler 与 training backup 当前 `CUT` |
| Body-token/support | **`KEEP_BACKUP`** | PILS-L，仅在 Planner novelty 早停时接棒 | support contraction、all-axis、smoothing、dead-token cleanup 当前 `CUT` |
| Mask-aware RL | **`CUT`** | 无 | 设计可留 future work；本轮不给训练/评测预算 |

这不是把三个方向按顺序全做。唯一允许的主实验 principal factor 是
formula-prefix reachability；唯一备线 principal factor 是 length-axis token
sharing。RL 在本投稿周期退出。

### 0.2 为什么不是 Body 或 RL 做主线

冻结证据已经定位了性能责任：

```text
Planner composition: 约 85%–88%，明确是最大瓶颈
Body completion:      984/1024，只有小幅 headroom
conditional structure:约 99.7%–99.8%，已接近饱和
strict/meta:          会极化，不能靠单一 proxy 推断
```

PILS-L 不能改变 P0 已冻结的 formula；它最多通过减少 body failure 提高 raw
all-attempt comp count，不能改善 conditional chemistry。RL 同样不能在 fixed
Plan 内获得 composition advantage。只有 Planner reachability 直接作用于当前
最大的 24/35 charge-invalid 桶。

### 0.3 总评分

分数越高越好；“实现风险”列按 `5=低风险/高可行性` 计。

| 方向 | Innovation | Expected effect | Evidence | 实现风险 | ICLR 时限 | 叙事一致性 | 总分 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CR-Plan | 3 | 4 | 4 | 4 | 4 | 5 | **24/30** |
| PILS-L | 2 | 2 | 3 | 3 | 3 | 3 | **16/30** |
| Mask-aware RL | 3 | 3 | 2 | 1 | 1 | 3 | **13/30** |

硬门高于总分。CR-Plan 若与 CrysVCD 方法实质重合，Innovation 立即降到 1–2
并退出主线；RL 即使 Innovation 仍为 3，也因依赖与截止期硬门被 CUT。

### 0.4 绝对时间与资源

| 方向 | 最早 paired-64 | 最迟 paired-64 | 最早 paired-256 | 最迟 paired-256 | reviewer 最大 A800 GPUh |
|---|---|---|---|---|---:|
| CR-Plan | 2026-08-13 | **2026-08-15** | 2026-08-21 | **2026-08-22** | 72 到 256；确认追加 96；总计 168 |
| PILS-L | 2026-08-14 | **2026-08-15** | 2026-08-18 | **2026-08-22** | 当前只解锁 Gate-1 12；若 08-08 接棒则 128 到 256、总计 176 |
| Mask-aware RL | 理论 2026-08-14 | — | 理论 2026-08-21 | — | **0 新 GPUh；当前 CUT** |

若 2026-08-15 没有主候选 paired-64 机制信号，不再切换新 principal factor；
论文退回冻结 H1 systems/diagnostic 主张。

## 1. Planner chemistry：`KEEP_MAIN`

### 1.1 只保留哪个 Planner 方法

主比较必须简化为：

```text
control:   frozen P0
candidate: same P0 + formula-prefix charge-reachability product automaton
```

不建议让新 1,024 P-control confirmation 阻塞或决定首个 CR-Plan anchor。
P-control 同时改变 continuation、数据流、更新数、LR 和 field-balanced loss，
又来自 post-selected discovery；把它先选成 anchor 会多引入一个 checkpoint
principal factor，并压缩 CR-Plan 的 32/64 时间。

因此：

- P0 现在就冻结为 primary causal anchor；
- P-control confirmation 可并行作为 supporting result，最迟 2026-08-12
  结束，但不改变 primary CR-Plan control；
- formula compiler 当前 CUT，不与 reachability 合并；
- reachability-mass SFT 当前 CUT；只有投稿后才处理模型/support mismatch。

### 1.2 成功与失败机制

最可能成功：

> 很多 charge-invalid trajectory 直到 formula newline 前仍可由局部
> reachability 判定阻断；只屏蔽不可到达 neutral witness 的 token/newline，
> 在原 logits 的 legal support 上单次重归一化，可用很小 distribution shift
> 减少 charge failure，并提高 raw comp/joint validity。

最可能失败：

> P0 在 neutral suffix 上的原始概率质量太低，product automaton 把采样推入
> rare element/count 尾部或 constrained dead-end；得到的 formula 虽
> charge-neutral，却更单一、更新颖但高 hull，最终 comp 上升而 meta 不升，
> 甚至出现新 chemical systems 的 MP coverage 缺口。

### 1.3 对真实瓶颈的预期

| 指标 | reviewer 预期 | 解释 |
|---|---|---|
| charge failure | 中高概率下降 50%+ | 方法直接目标 |
| raw comp-valid | 约 `+2~5 pp` 的合理范围 | 最大可解释 headroom |
| raw joint-valid | 小幅正向 | 取决于 Body/refiner completion |
| conditional struct-valid | 近似不变 | 已接近 ceiling，只要求非劣 |
| strict S.U.N. | 无先验正保证 | 必须非劣 |
| meta S.U.N. | 低置信 `0~+2 pp` | charge neutrality 不等于 low hull |

这个方向的主要论文价值是 comp/joint 因果改善。若 meta 只非劣而未提升，仍可
写；若把方法标题预先承诺为 “improves S.U.N.”，则证据边界错误。

### 1.4 新颖性碰撞

#### CrysVCD：高碰撞，必须是 kill gate

CrysVCD 已明确采用 transformer 生成 valence-balanced composition，再由
diffusion 生成结构。因而以下都不能作为本项目创新：

- composition-first / structure-second；
- valence-balanced generation；
- 将化学规则接入生成而不是只做 post-screening。

CR-Plan 只剩以下可能的真实差异：

1. rich Plan formula line 上逐 tokenizer-token 的 grammar × finite-atom-budget
   charge-reachability product automaton；
2. terminal oxidation witness/certificate；
3. single-pass legal-support renormalization；
4. 无 repair/retry/filter/rerank，失败保留在 all-attempt denominator；
5. 冻结 exact-length Body-DLM/refiner 下的 paired causal evaluation。

2026-08-08 前必须逐项阅读 CrysVCD 方法与 supplement。若其已有等价 prefix
DP/proof-carrying decoder，CR-Plan `CUT`，不能靠换名保留。

#### PLaID++

PLaID++ 已把 compact symmetry-informed Wyckoff text representation 和
preference alignment 用于稳定、novel、unique crystal generation。因此
CR-Plan 不能泛称“structured text improves physical generation”，也不能把
后续 reward alignment 写成新颖性。它必须坚持 symbolic reachability，而不是
偏好学习。

#### CRYSTAL

CRYSTAL 已覆盖 crystal LLM 的 coordinated multi-objective RL，以及
stability/validity/novelty/diversity 和 reward-hacking 叙事。CR-Plan 的安全
边界是 frozen-policy constrained decoding；不使用 multi-objective reward，
不声称 crystal RL。

#### Mask-Aware Policy Gradients

它位于 Body MDLM 的 token/masking two-stage action；CR-Plan 位于
autoregressive Planner formula prefix。两者作用层不同，当前不碰撞。
若把 Body RL 合并进主方法，碰撞与叙事冲突立刻出现。

### 1.5 因果归因与门禁

只要保持 P0、B0/R5-C、D1、model_494、evaluator、ordinals 和 denominator
冻结，`P0 vs P0+CR-Plan` 不污染既有 H1 归因。

会污染归因的做法：

- 第一次比较同时换 P-control；
- 同时开 compiler、safe-axis、PILS-L 或 RL；
- 根据 S.U.N. 改 oxidation table、mask 或 witness tie-break；
- 用 survivor-only denominator；
- 对 unknown/invalid formula 做 repair 或 replacement。

硬日期：

| 日期 | 必须到达 | STOP |
|---|---|---|
| **2026-08-07** | claim、oracle、token-DFA 接口冻结 | 不再加 Planner 架构 |
| **2026-08-08** | CrysVCD novelty gate | 实质重合即 CUT |
| **2026-08-11** | fixtures + paired-32 terminal | certificate/tokenizer 错即停 |
| **2026-08-13/14** | paired-64 terminal | comp `<+3/64`、charge reduction `<50%` 或 shortcut/collapse 即停 |
| **2026-08-20** | union MP snapshot/cache 可运行 | 不可运行则不能作 meta 主张 |
| **2026-08-22** | paired-256 + Direct/S.U.N. terminal | 不调门、不延至 08-24 |
| **2026-08-31** | independent panels/common evaluation | 未完成则降级 claim |
| **2026-09-05** | science freeze | 禁止新 principal factor |

reviewer cap：

```text
through paired-256 <= 72 A800 GPUh
independent confirmation additional <= 96 A800 GPUh
all-in <= 168 A800 GPUh
```

## 2. Body-token/support：`KEEP_BACKUP`

### 2.1 只保留 PILS-L

唯一保留的备线：

```text
<LA_k>, <LB_k>, <LC_k> -> <L_k>
```

axis 语义由 exact answer position决定。`7+4N`、D1、数值分辨率、
per-position 500-way nonzero length support、Planner、refiner 和 evaluator
全部不变。

它把：

- special token `2481 -> 1479`，减少 1,002；
- total model vocab `128830 -> 127828`，减少 0.778%；
- numeric stochastic identity union `2343 -> 1341`，减少 42.77%。

但它不缩小单个 length position 的合法 branching，也不改变 formula。

以下当前全部 CUT：

- B0-compatible support contraction：工程 safety，不足以做主创新；
- all-axis sharing：同时扰动 angle/coordinate，归因差；
- ordinal/neighbor smoothing：kernel/width/boundary 选择过多；
- 删除 `S/EMPTY/PAD`：纯清理；
- 与 RL 联合：违反 Gate −1 与单因素要求。

### 2.2 成功与失败机制

最可能成功：

> 三轴共享同一 length row，聚合 sparse ordinal supervision，改善
> length NLL/calibration，减少极端 lattice 与 refiner displacement，从而
> 小幅提高 completion 和 meta，strict 保持非劣。

最可能失败：

> full train 其实已覆盖关键 bins，或 B0 对 rare/unseen bins 的 legal mass
> 近零；此时 token reduction 只有工程效果。三轴 pooling 还可能抹去
> anisotropic axis prior，造成 lattice distribution 或 meta 恶化。

### 2.3 为什么只能是备线

1. 当前 `953/1013` unseen-numeric identity 来自 held-out，不是 full-train
   unseen 结论；
2. body completion 只有约 4% headroom；
3. conditional structure 已近 99.8%；
4. conditional comp-valid 完全由 frozen Planner 决定；
5. 新 tokenizer、row remap、matched C0/L1 one-epoch continuation 都要做，
   并非零成本 support fix；
6. PLaID++ 已强调 crystallographic representation 的重要性，PILS-L 只能
   主张 exact-schema position-indexed factorization，不能泛称“首次压缩
   Wyckoff/crystal representation”。

### 2.4 新颖性边界

- 相对 CrysVCD：无直接算法碰撞，但 PILS-L 不解决其核心 chemistry 约束问题；
- 相对 PLaID++：存在 representation-level 邻近。真正差异是 exact-length
  masked Body 中、position-disambiguated ordinal length sharing，不是新的
  Wyckoff text 表示；
- 相对 CRYSTAL：无 RL，不使用 multi-objective reward；
- 相对 Mask-Aware PG：无 policy gradient 或 reveal-policy 学习；仅改
  supervised output representation。

因此只有出现完整的：

```text
coverage/legal mass
  -> shared-L NLL/calibration
  -> lattice tail/refiner burden
  -> meta improvement with strict noninferiority
```

机制链，PILS-L 才可能升级为主文方法。只证明 vocab 变小不足以投稿。

### 2.5 备线的真实启动窗口

Body Gate-1 可以立刻只读运行，但 matched SFT 不应与 CR-Plan 主实验并行烧满
预算。

```text
2026-08-06: Gate-1 full-corpus coverage/legal-mass
2026-08-08: dynamic shared tokenizer/data/tests ready
2026-08-08: only if Planner novelty gate fails, PILS-L is promoted
2026-08-11: matched C0/L1 SFT terminal
2026-08-14: paired-64 mechanism terminal
2026-08-18: paired-256 target
2026-08-22: absolute paired-256 cut
2026-08-31: confirmation cut
```

关键限制：

- 若 CR-Plan 通过 2026-08-08 novelty gate，PILS-L 留作 backup/appendix，
  不再启动 full matched SFT；
- 若 CR-Plan 到 2026-08-14 才在科学 64 失败，已经没有时间把 PILS-L 从头
  变成主线；此时论文必须回退 frozen H1，而不是仓促切候选。

资源：

```text
immediate backup readiness: Gate-1 <= 12 A800 GPUh
if promoted by 2026-08-08:
  through paired-256 <= 128 A800 GPUh
  confirmation additional <= 48 A800 GPUh
  all-in <= 176 A800 GPUh
```

## 3. Mask-aware RL：`CUT`

### 3.1 不是方向错误，而是本轮不可承担

RL 提案最有价值的部分是：

- legal-support-aware token probability；
- token + reveal-position joint trace；
- D1 group-local K=1 Plackett–Luce；
- pre/post-refiner randomized control-variate reward；
- evaluator firewall 与 exact resume。

但这也正说明它不是“小 LoRA 就能试”的单变量补丁。旧 TraceRL 正式
NO-GO；必须重写 rollout、joint likelihood、trace、resume、reward、A/B
labels、independent evaluator 与 promotion pipeline。

它还有三重未冻结依赖：

1. Gate −1 token support；
2. frozen Planner/formula source；
3. `E_train/E_gate/E_final` 三层 evaluator 与 MP/DFT 路径。

在 2026-08-15 first-64、2026-08-22 paired-256、2026-08-31 independent
confirmation 的时钟下，任何一个依赖延误都会吞掉主线。

### 3.2 新颖性碰撞比提案自评分更严重

#### Mask-Aware Policy Gradients

该工作已经把 MDLM generation 定义为 token 与 mask/reveal position 的
two-stage action，并将 policy gradient 分成 token 与 masking 两项。
因此本项目不能把以下单独写成创新：

- joint token-position action；
- position likelihood；
- probabilistic remasking / Plackett–Luce；
- “token-only gradient is incomplete”。

剩余差异只有 crystallographic legal support、D1 group constraints、
pre/post-refiner randomized correction 和 all-attempt evaluator firewall。
这是有价值的 domain system combination，但不足以抵消当前实现风险。

#### CRYSTAL

CRYSTAL 已经主张 coordinated multi-objective crystal RL，并显式同时处理
physical correctness/stability、novelty 与 diversity，目标之一就是缓解
reward hacking。本文 RL 的 multiplicative stability×novelty×uniqueness
叙事与其高度邻近；不能再把 multi-objective crystal reward 写成主要创新。

#### PLaID++

PLaID++ 已采用 preference/post-training 来引导 stable、novel、unique
crystal generation，并强调 reward formulation、temperature 与 mode
collapse。本文不能泛称“首次对 crystal LM 做 post-training 以改善 S.U.N.”。

#### CrysVCD

RL 不直接碰撞其 valence-constrained Planner，但 frozen Plan 也意味着 RL
不能改善 composition，无法单独覆盖本项目最大瓶颈。

### 3.3 预期效果和失败机制

最可能成功：

> 在 legal support 与 D1 macro-groups 冻结时，joint policy 学到更好的
> reveal order/token coupling，把合法 proposal 推入更好的 refiner basin；
> meta-aware reward 避免 safe-axis 的 strict-positive/meta-negative 极化。

最可能失败：

> G=8 下 reward variance 太稀，joint action ratio 方差过高；cheap A 对
> exact800 B 排序不足，策略放大 reward proxy、rare tokens 或少数结构簇。
> 即使训练 reward 上升，independent gate 或 final CHGNet/MP 也可能下降。

### 3.4 为什么 reviewer 现在给 CUT

| 项目 | 审稿判断 |
|---|---|
| direct bottleneck | 只作用 strict/meta/basin，不作用 comp |
| local evidence | reveal schedule 有因果效应，但没有 RL efficacy |
| implementation | 旧 trainer 不可用，需新完整系统 |
| evaluator | independent judge 与 DFT 尚未冻结 |
| token dependency | Body representation 一改，RL support/trace 全失效 |
| novelty | core two-stage action 已被 Mask-Aware PG 占据 |
| deadline | first-64 前要同时过 Gate −1、R0、32、A/B/evaluator |

所以本轮：

```text
new RL A800 budget = 0
no first-64
no training
no evaluator construction for RL
archive proposal as post-submission design
```

这比“KEEP_BACKUP 但暗中占用工程资源”更诚实。若 ICLR 主方法在 8 月中失败，
也不得临时重启 RL。

## 4. 三方案之间的 confounding 与 dependency

### 4.1 不能合并的 principal factors

| 组合 | 污染 |
|---|---|
| CR-Plan + PILS-L | formula distribution 与 Body representation 同时改变，comp/meta 变化无法分层归因 |
| CR-Plan + RL | Planner composition/support 与 Body policy 同时改变；RL group reward 分布随 Plan 改变 |
| PILS-L + RL | tokenizer/support/action identity 变化，behavior likelihood 与 Gate −1 全部重做 |
| P-control + CR-Plan 首测 | checkpoint/recipe 与 legal support 同时改变 |
| safe-axis + 任一候选 | 已知 strict/meta 极化因子污染新机制 |
| compiler + CR-Plan | formula validity与 rich-field coherence 同时改变 |

### 4.2 依赖方向

```text
frozen P0
  -> CR-Plan single-factor main experiment
  -> frozen B0/R5-C + D1
  -> frozen model_494 + common Direct/S.U.N.

Gate-1 read-only
  -> only if CR-Plan novelty fails by 2026-08-08
  -> PILS-L matched C0/L1 backup

Mask-aware RL
  -> requires Planner, tokenizer/support, Body sampler and evaluators all frozen
  -> therefore cannot be an August backup
  -> CUT
```

### 4.3 共同 evaluator 纪律

三方向任何结果都必须使用：

- raw all-attempt denominator；
- proposal-before-refine 与 after-refine 分表；
- frozen Direct evaluator；
- frozen original A100/CHGNet S.U.N. 口径；
- treatment/control union common MP snapshot；
- 同样 novelty database 与 structure matcher；
- 无 retry/replacement/repair/filter/rerank。

历史 `9.4% strict / 47.4% meta` 与 coverage-adjusted
`9.71% / 48.94%` 不能混成同一 control。共同重算失败即退出主结果。

## 5. 绝对止损日历

| 日期 | 唯一允许的交付 | 止损 |
|---|---|---|
| **2026-08-06** | Body Gate-1；Planner oracle/claim draft | Body legal mass无信号则 backup CUT |
| **2026-08-07** | 三方向 claim 与架构冻结 | 不再引入新 principal factor |
| **2026-08-08** | CR-Plan vs CrysVCD novelty terminal；PILS-L readiness | CR-Plan novelty fail 才切 PILS-L |
| **2026-08-11** | 主候选 32 工程 terminal | 工程 gate fail 即退 frozen H1 |
| **2026-08-15** | 唯一主候选 paired-64 terminal | 无机制信号即不再换候选 |
| **2026-08-20** | common MP snapshot/cache freeze | 不完整则不作 meta 主张 |
| **2026-08-22** | paired-256 terminal | 不改 threshold/seed/method |
| **2026-08-31** | independent confirmation、共同评测、核心消融 | 未完成则降低 claim |
| **2026-09-05** | science freeze | 之后不启动 principal factor |
| **2026-09-12** | paper/supp/repro only-fix | 只修错，不扩结果 |
| **2026-09-18 AOE** | ICLR abstract | 真实披露生成式 AI 使用 |
| **2026-09-25 AOE** | ICLR full paper | 主文最多 9 页 |

## 6. 三大 paper-killing risks

### Risk 1：新颖性被前作吃掉

如果 CR-Plan 只剩“valence-balanced composition before diffusion”，CrysVCD
已覆盖；如果 RL 只剩“token+position policy 和 multi-objective crystal
reward”，Mask-Aware PG 与 CRYSTAL 已覆盖；如果 PILS-L 只剩“compact
crystallographic representation”，PLaID++ 会让贡献显得像实现细节。

杀伤方式：审稿人认为这是三个已知想法的材料领域拼接，而不是新方法。

### Risk 2：提高 comp 却没有可用 S.U.N. 故事

charge-neutral 不保证 low hull；R03 已实证 strict 与 meta 可反向。
CR-Plan 可能提高 comp 但引入高-hull/new-chemsys，PILS-L/RL 又不能改善
conditional comp。若最终没有 common-snapshot meta 非劣，论文的一句话主张
会裂成互不相干的 comp trick 与 stability diagnostic。

杀伤方式：主要 endpoint 与方法机制不一致，或 MP unknown/coverage 破坏公平
比较。

### Risk 3：截止前多因素和评测债务压垮因果证据

P-control、CR-Plan、compiler、PILS-L、RL、safe-axis 任意组合都会污染归因。
如果 8 月仍同时维护多个 tokenizer、checkpoint、reward judge、hull snapshot
和 seed panel，就无法在 8 月 31 日前完成独立确认，更无法在 9 月 12 日前
得到可信复现包和 9 页叙事。

杀伤方式：有许多漂亮点估计，却没有一个经过 paired-256、独立 seed 和共同
evaluator 的主结论。

## 7. 最终投稿策略

### 7.1 主线

主线只写：

> proof-carrying formula-prefix reachability，在一次 all-attempt sampling
> pass 中减少 charge-invalid rich Plans，并通过冻结 exact-length Body-DLM
> 和 refiner 隔离其对 raw comp/joint 与 S.U.N. 的影响。

这句只有在 CrysVCD novelty gate 和 64/256/confirmation 全部通过后进入摘要。

### 7.2 备线

PILS-L 只在 2026-08-08 前 CR-Plan novelty 失败时接棒。它若过 256，可以写：

> position-indexed lattice-length sharing pools axis-specific ordinal
> supervision without changing numerical resolution or denoising length.

若它只过 64 或仅缩小 vocab，降为 appendix/negative diagnostic。

### 7.3 2026-08-15 无信号时

不启动 RL，不切 compiler，不改 safe-axis。论文退回现有 H1：

1. rich PlanGraph + exact-length masked Body-DLM + frozen continuous refiner；
2. conditional post-refiner structure validity 约 99.7%–99.8%；
3. Planner chemistry 是 raw composition/joint 的主要瓶颈；
4. decoding schedule 对 duplicate/completion 与 strict/meta polarization
   具有可复现因果影响；
5. 所有结论采用 all-attempt、无 repair/retry/rerank 的证据合同。

这是可写的 systems/diagnostic paper；不能把 stopped safe-axis 说成全指标提升。

### 7.4 可以写与绝不能写

可以写进 9 页主文的一句：

> We introduce a proof-carrying formula-prefix reachability decoder and
> causally evaluate its chemistry gains through a frozen exact-length crystal
> diffusion language model and continuous refiner under an all-attempt
> protocol.

绝不能写的过度主张：

> We are the first to combine planning, diffusion, chemical constraints, and
> reinforcement learning, and our method universally improves composition,
> structure, strict S.U.N., and meta S.U.N.

## 8. 最终结论

第二轮不应追求“Planner + token + RL 三件都做”。冻结事实、近邻碰撞与
45/52 天时钟共同指向一个更窄的选择：

```text
KEEP_MAIN   = P0 + CR-Plan
KEEP_BACKUP = PILS-L, only if CR-Plan novelty fails by 2026-08-08
CUT         = mask-aware RL for ICLR 2027
```

CR-Plan 的优势是直接命中 composition bottleneck、实现可在一周内给 64
机制信号、并与 frozen H1 pipeline 形成清楚因果链；它的致命弱点是 CrysVCD
新颖性碰撞和 comp→meta 缺乏保证。

PILS-L 是唯一值得保留的早期备线，因为它有清楚的 coverage hypothesis 和
两周 256 路径；但它不能被包装成 chemistry 方法，也不能在 CR-Plan 存活时
并行发展成第二主因素。

RL 的方法设计有价值，但 core two-stage action 已被 Mask-Aware PG 占据，
multi-objective crystal RL 又与 CRYSTAL/PLaID++ 邻近；在本地旧 TraceRL
NO-GO、support/evaluator 未冻结的条件下，继续做 RL 最可能牺牲的是主候选的
独立确认与论文写作，而不是增加可提交贡献。

2026-08-15 是不可移动的总止损线：届时没有唯一主候选 paired-64 机制信号，
就回到冻结 H1 叙事，不再用第三个方向救场。
