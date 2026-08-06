# Planner 作者对 Body-token / Mask-aware RL 的第二轮交叉审稿 V1

Status: `cross_review_only_no_execution_authorization`

Date: 2026-08-04

Reviewer stance: Planner/chemistry 提案作者，同时按苛刻外部审稿人标准反审
Planner、Body-token 和 RL 三案。本文不授权训练、生成、refinement、API、
checkpoint promotion 或 automatic downstream。

完整评审输入：

- `ROOT_EVIDENCE_AND_DEADLINE_FRAME_V1.md`；
- `PLANNER_CHEMISTRY_PROPOSAL_V1.md`；
- `BODY_TOKEN_PROPOSAL_V1.md`；
- `MASK_AWARE_RL_PROPOSAL_V1.md`。

## 0. 强制结论：一条主线、一条备线、其余 CUT

| 方向 | 最终标签 | 本轮允许的角色 |
|---|---|---|
| Planner formula-prefix charge reachability | **`KEEP_MAIN`，条件性** | 唯一 active scientific main |
| Body PILS-L length sharing | **`KEEP_BACKUP`，早切换条件性** | 唯一 cold backup；不得与 Planner 同时训练/跑 endpoint |
| Mask-aware Body RL | **`CUT`** | 投稿后方向；ICLR 2027 前不占 GPU、judge 或 DFT 资源 |

强制删减：

- P-control 只保留为 2026-08-10 前结束的 supporting audit/anchor check，不是
  第二条方法线；
- Planner compiler 本轮 `CUT`；
- reachability-mass SFT 本轮 `CUT`；
- Body support contraction、all-axis sharing、neighbor smoothing 本轮 `CUT`；
- RL 的 R0、PL sampler、multi-fidelity training 和 DFT audit 全部从当前
  execution queue 移除；
- safe-axis 继续作为 frozen stopped evidence，不与任何候选组合。

这不是说 compiler、token smoothing 或 RL 没研究价值，而是 45/52 天窗口
只能支持一个可确认的方法贡献。三线并跑最可能得到三个不完整 pilot，而不是
一篇可审稿的论文。

## 1. 统一评分

分数均为 1–5。`Feasibility` 越高越好；`Implementation risk` 越高越危险。
硬门优先于总分。

| 方向 | 标签 | Innovation | Expected effect | Existing evidence | Feasibility | Implementation risk | Deadline fit | Paper coherence |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Planner CR-Plan | `KEEP_MAIN` | 3 | 4 | 4 | 4 | 3 | 4 | 4 |
| Body PILS-L | `KEEP_BACKUP` | 2 | 2 | 2 | 3 | 4 | 3 | 3 |
| Mask-aware RL | `CUT` | 3 | 3 | 2 | 1 | 5 | 1 | 3 |

### 1.1 为什么不是按“算法看起来最复杂”排序

当前冻结事实是：

- Planner chemistry 是最大、最明确的 direct-validity 瓶颈；
- Body completion 仍有小空间，但 conditional structure 已约
  `99.7–99.9%`；
- meta 是 geometry/energy/refiner-basin 问题，不能从 comp 或 token
  coverage 自动推出；
- old TraceRL 正式 NO-GO，新的 joint policy、support、resume 和 evaluator
  均未落地。

因此排序依据是“能否在 08-15/08-22 前形成一条完整证据链”，不是概念组件
数量。

## 2. Planner / chemistry 自我反审

### 2.1 保留理由

Planner route 有三项其他两案不具备的冻结机制证据：

1. safe-axis successful-refined 中 35 个 comp-invalid 有 24 个 charge
   failure；
2. 更大历史 Plan panel 的 142 个 invalid 中有 98 个 charge failure；
3. P-control discovery 结果说明 Planner 分布确有约数个百分点的可移动空间，
   尽管其 `+4.30 pp` 尚未确认。

Formula-prefix reachability 直接作用于已观察 failure taxonomy，且不需要新
checkpoint、Body tokenizer 或 reward judge。它因此是最可能在十天左右给出
明确 paired-64 机制信号的候选。

### 2.2 创新性：只有 3/5

Planner 原案对新颖性的态度仍不够保守。

CrysVCD 已经覆盖：

```text
valence-balanced composition generation
  -> conditional diffusion structure generation
```

所以以下说法均不新：

- 先生成 composition 再生成 structure；
- 用 valence/charge rule 提高 composition validity；
- 把 symbolic chemistry constraint 接到 generative model。

CR-Plan 只能争取以下组合差异：

- constraint 在 tokenizer-level formula prefix 上生效；
- grammar × atom-budget × charge-reachability product automaton；
- 每个 terminal formula 有可复算 oxidation witness；
- single renormalized sample；
- no repair/retry/replacement/filter/rerank；
- rich Plan→exact-length masked DLM→refine800 的 all-attempt 因果评估。

必须在 2026-08-07 前完成 CrysVCD 及相邻工作的 claim-by-claim 对照。如果
CrysVCD 或别的工作已经实现等价 prefix DP/certificate，Planner 立即从
`KEEP_MAIN` 降为 `CUT`，不得靠改方法名称维持主线。

### 2.3 预期效果：只对 comp 有强因果预测

| 指标 | 严格评审判断 |
|---|---|
| charge failure | 有直接下降机制，目标至少 `-50%` |
| raw comp-valid | 最可信，合理 planning prior `+2~5 pp` |
| raw joint-valid | 可能随更多合法 Plan downstream 成功而提高 |
| conditional struct-valid | 应近似不变，只能要求非劣 |
| strict S.U.N. | 无直接保证 |
| meta S.U.N. | 无直接保证；只能作安全门 |

原 Planner 提案把 paired-256 meta `+2 pp` 或 CI 正作为 promotion gate，
这与方法机制不一致，也容易在 256 小样本上把一个真实 comp 方法误杀。
第二轮建议改为：

- primary efficacy：comp `>=+3 pp` 且 charge failure `>=-50%`；
- hard safety：meta、strict、structure、novelty、uniqueness 非劣；
- 不在标题或摘要承诺 meta improvement。

若项目要求“一种方法同时显著改善 comp、meta、strict、struct”，Planner
单线做不到；正确反应是缩小 claim，而不是追加 compiler/safe-axis 来凑全正。

### 2.4 现有证据：4/5，但不是 5/5

支持：

- failure taxonomy 直接；
- formula 已确定为 Planner 责任；
- Body/refiner 可保持冻结；
- P0/P-control inference 和 all-attempt contract 已存在。

缺口：

- P-control 缺完整 shortcut/drift/paired audit；
- 当前 24/992 charge bucket 对固定 sample 的静态空间只有约 `2.42 pp`；
- prefix DP 生成的是新 formula，效果和 diversity 仍未知；
- “仍可由未来元素补成中性”的 oracle 可能对大多数内部 prefix 都过于宽松，
  最终实际 treatment 退化成 newline mask；
- tokenizer token 可跨 element/count/newline，工程 parity 尚未证明。

Gate-64 必须报告 mask activation 的位置分解。如果几乎所有 action 只在最后
newline 被阻止，论文不能夸大为强 planning algorithm；它仍可作为有效的
certified terminal constraint，但 novelty 要再次下调。

### 2.5 实现风险：3/5

主要风险：

- 128k tokenizer 中 formula-relevant token transition 的完整枚举；
- oxidation-state DP 与冻结 SMACT oracle 不完全一致；
- future-element reachability 的 memoization 状态爆炸；
- constrained/unconstrained trajectory 分叉后，同 seed 不是严格 token-level
  common random number；
- mask 把概率质量推到 rare element/count tail；
- 新 chemsys 需要 common MP snapshot。

这些风险高于普通 grammar FSM，但明显低于 tokenizer remap+双臂 SFT 或完整
RL stack。

### 2.6 对原 Planner 提案的强制收缩

原案应在本轮执行层面收缩为：

```text
P0 or time-boxed frozen P-control anchor
  -> CR-Plan only
  -> D1/B0/model_494
```

删除：

- compiler C2；
- reachability-mass SFT；
- CR-Plan 与 safe-axis interaction；
- 09-05 才做完 `4×256` 的宽松安排。

P-control fresh confirmation 若到 2026-08-10 未结束，直接使用 P0。它不能
阻塞 product automaton，也不能因旧 `456/512` 被默认晋级。

### 2.7 日期和资源

| 里程碑 | 最早日期 | 硬截止 |
|---|---|---|
| novelty/P-control read-only audit | 2026-08-06 | **2026-08-07** |
| anchor freeze | 2026-08-08 | **2026-08-10** |
| paired-32 | 2026-08-11 | **2026-08-12** |
| paired-64 | 2026-08-14 | **2026-08-15** |
| paired-256 | 2026-08-21 | **2026-08-22** |
| independent confirmation | 2026-08-29 | **2026-08-31** |
| common science freeze | 2026-09-05 | **2026-09-05** |

最大可接受预算：

- 到 paired-64：`<=12 A800 GPUh`；
- 到 paired-256 累计：`<=72 A800 GPUh`；
- 独立 confirmation 追加：`<=64 A800 GPUh`；
- 投稿路径总计：`<=136 A800 GPUh`。

原案给 `4×256` 追加 192 A800 GPUh 过于宽松。按现有
`256 exact800/arm <=6 A800 GPUh` 上界，paired `4×256` 的 refiner 核心约
48 A800 GPUh；预留 Body/Planner/assembly 后，64 是更合理的硬 cap。

### 2.8 Planner 最可能成功与失败的机制

最可能成功：

> 大量 charge-invalid formula 在离开 formula line 时仍无中性 witness；
> 阻止这些 terminal transition 即可用较小 distribution shift 获得 comp gain。

最可能失败：

> 合法中性 completion 在原模型下概率过低，constraint 把采样推入 rare
> chemistry tail；comp point 上升但 diversity/meta 下降，或 treatment 实际只
> 是一个缺乏方法新颖性的 newline mask。

## 3. Body-token / PILS-L 审稿

### 3.1 判定

`KEEP_BACKUP`，不是并行第二主线。

PILS-L 是三案中最干净的 fallback：

```text
<LA_k>, <LB_k>, <LC_k> -> <L_k>
```

axis meaning 由 fixed answer position 决定，exact `7+4N`、D1、resolution 和
per-position branch count 不变。Matched continuation control 能把
representation effect 与额外训练 exposure 分开。

### 3.2 创新性：2/5

优点：

- exact-length schema 使 position-indexed semantics 很自然；
- axis pooling 与 crystal lattice calibration 形成一个清楚的 domain-specific
  mechanism；
- 相对 all-axis sharing，变化范围单一。

但本质仍是：

- vocabulary factorization；
- 三组 embedding/output rows 的 arithmetic-mean tying；
- matched SFT continuation。

它没有减少单个 length slot 的 500-way branching，模型总 vocab 只减少
`0.778%`，示例 hidden width 下 matrix storage 只节省约 15.7 MiB。若 endpoint
没有显著改善，reviewer 很容易把它判为 tokenizer cleanup。

PLaID++ 已强化“crystal-specific representation matters”的宽泛叙事；PILS-L
不能宣称首次为晶体生成设计 position-aware representation，只能声称
exact-length DLM 中 axis-redundant numeric support 的特定因子化。

### 3.3 预期效果：2/5

| 指标 | 严格评审判断 |
|---|---|
| shared-L NLL/calibration | 有合理改善机制 |
| body completion | 小到中等可能 |
| refiner displacement | 方向未知 |
| meta S.U.N. | 有物理故事，但目前只是推测 |
| strict S.U.N. | 只能要求非劣 |
| conditional comp-valid | 精确为无直接作用 |
| conditional struct-valid | 已近 ceiling，预计变化很小 |

“length 决定 volume/density，所以 sharing 会改善 meta”缺少冻结因果证据。
R03H 只证明坐标/schedule 可改变 hull 分布，并没有证明 unseen length token 是
meta bottleneck。

### 3.4 现有证据：2/5

支持：

- held-out numeric unseen 中 `953/1,013` 来自 length；
- shared bin 可以聚合三轴 occurrence；
- fixed-slot scaffold 证明 row remap 和 position parser 可实现。

关键缺口：

- 完整 train 27,136 coverage 尚未恢复；
- held-out-unseen 不能写成 training-unseen；
- token identity coverage 不等于 legal probability mass；
- 三轴 length 每行都有 supervision，总 occurrence 并不稀缺；
- fixed-slot scaffold 不是 R5 exact-dynamic terminal；
- 当前 B0 completion 已高，conditional structure 已饱和；
- 没有 length calibration→refiner burden→meta 的本地因果结果。

因此 Gate-1 是真正的生死门，不是形式审计。以下任一成立即 `CUT`：

- train-unseen/rare mass 很小；
- `mean(m_zero+m_rare)<0.5%` 且 `p95<2%`；
- axis pooling 后 median positive target count 未提高 2 倍；
- endpoint-attributed rare length emissions没有 headroom。

### 3.5 实现风险：4/5

PILS-L 需要：

- 新 tokenizer 与 vocabulary SHA；
- B0 embedding/output row remap；
- 45,229 rows exact-dynamic rebuild/round-trip；
- C0/L1 两臂 matched one-epoch SFT；
- 2×A800 DDP；
- 新 formal sampler；
- model_494 exact800 与 S.U.N.。

它虽然是一个 representation factor，但实现上同时触及 data、tokenizer、
checkpoint、training 和 inference。Matched C0 是必要的，否则无法区分
sharing 与 continuation exposure。

提案的 08-06 Gate-1、08-08 conversion、08-11 双臂 SFT、08-14 paired-64
计划可行但没有返工空间。任何 tokenizer/package/DDP 问题都会吞掉 backup
窗口。

### 3.6 Novelty collision

- **CrysVCD：** direct collision 低；但“composition→diffusion”系统分层不
  能当 PILS-L 贡献。
- **PLaID++：** 中等宽泛碰撞；两者都强调 crystal representation 与生成质量，
  但 PILS-L 不是 Wyckoff text 或 preference alignment。
- **CRYSTAL：** direct collision 低；PILS-L 不做 RL。
- **Mask-Aware PG：** direct collision 低；但一旦 PILS-L 与 RL 同跑，
  legal support、policy likelihood 和 action identity 全部改变，无法归因。

### 3.7 作为备线的启动规则

允许在 2026-08-07 前完成：

- full-corpus Gate-1 read-only audit；
- source/tokenizer map design；
- CPU unit-test scaffold。

不允许在 Planner active 时进行：

- C0/L1 matched SFT；
- Body generation；
- refine800；
- S.U.N. endpoint selection。

只有 Planner 在 novelty/engineering 上不晚于 2026-08-10 被 CUT，才能启动
PILS-L matched SFT。若 Planner 到 08-10 仍是 active main，PILS-L 对本次投稿
保持 cold backup，不再执行。

如果 Planner 到 paired-64 的 08-15 才 scientific stop，此时不再晚切
PILS-L；论文直接回到 frozen H1 fallback。原因是从 08-15 再启动 tokenizer
remap+双臂 SFT 无法可靠满足 08-22 paired-256。

### 3.8 日期和资源

如果不晚于 2026-08-10 激活：

| 里程碑 | 最早日期 | 硬截止 |
|---|---|---|
| Gate-1 | 2026-08-06 | **2026-08-07** |
| conversion/unit terminal | 2026-08-09 | **2026-08-10** |
| C0/L1 matched SFT | 2026-08-11 | **2026-08-12** |
| paired-32 | 2026-08-13 | **2026-08-13** |
| paired-64 | 2026-08-14 | **2026-08-15** |
| paired-256 | 2026-08-19 | **2026-08-22** |
| independent confirmation | 2026-08-29 | **2026-08-31** |

最大可接受预算：

- Gate-1/engineering：`<=12 A800 GPUh`；
- matched SFT + paired-256 累计：`<=96 A800 GPUh`；
- `4×256` confirmation 追加：`<=48 A800 GPUh`；
- 总计：`<=144 A800 GPUh`。

### 3.9 Body 最可能成功与失败的机制

最可能成功：

> 三轴共享使 rare length bins 获得更稳定的 embedding/output calibration，
> 减少极端 lattice emission 和 refiner displacement，产生小而稳定的 meta
> gain。

最可能失败：

> full-train audit 显示 held-out identity gap 不对应实际 legal probability
> mass；sharing 只做了普通 parameter tying，C0/L1 endpoint 无差别，或新
> tokenizer/SFT 引入的 drift 大于任何 calibration gain。

## 4. Mask-aware RL 审稿

### 4.1 判定

**`CUT` for ICLR 2027。**

保留内部设计文档，投稿后再做；本轮不运行 Gate −1/R0/32，不预占 RL
judge、DFT 或 96 A800 GPUh。

这是时间和证据判断，不是说方向在长期无价值。

### 4.2 创新性：3/5，而非提案自评 4/5

Mask-Aware Policy Gradients 已明确：

- diffusion LM action 包含 token 与 reveal/remask position；
- greedy top-k 不能提供完整 position policy gradient；
- Plackett–Luce 可以参数化 reveal order。

因此 RL 案的以下核心不能再算本项目原创：

- token + position two-stage action；
- PL position likelihood；
- 优化 unmask/reveal order。

可争取的差异仅为：

- crystallographic legal support；
- D1 group-local constraint；
- frozen-Plan composition 不进 advantage；
- randomized pre/post-refiner Horvitz–Thompson correction；
- all-attempt evaluator firewall。

CRYSTAL 已覆盖 coordinated multi-objective crystal RL，PLaID++ 已覆盖
iterative preference alignment 和 evaluator isolation。把这些已知组件组合
到当前 pipeline 仍可能有方法价值，但不足以抵消实现与时间风险。

### 4.3 预期效果：3/5

优点：

- reveal order 已被 R03 证明会改变 strict/meta 分布；
- reward 可以直接作用于 refiner basin/stability；
- 理论上比 Planner/Body token 更直接面向 strict/meta。

限制：

- fixed Plan 下 comp advantage 为 0；
- conditional structure 已接近饱和；
- safe-axis 证明 strict 与 meta 可能极化；
- 1,024 rollouts、512 B labels、最多 20 optimizer steps 是否足以学到稳定
  policy 完全未知；
- reward 的 independent-evaluator predictivity 尚未建立。

### 4.4 现有证据：2/5

支持：

- R03 证明 reveal schedule 是因果变量；
- Mask-Aware PG 支持 two-stage action；
- exact token support 理论上可枚举。

缺失：

- full train legal-mass Gate −1；
- usable joint rollout implementation；
- exact replay/resume；
- locally frozen `E_train/E_gate/E_final`；
- A/B pre/post-refiner predictivity；
- nonzero group advantage rate；
-任何本地 RL efficacy result。

First-64 按 RL 提案定义仍然是 sampler/reward calibration、没有训练，不是
efficacy signal。也就是说 08-15 才知道“是否能开始训练”，而共同框架要求
主候选此时已有机制信号。这一时间结构本身就不适合作为当前 active line。

### 4.5 实现风险：5/5

RL stack 同时依赖：

1. frozen special-token support；
2. D1 greedy→K=1 PL sampler factor；
3. current-group all-candidate trace；
4. exact token/position joint likelihood；
5. high-variance normalized surrogate；
6. behavior/reference/old actor separation；
7. KL、ESS、clip 和 entropy telemetry；
8. transaction-level exact resume；
9. A/B label cache 与 nondeterministic model_494；
10. `E_train/E_gate/E_final` isolation；
11. common hull snapshot；
12. 192-structure DFT audit。

这是至少四个尚未闭合的 prerequisite stacks，不是一个“最小 LoRA”。

额外理论/统计风险：

- `N=20` 时一个 trajectory 记录约 651 candidate-token actions，exact joint
  ratio 高方差；
- token/position channel normalization 是稳定化 surrogate，不等于 exact
  sequence marginal policy gradient；
- `R_A + 2Z(R_B-R_A)` 虽无偏，但方差可能很大且可超出原 reward range；
- `G=8` 下 strict/meta/novelty reward 可能大量 zero-variance group；
- model_494 process nondeterminism 增加 B label noise；
- 同一小 panel 不能同时校准 reward、tau、judge 和证明 efficacy。

### 4.6 Novelty collision

- **CrysVCD：** direct RL collision 低；系统的 composition→diffusion 分层仍
  不是新贡献。
- **PLaID++：** 中高碰撞；iterative preference alignment、MLIP isolation、
  stability/novelty trade-off 已存在。
- **CRYSTAL：** 高碰撞；crystal multi-objective RL 与 diversity guards 已
  存在。Frozen composition 与 masked action 是差异。
- **Mask-Aware PG：** 最高碰撞；joint token/reveal action 和 PL 是直接近邻。

### 4.7 为什么不是 `KEEP_BACKUP`

程序已经只允许一条备线。PILS-L 相对 RL：

- prerequisite 更少；
- 无 reward judge hacking；
- 无 behavior-policy likelihood/replay/resume；
- 无 A/B/DFT 三层 evaluator；
- 64/256 前有清楚的 NLL/calibration mechanism；
- 失败不会污染 policy/evaluator。

RL 若作为第二备线，会重新制造三线并跑。最合理的处置是现在 `CUT`，保留
设计供投稿后执行。

### 4.8 日期和资源

审稿决定日：**2026-08-04 CUT。**

本次投稿最大可接受 RL 资源：

```text
0 A800 GPUh
0 new reward labels
0 DFT structures
```

RL 原案的 `96 A800 GPUh` 是一个长期设计预算，不是当前 program 获批预算。

RL 若投稿后重启，必须重新证明：

- support SHA 已在独立 Body decision 后冻结；
- Mask-Aware PG/CRYSTAL/PLaID++ novelty 对照；
- R0 exact replay/resume；
- independent judge availability；
- A/B predictivity；
- 单独的新 deadline 和授权。

### 4.9 RL 最可能成功与失败的机制

最可能成功：

> joint token/reveal optimization 学到更合适的 D1 group 内 reveal order，
> 把合法 proposal 推入更好的 model_494 basin，同时 meta-aware reward 避免
> strict-only 极化。

最可能失败：

> Mask-Aware PG 的核心已非新颖；本地 A/B reward 相关性弱且 group advantage
> 稀疏，normalized surrogate 优化 proxy/rare token 而非 independent
> stability，最终花完截止窗口仍只有工程 trace。

## 5. 四个近邻工作的 collision 总表

| 方向 | CrysVCD | PLaID++ | CRYSTAL | Mask-Aware PG |
|---|---|---|---|---|
| Planner CR-Plan | **高：valence constraint 与 composition→diffusion**；仅 prefix/certificate 可区分 | 低：不做 preference alignment | 低：不做 RL；但多指标 guard 不是新 | 低：Planner 是 AR prefix，不是 Body reveal |
| Body PILS-L | 低：只共享 length representation | 中：crystal-specific representation 宽泛碰撞 | 低 | 低；若以后进 RL，support identity 成 prerequisite |
| Mask-aware RL | 低 | 中高：preference/evaluator isolation | **高：crystal multi-objective RL** | **最高：joint token-position + PL** |

全项目共同 claim boundary：

- 不能说首次 composition→diffusion；
- 不能说首次保证 charge balance；
- 不能说首次 crystal RL；
- 不能说首次 token+reveal policy；
- 不能仅凭组件组合使用 “first”；
- 只有 frozen-factor、all-attempt、independent-seed 和 common-evaluator
  证据才能支撑实际系统 claim。

## 6. Confounding 与 dependency 审计

### 6.1 依赖图

```text
Body full-corpus Gate-1
  ├─ informs whether PILS-L has headroom
  └─ would be mandatory before any future RL

Planner CR-Plan
  └─ must use one frozen P0/P-control anchor

PILS-L
  └─ changes tokenizer, support, checkpoint and Body training

Mask-aware RL
  └─ requires tokenizer/support/checkpoint already frozen
```

由此可见，PILS-L 与 RL 不是两个可并行、独立的 Body factors。PILS-L 一旦
改变 token IDs/support，RL 的 old log-prob、trace、support SHA、policy action
和 replay fixtures 全部失效。

### 6.2 禁止的组合

| 组合 | 为什么 confounded |
|---|---|
| CR-Plan + PILS-L 首测 | 同时改变 formula distribution 与 Body representation |
| CR-Plan + RL 首测 | formula/Plan cohort 与 Body policy 同时变化 |
| PILS-L + RL | tokenizer/support/checkpoint 与 policy optimization 同时变化 |
| P-control + CR-Plan 首测 | checkpoint 与 decoding constraint 同时变化；必须先 freeze anchor |
| safe-axis + 任一新线 | safe-axis 已 strict+/meta- scientific stop，会混入已知 polarization |
| compiler + CR-Plan | formula validity 与 semantic-field source 同时改变 |
| training SFT + hard CR-Plan | learned support 与 decoding support 同时改变 |
| reward judge = final judge | evaluator hacking，无法作独立科学确认 |

### 6.3 共同 evaluator 和 denominator

所有保留路线必须：

- 使用 raw all-attempt denominator；
- 保留 Planner/body/refiner failure；
- no retry/replacement/repair/filter/rerank；
- proposal-before-refine 与 after-refine 分开；
- common Direct、novelty database 和 MP snapshot；
- 新 chemsys unknown 对两臂共同处理；
- model_494 process repeats 只估 variance，不冒充 scientific panels；
- 08-31 前至少完成一个真正独立 seed confirmation；
- 09-05 后不再启动 principal factor。

## 7. 唯一允许的执行/切换策略

### 7.1 2026-08-04 至 08-07

允许同时进行的仅是零 treatment 的只读工作：

- Planner novelty/P-control audit；
- Body full-corpus Gate-1；
- common evaluator/cache inventory；
- paper outline 和 baseline table audit。

RL 不运行 R0，也不准备独立 judge/DFT。

### 7.2 2026-08-07 至 08-10

优先激活 Planner：

- 08-07 novelty gate 通过才继续；
- P-control 若不能在 08-10 前无歧义确认，则固定 P0；
- product-automaton unit/preflight 必须在 08-10 前显示可实现。

Body 只可完成 CPU conversion/test scaffold，不可启动 C0/L1 SFT。

### 7.3 早切换到 Body 的唯一窗口

只有以下任一情况不晚于 **2026-08-10** 发生，才激活 PILS-L：

- Planner novelty collision 判定失败；
- product automaton tokenizer/oracle parity 工程失败；
- P0/P-control anchor evidence不可恢复且 P0 fallback 也无法形成合同。

切换后：

- Planner scientific generation 停止；
- compiler/training/RL 保持 CUT；
- PILS-L 成为唯一 active candidate；
- 08-15 paired-64、08-22 paired-256、08-31 confirmation 的共同截止不变。

### 7.4 不允许的晚切换

如果 Planner 到 08-15 paired-64 才 scientific stop：

- 不再启动 PILS-L training；
- 不启动 RL；
- 论文回到 frozen H1/R03 fallback。

原因：晚切换会迫使 backup 跳过 matched SFT、64 或 independent confirmation，
最终只能制造 post-hoc pilot。

### 7.5 若 Planner 通过

- 08-15 通过 64 后，Body backup 对本次投稿冻结；
- 08-22 前只扩 Planner paired-256；
- 08-31 前只做 Planner independent confirmation；
- 09-05 science freeze；
- 之后只写作、共同评测审计和复现包。

## 8. 三个最可能杀死论文的问题

### Paper-killer 1：新颖性被最近工作完全吃掉

- CR-Plan 若只是 CrysVCD valence mask 的 tokenizer 版本；
- PILS-L 若只是常规 token tying；
- RL 若只是 Mask-Aware PG + CRYSTAL reward 的领域复现；

那么即使指标正，9 页主文也缺少不可替代的方法贡献。08-07 的 literature
claim table 是硬门，不是写作末期补引用。

### Paper-killer 2：机制与目标 endpoint 错位

- Planner 最可能改善 comp，却被要求自动改善 meta；
- PILS-L 的 held-out token identity gap 未必是 legal-mass/meta bottleneck；
- RL proxy 可能与 independent stability 反向。

若为了“全指标都升”叠加 Planner、token、safe-axis、compiler 或 reward，
因果归因会消失。应接受一个清楚的 primary gain 加严格非劣门，而不是拼装
全正表。

### Paper-killer 3：时间内拿不到共同口径的独立证据

风险包括：

- model_494 非 bitwise deterministic；
- 新 chemsys MP coverage/API；
- tokenizer/package/DDP 或 joint-trace/resume 工程；
- independent seed 与 process repeat 混淆；
- H1-A2/CrysLLMGen 的 denominator/evaluator 口径不一致。

若 08-22 只有 64 pilot、09-05 仍无 common evaluator/independent confirmation，
论文会变成方法设想加不完整诊断。

## 9. 论文叙事和 claim

### 9.1 Planner 主线成功时

一句可进入 9 页主文：

> We introduce a proof-carrying formula-prefix reachability decoder that
> removes charge-infeasible Planner trajectories in a single all-attempt
> sample while keeping the exact-length Body DLM and continuous refiner
> frozen.

主表必须展示：

- P0/frozen anchor；
- anchor+CR-Plan；
- raw comp/structure/joint；
- conditional structure；
- proposal-before/after-refine；
- strict/meta/novel/unique；
- complete failure taxonomy；
- independent panel sign stability。

### 9.2 Body 备线成功时

一句可进入 9 页主文：

> We factor axis-redundant lattice-length tokens by their fixed exact-length
> positions, improving shared-length calibration and downstream stability
> without reducing numerical resolution or changing denoising actions.

必须同时有：

- full-train/legal-mass证据；
- matched continuation control；
- NLL/calibration；
- lattice distribution；
- refiner displacement；
- meta gain 与 strict/structure 非劣。

### 9.3 绝不能写

> We are the first to guarantee chemically valid crystals and our Planner,
> token compression, and reinforcement learning jointly improve composition,
> structure, strict S.U.N., and meta S.U.N.

这句话同时违反 novelty、single-factor 和冻结证据边界。

### 9.4 08-15 无信号时的 fallback

若唯一 active candidate 到 08-15 没有 paired-64：

1. 立刻停止新模型线；
2. 不从 CUT 方案中选一个小样本赢家；
3. 论文退回 frozen H1/R03：
   - rich PlanGraph + exact-length Body-DLM + frozen refiner；
   - D1/schedule-constraint interface；
   - 约 99.8% conditional structure validity；
   - Planner chemistry bottleneck；
   - safe-axis strict/meta polarization；
4. 新 Planner/Body/RL 只作 discussion/future work，不写 efficacy claim。

## 10. 最终审稿意见

### Planner：`KEEP_MAIN`

条件：

- 08-07 通过 CrysVCD novelty gate；
- 08-10 冻结 anchor/oracle；
- 08-15 paired-64 有 charge/comp 机制信号；
- 08-22 paired-256 完成；
- meta/strict/structure/novelty/uniqueness 均非劣；
- 08-31 独立 confirmation。

它是最直接作用于已知 bottleneck、实现依赖最少的路线。必须承认它主要改善
comp，而不是承诺 meta。

### Body-token：`KEEP_BACKUP`

条件：

- Gate-1 证明 full-train legal-mass headroom；
- Planner 不晚于 08-10 被 CUT；
- 之后成为唯一 active line；
- matched control、64/256/confirmation 不跳级。

它的 novelty 较弱、meta 链条尚未证明，但比 RL 更可执行、更易归因。

### Mask-aware RL：`CUT`

原因：

- 与 Mask-Aware PG/CRYSTAL/PLaID++ 的碰撞最强；
- prerequisite 与 evaluator stack 尚未闭合；
- first-64 仍无训练 efficacy；
- 96 A800 GPUh、512 exact800 train labels、4×256 和 DFT audit 无法在当前
  截止内以低风险完成；
- 它不能改善 frozen Planner composition。

本轮给 RL 的资源上限为零。保留设计，投稿后在 token/support 和 evaluator
各自冻结后重新注册。

### 最终唯一选择

```text
MAIN   = Planner CR-Plan
BACKUP = Body PILS-L, only for an early switch by 2026-08-10
CUT    = Mask-aware RL
```

这套选择牺牲了“所有想法都试一下”的覆盖率，换取一条在 08-22 前能被
证伪、在 08-31 前能独立确认、并可在 9 页主文中讲清楚的因果主线。
