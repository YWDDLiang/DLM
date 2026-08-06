# Planner / Body-token / Mask-aware RL 第二轮交叉评审 V1

状态：`cross_review_decision_only_no_execution_authorization`

日期：2026-08-04

审稿立场：mask-aware RL 提案作者，同时按 ICLR 2027 截止时间作苛刻的
portfolio reviewer。

本文完整审阅：

- `ROOT_EVIDENCE_AND_DEADLINE_FRAME_V1.md`；
- `PLANNER_CHEMISTRY_PROPOSAL_V1.md`；
- `BODY_TOKEN_PROPOSAL_V1.md`；
- `MASK_AWARE_RL_PROPOSAL_V1.md`；
- CrysVCD、PLaID++、CRYSTAL 与 Mask-Aware Policy Gradients 的一手论文。

本文只作方向选择与停止判定，不授权训练、生成、refinement、S.U.N.、API
查询、checkpoint promotion 或自动下游。

---

## 0. 强制结论：一主、一备、一砍

| 方向 | 最终标签 | 本轮保留的唯一版本 | 结论 |
|---|---|---|---|
| Planner / chemistry | **`KEEP_MAIN`** | P0 上的 formula-prefix charge-reachability product automaton | 最贴近已证实的 composition 瓶颈；8 月 15 日前必须过 novelty 与 paired-64 双门 |
| Body-token / support | **`KEEP_BACKUP`** | PILS-L：只共享 LA/LB/LC length family | 只在 full-train Gate−1 与 matched-control 64 同时通过时保留；不与主线组合 |
| Mask-aware RL | **`CUT`** | 无 | 从 ICLR 2027 principal-factor portfolio 中移除；设计保留到投稿后，不占训练/评测资源 |

这是角色预注册，不是 64 结果出来后再挑最好看的方向：

```text
Planner novelty + paired-64 PASS
  -> Planner 是唯一进入 paired-256 的方向
  -> Body 备线冻结，不做 256

Planner 在 novelty 或 paired-64 上 FAIL
  AND Body Gate−1 + paired-64 PASS by 2026-08-15
  -> Body 在 2026-08-15 当天一次性升为唯一主线

两者均未在 2026-08-15 前 PASS
  -> 新方法全部 CUT
  -> 退回冻结 H1 systems/diagnostic 论文
```

不得把 Planner、PILS-L 与 RL 的局部正结果拼成一个未经 factorial
拆分的“全指标系统”。

---

## 1. 统一评分

分数为 1–5；硬门优先，不能用总分抵消一个 fatal gate。

| 方向 | Innovation | Expected effect | Evidence | Feasibility | Deadline fit | Paper coherence |
|---|---:|---:|---:|---:|---:|---:|
| Planner CR-Plan | 3 | 4 | 4 | 4 | 4 | 5 |
| Body PILS-L | 2 | 3 | 2 | 3 | 3 | 4 |
| Mask-aware RL | 3 | 3 | 2 | 2 | 1 | 3 |

### 1.1 Planner CR-Plan：为什么是主线

- **Innovation = 3。** CrysVCD 已占据“valence-balanced composition
  generator + diffusion structure generator”的宽泛叙事，不能给 4–5。
  可争取的差异是现有 rich Plan formula prefix 上的 grammar × finite-budget
  charge-reachability product automaton、terminal witness、单次 legal-support
  renormalization 和 raw all-attempt 因果合同。
- **Expected effect = 4。** 现有 35 个 composition-invalid 中 24 个是
  charge failure；这是三条路线中唯一直接改变 conditional chemistry 的方法。
  它不能承诺 meta 或 conditional structure 提升。
- **Evidence = 4。** 本地 failure taxonomy 与 P-control discovery signal
  都支持 Planner 是 composition 瓶颈；机制有可复算 certificate。
- **Feasibility = 4。** 不训练大模型，不改 tokenizer/Body/refiner；主要工作是
  tokenizer-aware DFA、reachability DP、telemetry 和 paired generation。
- **Deadline fit = 4。** 只要不等待 P-control 重新选 anchor，08-14/15 的
  64 和 08-21/22 的 256 可达。
- **Paper coherence = 5。** 它自然连接 rich PlanGraph、exact-length Body-DLM
  和 frozen refiner，并服务 raw composition/joint-valid 主轴。

### 1.2 Body PILS-L：为什么只作备线

- **Innovation = 2。** position-indexed token sharing 是合理因子化，但总体
  vocabulary 只缩小 0.778%，单位置 branching factor 不变；若没有完整
  coverage→calibration→refiner-basin→meta 链，更像表示工程。
- **Expected effect = 3。** length calibration 可能改善 completion、volume
  tail、refiner displacement 与 meta；它不能改变 frozen Plan 的 conditional
  composition，conditional structure 又已接近 ceiling。
- **Evidence = 2。** 当前只有 held-out coverage；`953` 个 identity 的缺口
  不能冒充 train-unseen，且尚无 B0 legal probability mass 证据。
- **Feasibility = 3。** fixed-slot scaffold 可借鉴，但 exact-dynamic tokenizer、
  row remap、matched C0/L1 SFT 和 formal sampler 都需新实现。
- **Deadline fit = 3。** 若 08-06 Gate−1 立即通过，08-14 的 64 仍可能；
  两个 matched SFT arm 的实际 A800 GPUh 尚未被清楚换算，是显著风险。
- **Paper coherence = 4。** 成功时能解释 lattice-token calibration 如何影响
  frozen refiner basin；失败时不会破坏现有 B0/D1 锚点。

### 1.3 Mask-aware RL：为什么现在砍

- **Innovation = 3。** exact crystal legal support、D1 groups 和随机
  pre/post-refiner control variate 的组合有差异，但 joint token-position
  policy 与 Plackett–Luce reveal 已被 Mask-Aware PG/DCoLT 明确提出；
  crystal multi-objective RL 又被 CRYSTAL、PLaID++、CrysTune 覆盖。
- **Expected effect = 3。** 可能改善 strict/meta、novelty 与 refiner basin；
  fixed Plan 内 composition advantage 恒定，不能修复 comp_valid，structure
  又近饱和。
- **Evidence = 2。** R03 只证明 reveal order 会改变结果且 strict/meta 可极化，
  尚未证明 policy gradient 能产生正向效果。
- **Feasibility = 2。** 历史 TraceRL 正式 NO-GO；必须重写 rollout、
  legal likelihood、position probability、old/reference separation、reward、
  exact replay/resume 和 evaluator firewall。
- **Deadline fit = 1。** first-64 只能证明 sampler/reward calibration，
  不是 RL efficacy；随后还要训练、独立 judge、paired-256 与 4×256。
- **Paper coherence = 3。** 成功时故事强，但会把当前清楚的 Planner bottleneck
  叙事转成高风险的 post-training 论文，并引入最多的新依赖。

与 RL 提案自身的较乐观评分相比，本轮下降的原因不是方法价值改变，而是
portfolio 比较后必须只给一主一备；在 45/52 天窗口内，机会成本也是硬风险。

---

## 2. Planner 主线的审稿意见

### 2.1 接受的核心

保留的 intervention 只有：

```text
P0 formula autoregressive prefix
  + frozen grammar state
  + frozen oxidation table
  + finite N<=20 charge-reachability DP
  + legal-logit renormalization and one sample
  + terminal oxidation witness
```

以下 Planner 子路线不进入本轮方法 portfolio：

- P-control：只作 supporting audit/baseline，不是贡献；
- formula-derived compiler：`CUT`；
- reachability-mass SFT：`CUT`；
- typed two-stage Planner：`CUT`；
- P-control + reachability、reachability + compiler 等组合：`CUT`。

### 2.2 对原提案的必要修正

#### 修正 A：主实验立即冻结 P0，不等待 P-control 选 anchor

原提案先做新 1,024 P-control confirmation，再在 08-12 选择 P0/P-control，
会把 CR-Plan 的 32/64 压缩到两天，也增加一个 post-selection dependency。

本评审要求：

- 2026-08-07 前把 **P0 冻结为 CR-Plan primary anchor**；
- P-control 旧 ledger 只读审计可以并行；
- 新 P-control confirmation 若资源允许可作为 supporting result，但不得改变
  CR-Plan primary arm，也不得阻塞 64；
- 不得把 P-control 与 reachability 同时打开后归因给 CR-Plan。

#### 修正 B：删除全部 late grace

原提案中的 `08-24 late cut`、`09-05 final panels` 和 `09-10 result freeze`
与统一框架冲突，全部由以下日期覆盖：

- paired-64：目标 2026-08-14，**绝对最迟 2026-08-15**；
- paired-256：目标 2026-08-21，**绝对最迟 2026-08-22**；
- `4×256` 与独立 audit：**绝对最迟 2026-08-31**；
- science/results freeze：**2026-09-05**；
- paper/supp/repro only-fix：**2026-09-12**。

#### 修正 C：把 meta 当安全门，不把 charge-neutral 写成稳定性方法

在 64：

- charge failure 至少下降 50%；
- raw comp-valid 至少 `+3/64`；
- no shortcut/support collapse；
- meta point 不为负，只作 early safety，不作 efficacy claim。

在 256：

- raw comp-valid 至少 `+8/256`；
- charge failure至少下降 50%；
- raw joint-valid 为正；
- meta point `>=0`，one-sided paired lower bound `>-2 pp`；
- strict、raw/conditional structure、novelty、uniqueness 全部非劣。

只有 meta point 提升且独立确认稳定时，才可说 downstream meta 改善；否则论文
主张只落在 chemistry 与 raw joint-valid。

### 2.3 最可能成功与失败的机制

最可能成功：

> 大部分 charge-invalid path 在 formula 结束前仍有高概率的局部非法终止；
> 阻止这些 terminal/newline，而不显著删除内部高概率 token，就能用小
> distribution shift 换取 charge failure 的确定下降。

最可能失败：

> 中性 suffix 在 P0 下概率质量过低；prefix mask 把模型推入稀有
> element/count、alloy或高-hull composition，造成 dead-end、多样性下降，
> 或出现 comp 上升但 meta 下降。

### 2.4 算力上限

- Planner 主线从 Gate 32 到 `4×256` 的**累计上限为 `96 A800 GPUh`**；
- 08-15 前上限 `12 A800 GPUh`；
- P-control supporting sampling、Body/refine 和 independent evaluator 均计入；
- compiler、Planner training 或与 safe-axis 的 interaction 不得另开预算；
- 达到 cap 即停，不用减少 denominator、独立 panel 或 common evaluator
  换取继续。

---

## 3. Body-token 备线的审稿意见

### 3.1 有条件保留的原因

PILS-L 是比 support pruning、全轴 sharing 或 neighbor smoothing 更干净的
单 treatment：

```text
<LA_k>, <LB_k>, <LC_k> -> <L_k>
```

position 仍决定 a/b/c，bin resolution、每步 legal branches、exact
`7+4N`、D1、Planner、refiner 与 evaluator 均不变。它对 meta 的作用路径比
Planner charge constraint 更直接，因此是合理备线。

### 3.2 必须先杀掉的三个不确定性

1. **Train coverage 未知。** 08-06 前必须恢复 full train/val/test 并计算
   B0 constrained legal mass；held-out identity 数不能作为 treatment 理由。
2. **Compute 单位不清。** “两臂总计 24–48 小时”必须转换成明确的
   `A800 GPUh = GPU count × wall hours`；08-07 profiling 后若到 paired-64
   的预计累计超过 64 GPUh，备线直接 CUT。
3. **Matched control 才能归因。** Primary 必须是 L1 vs matched-continuation
   C0；L1 vs historical B0 只能描述。否则 continuation、row remap 与 sharing
   混在一起。

即使使用 matched C0，能归因的对象也是
“arithmetic-mean remap + shared representation + matched continuation”
这个完整 PILS-L package，不能进一步声称 row averaging 或 sharing 单独导致
endpoint 改善。

### 3.3 角色与资源纪律

- 2026-08-15 前允许完成 Gate−1、conversion/tests、matched SFT、32 和 64；
- 若 Planner 通过 novelty + 64，PILS-L 立即冻结，**不做 256**；
- 只有 Planner CUT 且 PILS-L 自己的 64 全过，PILS-L 才在 08-15 升为唯一
  主线；
- 升级后 paired-256 最迟 08-22，`4×256` 最迟 08-31；
- 禁止把 PILS-L 与 CR-Plan、safe-axis 或 RL 组合成 final candidate。

算力：

- 备线在 08-15 前累计上限：`64 A800 GPUh`；
- 只有正式升为主线后，完整路径累计上限放宽到 `128 A800 GPUh`；
- 若 matched SFT 本身已使 64 cap 不可达，则 Body 路线 CUT，不追加 GPU。

### 3.4 最可能成功与失败的机制

最可能成功：

> 三轴长度 bin 的共享监督改善 rare-bin likelihood/calibration，减少极端
> lattice 与 refiner displacement，从而让 meta 上升而 strict 不极化。

最可能失败：

> 三轴长度分布并不同质，position context 不足以恢复 axis-specific prior；
> row averaging 与共享训练反而抹平有用的各向异性，导致 volume tail、
> completion 或 meta 恶化。另一种更简单的失败是 full-train legal mass
> 证明 held-out identity gap 根本不影响 B0。

---

## 4. Mask-aware RL 的拒稿意见

### 4.1 `CUT` 的含义

`CUT` 只针对当前 ICLR 2027 principal-factor portfolio，不是否定长期方向。
投稿前：

- 新 RL 训练：不启动；
- RL paired-256/`4×256`：不启动；
- 两个长期 LoRA：不启动；
- 历史 TraceRL：不得用于正式结果；
- A800 增量预算：`0 GPUh`。

现有设计文档、trace schema 与 reward/evaluator 审计可保留到投稿后。

### 4.2 为什么不能给 `KEEP_BACKUP`

1. 旧 trainer 缺 legal-support renormalization、position log-prob、真实 online
   old log-prob 与 exact resume，不能作为最小修改起点。
2. RL 的 first-64 是 sampler、reward variance、A/B correlation 和 independent
   judge calibration；即使全过，也没有经过优化的 policy，不能与另两个方向的
   64 endpoint signal同义比较。
3. fixed Plan 下 composition reward无 variance；RL 不能服务当前最大、最明确的
   comp_valid 瓶颈。
4. conditional structure validity 已约 `99.7–99.9%`，可用结构 headroom 很小。
5. reward 与 final evaluator 必须隔离；这要求至少两个 MLIP checkpoint/family
   和独立 audit，资源依赖远高于 Planner/Body。
6. Mask-Aware PG 已使“token + reveal position 的 PL policy gradient”不再是
   独立新颖点；CRYSTAL/PLaID++ 又已占据 crystal multi-objective alignment。

### 4.3 投稿后最小恢复条件

投稿后若重启，仍建议一个 multi-fidelity LoRA，而不是两个长期模型，并先满足：

- Gate−1 support SHA 冻结；
- D1 K=1 PL no-RL control；
- joint token-position trace/replay/exact resume；
- E_train/E_gate/E_final isolation；
- pre/post-refiner randomized inclusion ledger；
- 不与 token representation change 同训。

---

## 5. 与一手近邻的 claim 碰撞

### 5.1 CrysVCD

[CrysVCD](https://arxiv.org/abs/2507.19799) 已经：

- 先生成 valence-balanced composition，再用 conditional diffusion 生成结构；
- 使用 valence-specific element/count token；
- 把 composition chemistry 作为前置约束；
- 进一步用 stability labels 做 conditional fine-tuning。

其方法部分还明确描述：ionic transformer 生成后附加 filter，只把
charge-balanced result 传给 diffusion。

因此 Planner 主线不能声称：

- first composition-before-structure；
- first valence-constrained crystal generator；
- charge balance 本身是主要新颖性；
- charge neutrality 自动带来 hull stability。

仍可能成立的差异是：

- 在现有 rich Plan 的**每个 formula token prefix**上做 grammar × charge
  reachability；
- 有限 atom budget 下保留“仍存在中性 suffix”的支持；
- terminal 输出可复算 witness；
- empty support/dead-end 留作 raw failure；
- 一次采样、无 output filter/retry/replacement/rerank；
- downstream Body/refiner 全冻结的 paired all-attempt 因果实验。

该差异足以让 Innovation 暂保 3，但必须在 2026-08-07 前完成 claim-by-claim
novelty kill gate；若发现等价 prefix DP/proof-carrying decoder，Planner 主线
立即 CUT。

### 5.2 PLaID++

[PLaID++](https://arxiv.org/abs/2509.07150) 已经：

- 使用 Wyckoff text representation；
- 做 iterative DPO；
- 联合 stability、novelty、space group 等 preference；
- 展示 representation 可缓解 RL mode collapse；
- 用 eqV2 建 preference、eSEN 做 evaluation，并以 1,000 个近 hull DFT
  样本审计 proxy。

碰撞结论：

- PILS-L 不能把“改 crystal text representation 提升 stability”写成宽泛创新；
- RL 不能把“多目标、稳定性、novelty 的 crystal post-training”写成创新；
- evaluator isolation 与 DFT audit 已是强近邻实践，不是本项目独有。

PILS-L 可争取的窄差异是 exact-length masked DLM 中 position-indexed
axis-token factorization，且每位置分辨率与 branching 不变。它必须用
calibration/refiner burden 机制链证明不是简单 vocabulary cleanup。

### 5.3 CRYSTAL

[CRYSTAL](https://openreview.net/pdf/94d95333b625bc19463eca098ff60038d639d590.pdf)
已经：

- 对 crystal autoregressive LM 做 group-relative multi-objective RL；
- 联合 stability、validity、novelty 与 diversity；
- 使用 multiplicative aggregation 缓解 reward hacking；
- 明确讨论 rollout group size、diversity 和单目标极化。

碰撞结论：

- “参考 CRYSTAL 设计 reward”不是方法新颖性；
- fixed-Plan Body 中 composition reward必须删除，而不是照搬；
- symmetric cluster uniqueness、meta guard 与 independent judge 是必要修正，
  但不足以在当前截止期单独支撑 RL 主线；
- Planner CR-Plan 不使用 energy reward，因此与 CRYSTAL 的直接算法碰撞最小。

### 5.4 Mask-Aware Policy Gradients

[Mask-Aware Policy Gradients](https://arxiv.org/abs/2607.15200) 已经：

- 把 masked-DLM generation写成 token 与 masking/reveal 的 two-stage action；
- 证明只优化 token likelihood 会漏掉 position decision；
- 用 Plackett–Luce probabilistic remasking；
- 在 `tau -> 0` 时恢复 greedy top-K；
- 把 position log-prob 纳入 trajectory policy gradient。

因此本项目 RL 不能把 joint token-position PL 本身称为贡献。可争取的只是：

- crystallographic legal special-token supports；
- D1 lattice→X→Y→Z group-local K=1 action；
- exact-length/Plan-match hard constraints；
- randomized pre/post-refiner multi-fidelity reward；
- all-attempt evaluator firewall 与 exact scientific resume。

这是组合差异，不足以抵消当前实现和时间风险。

---

## 6. 方案间 dependency 与 confounding

### 6.1 依赖图

```text
frozen P0 / B0 / D1 / model_494 / evaluator
  |
  +-- Planner CR-Plan
  |     changes formula distribution
  |     -> changes composition, chemsys coverage and downstream reward distribution
  |
  +-- Body PILS-L
  |     requires full-corpus Gate−1
  |     changes tokenizer + length representation + matched continuation
  |
  +-- Mask-aware RL
        also requires Gate−1
        requires frozen token representation/support
        requires E_train/E_gate/E_final and A/B label pipeline
```

### 6.2 五个禁止组合

1. **CR-Plan + PILS-L 首测：禁止。** 同时改变 formula distribution 与 lattice
   representation，comp/meta 的来源无法分解。
2. **PILS-L + RL：禁止。** RL behavior/support likelihood 在 tokenizer 与
   output rows改变后不再可比；这正是“representation change 与 RL 不同训”
   的硬边界。
3. **CR-Plan + RL：禁止。** Planner 改变 Plan/chemistry cohort，Body reward
   分布和 A/B calibration 也随之改变；没有 factorial 不能说二者可加。
4. **P-control + CR-Plan 合并：禁止。** P-control 是 post-selected recipe；
   同开会把 checkpoint 与 hard support混成两个 principal factors。
5. **safe-axis 与任一候选合并：禁止。** safe-axis 已有 strict-positive/
   meta-negative polarization，不能用另一因子事后“抵消”。

### 6.3 共同 evaluator 是三方向共享的外部依赖

最终候选必须与 H1-A2/CrysLLMGen 使用：

- 同一 raw all-attempt denominator；
- 同一 Direct parser/comp/structure contract；
- 同一 novelty database 与 matcher；
- 同一 MP hull snapshot 与 unknown policy；
- proposal-before-refine 和 after-refine 双表；
- 独立 scientific ledgers，不把 CUDA process repeats 当 seeds。

Planner 新 chemistry 最容易引入未缓存 chemsys；若只对 candidate 成功查询
或删除 unknown，会直接杀死论文。Body/RL 更容易通过 refiner/evaluator
优化 proxy；训练/选择与 final judge不隔离同样是 fatal。

---

## 7. 唯一可接受的倒排日历

| 绝对日期（CST） | Planner 主线 | Body 备线 | RL | 硬决定 |
|---|---|---|---|---|
| **2026-08-07** | CrysVCD/相邻 novelty audit、P0 anchor、oxidation table、claim 冻结 | full-train Gate−1、compute profiling、唯一 PILS-L 冻结 | 设计封存 | 之后不引入新架构 |
| **2026-08-10** | DFA/DP/certificate/telemetry R0 全过；common evaluator contract 冻结 | conversion、row remap、round-trip tests 全过 | 不执行 | 任一基础合同失败即对应路线 CUT |
| **2026-08-12** | paired-32 terminal | matched SFT terminal + paired-32 terminal | 不执行 | 不用真实 64 调工程 bug |
| **2026-08-14 target / 08-15 hard** | paired-64 mechanism gate | paired-64 mechanism gate | 不执行 | 按预注册优先级只选一个 256 主线 |
| **2026-08-21 target / 08-22 hard** | 若选中，paired-256 | 仅被提升时 paired-256 | 不执行 | 失败或逾期即回退 H1 |
| **2026-08-31** | 仅选中方向做 `4×256` + independent audit | 同左 | 不执行 | 未完成不得进摘要主结论 |
| **2026-09-05** | common evaluation、核心消融、science results freeze | 同左 | 不执行 | 不再启动 principal factor |
| **2026-09-12** | paper/supp/figures/anonymous repro only-fix | 同左 | 只可列 future work | 不因写作缺口补实验 |
| **2026-09-18 AOE** | abstract | abstract | — | 官方截止 |
| **2026-09-25 AOE** | full paper + supplement | full paper + supplement | — | 官方截止 |

### 7.1 绝对停止条件

#### Planner

- 08-07 新颖性无法与 CrysVCD/等价 prefix constraint 拉开：`CUT`；
- 08-10 product automaton/certificate 不能 100% fixture parity：`CUT`；
- 08-15 无 `charge -50%`、`comp +3/64` 或出现 shortcut/collapse：`CUT`；
- 08-22 paired-256 不完整，或 comp/joint 不正、meta/strict/structure/
  novelty/uniqueness 破非劣：`CUT`；
- 08-31 无独立确认：不得进摘要主结论。

#### Body

- 08-06 full-train coverage/legal mass不支持 length sharing：`CUT`；
- 08-07 projected 64 超过 64 A800 GPUh：`CUT`；
- 08-12 matched C0/L1 或 tokenizer exact-dynamic contract不闭合：`CUT`；
- 08-15 无 calibration/refiner-burden 与 nonnegative-meta 机制链：`CUT`；
- Planner 已过 64：备线冻结，不再申请 256；
- 若被提升后 08-22/08-31 逾期：`CUT`。

#### RL

- 本轮已经 `CUT`，不得用“只跑 R0/64”重新进入 portfolio；
- 任何 token representation/support 仍在变化时不得训练；
- 不得把历史 TraceRL 或同一 reward/final judge 包装为正式 RL。

### 7.2 预算总纪律

| 路线 | 08-15 前 cap | 完整路径 cap | 备注 |
|---|---:|---:|---|
| Planner CR-Plan | 12 A800 GPUh | **96 A800 GPUh** | 唯一默认可获得 post-64 预算的路线 |
| Body PILS-L | **64 A800 GPUh** | 仅被提升后 **128 A800 GPUh** | 训练 GPU 数必须计入 GPUh |
| Mask-aware RL | 0 | 0 | 投稿后另议 |

完整路径 cap 不能相加。08-15 后只有唯一主线可以继续烧 paired-256/
confirmation 预算。

---

## 8. 三大 paper-killing risks

### Risk 1：新颖性被近邻压扁

最危险的 reviewer 读法是：

```text
Planner = CrysVCD 的 valence constraint 换成另一种 decoder
Body    = 一个显然的 axis token tying trick
RL      = Mask-Aware PG + CRYSTAL reward 的领域移植
```

若 CR-Plan 不能用 prefix reachability、proof certificate、no-filter
all-attempt 与 rich-Plan interface 给出实质区别，整篇没有足够新方法。

### Risk 2：多个 principal factors 与 denominator 污染

把 P-control、CR-Plan、PILS-L、safe-axis、RL 或 refiner/evaluator 中任意两个
首次合并，会使 comp、meta、strict 的来源不可识别。使用 retry、filter、
replacement、survivor denominator、candidate-only MP coverage，哪怕数字更高，
也会直接破坏论文可信度。

### Risk 3：meta/evaluator 与独立确认赶不上截止

composition 的 64 信号可能很快，但 meta 需要 exact800、common hull snapshot、
unknown accounting 和真正独立 seed。若 08-22 只有 Direct，没有共同 S.U.N.；
或 08-31 只有同一 ledger 的 CUDA repeats，核心结果仍然不完整。不得用
“structure validity 很高”掩盖缺失的 meta 证据，因为 conditional structure
本来就已接近 ceiling。

---

## 9. 因果归因、fallback 与论文句子

### 9.1 H1 因果污染判定

- **Planner 单独执行：不污染。** P0、Body、D1、refiner、evaluator、seed role
  与 denominator 固定，唯一变化是 formula prefix legal support。
- **Body 单独执行：可控但需限定。** matched C0/L1 可归因给整个 PILS-L
  package，不能拆成 row averaging 或 continuation 的单独效果。
- **RL：当前不执行。** 若未来执行，no-RL PL 必须作为 control，token
  representation/support 必须先冻结。
- **任何跨方向组合：污染。** 当前时间不足以补完整 factorial，因此禁止。

### 9.2 08-15 无信号时的 fallback

论文退回已有 H1/R03 systems/diagnostic 主张：

1. rich Plan 与 exact-length masked Body-DLM 构成可审计的分层生成接口；
2. frozen Body/refiner 的 conditional structure validity 约
   `99.7–99.9%`；
3. Planner chemistry 是 composition/joint-valid 主瓶颈；
4. reveal schedule 是可复现因果接口，但 safe-axis 只造成 strict/meta
   polarization，不是全指标提升；
5. 新 Planner、token 与 RL 都作为诊断或 future work，不展示选择性 64
   小样本胜利。

### 9.3 主线若通过，可放进 9 页主文的一句

> We introduce a proof-carrying formula-prefix reachability decoder that
> constrains a rich crystal Planner to charge-neutral completions in a single
> all-attempt sampling pass, and isolate its effect through a frozen
> exact-length diffusion language model and continuous refiner.

### 9.4 若 Body 被正式提升，可替换为

> We factor axis-specific lattice-length tokens into a position-indexed shared
> vocabulary and show, under matched continuation and a frozen downstream
> pipeline, how improved length calibration changes refiner burden and
> meta-stable generation.

### 9.5 绝不能写的过度主张

> We are the first to guarantee chemically valid and structurally stable
> crystals, and our combined Planner, token compression, and RL method improves
> composition, structure, strict S.U.N., and meta S.U.N. simultaneously.

---

## 10. 最终审稿结论

### `KEEP_MAIN`：Planner CR-Plan

它是唯一直接命中当前最大、证据最清楚瓶颈的方向，也是唯一能在不训练
Body/refiner 的情况下形成完整因果链的候选。它的生死不取决于 GPU，而取决于
两件事：08-07 前能否证明不是 CrysVCD 的轻微变体，以及 08-15 前能否用
单次 all-attempt decoding 同时得到 charge failure 下降、comp-valid 上升且
无 shortcut。

### `KEEP_BACKUP`：Body PILS-L

它对 meta 的机制比 charge neutrality 更直接，但证据尚停留在 held-out token
identity，且必须重做 tokenizer、matched training 与 exact-dynamic sampler。
因此只能作为有 64 GPUh 前置 cap 的备线。Planner 主线一旦过 64，它不再扩到
256；Planner 失败时，它也只有在自己的 Gate−1 与 64 已独立通过后才能接替。

### `CUT`：Mask-aware RL

作为该方向作者，我认为最小算法是合理的，但当前提交窗口不合理：核心 trainer
需重写、reward/evaluator 需隔离、64 不是 efficacy、composition 又不属于
Body action。现在启动 RL 会同时威胁主线算力、科学冻结和 9 页叙事。最好的
决定是把完整设计保留到投稿后，而不是提交前做一个不可确认的小 RL 结果。

最终 portfolio 必须保持：

```text
one main  = Planner CR-Plan
one backup = Body PILS-L
cut        = Mask-aware RL
```

任何为了追求“comp、struct、strict、meta 全升”而在 08-15 后组合这三条路线的
行为，都应视为违反预注册并停止。
