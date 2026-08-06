# ICLR 2027 路线图独立红队审阅 V1

状态：`independent_decision_only_no_execution_authorization`

日期：2026-08-04

审阅对象：三份 proposal、三份 cross-review、证据/截止框架，以及
`ICLR2027_TIMEBOXED_FULL_IMPROVEMENT_ROADMAP_V1.md`。本审阅不替任何方案
辩护，也不授权训练、生成、refinement、Materials Project 查询或 checkpoint
promotion。

审阅时主路线图版本：

```text
SHA256 6151a4adf92a552b1aaae6c591e6723a907626d815b017d73916c59ec8e37ae2
```

---

## 0. 可直接引用的 executive decision

> **独立红队裁决：本轮唯一 MAIN 是以冻结 P0 为因果锚点的
> P0 vs P0+CR-Plan；不等待、也不以 post-selected P-control 替换锚点。
> 唯一 BACKUP 是 PILS-L，但仅允许在 2026-08-08 novelty gate 或最迟
> 08-10 engineering gate 早停 CR-Plan 后一次性接替；CR-Plan active
> 期间不得并行 matched SFT、64 或 256。Mask-aware RL、双 RL 模型、
> chemistry compiler 与 reachability-mass SFT 均为本次 ICLR CUT，RL
> 预算严格为 0 A800 GPUh。CR-Plan 只有在证明其相对 CrysVCD 的增量不只是
> terminal valence filter 加通用 constrained decoding，并在 08-15/08-22/
> 08-31 分别通过 64、256、独立确认门后，才可进入标题和摘要；否则回退到
> frozen H1 系统论文，不晚切候选、不叠加变量。**

结论是“**值得尝试，但只值得做这一条有条件主线**”，不是已经认可其新颖性
或预先承诺 meta/struct_valid 提升。

---

## 1. 唯一 MAIN / BACKUP / CUT

| 角色 | 唯一裁决 | 允许做什么 | 明确禁止 |
|---|---|---|---|
| `MAIN` | frozen `P0 -> P0+CR-Plan` | novelty audit、R0/32、64、256、通过后 4×256 confirmation | P-control 换锚点；叠加 Pauling/compiler/PILS/RL |
| `BACKUP` | `PILS-L Gate-1 cold backup` | full-split coverage、legal-mass、identity/readiness 审计 | CR active 时 matched SFT、32/64/256 |
| `CUT` | mask-aware RL（含两个模型）、compiler、reachability-mass SFT | 仅保留设计/future-work 文档 | 本轮训练、生成、reward labeling、endpoint |
| `FALLBACK` | frozen H1/PlanGraph-DLM | 系统论文、诊断、common evaluation | 08-15 后另起模型线追结果 |

这不是“三条并行路线”。即使存在独立 owner 或额外 GPU，也不能把 PILS-L
升级为并行竞赛者；共享的 evaluator、论文注意力和多重选择风险仍未独立。

### 1.1 P0 与 P-control

裁决：**P0 立即固定为 CR-Plan primary causal anchor。**

理由：

- P0 是当前冻结、可复现的系统锚点；
- P-control 的 `456/512` 相对 P0 的 `+4.30 pp` 是 post-selected discovery，
  且同时改变多个 recipe 因素，不能反向决定 CR-Plan 的 treatment/control；
- 等待 P-control 新 1,024 会消耗最稀缺的日期和注意力，并产生“先选强锚点再
  选方法”的第二层选择自由度；
- 若 P-control 最终更强，审稿人确实会质疑 P0 是弱 baseline；正确修复是如实
  报告 P-control 为 supporting baseline，而不是让它改写因果主比较。

执行边界：

1. 现在只做旧 512 的只读 shortcut/drift/identity audit；
2. 新 1,024 Plan-only confirmation 不得阻塞 CR-Plan，且不计入本轮必需交付；
3. 在 CR-Plan paired-256 与资源预留未闭合前，建议从队列删除新 1,024；
4. 无论其结果如何，primary comparison 始终是 `P0 vs P0+CR-Plan`。

### 1.2 PILS-L

裁决：**仅 Gate-1 cold backup，不允许 08-15 前与 Planner 并行 matched
SFT/64。**

只有以下两种早停可触发一次性切换：

- 08-08：CR-Plan 相对近邻无法形成可防守 novelty；
- 最迟 08-10：DFA/DP/witness/parity 工程合同无法闭合。

切换还要求 PILS-L Gate-1 全部通过。切换后 CR-Plan 停止，PILS-L 成为唯一
active candidate。若 CR-Plan 到 08-15 的 64 才失败，则不晚切，直接回退 H1。

### 1.3 RL 与两个模型

裁决：**本轮 RL = 0 A800 GPUh，投稿路径 CUT。**

“不经过 refine 的模型”和“经过 refine 的模型”在科学上只有当它们对应两个
真实部署产品时才值得分开训练。若最终部署固定经过 model_494 exact800，
pre/post-refiner 只是 label fidelity 或 curriculum，不是两个产品；训练两个
checkpoint 会翻倍标签、验证和选择自由度。更合理的投稿后方案是一个 policy，
使用预注册的随机多保真标签，最后按 post-refiner 目标确认。

本轮不允许 RL R0、first-64、LoRA、reward labeling 或 judge stack 占用关键
路径。形式化设计可以留在 future work，但不能成为论文实验贡献。

---

## 2. Paper-killing novelty attack

### 2.1 CrysVCD 对 CR-Plan

[CrysVCD](https://arxiv.org/html/2507.19799v1) 已经使用带价态的
`(element^v, count)` composition 表示生成 charge-balanced composition，再
条件化结构 diffusion；其推理还明确使用 charge-balance filter。

因此以下宽泛主张均已死亡：

- composition 先于 structure；
- valence/charge constraint；
- charge-balanced proposal；
- “价态序列本身提供化学证据”。

CR-Plan 唯一可能存活的增量是：在通用 formula-token prefix 上进行
finite-budget suffix reachability、记录可复算 sidecar witness，并在
single-pass all-attempt 协议下不靠事后 survivor filtering。即便如此，
[PICARD](https://aclanthology.org/2021.emnlp-main.779/) 和
[Grammar-Constrained Decoding](https://aclanthology.org/2023.emnlp-main.674/)
也已使“prefix masking”本身不具新颖性。

**致命消融：**

```text
grammar only
vs terminal-only fail-closed / STOP gate
vs full prefix reachability
```

必须报告换行前被移除的 probability mass、blocked prefixes、dead-end、
all-attempt completion/yield 与 diversity。若 full reachability 仅在 terminal
阻止换行，或不能优于 terminal-only gate，则不得称其为核心方法或
“proof-carrying planning”创新。

此外，CrysVCD 明确支持 mixed valence；当前“一元素一个 oxidation state”的
实现若不支持混合价，可能比近邻更不完整。必须先给出 train/val/test oracle
coverage 与 mixed-valence false-exclusion audit。

### 2.2 PLaID++ 对 PILS-L/系统主张

[PLaID++](https://arxiv.org/html/2509.07150v4) 已覆盖 Wyckoff/text
representation、迭代 preference alignment，以及 stability/novelty/symmetry
方向的后训练。PILS-L 若只是把 LA/LB/LC 三组 embedding 绑成 L，本质上是
普通 parameter sharing；没有 legal-mass 与 calibration→refiner-basin→meta
机制链，就不足以作为表示创新。

同时，PLaID++ 的部分报告协议会对不可解析 CIF 重采样。它与本项目
all-attempt denominator 不同；未经 common evaluator 重算，不得声称 SOTA，
但 denominator 更严格只是评测贡献，不会自动使方法新颖。

### 2.3 CRYSTAL 对 RL

[CRYSTAL](https://openreview.net/pdf/94d95333b625bc19463eca098ff60038d639d590.pdf)
已将 crystal LM 的 validity、stability、novelty、uniqueness 组织为协调的
多目标 RL，并直接报告 S.U.N.。因此“参考 CRYSTAL 设计 reward”、乘法聚合、
或 strict/stability 奖励均没有方法新颖性。

双模型也不是新贡献，只是两个训练目标。除非能证明 mask-aware exact policy
likelihood、多保真无偏估计和 independent-judge 防 reward hacking 三者均成立，
否则这是高成本复现近邻，不是投稿主线。

### 2.4 Mask-Aware Policy Gradients 对本项目 RL

[Mask-Aware Policy Gradients for DLMs](https://arxiv.org/html/2607.15200v1)
已经把 token 与 reveal/remasking decision 纳入策略，并使用
Plackett–Luce 类位置机制。因而本项目不能主张 joint token-position action、
PL position policy 或 mask-aware PG 本身。

“用于晶体”“按 group reveal”“加入 pre/post-refiner reward”最多是应用与目标
差异；当前提案中的 two-channel surrogate 还未证明等于完整轨迹 likelihood。
在没有本地 replay/resume、support parity 和 held-out 效果前，RL novelty
评分最多 2–3/5，deadline fit 1/5，支持本轮 CUT。

---

## 3. 64 / 256 / confirmation 门禁修正

| Stage | 红队裁决 |
|---|---|
| R0/32 | candidate 的非预注册 shortcut `terminal charge_fail` 必须为 **0**；这是实现正确性，不是“降低 50%”的效果指标。增加 frozen evaluator/table SHA、mixed-valence coverage、DP time/memory cap、brute-force parity。 |
| 64 | `comp +3/64` 可作为机制 screen；不要把稀疏 strict/meta 点估计符号当效果证明。`all-metal +2`、`unary +1` 与“shortcut ≤ gain 的 10%”在 gain=3 时数学冲突；改为 **candidate shortcut-valid count 不高于 control，且 shortcut 不计入 primary gain**。 |
| 64 distribution | “无明显 collapse”不可复现。64 只作预注册分布诊断；hard coverage/diversity gate 放到 256。 |
| 256 | `comp +8/256`、`joint +5/256` 是效应量 screen，不是显著性证明。candidate 非 shortcut charge failure 仍应为 0；control 为 0 时不得报告 reduction ratio。 |
| 256 meta | 单个 256 同时要求 point `>=0` 且 bootstrap lower `>-2 pp` 很可能低功效，并不符合 CR-Plan 不直接作用 meta 的机制。256 用冻结 point safety 决定是否继续；正式 `-2 pp` noninferiority 应在 pooled 4×256 confirmation 上裁决。 |
| 256 blocks | 两个 128 block 都正，在约 3 pp 效应下每块只差约 4 个样本，太噪；保留为诊断，不作单独硬停门。 |
| pairing | 约束一旦改变 token path，同 ordinal 不再是 token-level common random number。称“paired ordinal ledger”，不要过度宣称 CRN variance reduction。 |
| confirmation | 4 个真正独立 256 panels 是正确方向。将“每 panel 无 shortcut”改为“无 shortcut inflation/new failure class”，否则语义上会误要求输出中不存在任何 unary/all-metal。hierarchical paired interval 才是 primary，3/4 sign 为稳定性佐证。 |

其他必须补齐：

- witness 只能称“相对冻结 oxidation table 的可行性证书”，不能称真实氧化态；
- 明确 unary/all-metal 是被阻止、单独接受还是无 witness fail closed；当前
  “每个 terminal 必须 witness”与“shortcut 不视为合格 witness”存在语义缺口；
- sidecar 不可被 Body 消费；若消费，即成为第二个 principal factor；
- 加入 terminal-only baseline、oracle coverage、prefix affected-rate 和
  latency/DP-state 审计；
- `struct_valid` 已近天花板，只能主张 noninferiority；raw structure/joint
  上升可由 completion 带来，不能写成 conditional structure 改进；
- meta 不是 CR-Plan 的直接靶点。若只通过非劣门，只能声称 comp/joint 改善；
  meta 提升必须有独立正证据。

---

## 4. 绝对日期与预算

### 4.1 不可移动日期

| 日期 | 唯一含义 |
|---|---|
| 08-07 | P0、claim、evaluator/spec 冻结 |
| 08-08 | CR-Plan novelty 决定；只有此处可因 novelty 切 PILS-L |
| 08-10 | R0/工程与 token Gate-1；最晚一次性切换日 |
| 08-12 | paired-32 闭合 |
| 08-15 | 唯一候选 paired-64；失败即 H1 fallback |
| 08-22 | paired-256 硬截止，无 08-24 grace |
| 08-31 | 4×256 confirmation/common table/core ablation 硬截止，无 09-05 grace |
| 09-05 | science/table freeze，不是 confirmation 宽限 |
| 09-12 | only-fix |

日历只有在 08-22 前预留至少两路并发 refiner、冻结 common snapshot/API cache、
并明确 evidence owner 时才现实。若资源不能保证，不得压缩 denominator 或把
一个 panel 重复运行冒充四个；应降低论文 claim。

### 4.2 唯一预算

```text
CR-Plan target cap: 96 A800 GPUh
CR-Plan hard cap:   136 A800 GPUh
PILS-L training while CR active: 0 GPUh
RL: 0 A800 GPUh
```

主路线图仍有一个必须删除的预算矛盾：表中 96 GPUh 是 target cap，随后又写
“如果选择更严格的 96 GPUh program cap”，并建议只保至少一个 independent
panel，同时又说不能牺牲 4-panel 定义。96 不能既是目标预算，又是可选硬 cap；
“至少一个 panel”也不能同时算 4-panel confirmation。

红队建议固定：

- 96 为 planning target，136 为绝对 hard cap；
- 进入 confirmation 前必须已经预留完整 4-panel 资源；
- 资源不足则结果降为 preliminary，不把 1 panel 写成 confirmed；
- P-control 新 1,024 未列预算，paired-256 前从必需队列删除；
- PILS-L Gate-1 为 CPU/read-only；其“最多 12 GPUh”只能是诊断上限，不能用于
  SFT 或 endpoint；
- independent MLIP/DFT audit 若是 promotion hard gate，必须另列 owner、
  wall-clock、API/compute budget；否则只能称 robustness audit。

---

## 5. 主路线图剩余内部问题

主路线图已经正确修掉 P-control 换 anchor、PILS 并跑、RL R0、compiler
execution、08-24/09-05 宽限等冲突；仍需在采用前修正：

1. **预算自相矛盾：** 96 target 与“更严格 96 cap”并存，1-panel 与
   4-panel confirmation 并存。
2. **P-control A1 未预算：** optional 1,024 截止 08-12 仍会与唯一主线争抢
   owner/queue，应降为 main confirmation 后的 optional support。
3. **64 shortcut 门数学不一致：** 允许 `+2/+1` 与贡献 `<=10%` 无法在
   `+3/64` gain 下同时满足。
4. **化学语义未闭合：** mixed valence、covalent/all-metal/unary 与 witness
   的处理不清；可能把真实可行 formula 错删，且比 CrysVCD 表达能力弱。
5. **novelty 表述仍偏强：** “proof-carrying”与推荐标题应在
   terminal-only 消融和 prefix affected-rate 通过后启用，不应仅凭设计使用。
6. **统计门与机制错位：** 单 256 的 meta CI 与双 128 正向硬门低功效；把正式
   noninferiority 移至 pooled confirmation。
7. **配对措辞过度：** 同 ordinal 不是路径分叉后的 token-level CRN。
8. **独立评测未资源化：** independent evaluator/common baseline/core
   ablation 全塞进 08-31，却没有清晰成本、owner 和失败降级规则。
9. **目标预期需收窄：** CR-Plan 合理目标是 comp/joint；它不直接解决
   conditional struct ceiling，也没有机制保证 meta 提升。若项目目标要求三项
   同升，应承认本路线只能先解决其中一项并守住另外两项。

---

## 6. 最可能导致 ICLR 拒稿的五条意见与逐条修复

### R1. “CR-Plan 是 CrysVCD valence filter 加标准 constrained decoding。”

修复：完成逐项 claim matrix；加入 grammar-only、terminal-only fail-closed、
full prefix reachability 三臂消融；报告换行前真实 removed mass、blocked
prefix、all-attempt yield 与 diversity。若无前缀级增益，删掉方法标题和
proof-carrying 主张。

### R2. “所谓证书只是有限 oxidation table heuristic，还会错杀 mixed-valence、
covalent 或 alloy chemistry。”

修复：冻结 table/version/语义；在 train/val/test 报 oracle coverage、
mixed-valence false exclusions、Pauling/missing/shortcut taxonomy；将 witness
准确命名为 table-relative feasibility certificate；给 independent chemistry
failure audit，不把 neutrality 等同 stability。

### R3. “作者选择了较弱 P0，回避更强 P-control 和强外部 baseline；文献数字
还使用不同 denominator。”

修复：P0 保持预冻结因果锚点，但完整披露 P-control 的 post-selection、旧
512 和可用的新 confirmation；reported 与 common-evaluator 表分开；明确
PLaID++ resampling 与本项目 all-attempt 差异；没有 common baseline 就不写
SOTA。

### R4. “comp_valid 的提高没有证明产生更稳定的晶体；结果主要由 frozen
refiner 修复，struct_valid 已饱和。”

修复：把 raw comp/joint 定为 primary；strict/meta/structure/diversity 定为
预注册 noninferiority；按 attempt 报 before/after-refiner 和 failure
ownership。只有 independent positive meta 才声称 stability 改善，绝不宣称
conditional struct 提升。

### R5. “多路线 pilot、阈值与 panel 选择造成 winner's curse，确认统计功效
不足。”

修复：执行唯一 P0+CR MAIN、PILS cold only、RL/compiler/SFT CUT；冻结绝对
日期和 estimand；保留失败 ledger；单 256 只作 scientific screen，最终依赖
4×256 hierarchical paired analysis；不以双 128 block、单一 seed 或多个
endpoint 中偶然为正者改写 primary。

---

## 7. 最终红队结论

**可以尝试 CR-Plan，但批准的是一个可被 08-08 novelty attack 杀死的实验
假设，不是一个已经成立的论文贡献。** 当前证据支持优先解决 Planner
composition bottleneck；不支持为了同时改善 meta/struct_valid 而并跑
PILS-L 或 RL。最稳健的 deadline 决策是固定 P0、只跑 CR-Plan、把 PILS-L
保持冷备、RL 归零，并在任何一项 novelty、correctness、diversity、meta
noninferiority 或独立确认失败时诚实回退 H1。

本审阅没有启动或授权任何实验。
