# Roadmap Release Narrow Re-audit V1

状态：`release_ready_pending_external_manifest`  
审计日期：2026-08-04  
审计类型：旧问题的窄范围关闭复核，不授权实验、不修改 roadmap。

## 1. 绑定快照

| 对象 | SHA-256 |
|---|---|
| `ICLR2027_TIMEBOXED_FULL_IMPROVEMENT_ROADMAP_V1.md` | `47e0b9fe1ce1a891edf0df9744e8e8d68cd9fa6225f2339bc49e99608cbf2bc7` |
| 前序 `ROADMAP_RELEASE_CONSISTENCY_AUDIT_V1.md` | `8e368f5f046423e56f1f84c82e2703a00691461b2441b14a4eb97b3d38a4c1f8` |

本复核只检查前序审计的 B1–B3、M1–M5、三个 minor，以及
P0/CR-Plan/PILS-L/compiler/RL、日期和预算的一致性。前序 B4 本质上是发布
打包要求，不在正文内自引用解决；因此状态明确保留
`pending_external_manifest`。

## 2. 结论

**PASS：当前绑定快照没有剩余 release blocker。**

- 旧 B1–B3：全部关闭；
- 旧 M1–M5：实质问题全部关闭；
- 旧 minor：全部关闭；
- 主线、备线、CUT、日期和预算：一致；
- 发布前唯一剩余动作：生成外部 manifest，绑定最终 roadmap、本复核和输入
  评审的 SHA。

## 3. BLOCKER 关闭复核

### B1 — PASS：novelty gate 已可机械裁决

位置：677–711 行。

已冻结：

- 新且与 discovery 独立的 `512 raw attempts/arm` Plan-only ledger；
- original P0、grammar-only、terminal-only、full-prefix 四臂；
- raw yield、prefix affected-rate、terminal failure、parse/completion、
  diversity/coverage、shortcut 等数值门；
- DP-state、median/p95 latency、OOM/timeout/silent-fallback 工程门；
- 08-10 terminal report 截止；
- ordinal 结果打开后不得修改阈值。

旧 B1 的样本、estimand、阈值、资源上限和日期缺口均已关闭。

### B2 — PASS：PILS-L 接棒不再是无合同自动切换

位置：492–495、946–962、1387–1390 行。

早期 CR-Plan stop 现在只授予 `backup eligibility`。任何 conversion/SFT 写操作
前必须有独立、release-ready 且新授权的 `PILS_L_EXECUTION_ANNEX`；annex 必须
覆盖 matched SFT、32/64/256、`4×256`、common evaluator、owner 和预算，并在
08-10 前闭合。它继承 08-15/22/31/09-05 日期，CR-Plan 与 PILS-L 合计不得超过
`136 A800 GPUh`，否则直接回退 H1。

旧 B2 的执行合同与预算歧义已关闭。

### B3 — PASS：common 与 extra evaluator 语义已统一

位置：758–787、799–827、1467–1472 行。

- historical common Direct/S.U.N. 缺失：`HOLD`，硬日期后不确认；
- extra evaluator 缺失：不反向取消 common-evaluator comp/joint 因果确认，
  但禁止跨-judge broad stability 和宽泛 SOTA；
- extra evaluator 同向：按证据等级开放 stability claim；
- extra evaluator 反向：保留并披露 common comp/joint，关闭
  all-metric/broad-stability/SOTA；
- extra robustness audit 最多 16 GPUh，包含在 confirmation hard cap。

三处语义一致，旧 B3 已关闭。

## 4. MAJOR 关闭复核

| 旧项 | 状态 | 复核结果 |
|---|---|---|
| M1 统一评分 | `PASS_WITH_NOTE` | PILS-L 已改为交叉评审多数口径 `2/2/2/3/3/3`；RL Innovation 已由 4 改为三评审一致的 3；“算法新颖性最高”过度 claim 已删除 |
| M2 P-control 执行义务 | `PASS` | 263–264 行已限定为只有进入 confirmed supporting table 才需新 1,024，并明确不是当前投稿必需义务；569 行后仍保持新授权/不占主线资源 |
| M3 08-08/08-10 | `PASS` | 129–130 行已拆为 08-08 文献/claim novelty 与 08-10 工程/四臂机制门；细则一致 |
| M4 64 evaluator incomplete | `PASS` | 719–722 行冻结唯一状态 `HOLD_EVALUATOR_INCOMPLETE`；08-15 前未补齐即未通过并回退 H1 |
| M5 preliminary claim | `PASS` | 516–528 行 claim ladder 与 133 行一致：未完成 `4×256` 不得占用 CR-Plan 方法题目、摘要结果或 confirmed/SOTA |

M1 非阻断说明：当前 RL Expected effect 为 2，而三份交叉评审均给 3。这是向下
收缩、不是过度 claim；它不改变 `CUT FOR ICLR`、`0 GPUh`、路线排序或实验权限，
且正文 469–472 行已说明根因证据与 deadline 可达性弱。因此记录为保守综合
偏差，不恢复为 MAJOR，也不要求为该评分单独改动冻结正文。

## 5. MINOR 关闭复核

| 旧项 | 状态 | 证据 |
|---|---|---|
| malformed bullet | `PASS` | 977 行现为 `- 真实 behavior trace；` |
| proof-carrying 条件 | `PASS` | 1237–1265、1662–1664 行明确只有 novelty/terminal-only/prefix 增量门通过后才解锁该简称 |
| Gate 命名 | `PASS` | 全文统一为 `Gate −1`，未发现 `Gate-1`、`Gate−1` 或 `Gate -1` 残留 |

## 6. 主线、备线、CUT、日期与预算

### 路线角色 — PASS

| 项目 | 冻结状态 | 一致性 |
|---|---|---|
| P0 | primary causal anchor | 通过 |
| CR-Plan | only scientific main line，conditional | 通过 |
| PILS-L | Gate −1 cold backup；早停只给 eligibility，需 annex + 新授权 | 通过 |
| chemistry compiler | `CUT FOR ICLR`，只作 post-ICLR future work | 通过 |
| mask-aware RL | `CUT FOR ICLR / POST-ICLR`；本投稿不执行 | 通过 |
| 两个 RL 模型 | `CUT` | 通过 |

正文 45–88、475–495、848–869、946–985、1377–1398、1641–1660 行之间未发现
角色漂移或并行竞赛授权。

### 日期 — PASS

| 日期 | 冻结事件 |
|---|---|
| 08-08 | CR-Plan 文献/claim novelty |
| 08-10 | 工程、四臂 Plan-only、token Gate −1；PILS annex 最晚闭合 |
| 08-15 | 唯一 active candidate paired-64 |
| 08-22 | paired-256 与方法/evaluator 冻结 |
| 08-31 | `4×256` 独立确认；未完成自动 preliminary |
| 09-05 | science/table freeze |

顶层日历、方法门、备线 annex 和执行章节一致。

### 预算 — PASS

- CR-Plan target `96 GPUh`、absolute hard cap `136 GPUh`；
- paired-64 / cumulative paired-256 / confirmation hard caps 分别为
  `12 / 72 / 64 GPUh`，累计口径明确；
- extra evaluator 的最多 16 GPUh 包含在 confirmation hard cap；
- P-control 新 1,024 为 `0 GPUh` 且不在必需队列；
- CR-Plan active 时 PILS-L matched SFT 为 `0/0`；
- PILS-L 只有获授权 annex 后才可在总 `136 GPUh` 内接棒；
- RL 本投稿为 `0/0`；
- 未使用 Gate −1 diagnostic 预算不得结转给 PILS-L/RL。

未发现重复 hard cap、预算外自动接棒或 CUT 路线占用投稿 GPU 的授权。

## 7. Markdown 与发布检查

- 当前 roadmap SHA 已再次核对为
  `47e0b9fe1ce1a891edf0df9744e8e8d68cd9fa6225f2339bc49e99608cbf2bc7`；
- 共 52 个 fenced-code delimiter，数量成对；
- 未发现缺空格的 Markdown bullet；
- 1515、1519、1523、1526、1530 行末双空格是有意的 Markdown hard break；
- Gate 命名已统一；
- 表格和标题层级在本次窄范围检查中未发现断裂。

## 8. 发布动作

当前正文可进入发布打包，但还需外部 manifest 同时绑定：

1. roadmap SHA；
2. 三份 cross-review、root frame、red-team 的输入 SHA；
3. 前序 release audit SHA；
4. 本 narrow re-audit SHA。

manifest 完成前状态保持 `release_ready_pending_external_manifest`。本复核本身不
是实验授权；执行仍须遵守 roadmap 内部 gate、owner、资源预留与新授权要求。
