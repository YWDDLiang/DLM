# ICLR 主线：MLIP-free WQ 实验重心与路线图

状态：`AUTHORITATIVE_LOCAL_PLAN`  
日期：2026-07-25  
适用 run：`20260720_0401-crysllmgen-wq-final-v3`

## 1. 本文档的权威性

本文档是 2026-07-25 起后续实验的权威入口。它明确取代尚未产生任何科学
attempt 的：

- `20260725_mattersim_guided_wq_iclr_execution_plan.md`；
- `mattersim_guidance_chgnet_eval_v1.json`。

被取代文件继续保留为决策历史，不删除、不覆盖。MatterSim 草案只完成了本地
contract 准备，没有提交远端作业、没有生成科学样本、没有训练模型，因此取消
该分支不会改变任何已有科学结果。

### 2026-07-26 P3/P4 修订

P1/P2 的历史结论继续由本文档管理；P3/P4 从 2026-07-26 起由下列修订案管理：

`docs/experiment_program/20260726_wyckoff_tangent_bridge_iclr_plan.md`

修订后的 P3 不再因 schedule parity PASS 自动进入 bridge 短训，而是先运行
training-free、MLIP-free 的 Wyckoff tangent projection gate。修订后的 P4
比较 raw WQ、unconstrained schedule-correct parent 与 every-step tangent
bridge。本文档中与其冲突的旧 P3/P4 阈值和执行顺序仅保留为历史。

## 2. 当前实验重心

现在只保留一条 ICLR 主线：

> 先让 symmetry-native Wyckoff proposal 落在化学可完成的 support 上，再让
> continuous diffusion 在正确的 forward-noise / condition 契约下恢复几何；
> 最后用冻结、独立的 CHGNet R5-C S.U.N. 评估两项改进是否同时成立。

三层职责必须严格分开：

1. **WQ LLM / constrained decoder：composition 与 orbit inventory。**
2. **CrysLLMGen parent / WQ bridge：coordinates 与 lattice。**
3. **CHGNet：模型冻结后的 held-out evaluator。**

本主线中：

- 不使用 MatterSim；
- 不使用任何 MLIP 做 training loss、teacher、distillation、label generation、
  guidance、reranking、schedule tuning、checkpoint selection 或失败处理；
- CHGNet 只做冻结后的正式评估；
- 不做 DFT；
- 不使用 retry、replacement、best-of-N 或输出筛选；
- 所有注册 attempts 都进入分母。

## 3. 为什么把重心放在这里

### 3.1 已经确认的 composition 问题

当前冻结 256-panel：

- composition-valid：`208/256 = 81.25%`；
- 历史 CrysLLMGen1000：约 `89.2%`；
- 48 个 composition-invalid 全部在 WQ proposal 阶段已经形成；
- parent CSP32 没有改变这 48 个 proposal 的元素计量；
- 36 个属于 `no_charge_neutral_assignment`；
- 12 个属于 `pauling_rejection`。

因此固定 composition 的 geometry diffusion 不可能修复这 48 个问题。第一优先级
必须是 WQ proposal support，而不是再训练一个“更稳定”的 diffusion。

同时，12 个 Pauling-only 多为金属间化合物/硅化物，其中部分在 CHGNet/MP
意义下稳定，不能作为硬 invalid 被删除或强制改写。历史 headline 继续报告冻结
legacy composition metric，新增 family-aware 诊断只能并列报告。

### 3.2 当前 S.U.N. 不能被简单视为失败

MP reference 补齐后的 sensitivity：

- raw meta S.U.N.：`118/256 = 46.09%`；
- 历史 CrysLLMGen1000：`461/1000 = 46.10%`；
- raw strict sensitivity：`18/256 = 7.03%`；
- 历史 strict：`90/1000 = 9.00%`。

这说明冻结 lower-bound 中的大部分 meta 缺口来自 hull reference coverage，不是
模型本身。strict 点估计偏低，但 256 attempts 尚不足以证明显著退化。

当前最明确、最可行动的缺口是 composition validity，以及少量极端
volume/density/minimum-distance 几何异常。

### 3.3 quotient refiner 失败不是路线总失败

既有 quotient refiner 的 768/768 生成失败，首先暴露的是训练—推理状态不一致：

- 训练对 geometry state 在 timestep `t` 做 forward-noise；
- 失败 sampler 却把 clean WQ proposal 直接当作高噪声端点 state；
- clean proposal 没有作为独立 condition；
- endpoint schedule 与 released parent 的离散 cosine schedule 不一致。

因此后续 bridge 必须先证明 schedule parity，不能直接再开一次长训练。

## 4. 本轮要回答的三个科学问题

### Q1 — Chemistry support

在不改变 space group、Wyckoff topology、orbit multiplicity、元素集合、lattice
和 coordinates 的前提下，是否能通过 deterministic orbit-species projection
恢复无中性解 proposal？

这一步只做 mechanism diagnosis。它若成功，下一步才把可行性约束移入 WQ
decoder，使最终方法成为一次 constrained sampling，而不是 post-hoc 筛选。

### Q2 — Schedule-correct geometry refinement

当 noisy state 与 clean proposal condition 正确分离后，WQ-conditioned parent /
bridge 能否恢复有限、正体积、可重建的晶体？

这一步不使用任何 MLIP，只看 schedule reconstruction、finite trajectory、
geometry support 和 raw generation success。

### Q3 — End-to-end transfer

chemistry-aware WQ + schedule-correct geometry refinement 是否在全新 attempts 上：

- 提升 composition/joint validity；
- 提升 chemistry-gated meta S.U.N.；
- 同时保持 raw meta、strict、novelty 和 uniqueness？

该问题只在方法完全冻结后，由 exact CHGNet R5-C A100 protocol on A800 回答。

## 5. 目标与成功标准

### 5.1 机制目标

Composition mechanism 64：

- 36 个 no-neutral；
- 12 个 Pauling-only；
- 16 个原 comp-valid matched controls。

PASS 条件：

- 至少 `24/36` no-neutral 在固定 topology/元素集合下恢复 legacy comp-valid；
- `16/16` controls byte-identical；
- `12/12` Pauling-only byte-identical；
- 每个 attempt 最多一个 deterministic 输出；
- 无可行解直接保留原 invalid，不换元素、不重采样；
- 所有 64 attempts 保留在分母；
- parent generation success 至少 `61/64`；
- raw meta 与 raw N+U 没有明显下降。

该 64-panel 是富集机制 panel，不能用其 S.U.N. rate 代表总体性能。

### 5.2 Bridge parity 目标

固定：

- start timestep `t ∈ {100, 200, 400, 800}`；
- 每 cell 8 个 paired attempts；
- correctly forward-noised state；
- clean proposal as separate condition；
- released parent schedule、decoder 和 time embedding。

PASS 条件：

- 所有 trajectory finite；
- invalid lattice at first reverse step = 0；
- schedule reconstruction audit PASS；
- raw generation success 不低于当前 verified parent handoff；
- failures 全部保留且无 retry/replacement。

### 5.3 Confirmatory 256 目标（原 chemistry 路线历史目标）

全新、不与开发 panel 重叠的 256 attempts：

- composition-valid `>=89%`；
- joint-valid `>=88%`；
- chemistry-gated meta S.U.N. 相对 `C-WQ-BASE` 至少 `+2 pp`；
- raw meta 同方向；
- strict 非劣；
- N+U 下降不超过 `2 pp`；
- graph/reconstruction acceptance `>=95%`。

256-panel 只用于 promotion。最终论文显著性仍由冻结冠军与必要基线的
3 training seeds × 1000 attempts 给出。

由于 P1 正式 FAIL，以上 chemistry 路线目标当前不再 active。Wyckoff tangent
bridge 固定 composition，无法独立达到或声称 `composition-valid >=89%`；
当前 active 256 gate 见
`20260726_wyckoff_tangent_bridge_iclr_plan.md` 第 9 节。

## 6. 实验阶段

### P0 — Identity 与 evaluator 冻结

要做：

- 当前 `step2544` 只称为 `epoch-3 endpoint`，不称 selected checkpoint；
- 冻结 historical CrysLLMGen1000 与当前 panel 的 MP database/cache 口径；
- 保留 raw / chemistry-gated / joint-valid 三层指标；
- MP reference 只在登录节点预取，不使用 Slurm；
- CHGNet 固定为 `diff_meets_diff` 环境、package 0.4.2、model 0.3.0、
  exact R5-C A100 protocol on A800；
- 每个 A800 job 必须满足 `CPU <= 8 × A800`。

### P1 — 固定-topology composition mechanism

立即实现的本地模块：

```text
StratifiedState
  -> exact original element set
  -> enumerate deterministic whole-orbit species reassignments
  -> preserve SG/Wyckoff/multiplicity/geometry/atom count
  -> accept first objective-optimal legacy-valid state
  -> otherwise return explicit no-solution/budget-exhausted result
```

预注册目标函数，按以下顺序最小化：

1. changed orbit count；
2. affected primitive atom count；
3. composition-count L1 distance；
4. canonical assignment hash。

边界：

- 只处理 `charge_neutrality_fail`；
- single-element、all-metal、Pauling-only、already-valid 均 identity；
- 只用原元素集合，且修复后每个原元素仍至少出现一次；
- 不改变 orbit 数、multiplicity、coordinates、lattice；
- search budget 固定并记录；
- 不调用 MLIP、LLM、parent diffusion 或外部 API。

P1 通过后，projector 只作为机制证据。正式方法把“仍存在 charge-neutral
completion”变成 WQ decoding support mask；不能把 CPU 搜索出的多个候选当作
best-of-N 生成结果。

### P2 — Chemistry-aware WQ sampler

P1 PASS 后才做：

- 在 partial WQ state 中追踪各 orbit multiplicity 对 composition 的贡献；
- ionic branch 仅允许仍存在净电荷 0 completion 的 species/orbit token；
- metallic/intermetallic branch 单独建模，Pauling 只作 soft flag；
- 每个 sampling trajectory 仍只产生一个 proposal；
- 无 rejection sampling、重试或候选池；
- 先做 proposal-only 3×256，检查 comp-valid、元素/family 分布和 uniqueness；
- 若 proposal support 改善且 diversity 未塌缩，再做最多一轮短
  formula-plan / chemistry-aware SFT。

### P3 — Schedule-correct bridge

schedule-correct A800 parity 已由 job28081 完成：32/32 cells、first-step invalid
lattice 0、non-finite 0、positive-volume 32/32、schedule error 0。

当前 P3 改为：

- strict-load released CrysLLMGen parent；
- WQ LLM 固定 composition、SG、Wyckoff orbit/species/multiplicity；
- parent atom/lattice update 每步拉回 orbit 与 lattice chart；
- 先做 training-free 的 `U/F/T` 32-cell mechanics gate；
- 不生成新 proposal、不训练、不调用 MLIP/API；
- Mechanics PASS 也不自动授权训练。

### P4 — Confirmatory 256

最小方法矩阵：

1. raw WQ expanded structure；
2. `U`：unconstrained schedule-correct parent；
3. `T`：every-step Wyckoff tangent bridge；
4. released CrysLLMGen exact-panel baseline（若另行冻结并运行）；
5. `F` final-only projection 只作诊断，不参与 headline 选择。

所有 arms 共用：

- attempt IDs 与 paired noise；
- parent/reverse schedule；
- model/call budget accounting；
- MP reference/cache；
- CHGNet evaluator；
- all-attempt denominator；
- no retry/replacement/best-of；
- no DFT。

### P5 — 论文确认

只有 P4 PASS 才运行：

- released CrysLLMGen；
- `C-ATOM-MATCHED`；
- `C-WQ-BASE`；
- 唯一冠军。

执行 3 training seeds × 1000 attempts，报告：

- comp / struct / joint validity；
- raw / chemistry-gated / joint-valid strict 和 meta S.U.N.；
- N+U；
- all-metal / single-element shortcut；
- element、family、atom-count marginals；
- failure taxonomy、计算预算与 hierarchical CI。

## 7. 统计与防泄漏

- 机制 64、proposal development、bridge development、confirmatory 256
  互不重叠；
- binary paired outcomes 使用 exact McNemar / paired exact test；
- paired difference 使用 10,000 次 attempt-level bootstrap；
- uniqueness 在每次 bootstrap 内重新计算；
- hull unknown、generation failure、relaxation failure 均贡献 0；
- coverage-adjusted 只报告，不能选方法；
- CHGNet 结果首次可见前，方法配置与 checkpoint 必须独占冻结；
- 查看 CHGNet 后不得修改配置并复用同一 panel。

## 8. 明确不做的事

- MatterSim 或其他 MLIP guidance；
- MLIP-derived loss、teacher、labels 或 reranking；
- 新的 stability diffusion 长训，直到 P1/P2/P3 均有机制证据；
- 按当前 48 个失败样本做 post-hoc headline；
- 删除 Pauling-only intermetallic 后声称 comp-valid 提升；
- 用 chemistry-filtered 分母掩盖 raw S.U.N. 下降；
- top-k 选优后仍按一个 attempt 计数；
- 按 NLL 或 CHGNet 事后选择 epoch；
- 额外 DFT。

## 9. 立即执行顺序（2026-07-25 历史）

本节记录 P1 启动时的执行顺序。2026-07-26 的 active 工作指针已改为
`20260726_wyckoff_tangent_bridge_execution_tasks.md` 中的
`WT200--WT208`。

现在按下列顺序工作：

1. 写入本权威路线图和独立任务清单；
2. 创建 MLIP-free experiment contract 和 fail-closed preflight；
3. 实现固定-topology composition projection；
4. 写 invariant、objective、budget、identity tests；
5. 在合成状态上完成本地测试；
6. 读取现有 48+16 panel 前先冻结 panel identity 与 input hashes；
7. 只运行 proposal-only 机制审计；
8. gate PASS 后才准备 parent 64；
9. 不在本阶段提交长训练。

当前首个代码目标是 P1 projector，不是 sampler 重训。这样可以用最低成本先验证：
“Wyckoff multiplicity 约束下是否真实存在足够 composition headroom”。若连固定
topology projection 都无法恢复至少 24/36，就停止这条 chemistry repair 路线，
不把时间投入到更复杂的 constrained decoder。

## 10. P1 终态（2026-07-25）

Mechanism-64 已在 A800 登录节点、`diff_meets_diff` 环境中以 CPU-only
方式完成唯一一次科学调用：

- 冻结 panel：36 no-neutral + 12 Pauling-only + 16 pre-outcome matched
  valid controls；
- no-neutral 恢复：22/36（61.1%），低于预注册 24/36；
- controls：16/16 byte-identical；
- Pauling-only：12/12 byte-identical；
- 无 classifier error、无 search-budget exhaustion；
- 无 MLIP、LLM、parent diffusion、外部 API、retry、replacement 或新生成；
- 4/36 taxonomy no-neutral 被同一冻结分类器复判为
  `oxidation_state_missing`；该差异只作机制诊断，不改变主分母；
- 其余 32 个 `charge_neutrality_fail` 中 22 个可投影、10 个在冻结约束内无解。

因此 P1 正式 FAIL，当前 chemistry projection/constrained-decode 路线停止。
不得事后放宽为 22/32、改变 24/36 阈值，或继续 parent 64/短训来“补救”该
门控。下一步只能先离线分析 10 个 no-solution 与 4 个 taxonomy/classifier
分歧，再提出与当前失败机制实质不同、重新预注册的新方案。

## 11. P1 离线机制分析补记（2026-07-25）

用户明确认为 `22/36` 足以作为探索性信号继续分析。该判断不改变第 10 节的
预注册 FAIL，只允许设计新的、独立预注册的后续 gate。

离线穷举确认：

- 10 个 no-solution 在移除 6-orbit cap 后仍全部无解；
- 5 个由 primitive atom total 与可中和整数比例不相容导致；
- 5 个虽存在同 atom-total 有效比例，但当前 Wyckoff multiplicity
  partition 无法实现；
- 4 个 taxonomy/classifier 分歧全部含 Pm，来自 default/ICSD24 氧化态表
  对 Pm 无覆盖；
- 22 个成功中 15 个只改 1 orbit、7 个改 2 orbits；
- 22/22 通过 default 与 ICSD16 氧化态表，21/22 通过所检查四表交集。

因此不再考虑“扩大 projector 搜索”作为解决办法。新的最小候选机制是
MLIP-free topology-feasibility mask：在 proposal/decode 前同时检查
primitive atom-total feasibility 与 Wyckoff orbit-partition feasibility。

训练前先执行独立的 `existing-22 projection survival audit`：不生成新样本，
先测投影后 CrysLLMGen composition/structural validity；只有新预注册通过后
才进入 CHGNet relaxation/hull 与 exact S.U.N。详细分析见：

`docs/experiment_program/20260725_wq_composition_mechanism64_offline_analysis.md`

该 survival audit 已在结果可见前冻结为
`wq_existing22_projection_survival_v1`：

- 输入是现有 22 个唯一投影，exact all-22 denominator；
- 22/22 必须成功展开且 composition-valid；
- structural-valid 与 joint-valid 均至少 20/22；
- login-node CPU-only，不使用 Slurm、GPU、MLIP、MP API 或 S.U.N；
- 不重跑 projector、不重选候选、不生成、不训练；
- FAIL 立即停止；PASS 也只进入一个新的、独立预注册的 CHGNet/S.U.N
  gate。

冻结 contract SHA256：
`4743a2129597b0f68483aa0feb9e2dd286089da3926bbb9ab2b119bf783e540b`。

## 12. Schedule parity 终态与 active 路线（2026-07-26）

原 diagnostic job28054 在科学计算前因 result schema binding 失败；该失败不
代表 diffusion 失败。唯一 supersession job28081 已完成：

- `COMPLETED 0:0`；
- `t={100,200,400,800} × 8 = 32` cells；
- 32/32 terminal、finite、positive-volume；
- first-step invalid lattice 0；
- non-finite trajectory 0；
- strict parent load PASS；
- schedule max absolute error 0；
- 无 retry、replacement、新生成、MLIP 或 API。

终态审计 SHA256：

```text
4b77a10d632c33b53b2208d49db19f542081a7fd91d52f038ebe7e5280e2cf41
```

因此 schedule mismatch 已被纠正，但这只是 correctness gate，不是效果提升。
active 路线变为：复用现有 quotient charts/Jacobians，实现 released parent 的
training-free every-step Wyckoff tangent projection；先做本地数学/invariant
测试，再由独立授权决定是否执行一个 32-cell A800 mechanics gate。
