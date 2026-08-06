# ICLR MLIP-free WQ 执行任务

权威路线：
`docs/experiment_program/20260725_iclr_mlip_free_wq_experiment_focus_and_roadmap.md`

## 0. 路线冻结

- [x] `F000` 取消 MatterSim 分支，确认没有科学 attempt 或远端 submission。
- [x] `F001` 冻结训练与采样全过程 MLIP-free。
- [x] `F002` 冻结 CHGNet 只作 held-out evaluator。
- [x] `F003` 冻结每 A800 最多 8 CPU。
- [x] `F004` 写入实验重心、问题、阶段、目标和停止规则。

## 1. 本地 contract

- [x] `C100` 创建 `wq_iclr_mlip_free_v1.json`。
- [x] `C101` 检查配置中不出现 guidance/reranking/MLIP training。
- [x] `C102` 检查 all-attempt denominator、no retry/replacement/best-of。
- [x] `C103` 检查 CHGNet 仅出现在 held-out evaluation。
- [x] `C104` 检查 `CPU<=8×A800`。
- [x] `C105` 运行本地 preflight 与 mutation tests。

## 2. P1 固定-topology composition mechanism

- [x] `M200` 实现 `FixedTopologyCompositionProjector`。
- [x] `M201` 保留 SG、lattice、orbit IDs、Wyckoff type、multiplicity、
  primitive multiplicity、chart dimension 与 coordinates。
- [x] `M202` 保留原元素集合，每种原元素至少占一个完整 orbit。
- [x] `M203` 只允许 whole-orbit species reassignment。
- [x] `M204` 只处理 `charge_neutrality_fail`。
- [x] `M205` already-valid、Pauling-only、all-metal、single-element 必须 identity。
- [x] `M206` 固定 objective：changed orbits → affected atoms → count L1 → hash。
- [x] `M207` 固定 search budget，区分 no-solution 与 budget-exhausted。
- [x] `M208` 记录 evaluated assignments、before/after formula、changed orbit IDs。
- [x] `M209` 测试 deterministic、permutation invariance 与 input immutability。
- [x] `M210` 测试无重试、无外部 API、无 MLIP/LLM/parent 调用。

## 3. Mechanism-64

- [x] `P300` 冻结 36 no-neutral IDs。
- [x] `P301` 冻结 12 Pauling-only IDs。
- [x] `P302` 仅按处理前特征冻结 16 valid controls。
- [x] `P303` 写 panel JSON 与 input SHA：
  panel SHA `3da00d574fcf23416d4464b937480d2609f50a04a98d4c81f719c73c938d63f2`。
- [x] `P304` 运行唯一一次 proposal-only projector。
- [x] `P305` 核验 >=24/36 恢复：**FAIL，22/36**。
- [x] `P306` 核验 16/16 controls byte-identical：PASS。
- [x] `P307` 核验 12/12 Pauling byte-identical：PASS。
- [x] `P308` 生成不可覆盖 mechanism audit：
  `runs/remote_audit/20260725_wq_composition_mechanism64_v1_terminal.json`。
- [x] `P309` Gate 未通过，因此未准备、也不得运行同 parent/noise 的
  64 generation。

终态说明：10/32 个被 projector 复判为 `charge_neutrality_fail` 的状态在固定
元素集合、固定 topology 和 6-orbit budget 下无解；另有 4/36 个 taxonomy
no-neutral 状态被同一冻结 SMACT 分类器复判为 `oxidation_state_missing`。
主分母仍固定为 36，不因结果出现后改成 32。

## 4. P2 chemistry-aware sampler

**状态：STOPPED。** P1 未达到 24/36；不得沿当前 composition
projection/constrained-decode 路线继续 sampler 或训练。

- [ ] `S400` 从 projector 结果提取可完成性状态，不使用输出后筛选。
- [ ] `S401` 实现 ionic charge-completion support mask。
- [ ] `S402` 实现金属/intermetallic family 分支。
- [ ] `S403` Pauling 只作 soft flag。
- [ ] `S404` 保证每 trajectory 单一 proposal。
- [ ] `S405` 3×256 proposal-only diversity/validity audit。
- [ ] `S406` Gate PASS 后决定是否做一轮短 formula-plan SFT。

## 5. P3 schedule-correct bridge（独立探索线；正式 all-22 FAIL 不改写）

- [x] `B500` clean proposal 与 noisy state 分离。
- [x] `B501-local` 独立复现 released parent forward schedule。
- [x] `B501-A800` 与 strict-loaded parent schedule 精确 parity，max error 0。
- [x] `B502` strict-load parent decoder/time embedding/checkpoint。
- [x] `B503` 冻结 `t={100,200,400,800}` × 8 attempts。
- [x] `B504` 32-cell schedule reconstruction PASS；first-step invalid lattice
  `0/32`。
- [x] `B505` finite trajectory `32/32`、positive-volume `32/32`、失败分母和
  no-retry accounting PASS。
- [x] `B506-local` 冻结 MLIP-free validation 与 checkpoint/early-stop 禁泄漏合同。
- [x] `B507` job28081 Gate PASS；明确 **不自动授权短训**，转入新的
  Wyckoff tangent projection gate。

## 6. P4 confirmatory 256（原 chemistry matrix，已由 P3/P4 修订案取代）

- [x] `E600-old` P1 FAIL 后停止原 chemistry confirmatory matrix。
- [x] `E601-old` 不运行 `C-WQ-CHEM-HANDOFF/BRIDGE`。
- [x] `E602-old` 不事后把固定-composition geometry arm 要求为 comp>=89%。

active P4 任务见第 9 节与：

`docs/experiment_program/20260726_wyckoff_tangent_bridge_execution_tasks.md`

## 7. 停止规则

- [x] P1 恢复少于 24/36：已停止 chemistry projection/decode 路线
  （22/36，2026-07-25）。
- [ ] controls 或 Pauling-only 被改变：FAIL，不扩大规则。
- [ ] proposal diversity 明显塌缩：停止训练。
- [x] schedule parity PASS；继续 training-free tangent gate，不自动训练。
- [ ] confirmatory 256 FAIL：不跑 3×1000。
- [ ] 任一 job `CPU>8×A800`：claim 前 fail-closed。
- [ ] 任一输出 identity 已存在：禁止覆盖或重复提交。

## 8. P1 离线分析与探索性后续

- [x] `A700` 保留预注册 22/36 FAIL，不事后改变 24/36 阈值。
- [x] `A701` 记录用户将 22/36 接受为探索性机制信号。
- [x] `A702` 对 10 个 no-solution 穷举全部 element-preserving
  whole-orbit assignments。
- [x] `A703` 确认 10/10 均非 search budget 或 6-orbit cap 导致。
- [x] `A704` 将根因拆成 5 个 atom-total congruence 和 5 个
  Wyckoff-partition incompatibility。
- [x] `A705` 确认 4 个 taxonomy/classifier 分歧全部来自 Pm
  oxidation-state database coverage。
- [x] `A706` 完成 default/ICSD16/SMACT14/Wiki 氧化态表敏感性分析。
- [x] `A707` 写入详细离线分析和机器可读记录。
- [x] `A708` 新预注册 existing-22 projection survival audit：all-22，
  render/comp 22/22，struct/joint 至少 20/22；contract SHA256
  `4743a2129597b0f68483aa0feb9e2dd286089da3926bbb9ab2b119bf783e540b`。
- [x] `A709` 在 all-22 denominator 上完成唯一 CrysLLMGen survival audit：
  rendered 22/22、composition-valid 22/22、structural/joint-valid 17/22；
  低于冻结的 20/22，正式结果 **FAIL**。
- [x] `A710` 正确应用停止规则：A709 未 PASS，因此没有冻结或运行
  CHGNet relaxation/hull 与 exact S.U.N gate。
- [x] `A711` 正确停止当前 escalation：没有实现或训练新 sampler。
  若继续，必须先另行预注册 MLIP-free geometry feasibility +
  atom-total/orbit-partition gate。
- [x] `A712` 写入详细结果：
  `docs/experiment_program/20260725_wq_existing22_projection_survival_outcome.md`。
- [x] `A713` 保留原 20/22 gate 的正式 FAIL；单独记录用户接受 17/22
  作为探索性继续门，不事后改写历史。
- [x] `A714` 冻结独立 all-22 CHGNet R5-C S.U.N. 合同：17 个
  joint-valid 输入 evaluator，5 个 structural-invalid 作为固定 failed
  placeholder，仍保留 all-22 denominator。
- [x] `A715` 冻结方向阈值：对齐同口径 CrysLLMGen 1000-sample 的
  strict 9.0% 与 meta 46.1%，离散为 strict≥2/22、meta≥11/22。
- [x] `A716` 构建、验证并通过用户一次性授权传输唯一评测归档。
- [x] `A717` 严格一次提交 1×A800/8CPU exact CHGNet R5-C S.U.N.
  evaluation-only gate。
- [x] `A718` 终态核验 PASS/FAIL/INCONCLUSIVE_MP_COVERAGE，写入不可覆盖
  审计；任何结论均不得自动训练。

`A718` 终态：job27631 的八个 MP hull unknown 经另行授权在 A800
登录节点严格各查询一次；7 条解析成功，`BaYb3O11` 保留一条结构化
query error。固定 all-22 结果为 strict `2/22`、meta `6/22`、unknown
`1/22`。即使把最后 unknown 算作成功，meta 上限仍只有 `7/22`，低于
冻结阈值 `11/22`，因此结论为确定的 **FAIL**，停止当前
composition-projection escalation。终态审计：
`runs/remote_audit/20260725_wq_existing22_mp_completion_v1/terminal_audit.json`。

## 9. P3/P4 修订：Wyckoff-tangent bridge

权威计划：

`docs/experiment_program/20260726_wyckoff_tangent_bridge_iclr_plan.md`

完整任务：

`docs/experiment_program/20260726_wyckoff_tangent_bridge_execution_tasks.md`

- [x] `WT100--WT108` job28081 schedule-correct reference 完整 PASS。
- [ ] `WT200--WT208` 本地 atom/lattice step projector。
- [ ] `WT300--WT305` manifold-restricted paired noise。
- [ ] `WT400--WT408` 本地数学、周期边界、invariant 与 mutation tests。
- [ ] `WT500--WT509` WTB-32 contract/runner safety。
- [ ] `WT600--WT607` 独立授权后的唯一 A800 mechanics gate。
- [ ] `WT700--WT708` 仅在 WTB-32 PASS 后冻结 confirmatory 256。
- [ ] `WT800--WT803` 根据 256 证据决定 stop / training-free / 小 adapter。

当前唯一 active work item 是 `WT200--WT208`；没有远端提交授权，也没有训练
授权。
