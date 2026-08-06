# WTB-256、训练决策与论文级复现执行计划

状态：`ACCEPTED_PLAN_LOCAL_PREPARATION_PASS_REMOTE_EXECUTION_NOT_AUTHORIZED`  
日期：2026-07-26  
适用 run：`20260720_0401-crysllmgen-wq-final-v3`

本文件是 2026-07-26 mechanics 回归闭环后的当前权威执行计划。它不改写
job28185 的失败，也不把 development-only job28187 解释成科学效果证据。
旧计划继续作为历史设计与根因记录；若旧文件中的“当前工作指针”与本文件冲突，
以后者为准。

## 1. 当前结论与实验重心

已经确定的事实是：

1. job28185 暴露了有限步长 lattice chart 更新的真实数值缺陷；
2. `global_chart_retraction_v1` 在同一 development panel 上通过 job28187：
   F/T 各 `32/32`、全部 30 个 mechanics Gate 为真；
3. job28187 只证明实现能够稳定执行并保持固定 Wyckoff topology，不证明 T
   优于 U，也不授权训练；
4. 当前最重要的科学问题不是“能否运行”，而是：

   > 对新的、完全不重叠的 WQ proposals，every-step Wyckoff chart
   > retraction 是否能在不牺牲生成成功率、composition 和 novelty 的前提下，
   > 提升结构有效性与 held-out S.U.N.？

因此下一步是一次新的 `WTB-256` confirmatory evaluation，而不是先训练。

## 2. 证据分级

| Level | 必需证据 | 允许的结论 |
|---|---|---|
| L0 | job28187 mechanics PASS | 方法实现与数值修正闭环 |
| L1 | 新 WTB-256 的 MLIP-free direct metrics 完整 | 判断 geometry/topology 因果方向 |
| L2 | WTB-256 CHGNet R5-C S.U.N. 与 paired statistics PASS | 决定 stop / training-free / small adapter |
| L3 | 预注册多 seed × 1000、matched baselines、消融与完整审计 | 论文主结果达到投稿级 |

方法、动机、负结果、数值根因与 protocol 章节现在即可写。headline 效果表和最终
claim 必须等 L2；正式论文结果表必须等 L3。

## 3. WTB-256 冻结 panel

### 3.1 新 proposal panel

- WQ LLM：formal epoch-03 / step2544 adapter；
- training seed：`11`；
- sampling seed：`101`；
- ordinals：`512..767`，共 256；
- 与 development 来源（现有 ordinal `256..511` 中的 8 个 hash-selected
  sources）零重叠；
- attempt ID、pair ID、proposal text、proposal state 与 topology hash 全部写入
  append-only ledger；
- 每 ordinal 只允许一次 proposal；parse/expand 失败仍保留在 256 分母；
- 不 retry、replacement、best-of、rerank 或 outcome-dependent filtering。

不能用旧 ordinal `256..511` 作为本次 confirmatory 256：其中已经包含直接参与
WTB-32 开发和根因修复的来源。

### 3.2 主 arms

对每个已注册 source 构造三个 paired arms：

1. `R` / `raw_WQ`：WQ proposal 无损展开的 primitive structure；
2. `U` / `unconstrained_schedule_correct`：released CrysLLMGen parent，
   正确 forward-noised geometry，随后无约束 reverse；
3. `T` / `every_step_chart_retraction`：相同 parent schedule、相同 clean source
   和 paired base noise；forward noise 限制到 WQ manifold，并在每个
   corrector/predictor substep 后执行 `global_chart_retraction_v1`。

`F` 仅是 development mechanics diagnostic，不进入 headline confirmatory
选择。released CrysLLMGen 若没有同 proposal 的 paired 输入，只能作为历史
reference，不能标记成 exact-panel paired baseline。

### 3.3 固定 schedule 与 compute

- parent checkpoint：
  released CrysLLMGen MP20 `model_494`；
- reverse start：`t=800`；
- reverse grid：32 个固定 respaced steps；
- decoder calls：U/T 每个 terminal attempt 各 64；
- U/T forward 与 reverse base seeds 由同一 pair identity 派生；
- parent beta、alpha-bar、coordinate sigma、time embedding、decoder 和
  corrector step size均不改；
- `t=800` 来自 parent 官方 sampling contract，不从 WTB-32 四个 timestep 的
  outcome 中选择。

## 4. 评测顺序

### 4.1 Stage A：身份、mechanics 与 MLIP-free metrics

在读取新 CHGNet 输出之前，完成并冻结：

- 256 source attempts 与三 arm ledger；
- generation/render success；
- composition validity；
- structural validity；
- joint validity；
- exact space-group retention；
- exact Wyckoff multiset/topology retention；
- atom count、species 与 multiplicity preservation；
- minimum distance、volume、density 与 collision taxonomy；
- uniqueness、novelty；
- U/T paired outcome table；
- 所有失败在 all-attempt denominator 中记 0。

任何实现 hash、parent identity、schedule、proposal policy 或 resource contract
不一致都在科学 attempt 前 fail closed。

### 4.2 Stage B：冻结方法后的 CHGNet S.U.N.

只有 Stage A 已完整物化并且 method identity 不再变化，才运行：

- runtime：
  `/public/home/jiaosz/miniconda3/envs/diff_meets_diff`；
- CHGNet package `0.4.2`、model semantics `0.3.0`；
- exact `R5-C A100 protocol on A800`；
- strict：`E_hull <= 0.0 eV/atom`，主指标；
- meta：`E_hull <= 0.1 eV/atom`，次指标；
- denominator：每 arm 全部 256 registered attempts；
- failures、relaxation unknown、hull unknown 均在主分母计 0；
- coverage-adjusted 只报告；
- 不做 DFT；
- CHGNet 不用于训练、guidance、reranking、retry、checkpoint selection 或
  failure replacement。

MP hull cache若缺失，只允许在 A800 登录节点以既定 MP API 流程预取并冻结，
不得在 Slurm 内在线查询；query cache 的 identity/hash 必须进入终态审计。

### 4.3 Stage C：paired statistics

- binary outcomes：paired exact / exact McNemar；
- continuous/direct deltas：pair-level bootstrap 10,000；
- direct 与 S.U.N. 的 attempt-level binary labels 使用 paired bootstrap 10,000；
- novel-and-unique 的 promotion Gate 使用冻结 evaluator 的 paired point
  estimate；除非后续显式保留 equivalence classes 并在每个 replicate 内
  重算 uniqueness，否则不伪造该指标的 bootstrap 区间；
- 同时报告 point estimate、paired delta、95% interval 和 discordant counts；
- 对可合法 bootstrap 的指标同时报告 point estimate、paired delta、95%
  interval 和 discordant counts；不用 nominal p-value 替代 effect size 与区间。

## 5. 预注册 promotion Gate

以下阈值在 ordinal 512 的首个输出出现前冻结：

### 5.1 不可替代的 integrity Gate

- R/U/T registered attempts 都是 256；
- T composition 与其 R source 逐 attempt byte-identical；
- T exact topology retention `256/256`；
- U/T 使用相同 source、pair ID 和 frozen base-noise identity；
- T generation/render success 不低于 U；
- attempt retry/replacement/best-of 为 `false`；
- 三 arm all-attempt denominator 均为 256；
- 不读取 final test、DFT 或其他 MLIP 结果做选择。

任一 integrity Gate 失败都使本次 identity 终态 FAIL；不得静默修补或重跑。

### 5.2 科学 promotion Gate

- T joint-valid 相对 R 至少 `+3.0 percentage points`；
- T strict S.U.N. 点估计至少 `9.0%`；
- T meta S.U.N. 点估计至少 `46.1%`；
- paired `T-U` joint-valid delta 不得为负；
- paired `T-U` strict S.U.N. delta 不得为负；
- T 的 novel-and-unique rate 相对 U 下降不超过 `2.0 pp`。

composition-valid 不是 geometry-only bridge 的独立可优化终点：R/U/T 固定同一
composition。它必须报告并保持一致，但不能把 geometry projection 描述成
composition repair。

## 6. WTB-256 后的训练决策

### Branch A：无 paired gain

若 T 相对 U 的 joint-valid 与 strict S.U.N. 都无正向 paired signal：

- 终止 tangent bridge；
- 不训练 adapter；
- 将其作为 mechanics-correct but scientifically negative 的结果保留；
- 论文主线回到 WQ proposal chemistry 与 released-parent baseline。

### Branch B：有 gain，但 residual 明显

若方向为正但未过 promotion Gate，且 failure taxonomy 指向可学习的局部
projection residual：

- 另建新身份设计小 projection adapter；
- parent backbone、WQ LLM 与 topology 均冻结；
- checkpoint selection 完全 MLIP-free；
- 先做极短 smoke，再决定最多一次短训练；
- CHGNet/S.U.N. 不回流训练；
- 训练必须另有 contract、panel、hash 和用户授权。

### Branch C：training-free promotion PASS

若完整 promotion Gate PASS：

- 保留 training-free 方案作为首选；
- 不为了更高数字自动训练；
- 直接进入 L3 多 seed × 1000、matched baselines 与消融复现。

## 7. L3 论文级复现

training-free 路线最低配置：

- sampling seeds 至少 3 个；
- 每 seed 每 arm 1000 registered attempts；
- R/U/T 完全 matched；
- released CrysLLMGen 与 raw WQ historical/reference 语义明确；
- ablations 至少含 final-only F、无 lattice retraction、无 atom tangent
  projection；
- paired exact tests 与 bootstrap 10,000；
- 全失败 taxonomy、walltime、GPU-hours、peak memory 与 call budget；
- 三次独立运行的 immutable terminal audits。

若引入 trainable adapter，则额外要求至少 3 个 training seeds，且不能用
CHGNet/S.U.N. 选 checkpoint。

## 8. 资源、连接与不可覆盖边界

- 每 `1×A800` 最多 `8 CPU`；
- 当前 WTB-256 计划只使用 `1×A800 / 8 CPU`；
- 本机仅通过 `tmux wq-starteam:1.0`；
- A800 仅通过 starteam 用户维护的 `tmux ssha800:1.0`；
- 每次 A800 动作前必须确认 outer/nested：
  `pane_dead=0` 且 `pane_current_command=ssh`；
- nested 断开后停止，绝不由代理重连或本机直连 A800；
- starteam→A800 SCP 固定端口 `7001`，任何一次传输使用新冻结归档身份；
- 所有复合远端命令放入隔离 `bash -lc`；
- submission claim、record、outputs、terminal audit 均 exclusive-create；
- 不 overwrite、cancel、retry、replace 或拆分已经提交的科学 identity。

## 9. 当前执行边界

当前已经完成并通过本地 Gate 的是：

1. 写入本权威计划与任务清单；
2. 本地冻结 WTB-256 contract；
3. 实现/复用 panel、R/U/T、direct metrics、S.U.N. 与统计 runner；
4. 建立一次性 submission 与资源 fail-closed 检查；
5. 运行本地 tests、模拟安装与静态审计；
6. 冻结 exact source hashes、patch manifest 与唯一归档。

本文件作为归档内权威科学计划，不自指向最终 tar/manifest SHA；最终 exact
archive identity 由归档外的 local-preparation audit 与 transfer manifest 记录。
任何远端传输、安装与提交都必须取得绑定这些最终 SHA 的后续明确授权，不能使用
草稿 hash，也不能把本次本地准备授权扩张成 Slurm 授权。

## 10. 2026-07-27 identity amendment

job28195 已证明本文件 5.1 节中把 primitive atomic-number **有序序列**用于
composition byte identity 会错误拦截规范化 orbit 重排。该 job 和原 Gate
结果保持不可变；所有未来 identity 改用 exact element-count multiset、atom
count、canonical proposal payload 与 exact species-Wyckoff topology 作为硬门，
legacy ordered signature 只作诊断。

完整根因、修正边界、development mechanics Gate 与新 held-out 256 的执行顺序见：

`docs/experiment_program/20260727_wtb256_permutation_safe_identity_and_next_execution_plan.md`
