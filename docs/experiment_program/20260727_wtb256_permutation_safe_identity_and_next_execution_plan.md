# WTB-256 顺序无关身份修正与后续执行计划

状态：`IMPLEMENTED_LOCAL_GATE_IN_PROGRESS`  
日期：2026-07-27  
适用 run：`20260720_0401-crysllmgen-wq-final-v3`

## 1. 当前实验结论

job28195 的 Slurm 终态和科学判定保持不变：

- Slurm：`FAILED 2:0`；
- 决策：`invalid_integrity_stop_no_retry`；
- source 生成本身：`256/256` PASS；
- R/U/T 的 CrysLLMGen 与 S.U.N. 报告已经物化，但由于完整性 Gate
  失败，不能作为本 identity 的正式比较结果；
- 不允许覆盖、删除、重命名、重跑或事后把 job28195 改判为 PASS。

终态审计：

- 路径：
  `runs/remote_audit/20260727_wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1/terminal_failure_audit_job28195.json`
- SHA256：
  `124bb6e02d612687cd25a21b57b57e64773eff5836c788b8f5998754f1da76c9`

## 2. 根因

256 个 source 中，216 个在 R/U/T 三个 arm 中同时被判为：

`source_revalidation: ... successful source expansion identity changed`

该失败不是元素数目、Wyckoff topology 或原子数发生变化，而是以下执行顺序造成的
表示差异：

1. source runner 先展开内存中的 `StratifiedState`；
2. 随后调用默认 `canonical_storage=True` 的 `to_dict()`，按 storage key
   重新排列 orbit；
3. arm runner 从规范化 payload 重建 state 并再次展开；
4. v1 的所谓 composition signature 实际哈希 primitive atomic-number
   **有序序列**；
5. 两次展开的元素计数、atom count 和 topology 相同，但 orbit/atom 顺序不同，
   因此被错误当成化学身份变化。

这使一个 representation permutation 被错误升级为科学完整性失败。用户明确认为
该比较过严，要求去除或降为非拦截项。

## 3. 修正后的身份契约

### 3.1 继续作为硬门的项目

- 规范化 proposal payload 完全一致；
- `species × Wyckoff orbit × multiplicity` 的 topology hash 完全一致；
- 元素原子序数到计数的 multiset 完全一致；
- primitive atom count 完全一致；
- T arm 的终态 topology 完全一致；
- U/T decoder call budget 与 T chart-retraction call budget 完全一致；
- 所有轨迹审计值 finite；
- 无 retry、replacement、best-of 或 rerank。

### 3.2 降为诊断项的项目

- primitive atomic-number 的原始有序序列；
- legacy ordered signature 是否相等。

legacy 顺序不一致仍记录：

- stored ordered signature；
- canonical ordered signature；
- match/mismatch count。

但它不再阻止执行或 promotion。真正的元素计数、atom count、proposal 或 topology
变化仍然 fail-closed。

### 3.3 原子对应关系

放松 ordered signature 不意味着任意重排 tensor。唯一允许的 parent 输入是：

`persisted proposal → canonical_storage payload → canonical state → one canonical expansion`

R/U/T 都从这一份 canonical expansion 派生。因此每次 parent 调用内部的 atom
correspondence 唯一且稳定；旧顺序仅用于解释历史 mismatch。

## 4. 当前立即执行的 development mechanics Gate

新 identity：

`wq_wyckoff_identity_mechanics_sup28195_v1`

它只用于验证修正是否真实消除表示型失败：

1. 对 job28195 已冻结的全部 256 个 source 做 permutation-safe round-trip
   identity audit；
2. 复用其中前 32 个 source/noise cells（ordinals `512..543`）运行 R/U/T；
3. 不生成新 WQ proposal；
4. 不运行 CrysLLMGen direct metrics；
5. 不运行 CHGNet 或 S.U.N.；
6. 不训练；
7. 不把该 development panel 当成 confirmatory evidence。

资源固定为普通 `gpu`、`1×A800 / 8 CPU / 64 GiB / 1 h`。

PASS 必须同时满足：

- 256/256 permutation-safe source identity PASS；
- legacy ordered mismatch 仅报告、不拦截；
- R/U/T 各 32/32 generation success；
- 三 arm 终态 composition multiset 全部与 source 相同；
- T topology 32/32 完全保持；
- T 每 attempt 64 decoder calls 与 64 global chart retractions；
- 全部 chart audit finite；
- 无 retry/replacement。

任意其他失败都使新 mechanics identity FAIL，且不得重投。

## 5. mechanics PASS 后的科学步骤

mechanics PASS 本身不恢复 job28195 的科学有效性，也不证明 T 优于 R/U。下一步必须
使用全新且未暴露给此次修正的 panel：

- ordinals：建议 `768..1023`；
- 256 个 source；
- R/U/T paired；
- 同一 source、pair ID、forward noise 和 reverse base noise；
- composition identity 使用 element-count multiset；
- legacy order 只报告；
- 先完整物化 MLIP-free mechanics/direct metrics；
- integrity PASS 后才运行 exact R5-C A100-on-A800 CHGNet S.U.N.；
- 仍然不 retry、replacement、best-of、rerank 或结果过滤。

为了减少再次浪费长评测时间，新 confirmatory 建议拆成依赖 DAG：

1. source generation + R/U/T mechanics；
2. afterok 的不可覆盖 identity/mechanics audit；
3. afterok 的 CrysLLMGen direct metrics；
4. afterok 的 S.U.N.；
5. afterok 的 paired summary 和 promotion lock。

这样真实 identity/mechanics 失败会在 CHGNet 之前终止，不再消耗完整 S.U.N.
评测时间。

## 6. 科学决策

全新 256 的 all-attempt 指标才用于选择：

- 若 T 相对 U 的 joint validity 与 strict S.U.N. 都没有正向 paired signal：
  停止 tangent bridge，主线回到 WQ LLM chemistry；
- 若有稳定正向 signal 但未过原 promotion 阈值：
  只讨论新的、独立授权的小型 projection adapter，不自动训练；
- 若 training-free T 过 integrity 与 promotion Gate：
  进入多 seed × 1000 的 L3 论文复现，不为追数字自动引入训练。

job28195 已物化的成功子集率只能用于 debug 方向判断，不能替换 all-attempt
confirmatory 指标。

## 7. 不可覆盖边界

- job28195 的 claim、record、logs、outputs 与 terminal audit 不变；
- 当前 development Gate 使用新 claim、record、output 和 patch identity；
- 每 `1×A800` 最多 `8 CPU`；
- 本机仅经 `tmux wq-starteam:1.0`；
- A800 仅经用户维护的 `tmux ssha800:1.0`；
- nested 断开后代理不重连；
- 不直接从本机连接 A800；
- 不修改无关队列任务；
- mechanics Gate 不自动提交后续 256。

