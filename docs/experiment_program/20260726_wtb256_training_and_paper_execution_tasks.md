# WTB-256、训练决策与论文级复现任务清单

权威计划：
`docs/experiment_program/20260726_wtb256_training_and_paper_execution_plan.md`

状态：`LOCAL_PREPARATION_PASS_REMOTE_EXECUTION_NOT_AUTHORIZED`  
日期：2026-07-26

## 0. 证据边界

- [x] `WC000` job28185 保持 immutable scientific FAIL。
- [x] `WC001` job28187 development mechanics PASS，F/T 各 32/32。
- [x] `WC002` job28187 标记 `confirmatory_evidence=false`。
- [x] `WC003` 禁止把 WTB-32 mechanics PASS 当成训练授权。
- [x] `WC004` 禁止复用 ordinal 256..511 作为新的 held-out 256。

## 1. 权威计划

- [x] `WC100` 写入当前 accepted execution plan。
- [x] `WC101` 固定 L0/L1/L2/L3 证据等级。
- [x] `WC102` 固定 stop / small-adapter / training-free 三分支。
- [x] `WC103` 固定论文级最低复现要求。
- [x] `WC104` 固定 `CPU<=8×A800` 与两跳连接边界。

## 2. WTB-256 scientific contract

- [x] `WC200` 新 identity：
  `wq_wyckoff_chart_retraction_confirmatory256_v1`。
- [x] `WC201` 固定 formal epoch-03 WQ adapter 与全部 asset hashes。
- [x] `WC202` 固定 training seed11、sampling seed101、ordinals512..767。
- [x] `WC203` 静态证明与 development 8-source panel 零重叠。
- [x] `WC204` 固定 R/U/T arms 和 all-attempt denominator256。
- [x] `WC205` 固定 official start timestep800、32-step schedule 与64 calls。
- [x] `WC206` 固定 paired forward/reverse base-noise derivation。
- [x] `WC207` 固定 integrity 与 scientific promotion Gate。
- [x] `WC208` 固定 CHGNet exact R5-C evaluation contract。
- [x] `WC209` 固定 immutable output、claim、record 与 terminal audit paths。

## 3. Runner 与 evidence

runner 已实现并通过 `256×R/U/T` 合成证据闭环；以下复选框只在真实
held-out scientific artifacts 到达后勾选，不能以实现或 fixture 代替实验。

- [ ] `WC300` 生成 256 个 append-only WQ source attempts。
- [ ] `WC301` 物化 raw WQ expanded structures。
- [ ] `WC302` 对同 source/paired noise 执行 U。
- [ ] `WC303` 对同 source/paired noise 执行 T。
- [ ] `WC304` 记录每 attempt topology/composition/call/mechanics evidence。
- [ ] `WC305` failures 留在分母且禁止 retry/replacement/best-of。
- [ ] `WC306` 输出 evaluator-compatible R/U/T JSONL。
- [ ] `WC307` 运行三 arm CrysLLMGen direct metrics。
- [ ] `WC308` 运行 exact topology 与 collision taxonomy。
- [ ] `WC309` 方法冻结后运行三 arm CHGNet strict/meta S.U.N.。
- [ ] `WC310` 对 direct/S.U.N. binary labels 生成 paired exact tests 与
  bootstrap10,000；novel-and-unique 仅用冻结 paired point estimate，除非
  equivalence classes 可在 replicate 内诚实重算。
- [ ] `WC311` 生成不可覆盖 terminal acceptance/audit。

## 4. Submission safety

- [x] `WC400` Slurm 固定 `gpu / 1×A800 / 8CPU`。
- [x] `WC401` claim 前 fail closed 检查 `CPU<=8×A800`。
- [x] `WC402` claim 前检查 record/claim/output 均不存在。
- [x] `WC403` exact archive/install/authorization hashes 进入 job。
- [x] `WC404` 只允许一次 `submit_once.sh`。
- [x] `WC405` unrelated queue 只记录，不取消、不修改。
- [x] `WC406` Bash 4.2、offline 与 exact environment preflight。

## 5. Local verification

- [x] `WC500` contract/schema/static safety tests。
- [x] `WC501` deterministic panel/seed/non-overlap tests。
- [x] `WC502` R/U/T pairing 与 denominator tests。
- [x] `WC503` failure accounting 与 exclusive-write tests。
- [x] `WC504` metric/paired-statistics unit tests。
- [x] `WC505` compile、Ruff、shell syntax。
- [x] `WC506` archive manifest 与模拟原子安装。
- [x] `WC507` base+patch installed-byte Gate。
- [ ] `WC508` 生成 local preparation audit 与所有 exact SHA256。

## 6. Remote WTB-256

- [ ] `WC600` 冻结唯一 archive、target path 与 transfer manifest。
- [ ] `WC601` 按冻结连接策略传输与核验 exact bytes/SHA。
- [ ] `WC602` A800 唯一 staging 解包与原子安装。
- [ ] `WC603` 远端 targeted tests、environment、asset strict-load Gate。
- [ ] `WC604` 单次创建 claim 并提交唯一科学 pipeline。
- [ ] `WC605` 只读监控 stage、stderr、GPU CSV 与 artifacts。
- [ ] `WC606` 终态核验 Slurm、256 denominator、三 arm、metrics 与 hashes。
- [ ] `WC607` 生成不可覆盖 remote terminal audit。

## 7. 决策

- [ ] `WC700` integrity Gate 判定。
- [ ] `WC701` scientific promotion Gate 判定。
- [ ] `WC702` 无 paired gain：stop/no train。
- [ ] `WC703` 正 gain 但 residual 大：另建 small-adapter gate。
- [ ] `WC704` training-free PASS：进入多 seed ×1000。
- [ ] `WC705` 决策结果写入 immutable promotion lock。

## 8. 论文级复现

- [ ] `WC800` 3 sampling seeds ×1000 的新 contract。
- [ ] `WC801` matched R/U/T 与 reference baselines。
- [ ] `WC802` F / no-lattice / no-atom-tangent ablations。
- [ ] `WC803` paired statistics、failure taxonomy 与 compute table。
- [ ] `WC804` 三次 independent terminal audits。
- [ ] `WC805` 冻结 headline tables、figures 与 claim boundary。

## 9. 当前工作指针

```text
WC508 -> WC600 -> request exact archive-bound remote authorization
```

本文件作为归档内容不自指向归档 SHA。最终 local audit 与 transfer manifest
在归档外记录 `WC508/WC600`；当前不传输、不安装、不提交远端 job。
