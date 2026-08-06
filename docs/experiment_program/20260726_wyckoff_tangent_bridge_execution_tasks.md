# Wyckoff-tangent bridge 执行清单

权威方法计划：
`docs/experiment_program/20260726_wyckoff_tangent_bridge_iclr_plan.md`

状态：`JOB28185_FAIL_DIAGNOSED_V2_LOCAL_BUILT_REMOTE_NOT_AUTHORIZED`  
日期：2026-07-26

## 0. 历史边界

- [x] `WT000` 保留 P1 `22/36 < 24/36` 的正式 FAIL。
- [x] `WT001` 保留 existing-22 survival/S.U.N. 的正式 FAIL。
- [x] `WT002` 不恢复 MatterSim、MLIP guidance 或旧 constrained decoder。
- [x] `WT003` 不把 job28054 schema failure 解释为 diffusion failure。
- [x] `WT004` 不重试 job28054 或 job28081 身份。

## 1. Schedule-correct parent reference

- [x] `WT100` job28081 `COMPLETED 0:0`。
- [x] `WT101` strict-load released parent PASS。
- [x] `WT102` 32/32 cells terminal。
- [x] `WT103` first-step invalid lattice `0/32`。
- [x] `WT104` non-finite trajectory `0/32`。
- [x] `WT105` positive-volume output `32/32`。
- [x] `WT106` schedule max absolute error `0`。
- [x] `WT107` no retry/replacement/new generation。
- [x] `WT108` 终态审计 SHA
  `4b77a10d632c33b53b2208d49db19f542081a7fd91d52f038ebe7e5280e2cf41`。

## 2. 本地 projector 实现

- [x] `WT200` 新增 step-level `WyckoffTangentProjector`。
- [x] `WT201` 复用 `runtime.project_atom_scores`，不复制第二套求解器。
- [x] `WT202` 实现最小镜像 periodic atom update。
- [x] `WT203` 固定-site orbit 必须零更新。
- [x] `WT204` 记录每 orbit tangent/normal residual 和 rank。
- [x] `WT205` 新增 `LatticeChartProjector`。
- [x] `WT206` 冻结 lattice finite-difference step 与
  Tikhonov `lambda=1e-8`。
- [x] `WT207` decode 后检查 crystal system、正体积、condition number。
- [x] `WT208` 保持 state 输入 immutable，输出使用新 identity。

## 3. Manifold-restricted noise

- [x] `WT300` orbit chart Gaussian 到 full-atom correlated noise。
- [x] `WT301` Jacobian covariance normalization。
- [x] `WT302` lattice chart noise 到 full lattice noise。
- [x] `WT303` 保持 parent timestep/alpha-bar/sigma/time embedding 不变。
- [x] `WT304` 固定 base-noise seed 与 U/T paired transform。
- [x] `WT305` 审计该分布是 manifold-restricted，而非独立逐原子噪声。

## 4. 本地数学与 invariant tests

- [x] `WT400` synthetic 0D/1D/2D/3D orbit tests。
- [x] `WT401` rank-deficient/ill-conditioned Jacobian fail-closed tests。
- [x] `WT402` periodic-boundary crossing tests。
- [x] `WT403` lattice 1D--6D chart round-trip `<=1e-6`。
- [x] `WT404` all seven crystal-system invariant tests。
- [x] `WT405` SG/topology/species/multiplicity/atom-count immutability。
- [x] `WT406` deterministic/permutation-invariant projection。
- [x] `WT407` input/output aliasing 与 mutation tests。
- [x] `WT408` no MLIP/API/retry/replacement static checks。

## 5. WTB-32 contract 与 runner

- [x] `WT500` 只读加载 job28081 U evidence 与 hashes。
- [x] `WT501` 冻结 8 sources × 4 timesteps 与 paired noise。
- [x] `WT502` F arm 从 U terminal evidence 派生，不重跑 U。
- [x] `WT503` T arm 每 reverse step 做 atom/lattice projection。
- [x] `WT504` 每 cell append-only evidence。
- [x] `WT505` terminal report 覆盖 F/T 各 32 cells。
- [x] `WT506` claim 前检查输出不存在、identity 唯一。
- [x] `WT507` claim 前检查 `CPU<=8×A800`。
- [x] `WT508` 禁止训练、生成、CHGNet、MP API、MatterSim。
- [x] `WT509` local runner safety 与 Bash 4.2 tests。

## 6. 远端 gate

状态：`V1_TERMINAL_FAIL_NO_RETRY`。

- [x] `WT600` 本地全部 PASS 后冻结 source/contract/code hashes。
- [x] `WT601` 用户审阅并单独授权 v1 归档传输。
- [x] `WT602` A800 解包、原子安装、installed-byte audit PASS。
- [x] `WT603` 远端 targeted tests、exact import 与 parent strict-load PASS。
- [x] `WT604` 用户单独授权唯一 `1×A800/8CPU` diagnostic。
- [x] `WT605` job28185 严格一次提交。
- [x] `WT606` F `32/32`，T `31/32`；预注册 Mechanics gate 正式 FAIL。
- [x] `WT607` 不可覆盖 terminal audit 已生成；job28185 未重试/替换。

## 6A. job28185 根因与 v2 本地修复

状态：`LOCAL_PASS_REMOTE_NOT_AUTHORIZED`。

- [x] `WT900` 保留 job28185 `FAILED 2:0` 与全部旧 hashes。
- [x] `WT901` 唯一失败固定为 SG63 / timestep800 /
  `b-e2a6902801f056e71703433a`。
- [x] `WT902` 只读恢复 source conventional/primitive lattice 与 exact
  centering transform。
- [x] `WT903` 最小复现证明 v1 把有限倍数变化 \(f\) 误作 log-chart
  `f-1` 更新。
- [x] `WT904` 拒绝只放宽 absolute/relative tolerance。
- [x] `WT905` 实现 `global_chart_retraction_v1`。
- [x] `WT906` `ExpandedState` 携带并冻结 exact primitive transform。
- [x] `WT907` 新增 primitive transform、absolute/relative consistency、
  scale 与 condition audits。
- [x] `WT908` 新增失败 SG63/20× lattice regression。
- [x] `WT909` 新建独立 v2 contract/runner；不覆盖 v1 identity。
- [x] `WT910` 同一 8×4 panel 标注为 development mechanics only，
  `confirmatory_evidence=false`。
- [x] `WT911` 45 项本地 targeted tests：42 PASS、3 expected skip；
  compile/Ruff PASS。
- [ ] `WT912` 用户审阅 v2 exact hashes；当前未授权 archive/transfer/install/
  Slurm。
- [ ] `WT913` 若另行授权，制作唯一 archive 与 installed-byte/remote-test Gate。
- [ ] `WT914` 若远端 Gate PASS 且再次明确授权，最多一次 development F/T32。
- [ ] `WT915` v2 任一 cell 失败即停止且不重试；PASS 也不形成 confirmatory
  claim。

## 7. Confirmatory 256

状态：`BLOCKED_ON_V2_DEVELOPMENT_PASS_AND_NEW_HELD_OUT_CONTRACT`。

- [ ] `WT700` 冻结与 8-source development 不重叠的 256 attempts。
- [ ] `WT701` 冻结 raw/U/T 和 exact-panel released baseline arms。
- [ ] `WT702` 冻结 paired noise、call budget 与 all-attempt denominator。
- [ ] `WT703` 在结果可见前冻结 promotion thresholds。
- [ ] `WT704` 一次生成，无 retry/replacement/best-of。
- [ ] `WT705` MLIP-free direct metrics 与 exact topology audit。
- [ ] `WT706` 方法冻结后才做 CHGNet R5-C strict/meta S.U.N.。
- [ ] `WT707` paired exact tests 与 bootstrap 10,000。
- [ ] `WT708` 生成不可覆盖 WTB-256 terminal audit。

## 8. 训练决策

- [ ] `WT800` WTB-32 PASS 也不得自动训练。
- [ ] `WT801` WTB-256 无 paired gain：停止，不训练。
- [ ] `WT802` WTB-256 有 gain 但 residual 大：另行设计小 adapter gate。
- [ ] `WT803` training-free 已 PASS：优先保留简洁方法，不自动加训练。

## 9. 当前工作指针

本地冻结完成后的唯一 active item：

```text
WT912
```

job28185 已正式 FAIL 且保持不可变；其根因、受控复现、新 retraction、v2
contract/runner 与 local audit 已冻结。下一步仅为用户审阅 v2；`WT913` 及以后
仍需对新的 v2 hashes 另行明确授权。当前不得制作远端归档、传输、安装、申请
GPU、创建 Slurm claim 或训练。
