# ICLR 主线修订：Wyckoff-tangent CrysLLMGen bridge

状态：`AUTHORITATIVE_P3_P4_AMENDMENT_WITH_JOB28185_V2`  
日期：2026-07-26  
适用 run：`20260720_0401-crysllmgen-wq-final-v3`

## 1. 修订范围

本文档只修订
`20260725_iclr_mlip_free_wq_experiment_focus_and_roadmap.md` 的 P3、P4
以及与其直接相关的停止规则。下列历史结果继续原样保留：

- P1 固定-topology chemistry mechanism 的正式结果仍为 `22/36`，低于
  `24/36` gate，正式结论为 **FAIL**；
- existing-22 survival/S.U.N. escalation 仍为 **FAIL**；
- 不恢复旧 constrained-decoder 路线；
- 不追溯性修改任何阈值、分母、失败身份或审计；
- MatterSim 与其他 MLIP 继续完全排除在训练、采样、选择和 guidance 之外；
- CHGNet 继续只作方法冻结后的 held-out evaluator。

旧的
`20260726_wq_schedule_correct_bridge_exploratory_execution.md`
是 schedule parity 的执行历史，不再授权其末尾所述的短训练。本文档是该
parity PASS 之后的新科学计划。

## 2. 新的中心假设

当前最保守、同时有明确方法贡献的方案是：

> WQ LLM 固定离散的 composition、space group、Wyckoff orbit inventory 和
> species assignment；released CrysLLMGen 保持原 checkpoint 与 scheduler，
> 但其逐原子、逐晶格反向更新在每一步都被拉回固定 Wyckoff quotient 的切空间；
> CHGNet 仅在方法完全冻结后评估 S.U.N.

职责因此变为：

1. **WQ LLM：** 生成离散 chemistry/topology。
2. **released CrysLLMGen：** 提供已经学到的连续几何 score/update。
3. **Wyckoff tangent bridge：** 把 full-atom update 投影到固定 quotient chart。
4. **CHGNet：** 只作最终 held-out relaxation/hull/S.U.N. 评估。

这不是“重新训练一个 Wyckoff diffusion”。第一阶段完全 training-free，目标是
证明一个现有 parent 能否在不破坏 WQ 离散结构的情况下恢复几何。

## 3. 已完成的 schedule-correct gate

不可变参考：

- 原始 diagnostic `job28054` 在科学计算前因 result schema binding 失败；
  不把该失败解释为 diffusion 失败，也不重试该身份。
- 唯一 supersession `job28081`：
  `wq_schedule_correct_bridge_parity_sup28054_v1`；
- 资源：`1×A800 / 8 CPU / 64 GiB`；
- Slurm：`COMPLETED 0:0`，elapsed `00:00:34`；
- 矩阵：`t={100,200,400,800} × 8 = 32 cells`；
- terminal cells：`32/32`；
- finite positive-volume raw output：`32/32`；
- first-step invalid lattice：`0/32`；
- non-finite trajectory：`0/32`；
- schedule max absolute error：`0`；
- strict parent load：PASS；
- retry/replacement：`false`。

终态审计：

```text
runs/remote_audit/
  20260726_wq_schedule_correct_bridge_parity_sup28054_v1/
  terminal_audit.json
```

SHA256：

```text
4b77a10d632c33b53b2208d49db19f542081a7fd91d52f038ebe7e5280e2cf41
```

该结果只证明：

- released parent 可以在正确 forward-noise/schedule 契约下运行；
- 先前失败不能归因于 parent diffusion 本身；
- 可以安全进入 Wyckoff projection 的本地方法开发。

它不证明：

- composition validity 提升；
- structural/joint validity 提升；
- S.U.N. 提升；
- 论文贡献已经成立；
- 任何新训练已获授权。

## 4. 与既有工作的边界

已有工作已经证明 Wyckoff-aware diffusion 可行，包括 WyckoffDiff、SymmCD、
SGEquiDiff 和 DiffCSP++。因此论文不能把“diffusion 使用 Wyckoff”本身当作
新颖性。

本项目的可检验差异必须收窄为：

> 一个 LLM 先给出固定 WQ 离散状态，再用 schedule-correct、training-free
> 或极轻量 adapter 的 tangent projection，把 released full-atom parent 的
> 几何先验转移到该 quotient；全过程 MLIP-free。

第一阶段只验证 training-free 版本。若该版本没有形成可测的几何收益，不得用
长训练扩大计算来挽救叙事。

## 5. 现有实现基础

不新增第二套晶体表示。直接复用：

- `crystal_dlm/wqcodiff/state.py`
  - `StratifiedState`
  - `OrbitState`
- `crystal_dlm/wqcodiff/charts.py`
  - `AffineOrbitChart`
  - `LatticeChartCodec`
- `crystal_dlm/wqcodiff/runtime.py`
  - `expand_state`
  - `atom_to_orbit`
  - `orbit_jacobians`
  - `project_atom_scores`

现有 `project_atom_scores` 已实现带 Tikhonov 正则的最小二乘切空间拉回。新代码
只需把它封装成反向步级别的可审计 projector，并对 lattice chart 提供同等的
拉回/重解码路径。

## 6. 方法定义

### 6.1 固定离散状态

一个 attempt 在进入 diffusion 前冻结：

- composition 和每个 orbit 的 species；
- space group；
- Wyckoff type/letter identity；
- conventional 与 primitive multiplicity；
- orbit 数量与 atom count；
- orbit ID 的 canonical mapping。

任何反向步都不得：

- 增删 orbit；
- 改 species；
- 改 multiplicity；
- 改 space group；
- 做 topology revision；
- retry、replacement 或 best-of。

因此该方法可以保持 composition，但不能修复 proposal 阶段已经形成的 48 个
composition-invalid。该限制必须在论文中明确报告。

### 6.2 原子更新的切空间拉回

对 orbit \(i\)，展开后的 parent update 为
\(\Delta x_i \in \mathbb{R}^{3m_i}\)，Wyckoff chart Jacobian 为
\(J_i \in \mathbb{R}^{3m_i \times d_i}\)。注册的拉回为：

\[
\Delta q_i =
\left(J_i^\top J_i + \lambda I\right)^{-1} J_i^\top \Delta x_i,
\qquad \lambda=10^{-8}.
\]

然后：

1. 更新该 orbit 的 free coordinate；
2. 用 `AffineOrbitChart.decode` 重建 representative；
3. 用 exact symmetry expansion 重建全部 primitive atoms；
4. 记录 normal residual
   \(\|\Delta x_i-J_i\Delta q_i\|_2\)；
5. 固定点 orbit（\(d_i=0\)）的更新必须严格为零。

周期边界差值统一使用最小镜像，所有求解使用 float64；parent 推理 dtype
不因此改变。

### 6.3 v1 晶格更新（历史失败方法）

v1 曾对 parent 预测的 full lattice update 执行：

1. 用当前 `LatticeChartCodec.decode_matrix` 得到 \(L(z)\)；
2. 以冻结的中心差分步长构造局部
   \(J_L=\partial\operatorname{vec}L/\partial z\)；
3. 用同一正则最小二乘求 \(\Delta z\)；
4. 在 chart 中更新 \(z\)；
5. 重新 decode，强制满足 crystal-system 参数关系与正体积。

job28185 已证明该有限步长用法不成立：log-length chart 对倍数 \(f\) 的正确
有限位移是 \(\log f\)，局部线性解却近似 \(f-1\)，decode 后产生指数爆炸。
该 v1 方法及 job28185 的 FAIL 结论保持不可变，不再用于新执行。

### 6.3A v2 全局 chart retraction

v2 对绝对 proposed lattice 执行：

1. 从 `ExpandedState` 读取 exact conventional-to-primitive transform；
2. 用 linear solve 得到 proposed conventional lattice；
3. 直接用 `LatticeChartCodec.encode_matrix` 得到 proposed chart；
4. 只 decode 一次，得到满足 crystal system 的 canonical lattice；
5. 用同一个 exact transform 回到 primitive frame；
6. re-expansion 后核验 transform 与 lattice consistency。

局部 lattice Jacobian 只保留作诊断线性化，不再决定有限步长输出；lattice
Tikhonov 正则不用于 v2 retraction。每步记录：

- lattice projection residual；
- exact chart displacement 与 actual retracted update；
- determinant；
- condition number；
- encode/decode round-trip error；
- crystal-system invariant；
- primitive transform consistency；
- primitive lattice absolute/relative consistency 与 scale。

### 6.4 manifold-restricted forward noise

parent 的离散 timestep、\(\bar\alpha_t\)、\(\sigma_t\)、time embedding 与
reverse call budget全部保持不变。连续噪声改为：

- orbit free-coordinate chart 中的高斯噪声，经 Jacobian covariance
  normalization 后展开为 symmetry-correlated atom noise；
- lattice chart 中的高斯噪声，经局部 Jacobian normalization 后展开为
  crystal-system-compatible lattice noise。

该过程使用 parent 的同一 scalar schedule，但不是声称与 parent 的独立逐原子
噪声分布完全相同。报告中应称为
`parent-schedule / manifold-restricted forward process`。

## 7. 三个冻结 arms

| arm | 定义 | 用途 |
|---|---|---|
| `U` | job28081 的 unconstrained schedule-correct parent output | 不可变参考，不重跑 |
| `F` | 只把 `U` 的 terminal output 投影回固定 WQ manifold | final-only 诊断，不作为主方法 |
| `T` | manifold-restricted forward noise，且每个 reverse step 都做 tangent/lattice projection | 候选主方法 |

`F` 用于区分“最终几何合法化”与“整条对称约束轨迹”的收益；真正的因果比较是
同 source、同 timestep、同 base noise 的 `U` 与 `T`。

## 8. Gate WTB-32：无训练 paired preflight

### 8.1 输入

完全复用 job28081 的：

- 8 个 immutable source proposals；
- `t={100,200,400,800}`；
- paired base noise；
- released `model_494.pt`；
- 32 reverse steps；
- scheduler 与 call budget。

不重新生成 proposal，不重跑 `U`。新 diagnostic 只计算 `F` 与 `T`，并把
job28081 的 `U` evidence 作为只读 reference。

### 8.2 必须在提交前冻结

- 新 execution identity、input manifest 与 output path；
- source/job28081 evidence hashes；
- projector implementation hash；
- chart finite-difference step；
- Tikhonov \(\lambda=10^{-8}\)；
- exact base-noise transform；
- per-cell call budget；
- 1×A800 / 8 CPU resource envelope；
- no retry/replacement/overwrite contract。

### 8.3 Mechanics PASS

必须同时满足：

1. `F` 与 `T` 各 `32/32` cells 终态；
2. `F` 与 `T` first-step/terminal non-finite 均为 `0/32`；
3. `F` 与 `T` positive-volume output 均为 `32/32`；
4. `T` 每个 reverse step 的 space group 与固定 topology hash 不变；
5. `F/T` species、multiplicity、orbit count、atom count 均 byte-identical；
6. orbit chart round-trip max error `<=1e-6`；
7. lattice chart round-trip max error `<=1e-6`；
8. fixed-site orbit drift `<=1e-8`；
9. parent timestep/schedule tensor与 job28081 一致；
10. 每个 cell 只有一个 terminal output；
11. retry/replacement/best-of/MLIP/API/new generation 均为 0；
12. 不读取 CHGNet、S.U.N.、hull、final-test 结果。

本 gate 只选“mechanically valid or stop”，不得按 minimum distance、密度、
CHGNet 或 S.U.N. 在 `F/T` 中挑冠军。

### 8.4 Stop

任一 Mechanics 条件失败：

- 保留全部 cell 与残差证据；
- 不重试或替换；
- 不训练 adapter；
- 不进入 256；
- 回到表示/投影数学诊断。

## 9. 冻结后的 confirmatory 256

只有 WTB-32 PASS，且在查看任何新 CHGNet 输出前，才创建独立
`WTB-256` contract。

### 9.1 Panel

- 新的、与现有 8-source development 集合不重叠的 256 attempts；
- WQ LLM checkpoint、decoding policy 与 attempt IDs 冻结；
- 每 trajectory 单一 proposal；
- `U` 与 `T` 使用相同 source 与 paired noise；
- released CrysLLMGen baseline 若不能在相同 panel 重跑，只能标记为
  historical reference，不能冒充 paired baseline。

### 9.2 主要 arms

1. raw WQ expanded structure；
2. `U`：unconstrained schedule-correct bridge；
3. `T`：every-step Wyckoff tangent bridge；
4. released CrysLLMGen exact-panel baseline（若预算和冻结合同允许）。

`F` 只保留为诊断，不参与 headline 选择。

### 9.3 指标

先做完全 MLIP-free 的：

- generation/render success；
- composition/structural/joint validity；
- exact space-group retention；
- exact Wyckoff multiset/topology retention；
- minimum-distance、volume、density 与 collision taxonomy；
- uniqueness 与 novelty；
- all-attempt failure accounting。

方法冻结后再用统一的 `diff_meets_diff` + CHGNet R5-C evaluator 做：

- strict S.U.N.；
- meta S.U.N.；
- chemistry-gated sensitivity；
- all-attempt denominator；
- paired bootstrap 10,000；
- coverage-adjusted 只报告。

### 9.4 256 promotion gate

正式阈值必须在生成 `WTB-256` 输出前写入其独立 contract。当前冻结方向为：

- `T` composition 与其 WQ source 必须逐 attempt 完全一致；
- `T` exact topology retention 必须为 `256/256`；
- `T` generation/render success 不低于 `U`；
- `T` joint-valid 相对 raw WQ 至少提高 `3 pp`；
- `T` strict S.U.N. 点估计至少达到历史 CrysLLMGen `9.0%`；
- `T` meta S.U.N. 不低于历史 `46.1%`；
- paired `T-U` strict 与 joint 方向均不得为负；
- N+U 相对 `U` 下降不超过 `2 pp`；
- 无 retry/replacement/best-of。

这些 geometry-only arms 固定 composition，因此旧路线中的
`composition-valid >=89%` 不能作为 tangent bridge 的独立成功声明；它只能由
上游 WQ proposal 改善。旧的 `joint-valid >=88%` 也不能机械复用，因为当前
81.25% composition cap 使其在固定 proposal panel 上不可达。

## 10. 训练决策

WTB-32 PASS 不授权训练。WTB-256 的决定为：

- **若 T 无 paired geometry/S.U.N. 改善：** 停止 bridge，不训练。
- **若 T 改善但 residual 很大：** 只设计一个冻结 parent、训练小
  projection adapter 的新 gate。
- **若 T 已满足 promotion gate：** 以 training-free 方法作为首选论文方案；
  不为追求更大数字自动增加训练。

任何 adapter 都必须：

- 新身份、新 panel、新授权；
- parent backbone 默认冻结；
- checkpoint selection 完全 MLIP-free；
- CHGNet 结果不可反馈进训练；
- 最多先做短 smoke，不直接长训。

## 11. 论文贡献的保守表述

只有 WTB-256 PASS 后，才允许把主张写成：

1. LLM 离散 WQ 与 pretrained full-atom diffusion 的接口；
2. schedule-correct manifold-restricted forward/reverse bridge；
3. training-free Wyckoff tangent projection；
4. exact topology retention 与 held-out structural/S.U.N. 转移；
5. 全流程 MLIP-free generation，CHGNet 仅 held-out evaluation。

在此之前，28081 只能作为 negative-result correction 与 engineering
correctness evidence，不能作为 headline improvement。

## 12. 2026-07-26 本地实现状态

本地 training-free 实现已经完成，仍未获授权进行任何远端动作：

- `crystal_dlm/wqcodiff/crysllmgen/tangent_bridge.py`
  已实现 step-level atom/lattice tangent projection、manifold-restricted forward
  noise，以及复用 released CrysLLMGen decoder/scheduler 公式的 projected
  corrector/predictor reverse；
- atom projection 直接调用现有
  `crystal_dlm.wqcodiff.runtime.project_atom_scores`，没有复制第二套求解器；
- lattice projection 覆盖全部七个 crystal systems，并对 rank、condition
  number、正体积和 round-trip fail closed；
- WQ primitive lattice 与 CrysLLMGen 根据 lengths/angles 重建的 canonical
  lattice 可能只差一个刚性右旋转。实现新增 `ParentLatticeFrame`，只接受
  Gram-matrix 一致且行列式为正的正交 frame 映射；parent update 先映回 WQ
  frame 投影，再映回 parent frame。非刚性或反射映射直接失败；
- `scripts/a800/run_wq_wyckoff_tangent_bridge_preflight_v1.py`
  只读加载 job28081 的 32-cell U evidence；F 只做一次 terminal projection，
  T 使用 paired manifold noise 并在每个 corrector/predictor 子步后投影；
- WTB-32 scientific contract、Bash-4.2-compatible `preflight.sbatch` 与
  fail-closed `submit_once.sh` 已在本地冻结；
- wrapper 在 claim 前强制核验授权记录、installed patch、全部输入 hashes、
  唯一 identity、输出不存在及 `1×A800 / 8 CPU` 资源上限；
- 本地单测覆盖 0D/1D/2D/3D orbit、周期边界、固定 site、病态 Jacobian、
  七晶系、frame adapter、immutability、permutation invariance、call budget、
  append-only evidence 和无训练/MLIP/API 静态边界。

本机环境缺少远端锁定的 PyXtal、Torch 与完整 CrysLLMGen runtime，因此本地
测试只证明数学核心、调用契约和提交安全性；exact PyXtal round-trip、
released checkpoint strict-load 和 CUDA trajectory 仍属于未来远端 gate，
不能由本地 PASS 代替。

## 13. 当前唯一下一步

本地整体验证已经完成：61 项测试 PASS，6 项因本机缺少锁定的
PyXtal/spglib runtime 按预期跳过；JSON、Ruff 与 Bash syntax 均 PASS。
immutable local implementation audit 固定写入：

```text
runs/remote_audit/
  20260726_wq_wyckoff_tangent_bridge_local_v1/
  local_implementation_audit.json
```

当前下一步是把文档、代码、scientific contract、审计与冻结 hashes 交给用户
审阅。

只有另行明确授权后，才允许依次执行：

1. 制作并传输唯一归档；
2. A800 原子安装和 installed-byte audit；
3. 远端 exact imports 与测试；
4. 再次单独授权后，严格一次提交
   `1×A800 / 8 CPU / 64 GiB / 1 h` 的 WTB-32 diagnostic。

当前不传输、不安装、不提交、不训练，也不创建 Slurm claim。

## 14. 2026-07-26 job28185 终态修订

本节是对第 8、12、13 节执行状态的最新修订；旧文字保留为 v1 预注册历史。

### 14.1 v1 结果

WTB-32 已实际执行一次：

- identity：`wq_wyckoff_tangent_bridge_preflight_v1`；
- job：`28185`；
- Slurm：`FAILED 2:0`；
- F：`32/32`；
- T：`31/32`；
- 唯一失败：SG63、timestep800、first predictor lattice projection；
- retry/replacement：均为 `false`。

因此第 8.3 节的全-cell Mechanics gate 正式为 **FAIL**。该结论不因本地找到
bug 而更改，也不允许重试/替换 job28185。

### 14.2 根因

v1 把绝对 lattice proposal 的大有限位移当作当前 log-chart 的一阶增量。
受控复现中 20× 轴变化被写成约 `+19`，而非 `log(20)`，decode 后得到约
`3.38e8 Å` 的 primitive lattice 元素。primitive consistency exception 是该
爆炸的浮点症状。

完整推导、证据限制与复现表见：

```text
docs/experiment_program/
  20260726_wtb32_job28185_lattice_retraction_root_cause_and_v2_plan.md
```

机器可读审计：

```text
runs/remote_audit/
  20260726_wq_wyckoff_tangent_bridge_job28185_root_cause_v1/
  root_cause_audit.json
```

### 14.3 新 v2 身份

新方法：

```text
wq_wyckoff_chart_retraction_preflight_sup28185_v2
global_chart_retraction_v1
```

实现改为对 proposed conventional lattice 直接 encode/decode，并在
`ExpandedState` 中保留 exact primitive transform。新增 transform、
absolute/relative consistency、scale 与 condition Gate。

冻结文件：

```text
crystal_dlm/wqcodiff/crysllmgen/tangent_bridge.py
crystal_dlm/wqcodiff/runtime.py
scripts/a800/run_wq_wyckoff_chart_retraction_preflight_sup28185_v2.py
configs/experiments/wyckoff_codiffusion/
  wq_wyckoff_chart_retraction_preflight_sup28185_v2.json
```

本地 targeted validation：42 PASS、3 expected skip、0 FAIL；compile/Ruff
PASS。

### 14.4 防止 development leakage

v2 若复用 job28185 的 8×4 panel，只能是 mechanics regression。这个 panel 已
暴露失败并影响修复，所以其结果明确：

```text
development_panel_reused=true
confirmatory_evidence=false
```

它不能用于 T-vs-U promotion、validity/S.U.N. headline 或训练决策。v2
development PASS 后仍需独立、non-overlapping、输出前冻结的 held-out
contract。

### 14.5 当前唯一下一步

当前状态是：

```text
V2_LOCAL_BUILT_REMOTE_NOT_AUTHORIZED
```

未制作 v2 archive，未传输，未安装，未创建 claim，未提交 Slurm，也未训练。
任何远端动作都需要用户对 v2 的新 hashes 再次明确授权；即使未来 development
PASS，也不会自动授权 confirmatory 256 或训练。
