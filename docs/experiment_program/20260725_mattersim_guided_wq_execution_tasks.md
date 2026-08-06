# MatterSim-guided WQ 执行任务清单

状态：`LOCAL_PREPARATION`  
主计划：
`docs/experiment_program/20260725_mattersim_guided_wq_iclr_execution_plan.md`

规则：任务只能按依赖推进；任何 gate FAIL 都保留证据并停止该分支，不能静默
重试、换 panel、换 evaluator 或覆盖输出。

## A. 当前本地准备

- [x] `A001` 明确角色：MatterSim=guide，CHGNet=held-out 主评估。
- [x] `A002` 明确修订范围：不覆盖历史协议，只新增后续实验 addendum。
- [x] `A003` 核对 MatterSim 1.1.2、5M checkpoint 名称与 SHA。
- [x] `A004` 核对 CHGNet 0.4.2/model 0.3.0、checkpoint SHA 与
  `diff_meets_diff` 环境。
- [x] `A005` 固定 A800 资源硬约束 `CPU <= 8 × A800`。
- [x] `A006` 运行
  `diagnostics/preflight_mattersim_guidance_contract.py`。
- [x] `A007` 运行 `tests/test_mattersim_guidance_contract.py`。
- [x] `A008` 记录 config、plan、task list、validator、preflight 和 test SHA。
- [x] `A009` 独立复核：旧 MLIP 角色协议文件未被改写。
- [x] `A010` 冻结训练边界：MatterSim/CHGNet 均不得进入 loss、teacher、
  label generation、tuning 或 checkpoint selection。

产物：

- `configs/experiments/wyckoff_codiffusion/mattersim_guidance_chgnet_eval_v1.json`
- `crystal_dlm/wqcodiff/guidance_contract.py`
- `diagnostics/preflight_mattersim_guidance_contract.py`
- `tests/test_mattersim_guidance_contract.py`

## B. G1 — MatterSim CUDA 与有限差分 smoke

依赖：`A001-A009 PASS`

- [ ] `B001` 从 MP20 validation 冻结 8 个非 held-out 结构。
- [ ] `B002` 写入 structure IDs、CIF/structure JSON SHA、元素与晶系分层。
- [ ] `B003` 冻结有限差分方向、`epsilon`、能量/force 容差。
- [ ] `B004` 冻结 position trust-region 更新公式和符号审计。
- [ ] `B005` 写唯一 smoke config、sbatch 和 submission claim。
- [ ] `B006` fail-closed 检查 `1×A800, CPU<=8`。
- [ ] `B007` 在 isolated MatterSim runtime 导入 package/CUDA extension。
- [ ] `B008` 校验 checkpoint SHA
  `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5`。
- [ ] `B009` 检查 energy/forces/stress finite。
- [ ] `B010` 检查 deterministic repeat。
- [ ] `B011` 检查 directional derivative 与 force 符号/尺度。
- [ ] `B012` 检查 >=95% 小步能量不升。
- [ ] `B013` 检查没有新 overlap、非正体积、极端 volume/atom。
- [ ] `B014` 生成不可覆盖 smoke acceptance 与终态审计。

PASS 后产物：

- `runs/<run>/notes/mattersim_guidance_fd_smoke_submission.json`
- `runs/<run>/outputs/mattersim_guidance_fd_smoke_v1/report.json`
- `runs/remote_audit/<date>_mattersim_guidance_fd_smoke_terminal_v1.json`

FAIL：停止 MatterSim guidance；禁止改用 CHGNet guidance。

## C. G2 — Composition mechanism 64

依赖：可与 G1 并行准备，但科学结果在新鲜 panel 冻结前不得用于确认性结论。

- [ ] `C001` 冻结 36 no-neutral + 12 Pauling-only + 16 matched controls。
- [ ] `C002` 固定 controls 匹配规则，只使用处理前特征。
- [ ] `C003` 实现固定 topology、固定元素集合、完整 orbit species reassignment。
- [ ] `C004` 固定最少 changed orbits / atoms 和 hash tie-break。
- [ ] `C005` 对无解 proposal 返回 failure，不换元素、不重采样。
- [ ] `C006` 单元测试 formula、site species、multiplicity 和 atom count 一致。
- [ ] `C007` 跑 deterministic proposal-only audit。
- [ ] `C008` 跑同 parent/noise 的 64 paired mechanism panel。
- [ ] `C009` 检查 >=24/36 恢复、controls/Pauling identity、61/64 success。
- [ ] `C010` 检查 raw N+U 与 raw meta 未明显退化。
- [ ] `C011` 生成不可覆盖 chemistry mechanism audit。

FAIL：转向 chemistry-aware WQ decoding / formula-plan SFT；不训练 geometry
diffusion 来修 composition。

## D. G3 — Schedule-correct bridge

依赖：`B PASS` 不是必要条件；`C PASS` 是进入正式新模型前的必要条件。

- [ ] `D001` 将 clean proposal condition 与 forward-noised state 分离。
- [ ] `D002` 复用 released parent discrete alpha/sigma schedule。
- [ ] `D003` strict-load parent decoder 和 time embedding。
- [ ] `D004` 为 lattice-chart、atom-to-orbit tangent、condition projection 写测试。
- [ ] `D005` 冻结 `t={100,200,400,800}`、8 attempts/cell matrix。
- [ ] `D006` 校验 paired noise 和 schedule reconstruction。
- [ ] `D007` 运行非长训 parity smoke。
- [ ] `D008` 检查 finite、首步 lattice、raw success 与失败分母。
- [ ] `D009` 生成不可覆盖 bridge parity lock。
- [ ] `D010` 冻结 MLIP-free 训练信号：MP20 train statistics、解析
  distance/volume/lattice constraints、self-conditioning、parent trajectories。
- [ ] `D011` 检查训练代码和配置不导入/调用 MatterSim、CHGNet、MACE。
- [ ] `D012` 只用 MLIP-free validation 冻结 checkpoint/early-stop。

FAIL：不启动新 diffusion 长训。

## E. G4 — MatterSim-only Geometry-64

依赖：`B PASS`、`D PASS`

- [ ] `E001` 冻结与 held-out/confirmatory 不重叠的 Geometry-64。
- [ ] `E002` 固定 composition-valid、O 分层、密度/体积/距离分层。
- [ ] `E003` 在模型 checkpoint 已冻结后，实现推理期
  `predicted clean x0` position-only corrector。
- [ ] `E004` 固定网格 `K={0,1,2,4}`。
- [ ] `E005` 固定 `dmax={0.01,0.02,0.05} Å`。
- [ ] `E006` 固定 `t_start={25,50,100}`。
- [ ] `E007` 记录每 attempt MatterSim calls、energy、forces、geometry。
- [ ] `E008` 禁止 line search、retry、replacement、best-of。
- [ ] `E009` 用 MatterSim+非 MLIP geometry signals 选择唯一配置。
- [ ] `E010` 检查 finite=100%、energy nonincrease>=95%、无新 geometry failure。
- [ ] `E011` 在查看 CHGNet 前独占写 `guidance_freeze.json`。

## F. G5 — CHGNet held-out 64 transfer

依赖：`E011 PASS`；禁止用该 panel 调参。

- [ ] `F001` 冻结 64 个新 held-out attempt IDs 和 paired noise。
- [ ] `F002` 运行 unguided / frozen MatterSim-guided 两臂。
- [ ] `F003` 确认 generation 和 guidance 调用预算固定。
- [ ] `F004` 使用 `diff_meets_diff` exact R5-C A100 protocol on A800。
- [ ] `F005` 检查 CHGNet checkpoint SHA 与冻结 MP cache。
- [ ] `F006` 计算 raw/chem/joint strict/meta、N+U、failure counts。
- [ ] `F007` paired exact test + 10,000 bootstrap。
- [ ] `F008` 应用“不下降/strict最多1个/N+U最多2个” gate。
- [ ] `F009` 生成不可覆盖 transfer acceptance。

FAIL：MatterSim guidance 不进入 confirmatory 256；不得在该 panel 调参后重跑。

## G. G6 — Confirmatory 256

依赖：`C PASS`、`D PASS`、`F PASS`

- [ ] `G001` 冻结全新 256 attempts、MP cache 与 paired noise。
- [ ] `G002` 准备 `C-CRYSLLMGEN-RELEASED`。
- [ ] `G003` 准备 `C-ATOM-MATCHED`。
- [ ] `G004` 准备 `C-WQ-BASE`。
- [ ] `G005` 准备 `C-WQ-CHEM-HANDOFF`。
- [ ] `G006` 准备 `C-WQ-CHEM-MSGUIDE-HANDOFF`。
- [ ] `G007` 校验所有 arms matched model/call budget。
- [ ] `G008` 执行一次生成，不筛选、不重试。
- [ ] `G009` 执行 CrysLLMGen direct metrics。
- [ ] `G010` 执行 exact CHGNet R5-C S.U.N.。
- [ ] `G011` 计算 raw/chem/joint 指标和 paired statistics。
- [ ] `G012` 应用 comp>=89%、joint>=88%、chem-meta +2pp 等 gate。
- [ ] `G013` 冻结唯一 champion 或记录 no-promotion。

## H. G7 — 最终论文跑

依赖：`G PASS`

- [ ] `H001` 只保留 released、atom-matched、WQ base、champion。
- [ ] `H002` 冻结 3 training seeds × 1000 attempts。
- [ ] `H003` 预算检查；每个 A800 job 仍 `CPU<=8×A800`。
- [ ] `H004` 执行并完成全 attempt 审计。
- [ ] `H005` 报告 hierarchical CI、failure taxonomy、compute。
- [ ] `H006` 写论文表格、claim boundary 和复现清单。

## I. 停止规则

- MatterSim CUDA/finite-difference FAIL：停止 guidance。
- Chemistry-64 FAIL：优先修 planner，不进入新 diffusion 长训。
- Schedule parity FAIL：停止 bridge training。
- CHGNet transfer FAIL：删除 guidance arm 的 promotion 资格，但保留证据。
- Confirmatory 256 FAIL：不跑 3×1000，不用新的 post-hoc panel 挽救。
- 任一 A800 job 请求 `CPU > 8×A800`：在 claim 前 fail-closed。
- 任一 identity 的 claim/output 已存在：禁止覆盖和重复提交。
