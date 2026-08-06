# MatterSim-guided WQ：ICLR 最短执行计划

状态：`LOCAL_PREPARATION`  
日期：2026-07-25  
适用 run：`20260720_0401-crysllmgen-wq-final-v3`

## 1. 本次决策

本计划新增一份显式协议修订，不覆盖或改写任何历史实验：

- **MatterSim 1.1.2 / MatterSim-v1.0.0-5M** 是唯一、且仅在推理期使用的
  guidance MLIP；
- **CHGNet 0.4.2 / model 0.3.0** 继续使用历史 R5-C A100
  protocol on A800，作为主要、冻结且隔离的 S.U.N. evaluator；
- 新 diffusion 的训练全过程 **MLIP-free**：MatterSim 和 CHGNet 都不进入
  loss、teacher/distillation target、训练标签生成、schedule tuning、
  checkpoint 选择、重试或失败分支；
- MatterSim 不产生 headline S.U.N.，也不改变模型权重；
- 第一版 guidance 只允许在**固定 composition** 的轨迹内，对低噪声
  `predicted clean x0` 做有限次、固定预算、position-only corrector；
- 在 MatterSim-only 配置完全冻结之前，禁止查看对应 CHGNet 结果；
- 任何 A800 job 必须满足 `CPU <= 8 × A800`。

这项修订只改变“MLIP 角色映射”和后续新实验，不改变已经完成的 MatterSim、
CHGNet 或 MACE 评测身份，也不追溯修改历史结论。

## 2. 为什么这样设计

MatterSim 的 ASE calculator 可以提供能量、力和应力，因此可作为连续几何
corrector 的物理信号。第一版只使用力，是因为力的符号和单位可以通过固定
composition 的有限差分独立审计；cell/stress guidance 涉及晶格参数化、应力
符号和体积尺度，必须在独立审计通过后才能打开。

CHGNet 保留为历史主评估器，是为了继续使用当前已冻结的 CrysLLMGen
1000-sample 对比口径。若让 CHGNet 同时指导和评估，就无法区分真实泛化与
evaluator gaming。

MatterSim 也不进入训练。新的 diffusion 只从 MP20 train split、解析几何约束、
released parent 轨迹和标准 diffusion denoising objective 学习。MatterSim
corrector 是模型冻结后、预算固定的推理算子，因此可以单独报告：

1. MLIP-free generator 本身的改进；
2. 同一 generator 加固定 MatterSim inference guidance 的边际改进；
3. 两者向独立 CHGNet evaluator 的迁移。

公开的一般性 MLIP 研究还表明，通用 MLIP 在高能或分布外构型上可能出现
势能面软化。因此 MatterSim guidance 的改善必须迁移到独立 CHGNet
评估，MatterSim 自身能量下降不能作为论文结论。

主要技术依据：

- MatterSim 官方实现：<https://github.com/microsoft/mattersim>
- MatterSim 论文：<https://arxiv.org/abs/2405.04967>
- CHGNet 论文：<https://www.nature.com/articles/s42256-023-00716-3>
- 通用 MLIP 势能面软化分析：
  <https://www.nature.com/articles/s41524-024-01500-6>

## 3. 当前证据对新设计的约束

当前 256-panel 不能被概括为“整个方案失败”：

- composition-valid 为 `208/256 = 81.25%`，低于历史约 `89.2%`；
- 48 个 composition-invalid 在 WQ proposal 中已经形成，parent diffusion
  没有改 composition；
- 其中 36 个没有冻结氧化态下的电荷中性解，12 个是 Pauling rejection；
- MP 补查后 raw meta S.U.N. sensitivity 为 `118/256 = 46.09%`，与历史
  CrysLLMGen `461/1000 = 46.10%` 接近；
- strict sensitivity 为 `18/256 = 7.03%`，历史为 `9.00%`，当前样本量
  不足以确认显著退化；
- quotient refiner 的 768/768 失败首先暴露 clean proposal 被直接注入
  高噪声端点的训练—推理契约错误，不能据此判断所有 WQ diffusion 都无效。

因此新计划必须依次解决三个不同问题：

1. WQ proposal 的 composition support；
2. parent / bridge diffusion 的 forward-noise 与 condition 契约；
3. composition 固定后的 residual geometry / stability。

MatterSim 只用于第 3 项，不能跨 composition 比较绝对能量，也不能替代前两项。

## 4. 论文主线

建议收缩后的 ICLR 主张是：

> A chemistry-aware symmetry-native Wyckoff planner, combined with
> schedule-correct continuous refinement and independently evaluated
> MatterSim guidance, improves attempt-level chemical validity and
> CHGNet S.U.N. without filtering, retry, replacement, or extra candidate
> selection.

这个主张保留了三条原始主线：

- symmetry-native Wyckoff quotient representation；
- LLM proposal 与 diffusion geometry refinement 的职责分离；
- attempt-preserving、固定预算、无筛选的 S.U.N. 评测。

暂不把 geometry-triggered dimension-changing feedback 作为必须完成的主结论。
只有正确 schedule 的 bridge 和 MatterSim guidance 都通过机制 gate 后，才恢复
GEOREV 作为附加方法。

## 5. 严格角色防火墙

| 阶段 | MatterSim | CHGNet |
|---|---|---|
| 组成规则开发 | 不使用 | 不使用 |
| scheduler / bridge parity | 不用于选择 | 不使用 |
| diffusion 训练 | 禁止 | 禁止 |
| guidance finite-difference audit | 推理期主信号 | 禁止 |
| guidance 超参数冻结 | 推理期主信号 + 非 MLIP 几何指标 | 禁止 |
| held-out 64 promotion | 只运行已冻结 guidance | 唯一主 MLIP evaluator |
| confirmatory 256/1000 | 固定、不再调参 | 历史 R5-C 主评估 |

必须通过独占、带 hash 的 `guidance_freeze.json` 才能进入 CHGNet 阶段。若在
查看 CHGNet 后修改 corrector 次数、步长、起始 timestep、梯度裁剪或失败处理，
必须创建新开发 identity，旧 held-out panel 不得继续当确认集。

## 6. Guidance v1 的数学与执行契约

### 6.1 允许的信号

对同一 attempt、同一 composition 的低噪声预测结构
\(\hat{x}_0=(R,L,Z)\)，MatterSim 返回：

- 总能量与 eV/atom；
- Cartesian forces；
- stress，仅记录，不在 v1 更新中使用。

v1 corrector 只允许更新 Cartesian positions。其方向和归一化必须先由有限差分
审计确认；在审计前不把任何符号约定写死为科学实现。

### 6.2 固定预算

开发网格冻结为：

- corrector calls `K ∈ {0, 1, 2, 4}`；
- 每次最大 Cartesian displacement
  `dmax ∈ {0.01, 0.02, 0.05} Å`；
- 低噪声入口候选 `t_start ∈ {25, 50, 100}`，以 parent 的离散
  1000-step schedule 为准；
- 每个网格单元使用完全相同的 validation structures、attempt IDs 和 noise；
- 每个 attempt 固定调用数，不做 line search；
- 任一 MatterSim failure 直接保留为失败，不能回退到 unguided、重试或换候选。

### 6.3 明确禁止

- 跨 composition 比较 MatterSim 绝对能量；
- 为同一 attempt 生成多个候选后按 MatterSim 排名；
- best-of-K、rejection sampling、失败后 unguided replacement；
- 使用 CHGNet 选择 timestep、步长、corrector 次数或 checkpoint；
- 在 v1 中用 stress 更新 cell；
- 在 raw 高噪声 `x_t` 上直接调用 clean-structure MatterSim；
- 把 MatterSim energy 降低直接称为稳定性或 S.U.N. 改善。

## 7. 分阶段执行

### G0 — 协议、资产和本地 fail-closed 准备

输入：

- MatterSim package `1.1.2`；
- checkpoint `MatterSim-v1.0.0-5M.pth`；
- checkpoint SHA256
  `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5`；
- isolated runtime
  `/public/home/jiaosz/ywliang/models/wqcodiff/runtimes/mattersim-1.1.2-py310-v4`；
- CHGNet package `0.4.2`、model `0.3.0`；
- CHGNet checkpoint SHA256
  `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`；
- CHGNet environment
  `/public/home/jiaosz/miniconda3/envs/diff_meets_diff`。

Gate：

- 角色、hash、分母、资源、无重试规则通过本地 contract validator；
- 旧协议文件保持不变；
- 没有 Slurm submission 或 scientific attempt。

### G1 — MatterSim A800 单点与导数 smoke

数据：8 个冻结 MP20 validation structures，覆盖单/多元素、不同晶系和 O-rich
结构；不使用当前 256 held-out panel。

资源：`1×A800, <=8 CPU`，非科学 smoke。

检查：

- 运行时和 checkpoint 精确匹配；
- A800 可见且仅一张；
- energy、forces、stress 全 finite；
- 同一输入重复两次结果在冻结容差内一致；
- \(E(R+\epsilon v)-E(R-\epsilon v)\) 与 force 投影符号一致；
- 小 trust-region position step 的能量下降比例至少 95%；
- 不新增 overlap、非正体积或极端 volume/atom。

任一检查失败：停止 guidance 路线，不提交训练，不改用 CHGNet guidance。

### G2 — Composition mechanism 64-panel

该阶段独立于 MatterSim：

- 36 个 `no_charge_neutral_assignment`；
- 12 个 Pauling-only；
- 16 个按处理前特征固定匹配的 valid controls。

干预仅允许固定拓扑、固定元素集合、完整 orbit 的 deterministic
species reassignment；Pauling-only 与 controls 保持 identity。不得训练、
重采样或用 MLIP 选择。

晋级 gate：

- 至少 `24/36` no-neutral 恢复冻结 legacy comp-valid；
- `16/16` controls byte-identical；
- `12/12` Pauling-only 不被硬删或硬修；
- generation success 至少 `61/64`；
- raw N+U 与 raw meta S.U.N. 均不明显退化；
- 全 64 attempts 保留在分母。

如果该 gate 不通过，优先转向 chemistry-aware WQ decoding / formula-plan SFT，
而不是训练新 diffusion。

### G3 — Schedule-correct bridge parity

基于 `CrysLLMGenWQBridgeV2`：

- strict-load released parent decoder 和 time embedding；
- geometry state 必须按 parent forward process 在 timestep `t` 加噪；
- clean WQ proposal 是独立 condition，不能冒充 noisy state；
- 首先只训练 lattice-chart bridge、atom-to-orbit tangent bridge 和 proposal
  condition projection；
- 使用 `t={100,200,400,800}` 的 8-attempt/cell parity matrix；
- 不使用 MatterSim 或 CHGNet 选择 checkpoint。

晋级 gate：

- 所有轨迹 finite；
- 首步没有 invalid lattice；
- raw generation success 不低于已验证 parent handoff；
- 所有 failures 留在分母；
- schedule identity 和 forward-noise reconstruction audit PASS。

只有该阶段通过，才允许启动新的 diffusion 短训。

### G4 — MatterSim-only guidance development

使用独立 Geometry-64 validation panel，composition 全部固定且有效，覆盖：

- stable / unstable；
- O-rich / non-O；
- density、volume/atom、minimum-distance 分层；
- 不与 CHGNet held-out 64 或 confirmatory 256 重叠。

运行 G1 已通过的 position-only corrector 网格。只允许以下选择信号：

- MatterSim within-trajectory energy delta；
- max-force reduction；
- reconstruction、minimum distance、volume/atom、density；
- parent schedule recovery 和 finite rate。

冻结唯一配置的 gate：

- finite rate 100%；
- 至少 95% eligible cases 的 MatterSim energy 不升；
- geometry failure 不增加；
- reconstruction success 不下降；
- fixed-call budget 可审计；
- `guidance_freeze.json` 独占生成且 hash 固定。

### G5 — 一次性 CHGNet held-out 64 transfer gate

比较：

- A：schedule-correct unguided；
- B：完全冻结的 MatterSim-guided；
- 相同 attempt IDs、composition、noise、parent checkpoint 和 calls accounting。

CHGNet 使用 exact R5-C A100 protocol on A800，主分母为 64。主要晋级规则：

- raw `MLIP-SUN@0.1` 不下降；
- chemistry-gated 与 joint-valid `MLIP-SUN@0.1` 不下降；
- raw strict `@0.0` 不下降超过 1 个 attempt；
- N+U 不下降超过 2/64；
- generation/reconstruction failure 不增加；
- MatterSim 与 CHGNet 至少在方向上没有系统性反转；
- paired bootstrap 10,000 次和逐 attempt ledger 完整。

这是一次性 transfer gate。失败后不得调参并重用同一 held-out panel。

### G6 — 新鲜 confirmatory 256

冻结、互斥的新 attempt panel 上运行最小方法矩阵：

1. `C-CRYSLLMGEN-RELEASED`；
2. `C-ATOM-MATCHED`；
3. `C-WQ-BASE`；
4. `C-WQ-CHEM-HANDOFF`；
5. `C-WQ-CHEM-MSGUIDE-HANDOFF`。

若 G3 的 topology mechanism 另行通过，可加
`C-WQ-CHEM-MSGUIDE-GEOREV`，但它不是本轮关键路径。

所有方法共享：

- 256 个 attempt IDs 和 paired noise；
- frozen parent/reverse schedule；
- identical MP train/reference/cache；
- exact CHGNet R5-C evaluator；
- all-attempt denominator；
- 无 retry/replacement/best-of-N；
- 不做 DFT；
- coverage-adjusted 仅报告，不参与选择。

层级 gate：

1. composition-valid `>=89%`；
2. joint-valid `>=88%`；
3. chemistry-gated meta S.U.N. 相对 `C-WQ-BASE` 至少 `+2 pp`；
4. raw meta 同方向；
5. strict 非劣；
6. N+U 下降不超过 `2 pp`；
7. graph acceptance `>=95%`。

该 256-panel 只用于 promotion，不承担最终显著性主张。

### G7 — ICLR 确认性 3×1000

只对冻结冠军和必要基线运行：

- released CrysLLMGen；
- `C-ATOM-MATCHED`；
- `C-WQ-BASE`；
- 唯一冠军。

报告 raw / chemistry-gated / joint-valid strict 与 meta S.U.N.、comp/struct/joint
validity、N+U、failure taxonomy、all-metal shortcut、计算量和 paired /
hierarchical confidence intervals。

若 G6 不通过，停止长跑，把 WQ 路线写成因果诊断与负结果，不用 evaluator
specific tuning 挽救 headline。

## 8. 新 diffusion 的 MLIP-free 训练边界

新 diffusion 的长训练只有在 composition mechanism 与 schedule parity
通过后才允许；MatterSim G4 不再是训练前置条件，因为它不参与训练。模型先
独立冻结，随后才挂接推理期 corrector。

1. **Stage A：schedule parity**  
   冻结 parent backbone，只训练 bridge/proposal-condition 层。
2. **Stage B：proposal-conditioned short training**  
   正确 forward-noised state；最多一轮短训练，先证明 finite generation。
3. **Stage C：MLIP-free geometry regularization（可选）**  
   仅允许使用 MP20 train split 冻结统计与解析约束：元素/原子数条件下的
   log-volume/atom prior、晶格 condition-number prior、基于训练集或共价半径
   的平滑 minimum-distance barrier、symmetry reconstruction、self-conditioning
   和 released-parent trajectory consistency。所有统计先于 validation 冻结并
   记录 SHA；不得调用任何 MLIP 生成 loss 或标签。
4. **Stage D：topology revision（可选）**  
   只在 schedule parity 和 geometry mechanism 都通过后恢复。

模型 checkpoint、训练超参数和停止点只依据 MLIP-free validation：
denoising loss、proposal recovery、reconstruction、finite rate、distance/volume
support 与固定 attempt generation success。MatterSim/CHGNet 结果不能反向选择
或重训 checkpoint。

任何阶段不得从同一 attempt 产生多个候选再筛选。模型调用、MatterSim 调用和
失败都必须逐 attempt 记录。

## 9. 统计与审计

- binary paired metrics：exact McNemar / paired exact test；
- ratios：Wilson interval；
- paired differences：10,000 次 attempt-level bootstrap；
- uniqueness 在每个 bootstrap 内重算；
- MP hull unknown、relaxation failure、generation failure 贡献 0；
- MP API 只在登录节点预取 reference，不用 Slurm；正式 evaluator 只读冻结 cache；
- 开发 panel、held-out 64、confirmatory 256 必须互斥；
- CHGNet 结果不能反向修改 MatterSim 配置；
- 所有 config、checkpoint、attempt panel、cache、output 和 audit 都记录 SHA256。

## 10. 当前可执行结论

本地现在可以执行 G0：配置、角色防火墙、资源检查、预检和单元测试。

下一项真实 GPU 动作是 G1 的 MatterSim A800 非科学 smoke。它需要先冻结
8-structure panel、finite-difference 容差和唯一 sbatch identity；在这些文件
及归档通过本地审计前，不提交作业。当前也不提交新的长 diffusion 训练。
