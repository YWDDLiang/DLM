# ICLR 主线重启：Shared Plan + Diffusion Language Model

日期：2026-07-28  
状态：active scientific plan  
取代：`20260728_r5c_reactivation_plan.md` 中原 Stage 3 以后路线

## 一句话主张

一个简短、可干预、不可变的 Crystal Plan 同时协调离散 diffusion language
model 的结构草稿和连续 equivariant diffusion 的几何细化，比“Plan 在产生草稿后
就消失”的强基线更有效，并且不依赖 MLIP guidance、搜索或样本筛选。

主模型：

```text
goal
  -> Planner
  -> shared Plan P
       -> exact-length DLM q(D | P)
       -> Plan-conditioned diffusion p(X | D, P)
  -> crystal X
```

恢复资产和完整负结果复盘见
`workstreams/r5c_reactivation_20260728/ALL_POST_R5C_RESULTS_REVIEW.md`。

## 科学问题

R5-C 已经证明：给定一个正确 Plan，exact-length DLM body 与 CrysLLMGen refiner
能得到很强的 conditional 结果。H1-A2 则是目前最强的 fully-de-novo
Planner→DLM 路径。两者之间的主要差距不是 parse 或 structure validity，而是
Planner 产生的 draft 落入较差的稳定性 basin。

本主线检验一个具体、可证伪的解释：

> 当 draft 不完全表达 Planner 的高层结构意图时，让同一 Plan 在 continuous
> diffusion 阶段继续可见，能否在保持化学组成、有效性和多样性的同时，提高
> first-attempt S.U.N.？

不是主张：

- 更长的自然语言推理必然更好；
- diffusion 比 AR/D3PM 普遍更好；
- MLIP reward 或 test-time search 带来提升；
- conditional R5-C 等于 autonomous generation；
- 任意 Plan-conditioned adapter 都能提升稳定性。

## 冻结主干

| 组件 | 冻结选择 | 原因 |
|---|---|---|
| Planner | H1-A2 epoch 2 | 最强 fully-de-novo 历史主干：strict 9.71%，meta 48.94% |
| body | R5-C exact-length constrained DLM | parse/graph/chemistry 最成熟；默认约束 ablation 最稳健 |
| parent refiner | original CrysLLMGen MP-20 checkpoint | 保留强 parent 与历史可比性 |
| Plan refiner init | R5-C shared-Plan S2 best adapter | 最强 direct Plan signal；只在 null 修复后使用 |
| evaluator | original A100 scripts + CHGNet in `diff_meets_diff` | 与 9.0/46.1、10.3/72.2 历史行一致 |

不从 PlanBridge/PlanV2 恢复。PlanV2 的 256-lineage identity gate 已经是机制负结果，
继续增加样本只会更精确地估计失败机制。

## Shared Plan 合同

一个 canonical Plan 产生两个带同一 hash 的只读 view：

```text
Plan
  ├── DLMPlanView
  │     formula, elements, counts, N,
  │     lattice family, exact body length and legal-token constraints
  └── DiffusionPlanCondition
        immutable formula/count/N mask + lattice-family strategy
```

首轮不加入 volume、formation energy、band gap、hull、coordinates、MLIP score 或
Materials Project 查询结果。composition、atom count 和 atom order 在 refiner 中
不可改变。

## 关键实现修正：single-trajectory algebraic null

旧 S2 adapter 的单步 null 误差很小，但旧 gold gate 中分别 replay 的 R0-parent
与 R1-null 完整 800-step trajectory 明显不同。源码复查表明 adapter 本身已经在
所有 learned operations **之后**施加 condition mask：

```text
adapter_delta = condition_present * learned_adapter(features)
film_delta    = condition_present * learned_film(features)
lattice_delta = condition_present * learned_lattice_residual(features)
```

`condition_present=0` 时，三个 delta 必须逐元素精确为零，而且当前
`force_null` 已直接调用 frozen parent decoder。旧漂移来自两次独立 CUDA
scatter/reduction replay 不保证位级确定性，微小差异经过 800 steps 放大，而不是
已证实的 residual 泄漏。

新协议禁止把 parent 和 exact-null 独立 replay 后比较终态。它必须：

1. 在 denoiser boundary 验证 `force_null` 直达 parent；
2. 只执行一次 frozen-parent reverse trajectory；
3. 将同一个不可变 `RefinedCrystalState` 同时登记为 R0 与 R1；
4. 仅对 matched/shuffled Plan 各自执行 conditional trajectory。

## 执行阶段

### Stage 0 — 本地实现和冻结测试

不申请 GPU，不训练。

1. 从 `llm_plan_diff` 移植 S2 PlanLite encoder/refiner adapter 的最小代码；
2. 固化已有 post-transform `condition_present` hard gate；
3. 加入 parent/exact-null single-trajectory alias API；
4. 为每个 FiLM block、lattice residual、组合 denoiser和 alias receipt 写单测；
5. 加载 S2 checkpoint，确认旧权重能在新协议下读取；
6. 固定 source/config/asset hash 和一个 terminal report 格式。

退出条件：

- null residual bitwise zero、`force_null` 直达 parent；
- parent/null 只消费一次 reverse noise schedule 且共享一个 state/hash；
- matched 条件下 gradient/path 仍非零；
- stock/no-adapter 路径不被改写；
- 所有资源配置满足 `CPU <= 8 * A800`。

### Stage 1 — 条件式 null-repair / mechanism-256

不训练、不生成新 draft。使用冻结 R5-C valid drafts，same draft、same atom order、same
reverse noise：

| Arm | DLM draft | diffusion condition |
|---|---|---|
| R0 | frozen R5-C | single stock-parent trajectory |
| R1 | same | exact-null semantic alias of the same R0 trajectory |
| R2 | same | repaired adapter + matched Plan |
| R3 | same | repaired adapter + deterministically shuffled Plan |

先用 32 条只做完整轨迹 parity；通过后再用一个与旧 gold-gate 身份不重叠的固定
256 条 panel 做 mechanism gate。旧 256 只允许用于工程回归，不作为新的确认性
统计样本。

Stage 1 gate：

- R0/R1 是同一不可变 state，coordinate/lattice hashes 必须完全相同；
- R2-R3 lattice-family hit 差至少 +5pp 且 paired 95% CI lower bound `> 0`；
- R2 对 R0 的 comp/struct/joint/COV 非劣，Wasserstein 恶化不超过冻结 margin；
- 所有 256 identities、Plan hashes、draft hashes 和 noise hashes 一一对应；
- 无 retry、replacement、repair、rerank、S.U.N. 选择或 MLIP 调参。

Stage 1 是 conditional mechanism gate，不是论文 headline。

### Stage 2 — H1-A2 fully-de-novo paired-256

冻结 H1-A2 Planner 和 R5-C body。每个 goal/seed 只产生一次 Plan 和一次 draft：

| Arm | Planner/body | refiner |
|---|---|---|
| F10 | H1-A2 Plan -> R5-C DLM | stock/no Plan |
| F11 | same Plan、same draft | repaired matched Plan |
| F1S | same Plan、same draft | shuffled refiner Plan |

F10/F11/F1S 共享 draft、atom order、reverse schedule 和随机张量。F1S 只改变 refiner
读到的 Plan identity。CrysLLMGen direct metrics 完成后再运行 original
A100-script/CHGNet S.U.N.。

Stage 2 promotion：

- all-attempt denominator 恰为 256；
- graph yield、composition validity、structure validity、novelty 和 uniqueness 对
  H1-A2/F10 不发生实质下降；
- F11-F1S Plan identity effect 为正且 CI 不跨零；
- F11-F10 strict S.U.N. 点估计为正；
- meta S.U.N. 对 F10 非劣；
- 不因失败样本缩小分母。

256 panel 只决定是否扩样，不选 checkpoint。

### Stage 3 — 只允许一次的 refiner 训练分叉

如果 Stage 1 的 repaired frozen S2 保留 Plan identity，但 Stage 2 因 de-novo draft
distribution shift 失败，允许一次小型 same-source adapter continuation：

- Planner 和 R5-C body 全部冻结；
- 从 repaired S2 初始化；
- 只训练 Plan encoder/FiLM/lattice residual；
- matched/null/shuffled 三项损失同时记录；
- exact-null 由结构 hard gate 保证，不作为可学习目标；
- 训练数据只来自 train split 和 H1-A2 train-side drafts；
- checkpoint 只按 validation denoising/identity objective 选择，不读 S.U.N.。

Stage 2 的实际结果满足这一分叉条件：F11-F1S 的 lattice-family hit
差约为 `+0.1988` 且 paired interval 不跨零，说明 Plan identity 机制仍在；
但 F10/F11 strict S.U.N. 均为 `15/256`，冻结 S2 对 H1-A2 draft
distribution 没有带来终端收益。

因此 Stage 3 固定为一次性、same-source 的小型延续训练：

- 对 27,136 条 H1-A2 train-side rich-Plan 行做 outcome-blind hash 选择，
  生成前冻结 2,048 train / 256 validation；
- 每个 source identity 仅运行一次 frozen full-Plan R5-C body，不补失败样本；
- train 与 validation 各自要求至少 95% same-source valid paired coverage；
- 从 S2 step-250 best adapter 初始化，最多 500 updates，LR `1e-4`，
  每 50 updates 验证；
- checkpoint 必须同时满足：H1-A2 matched 严格优于 shuffled、旧 R5-C
  matched degradation 不超过 2%、H1-A2 matched 相对初始化改善至少 0.1%；
- 在满足上述门的 checkpoint 中只选 H1-A2 matched validation loss 最小者；
- exact-null 保持结构 hard gate，S.U.N./CHGNet/MLIP/DFT/MP API 以及任何
  generation metric 均不参与训练或 checkpoint 选择。

若没有 eligible checkpoint，分叉终止；若通过，只允许另行冻结一个 paired-256
确认实验，不自动提交。

若 Stage 1 本身不能恢复 matched-shuffled identity effect，则停止这个 refiner
architecture，不通过延长训练掩盖机制失败。

### Stage 4 — full-1000 与三 seed

只有 Stage 2 通过才执行：

1. 固定一个 model/config；
2. 完成 1,000 all-attempt paired F10/F11/F1S；
3. 再补三个 Planner seeds 的预注册重复；
4. 报告 CrysLLMGen、strict/meta S.U.N.、novelty/uniqueness、prototype/chemistry
   分层和 paired bootstrap；
5. conditional R5-C 只作为上界，不参与模型选择。

正式 promotion 目标：

- F11 strict 高于 H1-A2 历史 adjusted 9.71%，并且 paired F11-F10 lower CI `> 0`；
- meta 对 H1-A2 历史 adjusted 48.94% 非劣（margin 2pp），最好同时提高；
- comp/struct/joint、coverage、novelty/uniqueness 不以筛选换提升；
- 三个 Planner seeds 方向一致。

conditional R5-C 的 meta 74.38% 是 oracle Plan 上界，不是 fully-de-novo 的硬性
首轮 gate。

## 论文里程碑

### 现在即可写

- motivation、method、shared Plan contract；
- R5-C conditional 与 H1-A2 的历史基线；
- 全部 negative-result map；
- exact-null 与 same-draft/same-noise 因果设计；
- evaluator 与 no-search protocol。

### 可以形成 ICLR 主结果表

需要 Stage 2 paired-256 机制与质量共同通过，并完成一次冻结 full-1000。

### 可以冻结投稿 claim

需要三 seed fully-de-novo 结果、正的 Plan identity CI、相对 F10 的 strict 提升、
meta/validity/diversity 非劣，以及信息匹配的 shuffled/null controls。无需引入
MatterSim、RL、DFT 或 test-time search。

## 时间与审计策略

效果优先，保留最小可复现记录：

- 一份 immutable config；
- source/checkpoint/evaluator SHA；
- exact command、环境、resources、seeds、exit；
- one terminal report；
- CrysLLMGen 与 S.U.N. reports；
- 不为轮询生成侧车，不做无关 Gate-A 全仓审计。

每个阶段失败即按预注册分叉处理，不自动重投，不展开新模型家族。
