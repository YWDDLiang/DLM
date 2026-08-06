# H1 Body-DLM Mask-Aware RL 最小提案 V1

状态：`design_only_no_execution_authorization`

日期：2026-08-04

投稿时钟：

- ICLR 2027 摘要截止：2026-09-18 AOE；
- ICLR 2027 全文截止：2026-09-25 AOE；
- 从本文日期起分别只剩 45/52 天。

适用锚点：

```text
frozen P0 Planner
  -> frozen B0/R5-C exact-length Body DLM
  -> D1 group schedule
  -> frozen CrysLLMGen model_494 exact800
  -> Direct + strict/meta S.U.N.
```

本文是第一轮 mask-aware RL 主提案。它只审计并设计，不授权训练、生成、
refinement、评测、checkpoint selection、promotion 或自动下游。

---

## 0. 一句话判定

**当前决策是 `KEEP_BACKUP`：RL 是有明确 CUT 条件的可选增强，不是 ICLR
主线的必经路径。**

2026-08-15 前完成 first-64 只获得继续资格；2026-08-22 前通过 paired-256
后才有资格竞争共同主线；2026-08-31 前完成 `4×256` 独立 panel 与共同评测
确认后，才可正式进入摘要。任一硬门失败都回到冻结的
PlanGraph-DLM-refiner 主线，RL 移到投稿后工作；不得为了保住 RL 而同时改
token 表示、support、safe-axis、Planner 或 evaluator。

推荐的最小科学方案是：

> **冻结 D1 的 group 边界和全部特殊-token合法支持，只把 group 内 greedy
> reveal 改为 K=1 Plackett–Luce；把一次 diffusion transition 明确定义为
> “当前组全部 token candidates + 被提交的位置”，联合训练 token 与 position
> 两个概率项。只维护一个 RL LoRA，先用 pre-model_494 低保真标签，随后用
> 随机抽取的 post-model_494 标签作无偏 control-variate 校正，最后用 B-only
> 收尾。**

正式名称建议：

```text
Constrained D1-PL Mask-Aware Policy Optimization
with Randomized Pre/Post-Refiner Rewards
```

该方案的论文价值不在于“又做一次 GRPO”，而在于三个紧密耦合的贡献：

1. exact-length crystallographic DLM 上、合法特殊-token支持内的
   joint token-position policy；
2. 保留 D1 因果/约束边界的 group-local K=1 probabilistic remasking；
3. pre/post-refiner 随机多保真 reward，在不训练第二个长期模型的情况下控制
   exact800 标签成本和 reward bias。

---

## 1. 审计输入与不可改写的事实

### 1.1 已完整审计的本地材料

- `H1_BODY_DLM_COMPLETE_PROTOCOL_AND_RL_DESIGN_V1.md`；
- `H1_BODY_SPECIAL_TOKEN_COVERAGE_AUDIT_V1.md`；
- `H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md` 中 R03E/G/H；
- `scripts/llada_trace_rl.py`；
- `crystal_dlm/rl_utils.py`；
- `crystal_dlm/llada_generation.py`。

三份历史 TraceRL trainer 完全相同：

```text
scripts/llada_trace_rl.py
legacy_dlm_r5c/scripts/llada_trace_rl.py
workstreams/r5c_reactivation_20260728/baseline/scripts/llada_trace_rl.py
SHA-256 = 6de9cd40413404ed5492ccaff6130d7e2c27dba99ead87a998c72a1df7ef1d00
```

`crystal_dlm/rl_utils.py` 与 reactivation baseline 也完全相同：

```text
SHA-256 = 7f2a844741b4ff153732359d007b7f4accc194395caa4e88c68b61fa5d40ebb2
```

### 1.2 当前 DLM 已经强在哪里

R03D body-only：

| Arm | Body success |
|---|---:|
| D1 | 246/256 |
| safe-axis | 248/256 |

R03E 四个 CUDA process repeats 的 pooled raw Direct：

| Arm | Generation | Composition | Structure | Joint |
|---|---:|---:|---:|---:|
| D1 | 984/1024 | 848/1024 | 982/1024 | 846/1024 |
| safe-axis | 992/1024 | 852/1024 | 989/1024 | 851/1024 |

在 refine-success denominator 上，conditional structure validity 已经是
`99.7967%` 与 `99.6976%`。真正 post-refiner structure-invalid 只有 2/3 个。
因此首个 RL 不能以“修复大量结构非法”为故事；它要优化的是 conditioned on
Plan 的 refiner basin、stability 分布、novelty/uniqueness，而不是已近饱和的
parser/structure gate。

### 1.3 R03G/H 给 reward 的硬教训

completed-snapshot lower-bound pooled：

| Arm | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|
| D1 | 99/1024 = 9.67% | 523/1024 = 51.07% |
| safe-axis | 117/1024 = 11.43% | 496/1024 = 48.44% |
| Delta | +1.7578 pp | -2.6367 pp |

R03H 将变化分解为：

```text
strict       +18
meta-only    -45
above-meta   +36
ineligible    -9
unknown        0
```

strict `+18` 中 `+16` 来自 finite-hull crossing，meta `-27` 全部来自 finite
`0.1 eV/atom` threshold crossing。72 个 residual unknown 完全配对，对两个
效应贡献均为 0。

因此：

- strict binary reward 不能单独训练；
- meta 必须同时进入 reward shaping 和 checkpoint/promotion 硬门；
- safe-axis 不能作为本轮 RL 起点；
- 同一个 256 scientific ledger 的四次 CUDA repeat 不能冒充四个独立 panel。

### 1.4 composition 不属于 Body RL action

`N` 和 element slots 由 Planner 预填。同一个 Plan 的 `G` 个 rollout 拥有相同
composition。因此 composition reward 在 Plan group 内是常量，减去 group
baseline 后 advantage 为 0。

R03E 中 safe-axis 的 35 个 successful-but-composition-invalid：

- 24 个 Planner charge failure；
- 4 个 Pauling failure；
- 7 个其他 Direct-invalid。

正式处理：

- `comp_valid` 只作 Planner quality、分层统计和 all-attempt endpoint；
- 不进入 Body advantage；
- RL train Plan pool 可以在 rollout 前按冻结的 Planner-only
  chemistry flag 选择可训练 Plan，不能按 Body/refiner outcome 筛选；
- final 256/1024 仍使用完整 raw P0 all-attempt denominator。

### 1.5 Gate −1 不是可选项

held-out `9,046` records 中：

| 指标 | 数值 |
|---|---:|
| 定义特殊 token | 2,481 |
| held-out seen | 1,437 |
| held-out unseen | 1,044 |
| numeric stochastic token | 2,343 |
| numeric held-out unseen | 1,013 |
| 三个 length family unseen | 953 |

这不是 train-unseen 结论，但足以证明正式 RL 前必须恢复并扫描 train
`27,136`，计算 legal support 中 unseen/rare probability mass。

本提案的硬边界：

> 若 Gate −1 认为需要 axis sharing、新 token、embedding/output-row 训练或
> 其他 representation change，则 RL 从 ICLR 时间表中 CUT。表示变化单独做，
> 不能与 RL 同时训练。

若 Gate −1 决定保留原 support，或完成一个独立 support-only 决策，则必须先
冻结唯一 support SHA，之后 RL run 内不可改变。

---

## 2. 最新一手工作带来的设计选择

| 一手工作 | 可采用的结论 | 本项目不照搬的部分 |
|---|---|---|
| [Mask-Aware Policy Gradients for Diffusion Language Models](https://arxiv.org/abs/2607.15200), COLM 2026 | MDLM action 有 token 与 remask/unmask position 两阶段；greedy top-k 不能给完整 PG；Plackett–Luce 可给 position log-prob | 论文主实验是 8×H100、LoRA r128、G=6、约 6k–8k steps、12 inner updates、无 KL；本项目数据少、judge 可被攻击，不能复制这些数值 |
| [DCoLT/LLaDOU](https://arxiv.org/abs/2505.10446) | unmask order 是可优化 action，Plackett–Luce 是自然排序模型 | 不引入独立 UPM 和新的全局生成顺序 |
| [ESPO](https://arxiv.org/abs/2512.03759), ICLR 2026 | diffusion RL 应从 sequence/trajectory 级理解，importance ratio 需长度归一化，KL estimator 要稳健 | 本项目真实 transition 很短且可完整记录，不需要先上 ELBO proxy |
| [Learning Unmasking Policies](https://arxiv.org/abs/2512.09106) | 小 controller 可以单独学习 unmask policy | 第一枪不新增 controller；否则同时改变 parameterization 与 policy |
| [PLaID++](https://arxiv.org/abs/2509.07150) | 训练 MLIP 与评测 MLIP 分离，并用 DFT 验证；stability/novelty 优化易 mode collapse | 不改 Wyckoff/token 表示，不做 iterative DPO |
| [CRYSTAL](https://openreview.net/pdf/94d95333b625bc19463eca098ff60038d639d590.pdf) | validity/stability/novelty/diversity 要共同监控；单目标会 reward hijacking；更大 rollout group 有助 diversity | 其 AR-GRPO、128 rollouts 和 composition reward 不适合 frozen-Plan Body |
| [Chemeleon2](https://www.nature.com/articles/s42256-026-01262-4), Nature Machine Intelligence 2026 | continuous stability、creativity、marginal diversity reward 能缓解 novelty–stability 冲突 | 不引入新的 VAE latent space，不把 batch-dependent diversity 变成主 endpoint |
| [MatInvent](https://arxiv.org/abs/2511.03112) | diffusion trajectory RL 和多保真/昂贵属性标签有现实价值 | 首版不做长 experience replay，避免 stale-policy 与 resume 复杂度 |

Mask-Aware PG 的实验实现使用 max-logit position score；其理论定义又以 sampled
token log-likelihood 为 score。当前冻结 H1 sampler 实际排序的是 sampled-token
的**未温度化 confidence**。为了保持单变量和 `tau_pos -> 0` 的 baseline
连续性，本项目必须沿用当前 sampled-token confidence，不能为了和论文实现
逐字一致而再改一个 score factor。

---

## 3. 当前 TraceRL 的最终判定

### 3.1 正式实验：NO-GO

历史 TraceRL：

1. 从 final sequence 和 `step_map` 重建状态，不保存真实 rollout state；
2. 缺 `step_map` 时直接构造 `range(answer_len)`；
3. 默认 `trace_shrink=8`，会合并真实 transition；
4. old log-prob 在训练前用加载后的当前模型重算，不是 rollout 在线记录；
5. 对 128,830 全词表 raw `log_softmax`，没有 legal mask 后重归一化；
6. 没有 rollout temperature `0.7`；
7. 没有 position/remasking probability；
8. 没有保存当前 group 全部 sampled candidates；
9. advantage key 可能使每个 unique Plan 只有一个 rollout，直接得到零梯度；
10. `k3` selected-token KL 不是 frozen-reference joint KL；
11. checkpoint 只保存 model/tokenizer，缺 optimizer/scheduler/scaler/RNG/cursor，
    不是 exact resume；
12. 默认 reward 是 formula/SMACT/duplicate，服务于旧 fixed-slot composition
    generation，不适合 composition 已冻结的 Body。

### 3.2 仅可复用

- DDP/LoRA/optimizer 外壳；
- JSONL、日志和 checkpoint 命名习惯；
- group reward utilities 的基础结构。

rollout、legal probability、joint policy、reward、replay、resume 和 gates
必须新写，不能在旧脚本上打一个小补丁后声称 principled RL。

---

## 4. D1 K=1 PL joint token-position policy

### 4.1 不改变的边界

```text
prefilled N/elements
  -> one lattice group
  -> one all-X group
  -> one all-Y group
  -> one all-Z group
```

保持：

- exact `7+4N`；
- token IDs、tokenizer、B0 embedding/output head；
- `tau_tok=0.7`；
- CFG `0.0`；
- schema、zero-length、lattice-volume、duplicate-coordinate masks；
- lattice→X→Y→Z 宏观顺序；
- Planner、prompt、Plan、refiner、evaluator；
- 每步 K=1 committed position。

禁止：

- safe-axis；
- cross-group PL；
- mixed-axis group；
- token/support change；
- K>1；
- future group 的无效 candidate sampling。

### 4.2 状态

令当前 D1 group 内仍 masked 的位置为 `I_t`：

```text
S_t = (
  frozen Plan and prompt,
  exact partial sequence,
  current group/step,
  each position's legal special-token support,
  tokenizer/model/support/constraint/schedule identities
)
```

### 4.3 token candidate policy

对 `i in I_t`，应用全部 schema/dynamic masks 后：

```text
q_tok_theta(y | S_t, i)
  = softmax(masked_logits_theta(S_t, i) / 0.7)[y]

Y_i ~ q_tok_theta(. | S_t, i)
```

只在当前 group 采 candidate。非法 token 概率严格为 0；empty legal support
立即失败，无 fallback。

### 4.4 position policy

为精确延续当前 low-confidence greedy score：

```text
c_i = softmax(masked_logits_theta(S_t, i))[Y_i]
v_i = log(c_i)

p_pos_theta(J=j | Y, S_t)
  = exp(v_j / tau_pos)
    / sum_{i in I_t} exp(v_i / tau_pos)
```

然后：

```text
J ~ p_pos_theta
x_{t+1,J} = Y_J
other positions remain MASK
```

`tau_pos -> 0` 趋近当前 sampled-confidence greedy top-1。若出现相同 score，
必须冻结 tie policy 和 stateless uniform role，不能依赖 CUDA `topk` 的隐式
顺序。

### 4.5 扩展 action 与精确概率

一步 action：

```text
a_t = ( {Y_i : i in I_t}, J )
```

精确扩展 action probability：

```text
pi_theta(a_t | S_t)
  = [ product_{i in I_t} q_tok_theta(Y_i | S_t, i) ]
    * p_pos_theta(J | Y, S_t)
```

trajectory：

```text
log pi_theta(tau)
  = sum_t [
      sum_{i in I_t} log q_tok_theta(Y_i | S_t, i)
      + log p_pos_theta(J_t | Y, S_t)
    ]
```

这是首版需要审计的“joint”。只对 committed token 算 log-prob 会漏掉影响
position decision 的未提交 candidates；把未来 group 的无效 candidates 算入
likelihood 则会加入不影响 transition 的随机量。两者都不允许。

### 4.6 需要承认的数值事实

当 `N=20` 时只有 66 个 committed actions，但若每一步记录当前 group 全部
candidates，candidate action 数约为：

```text
6+5+...+1 + 3 × (20+19+...+1) = 651
```

因此 raw trajectory ratio 可审计但可能高方差。训练使用预注册的两个 channel
归一化 surrogate：

```text
ell_tok = mean over all logged candidate-token log-probs
ell_pos = mean over all non-singleton position log-probs
```

token 与 position 各自 clip，再等权进入 loss；同时保存未归一化 exact joint
log-prob 和 exact joint ratio 作审计。必须把这写成稳定化 surrogate，不能把它
误称为最终序列 marginal likelihood。

---

## 5. 最小优化器

### 5.1 参数边界

1. 加载 frozen base 与 B0；
2. 验证 128,830 vocabulary、2,481 token map 和三份 tokenizer SHA；
3. 冻结 B0，包括 `wte` 与 `ff_out`；
4. 新增 RL-only LoRA：

```text
r=8
alpha=32
dropout=0.05
targets=q_proj,k_proj,v_proj,ff_proj,up_proj
modules_to_save=None
```

5. trainable-parameter manifest 必须只出现新 LoRA。

不能把 Mask-Aware PG 的 `r=128` 当默认值。当前 B0 adapter 约 6.39 GB，
历史 `modules_to_save` 包含完整 embedding/output head；误解冻会把实验从
“小 LoRA RL”变成另一个模型因子。

### 5.2 group 与 advantage

每个 Plan 采 `G=8`：

```text
A_g = R_g - mean_{h != g}(R_h)
```

使用 leave-one-out baseline，不除 group standard deviation。原因：

- strict 约 10%，小 group 的标准差经常接近 0；
- refiner/judge 有过程噪声；
- CRYSTAL 表明小 group 容易失去 diversity 协调，但 exact800 不允许直接复制
  128 rollouts；
- 本项目用 continuous meta-aware stability shaping 增加 group 内密度。

`zero_variance_group_rate` 必须报告；64 阶段若少于 25% Plan groups 有非零
advantage，直接 CUT，不在结果后临时改 reward。

### 5.3 更新方式

最小训练不引入 critic，不使用长期 replay：

- fresh on-policy rollout；
- behavior actor 在线保存 old log-prob；
- 每个 rollout batch 最多 2 个 inner updates；
- actor version 最多滞后一版；
- RLOO/group-relative advantage；
- token/position 两个 normalized ratios 分别 clip `[0.8,1.2]`；
- frozen B0 reference KL；
- stale trace 在 ESS 不过门时立即丢弃。

论文中的 12 inner updates、reference 每 64 steps 同步和 no-KL 配置不适用于
当前昂贵科学 judge。这里宁愿更新少，也不允许 judge hacking 被多次放大。

### 5.4 ratio、KL 与 ESS

对同一 logged action：

```text
Delta_tok =
  sum_{t,i in I_t} [
    log q_new(Y_i | S_t,i) - log q_old(Y_i | S_t,i)
  ]

Delta_pos =
  sum_t [
    log p_new(J_t | Y,S_t) - log p_old(J_t | Y,S_t)
  ]

rho_joint_exact = exp(Delta_tok + Delta_pos)
```

同时计算：

```text
rho_tok_norm = exp(Delta_tok / number_of_candidates)
rho_pos_norm = exp(Delta_pos / number_of_nontrivial_steps)
```

初始 hard gates：

- mean legal-support categorical token KL `<=0.01 nat`，`>0.02` 停止；
- mean categorical position KL `<=0.01 nat`，`>0.02` 停止；
- frozen-B0 trajectory KL per logged action `<=0.05 nat`；
- normalized transition ESS `>=0.70`；
- trajectory ESS `>=0.50`；
- clip fraction `<=20%`，任一 Plan group `>30%` 停止；
- 任一 NaN/Inf、empty support、ratio overflow 立即停止。

按 `N`、lattice/X/Y/Z、Plan chemistry family、rare-token bucket 报告 KL、
ESS、entropy、clip fraction 和 max ratio。

---

## 6. Reward、constraint 与 evaluator isolation

### 6.1 保存 reward vector，不只保存 scalar

```text
z = (
  body_completion,
  exact_length,
  plan_match,
  pre_structure_valid,
  pre_refiner_margin,
  post_refiner_completion,
  post_structure_valid,
  E_hull,
  strict,
  meta,
  novelty,
  symmetric_uniqueness,
  refiner_displacement,
  reward_judge_identity
)
```

`comp_valid` 另存为 `Q_plan`，不进入 Body advantage。

### 6.2 两个 fidelity

低保真 A：

```text
raw DLM proposal -> frozen cheap judge
```

必须称 `pre-model_494 proxy`。若 A 内含 cheap MLIP relaxation，就不能称
“完全未 refine”。

高保真 B：

```text
same raw DLM proposal
  -> frozen model_494 exact800, batch1, frozen seed
  -> same reward semantics
```

B 是部署一致标签。

### 6.3 meta-aware stability utility

首版冻结：

```text
S(E_hull) =
  1.00                                      if E_hull <= 0
  0.50 + 0.25 * (0.10-E_hull)/0.10         if 0 < E_hull <= 0.10
  0.00                                      if E_hull > 0.10
  0.00                                      if unknown/judge failure
```

meta-only 至少保留 0.5，且越接近 hull 奖励越高。这样避免只用 strict binary
把 mass 从 meta-only 推到 above-meta。

### 6.4 CRYSTAL-inspired、但适合 frozen Plan 的 scalar

先在固定 cohort 上用 frozen StructureMatcher 做对称 clustering。若一个结构
属于大小为 `k` 的重复簇：

```text
U_cluster = 1/k
N = frozen novelty indicator
V = exact_length AND plan_match AND structure_valid

R_F = V * N * U_cluster * S_F(E_hull)
```

pipeline/body/refiner failure 使用预注册值 `-0.25`；unknown hull 为 0，不重试。

相对 CRYSTAL 的修改：

- 去掉 composition reward；
- 去掉 sequential first-seen uniqueness；
- 不用 strict-only；
- 不用 128 rollouts；
- diversity 作为 symmetric cohort property 和 validation guard，不让
  batch order 决定主 reward。

如 64 阶段该 reward 仍过稀，结论是当前在线 RL 不可行，直接 CUT；不能在
看过 endpoint 后继续调权重。

### 6.5 三层 evaluator firewall

| 角色 | 最小合同 |
|---|---|
| `E_train` | frozen A/B reward judge、checkpoint、hull snapshot、cache；只看 train rollouts |
| `E_gate` | 独立 architecture 或至少独立 checkpoint；只用于 64/256 gate，不反向调 reward |
| `E_final` | 历史 Direct + CHGNet/MP-hull 保持可比性，并加独立 MLIP/DFT audit；final panel 只开启一次 |

首选可执行映射：

```text
E_train : one frozen MLIP family
E_gate  : a different MLIP family
E_final : historical CHGNet/MP snapshot + stratified DFT audit
```

PLaID++ 使用 eqV2 建偏好数据、eSEN 做生成评测，并用 1,000 个 DFT 样本验证
MLIP；这正是本项目需要的 isolation 原则。具体选 eqV2/eSEN/当前 CHGNet 中
哪一个作哪一层，必须在 first-64 前依据本地可用 checkpoint 冻结，不能在看到
RL 结果后互换。

`E_gate` 的合同、checkpoint 与阈值最迟在 2026-08-10 冻结，并须在
2026-08-12 前可运行。若到 2026-08-15 仍未通过 first-64 smoke，则：

- 可以继续做工程 trace；
- 不得做科学 RL claim；
- ICLR RL 主线 CUT。

若训练 reward 使用 CHGNet，final 的 CHGNet/MP 只能叫“历史口径上的被优化
proxy”；科学确认必须依赖独立 MLIP/DFT 方向一致。

### 6.6 约束与 reward 分工

以下是 hard constraint，不靠 reward 学：

- exact length；
- Plan count/element 不变；
- legal token；
- no duplicate coordinate；
- positive lattice volume；
- D1 group order；
- no retry/repair/replacement/filter/rerank。

reward 只在合法 proposal 之间排序。任何 hard constraint regression 都是
程序/策略失败，不因 strict 上升而豁免。

---

## 7. 一个多保真模型，还是两个模型

### 7.1 当前判定

**一个长期 LoRA：GO。两个长期 LoRA：NO-GO。**

如果部署始终经过 `model_494 exact800`，A 与 B 是 label fidelity，不是两个
产品。维护 A-LoRA 与 B-LoRA 会：

- 近似翻倍 rollout/selection/label 成本；
- 增加 checkpoint selection 自由度；
- 混淆“fidelity”与“训练步数”；
- 最终仍需选一个 post-refine 最优策略。

只有未来存在真正的“无 model_494 在线部署产品”时，才维护第二个 A 模型。

### 7.2 单模型 curriculum

若 64 的 A/B calibration 通过：

```text
B0
  -> A-only warm-up
  -> randomized A/B mixed-fidelity
  -> B-only final
```

A-only checkpoint 只是中间状态，不是长期产品，也不参与 final 多选一。

### 7.3 最小无偏多保真校正

对每个 mixed-fidelity rollout：

```text
Z ~ Bernoulli(p),  p=0.5 fixed before rollout

R_MF =
  R_A + Z/p * (R_B - R_A)
```

因为 B inclusion 是随机的且 `p` 已记录：

```text
E[R_MF | proposal] = R_B
```

这是以 A 为 control variate 的 Horvitz–Thompson 校正，不需要再训练一个
surrogate。它比“只 refine A 预测最好的 proposal”更可审计，后者会产生
survivorship bias。

禁止：

- 根据 A score 决定 `Z`；
- 忘记保存 inclusion probability；
- 对 `R_MF` 事后 clipping；
- 缺 B label 时重抽 proposal；
- judge failure 后补跑到成功。

### 7.4 A/B calibration gate

first-64 实际生成 64 attempts/arm，因此可获得最多 128 个 A/B paired labels。
通过条件：

- finite paired labels `>=80`；
- continuous score Spearman point `>=0.30`；
- B-meta AUROC `>=0.60`；
- top-A quartile 的 B-meta rate 至少比 bottom quartile 高 `10 pp`；
- strict positives `>=8` 时，B-strict AUROC `>=0.60`；
- `strict/meta-only/above-meta/ineligible` transition 无明显反向排序。

另对固定 16 proposals 做第二个 CUDA process 的 exact800：

- 只估计 process noise，不作为新 science seeds；
- meta/pipeline label disagreement不得超过 `2/16`；
- 若超过，A/B curriculum CUT；不得通过重复直到一致来制造标签。

若相关性失败：

1. A-only 与 mixed-fidelity 路径 CUT；
2. 只允许一个 B-direct LoRA，B training labels cap 512；
3. 若 B-direct 不能在总预算与 2026-08-22 之前完成，整个 RL CUT。

---

## 8. Label 与 A800 预算

### 8.1 最小训练 curriculum

`G=8`，所有 actor batch 使用未重复的 train Plan groups：

| Phase | Actor versions | Plans/version | Rollouts | A labels | B labels | Inner updates |
|---|---:|---:|---:|---:|---:|---:|
| A warm-up | 4 | 8 | 256 | 256 | 0 | 1–2 |
| mixed, `p=0.5` | 4 | 16 | 512 | 512 | 256 expected/exact ledger target |
| B-only final | 2 | 16 | 256 | 256 diagnostic | 256 | 1–2 |
| 合计 | 10 | 128 Plan groups | 1,024 | 1,024 | 512 | 最多 20 optimizer steps |

`Z` ledger 应预先构造为每个 mixed actor version 恰好一半 B labels，同时在
Plan/ordinal 内随机排列，保证预算固定而非仅在期望上固定。

训练 schedule 预注册后只保留 terminal checkpoint。不要用同一个小 validation
panel 在十个 checkpoints 中挑最好者。

### 8.2 各阶段 label 预算

| Stage | Body proposals | A labels | B exact800 labels |
|---|---:|---:|---:|
| R0 | synthetic/fixed trace | 0 | 0 |
| paired-32 | 64 | 0 或 cheap diagnostic | 0 |
| first-64 | 128 + 16 process repeats | 128 | 144 |
| RL train | 1,024 | 1,024 | 512 |
| paired-256，RL vs PL | 512 | 512 diagnostic | 512 |
| final `4×256`，RL vs PL | 2,048 | optional | 2,048 |

不在 core budget 中增加 final greedy 第三臂。当前 frozen greedy D1 用作历史/
secondary context；若要在新 1,024 panel 上重跑第三臂，必须另行授权约
`+24 A800 GPUh`。

### 8.3 A800 hard cap

现有 exact800/batch1 allocation 上界：

```text
256 B labels <= 6 A800 GPUh
```

因此：

| Stage | model_494 上界 | 含 Body/backward/independent judge 的 stage cap | 累计 cap |
|---|---:|---:|---:|
| Gate −1/R0 | 0 | 0 | 0 |
| paired-32 | 0 | 2 | 2 |
| first-64 | 3.4 | 6 | 8 |
| train + paired-256 | 24 | 32 | 40 |
| final paired `4×256` | 48 | 56 | 96 |

**RL 投稿路径最大累计预算：96 A800 GPUh。**

DFT CPU/node 预算另计，但必须在 2026-08-15 前确认资源。任何阶段实际 wall/GPU
time 超出预估 25%，先停止扩容并重估；累计达到 cap 后不得用“只差一点”为由
继续。

---

## 9. 时间表与门禁

### 9.1 硬时间表

| 阶段 | 目标日期（CST） | 最迟日期（CST） | 超时后 |
|---|---|---|---|
| 只读审计、设计与一句话 claim | 2026-08-07 | **2026-08-07** | 不再引入新 RL 架构 |
| Gate −1 support/coverage 决策 | 2026-08-09 | **2026-08-10** | RL 从 ICLR 路径 CUT |
| R0 tests + exact resume | 2026-08-09 | **2026-08-10** | RL 从 ICLR 路径 CUT |
| paired-32 terminal | 2026-08-11 | **2026-08-12** | first-64 不得启动 |
| first-64 terminal | 2026-08-14 | **2026-08-15** | RL 退出 ICLR 主线 |
| RL train terminal | 2026-08-18 | **2026-08-19** | paired-256 不扩 |
| paired-256 terminal | 2026-08-21 | **2026-08-22** | 不启动 `4×256` |
| final `4×256` + independent audit | 2026-08-29 | **2026-08-31** | RL 不进摘要主结论 |
| common evaluation/science freeze | 2026-09-05 | **2026-09-05** | 不再启动 principal factor |
| paper/supp/repro package only-fix | 2026-09-12 | **2026-09-12** | 仅写冻结证据 |

Time-to-first-64：

```text
target = 10 calendar days (2026-08-14)
hard maximum = 11 calendar days (2026-08-15) from 2026-08-04
```

### 9.2 Gate −1：零训练

必须完成：

- train/validation/test、R03D/E generated token coverage；
- fixed-panel B0 legal probability mass；
- unseen/rare emission 与 endpoint attribution；
- frozen tokenizer/token map/embedding identity；
- 唯一 legal-support SHA。

通过：

- 保留原 support，或一个独立 support-only 决策已结束；
- 不需要 token/representation 同步训练。

CUT：

- train corpus 无法在 8 月 10 日前恢复；
- 需要 axis sharing/new token/row-level embedding training；
- support 仍会在 RL 中动态修改；
- unseen/rare mass 可能构成主要 exploit，但无法先隔离。

### 9.3 R0：无科学 GPU

必须全通过：

- manual constrained log-softmax；
- PL K=1 probability sum=1；
- `tau_pos -> 0` 与当前 score ranking 一致；
- legal-mask parity；
- group/axis/exact-length invariant；
- all-candidate trace；
- new=old：ratio=1、KL=0；
- fixed behavior replay；
- uninterrupted vs interrupted exact resume；
- batch/rank/order-independent ordinal RNG；
- empty support fail closed。

任一失败都不允许用真实 32 调试。

### 9.4 paired-32：工程，无训练

比较：

```text
frozen B0 + D1 greedy
frozen B0 + D1 K=1 PL
```

使用 stateless RNG，使两臂的 `(ordinal, step, position, token-role)` candidate
uniform 可配对；position uniform 使用独立 role。

`tau_pos` 候选只允许：

```text
{0.05, 0.10, 0.20, 0.50}
```

选最小且同时满足：

- median normalized position entropy 在 `[0.02, 0.25]`；
- 与 greedy position agreement `>=80%`；
- finite、nonzero entropy；
- 全部工程 gate 通过。

然后冻结。不得按 S.U.N. 选择。

paired-32 hard gate：

- exact length 100%；
- Plan match 100% among body outputs；
- illegal/cross-group/mixed-axis/Z-before-XY 为 0；
- no new failure；
- PL body completion `>= greedy -1/32`；
- duplicate-coordinate 不增加；
- trace/replay/resume 全过。

### 9.5 first-64：sampler + reward calibration，无训练

设计：

- 64 attempts/arm；
- 8 frozen train-calibration Plans × `G=8`；
- greedy vs PL-noRL；
- 两臂所有 proposals 均取 A/B labels；
- 另选 16 个 proposal 做第二 CUDA process；
- 独立 `E_gate` smoke。

hard gate：

- PL completion `>= greedy -1/64`；
- post-refiner structure 不差超过 2 pp；
- exact/Plan/group/duplicate gate 全过；
- 至少 25% Plan groups 有 nonzero reward advantage；
- A/B calibration 通过第 7.4 节；
- refiner process-noise gate 通过；
- reward scalar、threshold、judge 和 `tau_pos` 从此冻结。

first-64 是统一时间框架所要求的 **paired-64 机制信号**，不是 efficacy
结论：它证明 probabilistic reveal、joint trace、reward variance 和 A/B
calibration 至少可工作，但不因 strict 偶然增加而跳过 256。

### 9.6 RL train + paired-256

若 A/B 通过，执行一个 LoRA 的 A→mixed→B curriculum；否则执行一个
B-direct fallback。不得同时训练两个完整候选再挑赢家。

paired-256：

```text
candidate terminal RL
vs
frozen B0 D1-PL no-RL control
```

要求：

- fresh 256 P0 attempts；
- 两个独立 128 scientific-seed blocks；
- common-random-number ledger；
- 两臂都走 model_494 exact800；
- raw all-attempt；
- historical evaluator + independent `E_gate`；
- no checkpoint reselection。

晋级 `4×256`：

- strict point delta `>0`；
- 两个 128 blocks 的 strict 方向一致；
- meta point delta `>=0`；
- completion/structure loss `<=2 pp`；
- novelty、uniqueness 各不下降超过 5 pp；
- reward proxy 与 independent gate 同方向；
- KL/ESS/clip/entropy 全通过；
- no new failure；
- rare/unseen emission 不上升到 Gate −1 的 hard threshold。

若 proxy 上升但 independent gate 下降，判定 reward hacking，立即停止。

### 9.7 final `4×256`

不是四个 CUDA repeats，而是：

```text
4 independent scientific panels
× 256 raw P0 attempts
× 2 arms (RL, no-RL PL)
```

每个 panel 内做 paired common-random-number，panel 之间科学 seed 独立。

确认门：

- strict：至少 3/4 panel 为正，mean `>0`，hierarchical paired bootstrap
  95% CI lower `>=0`；
- meta：至少 3/4 panel 非负，mean `>=0`，one-sided 95% lower `>-2 pp`；
- structure noninferiority lower `>-2 pp`；
- novelty/uniqueness 不破预注册 guard；
- no new failure；
- independent evaluator 方向一致；
- DFT audit 不出现反向结论。

建议 DFT audit 预注册 192 个结构：

- 64 个 proxy-strict/strict-discordant strata；
- 64 个 meta-only strata；
- 64 个 above-meta valid strata；
- arm/panel 平衡，固定随机种子和 inclusion probability。

DFT 只作独立 audit，不把这 192 个条件样本直接当 raw S.U.N. 分母。

未通过：

- checkpoint 标记 `diagnostic_only`；
- 不 promotion；
- 不自动修改 reward 重训；
- ICLR 主线回到冻结非-RL系统。

---

## 10. Trace、replay 与 exact resume

### 10.1 每步 trace

必须保存：

- Plan/prompt/schedule/constraint/source SHA；
- tokenizer/vocabulary/2,481-token map/support SHA；
- parent B0、behavior old、frozen reference、RL adapter SHA；
- scientific panel、ordinal、body/refiner seeds；
- partial token IDs；
- current D1 group/axis/step；
- 当前 group 全部 candidate token IDs；
- selected position；
- 每个 candidate 的 legal-support IDs/bitset hash；
- old token/position/joint log-prob；
- token/position temperature；
- token/position RNG counter roles；
- reward vector、A/B fidelity、`Z/p`、judge provenance；
- raw output-to-attempt mapping。

### 10.2 replay

同一 behavior checkpoint：

- state/support/candidates/J/final tokens 一致；
- old log-prob tolerance 内一致；
- new=old ratio≈1；
- KL≈0；
- tie policy 一致；
- 任一 support/hash 差异 fail closed。

### 10.3 checkpoint transaction

原子 checkpoint 至少包括：

- RL adapter；
- parent B0/tokenizer/support identities；
- behavior/reference identities；
- optimizer/scheduler/scaler；
- global update、actor version、inner update；
- Python/NumPy/Torch CPU/CUDA/all-rank RNG；
- stateless seed ledger/counters；
- Plan sampler cursor；
- rollout manifest cursor；
- A/B label-cache cursor；
- mixed-fidelity `Z/p` ledger；
- source/config/reward/evaluator manifests；
- write-complete marker。

推荐三段事务：

```text
ROLLOUT_COMMITTED
  -> LABELS_COMMITTED
  -> UPDATE_COMMITTED
```

model_494 不是 bitwise deterministic。resume 必须复用已经 committed 的 B
label payload，不能重新跑 refiner 再称“exact resume”。恢复后的第一个动作是
trajectory parity，不是继续 optimizer。

---

## 11. 明确 CUT 清单

满足任一条，RL 不再占用 ICLR 主线资源：

1. Gate −1 到 8 月 10 日仍未冻结 support；
2. 需要 token/axis-sharing/embedding 表示变化；
3. R0 到 8 月 10 日仍无法 exact replay/resume；
4. paired-32 引入 illegal/cross-group/duplicate/new failure；
5. 没有可用的非零 position entropy；
6. paired-32 晚于 8 月 12 日，或 first-64 晚于 8 月 15 日；
7. 少于 25% Plan groups 有 reward variance；
8. A/B 相关性或 process-noise gate 失败，且 B-direct 超预算；
9. 独立 `E_gate` 到 8 月 12 日仍不可运行，或 8 月 15 日 smoke 仍失败；
10. token/position KL、ESS 或 clip gate 连续两次失败；
11. rare/unseen action mass 被 RL 放大；
12. reward proxy 上升而 independent evaluator 下降；
13. paired-256 strict 不正、meta 为负或两个 128 blocks 方向不一致；
14. completion/structure 下降超过 2 pp；
15. novelty/uniqueness 下降超过 5 pp；
16. paired-256 晚于 8 月 22 日；
17. final terminal 或独立 audit 晚于 8 月 31 日；
18. 累计达到 96 A800 GPUh；
19. 需要靠 retry、repair、replacement、filter 或 rerank 才过门；
20. 为挽救结果而在看过 endpoint 后改 reward、threshold、tau、judge 或 support。
21. 9 月 5 日仍无法冻结共同评测下的科学结果。

“CUT”不等于否定方向；它表示在当前 45/52 天提交窗口中停止把不完整 RL 当成
论文承诺。

---

## 12. 论文主线价值与角色升级规则

### 12.1 今天的角色

```text
可选增强，不是主线依赖
```

原因：

- frozen Body/refiner mechanics 已很强；
- Planner chemistry 才是 comp/joint 主瓶颈；
- strict/meta 已出现真实冲突；
- Gate −1、joint trace、independent evaluator 尚未落地；
- 最新 Mask-Aware PG 的原始训练规模远大于当前时间/标签预算。

### 12.2 何时升级

| Evidence | 论文角色 |
|---|---|
| 只过 R0/32 | engineering appendix 或 future work |
| first-64 过门 | 保持 `KEEP_BACKUP`，获得继续资格，不写 efficacy claim |
| paired-256 过门 | 获得 `KEEP_MAIN` 候选资格，开始写 method/results 草稿 |
| `4×256` + independent audit 过门 | 正式共同主线；可把 constrained mask-aware RL 写进摘要 |
| 任一 CUT | 从摘要/主结果移除；冻结主线继续投稿 |

即使最终通过，也不应宣称“RL 修复 comp_valid”。准确 claim 是：

> 在冻结 Planner composition、exact-length special-token support 与
> post-refiner pipeline 下，joint token-position mask-aware optimization
> 改善 end-to-end strict S.U.N.，同时保持 meta、结构与多样性非劣。

### 12.3 潜在创新 claim 的边界

有条件支持：

- group-constrained、legal-support-aware mask-aware RL for crystallographic
  language diffusion；
- D1 K=1 PL 作为当前 greedy sampler 的单变量概率化；
- randomized pre/post-refiner control-variate reward；
- exact trace/resume 与 evaluator firewall。

在完成外部文献查重和 `4×256` 前，不使用“first”或“SOTA”。

### 12.4 统一评分与 reviewer 输出

按共同框架评分，不用加权总分覆盖硬门：

| 维度 | 分数（1–5） | 依据 |
|---|---:|---|
| Innovation | 4 | constrained joint token-position policy 与随机 pre/post-refiner control variate 的组合有清楚方法差异，但不声称首创 crystal RL |
| Expected effect | 3 | 直接作用于 refiner basin、strict/meta 与多样性；不能改善 frozen Planner 的 composition |
| Evidence | 3 | R03 已证明 reveal order 可改变 strict/meta，外部工作支持 two-stage action；本地尚无 RL 因果结果 |
| Feasibility | 2 | 旧 TraceRL 正式 NO-GO，需新 joint trace、resume 与 evaluator firewall |
| Deadline fit | 2 | 8 月 15 日拿 64 可行但紧，8 月 22 日 256 与 8 月 31 日独立确认依赖并行资源 |
| Paper coherence | 4 | 若成功，直接连接 rich PlanGraph、exact-length DLM 与 frozen refiner；失败也不破坏 H1 主线 |

统一结论：

- **当前状态：`KEEP_BACKUP`。** 只过 R0/32 时为 `APPENDIX_ONLY`；8 月 15 日
  无 paired-64 信号、8 月 22 日无 paired-256 或任一科学硬门失败时为 `CUT`；
  只有 256 与独立确认依次通过才有资格 `KEEP_MAIN`。
- **最早日期：** paired-64 为 2026-08-14（最迟 08-15）；paired-256 为
  2026-08-21（最迟 08-22）。
- **最大可接受预算：** 累计 `96 A800 GPUh`，其中 paired-256 前累计上限
  `40 A800 GPUh`；不含第三 greedy arm，达到 cap 即停。
- **最可能成功机制：** 在 legal support、D1 groups、Planner 与 refiner 全冻结
  时，学习 reveal order 与 token choice 的联合策略，把合法 proposal 推入更好的
  refiner basin；meta-aware reward 防止 strict-only 极化。
- **最可能失败机制：** A 对 B 的排序相关性不足，或小 `G=8` 下 reward
  稀疏/联合 action 方差过高，导致策略只放大 proxy、稀有 token 或少数结构簇。
- **H1 因果归因：** 按本文冻结边界执行时不污染，因为唯一 principal factor 是
  `D1 greedy/no-RL PL -> D1-PL RL`，且保留 no-RL PL 对照；若同时改 token
  representation、support、Planner、safe-axis、refiner 或 evaluator，则立即
  污染并触发 CUT。
- **真实新颖性：** 相对 CrysVCD，不把 composition→diffusion 分层本身当贡献；
  相对 PLaID++/CrysTune，不依赖 Wyckoff 表示或迭代偏好对齐；相对 CRYSTAL，
  action 是 exact masked-DLM 的 legal token + reveal position，composition 不进
  Body advantage；相对 Mask-Aware PG，新增 crystallographic group/support
  constraints、pre/post-refiner 随机无偏校正和 all-attempt evaluator firewall。
  可主张的是这一组合，不能主张任一组件的 “first”。
- **08-15 无信号时的 fallback：** 论文回到
  “rich PlanGraph + exact-length masked Body DLM + frozen continuous refiner”
  的冻结 H1 主张，报告约 99.8% conditional structure validity，并把
  Planner chemistry reachability 定位为 composition 主瓶颈；RL 只留设计或
  future work。
- **可放入 9 页主文的一句：**
  “We formulate exact-length crystal DLM decoding as a constrained two-stage
  token–reveal policy and optimize it with randomized pre/post-refiner rewards
  while preserving all-attempt validity and meta-stability guards.”
- **绝不能写的一句：**
  “Our RL fixes composition validity and establishes the first state-of-the-art
  stable crystal generator.”

---

## 13. 最小实现清单

为避免污染 legacy，建议新 workstream 只新增：

1. `body_legal_policy.py`
   - pure legal support；
   - constrained token probability；
   - support hash。
2. `body_d1_pl_policy.py`
   - group-local K=1 PL；
   - stateless token/position RNG；
   - exact joint trace。
3. `body_multifidelity_reward.py`
   - A/B vector；
   - fixed `Z/p` ledger；
   - control-variate reward；
   - symmetric uniqueness。
4. `rollout_body_mapg.py`
   - G=8 online rollout；
   - immutable trace/label transactions。
5. `train_body_mapg.py`
   - old/reference separation；
   - two-channel normalized surrogate；
   - KL/ESS/clip hard gates；
   - exact checkpoint/resume。
6. `assemble_body_mapg.py`
   - raw all-attempt；
   - paired McNemar；
   - hierarchical bootstrap；
   - panel sign stability；
   - no promotion side effect。

先写 tests，再写 32 launcher。不得修改历史 TraceRL、R03 或 frozen sampler 作为
就地实验源。

---

## 14. 一页式结论

### 判定

Mask-aware Body RL **可尝试，但当前只能是有止损线的可选增强**。在 ICLR
2027 剩余 45/52 天的条件下，最小可行方案不是训练两个长期模型，而是：

```text
one frozen-parent RL LoRA
  + D1 group-local K=1 PL
  + joint token-position likelihood
  + A -> randomized A/B -> B curriculum
```

### 为什么值得试

- 当前 sampler 的 reveal order 已被 R03 证明是因果变量；
- 最新 Mask-Aware PG 证明 token-only gradient 会遗漏 position 方向；
- exact crystal special tokens 使每个 legal categorical support 可精确计算；
- pre/post refiner 是天然多保真标签；
- R03H 已给出明确 reward 目标：strict 要升，但 meta 不能再掉。

### 为什么不能直接开训

- 2,343 个 stochastic numeric tokens 有明显 coverage 长尾；
- 旧 TraceRL 的 behavior probability 与真实 sampler 不一致；
- current greedy top-k 没有 position log-prob；
- composition 在 frozen Plan 内没有 advantage；
- 同一 judge 训练和证明会造成 evaluator hacking；
- model_494 CUDA repeats 不是独立 science seeds。

### 首版算法

在每个 D1 group 内，对每个剩余位置从
`softmax(legal_logits/0.7)` 采 token candidate；用当前 sampled-token 的未温度化
confidence 构造 K=1 PL position distribution，再提交一个位置。保存当前 group
全部 candidates、legal supports、token/position log-prob。只训练新的 r=8 LoRA，
B0 embedding/output head 冻结。

### Reward

```text
R_F =
  exact/Plan/structure validity
  × novelty
  × symmetric cluster uniqueness
  × meta-aware S(E_hull)
```

`E_hull<=0` 得 1；meta-only 得 0.5–0.75；above-meta/unknown 得 0。
composition 不进入 Body reward。

多保真 mixed phase：

```text
R_MF = R_A + Z/0.5 * (R_B - R_A)
```

`Z` 必须在看 A score 前随机冻结。这样只维护一个 LoRA，且估计目标仍是 B。

### 最小规模与预算

- first-64：64/arm，128 A/B pairs，外加 16 B process repeats；
- train：1,024 rollouts，1,024 A labels，512 B labels；
- held-out：256/arm；
- confirm：4 个独立 256 panels，RL vs no-RL PL；
- 最大累计：96 A800 GPUh；
- final 不含第三 greedy arm。

### 时间硬门

- Gate −1：最迟 8 月 10 日；
- 审计/设计/一句话 claim：8 月 7 日冻结；
- Gate −1 与 R0：最迟 8 月 10 日；
- first-64：最迟 8 月 15 日；
- paired-256：最迟 8 月 22 日；
- final `4×256` 与独立 audit：最迟 8 月 31 日；
- science freeze：9 月 5 日；paper/supp/repro only-fix：9 月 12 日。

### 何时砍掉

support 未冻结、exact resume 不闭合、PL 引入新失败、reward group 无方差、
A/B 不相关、独立 evaluator 未就绪、proxy/independent 方向冲突、256 strict
不正或 meta 为负、预算超过 96 A800 GPUh、8 月 15/22/31 任一关键 terminal
逾期：立即从 ICLR 主线 CUT。

### 投稿角色

今天：**`KEEP_BACKUP`，可选增强**。

256 通过：**候选共同主线**。

`4×256` 与独立 audit 通过：**正式共同主线，可进摘要**。

否则：**投稿后工作，不拖累冻结的 PlanGraph-DLM-refiner 主线**。
