# H1 Body-DLM 完整方案审计与后续 RL 设计 V1

状态：`design_only_no_execution_authorization`

日期：2026-08-04

适用锚点：H1-A2 `P0 + B0/R5-C exact-length DLM + D1 + CrysLLMGen
model_494 exact-800`

本文汇总当前 Body-DLM 从数据、特殊 token、SFT、生成、refinement 到
Direct/S.U.N. 的完整合同，并给出可审计的 RL 方案。本文不授权训练、生成、
refinement、评测、checkpoint selection、promotion 或自动下游。

## 1. 结论

### 1.1 总判定

| 项目 | 判定 |
|---|---|
| 当前 B0/R5-C exact-length DLM | **可保留，是成熟锚点** |
| 当前 D1 greedy low-confidence sampler | **保留为冻结基线，不直接用于严格 policy gradient** |
| safe-axis 作为 RL 起点 | **NO-GO**；strict 上升但 meta gate 已失败 |
| 现有 `scripts/llada_trace_rl.py` | **正式实验 NO-GO**；只能作历史 heuristic scaffold |
| 新的 mask-aware Body-DLM RL | **CONDITIONAL GO** |
| 一开始长期训练两个独立 RL 模型 | **NO-GO** |
| 一个多保真 RL 模型，A→B curriculum | **CONDITIONAL GO，首选** |

建议方案是：

```text
冻结 B0 + D1
  -> Gate -1：完整 token coverage + legal probability mass 审计
  -> support-only 单变量筛选，或显式保留原 support
  -> 冻结一个 policy/reference 共用的 action support
  -> 特殊 token / legal-support / trajectory 概率合同
  -> D1 组内 K=1 Plackett–Luce，无训练 32
  -> 无训练 64：greedy↔PL 安全性 + pre/post reward 相关性
  -> 小 LoRA 256：no-RL PL / A-only / B-direct / 可选 A→B
  -> 只冻结一个胜者
  -> 4 × 256 真正独立 scientific ledgers 做 1,024 确认
```

核心算法不是普通 token PPO，而是：

> **保持 D1 的硬 group 边界，在组内对 token 候选与 reveal 位置建立联合
> 概率，使用 K=1 Plackett–Luce probabilistic remasking，再做
> mask-aware、sequence-normalized policy optimization。**

### 1.2 能优化什么，不能优化什么

Body-DLM 的 `N` 和每个 atom slot 的 element token 已由 Planner 预填。
同一 Plan 的多个 rollout 拥有完全相同的 composition，因此：

- Body RL 可以优化 body completion、离散几何、refiner basin compatibility、
  post-refiner structure validity、strict/meta stability、novelty 和 uniqueness；
- Body RL 不能把同一 Plan 内的 composition reward 变成有效 advantage；
- `comp_valid` 的主要改进仍必须来自 Planner/formula/chemistry；
- Body RL 的论文表述必须是 **conditioned on a frozen Plan**。

## 2. 当前 Body-DLM 完整链路

### 2.1 冻结模型身份

| 字段 | 冻结值 |
|---|---|
| Base model | LLaDA-8B-Instruct |
| Body role | `B0`, historical R5-C exact-length |
| Adapter | `runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final` |
| Adapter bytes | `6,391,016,776` |
| Adapter SHA-256 | `5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d` |
| Tokenizer size | `128830` |
| Vocabulary SHA-256 | `3acc073da85047265769f2dccd93543fa9d7cbfa95021aef54ef282b13ce2f37` |
| `tokenizer.json` SHA-256 | `3a21588abca8e56155cc7b6cabb81df51992ccd2e89704aec770912f24e75509` |
| `tokenizer_config.json` SHA-256 | `8e89acaa54a8fb8fc7d228165ac483f61b7fef7c4c9761214092511190f75de2` |

冻结证据见
[H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md](H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md)。

### 2.2 特殊 token 是模型设计本身

当前 DLM 使用 `2,481` 个晶体特殊 token，而不是让普通文本 tokenizer
分解晶体字段：

| 类别 | 数量 | 示例 | R5 exact-length 中的作用 |
|---|---:|---|---|
| Atom count | 20 | `<N_004>` | 位置 0，按 Plan 预填 |
| Lattice length | 1,503 | `<LA_042>` | LA/LB/LC，各 501 个合法 bin |
| Lattice angle | 537 | `<AA_090>` | AA/AB/AG，各 179 个合法 bin |
| Fixed-slot index | 20 | `<S00>` | 旧 fixed-slot 兼容；R5 exact 不生成 |
| Element | 94 | `<E_O>` | 每个 atom slot 按 Plan 预填 |
| Fractional coordinate | 303 | `<X_050>` | X/Y/Z，各 101 个合法 bin |
| Empty/pad | 4 | `<EMPTY>`, `<X_PAD>` | 旧 fixed-slot 兼容；R5 exact 不生成 |
| 合计 | 2,481 | — | 冻结 checkpoint tokenizer 的完整 data vocabulary |

特殊 token 带来四个重要结果：

1. 每个晶体语义字段恰好是一个动作，不存在多 token 数字或元素名边界。
2. R5 exact-length 的序列长度严格为 `7 + 4N`，不是近似 token budget。
3. 形式上虽然模型输出词表有 128,830 项，但某个 body 位置的合法支持只有：

   - count：1；
   - element：预填后不再采样；
   - length：最多 501；
   - angle：179；
   - coordinate：最多 101，再被动态 mask 收紧。

4. RL 的概率必须在 **mask 后的特殊-token合法支持上重新归一化**；
   对完整 128,830 词表直接做 `log_softmax` 是错误的 behavior probability。

### 2.3 特殊 token 的不可变合同

RL 前必须把以下条件设为 fail-closed：

- tokenizer 必须从 frozen B0 checkpoint 加载，禁止从 data vocabulary 重建；
- vocabulary、`tokenizer.json`、`tokenizer_config.json` 和所有 2,481 个
  token ID 必须逐项一致；
- `MASK_TOKEN_ID`、pad/eos ID 和 model embedding/output vocabulary size
  必须一致；
- 每一步 legal-support token ID 集和其 SHA 必须保存；
- tokenizer 或 token ID 任何变化都会使历史 old log-prob、replay 和 KL
  全部失效。

这不是理论风险。V6 曾因使用 standalone 重建 tokenizer 的 vocabulary SHA，
而不是 checkpoint tokenizer identity，在 body generation 前 fail closed。
V7 审计确认 B0/B1/B2 checkpoint 的 2,481 个 data token ID 完全一致。

### 2.4 定义 vocabulary 不等于有效 action vocabulary

新增只读覆盖审计见
[H1_BODY_SPECIAL_TOKEN_COVERAGE_AUDIT_V1.md](H1_BODY_SPECIAL_TOKEN_COVERAGE_AUDIT_V1.md)。

在本地冻结的 `9,046` 条 exact-length held-out records 中：

- `2,481` 个特殊 token 只出现 `1,437` 个；
- `1,044` 个未出现；
- H1 Body policy 的 `2,343` 个 numeric stochastic-action token 中只出现
  `1,330` 个，`1,013` 个未出现；
- 三个 length family 单独贡献 `953` 个 held-out-unseen token；
- 旧 fixed-slot 的 `S/EMPTY/PAD` 共 `24` 个在 R5 exact 中结构性不可达。

这是 held-out coverage，不是完整训练覆盖。正式 support contraction 前必须扫描
冻结的 `27,136` 条 train corpus，并测量 B0 在 legal mask 后分配给
`train-unseen/rare` token 的概率质量。仅统计 token type 数量不足以证明它会影响
sampling。

这个发现不改变“Planner 是当前 composition 主瓶颈”的结论，也不能解释已经
达到约 `99.7–99.8%` 的 conditional post-refiner structure validity。但它会
影响：

- lattice tail calibration；
- refiner basin burden；
- meta S.U.N. 尾部；
- RL 对低基准概率 action 的投机风险。

因此 token coverage/legal-mass audit 是 RL 的 **Gate −1**。support
representation 或 mask 的变化必须先作为独立 sampler factor 完成，不能与
policy optimization 同时改变。

### 2.5 特殊 embedding/output head 的训练边界

历史 B0 训练不仅保存普通 LoRA，还将：

- `model.transformer.wte`
- `model.transformer.ff_out`

作为 `modules_to_save` 保存。这解释了 B0 adapter 约 6.39 GB，远大于普通
rank-8 LoRA。RL 不能不经审计地以 `is_trainable=True` 恢复整个 B0 adapter，
否则可能同时更新完整 embedding/output head，而不再是“小 LoRA RL”。

首个 RL 协议建议：

1. 加载 base model；
2. resize 到 frozen 128,830 vocabulary；
3. 加载 B0，包括已训练的特殊-token embedding/output head；
4. 将 B0 合并或封装成一个新的只读 parent body base，并冻结全部参数；
5. 在 q/k/v/ff/up 模块上附着新的 RL-only LoRA；
6. 新 RL LoRA 的 `modules_to_save=None`；
7. 特殊-token embedding 和 output head 在第一阶段保持冻结。

隐藏状态经 LoRA 改变后，冻结 output head 仍可改变各特殊 token 的 logits。
若未来要单独训练 2,481 个特殊 token 行，必须作为第二个独立因子，使用
row-level gradient mask，并重新做 tokenizer/checkpoint/KL 审计；不能与第一轮
RL 同时改变。

### 2.6 数据与 SFT

R5 exact-length 数据由 MP-20 的结构数组构建：

| Split | Rows | Build failures | Formula matches Plan |
|---|---:|---:|---:|
| Train | 27,136 | 0 | 100% |
| Validation | 9,047 | 0 | 100% |
| Test | 9,046 | 0 | 100% |
| Total | 45,229 | 0 | 100% |

每条记录包含：

- rich Plan/body prompt；
- exact answer；
- `N/elements/counts`；
- lattice lengths/angles；
- element/XYZ atom blocks；
- exact semantic length；
- raw structure metadata。

注意：这些 Plan 是由参考结构构建的 teacher Plan，适合训练 Body-DLM，
不能被当作 de novo Planner 的独立成功证据。

Canonical B0 SFT：

- one epoch；
- two A800 DDP；
- per-device batch `1`；
- gradient accumulation `8`；
- effective global batch `16`；
- LR `5e-5`；
- cosine scheduler；
- warmup `100` updates；
- min LR ratio `0.2`；
- bf16；
- LoRA `r=8, alpha=32, dropout=0.05`；
- targets `q_proj,k_proj,v_proj,ff_proj,up_proj`；
- answer-only IID mask corruption。

训练不是 AR next-token loss。`forward_process` 对 answer positions 做随机
mask，模型预测原始特殊 token；masked CE 除以 `p_mask`，再按 answer position
做 sample-level normalization。

### 2.6 Exact-length representation

给定 `N`，answer positions 为：

```text
0           <N_NNN>
1..3        <LA_*>, <LB_*>, <LC_*>
4..6        <AA_*>, <AB_*>, <AG_*>
7+4k        <E_*>
8+4k        <X_*>
9+4k        <Y_*>
10+4k       <Z_*>
```

其中 `k=0..N-1`。因此总长度是：

```text
1 + 6 + 4N = 7 + 4N
```

正式 H1 中 count 和所有 element positions 均从 Plan 预填。真正的随机
Body action 数量只有：

```text
6 lattice/angle actions + 3N coordinate actions
```

当 `N=20` 时也只有 66 个 committed actions。

### 2.7 Canonical D1 与 safe-axis

Canonical D1 group：

```text
[N] -> [all elements] -> [six lattice fields] -> [all X] -> [all Y] -> [all Z]
```

count/elements 已预填，所以实际 stochastic groups 是：

```text
[six lattice fields] -> [all X] -> [all Y] -> [all Z]
```

safe-axis 保留 composition/lattice group，但按 PlanGraph site group 将 X、
Y、Z 进一步拆分，并保证：

- no mixed-axis group；
- `z_before_xy=0`；
- all XY precede all Z。

safe-axis 是有价值的独立 ablation，但不能作为新 RL 的默认起点，因为它已
观察到 strict/meta 极化并触发 scientific stop。

### 2.8 正式生成

H1 formal body generation 冻结为：

- temperature `0.7`；
- CFG `0.0`；
- remasking `low_confidence`；
- exact `7+4N`；
- count/element prefill；
- schema logit mask；
- duplicate-coordinate mask；
- lattice-volume mask；
- method-independent ordinal/body-noise ledger；
- all-attempt denominator；
- no retry/replacement/repair/filter/rerank。

给定当前状态，sampler：

1. 对整段序列 forward；
2. 应用 schema 和动态 mask；
3. 对仍 masked 的位置用 Gumbel-max 采候选 token，分布等价于
   `softmax(masked_logits / 0.7)`；
4. 使用 **未温度化** `softmax(masked_logits)` 下的 sampled-token
   probability 作为 confidence；
5. 在当前 group 内 greedy top-k reveal；
6. 提交 selected token，其他候选丢弃并在下一步重采。

当前每个 group 的 step 数等于该 group 的待生成位置数，因此正常情况下
每步每行提交 `K=1`。

### 2.9 Frozen refiner 与评测

每个 body-success proposal 都必须：

1. 进入同一个 frozen CrysLLMGen `model_494`；
2. exact 800 reverse steps；
3. batch size 1；
4. ordinal/refiner seed 冻结；
5. 完成后才进入 Direct evaluator 与 S.U.N.。

Primary reporting 使用 raw all-attempt denominator。失败、unknown、非 novel、
非 unique 都保留并记为 false。

Direct 报告：

- generation completion；
- composition validity；
- structure validity；
- joint validity；
- unique、novel、novel-unique。

S.U.N. 报告：

- strict：`E_hull <= 0.0 eV/atom`；
- meta：`E_hull <= 0.1 eV/atom`；
- 同时要求 stable/metastable、unique representative、novel。

## 3. 当前证据告诉我们什么

### 3.1 DLM mechanics 已经较强

R03D body-only：

| Arm | Success |
|---|---:|
| D1 | `246/256` |
| safe-axis | `248/256` |

R03E 四个 process repeats 的 pooled raw Direct：

| Arm | Generation | Composition | Structure | Joint |
|---|---:|---:|---:|---:|
| D1 | 984/1024 | 848/1024 | 982/1024 | 846/1024 |
| safe-axis | 992/1024 | 852/1024 | 989/1024 | 851/1024 |

在 refine-success denominator 上：

| Arm | Composition valid | Structure valid | Joint valid |
|---|---:|---:|---:|
| D1 | 86.1789% | 99.7967% | 85.9756% |
| safe-axis | 85.8871% | 99.6976% | 85.7863% |

真正 post-refiner structure-invalid 只有 D1 两个、safe-axis 三个。raw
structure-invalid 主要来自 Planner/body upstream failure，不是 model_494
结构判据。

### 3.2 safe-axis 没有通过 broad stability gate

completed-snapshot lower-bound pooled：

| Arm | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|
| D1 | `99/1024 = 9.67%` | `523/1024 = 51.07%` |
| safe-axis | `117/1024 = 11.43%` | `496/1024 = 48.44%` |
| Delta | `+1.76 pp` | `-2.64 pp` |

strict 四个 process repeats 都为正，但 meta 为三负一零。它证明生成顺序是
有效因子，也证明只优化 strict 会把质量从 meta-only 区间推到 `>0.1` 的错误
一侧。

### 3.3 对 RL 的直接含义

- 不能从 safe-axis checkpoint 或 safe-axis policy 默认起步；
- 不能把 strict binary reward 作为唯一主 reward；
- meta 必须是 checkpoint-selection 的硬非劣约束；
- composition 必须从 body reward 中剥离；
- 训练前先验证 reward 是否在同 Plan 的 group 内产生足够 variance。

## 4. 正式 policy 的 state、action 与 transition

### 4.1 当前 greedy sampler 的真实 action

令当前 D1 group 中仍 masked 的位置集合为 `I_t`。状态：

```text
S_t = (
  frozen Plan and prompt,
  exact 7+4N layout,
  partial token sequence,
  current D1 group and step,
  per-position legal special-token support,
  model/tokenizer/constraint/schedule identities
)
```

对每个 `i in I_t`：

```text
Y_i ~ q_tok_theta(. | S_t, i)
q_tok_theta = softmax(masked_logits_i / tau_tok)
tau_tok = 0.7
```

当前实现再计算：

```text
c_i = softmax(masked_logits_i)[Y_i]
```

并 greedy 选择 `c_i` 最高的位置 `J`，提交 `(J, Y_J)`。因此位置选择取决于
组内所有 sampled candidates；只保存 final tokens 和 reveal step 无法恢复
behavior trajectory probability。

### 4.2 为什么 greedy top-k 不能直接做严格 policy gradient

greedy `argmax/top-k`：

- 依赖模型参数产生的 confidence；
- 对参数不可微；
- 没有显式、可求 log-prob 的 position policy；
- token-only gradient 会漏掉改变 reveal order 的改进方向。

[Mask-Aware Policy Gradients for Diffusion Language Models](https://arxiv.org/abs/2607.15200)
将 MDLM 生成形式化为 token decision 和 remasking/unmasking decision 的
两阶段 MDP，并证明 position term 不能被 token-only policy gradient 忽略。

### 4.3 推荐的组内 K=1 Plackett–Luce policy

保持 D1 的 group 边界完全不变。只把 group 内 greedy top-k 改为概率化
position selection。

首先对当前 group 的每个剩余位置采 token：

```text
Y_i ~ q_tok_theta(. | S_t, i)
```

再以当前实现的 sampled-token confidence 定义：

```text
v_i = log softmax(masked_logits_i)[Y_i]
```

位置策略：

```text
p_pos_theta(J=j | Y, S_t)
  = exp(v_j / tau_pos)
    / sum_{i in I_t} exp(v_i / tau_pos)
```

K=1 的 step probability：

```text
pi_theta(a_t | S_t)
  = product_{i in I_t} q_tok_theta(Y_i | S_t, i)
    * p_pos_theta(J | Y, S_t)
```

transition：

```text
x_{t+1,J} = Y_J
other positions in I_t remain MASK
```

只需保存当前 group 内全部 candidates；对未来 group 采出的无效随机量应在新
实现中删去或积分掉，不能把没有影响 transition 的随机数塞进 likelihood。

`tau_pos -> 0` 才趋近 greedy。RL training 和 primary deployment 必须使用
同一个冻结的 `tau_pos > 0`，不能训练 PL、评测时偷偷切回 greedy。

### 4.4 为什么从 D1-PL 开始

D1-PL 只改变 D1 group 内的 reveal stochasticity：

- exact length 不变；
- special-token schema 不变；
- count/element prefill 不变；
- lattice→X→Y→Z 的宏观顺序不变；
- duplicate/volume constraints 不变；
- Planner、model_494、Direct/S.U.N. 不变。

safe-axis 则已经是一个有独立 endpoint 结果的 treatment。将它与 PL+RL
合并会一次改变两个 principal factors，并把已有 meta 失败带入新实验。

### 4.5 Singleton D1 的角色

把 D1 展开成预声明 singleton reveal order，可使 position 完全确定，policy
退化为 constrained token categorical。它适合：

- 验证 legal-support log-prob；
- 验证 exact replay；
- 验证 PPO/GSPO identity；
- 作为 token-only debug oracle。

但 singleton 顺序不是当前 groupwise greedy D1 的 bitwise no-op，也失去学习
unmask order 的能力。因此它只作 R0 工程 control，不作为正式 RL candidate。

## 5. 现有 TraceRL 为什么是 NO-GO

`scripts/llada_trace_rl.py` 和 `crystal_dlm/rl_utils.py` 的主要缺口：

1. 从 final sequence 和 `step_map` 重建 state，而不是记录真实 rollout state；
2. 缺 `step_map` 时人工使用 `range(answer_len)`，可构造并不存在的轨迹；
3. 默认 `trace_shrink=8`，进一步合并真实 denoising states；
4. old log-prob 在训练前用当前加载模型重算，不是 rollout 时在线捕获；
5. 对完整 vocabulary 做 raw `log_softmax`，没有 schema/dynamic mask 后重归一化；
6. 没有 temperature `0.7`；
7. 没有 position/remask probability；
8. 没有保存组内未提交 token candidates；
9. 没有 behavior/reference/action-support SHA；
10. 没有 ESS、max ratio、clip fraction、position entropy 等 fail-closed 指标；
11. `k3` KL 指数项在 diffusion sequence proxy 上有不稳定风险；
12. checkpoint 没有完整保存 optimizer/scheduler/scaler/RNG/replay cursor，不能
    称 exact resume；
13. 旧 reward 主要是 formula/SMACT/duplicate，适用于生成 composition 的旧
    fixed-slot RL，不适用于 composition 已冻结的 Body-DLM。

可复用：

- LoRA/optimizer/DDP 外壳；
- group reward normalization 的基础工具；
- clipping/KL 日志框架；
- JSONL I/O 和 checkpoint 命名习惯。

必须重写：

- rollout；
- online behavior log-prob；
- legal-support abstraction；
- token+position joint probability；
- replay；
- reward；
- exact resume；
- statistics/gates。

## 6. 推荐 RL 优化器

### 6.1 名称与原则

建议协议名称：

> `H1 Body MAPG-PL-GSPO`

含义：

- Mask-Aware Policy Gradient；
- D1 group 内 K=1 Plackett–Luce；
- group-relative advantage；
- sequence/trajectory normalized ratio；
- constrained special-token supports。

### 6.2 Rollout group

每个 frozen Plan 采 `G=8` body rollouts：

- `G=4` 在 strict≈11% 时至少一个 strict positive 的概率约 38%；
- `G=8` 约 61%；
- 更大的 G 会显著增加 exact800 标签成本。

advantage 建议使用 leave-one-out 或减 group mean；第一版不除以很小的
group standard deviation，避免放大 refiner/judge noise。reward 完全相同的
Plan group 自然产生零更新，并单独计入 `zero_variance_group_rate`。

### 6.3 Importance ratio

每步记录：

```text
Delta_tok,t =
  sum_{i in I_t} [
    log q_new(Y_i | S_t, i) - log q_old(Y_i | S_t, i)
  ]

Delta_pos,t =
  log p_new(J | Y, S_t) - log p_old(J | Y, S_t)

rho_joint,t = exp(Delta_tok,t + Delta_pos,t)
```

完整 exact trajectory ratio 可审计；优化时使用按 sampled candidate 数或
committed action 数归一化的 sequence ratio，以控制 `N` 不同导致的长度尺度。
[ESPO](https://arxiv.org/abs/2512.03759) 的主要启示是：diffusion ELBO/token
proxy 不应机械拆成 AR token-level ratio，sequence-level normalized ratio
更稳定。

### 6.4 初始稳定性设置

这些值只能作为预注册初值，不是已经验证的最佳超参数：

- RL LoRA：`r=8`；必要时单独筛 `r=16`，不从大 rank 起步；
- clip：`[0.8, 1.2]`；
- inner updates：最多 1–2；
- stale replay：最多一个 actor version；
- token temperature：冻结 `0.7`；
- position temperature：只在无训练 32/64 calibration panel 上，从
  `{0.05, 0.1, 0.2, 0.5}` 中预注册选择一次；
- 每次 RL rollout 刷新后只做极少 optimizer steps；
- frozen B0 是 reference policy；
- rollout behavior old policy 与 frozen reference policy 分开保存。

### 6.5 KL 与 ESS

联合 KL 可拆为：

```text
KL_joint =
  KL(token candidate policy)
  + E_Y KL(position policy | sampled candidates)
```

建议初始 fail-closed：

- mean normalized token KL `<=0.01 nat`；`>0.02` hard stop；
- mean position KL `<=0.01 nat`；`>0.02` hard stop；
- 对 frozen B0 的累计 trajectory/committed-token KL `<=0.05`；
- exact joint transition normalized ESS `>=0.70`；
- trajectory ESS `>=0.50`；
- clip fraction `<=20%`，任一 group `>30%` 停止；
- ratio、KL、ESS、entropy 按 N、axis/group、chemistry family 分层报告。

旧 TraceRL 使用的 `k3` KL 包含指数项。ESPO 报告其在 diffusion proxy 上可能
产生尖峰；第一版应比较 exact categorical KL 与稳定的 squared-log-difference
`k2` regularizer，并把选择冻结在训练前。

## 7. Reward 与 evaluator isolation

### 7.1 Reward 的层级

训练必须保留 reward vector，不只保存一个标量：

```text
z = (
  body_completion,
  exact_length_and_plan_match,
  pre_structure_margin,
  post_refiner_completion,
  post_structure_margin,
  stability_continuous,
  strict_indicator,
  meta_indicator,
  novelty,
  symmetric_uniqueness,
  OOD_penalty,
  refiner_displacement,
  refiner_seed_sensitivity
)
```

composition validity 单独记录为 `Q_plan`，不进入同 Plan 的 Body advantage。

### 7.2 不直接复制 CRYSTAL reward

[CRYSTAL](https://openreview.net/pdf/94d95333b625bc19463eca098ff60038d639d590.pdf)
的 validity × stability × novelty × diversity 多目标思路值得借鉴，但不能
原样复制：

- CRYSTAL 是 autoregressive policy，本项目是 masked diffusion；
- CRYSTAL 的 composition 是模型 action，本项目的 Body composition 已预填；
- 多个 binary reward 直接相乘会让当前 strict≈10% 的多数 group 全零；
- 其稳定性/S.U.N. 口径不能替代本项目 strict/meta 两个阈值；
- sequential first-seen uniqueness 会引入 order bias。

第一版训练 stability shaping 可以是预注册的分段连续 utility，例如：

```text
E_hull <= 0.0       -> 1.0
0.0 < E_hull <=0.1 -> 0.5 .. 0.75 continuous
E_hull > 0.1        -> 0
unknown/failure     -> 0
```

这样 meta-only proposal 至少保留正价值，减少“strict 上升、meta 下跌”的
极化风险。具体分段、judge 和缩放必须在 calibration split 上冻结。

### 7.3 训练目标与 checkpoint selection 分离

训练可以使用连续 scalar advantage，但 checkpoint selection 必须
lexicographic/constrained：

1. completion、exact length、Plan match 不过门，拒绝；
2. structure validity 不过非劣，拒绝；
3. meta 不过非劣，拒绝；
4. novelty/uniqueness/OOD 不过门，拒绝；
5. 只在幸存 checkpoint 中比较 strict；
6. final test 不参与 reward weight、threshold、early stop 或 checkpoint
   selection。

这套分离来自 advanced-evaluation 原则：reward proxy、model selector 和 final
evaluator 必须是不同角色，不能由同一个数字同时训练、选模和证明成功。

### 7.4 三层 evaluator

| 层 | 用途 | 约束 |
|---|---|---|
| `E_train` | rollout reward | frozen judge/checkpoint/cache；只看 train/replay split |
| `E_select` | validation 选模 | 不同 architecture 或至少独立 checkpoint；不看 final |
| `E_final` | 一次性科学结论 | 保持历史 Direct + CHGNet/MP-hull，同时加独立 MLIP/DFT audit |

若 `E_train` 使用 CHGNet，则最终主结论不能只由同一个 CHGNet 给出。Plan、
chemical system、seed 和 structure-neighbor leakage 都应在 split 层隔离。

novelty reference、StructureMatcher 参数、hull snapshot、MLIP checkpoint、
model_494、代码 manifest 和 threshold 全部锁 SHA。

## 8. Pre-refine A 与 post-refine B

### 8.1 两个 fidelity 的严格定义

模型 A：

```text
DLM proposal -> frozen low-fidelity reward judge
```

应称 `pre-model_494 proxy`。若 A 内仍有 cheap MLIP relax，就不能称
“完全不 refine”。

模型 B：

```text
DLM proposal
  -> frozen model_494 exact800, batch1
  -> same reward-judge family
```

B 更接近部署目标，但标签更贵。

### 8.2 是否长期保留两个 LoRA

若最终部署始终包含 model_494 exact800，长期保留两个独立模型没有足够理由：

- 两套 rollout 和 B labels 近似双倍成本；
- 多一个模型增加 selection degrees of freedom；
- pre/post fidelity 与训练步数容易混淆；
- 最终仍只能部署一个 post-refine 最优策略。

推荐：

```text
B0
  -> 可选 A warm-up
  -> randomized mixed-fidelity
  -> B-direct final stage
```

两个 LoRA 只在 256 阶段作为因果消融；终局只冻结一个 winner。只有存在真正
的“无 model_494 部署产品”时，才长期维护 A 和 B 两个模型。

### 8.3 A 是否有资格 warm-start B

在同一批 raw proposals 上同时取 A/B labels，报告：

- continuous score Spearman；
- strict/meta AUROC；
- top-A quartile 对 B-meta/B-strict 的 enrichment；
- `strict / meta-only / above-meta / ineligible` 四状态转移；
- refiner displacement；
- model_494 process variance。

建议 gate：

- 64 exploratory：Spearman point `>=0.30`，top quartile 的 B-meta 至少比
  bottom quartile 高 `10 pp`，无明显反向排序；
- 256 confirmation：bootstrap 95% lower bound `rho>0.10`，meta AUROC
  `>0.60`；strict positives 足够时 strict AUROC 也 `>0.60`。

若失败：

- A-only 和 A→B 均停止；
- 转 B-direct，或用随机 B 子样本训练带 uncertainty/OOD rejection 的
  surrogate。

### 8.4 Mixed-fidelity 必须随机抽 B

不能只 refine A 预测最好的 proposal，否则产生 survivorship bias。B 子样本
必须按预注册概率随机抽取并保存 inclusion probability。

可以评估 doubly robust estimator：

```text
R_MF =
  Rhat_B(A, x)
  + Z/p * (R_B - Rhat_B(A, x))
```

其中 `Z` 表示是否查询 exact800，`p` 为冻结抽样概率。它只在 surrogate
calibration 和 overlap 通过后使用。

## 9. Rollout、seed 与 exact resume 合同

### 9.1 每一步必须保存

- Plan、prompt、schedule、constraint、source manifest SHA；
- tokenizer/vocabulary/2,481-token ID map SHA；
- parent B0、behavior old、reference B0、RL adapter SHA；
- ordinal、body seed、refiner seed、scientific panel；
- partial state token IDs；
- current D1 group/axis/step；
- group 内全部 candidate token IDs；
- selected position；
- legal-support token IDs 或可验证 bitset/hash；
- token/position/joint behavior log-prob；
- token/position temperatures；
- RNG counter/role；
- reward vector、judge provenance 和 output mapping。

只保存 final tokens、step map 或 sampled seed 不足以形成正式 trajectory。

### 9.2 Replay gate

同一 behavior checkpoint 重算时：

- state token IDs 完全一致；
- legal support 完全一致；
- candidates 和 selected position 完全一致；
- old log-prob 在预注册 tolerance 内一致；
- `new == old` 时 ratio≈1；
- KL≈0；
- final sequence bitwise 一致；
- 任一 support/tie/hash 差异 fail closed。

### 9.3 Exact resume

原子 checkpoint 至少包括：

- RL adapter；
- frozen parent B0 identity；
- tokenizer identity；
- optimizer/scheduler/scaler；
- global update、rollout version、replay cursor；
- Python/NumPy/Torch CPU/CUDA/all-rank RNG；
- behavior/reference SHA；
- seed ledger；
- source/config manifests；
- completed marker。

恢复后先执行一段 trajectory parity；不能只因 adapter 可加载就称 exact resume。

### 9.4 科学 seed 与 process repeat

最终 `1,024` 不是同一个 256 ledger 重复运行四次，而是：

```text
4 independent scientific panels × 256 ordinals
```

每个 panel 内 control/candidate 使用 common-random-number roles。相同
scientific seeds 的 CUDA/model_494 repeats 只能估计 process noise，不能冒充
四个独立科学样本。

## 10. 实现文件建议

保留 legacy 实现不动，新建独立 workstream：

1. `crystal_dlm/body_special_token_contract.py`

   - checkpoint tokenizer identity；
   - 2,481-token map；
   - R5 active vs compatibility-only token categories；
   - embedding/output-head freeze audit。

2. `crystal_dlm/body_legal_policy.py`

   - pure legal-support function；
   - mask 后 constrained log-prob；
   - empty-support fail closed；
   - special-token support SHA。

3. `crystal_dlm/body_pl_remasking.py`

   - D1 group 内 K=1 Plackett–Luce；
   - token/position/joint probability；
   - explicit generators/stateless RNG；
   - no cross-group actions。

4. `scripts/rollout_llada_body_mapg.py`

   - online trajectory capture；
   - group rollouts；
   - immutable manifests；
   - reward vector mapping。

5. `scripts/train_llada_body_mapg.py`

   - old/reference separation；
   - sequence-normalized ratio；
   - token+position terms；
   - KL/ESS/clip gates；
   - exact checkpoint/resume。

6. `crystal_dlm/body_multifidelity_reward.py`

   - A/B label contract；
   - randomized B inclusion；
   - symmetric uniqueness；
   - reward/evaluator separation。

7. `scripts/assemble_body_mapg.py`

   - raw all-attempt；
   - paired McNemar；
   - hierarchical paired bootstrap；
   - panel sign stability；
   - unknown/failure false；
   - no automatic promotion。

## 11. 必须新增的测试

| 测试 | 硬门 |
|---|---|
| Special-token identity | 128,830 vocab、2,481 data tokens、所有 token ID 与三份 tokenizer SHA 一致 |
| Active support | R5 exact 不允许 slot/empty/pad token 进入 action support |
| Embedding boundary | B0 wte/ff_out frozen；RL trainable parameter list只含注册 LoRA |
| Legal-mask parity | 新 pure function 与 frozen sampler 逐 token 相同 |
| Constrained probability | 与手算 `log_softmax(masked_logits/T)` 相同；非法概率严格 0 |
| PL probability | K=1 categorical sum=1；极低 temperature 趋近 greedy |
| Group invariant | 不跨 lattice/X/Y/Z group；无 mixed-axis；exact coverage |
| Trace replay | state/support/candidates/selection/log-prob 完全复现 |
| Identity update | new=old 时 ratio=1、KL=0、梯度/损失符合预期 |
| Ordinal invariance | batch/rank/world size/order 不改变单 ordinal trajectory |
| Empty support | 立即失败，无 fallback |
| Exact length | 每个 N 保持 `7+4N` |
| Immutable Plan | N/element prefill 永不成为 action |
| Denominator | 所有 attempt 保留，无 retry/replacement/filter/rerank |
| Refiner freeze | no grad、model_494 exact800、batch1、seed replay |
| Reward attribution | composition 不进入同 Plan Body advantage |
| Exact resume | 中断恢复后的下一段 trajectory 与 uninterrupted run 一致 |

## 12. 32→64→256→1024 实验阶梯

### Gate −1：token coverage 与 legal mass，零训练

恢复并只读扫描 train/validation/test、R03D/R03E generated bodies 和 B0
fixed-panel constrained logits：

- 每个 token 的 target count、position 和 train-frequency bucket；
- 每步 legal support 中 train-unseen/rare probability mass；
- emitted unseen/rare rate；
- 与 body/refine/Direct/strict/meta outcome 的 paired attribution。

若 unseen/rare mass 可忽略，明确保留原 support；若不可忽略，先完成一个
B0-compatible support-only paired-32/64 screen。两种情况下都必须冻结唯一
support SHA 后才能进入 R0。不得根据 endpoint 逐次修改 support。

### R0：单元测试与 synthetic logits

无科学 GPU 任务：

- 特殊 token/tokenizer 合同；
- legal-support parity；
- PL probability；
- token+position log-prob；
- identity ratio/KL；
- exact resume；
- singleton debug oracle。

全部通过才允许真实 rollout。

### R1：32，纯工程，无训练

比较 frozen B0：

- D1 greedy；
- D1-PL K=1。

只看工程安全：

- exact length 100%；
- Plan match 100% among body outputs；
- no illegal token；
- no cross-group action；
- duplicate-coordinate 不增加；
- body completion 最多差 `1/32`；
- position entropy finite 且非零；
- replay/identity/resume 全通过；
- no new failure class。

不得根据 32 的 S.U.N. 选择算法或 reward。

### R2：64，sampler 安全与 A/B calibration，无训练

- greedy vs PL-noRL paired；
- 同一 proposals 同时取 A/B labels；
- 预注册 `tau_pos` calibration 只做一次；
- 至少 25% 的 `G=8` Plan groups 有非零 advantage；
- PL completion `>= greedy -1/64`；
- structure 不差超过 `2 pp`；
- no duplicate/Z-before-XY/new failure；
- A/B 相关性过门；
- 对 16 个 proposals 做双 process exact800，只估计 process noise。

若 PL 安全或 trajectory probability 不闭合，RL 工程停止。

### R3：256，小 LoRA 因果筛选

候选 arms：

1. no-RL PL control；
2. A-only；
3. B-direct；
4. A→B，仅在 A/B calibration 通过时。

要求：

- 同一 frozen P0 Plans；
- 同一 parent B0；
- 相同 optimizer update budget；
- 相同 group size；
- 相同 B-label budget；
- fresh held-out 256；
- validation 只选一个 winner；
- final evaluator 此时仍封存。

晋级门：

- strict point delta `>0`；
- meta point delta `>=0`，且不破预注册 noninferiority margin；
- completion/structure loss `<=2 pp`；
- novelty、uniqueness 各不下降超过 `5 pp`；
- exact length、Plan match、no-new-failure 全通过；
- KL/ESS/clip/entropy 全通过；
- 至少两个独立 128 seed block 方向一致；
- reward proxy 上升、独立 selector 下降时立即停止。

### R4：1,024，一次性确认

- `4 × 256` 独立 scientific panels；
- primary causal comparison：winner RL vs no-RL PL control；
- secondary system comparison：现有 frozen greedy D1；
- same frozen Planner；
- same model_494 exact800；
- common completed hull snapshot；
- unknown/failure false；
- final panel 一次性开启。

建议门：

- strict：至少 3/4 panels 为正，mean `>0`，hierarchical paired 95% CI
  lower `>=0`；
- meta：至少 3/4 非负，mean `>=0`，one-sided 95% lower `>-2 pp`；
- structure noninferiority lower `>-2 pp`；
- no new failure class；
- novelty/uniqueness 无预注册恶化；
- independent MLIP/DFT audit 方向一致；
- exact McNemar、paired bootstrap、panel sign stability 全报告。

未通过只保留 diagnostic checkpoint，不 promotion。

## 13. 粗略算力边界

依据现有 exact800/batch1 allocation 上界，仅用于规划：

| Labels/arm | 单臂 model_494 上界 |
|---:|---:|
| 64 | `<=1.5 A800 GPUh` |
| 128 | `<=3 A800 GPUh` |
| 256 | `<=6 A800 GPUh` |
| 1,024 | `<=24 A800 GPUh` |

paired control/candidate：

- 256：约 `<=12 A800 GPUh`；
- 1,024：约 `<=48 A800 GPUh`；
- 再加 greedy 第三臂约 `<=72 A800 GPUh`。

一次 online round 若为 `32 Plans × G8 = 256` 个 B labels，约
`<=6 A800 GPUh`；四轮约 `<=24 GPUh/model`，尚未计 Body rollout、
backward 和独立 evaluator。因此长期维护两个 RL LoRA 没有成本优势。

这些是上界，不是实测承诺。R1 必须记录 Body rollout、A judge、B exact800、
backward 的实际 wall/GPU time。

## 14. 主要失败模式

1. tokenizer 重建或 special-token ID 漂移；
2. B0 embedding/output head 被意外解冻；
3. full-vocab 而非 legal-support log-prob；
4. greedy top-k 漏 position gradient；
5. 只记录 selected token，漏组内 candidates；
6. temperature/mask/CFG 不一致产生伪 importance ratio；
7. strict reward 把 meta-only 推到 `>0.1`；
8. 同一 judge 训练、选模、最终评测；
9. A/B 负相关导致 warm-start negative transfer；
10. 只 refine A 预测最好样本造成 selection bias；
11. stale replay 导致 ESS collapse；
12. checkpoint 只能加载、不能 exact resume；
13. composition reward 在 frozen Plan 内零 variance；
14. uniqueness 定义有 batch order bias；
15. model_494 CUDA/process noise 被当作 scientific gain；
16. 同一 256 ledger 四次重复被误写成四个独立 panels；
17. retry/filter/replacement 破坏 all-attempt denominator；
18. 将 PL 与 safe-axis、Planner 新方案一次性组合。
19. 未审计 train-unseen/rare action mass 就启动 RL；
20. 在同一次实验同时做 token compression、support contraction 和 RL。

## 15. 最终建议

### 现在可以做

1. 新建 RL workstream 和协议，不修改 V3 历史证据；
2. 先完成 Gate −1 token coverage/legal-mass audit；
3. 独立决定保留原 support，或做一次 support-only screen；
4. 实现特殊 token、legal-support、PL joint likelihood、trace/replay；
5. 冻结 B0 embedding/output head，只训练新 RL LoRA；
6. 跑 R0 tests；
7. 跑无训练 32；
8. 再跑无训练 64 和 A/B calibration。

### 现在不能做

- 直接启动两个长期 LoRA；
- coverage 未审计就启动 RL；
- 同时更换 token representation 和训练 RL；
- 直接用旧 TraceRL；
- 直接在 safe-axis 上训；
- 用 full-vocab token log-prob；
- 训练时解冻整套 6.39 GB B0 modules-to-save；
- 把 composition reward 塞进 frozen-Plan Body advantage；
- 用 final CHGNet/MP endpoint 调 reward；
- 先看结果再选 `tau_pos`、reward weights 或 threshold；
- 自动扩到 256/1,024。

### 最有价值的下一步

> 先发布冻结的
> `train/val/test/generated × token-frequency × legal-mass × endpoint`
> coverage ledger，决定保留原 support 还是先做 support-only screen。随后才做
> 不训练模型的 `R0 + paired-32 D1-greedy vs D1-PL`，证明特殊 token 身份、
> 动态合法支持、token+position behavior log-prob、trajectory replay、exact
> resume 和 D1 group 不变量全部闭合。

## 16. 主要一手依据

- [Large Language Diffusion Models / LLaDA](https://arxiv.org/abs/2502.09992)
- [Mask-Aware Policy Gradients for Diffusion Language Models](https://arxiv.org/abs/2607.15200)
- [DCoLT / LLaDOU](https://arxiv.org/abs/2505.10446)
- [Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective / ESPO](https://arxiv.org/abs/2512.03759)
- [MaskGRPO](https://openreview.net/forum?id=9nxCJP4q0i)
- [CRYSTAL: Coordinated Multi-Objective Reinforcement Learning for Crystal Generation](https://openreview.net/pdf/94d95333b625bc19463eca098ff60038d639d590.pdf)
- [PLaID++: A Preference Aligned Language Model for Targeted Inorganic Materials Design](https://arxiv.org/abs/2509.07150)
