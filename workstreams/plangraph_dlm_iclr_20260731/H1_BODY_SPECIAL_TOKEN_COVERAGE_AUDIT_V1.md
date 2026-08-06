# H1 Body-DLM 特殊 Token 覆盖审计 V1

状态：`read_only_audit_design_only`

日期：2026-08-04

适用锚点：H1-A2 `P0 + B0/R5-C exact-length DLM + D1`

本文回应一个关键问题：冻结 tokenizer 中定义了 `2,481` 个晶体特殊
token，但当前 exact-length Body-DLM 实际使用、采样和得到正监督的 token
远少于这个数字。

本文不授权修改 tokenizer、checkpoint、数据、sampler、训练、RL、refinement
或评测。

## 1. 结论

这个问题真实存在，但必须区分三种“没有用上”：

1. **R5-C exact-length 永远不可达**
   - `20` 个 `<Sxx>`；
   - `<EMPTY>`、`<X_PAD>`、`<Y_PAD>`、`<Z_PAD>`；
   - 共 `24` 个。
   - 此外，formal H1 lightweight decoding mask 永远禁止
     `<LA_000>/<LB_000>/<LC_000>`；它们属于 exact schema vocabulary，
     但不是实际 legal action。
2. **Body policy 不采样，但作为条件使用**
   - `20` 个 atom-count token；
   - `94` 个 element token；
   - H1 推理时 `N` 和 element slots 由 Planner 预填。
3. **理论上可采样，但数据覆盖稀疏或未覆盖**
   - 三组 length、angle、coordinate token；
   - 这是对后续 RL 真正危险的部分。

在本地冻结的 `9,046` 条 MP-20 exact-length held-out records 中：

| 指标 | 数值 |
|---|---:|
| 定义的特殊 token | 2,481 |
| 至少出现一次 | 1,437 |
| 没有出现 | 1,044 |
| observed coverage | 57.92% |
| schema-defined numeric stochastic token union | 2,343 |
| 其中至少出现一次 | 1,330 |
| 其中没有出现 | 1,013 |
| stochastic-support observed coverage | 56.76% |

因此，用户关于“大量特殊 token 没有用上”的判断成立。

但当前本地只有完整 held-out JSONL，没有完整 `27,136` 条训练 JSONL。
所以 `1,044` 是 **held-out unseen**，不能直接写成
“training unseen”。任何正式 support pruning 前，必须先扫描冻结 train
corpus。

## 2. 冻结证据

| 文件 | SHA-256 |
|---|---|
| `data/dlm_sft/mp_20_r5_exact_length/test.jsonl` | `9a5c542888029ee785c3f922e2922a61a6cb92abf028b1fe197c8920d48efe8b` |
| `data/dlm_sft/mp_20_r5_exact_length/stats.json` | `f3eda4befe498d948bed75315a1a7567164b6e17528469210e152d01570ae5ca` |
| `crystal_dlm/fixed_slot.py` | `aa06e8c240a47c3f3696239bcf07adbcf73c95cab1b3f0d7be07ba560a6476eb` |
| `crystal_dlm/dynamic_crystal.py` | `911e032eefbaf9b38dd856732adaf261307dad6793d6b7c044ffc3f9f87146e5` |
| historical `r5_dynamic_length.py` | `c022ddda92caac1c60b91b239c1bc155c734cad86b595282ce9724e239d002c6` |
| historical `llada_sft.py` | `b6c6a55f6fd018d1e132f0fe9f7a3fd7662643b871b9f7a28bf7564083af371e` |

统计只读取每条记录的 `answer` 并使用冻结
`tokenize_answer_text()` 提取晶体 token。

## 3. 逐 family 覆盖

| Family | Defined | Seen | Unseen | Seen rate | Held-out occurrences | Seen token 中频次 ≤1 | Seen token 中频次 ≤10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `N` | 20 | 20 | 0 | 100.0% | 9,046 | 0 | 0 |
| `LA` | 501 | 157 | 344 | 31.3% | 9,046 | 33 | 80 |
| `LB` | 501 | 158 | 343 | 31.5% | 9,046 | 34 | 75 |
| `LC` | 501 | 235 | 266 | 46.9% | 9,046 | 60 | 132 |
| `AA` | 179 | 160 | 19 | 89.4% | 9,046 | 9 | 57 |
| `AB` | 179 | 160 | 19 | 89.4% | 9,046 | 10 | 56 |
| `AG` | 179 | 160 | 19 | 89.4% | 9,046 | 6 | 40 |
| `S` | 20 | 0 | 20 | 0.0% | 0 | 0 | 0 |
| `E` | 94 | 87 | 7 | 92.6% | 93,158 | 0 | 3 |
| `X` | 101 | 100 | 1 | 99.0% | 93,158 | 0 | 0 |
| `Y` | 101 | 100 | 1 | 99.0% | 93,158 | 0 | 0 |
| `Z` | 101 | 100 | 1 | 99.0% | 93,158 | 0 | 0 |
| empty/pad | 4 | 0 | 4 | 0.0% | 0 | 0 | 0 |

主要问题非常集中：

- 三个 length axis 合计有 `953` 个 held-out unseen token；
- 三个 angle axis 合计有 `57` 个 held-out unseen token；
- 三个 coordinate axis 各自只有 bin `100` 未出现；
- held-out 中未出现的 element 为
  `Ne/Kr/Po/At/Rn/Fr/Ra`，但 element 在 H1 Body policy 中是预填条件，
  不是随机动作。

Observed numeric ranges：

| Family | Observed bin range |
|---|---|
| `LA` | 24–367 |
| `LB` | 24–367 |
| `LC` | 25–467 |
| `AA` | 7–171 |
| `AB` | 7–171 |
| `AG` | 7–170 |
| `X/Y/Z` | 0–99 |

当前 length vocabulary 为每个 axis 独立定义 `0–500`，即 `0–50 Å`。
formal sampler 会额外 mask 三个 zero-length token，所以实际全局可达的 numeric
action union 至多为 `2,340`；其余范围为兼容极端结构保留，但大量 bin 没有
得到 held-out 正覆盖。

## 4. 为什么这不仅是参数浪费

### 4.1 初始化

历史 R5-C 创建新 token 时：

- 所有新 input embedding 先初始化为旧 vocabulary embedding 的均值；
- 所有新 output rows 同样初始化为旧 output rows 的均值；
- R5-C launcher 没有启用后来的 element semantic initialization。

因此，numeric token 最初没有 ordinal 邻近关系：

```text
<LA_100> 与 <LA_101>
```

在参数空间中不天然比 `<LA_100>` 与 `<LA_400>` 更接近。

### 4.2 SFT 目标与推理支持不一致

历史 R5-C SFT：

- 随机 mask answer positions；
- 对 masked target 使用 full-vocabulary one-hot cross entropy；
- 没有在 loss 中按 position 把 softmax 限制到该字段的 legal support。

因此没有作为 target 出现的 token：

- input row 没有正输入样本；
- output row 没有正 target；
- 但会作为 full-vocabulary negative class 接受梯度。

推理时才应用字段 mask，使这些 token 又成为对应 family 的合法候选。
这是明确的 train–inference support mismatch。

### 4.3 对 RL 更危险

普通 B0 sampling 可能已经把 unseen token 的 logit 压得很低；但 RL 会主动寻找
reward 梯度。如果 unseen/rare action 偶然得到高 reward，policy optimization
可能把它放大，形成：

- lattice tail exploitation；
- 极端但 parser-valid 的晶胞；
- refiner 承担过大的 basin correction；
- strict 提升但 meta 下降；
- importance ratio 和 KL 被低基准概率放大。

所以在 action support 未冻结前，不应该启动正式 Body RL。

## 5. 它是不是当前主要性能瓶颈

**不是已经证明的主瓶颈。**

现有 R03E 在 refine-success denominator 上：

- D1 structure validity：`99.7967%`；
- safe-axis structure validity：`99.6976%`。

真正 post-refiner structure-invalid 只有 2/3 个。当前 `comp_valid` 和 joint
validity 的主要瓶颈仍是 Planner chemistry，而不是特殊-token 覆盖。

更准确的判断是：

| Endpoint | 当前判断 |
|---|---|
| Body completion | 可能受 rare tail 影响，但尚未归因 |
| Post-refiner structure validity | 已近饱和，不是主要解释 |
| Refiner displacement/basin burden | 很可能相关，尚未量化 |
| strict/meta S.U.N. | 可能相关，尤其是 meta 尾部；尚无 paired attribution |
| RL stability | 明确是前置风险 |

正式因果判断需要把每个 generated token 的 train frequency 与：

- body failure；
- refine displacement；
- Direct failure；
- `E_hull`；
- strict/meta transition

按 ordinal 连接。

## 6. Axis sharing 的定量价值

现有 token 把 axis 信息编码在 token identity 中：

```text
<LA_041>, <LB_041>, <LC_041>
<AA_090>, <AB_090>, <AG_090>
<X_050>,  <Y_050>,  <Z_050>
```

但 sequence position 已经唯一确定 axis。因此可以改成：

```text
<L_041>
<A_090>
<C_050>
```

在 held-out corpus 中合并 axis 后：

| Shared family | Defined | Seen | Unseen | Occurrences |
|---|---:|---:|---:|---:|
| `L` | 501 | 242 | 259 | 27,138 |
| `A` | 179 | 163 | 16 | 27,138 |
| `C` | 101 | 100 | 1 | 279,474 |

H1 Body policy 的 schema-defined numeric stochastic union 可从：

```text
current: 3×501 + 3×179 + 3×101 = 2,343
shared:      501 +     179 +     101 =   781
```

下降 `66.67%`，同时每个 bin 聚合三个 axis 的正监督。

仓库已经存在 fixed-slot 的实验 scaffold：

- `crystal_dlm/fixed_slot_compressed.py`；
- `scripts/convert_fixed_slot_checkpoint_to_compressed.py`；
- `run_mp20_stcompress_ablation.sh`。

但它不是当前 exact-dynamic R5-C 的已验证结果，本地也没有恢复出的正式
terminal。因此它只能作为实现先例，不能当成 axis sharing 已经成功的证据。

## 7. 建议的单变量阶梯

### T0：完整 coverage audit，零模型变化

先恢复/只读扫描冻结的：

- train `27,136`；
- validation `9,047`；
- test `9,046`；
- R03D/R03E generated body token；
- B0 fixed-panel constrained logits。

每个 token 必须记录：

- train/validation/test target count；
- 在哪个 generation position 出现；
- B0 legal-support probability mass；
- generated emission count；
- body/refine/Direct/S.U.N. outcome；
- `seen / rare / train-unseen / unreachable` 标签。

最重要的两个统计是：

```text
legal probability mass assigned to train-unseen tokens
legal probability mass assigned to rare tokens
```

不能只统计 token type coverage；如果 B0 已经给 unseen token 近零概率，它们
对当前 sampling 的实际危害较小。

### T1：B0-compatible support screen

只有 T0 证明 unseen/rare mass 非忽略时，注册一个 sampler-only 单变量：

- tokenizer ID、B0 checkpoint、D1、temperature、Plan、seed 全冻结；
- 不删除、不重排 tokenizer；
- 只从 legal support 中移除冻结 train corpus 零正样本的 numeric token；
- count/element prefill 不变；
- 无 retry、repair、filter、rerank；
- 不训练。

先 paired-32 工程，再 paired-64：

- completion 不下降；
- duplicate/volume failure 不增加；
- numeric distribution 不坍缩；
- structure/novelty/unique 均作非劣门槛；
- 不根据 32/64 的 S.U.N. 调 support。

这一步的结果必须单独归因于 **support contraction**，不能同时加入 RL。

### T2：exact-dynamic axis-shared SFT

这是更干净的长期方案，但属于新表示与新 checkpoint：

- 保持 `7+4N`；
- 保持 count/element Plan prefill；
- 只把 axis-specific numeric tokens 合并；
- position 继续表达 axis；
- 从冻结训练数据重新构建；
- 用独立的 SFT/Direct/S.U.N. gate；
- 不同时改变 schedule、reward 或 RL。

建议先测试：

```text
length sharing only
```

因为 `953/1,503` 个 held-out-unseen length token 是当前最大覆盖缺口。
只有该单变量通过后，再考虑 angle/coordinate sharing。

### T3：ordinal numeric modeling

若 axis sharing 后 length 仍有明显长尾，再考虑：

- 邻域平滑 target；
- axis embedding + numeric-bin embedding；
- Fourier/spline ordinal residual；
- coordinate 的 periodic smoothing。

这比简单删除 token 更能表达连续邻近关系，但模型改动最大，不应作为第一枪。

## 8. 对 RL 方案的修订

原 RL 阶梯前新增 **Gate −1**：

```text
Gate −1: token coverage + legal-mass audit
  -> T1 support-only screen, or保留原 support
  -> freeze one sampler support
  -> R0 exact trajectory/logprob
  -> paired-32 D1-greedy vs D1-PL
  -> reward calibration
  -> RL
```

RL 合同必须增加：

- policy/reference 使用完全相同的 frozen support；
- 每步保存 support SHA；
- 保存 action 的 train-frequency bucket；
- 报告 unseen/rare probability mass 与 emission rate；
- support 在一个 RL run 内不可变化；
- 不允许 policy 通过恢复被 mask 的 dead token 投机；
- token representation change 与 reward/RL 永远不是同一实验因子。

## 9. 最终判定

- “大量特殊 token 没有用上”：**确认，held-out 层面非常明显**。
- 它是当前 Planner composition 问题的解释：**不是**。
- 它是启动 Body RL 前必须解决的风险：**是**。
- 立即删除/重排 2,481-token vocabulary：**NO-GO**，会破坏 B0 checkpoint。
- 先做完整 coverage/legal-mass audit：**GO，最高优先级**。
- B0-compatible zero-train-support mask：**CONDITIONAL GO**。
- exact-dynamic length-axis sharing：**CONDITIONAL GO，最有价值的新表示单变量**。
- 同时改 token、schedule 和 RL：**NO-GO**。

最有价值的下一步不是训练，而是发布一个冻结的
`train/val/test/generated × token-frequency × legal-mass × endpoint`
coverage ledger。它将决定是先做 T1 support contraction，还是直接保持 B0
support 进入 mask-aware RL。
