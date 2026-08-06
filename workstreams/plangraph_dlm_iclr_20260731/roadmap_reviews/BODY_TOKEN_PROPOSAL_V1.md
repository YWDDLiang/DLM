# H1 Body-DLM Token/Support 单变量提案 V1

状态：`design_only_no_execution_authorization`

日期：2026-08-04

适用锚点：

```text
H1-A2 epoch-2 Planner P0
  + B0/R5-C exact-length Body-DLM
  + canonical D1
  + frozen CrysLLMGen model_494 exact-800
  + frozen Direct / original A100-CHGNet S.U.N.
```

时间硬约束：

- ICLR 2027 摘要截止：2026-09-18 AOE，距本文日期 45 天；
- ICLR 2027 全文截止：2026-09-25 AOE，距本文日期 52 天；
- 本路线只保留一个必须在两周内、即不晚于 2026-08-18，给出
  paired-256 机制信号的模型候选；
- 本路线不与 RL、schedule、Planner、refiner、reward 或 evaluator 修改组合。

本文是只读审计后的路线建议，不授权数据恢复、代码修改、训练、生成、
refinement、S.U.N.、提交作业或自动下游。

## 1. 决策

### 1.1 唯一保留的两周候选

只保留：

> **PILS-L：Position-Indexed Length Sharing**
>
> 保持 exact `7+4N`、D1、Plan composition、角度 token、坐标 token 和所有
> downstream 不变，只把 `<LA_k>/<LB_k>/<LC_k>` 合并为一个
> `<L_k>` family；A/B/C 轴语义由 answer position 唯一决定。

选择它的原因：

1. held-out 的 numeric coverage 缺口几乎全部来自 length：
   `953/1,013 = 94.08%` 的 held-out-unseen numeric token identity 是
   LA/LB/LC；
2. lattice length 直接决定体积、密度和 refiner 所处 basin，因而比删 dead
   token 更可能影响 completion、refiner displacement 和 meta S.U.N.；
3. 它是一个单一 representation factor，不改变每一步的数值分辨率、合法
   bin 数、D1 顺序或物理约束；
4. 仓库已有 fixed-slot shared-token、row-average checkpoint conversion
   和 round-trip test scaffold，虽然不能直接用于 R5 exact-dynamic，但可
   降低实现风险；
5. 相比 all-axis sharing，它集中处理真正稀疏的 family，不同时扰动已经
   约 89%/99% 覆盖的 angle/coordinate family；
6. 相比 neighbor smoothing，它没有 kernel bandwidth、periodic boundary
   或 label-distribution 超参数搜索。

### 1.2 本轮不运行的方案

| 方案 | 本轮处置 | 原因 |
|---|---|---|
| Gate-1 full-corpus coverage/legal-mass | 必做前置测量 | 零模型变化，不是 treatment |
| B0-compatible support contraction | 只计算反事实 removed mass，不运行 | 快但主要是 sampler safety/工程清理，论文创新弱；与 PILS-L 同跑会增加选择自由度 |
| PILS-L length-axis sharing | **唯一 active candidate** | 两周内最有希望给 completion/meta 机制信号 |
| 全轴 sharing | 封存 | angle/coordinate 已高覆盖；改动面更大，难在两周内归因 |
| ordinal/neighbor smoothing | 封存 | 有方法潜力，但需要 kernel/boundary calibration，时间内不是单一固定超参 |
| 删除 `S/EMPTY/PAD` | NO-RUN | 对 R5 exact 动作不可达，只是 checkpoint/tokenizer 清理 |
| RL | NO-RUN | token/support 与 RL 不能在同一实验改变 |

若 PILS-L 在任一 gate 停止，本轮 token workstream 结束；不得静默切换到
support contraction、all-axis 或 smoothing。

## 2. 冻结事实与因果边界

### 2.1 当前 R5-C Body

权威实现：

- `crystal_dlm/r5_dynamic_length.py`：
  exact layout、Plan match、schema support、D1；
- `scripts/llada_sft.py`：
  answer-only IID masked-denoising SFT；
- `crystal_dlm/llada_generation.py`：
  schema/count/duplicate/volume masks 和 low-confidence reveal；
- `scripts/sample_llada_h1a2_factorial_body.py`：
  H1 formal ordinal/batch-1 sampling；
- `H1_R03_D2_SCHEDULE_DIAGNOSIS_V1.md`：
  mixed-axis D2 的 duplicate collapse；
- `H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md`：
  32→64→256→refine/S.U.N. 完整证据。

给定 Plan 中的 `N`，answer 为：

```text
0           N
1..3        LA, LB, LC
4..6        AA, AB, AG
7+4k        element
8+4k        X
9+4k        Y
10+4k       Z
```

正式 H1 在生成前预填 count 和全部 element positions，因此随机动作数为：

```text
6 + 3N,  N in [1,20]
```

即每个 body 只有 9 到 66 个 committed stochastic actions。

### 2.2 comp_valid 不能被误归因

P0 Plan 已固定 `N/elements/counts`，B0 sampler 又把 count/element token
预填并冻结。同一 Plan 下，PILS-L 不改变 formula。

因此：

- conditional composition validity 应与 Plan 一致；
- raw all-attempt composition count 可能因 body completion 增加而上升；
- 这种 raw 增长必须写成“更多 Plan 成功进入/完成 downstream”，不能写成
  “Body 学会了更好的化学计量”；
- 真正提升 SMACT chemistry 的主干仍是 Planner，不应与本 token 实验同时改。

PILS-L 的合理目标是：

1. length-token calibration；
2. body completion；
3. proposal lattice/volume 分布；
4. refiner displacement/basin burden；
5. 在 strict 不退化的条件下改善 meta S.U.N.。

## 3. 当前 coverage 与 action-space 核算

### 3.1 定义 vocabulary

冻结 checkpoint 有 2,481 个晶体特殊 token：

| Family | Token 数 |
|---|---:|
| N | 20 |
| LA/LB/LC | 1,503 |
| AA/AB/AG | 537 |
| S | 20 |
| E | 94 |
| X/Y/Z | 303 |
| EMPTY/PAD | 4 |
| 合计 | 2,481 |

H1 Body stochastic numeric token identity union 为：

```text
3*501 + 3*179 + 3*101 = 2,343
```

formal volume mask 另外禁止三个 zero-length token，所以当前实际可达的
numeric token identity union 上限为：

```text
3*500 + 3*179 + 3*101 = 2,340
```

### 3.2 held-out 证据

当前本地只有完整 held-out `9,046` records，不能把它冒充 train coverage。

| Family | Defined | Seen | Unseen | Held-out occurrence |
|---|---:|---:|---:|---:|
| LA | 501 | 157 | 344 | 9,046 |
| LB | 501 | 158 | 343 | 9,046 |
| LC | 501 | 235 | 266 | 9,046 |
| AA | 179 | 160 | 19 | 9,046 |
| AB | 179 | 160 | 19 | 9,046 |
| AG | 179 | 160 | 19 | 9,046 |
| X/Y/Z each | 101 | 100 | 1 | 93,158 each-axis total |

三轴 length token identity：

```text
defined = 1,503
seen identities = 157 + 158 + 235 = 550
identity seen rate = 36.59%
unseen identities = 953
```

合并后：

```text
defined shared-L bins = 501
seen bin union = 242
seen rate = 48.30%
unseen bins = 259
total positive occurrences = 27,138
```

这不是说每个 shared bin 都有三倍频次，但每个出现过的数值 bin 可以聚合
来自三个 axis 的监督。

### 3.3 必须区分三种“动作空间”

| 量 | 当前 | PILS-L | 变化 |
|---|---:|---:|---:|
| 晶体特殊 token 总数 | 2,481 | 1,479 | `-1,002`, `-40.39%` |
| 模型总 vocab | 128,830 | 127,828 | `-1,002`, `-0.778%` |
| numeric stochastic identity union | 2,343 | 1,341 | `-1,002`, `-42.77%` |
| 可达 numeric identity union | 2,340 | 1,340 | `-1,000`, `-42.74%` |
| 单个 length position legal branches | 500 after zero mask | 500 | **不变** |
| 单个 angle position legal branches | 179 | 179 | 不变 |
| 单个 coordinate position legal branches | up to 101 | up to 101 | 不变 |
| committed steps | `6+3N` | `6+3N` | 不变 |

所以 PILS-L 缩减的是：

- output/embedding 中互相重复的 token identity；
- 三轴分开的监督稀疏性；
- 全局 active vocabulary。

它**不**通过硬裁剪减少每个 length action 的数值选择数，也不降低分辨率。
若论文声称“每步 branching factor 缩小 3 倍”将是错误的。

若 hidden width 为 `d`，仅 input/output 两个 vocab-shaped matrix 的 row
减少为：

```text
2 * 1,002 * d parameters
```

例如 `d=4,096`、BF16 时约 15.7 MiB。相对 8B 模型和 128,830 总 vocab，
存储/softmax 节省很小；PILS-L 的主张应是监督共享与校准，不是算力压缩。

## 4. 五种方案的量化比较

### 4.1 Gate-1：full-corpus coverage/legal-mass

模型、tokenizer、sampler、checkpoint 全部不变。

必须恢复并锁定：

- train 27,136；
- validation 9,047；
- test 9,046；
- R03D body attempts；
- R03E/G/H endpoint mapping；
- frozen B0 fixed 256 panel 的 constrained logits/trajectory states。

对每个 token `v` 记录：

```text
train_count(v)
val_count(v)
test_count(v)
position/axis
R03 emission_count(v)
body/refine/Direct/S.U.N. endpoint
```

对每个 B0 state `s` 和 family `f`：

```text
m_zero(s,f)
  = sum_{v legal at (s,f), train_count(v)=0} p_B0(v | s,f)

m_rare(s,f)
  = sum_{v legal at (s,f), 1<=train_count(v)<=10} p_B0(v | s,f)
```

其中：

```text
p_B0 = softmax(schema_and_dynamic_masked_logits / 0.7)
```

必须报告 mean/p50/p95/max、按 axis/N 分层、真实 emission rate；不能只报
token type coverage。

Gate-1 允许 PILS-L 进入实现的预注册条件：

1. 三个 split 行数、hash、formula/Plan match 与历史合同一致；
2. length 占全部 numeric `train-unseen + rare` identity 至少 50%；
3. axis pooling 后 shared-L 的 median positive target count 至少提高 2 倍；
4. B0 length 的 `mean(m_zero+m_rare) >= 0.5%`，或
   `p95(m_zero+m_rare) >= 2%`；
5. 不查看/使用 S.U.N. 来决定 token support 或 sharing 范围。

若第 2–4 条全部不满足，PILS-L 停止，回到 Planner 主线。

预计成本：

- full-corpus CPU ledger：数小时；
- fixed-256 legal-mass capture：一张 A800，不超过半个 GPU-day 的规划上界；
- 最迟 2026-08-06 冻结报告和唯一 candidate decision。

### 4.2 B0-compatible support contraction

定义：

- tokenizer/vocab/checkpoint 完全不变；
- 只从每个 position 的 legal support 删除 `train_count=0` numeric tokens；
- `N/E` prefill、D1、temperature、seeds、refiner 全不变；
- 不删 rare positive tokens；
- 不训练。

量化：

```text
total vocab:       128,830 -> 128,830
special tokens:      2,481 -> 2,481
active union:        2,343 -> 2,343 - U_train_zero
per-position size:   |L_f| -> |L_f| - U_train_zero,f
```

held-out-only 的 `1,330/2,343` 不能作为正式 support；它仅说明上限上可能有
`1,013` 个稀疏 identity。

兼容性：

- B0 byte-compatible：是；
- tokenizer 改变：否；
- 需要重训：否；
- sampler support SHA 改变：是；
- historical old log-prob 仍不可直接复用，因为 behavior support 改变。

预期：

- completion：若 `m_zero` 明显，可能改善；
- meta：方向未知；可能去除 lattice tail，也可能损失有效稀有晶胞；
- comp_valid：不会改善 conditional chemistry；
- 方法创新：弱，更像安全校准/工程清理。

即时资源下的估计：

- time-to-64：Gate-1 后 2–3 天；
- time-to-256：Gate-1 后 4–6 天；
- 无 SFT 成本。

本轮不运行。Gate-1 只保存它会移除的概率质量作为反事实审计。

### 4.3 PILS-L：length-axis sharing

定义：

```text
<LA_k>, <LB_k>, <LC_k> -> <L_k>
```

position 1/2/3 仍分别解释为 a/b/c。角度、坐标、count、element 和 compatibility
token 不变；本轮不顺便删除 `S/EMPTY/PAD`。

兼容性：

- B0 tokenizer byte-compatible：否；
- B0 transformer/LoRA warm-start compatible：是，但必须显式 row remap；
- exact `7+4N`：是；
- D1 和 action step 数：不变；
- downstream arrays/graphs/refiner：保持相同接口；
- 需要新 tokenizer SHA、token map、checkpoint SHA：是；
- 需要 SFT：是。

初始化：

```text
wte(<L_k>) =
  mean(wte(<LA_k>), wte(<LB_k>), wte(<LC_k>))

ff_out(<L_k>) =
  mean(ff_out(<LA_k>), ff_out(<LB_k>), ff_out(<LC_k>))
```

其他 vocab-shaped rows按 token string 一一复制；LoRA tensors byte-for-byte
复制。row averaging 是 representation factor 的一部分，不得再做 semantic、
neighbor 或 frequency-weighted initialization。

为了隔离额外 continuation training，必须有 matched control：

| Arm | 初始化 | Token representation | Training |
|---|---|---|---|
| C0 | frozen B0 | 原 LA/LB/LC | matched one-epoch continuation |
| L1 | row-remapped frozen B0 | shared L | matched one-epoch continuation |

两个 arm 冻结：

- 同一 27,136 train rows、9,047 validation rows；
- same record order 和 stateless IID corruption keys；
- one epoch；
- per-device batch 1；
- two A800 DDP；
- grad accumulation 8；
- effective global batch 16；
- exactly `27,136 / 16 = 1,696` optimizer updates；
- BF16；
- cosine；
- warmup 100；
- min LR ratio 0.2；
- continuation LR 预注册为 `1e-6`，不做 LR sweep；
- existing B0 LoRA targets不变；
- wte/ff_out trainable boundary在两臂一致；
- full-vocabulary CE 保持不变；
- 不加入 legal-support CE、neighbor smoothing 或 D1 planned corruption；
- 使用 final step 1,696，不用 S.U.N. 选 checkpoint。

总训练预算：

```text
2 arms * 1,696 updates = 3,392 optimizer updates
```

每个 update 使用 2 张 A800。queue-excluded 规划上界为两臂总计 24–48 小时；
首个实际 run 必须记录真实 wall/GPU time，本数字不是性能承诺。

预期：

- completion：中等可能；更平滑的 length head 可减少非法/极端 lattice；
- meta：本轮 token 方案中最高的可执行潜力，因为 density/volume/basin 与
  hull tail 有直接联系；
- strict：目标是非劣，不允许复制 safe-axis 的 strict/meta 极化；
- comp_valid：只可能通过 upstream completion 改变 raw count；
- 论文创新：中等；必须由 coverage→NLL/calibration→refiner burden→meta 的
  机制链支撑，而不是只报 token 数变少。

### 4.4 全轴 sharing

定义：

```text
LA/LB/LC -> L
AA/AB/AG -> A
X/Y/Z    -> C
```

若沿用 fixed-slot compatibility token：

| 量 | 当前 | 全轴 shared | 变化 |
|---|---:|---:|---:|
| 特殊 token | 2,481 | 917 | `-1,564`, `-63.04%` |
| 总模型 vocab | 128,830 | 127,266 | `-1,564`, `-1.214%` |
| numeric stochastic union | 2,343 | 781 | `-1,562`, `-66.67%` |
| 可达 numeric union | 2,340 | 780 | `-1,560`, `-66.67%` |
| per-position branching | 不变 | 不变 | 0 |

但 held-out 合并后的覆盖为：

```text
L: 242/501
A: 163/179
C: 100/101
```

angle/coordinate 已经高覆盖，全轴 sharing 的新增收益主要是参数 tying，而
不是修复大 coverage 缺口。它还同时改变 lattice、angle、periodic coordinate
三个语义族，若 endpoint 变化无法判断是哪一族导致。

兼容性：

- 需要新 tokenizer、row remap 和 SFT；
- exact layout/downstream 可兼容；
- B0 不 byte-compatible；
- fixed-slot scaffold 可作实现先例，但没有 exact-dynamic terminal。

queue-excluded 估计：

- time-to-64：8–10 天；
- time-to-256：12–15 天；
- 已经吃掉全部两周窗口，且归因较差。

结论：本轮封存，属于后续 representation-scope ablation，不是第一枪。

### 4.5 Ordinal / neighbor smoothing

候选形式：

```text
target bin k:
  (1-lambda) * one_hot(k)
  + lambda * normalized_kernel(k-d ... k+d)
```

它可保持原 tokenizer、B0 初始化和 sampler support，但需要修改 SFT loss。

必须分别定义：

- length 的线性边界；
- angle 的物理边界；
- coordinate 0/100 的周期等价；
- kernel width；
- lambda；
- train-unseen bin 是否获得正质量；
- inference 是否仍使用原 categorical support。

量化：

```text
vocab reduction: 0
active union reduction: 0
per-step branching reduction: 0
requires retraining/continuation: yes
```

预期：

- completion：低到中；
- meta：科学潜力高，可能平滑 density/hull tail；
- 工程风险：高于 PILS-L；
- 选择自由度：至少 kernel/lambda/boundary 三类；
- time-to-64：6–8 天；
- time-to-256：10–13 天，但没有 calibration/返工 buffer。

它不是单纯工程清理，而是潜在更强的方法；但在当前截止期内不保留，因为一个
失败的 bandwidth 不能区分“ordinal idea 错”还是“kernel 选错”。

## 5. fixed_slot_compressed scaffold 的可复用边界

已有：

- `crystal_dlm/fixed_slot_compressed.py`；
- `scripts/build_fixed_slot_compressed_sft_data.py`；
- `scripts/convert_fixed_slot_checkpoint_to_compressed.py`；
- `tests/test_fixed_slot_compressed.py`；
- legacy `run_mp20_stcompress_ablation.sh`。

可复用：

1. `share_lengths/share_angles/share_coordinates` config pattern；
2. token-source map；
3. shared row arithmetic-mean conversion；
4. vocab-shaped tensor remap；
5. axis-by-position parser；
6. full↔compressed round trip；
7. per-position allowed support tests。

不可直接复用：

1. scaffold 固定 107 semantic positions；
2. 使用 `<Sxx>/<EMPTY>/<*_PAD>`；
3. builder 输入是 fixed-slot `data/dlm_sft/mp_20`；
4. sampler 是旧 `sample_llada_crystals.py`；
5. loss weights 针对 fixed empty/nonempty slots；
6. launcher 使用旧 checkpoint/fallback；
7. 本地没有可接受的 exact-dynamic terminal；
8. 没有 H1 ordinal、Plan prefill、D1、model_494 或 S.U.N. 合同。

所以它只能证明“token family sharing 和 row remap 可实现”，不能证明 PILS-L
已经通过、也不能直接作为执行 source。

## 6. PILS-L 最小工程边界

为避免修改历史证据，建议新 workstream 新增：

1. `crystal_dlm/r5_dynamic_length_shared.py`
   - length-shared config；
   - exact `7+4N` encode/decode；
   - position-aware L parser；
   - shared-L legal support；
   - Plan match。
2. `scripts/build_r5_exact_length_shared_sft_data.py`
   - 直接从 frozen R5 exact records/arrays 构建；
   - 不经过 107-position fixed-slot；
   - train/val/test 45,229 rows全量 round trip。
3. `scripts/convert_r5_checkpoint_to_length_shared.py`
   - token-string row map；
   - only LA/LB/LC arithmetic mean；
   - other rows exact；
   - LoRA exact；
   - tokenizer/config/conversion manifest。
4. `scripts/sample_llada_h1a2_length_shared_body.py`
   - formal batch-1 ordinal sampler；
   - P0 ledger、D1、temperature 0.7、CFG 0；
   - count/element prefill；
   - duplicate/volume masks；
   - no retry/repair/filter/rerank。
5. tests
   - token counts/IDs；
   - exact round trip for N=1..20；
   - LA/LB/LC position semantics；
   - old axis token不可进入 L1 active support；
   - row-average exactness；
   - C0/L1 data pairing；
   - D1 schedule equality；
   - ordinal invariance；
   - parser/graph smoke；
   - all-attempt denominator。

本轮不得顺便：

- legal-support CE；
- support contraction；
- angle/coordinate sharing；
- neighbor smoothing；
- D1→safe-axis；
- RL；
- Planner chemistry；
- refiner/evaluator change。

## 7. 实验阶梯与停止条件

### 7.1 E0：Gate-1，截止 2026-08-06

PASS：

- full train/val/test 恢复并 hash；
- coverage ledger 完整；
- B0 legal-mass capture 完整；
- 第 4.1 节预注册条件通过；
- no endpoint-based support tuning。

STOP：

- train corpus 无法按时恢复；
- tokenizer/records 与 frozen B0 不一致；
- length 不再是主要 rare/unseen family；
- rare/unseen length legal mass 可忽略；
- 任何使用 final S.U.N. 反向选 token 范围的要求。

### 7.2 E1：data/conversion/unit gate，截止 2026-08-08

必须：

- 45,229 rows conversion 100%；
- `7+4N` 100%；
- arrays 在原量化精度下 round-trip；
- formula/N/elements/counts 100%；
- C0/L1 row IDs 与 source map 可验证；
- non-length rows exact copied；
- shared rows等于三轴 arithmetic mean；
- D1 groups、Plan prefill、constraints byte-equivalent in semantics；
- focused tests 100%。

STOP：

- 任一 row build failure；
- tokenizer additive boundary 失败；
- 非 length token 被重映射到错误语义；
- 需要修改 angle/coordinate/parser容错才能通过；
- fixed-slot padding 逻辑泄漏到 exact-dynamic。

### 7.3 E2：matched SFT，截止 2026-08-11

必须：

- C0/L1 各 exactly 1,696 updates；
- 无 NaN/OOM/divergence；
- same record order/corruption ledger；
- exact trainable-parameter boundary；
- final checkpoint 与 tokenizer manifests 完整；
- L1 shared-L validation NLL 相对 C0 的 axis-aggregated length NLL 更低；
- C0 相对 frozen historical B0 无明显 validation collapse。

STOP：

- 任一 arm 训练未完整；
- C0/L1 update、data、seed、optimizer budget不一致；
- L1 length NLL 不改善且 rare-bin calibration不改善；
- C0 continuation 本身明显崩坏；
- 需要 LR/checkpoint sweep 才能找结果。

### 7.4 E3：paired-32 body-only，截止 2026-08-12

同一 P0 Plans、ordinals、D1 macro schedule：

- C0 vs L1；
- no refinement/S.U.N.；
- candidate tokenizer/action support不同，因此 seed 只表示同一注册 ordinal，
  不声称 token-level common random number。

PASS：

- exact length/Plan match 100% among emitted bodies；
- no old LA/LB/LC token in L1 output；
- no illegal token；
- no new failure class；
- L1 completion `>= C0 - 1/32`；
- duplicate/volume failure不增加；
- lattice distribution不坍缩到单一/极少数 bins。

STOP：

- tokenizer/model system failure；
- `>=2/32` excess completion loss；
- new duplicate/volume/parser failure；
- extreme lattice tail明显增加。

32 不看 S.U.N. 选方案。

### 7.5 E4：paired-64 mechanism gate，截止 2026-08-14

进行 exact800、Direct 和冻结 S.U.N.，但只作为预注册早停，不调任何超参。

PASS：

- L1 completion `>= C0 - 1/64`；
- structure/joint raw loss不超过 2 pp；
- meta point delta `>=0`；
- strict point delta `>=-1/64`；
- novelty/uniqueness各不差超过 5 pp；
- length rare/unseen-bin emission 或校准误差相对 C0 改善；
- paired refiner displacement median不增加，p90不增加超过 5%；
- no new failure class。

STOP：

- meta 方向为负；
- completion excess loss `>=2/64`；
- strict、novelty 或 uniqueness出现明显 collapse；
- 机制指标与 endpoint 反向；
- 需要修改 LR、support、kernel 或 schedule 才能继续。

达到这里即得到“两周内机制信号”的最小证据；但不能写最终成功。

### 7.6 E5：fresh paired-256，硬截止 2026-08-18

Primary：L1 vs matched C0。

Secondary：两者与 frozen historical B0 的描述性对照，不用历史 B0 选择
checkpoint。

PASS/进入确认：

- strict point delta `>=0`；
- meta point delta `>0`；
- meta paired bootstrap 95% lower `>-2 pp`；
- completion/structure/joint 各不差超过 2 pp；
- novelty/uniqueness各不差超过 5 pp；
- 两个独立 128-ordinal block 的 meta delta 均非负；
- Plan composition完全守恒；
- raw comp_valid 变化可由 completion accounting解释；
- shared-L NLL/calibration、refiner displacement 与 meta 方向一致；
- no new failure class。

STOP：

- 到 2026-08-18 仍无完整 paired-256；
- meta point delta `<=0`；
- safe-axis 式 strict-positive/meta-negative 极化；
- completion/structure/joint 破非劣；
- 只在一个 128 block 为正；
- 需要 post-hoc support/seed/checkpoint 选择。

若停止，PILS-L 只能保留为 negative/diagnostic result，提交前不替换成第二个
token candidate。

### 7.7 E6：confirm，目标完成 2026-08-31

仅在 E5 PASS 后：

```text
4 independent scientific panels * 256 ordinals
C0 and L1 paired within panel
```

不是同一 256 ledger 的四个 CUDA repeats。

冻结：

- final C0/L1 checkpoints；
- P0；
- D1；
- model_494 exact800 batch1；
- evaluator/cache/hull snapshot；
- all-attempt denominator；
- no retries。

确认门：

- meta：至少 3/4 panels 正，mean `>0`，hierarchical paired 95% CI
  lower `>=0`；
- strict：至少 3/4 非负，mean `>=0`，one-sided lower `>-2 pp`；
- completion/structure/joint one-sided lower `>-2 pp`；
- novelty/uniqueness无预注册恶化；
- no new failure class；
- mechanism metrics 方向稳定；
- process repeats 只作为 model_494 variance，不冒充 scientific panels。

确认预算依据现有上界：

- 256 exact800 每 arm `<=6 A800 GPUh`；
- 4×256 paired C0/L1 `<=48 A800 GPUh`；
- queue-excluded 约 3–5 天，取决于并发。

## 8. 日历与最迟 cutline

| 日期 | 必须完成 | 失败处置 |
|---|---|---|
| 2026-08-06 | Gate-1 full corpus/legal mass | 停 token 路线 |
| 2026-08-08 | dynamic shared data/conversion/tests | 停 PILS-L |
| 2026-08-11 | C0/L1 matched SFT | 停 PILS-L |
| 2026-08-12 | paired-32 | 停或进 64 |
| 2026-08-14 | paired-64 mechanism signal | meta/机制不符即停 |
| 2026-08-18 | fresh paired-256 | **两周硬 cutline**；未完成不作为 ICLR 主方法 |
| 2026-08-21 | confirm source/manifest freeze | 不再改 method |
| 2026-08-31 | 4×256 confirmation | 未完成则降为 preliminary |
| 2026-09-01 | 方法、threshold、统计完全冻结 | 禁止新 token family |
| 2026-09-08 | 主表/图/机制 appendix 初稿 | 只补审计，不改实验 |
| 2026-09-12 | submission figures/results freeze | 留 6 天摘要缓冲 |
| 2026-09-18 AOE | ICLR abstract | 官方截止 |
| 2026-09-25 AOE | ICLR full paper | 官方截止 |

以上 time-to-result 是资源可用、queue-excluded 规划；排队延误不能用降低 gate、
更换 denominator 或跳过 64/256 来弥补。

## 9. 哪些变化最可能有效

| 方案 | completion 潜力 | meta 潜力 | comp_valid 潜力 | 性质 |
|---|---|---|---|---|
| Gate-1 audit | 无直接效果 | 无直接效果 | 无 | 测量/工程前置 |
| support contraction | 中，取决于 removed mass | 不确定 | 仅经 completion | safety/工程校准 |
| PILS-L | 中 | **当前最高且可在两周验证** | 仅经 completion | 最小方法候选 |
| 全轴 sharing | 低到中 | 不确定 | 仅经 completion | 主要是过度压缩/工程 |
| ordinal smoothing | 低到中 | 高潜力但高调参风险 | 仅经 completion | 后续方法候选 |
| 删除 dead compatibility token | 无 | 无 | 无 | 纯清理 |

PILS-L 若成功，合理的因果链应同时出现：

```text
axis pooling
  -> shared-L train count / NLL / calibration 改善
  -> rare/extreme lattice emission更受控
  -> body completion或 refiner displacement 改善
  -> meta S.U.N. 改善且 strict 非劣
```

若只有 tokenizer 更小而没有中间机制或 endpoint 改善，不能作为主贡献。

## 10. 最小 ICLR 方法版本

建议名称：

> **PILS-L: Position-Indexed Lattice-Length Sharing for Exact-Length
> Diffusion Crystal Generation**

最小方法只包含三点：

1. 利用 fixed exact-length schema 的 position semantics，把三套冗余
   axis-specific ordinal length vocabulary 因子化为一个 shared ordinal family；
2. 用三轴 B0 rows 的 arithmetic mean 做无额外语义先验的 warm-start；
3. 用 matched full-corpus continuation 让 shared rows聚合三个 axis 的正监督。

明确不是方法的一部分：

- support pruning；
- angle/coordinate sharing；
- neighbor smoothing；
- new schedule；
- RL；
- MLIP guidance；
- Planner chemistry；
- refiner retraining。

若 E5/E6 通过，论文贡献可表述为：

1. 发现并量化 exact-length crystal DLM 的 train/inference special-token
   support mismatch；
2. 提出 position-indexed ordinal token sharing，在不减少数值分辨率或改变
   denoising steps 的情况下将 active numeric token identity union 减少
   42.77%；
3. 用 matched H1-A2/P0、D1、exact800、all-attempt 实验证明监督共享如何改变
   length calibration、refiner burden 和 strict/meta stability。

若只过 E4 而没有 E5/E6，最多写成 preliminary/ablation，不应占主方法位置。

## 11. 一页式结论

### 现在做什么

1. 2026-08-06 前完成 full train/val/test + B0 fixed-256
   coverage/legal-mass Gate-1。
2. Gate-1 通过后，只实现 PILS-L：

   ```text
   LA/LB/LC -> shared L
   ```

3. 建 matched control C0 和 candidate L1，各做一轮完全同预算的
   1,696-update continuation。
4. 按 32 body-only → 64 full mechanism → 256 fresh endpoint 推进。
5. 2026-08-18 是 paired-256 硬 cutline。
6. 通过后才做 4×256 独立 panel confirmation，目标 2026-08-31 完成。

### 为什么选它

- held-out numeric unseen identity 的 94.08% 来自 length；
- length sharing 把晶体特殊 token 从 2,481 降到 1,479，把 numeric
  stochastic identity union 从 2,343 降到 1,341；
- 每个 position 的 0.1 Å 分辨率、500 个合法非零 length bins、exact
  `7+4N` 和 `6+3N` denoising actions 全部不变；
- 它直接针对 lattice/density/refiner basin，最可能在两周内给 meta 机制信号；
- 它比 all-axis 和 smoothing 少得多的自由度。

### 不做什么

- 不跑 support contraction、全轴 sharing 或 smoothing 作为并行候选；
- 不删 compatibility token；
- 不改 D1/safe-axis；
- 不改 Planner、refiner、evaluator；
- 不做 RL；
- 不根据 32/64 S.U.N. 反复改 support、LR 或 token family。

### 成功标准

必须同时满足：

- shared-L NLL/calibration 改善；
- completion/structure/joint 非劣；
- refiner displacement 不恶化；
- paired-256 meta point delta `>0`；
- strict 不出现明显退化；
- 无新 failure class；
- 最终 4×256 至少 3/4 panels meta 为正。

### 失败时

任一 gate 失败即停止 PILS-L，不在提交前自动换成第二个 token treatment。
token 路线失败不改变既有结论：B0/R5-C+D1 是成熟 Body 锚点，conditional
structure 已接近饱和，真正的 composition 主瓶颈仍是 Planner chemistry。
