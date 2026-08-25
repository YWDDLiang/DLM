# Candidate A counterfactual grounding：四重复终报

## 结论

> **Candidate A没有通过冻结的论文贡献判据，不能作为论文最后一个已成立的技术贡献。**

它通过了机制、Strict方向稳定性、body、Direct、novelty和composition六项检查，
但没有通过Meta S.U.N.非劣门：pooled hull-known Meta从`472/985 = 47.92%`
降到`460/988 = 46.56%`，差值`-1.36 pp`，低于预注册的`-1.0 pp`下限。
因此保留标准H1-A2为fallback；公开`105/1000 Strict、488/1000 Meta`不变。

这里Candidate恰好也得到`105`个Strict成功，但它是`105/1024 attempts`或
`105/988 hull-known`，绝不能与公开`105/1000`混为同一结果。

## 冻结实验合同

- 同一冻结256-Plan cohort、同一control/candidate checkpoint、同一model494 refiner；
- 标准H1-A2 D1 `lattice → X → Y → Z`，temperature `0.7`，refine `800`步；
- 不使用safe-axis，不retry、replacement、filter或rerank；
- DLM seeds：`17117, 17217, 17317, 17417`；
- refiner seeds：`27117, 27217, 27317, 27417`；
- 每个repeat内按`base_seed + sample_idx`逐ordinal严格配对；
- official MP：fresh empty cache，`compatible_only=True`，`GGA_GGA+U`；
- hull unknown排除hull-known分母，绝不映射成unstable。

## 执行谱系

| Job | 状态 | 作用 |
|---|---|---|
| `34700` | FAILED after training | control/candidate训练均完成；后续缺少`scripts`包入口，checkpoint冻结复用 |
| `34710` | FAILED pre-science | 未传conda环境名，2秒退出 |
| `34711` | COMPLETED | fixed-256 screen；两臂均`256 parsed / 253 body / 253 refined` |
| `34714` | COMPLETED | 8×A800四重复body/refine |
| `34719` | FAILED pre-eval | V3 protocol仅接受1000/1200分母；8份generation已组装，无评价科学结果 |
| `34721` | COMPLETED | 8-cell Direct、N/U、CHGNet及official-input收集 |
| login query | COMPLETED | 251 chemsys fresh official MP查询与8-cell finalization |

## 机制结果

| Validation step | Control factual CE | Candidate factual CE | Candidate − Control |
|---:|---:|---:|---:|
| 500 | 1.915292 | 1.623997 | -0.291295 |
| 1000 | 1.316164 | 1.367862 | +0.051698 |
| 1500 | 1.289004 | 1.288558 | -0.000446 |

训练日志中true-vs-counterfactual margin的sample-weighted mean为`+0.7592`，
`79.29%`的记录事件为正。形式机制门通过，但最终factual CE只改善`0.000446`，
应诚实解释为“最终CE基本打平”，不能继续使用step500的较大差距代表最终模型。

## 四重复完整结果

`Strict K`和`Meta K`使用hull-known reconstructed分母；最后两列使用全部256请求分母。

| R | Arm | Req | Parsed | Body | Refined | Recon | Direct C/S/J | N/U/N∩U | Hull K/U | Strict K | Meta K | Strict/256 | Meta/256 |
|---:|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|
| 1 | Control | 256 | 256 | 255 | 255 | 255 | 231/254/230 | 231/254/230 | 247/8 | 25/247 = 10.12% | 125/247 = 50.61% | 9.77% | 48.83% |
| 1 | Candidate | 256 | 256 | 255 | 255 | 255 | 231/255/231 | 224/254/223 | 247/8 | 25/247 = 10.12% | 112/247 = 45.34% | 9.77% | 43.75% |
| 2 | Control | 256 | 256 | 254 | 254 | 254 | 230/254/230 | 223/253/222 | 246/8 | 25/246 = 10.16% | 115/246 = 46.75% | 9.77% | 44.92% |
| 2 | Candidate | 256 | 256 | 255 | 255 | 255 | 231/255/231 | 228/254/227 | 247/8 | 27/247 = 10.93% | 117/247 = 47.37% | 10.55% | 45.70% |
| 3 | Control | 256 | 256 | 255 | 255 | 255 | 231/255/231 | 228/254/227 | 247/8 | 27/247 = 10.93% | 123/247 = 49.80% | 10.55% | 48.05% |
| 3 | Candidate | 256 | 256 | 255 | 255 | 255 | 231/254/230 | 228/254/227 | 247/8 | 25/247 = 10.12% | 118/247 = 47.77% | 9.77% | 46.09% |
| 4 | Control | 256 | 256 | 253 | 253 | 253 | 229/253/229 | 223/252/222 | 245/8 | 26/245 = 10.61% | 109/245 = 44.49% | 10.16% | 42.58% |
| 4 | Candidate | 256 | 256 | 255 | 255 | 255 | 231/255/231 | 230/254/229 | 247/8 | 28/247 = 11.34% | 113/247 = 45.75% | 10.94% | 44.14% |

## Pooled结果

| Metric | Control | Candidate | Candidate − Control |
|---|---:|---:|---:|
| Requested / parsed | 1024 / 1024 | 1024 / 1024 | 0 |
| Body / refined / reconstructed | 1017 / 1017 / 1017 | 1020 / 1020 / 1020 | +3 |
| Direct composition / structure / joint | 921 / 1016 / 920 | 924 / 1019 / 923 | +3 / +3 / +3 |
| Novel / unique / N∩U | 905 / 1013 / 901 | 910 / 1016 / 906 | +5 / +3 / +5 |
| Hull known / unknown | 985 / 32 | 988 / 32 | +3 / 0 |
| Strict S.U.N., hull-known | 103/985 = 10.46% | 105/988 = 10.63% | **+0.171 pp** |
| Meta S.U.N., hull-known | 472/985 = 47.92% | 460/988 = 46.56% | **-1.360 pp** |
| Strict S.U.N., all attempts | 103/1024 = 10.06% | 105/1024 = 10.25% | +0.195 pp |
| Meta S.U.N., all attempts | 472/1024 = 46.09% | 460/1024 = 44.92% | -1.172 pp |
| Body rate | 99.32% | 99.61% | +0.293 pp |
| Direct joint rate | 89.84% | 90.14% | +0.293 pp |

## Repeat稳定性与配对检验

| Repeat | Δ Strict known | Δ Meta known | Strict discordance C-only/Cand-only | Meta discordance C-only/Cand-only |
|---:|---:|---:|---:|---:|
| 1 | +0.000 pp | -5.263 pp | 2 / 2 | 28 / 15 |
| 2 | +0.769 pp | +0.620 pp | 4 / 6 | 17 / 19 |
| 3 | -0.810 pp | -2.024 pp | 5 / 3 | 27 / 21 |
| 4 | +0.724 pp | +1.259 pp | 2 / 4 | 20 / 21 |

- Strict repeat差：mean `+0.171 pp`，SD `0.742 pp`，95% t-CI
  `[-1.011, +1.352] pp`；
- Meta repeat差：mean `-1.352 pp`，SD `2.970 pp`，95% t-CI
  `[-6.077, +3.373] pp`；
- pooled known-both：`981`对；
- Strict exact McNemar：control-only `13`，candidate-only `15`，`p=0.8506`；
- Meta exact McNemar：control-only `92`，candidate-only `76`，`p=0.2471`。

Strict方向满足“3/4非负且pooled为正”，但效应很小且不显著。Meta在2/4 repeats
为负，最主要的问题是repeat 1的`-5.26 pp`，最终使pooled差低于非劣下限。

## Novelty与化学分布

- 完整Plan cohort严格相同，因此input-stage family、arity、N-bin和all-metal TVD均为0；
- pooled novel rate：Control `88.99%`，Candidate `89.22%`，差`+0.228 pp`；
- pooled unique rate：Control `99.607%`，Candidate `99.608%`，基本相同；
- reconstructed-stage最大family TVD为`0.0093`，最大all-metal rate绝对差为
  `0.192 pp`；
- N∩U-stage最大family TVD为`0.0228`，最大N-bin TVD为`0.0248`。

因此没有novelty、uniqueness或easy-chemistry collapse证据；性能差异来自同一Plan
cohort上的realization随机结果，而不是改变proposal mix。

## Official hull覆盖

- requested chemsys：`251`；
- resolved：`243`；unresolved：`8`；
- 8个unresolved全部因缺少Yb unary reference，保留为hull unknown；
- 每个cell均有8个reconstructed hull unknown；未把它们计为unstable。

## 冻结贡献门

| Gate | 结果 | 判定 |
|---|---|---|
| factual CE更优且true-vs-CF margin为正 | final CE `-0.000446`；margin `+0.7592` | PASS |
| ≥3/4 Strict差非负且pooled为正 | 3/4；`+0.171 pp` | PASS |
| Meta pooled ≥ -1.0 pp且不能3/4为负 | `-1.360 pp`；2/4为负 | **FAIL** |
| Body pooled差 ≥ -1.0 pp | `+0.293 pp` | PASS |
| Direct joint pooled差 ≥ -1.0 pp | `+0.293 pp` | PASS |
| 无novelty collapse | Novel `+0.228 pp`，Unique持平 | PASS |
| 无composition collapse | input完全配对，最大recon family TVD `0.0093` | PASS |
| Overall | 一项关键门失败 | **FAIL** |

## Reviewer式判断

优点：

- fixed Plans和逐ordinal随机流把proposal变化排除在外；
- body、Direct joint、novelty和Strict方向一致地没有退化；
- 结果不依赖safe-axis，也没有挑最好repeat；
- fresh official hull和unknown政策完整保留。

不足：

- 目标训练没有带来可靠的Meta改进，反而跨重复pooled下降；
- Strict只有`+2/1024`，McNemar完全不显著；
- 训练早期CE优势在最终checkpoint几乎消失；
- 当前证据只能说明grounding objective改变了训练机制，不能说明它形成了更好的
  最终structural-realization方法。

最终决定：Candidate A作为内部负/混合结果归档，不进入public方法贡献；Candidate B
此前也已判负。论文继续使用冻结H1-A2，最后一个新增技术贡献要求**尚未解决**。

机器可读证据：

- `GROUNDING_FINAL_REPEAT4.json`
- `GROUNDING_FINAL_REPEAT4.csv`
- `GROUNDING_OFFICIAL_INPUT_MANIFEST.json`
- `GROUNDING_OFFICIAL_COMPLETION_MANIFEST.json`
- `GROUNDING_UNRESOLVED_CHEMSYS.jsonl`
