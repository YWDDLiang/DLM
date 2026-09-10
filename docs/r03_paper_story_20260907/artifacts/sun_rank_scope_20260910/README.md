# SUN 排序器权重

本目录包含本阶段真实训练并经完整 E3 重载核验的两个 300 参数判别组件。它们需要模型定义中的冻结 E3、同一 tokenizer、当前连续结构与 token 视图，不能单独替代完整晶体编辑模型。

| 文件 | 输出 | 固定策略 | SHA256 |
|---|---|---|---|
| `delta_sun_ranker.pt` | 新颖 Stable / 新颖 MS 的预测增量 | SUN 增量≥0.05，MS 增量≥0 | `522a935509ac9cad844b8c72e9711440f21c592a049a945c9581669588c993b5` |
| `nested_sun_ranker.pt` | 由两个 logits 构成的 P(NS)、P(NMS) | P(NS)≥0.2，候选内 SUN 优先 | `0f8c430c70edb39404862c638fcc0d9767dc7803493d8ab0e3b1acdc8505295b` |

输入为冻结 E3 quality 隐藏层的 128 维表示及 21 个可观测特征，顺序见 [MODEL_BUNDLES.json](MODEL_BUNDLES.json)。训练归一化已折叠进 `weight` 和 `bias`，推理直接计算 `x @ weight.T + bias`，不要再次标准化。差值模型直接使用两个输出；终态模型使用 P(NS)=sigmoid(z0)×sigmoid(z1)、P(NMS)=sigmoid(z1)。概率分数没有单独做校准验证。

取特征、推理及核验实现分别见仓库 `src/scripts/train_sun_ranker.py`、`src/crystal_dlm/sun_ranker.py`、`src/scripts/audit_sun_ranker.py`。推理读取已经评价的当前状态，不读取待选候选的物理评价结果。候选先构造一次连续结构，决策后取其绑定的原记录；KEEP 返回原连续结构。

终态模型是新终测前按 DEV 选定的主候选。完整采用门槛未通过，不能把对照在新终测上的较好均衡结果解释为已重新选择默认模型。完整计数、损失和科学含义见[结果报告](../../SUN_RANK_SCOPE_RESULTS_20260910.md)。
