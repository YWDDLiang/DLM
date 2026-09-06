# H-P33 数据与训练调度实现审查

2026-09-06。审查范围是 `mixed_geometry_training_data.py` 与 `train_mixed_geometry_dlm.py`，以及判断其语义所需的旧 T objective、连续风险和启动接口。只读审查实现；仅新增本报告与独立 CPU 证据。没有模型 forward、GPU、DDP 运行、SSH、训练、采样、MLIP 或 relaxation。

**结论：截至本报告绑定的代码快照，没有尚未关闭的数据或调度代码阻断项。** 每源一份 T construction 和一份 G、来源随机流、4/6 卡共享来源窗口、零权重 padding 和固定最终 checkpoint 规则均实现了注册定义。验证集噪声种子与归档目录中的 `git rev-parse` 两个问题已修复。可进入既定的真实多卡验收；本结论不代替该验收，也不证明 SUN 收益。

## 1. 审查对象与证据

| 对象 | SHA256 |
|---|---|
| [数据实现](../../src/crystal_dlm/mixed_geometry_training_data.py) | `f9379c74bfece42068da2e141f355e4f65074c0d4a333bc9cb94f6caa3db6de1` |
| [训练入口](../../src/scripts/train_mixed_geometry_dlm.py) | `1aaa8937aeee1bf00e62c625a0741039a4394411c78fb516715cb39cf5ef0473` |
| [连续风险与数值实现](../../src/crystal_dlm/mixed_geometry_diffusion.py) | `24ebd6d8c9def0205289b295691eb7a4074d23b58cbdfa4977f37996f0ac86dc` |
| [继承的 T objective](../../src/crystal_dlm/periodic_v2_objective.py) | `d12a28e0348ad9e27f4b5286efd9795a8f939aa73e27f27f69e9e3b58e2b9b04` |
| [注册定义](V3_H_P33_SPECIFICATION.md) | `ad61fd8a3e2d48ca1abdcf85d88f343054c89db383ad6002098514459be4b843` |

独立复核见 [CPU 程序](evidence_implementation_data_schedule/check_source_windows.py) 与 [结果](evidence_implementation_data_schedule/check_source_windows.json)。环境为 Torch `2.8.0+cpu`，12 个来源调度案例全部通过；额外实际调用数据构造器和 loss，对验证种子、随机流及整批 padding 做了检查。阅读了 [已有七项测试](../../tests/test_mixed_geometry_training_data.py)，没有将同一套测试再跑一遍冒充独立证据。

来源全量核验使用父任务提供的 [40009 原始结果](evidence/CONTINUOUS_SOURCE_IDENTITY_40009.json)，数值与 train normalizer 使用 [40013 结果](evidence/GEOMETRY_NUMERICS_40013.json)。本通道没有重新读取远程完整 JSONL，因此全量原始来源结论仍由这些具名凭据支持；合成 fixture 只检验读取它们的代码路径。

## 2. 原始连续标签、T/G 视图与来源绑定

`MixedGeometrySources` 按 `(source_split, source_row_idx)` 连接 prepared 行与 identity 行，要求两边无重复、key 集完全相同且只含目标 split。逐行验证 `identity_verified is True`、原始 `source_answer` 的字节 SHA，以及 `answer == source_answer`。随后旧的 `prepare_periodic_base_source` 检查 Plan 的 N/E 与原 token 答案、站点顺序相容。只匹配 formula、直接用行序号、缺行后继续或失败后退回 token 解码标签的路径均不存在。

G 的连续标签读取 `continuous_aligned.lattice_matrix_A` 与 `fractional`，检查站数、元素序列和有限形状，以 FP64 保存。F 只取周期余数；晶格经 train normalizer 编码。G 的 `input_body` 与 `old_body` 中六个晶格槽和所有 XYZ 槽均为 MASK，N/E 保留。其实际几何通过显式 `GeometryState` 传入；未把连续标签重新量化再送回 G。独立 fixture 的 F 值 `0.003137` 精确保留在 FP64 来源中，进入网络状态时按定义转换为 FP32。

T 直接调用原 `make_periodic_v2_training_example(..., view=0)`，保留 construction 的 0.75 prefix / 0.25 dense 抽样与原 support/objective。G 占新增的另一个 view，因此本轮不再训练旧的 view1 repair CE。每源每新增 epoch 各一份 T/G，与注册定义一致。

两种视图的来源 metadata 由旧白名单编译，未复制能量、SUN、旧 rollout 的 forced mask 等字段。本次 fixture 注入这些字段后确认其不出现在 G example 中。固定的 C/S/P 输入来自原 prepared 文件；完整 provenance 仍可通过来源 key 与启动时绑定的原文件追溯，不从新的生成结果构造标签。

40009 与 40013 提供的正式输入链如下；训练入口会对这四个文件及 normalizer 共五项计算完整文件 SHA，并与 `launch.input_sha256` 作整体相等检查。

| 正式输入 | 全量 SHA256 / 来源 |
|---|---|
| prepared train，27136 行 | `1a3fdea6c710d7b2dd397b6de332918ed239c6a0f46b6aab8b662bf4011e6b04` |
| prepared val，9047 行 | `7e598846dcfea5673dc16d67f887658a6bf34ad0df00d097d7c7053ff9b637d2` |
| train source identity | `5603f3bdb97b474bb33459ef6e55a6b943f9d83af444e64769067e6f3374e321` |
| val source identity | `30ededa3854f6fcbde209fed709df86c00a9a6ad6ab0994c9bc9121e8ee4adb3` |
| normalizer | 40013 在全部 27136 train 来源上拟合，ddof=0；正式文件字节 SHA 由启动清单固定 |

`normalizer.source_count == len(train)` 也在构造时检查；启动入口已有 `expected_train_sources` / `expected_val_sources` 参数。正式 wrapper 尚未在本次快照中出现，故“实际启动已绑定这些文件、27136/9047 计数与资源”仍是待集成凭据，不提前记为已完成。无需重新打开已经通过的 40009 原始 CIF 对齐问题。

来源白名单允许有限非负 `sample_weight`；实际公式会忠实保留该权重。正式等来源权重的语义由固定 prepared 文件支持，不能把接口允许非均匀权重本身称作本轮重加权。此处未从少量 fixture 推断全量权重分布。

## 3. 随机流与验证集种子

G 的种子为 SHA256 派生的 `(seed, split, source_row_idx, epoch, stream)`；训练使用 `geometry:training`，不同验证 band 使用独立 stream。训练 t 先按 LogUniform 抽样，再由同一局部 generator 产生该 view 的几何噪声；该操作不读全局 rank、微批位置或调用顺序。T 保留旧的独立 `periodic-v2` 来源/epoch/view/stream 种子。

数据初始化 seed 为 `202609062`，验证数据实例现已使用 `202609063`。先前验证 selection 使用后者、实例噪声误用前者的问题已关闭；独立 CPU 直接调用 `make_sources` 检查了两个实例的 seed。没有将这处修复称为旧 V2 科学结果的原因。

CPU 还确认：来源行和 identity 行同时换序、真实样本改为 padding 后，指定来源/epoch 的 t、z/F、v/u target 逐元素相同；改变 epoch 会改变 G 噪声。调用其他训练样本不会改变验证输入；相同验证 band 重复调用完全一致，换 band stream 改变噪声。

验证来源由 `validation_selection` 的来源 hash 事先选 100 个；每个来源固定一份 T 与六份 G。G 的 t 为 `0.004472135955, 0.022360679775, 0.1, 0.316227766017, 0.670820393250, 0.948683298051`。两轮使用相同 700 个监测状态，按 T/G 和 band 输出。

六个固定中心的等权平均是已注册诊断；它不是训练 LogUniform 风险的无偏积分估计。高 t 的单列结果应单独阅读。训练不根据这些监测值早停或选择 checkpoint，normalizer 也不从 val 拟合。

## 4. 4/6 卡共享窗口、覆盖与 padding

每个优化窗口先确定同一排列中的 12 个来源，再让每个来源各出现一次 T 和 G。每个微步全 rank 同模式，按 T/G 交替；4 卡分成三对微步，6 卡分成两对微步。改变 world size 只改变这 12 个来源如何分配给 rank，保持来源与模式的窗口组成相同。

| 正式调度量 | 4 卡 | 6 卡 |
|---|---:|---:|
| 每 rank microbatch | 1 | 1 |
| accumulation | 6 | 4 |
| 每窗口来源 / T+G 状态 | 12 / 24 | 12 / 24 |
| 每 epoch 更新 | 2262 | 2262 |
| 每 epoch 真实状态 | 54272 | 54272 |
| 每 epoch 零权重 padding 状态 | 16 | 16 |
| 两个新增 epoch 更新 / 真实状态 | 4524 / 108544 | 4524 / 108544 |

27136 个来源的最后窗口只有 4 个真实来源，即 4 T + 4 G；其余 8 个来源槽复制有效来源来构造正常输入，但两个视图的权重均为零。没有空晶体 padding，也没有 `drop_last`。每 epoch 各来源每种模式真实出现一次；训练结束逐源/逐模式 all-reduce coverage 必须均为 2。

独立 CPU 枚举 `N=1,11,12,13,27,27136` 与 world4/6，共 12 例；检查了所有 rank 同步模式、每个窗口 T/G 配对、真实覆盖、padding 数与两种 world size 的完整窗口序列一致。正式规模下两者窗口摘要均为 `0d5a37b2d6324a30a1477e8b547ed9ce6754289cc20f3ee3cc1e82e73e09165b`。这证明数据分配等价，不保证跨 world size 的随机网络计算、BF16 归约或最终权重逐位相同。

## 5. 固定分母的梯度系数

记来源数 N，窗口数 J=ceil(N/12)，补齐来源数 Np=12J，world size W，微批大小 m，累计步数 A=24/(Wm)。入口反传系数为 `(Np/N)/A`。T objective 已按完整微批取均值；G 明确使用 `(per_example * source_weights).mean()`，没有使用按有效权重重新归一化的 `risk.total` 作反传 loss。

若在固定参数值比较梯度，一窗口的 DDP 平均累计梯度为：

`g_window = (Np/N)/24 * sum_real_in_window [w_i * grad(loss_i)]`。

再对 J 个窗口取算术平均，系数正好化为 `1/(2N)`，得到每源 T/G 等权的注册总体风险。G 内部仍是 `0.5*mean6(v_error²) + 0.5*mean3N(u_error²)`；站数只影响坐标平均的分母，不把大晶体作为更多来源。

正式 `Np/N = 27144/27136 = 1.0002948113207548`。每个真实状态在其窗口中的系数同为 `1131/27136`，平均到整个 epoch 后同为 `1/54272`。尾批真实来源不因局部有效数少而被放大。本次有理数检查直接验证这一等式，而非用浮点近似接受小偏差。

该等式仅是损失权重与线性归约的说明；逐窗口 AdamW、梯度裁剪和参数变化不会因此等价于一次完整 epoch 的全批优化。源配对也会影响微批梯度协方差，这是已指定的调度。

全 padding 微批在正式最后窗口的部分 rank 上会发生。T 通过旧 logits 的连图零值，G 通过有效形状的零权重风险，均保留 backward 图。本通道分别对整批 T/G 做了张量反传：loss 严格为零、相关 prediction tensor 的梯度存在且严格为零。既有七项测试另覆盖了同一 G 批内真实与 padding 混合时的固定分母。

DDP 每微步同步，使用 `find_unused_parameters=True`；没有依赖未验证的 `no_sync` 交替模式策略。验证调用未包装 model，各 rank 处理来源切片，最终只归约可加指标，所以 100 个来源不能被 6 整除不会导致不同数量的 DDP forward collective。真实多卡、activation checkpointing 和交替未用参数的正确性仍由既定 preflight 的实际梯度对照确认。

## 6. 更新、最终 policy 与已关闭项

训练采用 fresh AdamW、注册两阶段 LR 与 clip_norm1，每个完整窗口恰好更新一次。每 epoch 保存状态；第一轮 checkpoint 的 `eligible_policy=False`，完成两轮的末次 checkpoint 才成为 `POLICY_PATH`。最终还核对总更新数、真实状态数与逐源 T/G coverage，成功标记在这些检查之后写出。

| 审查中发现的事项 | 当前裁决 |
|---|---|
| val 实例沿用 training data seed，未使用已指定 validation seed | 已修复；独立 `make_sources` 回归通过 |
| trainer 在无 `.git` 的归档代码中执行 `git rev-parse HEAD`，可能失败或读取外层 HEAD | 已修复；trainer 使用启动清单的代码对象 ID，启动方先验证完整对象 ID 再做不可变 `git archive`；不在归档内反查仓库 |
| 正式训练 wrapper 与完整 launch/entry/input 凭据 | 父任务正在集成，当前不冒称已运行；沿已存在的归档方式完成即可 |
| 真正 4/6 卡模型图、梯度归约、吞吐和显存 | 尚无本通道执行证据；交给已注册 preflight，不由 CPU 调度证明替代 |
| 正式训练效果、连续端点、SUN | 未测量；不从数据完整性、loss 下降或梯度存在推断 |

源身份与数值 gate 已由 40009/40013 关闭，数据调度代码已完成上述复核。后续只需完成既定启动集成与真实模型验收，再按冻结的两轮/末次 policy 规则推进；本审查没有提出新的损失、时间分布、GNN、teacher 或物理算子变更。
