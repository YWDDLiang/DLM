# V3 最终权重的 T construction 部署可行性

2026-09-06。只读接口映射；**未修改采样代码、未新增采样注册、未提交 GPU，也没有 T 生成结果。** 本文不改变已经确定的 H-G／Q 评测。

**结论：可行，现有数值模型和 `ProgrammedPathSampler` 内核无需新增生成算法。** 完成的 mixed checkpoint 中保留了经过 T+G 共同训练的 `token_model`；取出它，显式关闭所有 construction 之后的修订，即可保留原“条件／程序→离散 DLM 执行”的部署路径。需要小范围脚本接入，不能直接给旧命令换 checkpoint 路径。

## 1. 已有代码提供了什么

- [MixedGeometryDLM.forward](../../src/crystal_dlm/mixed_geometry_model.py) 的 `mode="token"` 直接调用同一个 `token_model`，不调用 G time/head、连续先验、ODE 或最终 Q；[保存／加载](../../src/crystal_dlm/mixed_geometry_model.py) 同时保存并核验 token 组件、G head 和 normalizer。T 并非只留下一个未训练的兼容入口：[数据实现](../../src/crystal_dlm/mixed_geometry_training_data.py) 的 T 每源每 epoch 使用原 V2 `view=0` construction 样本。
- [ProgrammedPathSampler.run](../../src/crystal_dlm/programmed_path_runtime.py) 已接受四个独立布尔参数。仅 construction 应显式使用 `construct=True, cooperative=False, closure=False, full_cell_repair=False`；默认的后三项不是这个端点的完整合同。
- 构造先固定 N/E，再依 [spad_predictor_position_groups](../../src/crystal_dlm/spad_program.py) 依次生成六个晶格标量、按 P 排列的各 species anchor XYZ、再按 P 排列的剩余站点 XYZ。健康请求有 `6+3N` 次标量决策；N=1/20 分别为 9/66 次，不沿用 G 的 33 NFE。
- `token_model` 是原 `PeriodicV2DLM` 子类，保留 `raw_initialization` 属性。因此 construction 的 `active_context` 仍为所有尚未揭示的数值槽，而非只标当前标量。每步 `old=x.clone()`，`task_id=construct`，`numeric_noise_level=-1`，全部走原整数状态、conditioner、V2 attention bias 和词表头。
- [真实 40045 预检](evidence/MIXED_PREFLIGHT_40045.json) 四 rank 的 `token_step_zero.max_abs` 均为 0，支持 mixed T 路由与其内置 token model 相同；它不证明未来最终权重的 construction SUN 或完整轨迹已经通过。

## 2. 最小接口接入

1. **先核验 mixed 最终策略，再提取 T。** 复用 [validate_final_policy](../../src/crystal_dlm/mixed_geometry_sampling.py)：核验 mixed method、checkpoint completed/expected epochs、TRAIN_FINAL updates/epochs/policy_path 和训练 `_SUCCESS`。随后 `load_mixed_geometry_model(..., trainable=False)` 完成组件 hash、原始 LLaDA／词表／LoRA、V2 父代与新增组件核验，最后取 `mixed.token_model.requires_grad_(False).eval()`。不要把 mixed 目录交给 `load_periodic_v2_model`、改写其 marker，或绕过 mixed 检查直接调用无策略选择的 architecture loader。
2. **增加一个显式 construction 入口。** [旧脚本](../../src/scripts/sample_state_programmed_paths.py) 第 170 行见 `periodic_repair_config.json` 就要求 `--full-cell-repair`；[load_path_model](../../src/crystal_dlm/programmed_path_data.py) 又优先识别 `periodic_v2_config.json`，最终会被 [V2 method/4524 专用门](../../src/crystal_dlm/periodic_v2_initialization.py) 拒绝 mixed。最小做法是在旧脚本增加明确的 mixed-token-construction 分支，先识别 mixed marker、采用上面的 loader，并显式传入四个运行参数；只对这一分支替换旧 full-cell 要求，保留 V2／reference 入口的原约束。复用原 `conditions_for_run`、`compile_condition`、`make_sampler` 和 `sampling_batches`，不用再写采样器。
3. **端点记录及 replay 与 construction 相符。** 记录 `method=mixed_geometry_h_p33`、实际最终 checkpoint、`deployment_mode=token`、`path_mode=construction_only` 和全部 phase/seed/batch 元数据。旧 `replay()` 末尾要求在 cooperative phase 测得 `state_effect>0`，仅 construction 必然不满足；可以复用 `replay_scalar_states`／`processed_logits` 检查原条件、每次 draw 的 log probability 和末态重建，去掉这一不适用的 cooperative 断言，另核验所有 draw phase 都是 construct。无需修改概率变换。

优先传 `.token_model`，而非把外层 mixed 对象作为所有旧接口的替身：旧 replay 调用 `model.geometry_inputs(context)`，外层同名函数是 G 的多参数接口。原 token model 的接口匹配现有调用。

## 3. 必须原样复用的比较口径

| 项目 | T construction 的固定复用内容 |
|---|---|
| 条件与程序 | 同一固定开发 256 条；conditions SHA256 `f1bbc29891bd3e42bcc452ff3ea1585f038ef83f0c7ba9327239192d67a65f68`。保留 sample_idx/group_id、C/S/Plan、species_program/source 与显式 fallback；prompt 仍由原编译器 `rstrip()+"\n"` 编码，不重问 LLM／pointer。 |
| 词表与槽位 | 同一 tokenizer／原子身份／`7+4N` 布局；N/E 由 Plan 和 canonical compiler 固定，六晶格与 XYZ 从 MASK 开始，不借用 G 生成结果作初态。 |
| 支持与 alias | 原 `exact_dynamic_schema_constraints` 和 `process_path_logits`；duplicate-coordinate、lattice-volume、`min_lattice_rad=1e-4`、周期 minimum-distance 0.5 Å、image radius 2；坐标 100→0 的 logaddexp 概率合并及 alias 屏蔽。末态仍过 `complete_geometry_supported`。 |
| 随机性 | 温度 0.7；base seed `20260905`、round 0、occurrence 0；原 `path_seed` 的 63-bit 整数完整存储，construct salt `100000000+10007*step`；原逐行 GPU FP64 Gumbel 变换不变。T 不使用 G 的 CPU FP64 连续先验；共用 seed 名称不代表 T/G 是同一随机 draw。 |
| 批次与数值 | 原 logical batch cap 4、layout world size 2，先按 `ordinal%2,N,prompt_length` 分桶，再把完整 logical batch 分派到实际 4/6 worker；不按新 worker 数重分组。保持原 CUDA BF16 autocast，另外记录实际神经调用和失败后的批内开销。 |
| 导出与分母 | T 使用 `--native-source body`，通过原严格 parser／graph 路径；全部 256 请求留在账本。新 T 分支明确把失败标成 `parseable=False`，无 active body/structure，尝试状态留在 diagnostic/trace；避免旧 body 导出器为兼容历史数据而解析一个 `success=False` 的完整预览并重新接纳。 |

[实际 V2 种子与条件证据](evidence/V2_ACTUAL_EVALUATION_SEEDS.json) 中 eval:0 的 seed 为 `8732798746829386280`，不能经过 JavaScript Number 再写回。支持规则不改；其物理局限不能因换成 T 端点而被隐藏。

若后续接 τ800，复用同一冻结 refiner checkpoint、800 steps、`num_evals=1`、batch size 1 和 graph `refiner_seed=(20260905+原sample_idx)%2**32`。跳过无 graph 的请求也保留原索引和总分母；不根据 raw/τ800 的能量或 SUN 逐样本选端点。这里仅说明现有接口可复用，没有注册该实验。

## 4. 合理对照和可支持的主张

| 比较 | 当前证据状态 | 可以回答的问题 |
|---|---|---|
| mixed 最终 T construction raw 对 V2 construction raw | 前者未采样；后者固定 256 上 Strict/Meta SUN **12/44**（4.6875%/17.1875%），verified 为 4/21。 | T+G 继续训练后的离散程序执行端点，是否优于现有离散 construction。不能单独归因于 G：新增两个 epoch、T 更新及共享 G 更新一起改变了权重。 |
| mixed T construction→τ800 对 V2 construction→τ800 | **两者均未测。** | 在相同离散 proposal→同一外部晶体扩散接口下比较最终系统；必须保留两者的 raw 结果。 |
| mixed T 与同一最终权重的 H-G／同样本 Q | T 未测；G/Q 按原方案准备评测。 | 同一训练产物不同部署机制的表现和成本；T/G 的状态、更新、输出及中间支持不同，不能把差别仅归于离散/连续表示，也不是 P 的净收益消融。 |

[12/44 来自完整 construction 前缀评测](evidence/V2_CONSTRUCTION_AND_REPAIR_39998.json)，[提取脚本](../../scripts/extract_periodic_construction_endpoint.py) 从已记录的 V2 construction→repair 轨迹截到 repair 前，没有新采样或按结果选择。已有 V2 **8/50** 是 repair 后 native；已有 **16/113** τ800 是 **repair→τ800**，不能移作 construction→τ800，[两端点证据](evidence/V2_EVALUATION_SUMMARIES_39998.json)。所有后续 raw/τ 比较仍使用同一 N/U 源、缓存、共同 CHGNet 终态评估与 headline/verified 双报告，“raw”不表示免做共同稳定性评估。

若 T 最终有效，可诚实表述为“连续周期几何去噪参与共享主干训练，部署时由语言程序驱动离散 DLM 生成，再按固定协议精修”；仍要披露原 CIF 连续监督及新增曝光。T 的存在只能证明执行接口保留；T 提升也不能单独证明 LLM 预训练、soft 条件或 learned P 的价值。H-G 提升尤其不能代替这几个结论。

## 5. 接入前仅需的有界验收

未来若决定实施以上入口，核验最终 gate/组件 hash、T wrapper与 `.token_model` 同输入输出相等、无 G forward、全 trace 仅 construct、原 P 顺序与支持/alias、条件/seed/logical batch 身份、失败账本及 fresh scalar replay 即可。健康源的决策数应为 `6+3N`，不要求误报为 33。不要用工程 probe 选择 checkpoint，也不要把本映射或训练中 T loss 当作新增端点已经验证。
