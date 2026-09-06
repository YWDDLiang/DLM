# 旧 LLM→DLM 顺序控制与当前 raw runtime 的只读代码追踪

本报告仅基于本地代码读取、文件 hash 比较；没有执行模型、SSH、GPU 检查、作业/自动化或实现修改。旧工作树读取时 HEAD 为 `cc0face868d5566d607ee71bfddd4d670ac9fb35`，新工作树为 `87978683121751a91b1cab1e2943f63d2ea81fd3`。以下记录的是当时工作文件调用链，不替代远端已运行快照的确认。

**结论：旧版的 Llama 顺序控制是真实调用链，不是只存在于文档；当前 raw 分支仍保留该控制。** Llama 决定 unique-species permutation；compiler 让它改变 anchor、site 和部分修订的访问顺序。它不决定六晶格/XYZ 的内部轴顺序，也不在每个 DLM token 前重新运行。新 raw 的 construction 与 full-cell repair 都消费同一 program；但它明确关闭了旧 cooperative 与 reverse-species closure，且 MP20 训练 program 主要来自 contact-tree teacher 而不是部署的 Llama 预测，因此不能宣称完整保留了旧方法或已经完成训练/部署顺序分布对齐。

## 1. 端到端字段与函数链

```text
C³FD 当前化学状态与可达动作支持
  → sample_single_trajectory：typed Llama 与 C³FD 在同一支持上融合概率
  → 已确定组成 + 预测 soft Plan + terminal typed sequence
  → ProductionRuntime.species_program
  → PlanConditionedSpeciesPointer.decode
  → species_program / species_program_indices / species_program_source
  → freeze_requested / validate_program → frozen plans_for_dlm.jsonl
  → compile_condition → program_from_element_order
  → SpeciesProgram(entries, canonical slot_indices, anchor_slot)
  → ProgrammedPathSampler.run / spad_predictor_position_groups
  → construction → [旧 cooperative → reverse-species closure]
                 → [当前 raw full-cell repair]
```

| 环节 | 实际函数/字段/调用点 | 精确行为 |
|---|---|---|
| 生成 Plan | 旧 [sample_c3fd_llama_typed_planner.py:381](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/scripts/sample_c3fd_llama_typed_planner.py:381) `sample_single_trajectory`；[520](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/scripts/sample_c3fd_llama_typed_planner.py:520) terminal sequence/soft fields | 化学动作生成完成后构造 `terminal_sequence`；C³FD 与 Llama 的 soft-field logits 用 `unit_weight_poe_log_probs` 融合，采样 lattice/SG/VPA。 |
| 提取 Llama 状态 | 同文件 [802](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/scripts/sample_c3fd_llama_typed_planner.py:802) `_sequence_hidden`；[826](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/scripts/sample_c3fd_llama_typed_planner.py:826) `ProductionRuntime.species_program` | typed embeddings 经同一个 Planner Llama，取 terminal hidden。pointer 的其他输入为 canonical 元素原子序数、counts、valid mask，以及此次实际选中的 soft-field IDs。 |
| 预测物种排列 | 旧 [species_program_pointer.py:176](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/species_program_pointer.py:176) `permutation_logits`；[258](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/species_program_pointer.py:258) `decode` | context 由 terminal projection 与三个 soft embeddings 组成；逐步将已选 species mask 掉，query 还读已选 candidate 的聚合与 step embedding。部署使用 argmax，产生全部 unique elements 的确定性排列；不是对所有原子位置独立排序。 |
| 持久化 | 旧 [sample_c3fd_llama_typed_planner.py:561](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/scripts/sample_c3fd_llama_typed_planner.py:561)、[593](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/scripts/sample_c3fd_llama_typed_planner.py:593) | 输出 `plan_state`、`prompt`、`species_program`、`species_program_indices`、`species_program_source`。learned 路径来源是 `planner_llama_pointer`。不加载 pointer 时有 `canonical_control`/`canonical_compatibility`，不能把这些叫 learned program。 |
| 真实 prospective 入口 | 旧 [172 wrapper:50](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/slurm/172_freeze_spad_prospective_seed23.sbatch:50)；[freeze_spad_prospective_plans.py:25](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/scripts/freeze_spad_prospective_plans.py:25) | wrapper 显式给 `--species-pointer-state`。冻结器校验 permutation、indices 一致并要求 `species_program_source == planner_llama_pointer`；因此登记的 prospective 路径会拒绝 canonical fallback。 |
| 编译条件 | 旧 [programmed_path_data.py:25](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/programmed_path_data.py:25) `compile_condition` | 从记录的 `plan_state/species_program/source` 编译；创建 exact `7+4N` MASK canvas，预填 N 和每个 canonical slot 的元素；prompt 由 DLM 自身 tokenizer 编码。 |
| 编译 program | 旧 [spad_program.py:84](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/spad_program.py:84) `_expanded_plan_slots`；[104](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/spad_program.py:104) `program_from_element_order` | slots 由 canonical composition 决定，元素排列只改变 `entries` 次序；每个 entry 的第一个 canonical slot 是 anchor，其余是 remaining slots。严格拒绝重复/缺失元素和计数不一致。 |
| 传入 sampler | 旧 [sample_state_programmed_paths.py:101](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/scripts/sample_state_programmed_paths.py:101) `make_sampler`；新对应 [103](D:/codex_work/ai4s/DLM_periodic_self_repair/src/scripts/sample_state_programmed_paths.py:103) | `programs=[c['program'] for c in compiled]` 直接传入 `ProgrammedPathSampler`，不是写进日志后遗忘。 |

`semantic_trace` 本身不能充当 learned program。旧 [spad_program.py:132](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/spad_program.py:132) `program_from_planner_trace` 明确把 trace first-occurrence 次序记为 `typed_planner_trace`，并不声称这种 chemical species 次序被自由学习。已实现的 Llama 顺序通道是额外的 learned pointer。

## 2. 旧主线各阶段究竟怎样受 program 控制

| 阶段 | 函数与位置 | program 决定的部分 | 固定/其他决定的部分 |
|---|---|---|---|
| Construction | 旧 [spad_program.py:196](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/spad_program.py:196) `spad_predictor_position_groups`；旧 [programmed_path_runtime.py:232](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/programmed_path_runtime.py:232) | 先按 entries 顺序访问各物种 anchor，再按 entries 顺序访问各物种 remaining slots。native 存储顺序与生成访问顺序可以不同，已生成的较后 native anchor 可成为较早 native slot 的上下文。 | N/元素预填；六晶格位置固定 `1..6`；每个 site 固定 `X→Y→Z`；数值 token 来自 DLM，不来自 Llama pointer。每步只采一个 scalar。 |
| Cooperative region | 旧 [programmed_path_runtime.py:82](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/programmed_path_runtime.py:82) `cooperative_slots` | 以 program 第一个 species 的 anchor 起始；优先加入尚未出现、program rank 最早的 species 的 site；同等条件使用 program rank 做 tie-break。 | region 大小由固定公式决定；优先 species 内选与已选集合周期距离最近的 site。它是 program-rooted 几何规则，非 Llama 在线阅读结构后再挑 region。 |
| Cooperative transaction | 旧 [programmed_path_runtime.py:253](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/programmed_path_runtime.py:253) | program 通过已选 region 与 site 次序影响坐标访问。 | 整体 MASK 六晶格+region XYZ，先 lattice 再按 region site 的 XYZ scalar 顺序；old snapshot 固定；空支持/末态无效时整项回滚。 |
| Species closure | 旧 [spad_program.py:236](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/spad_program.py:236) `reverse_species_block_revision_slots`；旧 [programmed_path_runtime.py:270](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/programmed_path_runtime.py:270) | 物种按 `reversed(program.entries)`；物种内部先 reversed remaining slots，再 anchor。原先最早 anchor 得到更多未来可见上下文。 | 每个 block 整体 MASK；逐 site、逐 XYZ 条件采样。单 site 空支持先恢复该 site，末态验证失败恢复整个 species block。 |
| 每一步模型条件 | 旧 [programmed_path_runtime.py:159](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/programmed_path_runtime.py:159) `processed_logits`；新 [state_conditioned_model.py:231](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/state_conditioned_model.py:231) `context_from_programs` | entries 还映射成每个 canonical site 的 `program_rank`，进入 conditioner。program 不仅影响调度，也影响数值状态条件。 | `x` 为当前 token canvas；`old` 是构造当前前缀或修订起点 snapshot。合法支持/温度/alias 由 runtime 定义，所有概率来自 DLM 的逐次 forward。 |

旧 K4/K8 的正常入口是 [sample_state_programmed_paths.py:228](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/scripts/sample_state_programmed_paths.py:228)：非 reference 时 `construct/cooperative/closure` 都开启。不要混同历史参考 `close_reference`：它在 predictor 后调用单独 `revise_spad_cell`，然后 `revise_spad_species_blocks(reverse_species_block_revision_slots(...))`，其 stage ledger 是 `predictor→cell→reverse_species_blocks`。[programmed_path_reference.py:29](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/programmed_path_reference.py:29)

更早的 SPAD/BS 路径也有真实顺序执行：旧 [sample_llada_r5_exact_length.py:940](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/scripts/sample_llada_r5_exact_length.py:940) 从 `prompt_record.species_program` 编译 `row_schedules`，随后在 [980](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/scripts/sample_llada_r5_exact_length.py:980) 将它传给 `generate(... generation_position_groups_by_batch=row_schedules)`；[llada_generation.py:739](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/crystal_dlm/llada_generation.py:739) 确实逐 batch-row/position-group 执行。可选 backfill/closure 又调用实际 revision 函数。这不是仅为论文新增的叙述。

## 3. 当前 raw 分支保留与替换的部分

本次逐文件 SHA256 比较中，以下旧/新文件完全相同：`spad_program.py`、`spad_generation.py`、`species_program_pointer.py`、`sample_c3fd_llama_typed_planner.py`。compiler 的共同 SHA256 是 `AE160F19F9E9E02EACBCF71427EB4BA639A7C53E0E74D353721C6F1B05573F3C`。差异集中在 `programmed_path_runtime.py` 与 `sample_state_programmed_paths.py` 等新接口。

| 接口 | 当前 raw 真实行为 | 对论文表述的影响 |
|---|---|---|
| 冻结 Plan/program | 新 [241 wrapper:52](D:/codex_work/ai4s/DLM_periodic_self_repair/slurm/241_evaluate_periodic_dlm_native.sbatch:52) 读取 `spad_prospective_seed23_256_v1_20260903/plans_for_dlm.jsonl`，传 `--full-cell-repair`；`conditions_for_run` 只补 group/split 字段，不重写 program。 | 原来通过 learned-pointer gate 冻结的队列仍可使用；该评估不会重新调用 Llama 改顺序。具体已部署 artifact 的 hash 未在本次只读本地审查中复核。 |
| raw 模型分支识别 | 新 [programmed_path_data.py:134](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/programmed_path_data.py:134) 看到 `raw_periodic_initialization.json` 则加载 `FreshPeriodicRepairDLM`，其 [初始化类:76](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_repair_initialization.py:76) 设置 `raw_initialization`。 | phase 选择确实受实际模型类型影响。不能只检查 CLI 默认值。 |
| Construction | 新 [programmed_path_runtime.py:255](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/programmed_path_runtime.py:255) 仍由 `spad_predictor_position_groups(p)` 构造 scalar 访问顺序。raw 的 active-context scope 改为全部尚 MASK 的 numeric positions，保存在 trace。 | Llama program 仍控制 construction 顺序；active scope 改变不等于取消 program。 |
| Cooperative/closure | 新 [sample_state_programmed_paths.py:235](D:/codex_work/ai4s/DLM_periodic_self_repair/src/scripts/sample_state_programmed_paths.py:235) 对 `hasattr(model,'raw_initialization')` 显式令二者 false。 | raw 不能沿用「同时执行旧协同半胞+反向物种闭合」的描述。旧函数仍在代码库不代表实际路径执行它们。 |
| Full-cell repair | 新 [programmed_path_runtime.py:128](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/programmed_path_runtime.py:128) `full_cell_transaction_positions(program)` 直接返回相同 predictor schedule；[321](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/programmed_path_runtime.py:321) 整体 MASK 全 lattice/XYZ，再按该列表采样。 | repair 的 lattice-first、species-anchor-first 顺序仍由 Llama program 部分控制；不是 reverse-species closure，也不是所有坐标边缘概率同时独立采样。 |
| Full-cell 可见上下文 | 同一 `_begin` 保留完整 `old`，当前 numeric canvas 全 MASK；逐次 DLM forward 接收当前 `x` 与固定旧 snapshot。 | 旧完整晶体通过 conditioner/geometry bias 进入；不是保持旧未来 numeric tokens 可见的单 anchor backfill。construction/repair 后半段仍可能看见本次新生成的 native-suffix anchors，两种「未来信息」须区分。 |
| program rank | 新 [state_training.py:43](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/state_training.py:43) 与推理 `context_from_programs` 都把 program entries 编译为 rank；[periodic_state_conditioning.py:212](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_state_conditioning.py:212) 将 rank/relative rank 用于 site/pair conditioner。 | 新版不是只保留 `species_program` JSON；序列调度与模型条件都有实连接。但 learned rank 有效不等于能量收益已证实。 |
| 回放 | 新 [programmed_path_runtime.py:397](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/programmed_path_runtime.py:397) `replay_scalar_states` 根据事件实际 position、phase、snapshot 和 `construct_active_scope` 重建状态；[sample CLI:249](D:/codex_work/ai4s/DLM_periodic_self_repair/src/scripts/sample_state_programmed_paths.py:249) 验证 trace 终态等于输出。 | 新版 phase/order/active scope 应以 event trace 记录为准，不能只凭文档或函数存在性。 |

一个仅由 compiler 规则推得的例子（非模型采样结果）：canonical slots 为 `O0,O1,Na2`，program 为 `Na→O` 时，construction 与 raw full-cell repair 都访问 `lattice→Na2 XYZ→O0 XYZ→O1 XYZ`；旧 reverse-species closure 则访问 `O1 XYZ→O0 XYZ→Na2 XYZ`。改 program 而不改 canonical canvas，会改变访问顺序与 conditioner rank，故顺序控制不是名义上的字段。

## 4. 真正存在的训练/部署分布差异

### 4.1 旧能量路径 program 与当前 raw MP20 program 来源不同

旧 K4/K8 条件准备入口 [224:21](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/slurm/224_state_train_conditions.sbatch:21) 调 `prepare_state_train_conditions.py`。该脚本 [107–141](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/src/scripts/prepare_state_train_conditions.py:107) 实际运行 frozen Llama，先对 soft fields 取 C³FD+Llama PoE 的 MAP，再把 predicted soft IDs 输入 pointer。输出来源是 `frozen_llama_pointer_predicted_soft`；组成来自固定训练条件，未重采。该分支确实已经让旧能量路径使用预测 program，不是 teacher contact order。

当前 raw trainer 则不同：[240:28](D:/codex_work/ai4s/DLM_periodic_self_repair/slurm/240_train_periodic_dlm_from_base.sbatch:28) 读取既有 `spad_basin_closure_sft_v1_20260904` 的 MP20 行。其登记 builder [195:16](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/slurm/195_build_spad_basin_closure_sft_data.sbatch:16) 读取 canonical teacher 数据与原始 `spad_species_pointer_v1_20260903` 数据；后者 [build_spad_species_pointer_data.py:165](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/scripts/build_spad_species_pointer_data.py:165) 写的是 `contact_tree_order_symbols/indices`。closure builder 的 [_program_from_row:98](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/scripts/build_spad_basin_closure_sft_data.py:98) 将这些识别为 `contact_tree_teacher`，缺语义行通过 [_resolve_program:125](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/scripts/build_spad_basin_closure_sft_data.py:125) 显式 canonical fallback。

新 [prepare_periodic_base_source:133](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_base_training_data.py:133) 只继承来源/program，并据此计算 full-cell transaction，不重新调用 frozen Llama/pointer。所以「训练有 species program」正确，「当前 raw 全部训练都用部署 Llama 预测顺序」不正确。上述判断追溯的是注册 builder 与调用参数，实际每个 source 的来源计数仍应从原数据 manifest/逐行记录确认，不能凭代码推定未读取的数据绝无人工变更。

此外，旧训练条件用 predicted **MAP** soft fields，prospective Planner 使用实际 sampled soft fields；两者都不是 teacher 真值，但也不是完全相同的输入分布。下一版若强调训练/部署一致，应注明这一差别。

### 4.2 当前 raw 只匹配编译顺序，没有匹配全部条件状态分布

新 [make_periodic_base_training_example:254](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_base_training_data.py:254) 的 construction 是 clean MP20 上独立随机 numeric MASK；部署 construction 是当前模型顺序采样形成的前缀。两边都标记尚 MASK 的 numeric positions 为 active，解决了接口 scope 的一类错配，但输入错误分布仍不同。

repair 在 [264–275](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_base_training_data.py:264) 从 full-cell schedule 中选一个监督 scalar，旧状态来自 structured corruption，前缀全部填 clean target；部署前缀由当前模型生成，可能有错误或触发回滚。由于 train/inference 调同一 `full_cell_transaction_positions`，没有发现顺序函数本身不一致；问题是 teacher order/teacher prefix/synthetic old-state 对预测 order/self prefix/own old-state 的迁移。

## 5. 对新版与论文的直接约束

- 可以保留并准确表述：**冻结 Llama 根据化学状态与软计划预测物种级程序，compiler 将该程序转成 DLM 的 native-slot 条件访问顺序；DLM 以周期几何交互和合法支持完成构造/修订。** 它是先生成一次、后确定性编译的 program，不是 Llama 每一步在线指挥 DLM 的 agent loop。
- 必须区分 Llama 决定的 species 次序、compiler 固定的 anchor/remaining/XYZ 模板，以及 geometry-aware cooperative 选择。不能把整个 phase 顺序、晶格顺序或 XYZ 轴顺序都归给 Llama 学习。
- 如果新版继续只用原始 MP20、排除新 K4/K8，可在该范围内准备 frozen Planner/pointer 预测的软条件与程序覆盖；缺 typed metadata 的 source 必须保持可审计 fallback，不伪造 Llama transcript。具体是否重建数据由主任务决定；本报告未实施。
- 数据和 runtime 应共用一个明确的 program→positions compiler，同时记录 `program_source`、permutation、phase、active scope 与实际 draw position。下一版若改 full-cell 次序或采用 GEM-style noisy state，要同步训练 mask/teacher prefix/replay，不能只把旧 JSON 搬过去。
- 不能把当前 raw 写成完整继承旧 suffix-visible reverse closure。若最终新版确实保留该机制，应看到它进入实际执行入口并有相应训练状态；若采用 whole-cell revision，应按真实已实现的状态信息来源表述。
- 下一版代码冻结前，需要使用固定合法小例子验证「同一 composition/prompt、不同 program，compiled 访问顺序与 rank 会变化」以及「同 program 的 train/runtime positions 一致」。模型机制证据还需要实际 trace 与 matched program 对照；仅本次静态追踪不证明 Llama 顺序有 SUN 优势。历史效果边界见同目录 `EVIDENCE_REVIEW_AGENT.md`。

当前 raw wrapper 中虽然仍有后训练 stage/旧收集入口的兼容代码，本报告未将这些入口视为新一轮方案，也没有执行它们。后续新模型训练、推理与资源释放后的自动接续由主任务按用户最新要求单独安排；本报告仅交付上述本地调用链与适配约束。
