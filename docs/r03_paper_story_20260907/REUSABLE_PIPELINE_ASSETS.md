# R03 链路可复用资产与参数化整理清单

更新日期：2026-09-08。范围：P0 → C3FD/程序头 → B0 构造/修订 → model494 精修 → 物理标注 → 官方参考缓存 → SUN，以及显式训练反馈链路。

v0.5新增可复用边界：`sun_feedback_contract.py`核验训练来源与独立MAIN的组成隔离；原label和SUN CLI通过`training_feedback`用途接入，保留原物理和N/U算法。`generate_sun_feedback_candidates.py`负责有限教师候选、独立图构造及精修导出身份校验，原`expert_composition_probe.py`增加训练反馈模式。`analyze_sun_headroom.py`只汇总已完成候选及评分回执，检查跨噪声支持和未知标签，不执行物理筛选或部署推理。现阶段状态与清单见[教师候选空间执行记录](execution/expert_self_edit/sun_v05/EXECUTION_STATUS.json)；这些入口不表示SUN策略训练已经实现。

用户已要求的收尾使用统一[归档与明确权重清理工具](../../operations/archive_experiment.py)：数据逐文件校验并压缩到实验树外，核验归档每个成员后才允许清理清单内的失败编辑权重；显式保留的参考checkpoint、生成数据、轨迹、标签和模型元数据保持可用。`archive`与`prune`为分开的动作，活跃/等待作业、归档变化或未归档数据都会阻止清理。

**当前状态：2026-09-08用户已授权专家编辑实验。旧R03路线保持停止；新路线复用本表入口并通过manifest绑定独立源码和运行目录。2048条新请求构造已完成，短训准备中；正式1200请求主面板尚未运行编辑器。**

目标是继续使用现有核心实现，把实验差异放进参数和配置。新任务可以产生新的配置、运行目录、结果和回执；不应因此产生一套新的采样、精修、评分、部署代码。

## 正式决定：评测脚本作为长期复用入口

2026-09-07 按用户要求，将现有评测脚本正式定为后续实验默认复用的工程资产。本决定确立入口及维护规则；完整编排的通用参数化仍按第 9 节逐项完成，不将尚未实跑的路径标记为已验证。

| 职责 | 正式复用入口 | 使用约定 |
|---|---|---|
| 组成和结构有效率 | [run_direct_validity_fast.py](../../scripts/run_direct_validity_fast.py) | 使用现有 `comp_struct` 模式 |
| 物理标注 | [label_programmed_paths.py](../../scripts/label_programmed_paths.py) | 结构及协议身份匹配时复用已有标签；需要新标注时使用此入口 |
| N/U、Strict/Meta SUN 评分 | [evaluate_programmed_paths.py](../../scripts/evaluate_programmed_paths.py) | 统一核心评分 CLI；输入结构、标签、固定协议、官方缓存及全请求数均显式传入 |
| 输入导出与多组件合并 | [export_r03_evaluation_inputs.py](../../src/scripts/export_r03_evaluation_inputs.py) | 按现有组件或 manifest 接口接入新方法输出 |
| 完整性检查和评分编排 | [evaluate_trial.py](../../operations/r03_c3fd_main_20260907/evaluate_trial.py)、[run_evaluation_stage.py](../../operations/r03_c3fd_main_20260907/run_evaluation_stage.py) | 复用现有实现；当前仍有本轮特例，后续在原入口扩参数，不复制任务专属版本 |
| 官方参考缓存准备与查询 | [prepare_hull_union.py](../../operations/r03_c3fd_main_20260907/prepare_hull_union.py)、[run_hull_query.py](../../operations/r03_c3fd_main_20260907/run_hull_query.py) | 保留来源、完成状态及实际端点覆盖验证 |

后续执行规则：

- 新方法、新 seed、新规模优先只增加输入 manifest、配置和结果目录；不另写一套 SUN 或有效率评测器。
- 参数不足时扩展上述原入口；接入问题在输入适配或编排层修复，核心科学口径保持明确的版本身份。
- 每次评测记录源码身份、输入与标签身份、参考库和固定协议、端点、全请求分母。协议发生变化时明确登记，不能将两种定义混成同口径结果。
- 保留无效、未完成、未知及工程错误；不得只对成功样本计算请求级产率。多 seed 的唯一性在合并 cohort 上重算。
- 已有结果按第 7 节的依赖和身份规则复用；现存输出保留，新执行写入新目录。
- 当前端到端回归参照为 R/I tau800 各 256 条的已完成结果。原生 N/U 阻塞、任意规模编排及正式双 seed 的验证缺口继续保留，不能因“正式复用”而视为已解决。

本次落实的是复用决定和维护约定，没有改动评分算法、运行评测或完成第 9 节的通用化改造。

## 1. 先看结论：哪些直接复用，哪些先整理

| 层次 | 现有资产 | 后续处理 |
|---|---|---|
| 模型与训练产物 | P0、B0、tokenizer、pointer、修订专用物理适配、model494 | 按路径和哈希引用；不为新任务复制或重训相同资产 |
| 核心算法 | formula 支持、周期几何桥接、程序执行、masked 修订、物理标签绑定 | 保留现有模块；算法变化才新增模式或实现 |
| 阶段 CLI | Planner、程序导出、body、refinement、输入合并、label、SUN | 大部分已有参数；优先直接调用 |
| 实验编排 | `run_component.py`、`run_trial.py`、`run_evaluation_stage.py`、`submit_stage.py` | 继续在原入口参数化，移出日期、作业号、角色组合和模型路径硬编码 |
| 缓存与结果复用 | 官方 hull 缓存、特征缓存、已有结构及对应物理标签、已完成 SUN 单元 | 按输入和协议身份复用；只重算受修改影响的下游阶段 |
| 运维 | tmux relay、持久连接、Git bundle、分块传输与哈希验证 | 归并现有实现；停止新增 `deploy_<commit>`、`probe_<issue>`、`status_<task>` 文件 |

**不要把“有代码”“有参数”“完成真实验证”混为一谈。** 正式双 seed 编排及合并入口已有实现和局部检查，但本轮没有实际完成正式运行。

## 2. 位置与版本：从哪里找回这条链路

本地仓库：`D:/codex_work/ai4s/DLM_periodic_self_repair`。分支：`codex/r03-c3fd-rich-main`。

以下别名仅用于本文缩短远端路径，不是代码已经支持的配置变量：

```text
PROJECT = /public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
GROUNDING = PROJECT/workstreams/proposal_realization_candidates_20260826/grounding
RUN = GROUNDING/experiments/r03_c3fd_rich_20260907/run_20260907T033526Z
OPS = operations/r03_c3fd_main_20260907
```

| 身份 | 本轮值 | 用途 |
|---|---|---|
| 256 生产源码 | `8167ae54f545936ac93005d336358d61b6e85d8d` | `RUN/code/<commit>`；R/I/G/P 实际生成使用的不可变目录 |
| 几何 canary 生产源码 | `82050dd8a3f19297d4ae293399b02035c131933a` | 16 条几何复测及匹配 I 对照 |
| 最新操作/评价源码 | `269d7dadc7c62443d89a2ceb0c15bd23fb163bc4` | 修复时间格式与 hull 阶段标识，支持精修端点预览；不是另一次生产模型训练 |
| 官方评价配置 | `PROJECT/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/CONFIG.json` | N/U、参考训练集、物理协议和代码身份 |
| 停止记录 | `RUN/USER_STOPPED_REMAINING.json` | 停止检查时本任务没有剩余运行/排队作业；不能据此推断每个结果阶段均成功 |

本地入口与证据：

- [基础资产身份与完整 SHA256](execution/assets/ASSET_IDENTITIES.json)
- [执行过程记录](execution/CURRENT_EXECUTION.md)（包含较早的阶段状态，停止状态以本文件及停止回执为准）
- [几何 canary 原始资产](execution/assets/canary_geometry_40457/SOURCE_RECEIPTS.json)
- [R/I tau800 SUN 原始报告](execution/assets/ri_tau800_256/RI_tau800_SUN.json)、[可读表格](execution/assets/ri_tau800_256/RI_tau800_SUN.md)、[源文件哈希](execution/assets/ri_tau800_256/SOURCE_RECEIPTS.json)
- [停止回执](execution/remote_receipts/persistent_1788781439428_77ba410305cb.json)

截至整理时，最新 R/I 结果、部分运维辅助文件和后期回执仍是本地未跟踪文件。**其他 checkout 或远端 Git 分支不一定包含它们。** 后续发布可复用包时应逐项纳入必要文件，不能把整个回执目录直接加入版本库；其中也有历史探测输出和临时 bundle。

## 3. 模型、特征与离线训练数据

| 资产 | 路径 | 可复用内容与边界 |
|---|---|---|
| P0 基础模型 | `/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B` | 原 H1A2 Planner 的基础模型 |
| P0 adapter | `PROJECT/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final` | 本轮没有更新 P0 的组成/rich LM 权重 |
| B0 基础模型 | `/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct` | 原构造与修订执行器的基础模型 |
| B0 adapter/tokenizer | `PROJECT/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final` | 原构造权重与 token ABI；不能随意换 tokenizer 或拼接其他 LoRA |
| 原 R03 runtime | `PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1` | 原 rich prompt、schema、safe-axis 与 paired-noise 实现 |
| 原始 Planner prompt | `RUN/pointer_40395/assets/P0_NATIVE_PROMPT.txt` | 保持字节、tokenizer 和 prompt-style 身份，不擅自裁剪换行 |
| C3FD 实际域 | `RUN/pointer_40395/assets/C3FD_DOMAIN.json` | 88 元素、393 strata、N≤20；当前为单价态见证，不是混合价实现 |
| C3FD 源资产 | 基础资产身份 JSON 中的 C3FD checkpoint 与 vocabulary | 本轮复用支持机制和语义资产；不把该 checkpoint 当作替代 P0 的独立 Composer |
| 程序控制头 | `RUN/pointer_40395/train/r03_control_pointer.pt` | 已完成一次训练；绑定 P0 特征、物种定义、prompt 与监督格式 |
| P0 特征缓存 | `RUN/pointer_40395/train/{train,val}_formula_features.pt` | 训练器已有 `--feature-cache-dir`；必须校验元数据后复用 |
| 物理训练输入清单 | `RUN/pointer_40395/assets/PHYSICS_SOURCES.json` | 通过来源清单定位 K4 consistent v2 / K8 原 teacher、路径和标签，不另造路径表 |
| 物理训练准备产物 | `RUN/physics_40399/data/` | 已完成兼容视图、权重及来源处理；查看 `PREPARATION_FINAL.json` 和训练清单 |
| 修订专用物理适配 | `RUN/physics_40399/train/checkpoints/step-128` | 完成固定 128 更新；以 `train/POLICY_PATH`、`TRAIN_FINAL.json`、checkpoint 内 `R03_REPAIR_TRANSFER.json` 为身份依据；不能用它替换原构造 B0 |
| model494 | `/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt` | 已实跑固定 tau800；它是独立连续精修模型，不是 DLM 自身修复能力 |

主要固定身份：

```text
P0 adapter: 65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a
B0 adapter: 5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d
B0 tokenizer.json: 3a21588abca8e56155cc7b6cabb81df51992ccd2e89704aec770912f24e75509
model494: 573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e
实际 C3FD_DOMAIN: a2b4a49eff72790cabaa75977414dba9eba45812ad02944d46e023a9b1d804a4
pointer: bc62f0606c21dcd74911358ba451b464016f2f97363d5a3f7e0be0eceb945266
```

物理 adapter 的哈希应读取其 `TRAIN_FINAL.json`，本文不猜测或补造。K4/K8 的条件重叠、标签兼容边界和已有来源索引见 [K4/K8 复用说明](K4_K8_REUSE_PLAN.md)。旧训练数据的能量可以复用到同一物理结构，不能修改坐标后沿用旧标签，也不能称为当前 R03 策略新采集的 on-policy 数据。

## 4. 可复用核心代码与当前参数

下表是现有 API；不是拟议的新命令。具体默认值以所选源码版本为准。

| 功能 | 现有入口 | 关键参数 / 模块 |
|---|---|---|
| 原 P0 生成，C3FD 可选 | [sample_r03_h1a2_plans.py](../../src/scripts/sample_r03_h1a2_plans.py) | `--model-path --checkpoint-path --native-prompt-file --num-samples --seed --sample-index-offset --batch-size`；传 `--c3fd-domain` 才启用约束 |
| 程序头训练/特征复用 | [train_r03_control_pointer.py](../../src/scripts/train_r03_control_pointer.py) | `--feature-cache-dir --teacher-data-dir --pointer-data-dir --mp20-dir --prompt-style --prompt-text-file` |
| 从实际 Plans 导出程序 | [export_r03_control_programs.py](../../src/scripts/export_r03_control_programs.py) | `--plans-jsonl --pointer-checkpoint --output-jsonl --batch-size`，以及原 P0/prompt 身份参数 |
| 原 B0 构造和可选修订 | [run_r03_integrated_body.py](../../src/scripts/run_r03_integrated_body.py) | `--plans-jsonl --b0-checkpoint --frozen-runtime-root --seed --expected-requests --batch-size`；构造约束 `--construction-geometry`；修订 `--repair --geometry-support`；P 模式另传 `--repair-checkpoint` |
| 公式、几何、程序模块 | [r03_formula_bridge.py](../../src/crystal_dlm/r03_formula_bridge.py)、[r03_geometry_bridge.py](../../src/crystal_dlm/r03_geometry_bridge.py)、[r03_control_pointer.py](../../src/crystal_dlm/r03_control_pointer.py) | 直接复用类和函数；不要为 R/I/G/P 分别复制算法 |
| 物理数据适配/训练 | [train_r03_physics_transfer.py](../../src/scripts/train_r03_physics_transfer.py)、[r03_physics_transfer.py](../../src/crystal_dlm/r03_physics_transfer.py) | `--sources-manifest --prepare-only` 或 `--prepared-dir`；`--updates --learning-rate --effective-batch --seed`。已有离线质量加权 CE，不是仅有 reward 接口 |
| 连续 diffusion refinement | [refine_dlm_with_crysllmgen.py](../../src/scripts/refine_dlm_with_crysllmgen.py) | `--proposal-graphs --checkpoint --crysllmgen-dir --diff-steps --num-evals --batch-size --seed --seed-by-sample-index --max-proposals` |
| 单组件及同臂多 seed 端点导出 | [export_r03_evaluation_inputs.py](../../src/scripts/export_r03_evaluation_inputs.py) | 单组件 `--body-dir`；多组件 `--input-manifest`；`--endpoint native\|tau800 --refined-pt --expected-requests --method-id`。多组件各自声明 refined tensor，不能再传一个公共 `--refined-pt` |
| 基础组成/结构有效率 | [export_direct_view.py](../../operations/r03_c3fd_main_20260907/export_direct_view.py)、[run_direct_validity_fast.py](../../scripts/run_direct_validity_fast.py) | 前者转成旧评价器视图；后者用 `--metrics comp_struct`，不需要执行全套 Direct 分布/指纹指标 |
| CHGNet 物理标注 | [label_programmed_paths.py](../../scripts/label_programmed_paths.py)、[terminal_energy_consistency.py](../../src/crystal_dlm/terminal_energy_consistency.py) | `--input-jsonl --output-dir --purpose --gpu-count --workers-per-gpu`；保留原始/终态能量、结构、收敛与核验状态 |
| 共同官方参考 | [prepare_hull_union.py](../../operations/r03_c3fd_main_20260907/prepare_hull_union.py) | `prepare / finalize / verify-endpoints`；`--inputs-manifest`；可重复 `--known-cache`；实际端点必须验证覆盖 |
| 缺失参考查询 | [run_hull_query.py](../../operations/r03_c3fd_main_20260907/run_hull_query.py) | `--hull-root`；读取已登记查询命令和现有私有凭据 provider，不在清单或日志中保存密钥 |
| N/U 与 SUN | [evaluate_programmed_paths.py](../../scripts/evaluate_programmed_paths.py) | `--paths-jsonl --labels-jsonl ... --frozen-config --official-cache --expected-requests --endpoint --cohort-role --policy-stage`；支持多个原始 label JSONL，合并 seed 时无需伪造新标签 |
| 预检、结果校验及复用 | [evaluate_trial.py](../../operations/r03_c3fd_main_20260907/evaluate_trial.py) | `--manifest --output-dir`；校验完成状态、样本身份、标签绑定、实际 hull 覆盖与已有评分来源 |

本轮 R/I/G/P 的差异本来就应是配置差异：

| 角色 | Formula C3FD | 构造几何支持 | 程序修订 | 修订权重 |
|---|---|---|---|---|
| R | 关 | 关 | 关 | 无 |
| I | 开 | 关 | 关 | 无 |
| G | 开 | 开 | 开 | 原 B0 |
| P | 开 | 开 | 开 | step-128 物理适配 |

256 试跑的 I/G/P 复用同一份实际 I Plans，G/P 复用同一份导出程序。R/I 的相同全局 seed 不保证组成逐条相同：原 Planner 是进程开始时播种一次，随后 RNG 随生成推进。

## 5. 已经得到的数据与可直接接续的中间产物

```text
RUN/
  pointer_40395/{assets,train}/
  physics_40399/{data,train}/
  canary_geometry_40457/{G,P,I_batch1_reference}/
  pilot_256/
    shared_I/planner/plans_for_dlm.jsonl
    shared_I/plans_with_programs.jsonl
    {R,I,G,P}/
      planner/                         # 自行生成 Planner 的组件才有
      body/
        run_config.json
        body_tokenizer_identity.json
        batch_partition.json
        raw_generations.jsonl
        proposal_graphs.pt
        construction_raw_generations.jsonl  # 修订臂保留的原始构造
        construction_sample_metrics.json
        repair_progress.jsonl
        sample_metrics.json
      native/{paths.jsonl,EVALUATION_INPUTS_FINAL.json}
      refine/{run_config.json,refinement_metrics.json,...}
      tau800/{paths.jsonl,EVALUATION_INPUTS_FINAL.json}
      {native,tau800}_validity/{report.json,attempt_metrics.jsonl}
      {native,tau800}_labels/{labels.jsonl,LABEL_FINAL.json,trajectories/}
      COMPONENT_CONFIG.json
      COMPONENT_FINAL.json
      *.command.json / *.stage.json / *.out / *.err
  hull_canary/
  hull_pilot256/
  evaluation_pilot256_preview_R_I_tau800/
```

这是阶段产物约定，不表示所有目录中每个文件都已存在。尤其 refined tensor 应读取 `refinement_metrics.json.output_file`，不要根据名字猜路径。

| 数据 | 已确认范围 | 后续复用方式 |
|---|---|---|
| 几何 canary 40457 | G/P 各16及匹配 I 的构造、修订、精修和标签；关键原始 JSON 已传回并校验哈希 | 小规模接口/配对回归样本；不是正式主结果 |
| R/I 256 | 两端生成和物理标签完成；tau800 SUN 预览完成 | 直接重用实际 Plans、body、refined tensor、标签及对应评分单元 |
| G/P 256 | 已核实构造/修订、native 指标和 native 标签；最后留存进度回执中 refinement 仍在运行 | 停止后未重新清点最终端点；接续前读取各组件完成标记和最终清单，不假定全链完成 |
| G/P 构造失败 | 两臂均38条 `construction_geometry_no_legal_support`，218条完成 | 几何恢复的工程诊断样本；必须同时保留成功请求和全256分母 |
| R/I 原生 N/U 异常 | 首个坏原生结构进入旧 matcher 后长时间未推进，未获得完整原生 SUN 表 | 保留异常样本和日志；不能把“有代码”“有预览目录”写成已完成原生评测 |
| 正式双 seed 路径 | 参数入口与合并代码存在，未实跑 | 后续工程验证对象；本轮无正式数据可复用 |

R/I tau800 已验证结果可作为端到端回归参照：R 的 Strict/Meta SUN 为 **22/117**，I 为 **17/113**，分母均256；comp/struct 分别为 R **222/249**、I **252/252**。这不是新方法收益保证。通过终点核验的 SUN 子集另计，不能替换主口径或当成 DFT 验证。

**训练用途边界：** 本轮 `pilot_256` 及 canary 是工程/评价资产，不自动变为新训练集。未来若明确将其用于开发或训练，需要重新划分未见评估集。已有轨迹不自动等于对齐完毕的 `旧状态 → 局部专家修正` 监督；原子对应、周期表示、量化后的实际编辑效果仍需另行验证。

## 6. 缓存和环境：最容易节省重复成本的部分

| 项目 | 已有位置/能力 | 复用要求 |
|---|---|---|
| 基础官方 MP 缓存 | `GROUNDING/runs/spad_basin_closure_official_20260904_v1/official_mp_cache` | 原协议与来源身份匹配 |
| 本轮增量参考 | `RUN/hull_canary/missing_query/official_mp_cache`、`RUN/hull_pilot256/missing_query/official_mp_cache` | 后续 prepare 作为 `--known-cache` 输入；不要重新查询已解析系统 |
| 本轮合并参考 | `RUN/hull_{canary,pilot256}/official_mp_cache` | 查询/合并完成不等于所有系统均有已知凸包；保留 official unresolved |
| 参考库核验产物 | `completion_manifest.json`、`completion_SUCCESS`、`cache_provenance.jsonl`、端点覆盖报告 | 准备时用 Plans 确定上界；评分前验证实际 native/refined 组成是其子集 |
| 训练集 N/U 索引 | 由固定 `CONFIG.json` 的 `training_index_cache` / `train_csv` 定位 | 训练集身份变化时重建；不另复制一份来源不明的索引 |
| 普通生产/标注 Python | `/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python` | 本轮实跑环境为 Python 3.10；可复用环境，不重装整套依赖 |
| 官方查询 Python | `/public/home/jiaosz/ywliang/ai4s/.venvs/mp_api_0_45_13_emmet0_85_1_py310_v4_system/bin/python` | 使用查询清单已登记的解释器和 MP 协议 |
| 本地轻量工具 Python | `C:/Users/admin/miniconda3/envs/exp1/python.exe` | 文档、传输和回执处理 |
| 本地结构分析 Python | `D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/.venv/Scripts/python.exe` | 已用于结构/接口检查；不因此授权本地重新跑模型或标注 |

连接路径为：本地 → `starteam5090` → 已有 tmux `ssha800:1.0` → 已有 A800 SSH。计算节点无网，官方 API 查询在登录节点运行；采样、MLIP 标注、N/U 评分使用相应计算资源。凭据只通过私有 provider 在内存中读取，配置与输出只保存 provider 引用，不保存密钥。

已有运维实现：[tmux_relay.py](../../operations/r03_c3fd_main_20260907/tmux_relay.py)、[prepare_persistent_relay.py](../../operations/r03_c3fd_main_20260907/prepare_persistent_relay.py)、[accept_persistent_relay.py](../../operations/r03_c3fd_main_20260907/accept_persistent_relay.py)、[receive_geometry_transfer.py](../../operations/r03_c3fd_main_20260907/receive_geometry_transfer.py)。持久会话的临时进程号不是资产，不应写进下一次任务配置。

本轮确认可在已有 Slurm allocation 内用 `srun --jobid … --overlap --exact --cpus-per-task=1 --gres=gpu:0` 运行 CPU 评分步骤。它仍共享原作业的 CPU、内存和寿命：父作业结束会影响步骤，不能把它当成独立长期作业，也不能据此超出总资源上限。

## 7. 什么改变了，才需要重算

| 修改 | 必须重做 | 可以保留 |
|---|---|---|
| 仅表格、图、展示顺序 | 展示产物 | Plans、结构、标签、已完成且身份匹配的评分 |
| 时间格式或阶段名接入错误 | 修复接入并重新核对受影响阶段；原失败回执保留 | 未改变的物理输入和标签 |
| 官方 hull 缓存更新 | hull 能量差与 SUN；核对来源 | 生成、refinement、原物理标签；训练参考未变时 N/U 可以复用 |
| N/U 训练参考或匹配定义更新 | N/U 与 SUN | 结构及物理标签 |
| CHGNet/弛豫/核验协议更新 | 对相应结构重新标注与评分 | 结构生成资产 |
| refiner/checkpoint/步数改变 | refined 端点及其标注/评价 | 不变的 native 结构和 native 标签 |
| 修订策略/权重改变 | 修订后的 native 与其下游 | 若条件完全一致，可保留修订前构造作为共享起点；当前 body CLI 尚没有完整的“从已构造状态继续修订”入口 |
| P0、化学域、实际 Plans 改变 | 受影响请求的构造及下游 | 原模型文件、训练特征中仍兼容的部分、已知官方参考缓存 |

标签复用必须同时匹配**物理结构、trajectory/请求身份和评价协议**。评分复用还要匹配输入文件、标签、参考库/训练参考和评分实现身份。不能仅因 seed 或文件名相同就复用。

多 seed 合并时保留全局 `sample_idx`，重新在该臂整体 cohort 上计算 N/U；不能相加各 seed 的 unique 数。无效、未完成、未知和工程错误分别记账，不补抽成功样本来更换分母。

## 8. 已有参数的调用示意

以下仅说明现有接口，**本轮不执行**。占位符代表另行登记的配置值；输出目录必须为新目录，已有结果只读。阶段 CLI 不会自动替使用者申请 Slurm 资源。

```bash
# 直接复用实际 Plans；不再生成一遍 Planner。
python "$SOURCE/operations/r03_c3fd_main_20260907/run_component.py" \
  --run-root "$RUN_ROOT" --output-dir "$NEW_COMPONENT" \
  --role G --method-id R03_G --requests 256 \
  --planner-seed "$PLAN_SEED" --body-seed "$BODY_SEED" \
  --refiner-seed "$REFINE_SEED" --sample-index-offset 0 \
  --plans-jsonl "$PLANS_WITH_PROGRAMS" --plans-include-programs \
  --construction-geometry --body-batch-size 1

# 独立复用已保存图做精修，无需重跑 body。
python "$SOURCE/src/scripts/refine_dlm_with_crysllmgen.py" \
  --proposal-graphs "$PROPOSAL_GRAPHS" --checkpoint "$REFINER" \
  --crysllmgen-dir "$CRYSLLMGEN" --output-dir "$NEW_REFINE_DIR" \
  --diff-steps 800 --num-evals 1 --batch-size 1 \
  --seed "$REFINE_SEED" --seed-by-sample-index --max-proposals 256

# 已完成 R/I 的精修评分预览：仍调用同一套 SUN 算法。
python "$SOURCE/operations/r03_c3fd_main_20260907/run_evaluation_stage.py" \
  --run-root "$RUN_ROOT" --phase pilot256 \
  --preview-roles R I --preview-endpoints tau800
```

限制：`run_component.py` 仍硬编码了一些资产路径、阶段顺序和 tau800；`run_evaluation_stage.py` 仍绑定本轮 phase/目录和请求数。本轮预览目录已经存在，不能原封不动重跑上面的预览命令覆盖结果。另一个任意规模实验应使用底层已参数化 CLI，并在后续统一编排改造中补齐配置入口。

## 9. 后续 general 化：扩现有入口，不新建一套 pipeline

| 现有位置 | 当前任务特例 | 建议移到参数/配置 |
|---|---|---|
| `run_component.py` | `PROJECT/P0/B0/LLAMA/LLADA/CRYS/FROZEN/REFINER`、pointer_40395、固定阶段列表 | 资产 manifest；显式 `stages`、共享 Plans/构造输入、端点与并行参数 |
| `run_trial.py` | canary/pilot/formal 枚举、16/256/500、角色配对、目录和 seed、固定训练作业路径 | 方法列表、seed 列表、每 seed 数量、全局索引、资源 lane、实际资产 ID |
| `run_evaluation_stage.py` | phase 与物理目录耦合、预览文件命名、固定 R/I 复用路径 | component manifest、`endpoints`、独立 `hull_phase`、明确的 `reuse_report` 与输出位置 |
| `evaluate_trial.py` | 试跑角色/数量限制、方法选择规则和评分一起编排 | 通用完整性/配对/评分层；实验专属 adoption 与 freeze 规则放独立配置 |
| `submit_stage.py` / `.sbatch` | GPU 型号、分区、2卡8CPU、依赖对象、作业号、2026-09-07 截止时间 | 资源配置、截止时间、任意已登记依赖、CPU step 与独立作业模式 |
| `run_hull_stage.py` / `run_hull_query.py` | 固定 CONFIG/base cache/阶段/凭据 provider | reference manifest、缓存列表、查询运行位置和 provider 引用 |
| `train_*.sbatch` / `prepare_assets.py` | 固定训练目录、checkpoint、预算、命名 | assets/data/training 配置；优先使用已有 prepared-dir 和 feature-cache-dir |
| 部署、状态、传输脚本 | 主机、pane、commit、固定角色、固定文件清单 | 原 transport 入口的 `host/pane/source-commit/files-manifest/output-dir` 参数 |

配置分为三部分即可：

1. **assets**：模型、tokenizer、prompt、domain、训练/参考清单及哈希。
2. **experiment**：实际请求、seed、方法参数、阶段依赖、共享条件、输出目录。
3. **execution/evaluation**：环境、资源、查询位置、指标协议、端点、恢复/缓存规则。

下面是未来配置的示意，**当前没有这样的统一解析器，不能把它当成已支持的 API**：

```yaml
assets_manifest: assets.json
run_id: experiment_name
requests:
  count_per_seed: 256
  seeds: [seed_a, seed_b]
methods:
  - id: baseline
    formula_constraint: none
    construction_geometry: false
    repair: none
  - id: candidate
    formula_constraint: c3fd_domain_ref
    construction_geometry: true
    repair: {checkpoint_ref: repair_adapter_ref}
stages: [plans, programs, construct, repair, refine, validity, labels, sun]
evaluation:
  endpoints: [native, tau800]
  schedule_order: [tau800, native]
  validity_metrics: [comp_valid, struct_valid]
  protocol_ref: frozen_evaluation_config
  hull_phase: shared_reference_cohort
execution:
  resource_profile: two_gpu_profile
  query_location: login
  deadline_utc: explicit_deadline
```

改造顺序：先抽出硬编码资产与资源配置；再让现有编排接受方法/阶段/端点清单；接着补“从现有阶段产物继续”的入口与严格缓存复用；最后归并运维脚本。先保持科学默认值和原始输出协议不变，再单独登记任何方法变化。

## 10. 已踩过的坑，直接作为通用化要求保留

| 问题 | 本轮事实 | 通用要求 |
|---|---|---|
| 科学启发式误拒绝 | 当前 C3FD 是单价态支持；混合价和部分元素资料覆盖存在边界 | 域版本、见证语义与 UNKNOWN 状态显式记录；不能用配置重命名掩盖定义改变 |
| 几何前缀死路 | G/P 各38/256无合法延伸 | 保留失败前缀和原因；未来恢复策略单独配置，不能默认无限重采样 |
| N/U 被坏原生结构阻塞 | R native 首个原子重叠结构的旧 matcher 长时间不推进；tau800 旧函数正常完成 | 端点独立排程/恢复；超时必须显式记录，不能记成 novel、stable 或科学失败；不偷偷跳过匹配 |
| 时间解析 | PowerShell 七位小数秒，远端 Python 3.10 不接受原形式 | 统一时间序列化/兼容解析；保留原时间戳 |
| 参考库 phase 不一致 | 预览显示名被当成缓存实验阶段 | `run/preview` 名称与 `hull_phase` 分开；覆盖校验仍保留 |
| 计算节点无网 | HTTP 查询作业失败，改登录节点成功 | 网络查询与计算阶段分离，不在节点反复重试联网 |
| 日志污染结束标记 | tqdm 回车/ANSI 覆盖 relay 标记 | 结构化 JSON、独立完整帧、nonce、实际退出码；不靠日志尾部猜成功 |
| GitHub 同步延迟 | 小修复也可能 fetch 超时 | 已用增量 Git bundle + 分块 SHA 校验 + 原 archive 路径；避免另写特定 commit 部署器；限制自动 gc/pack 线程 |
| 全流程重跑 | 标签、参考缓存、评分都有独立身份 | 根据依赖只失效下游；不同输出名不代表数据可以重新计数 |

新通用入口的验收重点：用同一配置重放可得到一致的实际命令；能从现有资产跳过已完成阶段；不串用 native/refined 标签；不混训练/评价；不跨方法错误合并；失败/unknown 分母正确；能在不新建任务脚本的情况下换 seed、规模、角色组合、端点和资源。已有相关测试可选取必要部分，不为每个新任务复制整套测试。

## 11. 文件治理约定

2026-09-08静态盘点：`operations/r03_c3fd_main_20260907/`顶层有57个文件（包含.gitignore和本地.relay_state.json）、43个Git跟踪文件，其中deploy/probe/status分别7/12/2个。文件名不证明每个文件无用，但反映同类运维入口已经分散。整改依据见[计划审计](EXPERT_REPAIR_PLAN_AUDIT.md)；当前只登记，未实施归并。

| 职责 | 优先复用的现有位置 | 下轮收敛要求 |
|---|---|---|
| 编辑模型、表示、动作 | `state_conditioned_model.py`、既有token codec与`programmed_path_runtime.py` | 缺失机制进入通用模块；不得把旧合同原样改名当新方法 |
| 数据与任务模式 | `r03_physics_transfer.py`、`export_r03_evaluation_inputs.py`、`label_programmed_paths.py` | 明确编辑数据合同和train/evaluation用途，复用读取/结构/标签接口 |
| 训练与独立修订 | `train_r03_physics_transfer.py`、`run_r03_integrated_body.py` | 参数化任务、完整恢复、DDP与保存初稿输入；科学默认值按模式版本化 |
| 阶段与恢复 | `run_component.py`、`run_trial.py`、`submit_stage.py`、`run_evaluation_stage.py` | 同一编排接资产、方法、端点、资源与截止时间manifest |
| 传输与诊断 | `tmux_relay.py`、持久relay准备/回收及现有部署/状态入口 | 一个规范部署入口、一个状态/诊断入口；commit/文件清单变参数，不再新增每个hash的脚本 |

最新双任务编辑设计仍在这些边界内实现，不为几何/稳定性分别复制训练器或推理器。复用状态模块时补实task/预算条件和异常晶格原始数值通道；不把已有但未进入forward的字段当功能。旧概率均值几何loss与纠错位置的B0 reference KL需要重新审查，保留codec、读取/保存及协议接口，不整段照搬科学目标。

- 算法放在 `src/crystal_dlm/`，阶段 CLI 继续放在 `src/scripts/` 或现有 `scripts/`；编排只组织阶段，不再复制算法。
- 后续新实验原则上只新增配置、manifest、运行目录和报告。
- `deploy_<hash>_remote.py`、同功能 `status_*`、为单个问题建立的 `probe_*` 应归并到已有入口的参数/子命令。历史文件先留档，不在本次整理中删除。
- 不为 native/tau800、R/I/G/P、16/256/512 各复制一份核心执行器。
- 只有数据模型或算法机制确实不同，且现有模块无法合理表达时才新增代码文件；写明边界和复用点。
- 本次新机制优先集中在一至两个通用核心模块，分别负责编辑/评分/采样与数据编译；这不是强制文件数上限，合理职责拆分仍优于巨型脚本。
- Git只加入源码、必要测试、配置、数据/模型manifest和精选结果。权重、大轨迹、临时bundle及整目录回执不随实验批量提交；每次显式选文件，不用全仓库暂存掩盖临时产物。
- 先确认活动入口和调用关系，再归并历史脚本；活动调用保留兼容或明确退役记录，历史结果及证据不因整理丢失。完成后用另一规模/seed/端点配置验证无需新增执行脚本。
- 本文件为后续整理的主索引。完成某一项参数化后，在这里更新“现有参数”和“验证范围”，不要再创建一份平行的任务专用资产清单。

本次交付：资产和接口盘点、复用条件、硬编码清单、具体归并职责及改造顺序。**没有实施上述 general 化，没有迁移/删除模型资产，也没有重新启动实验。**
