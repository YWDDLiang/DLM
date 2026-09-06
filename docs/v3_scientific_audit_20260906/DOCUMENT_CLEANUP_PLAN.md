# 文档清理批准执行清单

本清单只规划本工作树 docs。主审已协调阅读作者并明确批准 historical_docs、component_reference、六份重复复用及本树 PMTR 历史说明15/16；本清单构建工具本身不执行迁移，实际结果见随后生成的迁移报告。

本次登记 **442** 个文件，其中 audit 外 **162** 个。动作计数：`{"archive": 140, "move_reference": 2, "reuse_immutable_archive": 6, "retain": 294}`。逐源原 SHA、原库存对照、分类原因、双向链接、代码/测试引用和建议替换均在 [JSON](DOCUMENT_CLEANUP_PLAN.json)。

源库存的 209 是全仓库 Markdown 数，其中 docs 有 136；559 是失败文本线索数，不是独立失败实验数。当前还纳入 JSON 统计、推导脚本、图和新增审计证据，故总数不可直接与209相减解释为新增报告。

## 已协调迁移的具体范围

- `historical_docs`：除下表保护项外，旧 top-level 计划/状态/交接/结果、teacher_feedback 历史（含本树PMTR说明15/16）、旧 periodic review_notes 和历史 paper 说明，镜像保存在本审计目录 `historical/docs/`。
- `component_reference`：两份仍有组件参考价值的 typed Planner/soft-field 语义说明，移到 `reference/component_specs/`；不把旧执行预算或结果改称当前事实。
- `reuse_immutable`：六份已重新核 SHA 的 C³FD docs 副本复用既有 immutable archive。归档原件不动；重复源逐文件去重，待主审统一改其负责的入口/审计引用。原库存另五组 results 重复不在本任务范围。

主审已完成旧文档读取并通知上游作者使用冻结 Git 版本或映射。JSON 保留逐引用边界；本子任务只改获准的历史文档、剩余 legacy 文档及稳定 root WORKFLOW/PAPER_PIPELINE/REPRODUCTION 的链接。

## 保持原位的入口与执行依赖

| 文件 | 原因 |
|---|---|
| `docs/CURRENT_ARCHITECTURE.md` | 当前架构入口；主审正在维护。 |
| `docs/CURRENT_STATE.md` | 当前状态入口；主审正在维护。 |
| `docs/DATA_LICENSES.md` | 数据许可证仍需直接可用。 |
| `docs/MODEL_LICENSES.md` | 模型许可证仍需直接可用。 |
| `docs/PLACEHOLDER_ASSETS.md` | 现有环境/下载/提交脚本仍把此路径显示为帮助入口；是发布资产说明。 |
| `docs/SEEDS.md` | slurm/_common.sh 的帮助文本仍指向此历史种子清单。 |
| `docs/paper/METHOD_AT_A_GLANCE.md` | test_paper_pipeline_layout 读取并校验历史 G2 术语，暂保留字节。 |
| `docs/paper/README.md` | test_paper_pipeline_layout 读取的历史 G2 兼容入口，不宣称当前方法。 |
| `docs/periodic_self_repair_v1/CHECKLIST.md` | V2 当前执行/设计/验收/初始化链仍在使用；状态段可过期，但本次保持路径与内容。 |
| `docs/periodic_self_repair_v1/LOSS_GEOMETRY_AND_V2_PROPOSAL_20260906.md` | V2 当前执行/设计/验收/初始化链仍在使用；状态段可过期，但本次保持路径与内容。 |
| `docs/periodic_self_repair_v1/RAW_LLADA_DESIGN.md` | V2 当前执行/设计/验收/初始化链仍在使用；状态段可过期，但本次保持路径与内容。 |
| `docs/periodic_self_repair_v1/V2_EXECUTION_PACKAGE_20260906.md` | V2 当前执行/设计/验收/初始化链仍在使用；状态段可过期，但本次保持路径与内容。 |
| `docs/periodic_self_repair_v1/V2_IMPLEMENTATION_REVIEW_20260906.md` | V2 当前执行/设计/验收/初始化链仍在使用；状态段可过期，但本次保持路径与内容。 |
| `docs/periodic_self_repair_v1/V2_LAUNCH_MANIFEST.json` | V2 当前执行/设计/验收/初始化链仍在使用；状态段可过期，但本次保持路径与内容。 |

当前 audit 目录整体保留，包含主审已存档的六个 root 入口。`PAPER_PIPELINE.md` 的旧 CLI 已明确是历史 G2；本任务不改父审正在维护的 root README 或入口。

依赖核验：paper 两文件由 `tests/test_paper_pipeline_layout.py` 读取；V2 manifest 由 `tests/test_periodic_v2_queue.py` 读取。`PLACEHOLDER_ASSETS.md` 与 `SEEDS.md` 仍是脚本帮助链接；`src/crystal_dlm/d3po.py:5` 对 D3PO 合同是 docstring 引用，不是运行时读取。迁移该旧合同后需更新这个已定位的引用。

两份旧分析脚本也已核路径假设：`continuous_math_checks.py` 写相邻同名 JSON，因此与 JSON 一起归档；`analyze_round0_gap_change.py` 从 argv 接受 run root，不依赖所在 docs 深度。它们只作为历史证据移动，不在清理中执行。JSON 台账内记录的历史来源路径保留原值，不当成 Markdown 链接批量替换。

## 六份可复用的逐字节重复

| 当前副本 | 既有原件 | SHA256 |
|---|---|---|
| `docs/C3FD_V21_CORRECTION_CONTRACT.md` | `archives/successful_contributions_20260828/c3fd_v2_5/docs/C3FD_V21_CORRECTION_CONTRACT.md` | `04a6bc199ac2107e4aad1be2808fedd127b440649977ea9f932c321472e91c9e` |
| `docs/C3FD_V22_JOINT_REACHABILITY_CONTRACT.md` | `archives/successful_contributions_20260828/c3fd_v2_5/docs/C3FD_V22_JOINT_REACHABILITY_CONTRACT.md` | `c7780c43c376d0343d3131cd2ef75e5b7ad9a81cd4ee6ffb6b2bd1df047a4ef7` |
| `docs/C3FD_V23_PAULING_WITNESS_CONTRACT.md` | `archives/successful_contributions_20260828/c3fd_v2_5/docs/C3FD_V23_PAULING_WITNESS_CONTRACT.md` | `49663c34c231d6b085e325ad6d835fad444efe64733ba97df7309ea845e533a0` |
| `docs/C3FD_V24_BITSET_WITNESS_CONTRACT.md` | `archives/successful_contributions_20260828/c3fd_v2_5/docs/C3FD_V24_BITSET_WITNESS_CONTRACT.md` | `a638961e507e75caf47adecc74df0c4596b704fa3b5966e4cc0ea5f8b8de243f` |
| `docs/C3FD_V25_ONLINE_CANARY_CONTRACT.md` | `archives/successful_contributions_20260828/c3fd_v2_5/docs/C3FD_V25_ONLINE_CANARY_CONTRACT.md` | `3fff07f59465d456a3267a57c00e27afe771ba58dc87ecea7bf1ae42389a3c9a` |
| `docs/CCFD_V2_SEMANTIC_COMPILER.md` | `archives/successful_contributions_20260828/c3fd_v2_5/docs/CCFD_V2_SEMANTIC_COMPILER.md` | `680b40d47bff5a0ee7250afd3a9326343664f1c1d5a0b8aab3cc1a7146751e77` |

## 逐文件迁移表

JSON 保留所有受保护审计证据的逐文件登记。下表列出 audit 外所有建议迁移/复用项；原始科学证据和配套 JSON/推导脚本一起保存。

| 源文件 | 动作 / 分类 | 目标或复用位置 | 入链 / 出链 / 代码测试引用 |
|---|---|---|---|
| `docs/36H_FINAL_REPORT_C3FD_G2_20260901.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/36H_FINAL_REPORT_C3FD_G2_20260901.json` | 1 / 0 / 0 |
| `docs/36H_FINAL_REPORT_C3FD_G2_20260901.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/36H_FINAL_REPORT_C3FD_G2_20260901.md` | 3 / 0 / 0 |
| `docs/ADR_C3FD_LLAMA_COMPACT_V2_MAINLINE_20260831.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/ADR_C3FD_LLAMA_COMPACT_V2_MAINLINE_20260831.md` | 0 / 1 / 0 |
| `docs/ADR_C3FD_LLAMA_FUSED_TYPED_PLANNER_20260831.md` | `move_reference` / `still_relevant_component_reference` | `docs/v3_scientific_audit_20260906/reference/component_specs/ADR_C3FD_LLAMA_FUSED_TYPED_PLANNER_20260831.md` | 4 / 0 / 0 |
| `docs/ASSET_TRANSFER_LEDGER.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/ASSET_TRANSFER_LEDGER.md` | 5 / 0 / 0 |
| `docs/BUILD_STATUS.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/BUILD_STATUS.md` | 0 / 30 / 0 |
| `docs/C3FD_FULL_RICH_PLANNER_SUN_CHECKLIST_DRAFT_V5.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_FULL_RICH_PLANNER_SUN_CHECKLIST_DRAFT_V5.json` | 0 / 0 / 0 |
| `docs/C3FD_FULL_RICH_PLANNER_SUN_CHECKLIST_DRAFT_V5.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_FULL_RICH_PLANNER_SUN_CHECKLIST_DRAFT_V5.md` | 0 / 0 / 0 |
| `docs/C3FD_H1A2_FUSION_SUN_CHECKLIST_36H_V4.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_H1A2_FUSION_SUN_CHECKLIST_36H_V4.json` | 0 / 0 / 0 |
| `docs/C3FD_H1A2_FUSION_SUN_CHECKLIST_36H_V4.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_H1A2_FUSION_SUN_CHECKLIST_36H_V4.md` | 0 / 0 / 0 |
| `docs/C3FD_LLAMA_DEVELOPMENT_SUN_FINAL_20260831.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_LLAMA_DEVELOPMENT_SUN_FINAL_20260831.json` | 0 / 0 / 0 |
| `docs/C3FD_LLAMA_DEVELOPMENT_SUN_FINAL_20260831.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_LLAMA_DEVELOPMENT_SUN_FINAL_20260831.md` | 0 / 0 / 0 |
| `docs/C3FD_LLAMA_DLM_SUN_CHECKLIST_V6.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_LLAMA_DLM_SUN_CHECKLIST_V6.json` | 2 / 4 / 0 |
| `docs/C3FD_LLAMA_DLM_SUN_CHECKLIST_V6.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_LLAMA_DLM_SUN_CHECKLIST_V6.md` | 1 / 4 / 0 |
| `docs/C3FD_LLAMA_DLM_TWO_ROUTE_METHOD_V1.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_LLAMA_DLM_TWO_ROUTE_METHOD_V1.md` | 2 / 0 / 0 |
| `docs/C3FD_NATIVE_DLM_SUN50_CHECKLIST_36H_V3.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_NATIVE_DLM_SUN50_CHECKLIST_36H_V3.json` | 0 / 0 / 0 |
| `docs/C3FD_NATIVE_DLM_SUN50_CHECKLIST_36H_V3.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_NATIVE_DLM_SUN50_CHECKLIST_36H_V3.md` | 2 / 0 / 0 |
| `docs/C3FD_NATIVE_SFT_CANARY_GENERATION_FINAL_20260831.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_NATIVE_SFT_CANARY_GENERATION_FINAL_20260831.json` | 0 / 0 / 0 |
| `docs/C3FD_NATIVE_SFT_CANARY_GENERATION_FINAL_20260831.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_NATIVE_SFT_CANARY_GENERATION_FINAL_20260831.md` | 0 / 0 / 0 |
| `docs/C3FD_NATIVE_SFT_CANARY_OFFLINE_FINAL_20260831.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_NATIVE_SFT_CANARY_OFFLINE_FINAL_20260831.json` | 0 / 0 / 0 |
| `docs/C3FD_NATIVE_SFT_CANARY_OFFLINE_FINAL_20260831.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_NATIVE_SFT_CANARY_OFFLINE_FINAL_20260831.md` | 1 / 0 / 0 |
| `docs/C3FD_NATIVE_TEACHER_SFT_DATA_FINAL_20260831.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_NATIVE_TEACHER_SFT_DATA_FINAL_20260831.json` | 1 / 0 / 0 |
| `docs/C3FD_NATIVE_TEACHER_SFT_DATA_FINAL_20260831.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_NATIVE_TEACHER_SFT_DATA_FINAL_20260831.md` | 0 / 0 / 0 |
| `docs/C3FD_NATIVE_TEACHER_SFT_FINAL_20260831.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_NATIVE_TEACHER_SFT_FINAL_20260831.json` | 1 / 0 / 0 |
| `docs/C3FD_NATIVE_TEACHER_SFT_FINAL_20260831.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_NATIVE_TEACHER_SFT_FINAL_20260831.md` | 2 / 0 / 0 |
| `docs/C3FD_NATIVE_TRAIN_INFERENCE_INTERFACE_AUDIT_20260831.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_NATIVE_TRAIN_INFERENCE_INTERFACE_AUDIT_20260831.json` | 0 / 0 / 0 |
| `docs/C3FD_NATIVE_TRAIN_INFERENCE_INTERFACE_AUDIT_20260831.md` | `archive` / `historical_analysis` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_NATIVE_TRAIN_INFERENCE_INTERFACE_AUDIT_20260831.md` | 0 / 0 / 0 |
| `docs/C3FD_RICH_FIELD_SEMANTICS_AUDIT_V2.md` | `move_reference` / `still_relevant_component_reference` | `docs/v3_scientific_audit_20260906/reference/component_specs/C3FD_RICH_FIELD_SEMANTICS_AUDIT_V2.md` | 3 / 0 / 0 |
| `docs/C3FD_RICH_INTERFACE_V1_CONTRACT.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/C3FD_RICH_INTERFACE_V1_CONTRACT.md` | 0 / 2 / 0 |
| `docs/C3FD_V21_CORRECTION_CONTRACT.md` | `reuse_immutable_archive` / `exact_duplicate_component_history` | `archives/successful_contributions_20260828/c3fd_v2_5/docs/C3FD_V21_CORRECTION_CONTRACT.md` | 1 / 0 / 0 |
| `docs/C3FD_V22_JOINT_REACHABILITY_CONTRACT.md` | `reuse_immutable_archive` / `exact_duplicate_component_history` | `archives/successful_contributions_20260828/c3fd_v2_5/docs/C3FD_V22_JOINT_REACHABILITY_CONTRACT.md` | 0 / 0 / 0 |
| `docs/C3FD_V23_PAULING_WITNESS_CONTRACT.md` | `reuse_immutable_archive` / `exact_duplicate_component_history` | `archives/successful_contributions_20260828/c3fd_v2_5/docs/C3FD_V23_PAULING_WITNESS_CONTRACT.md` | 0 / 0 / 0 |
| `docs/C3FD_V24_BITSET_WITNESS_CONTRACT.md` | `reuse_immutable_archive` / `exact_duplicate_component_history` | `archives/successful_contributions_20260828/c3fd_v2_5/docs/C3FD_V24_BITSET_WITNESS_CONTRACT.md` | 0 / 0 / 0 |
| `docs/C3FD_V25_ONLINE_CANARY_CONTRACT.md` | `reuse_immutable_archive` / `exact_duplicate_component_history` | `archives/successful_contributions_20260828/c3fd_v2_5/docs/C3FD_V25_ONLINE_CANARY_CONTRACT.md` | 0 / 0 / 0 |
| `docs/CCFD_V2_SEMANTIC_COMPILER.md` | `reuse_immutable_archive` / `exact_duplicate_component_history` | `archives/successful_contributions_20260828/c3fd_v2_5/docs/CCFD_V2_SEMANTIC_COMPILER.md` | 0 / 1 / 0 |
| `docs/CTV_DLM_CONTRIBUTION_STACK_V1.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/CTV_DLM_CONTRIBUTION_STACK_V1.md` | 0 / 0 / 0 |
| `docs/CTV_DLM_DECISION_LOG_V1.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/CTV_DLM_DECISION_LOG_V1.md` | 0 / 0 / 0 |
| `docs/CTV_DLM_IMPLEMENTATION_STATUS_V1.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/CTV_DLM_IMPLEMENTATION_STATUS_V1.md` | 0 / 0 / 0 |
| `docs/CTV_DLM_RELATED_WORK_POSITIONING_V1.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/CTV_DLM_RELATED_WORK_POSITIONING_V1.md` | 0 / 0 / 0 |
| `docs/CTV_DLM_STABILITY_CONTRACT_V1.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/CTV_DLM_STABILITY_CONTRACT_V1.md` | 2 / 0 / 0 |
| `docs/CTV_DLM_V1_FINAL_NO_GO.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/CTV_DLM_V1_FINAL_NO_GO.md` | 2 / 0 / 0 |
| `docs/CTV_Q_HEAD_CONTRACT_V1.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/CTV_Q_HEAD_CONTRACT_V1.md` | 0 / 1 / 0 |
| `docs/D3PO_256_MIN_CONTRACT_V1.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/D3PO_256_MIN_CONTRACT_V1.md` | 3 / 2 / 1 |
| `docs/D3PO_37966_ENGINEERING_FAILURE.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/D3PO_37966_ENGINEERING_FAILURE.md` | 2 / 0 / 0 |
| `docs/D3PO_38034_ENGINEERING_FAILURE.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/D3PO_38034_ENGINEERING_FAILURE.md` | 2 / 0 / 0 |
| `docs/D3PO_38233_ENGINEERING_FAILURE.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/D3PO_38233_ENGINEERING_FAILURE.md` | 2 / 0 / 0 |
| `docs/D3PO_FALLBACK_DECISION_TREE_V1.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/D3PO_FALLBACK_DECISION_TREE_V1.md` | 3 / 1 / 0 |
| `docs/DE_NOVO_SCOPE_INTERNAL.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/DE_NOVO_SCOPE_INTERNAL.md` | 0 / 0 / 0 |
| `docs/DLM_CAPABILITY_REGRESSION_36H.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_CAPABILITY_REGRESSION_36H.md` | 1 / 0 / 0 |
| `docs/DLM_DIRECT_STABILITY_DECISION_20260902.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_DIRECT_STABILITY_DECISION_20260902.md` | 0 / 6 / 0 |
| `docs/DLM_EXPERIMENT_LINEAGE_FOR_D3PO_V1.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_EXPERIMENT_LINEAGE_FOR_D3PO_V1.md` | 1 / 0 / 0 |
| `docs/DLM_NOISY_STATE_ENERGY_CRITIC_V1.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_NOISY_STATE_ENERGY_CRITIC_V1.md` | 1 / 4 / 0 |
| `docs/DLM_ORIGINAL_ARTIFACT_EVIDENCE_LEDGER_36H_V1.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_ORIGINAL_ARTIFACT_EVIDENCE_LEDGER_36H_V1.json` | 1 / 7 / 0 |
| `docs/DLM_ORIGINAL_ARTIFACT_EVIDENCE_LEDGER_36H_V1.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_ORIGINAL_ARTIFACT_EVIDENCE_LEDGER_36H_V1.md` | 3 / 6 / 0 |
| `docs/DLM_POST_FM_STRUCTURAL_LEARNING_AND_REFINER_FEEDBACK_PLAN_V2.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_POST_FM_STRUCTURAL_LEARNING_AND_REFINER_FEEDBACK_PLAN_V2.json` | 0 / 0 / 0 |
| `docs/DLM_POST_FM_STRUCTURAL_LEARNING_AND_REFINER_FEEDBACK_PLAN_V2.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_POST_FM_STRUCTURAL_LEARNING_AND_REFINER_FEEDBACK_PLAN_V2.md` | 3 / 0 / 0 |
| `docs/DLM_RICH_PLANNER_DECISION_LOG_V2.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_RICH_PLANNER_DECISION_LOG_V2.md` | 1 / 0 / 0 |
| `docs/DLM_RICH_PLANNER_STABLE_DLM_CHECKLIST_36H_V2.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_RICH_PLANNER_STABLE_DLM_CHECKLIST_36H_V2.md` | 1 / 2 / 0 |
| `docs/DLM_RICH_PLAN_STABILITY_TRAINING_PLAN_V1.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_RICH_PLAN_STABILITY_TRAINING_PLAN_V1.md` | 0 / 0 / 0 |
| `docs/DLM_STABILITY_LITERATURE_DECISION_20260830.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_STABILITY_LITERATURE_DECISION_20260830.md` | 0 / 10 / 0 |
| `docs/DLM_STABILITY_OPTION_CHECKLIST_36H_V1.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_STABILITY_OPTION_CHECKLIST_36H_V1.json` | 0 / 0 / 0 |
| `docs/DLM_STABILITY_OPTION_CHECKLIST_36H_V1.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_STABILITY_OPTION_CHECKLIST_36H_V1.md` | 0 / 0 / 0 |
| `docs/DLM_STABILITY_PROGRAM_AUDIT_20260830.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_STABILITY_PROGRAM_AUDIT_20260830.json` | 0 / 0 / 0 |
| `docs/DLM_STABILITY_PROGRAM_AUDIT_20260830.md` | `archive` / `historical_analysis` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_STABILITY_PROGRAM_AUDIT_20260830.md` | 2 / 0 / 0 |
| `docs/DLM_SUN_STABILITY_MECHANISM_DEEP_DIVE_V2.md` | `archive` / `historical_analysis` | `docs/v3_scientific_audit_20260906/historical/docs/DLM_SUN_STABILITY_MECHANISM_DEEP_DIVE_V2.md` | 2 / 12 / 0 |
| `docs/DUAL_CANDIDATE_PROGRAM_INTERNAL.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/DUAL_CANDIDATE_PROGRAM_INTERNAL.md` | 0 / 0 / 0 |
| `docs/DUAL_TRACK_COMPOSITION_STABILITY_PLAN_V1.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/DUAL_TRACK_COMPOSITION_STABILITY_PLAN_V1.md` | 2 / 0 / 0 |
| `docs/EXPERIMENT_PRIORITIES_INTERNAL.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/EXPERIMENT_PRIORITIES_INTERNAL.md` | 1 / 0 / 0 |
| `docs/FAILED_METHODS.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/FAILED_METHODS.md` | 10 / 0 / 0 |
| `docs/FORCE_SCORE_DLM_CHECKLIST_V1.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/FORCE_SCORE_DLM_CHECKLIST_V1.md` | 2 / 0 / 0 |
| `docs/G2_FULL_EPOCH_AB_CHECKLIST_V1.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/G2_FULL_EPOCH_AB_CHECKLIST_V1.json` | 0 / 1 / 0 |
| `docs/G2_FULL_EPOCH_AB_CHECKLIST_V1.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/G2_FULL_EPOCH_AB_CHECKLIST_V1.md` | 0 / 1 / 0 |
| `docs/G2_FULL_EPOCH_AB_FINAL_20260901.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/G2_FULL_EPOCH_AB_FINAL_20260901.json` | 1 / 0 / 0 |
| `docs/G2_FULL_EPOCH_AB_FINAL_20260901.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/G2_FULL_EPOCH_AB_FINAL_20260901.md` | 1 / 0 / 0 |
| `docs/GPT6_AUDIT_HANDOFF_20260905.md` | `archive` / `superseded_handoff` | `docs/v3_scientific_audit_20260906/historical/docs/GPT6_AUDIT_HANDOFF_20260905.md` | 4 / 40 / 0 |
| `docs/HANDOFF_STATE_PROGRAMMED_SPAD_20260905.md` | `archive` / `superseded_handoff` | `docs/v3_scientific_audit_20260906/historical/docs/HANDOFF_STATE_PROGRAMMED_SPAD_20260905.md` | 1 / 19 / 0 |
| `docs/ICLR_REVIEW_STRATEGY.md` | `archive` / `historical_analysis` | `docs/v3_scientific_audit_20260906/historical/docs/ICLR_REVIEW_STRATEGY.md` | 0 / 8 / 0 |
| `docs/INTERNAL_RESULTS.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/INTERNAL_RESULTS.md` | 0 / 0 / 0 |
| `docs/NEXT_CONVERSATION_PROMPT.md` | `archive` / `superseded_handoff` | `docs/v3_scientific_audit_20260906/historical/docs/NEXT_CONVERSATION_PROMPT.md` | 0 / 7 / 0 |
| `docs/PAPER_STORY_INTERNAL.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/PAPER_STORY_INTERNAL.md` | 1 / 9 / 0 |
| `docs/PLAN1200_TAU800_FINAL_20260902.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/PLAN1200_TAU800_FINAL_20260902.md` | 5 / 0 / 0 |
| `docs/PLANNER_CRYSVCD_PHYSICS_EMBEDDING_FEASIBILITY_V1.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/PLANNER_CRYSVCD_PHYSICS_EMBEDDING_FEASIBILITY_V1.md` | 0 / 0 / 0 |
| `docs/PLANNER_PROMPT.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/PLANNER_PROMPT.md` | 1 / 0 / 0 |
| `docs/RELATED_WORK_INTERNAL.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/RELATED_WORK_INTERNAL.md` | 1 / 20 / 0 |
| `docs/RICH_DLM_EXECUTION_GAP_AUDIT_V1.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/RICH_DLM_EXECUTION_GAP_AUDIT_V1.md` | 1 / 3 / 0 |
| `docs/RICH_RECOVERY_CANARY_OFFLINE_FINAL_20260830.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/RICH_RECOVERY_CANARY_OFFLINE_FINAL_20260830.md` | 2 / 0 / 0 |
| `docs/ROLLOUT_MATCHED_DLM_24H_CHECKLIST_V1.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/ROLLOUT_MATCHED_DLM_24H_CHECKLIST_V1.md` | 5 / 1 / 0 |
| `docs/RRC_DLM_V1_PROPOSED_CONTRACT.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/RRC_DLM_V1_PROPOSED_CONTRACT.md` | 1 / 1 / 0 |
| `docs/SECOND_CONTRIBUTION_CCFD_DLM_REVIEW_V1.md` | `archive` / `historical_analysis` | `docs/v3_scientific_audit_20260906/historical/docs/SECOND_CONTRIBUTION_CCFD_DLM_REVIEW_V1.md` | 1 / 8 / 0 |
| `docs/SGTC_DLM_L7_CONTRACT_V1.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/SGTC_DLM_L7_CONTRACT_V1.md` | 2 / 0 / 0 |
| `docs/SGTC_DLM_V1_CONTRACT.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/SGTC_DLM_V1_CONTRACT.md` | 3 / 0 / 0 |
| `docs/SI_LWA_V1_CONTRACT.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/SI_LWA_V1_CONTRACT.json` | 0 / 0 / 0 |
| `docs/SI_LWA_V1_CONTRACT.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/SI_LWA_V1_CONTRACT.md` | 0 / 0 / 0 |
| `docs/STORY_REVIEW_INTERNAL.md` | `archive` / `historical_analysis` | `docs/v3_scientific_audit_20260906/historical/docs/STORY_REVIEW_INTERNAL.md` | 0 / 0 / 0 |
| `docs/TOKEN_NATIVE_PBC_GEOMETRY_EXECUTOR_V1.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/TOKEN_NATIVE_PBC_GEOMETRY_EXECUTOR_V1.md` | 1 / 0 / 0 |
| `docs/paper/EXPERIMENT_MATRIX.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/paper/EXPERIMENT_MATRIX.md` | 1 / 6 / 0 |
| `docs/paper/LEGACY_TO_MAINLINE_MAP.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/paper/LEGACY_TO_MAINLINE_MAP.md` | 1 / 0 / 0 |
| `docs/paper/METHOD_MAINLINE.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/paper/METHOD_MAINLINE.md` | 1 / 0 / 0 |
| `docs/paper/PAPER_STORY.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/paper/PAPER_STORY.md` | 1 / 1 / 0 |
| `docs/paper/PIPELINE_INVENTORY.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/paper/PIPELINE_INVENTORY.md` | 1 / 0 / 0 |
| `docs/paper/PSTR_METHOD_CONTRACT.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/paper/PSTR_METHOD_CONTRACT.md` | 0 / 0 / 0 |
| `docs/paper/REPRODUCIBILITY.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/paper/REPRODUCIBILITY.md` | 2 / 1 / 0 |
| `docs/paper/SCIENTIFIC_QUESTION_AND_CONTRIBUTIONS.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/paper/SCIENTIFIC_QUESTION_AND_CONTRIBUTIONS.md` | 1 / 0 / 0 |
| `docs/paper/architecture_mainline.mmd` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/paper/architecture_mainline.mmd` | 2 / 0 / 0 |
| `docs/periodic_self_repair_v1/DESIGN_AND_AUDIT.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/periodic_self_repair_v1/DESIGN_AND_AUDIT.md` | 3 / 4 / 0 |
| `docs/periodic_self_repair_v1/GPT56_HANDOFF_20260906.md` | `archive` / `superseded_handoff` | `docs/v3_scientific_audit_20260906/historical/docs/periodic_self_repair_v1/GPT56_HANDOFF_20260906.md` | 0 / 3 / 0 |
| `docs/periodic_self_repair_v1/K8_RESULTS_AND_REGRESSION_ANALYSIS_20260906.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/periodic_self_repair_v1/K8_RESULTS_AND_REGRESSION_ANALYSIS_20260906.md` | 3 / 2 / 0 |
| `docs/periodic_self_repair_v1/review_notes/EVIDENCE_REVIEW_AGENT.md` | `archive` / `historical_analysis` | `docs/v3_scientific_audit_20260906/historical/docs/periodic_self_repair_v1/review_notes/EVIDENCE_REVIEW_AGENT.md` | 1 / 76 / 0 |
| `docs/periodic_self_repair_v1/review_notes/K8_NATIVE_REGRESSION_ANALYSIS.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/periodic_self_repair_v1/review_notes/K8_NATIVE_REGRESSION_ANALYSIS.md` | 1 / 0 / 0 |
| `docs/periodic_self_repair_v1/review_notes/K8_TAU800_REGRESSION_ANALYSIS.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/periodic_self_repair_v1/review_notes/K8_TAU800_REGRESSION_ANALYSIS.md` | 1 / 0 / 0 |
| `docs/periodic_self_repair_v1/review_notes/LOSS_REVIEW_AGENT.md` | `archive` / `historical_analysis` | `docs/v3_scientific_audit_20260906/historical/docs/periodic_self_repair_v1/review_notes/LOSS_REVIEW_AGENT.md` | 0 / 1 / 0 |
| `docs/periodic_self_repair_v1/review_notes/OLD_LLM_DLM_ORDER_TRACE.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/periodic_self_repair_v1/review_notes/OLD_LLM_DLM_ORDER_TRACE.md` | 2 / 51 / 0 |
| `docs/periodic_self_repair_v1/review_notes/V2_MATH_DESIGN_REVIEW.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/periodic_self_repair_v1/review_notes/V2_MATH_DESIGN_REVIEW.md` | 0 / 1 / 0 |
| `docs/teacher_feedback_unified_v1/00_UNIFIED_METHOD_PLAN.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/00_UNIFIED_METHOD_PLAN.md` | 2 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/01_TRACK_A_PURE_LLM.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/01_TRACK_A_PURE_LLM.md` | 2 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/02_TRACK_B_LLM_GUIDED_DLM.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/02_TRACK_B_LLM_GUIDED_DLM.md` | 2 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/03_CROSS_REPRESENTATION_AND_DIFFUSION.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/03_CROSS_REPRESENTATION_AND_DIFFUSION.md` | 2 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/04_EXECUTION_CHECKLIST.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/04_EXECUTION_CHECKLIST.md` | 8 / 14 / 0 |
| `docs/teacher_feedback_unified_v1/05_DECISION_LOG.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/05_DECISION_LOG.md` | 2 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md` | `archive` / `historical_analysis` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md` | 11 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/07_BASIN_POSTERIOR_ALIGNMENT.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/07_BASIN_POSTERIOR_ALIGNMENT.md` | 1 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/08_DLM_NATIVE_STABILITY_AND_DIFFUSION_FALLBACK.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/08_DLM_NATIVE_STABILITY_AND_DIFFUSION_FALLBACK.md` | 2 / 5 / 0 |
| `docs/teacher_feedback_unified_v1/09_EFFICIENCY_FIRST_POTENTIAL_CLOSURE_PLAN.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/09_EFFICIENCY_FIRST_POTENTIAL_CLOSURE_PLAN.md` | 5 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/10_FULL_MP20_BASIN_ACTION_VALUE_PLAN.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/10_FULL_MP20_BASIN_ACTION_VALUE_PLAN.md` | 1 / 1 / 0 |
| `docs/teacher_feedback_unified_v1/11_LATEST_RESULTS_WORKLOG_AND_NEXT_STEPS.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/11_LATEST_RESULTS_WORKLOG_AND_NEXT_STEPS.md` | 2 / 2 / 0 |
| `docs/teacher_feedback_unified_v1/12_LLAMA_PROGRAMMED_BASIN_CLOSURE.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/12_LLAMA_PROGRAMMED_BASIN_CLOSURE.md` | 5 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/13_STREAM19_DIAGNOSIS_AND_FINAL_ITERATION.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/13_STREAM19_DIAGNOSIS_AND_FINAL_ITERATION.md` | 7 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/14_PCTP_SCIENTIFIC_ALIGNMENT_AND_EXECUTION.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/14_PCTP_SCIENTIFIC_ALIGNMENT_AND_EXECUTION.md` | 1 / 1 / 0 |
| `docs/teacher_feedback_unified_v1/15_PMTR_SCIENTIFIC_METHOD_AND_EXECUTION.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/15_PMTR_SCIENTIFIC_METHOD_AND_EXECUTION.md` | 5 / 4 / 0 |
| `docs/teacher_feedback_unified_v1/16_PMTR_CODE_GROUNDED_ARCHITECTURE_AUDIT.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/16_PMTR_CODE_GROUNDED_ARCHITECTURE_AUDIT.md` | 4 / 1 / 0 |
| `docs/teacher_feedback_unified_v1/17_STATE_CONDITIONED_TERMINAL_BASIN_PLAN.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/17_STATE_CONDITIONED_TERMINAL_BASIN_PLAN.md` | 5 / 27 / 0 |
| `docs/teacher_feedback_unified_v1/18_DUAL_OBJECTIVE_REVIEW_AND_DECISIONS.md` | `archive` / `historical_analysis` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/18_DUAL_OBJECTIVE_REVIEW_AND_DECISIONS.md` | 4 / 1 / 0 |
| `docs/teacher_feedback_unified_v1/19_RESUMED_ARCHITECTURE_AND_EXECUTION.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/19_RESUMED_ARCHITECTURE_AND_EXECUTION.md` | 4 / 3 / 0 |
| `docs/teacher_feedback_unified_v1/20_DATA_SUFFICIENCY_AND_DELIVERY_20260906.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/20_DATA_SUFFICIENCY_AND_DELIVERY_20260906.md` | 6 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/21_TERMINAL_REPRODUCIBILITY_AUDIT_20260906.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/21_TERMINAL_REPRODUCIBILITY_AUDIT_20260906.md` | 3 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/22_CONTINUOUS_DIFFUSION_EXTENSION_ASSESSMENT.md` | `archive` / `historical_analysis` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/22_CONTINUOUS_DIFFUSION_EXTENSION_ASSESSMENT.md` | 1 / 5 / 0 |
| `docs/teacher_feedback_unified_v1/23_CONTINUOUS_DIFFUSION_MATHEMATICAL_FEASIBILITY.md` | `archive` / `historical_analysis` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/23_CONTINUOUS_DIFFUSION_MATHEMATICAL_FEASIBILITY.md` | 1 / 13 / 0 |
| `docs/teacher_feedback_unified_v1/24_CONTINUOUS_DIFFUSION_TRAINING_AND_SDE_AUDIT.md` | `archive` / `historical_analysis` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/24_CONTINUOUS_DIFFUSION_TRAINING_AND_SDE_AUDIT.md` | 6 / 15 / 0 |
| `docs/teacher_feedback_unified_v1/25_ROUND0_MATCHED_EVALUATION.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/25_ROUND0_MATCHED_EVALUATION.md` | 5 / 1 / 0 |
| `docs/teacher_feedback_unified_v1/26_ROUND0_GAP_INCREASE_DIAGNOSIS.md` | `archive` / `historical_result_or_failure_evidence` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/26_ROUND0_GAP_INCREASE_DIAGNOSIS.md` | 9 / 3 / 0 |
| `docs/teacher_feedback_unified_v1/27_SELF_IMPROVEMENT_REPAIR_PLAN.md` | `archive` / `historical_design_or_component_contract` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/27_SELF_IMPROVEMENT_REPAIR_PLAN.md` | 8 / 1 / 0 |
| `docs/teacher_feedback_unified_v1/28_FINAL_SPRINT_AND_SELF_REPAIR_CHECK.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/28_FINAL_SPRINT_AND_SELF_REPAIR_CHECK.md` | 1 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/29_PERIODIC_SELF_REPAIR_12H.md` | `archive` / `superseded_status_or_execution_plan` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/29_PERIODIC_SELF_REPAIR_12H.md` | 0 / 6 / 0 |
| `docs/teacher_feedback_unified_v1/README.md` | `archive` / `historical_context_or_reader_entry` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/README.md` | 0 / 9 / 0 |
| `docs/teacher_feedback_unified_v1/analyze_round0_gap_change.py` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/analyze_round0_gap_change.py` | 1 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/continuous_math_checks.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/continuous_math_checks.json` | 3 / 0 / 0 |
| `docs/teacher_feedback_unified_v1/continuous_math_checks.py` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/continuous_math_checks.py` | 2 / 1 / 0 |
| `docs/teacher_feedback_unified_v1/self_improvement_train_diagnostic.json` | `archive` / `historical_supporting_data_or_tool` | `docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/self_improvement_train_diagnostic.json` | 1 / 0 / 0 |

## 执行方案

配置已经包含逐源 SHA 与目的路径；迁移工具默认只预检，默认不会改引用或文件。源内容在预检后有变更、目标已存在、路径越界或遇到 reparse point 时停止该批次。

```powershell
& 'D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_cleanup/migrate_documents.ps1' -Batch historical_docs
# 主审明确协调范围与引用作者后，才对同一选定批次加 -Apply。
# 也可使用 -SourcePath 'docs/具体文件.md' 仅预检/执行选定源。
```

工具只用原生 PowerShell 的 `Move-Item -LiteralPath` 搬移文件；仅对 SHA 完全相同且归档原件再次验证的重复副本使用无递归 `Remove-Item -LiteralPath`。移动和去重均在写操作之前核对解析后的绝对路径属于本工作树 docs，目标位于指定 audit 子目录或既有只读 archive，并留下逐文件 journal。

本次发现 338 条迁移后需要检查的链接/路径记录。`link_updates` 给出原行、迁移后源/目标、建议新 target；basename 提及不自动改，immutable archive 与证据快照中的历史路径不自动改。移动后仅更新主审明确指定的引用；清单构建阶段不运行测试或迁移。
