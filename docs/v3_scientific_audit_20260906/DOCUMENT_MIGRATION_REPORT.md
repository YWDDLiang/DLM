# 文档迁移与链接核验结果

已按主审协调范围处理 148 个源：142 个逐字节移动、6 个已复核完全重复的 docs 副本复用既有 immutable archive。原 archive 未修改。

随后只在获准的 18 份文档内更新 115 处链接/文档路径，所有行数保持不变；科学正文、审稿记录中的被审版本 SHA 没有改写。原 SHA、移动后 SHA 和链接改写后 SHA 均在 [逐文件映射](DOCUMENT_MIGRATION_MAP.json)。

| 核验 | 结果 |
|---|---|
| 活跃文件/引用 | 27 / 189 |
| 活跃引用状态 | `{"valid": 152, "pending_parent_replacement": 37}` |
| 已迁移历史文档引用 | 234；`{"valid": 229, "preexisting_missing_target": 5}` |
| 本次迁移意外引用问题 | 0 |

当前状态/架构、root README/EXPERIMENT、V2 运行合同、在写 audit 正文和 d3po 源码 docstring 均按分工保留。需要主审替换的精确行和目标见 [替换清单](PARENT_REFERENCE_REPLACEMENTS.json)；不把这些已明确留给主审的项混报为已全部修完。自动化 prompt 由主审依映射处理，本子任务未访问或修改。

## 主审剩余替换范围

| 文件 | 待替换引用数 |
|---|---|
| `docs/periodic_self_repair_v1/LOSS_GEOMETRY_AND_V2_PROPOSAL_20260906.md` | 1 |
| `docs/periodic_self_repair_v1/V2_EXECUTION_PACKAGE_20260906.md` | 1 |
| `docs/periodic_self_repair_v1/V2_IMPLEMENTATION_REVIEW_20260906.md` | 1 |
| `docs/v3_scientific_audit_20260906/C3FD_LLM_UPSTREAM_AUDIT.md` | 2 |
| `docs/v3_scientific_audit_20260906/CLAIM_AUDIT.md` | 5 |
| `docs/v3_scientific_audit_20260906/CRYSTAL_DIFFUSION_MECHANISMS.md` | 4 |
| `docs/v3_scientific_audit_20260906/DLM_PROBABILITY_AND_KERNEL_AUDIT.md` | 1 |
| `docs/v3_scientific_audit_20260906/PHYSICS_PIPELINE_AND_STORY_AUDIT.md` | 21 |
| `docs/v3_scientific_audit_20260906/ROUND2_PHYSICS_ON_DLM_AND_CLAIMS.md` | 1 |
| `src/crystal_dlm/d3po.py` | 1 |

原有缺失资源仍单列 `missing_not_in_migration_map` 或 `preexisting_missing_target`，没有擅自恢复旧数据/模型。全部逐引用结果、保留合同哈希与移动 journal 见 [完整报告](DOCUMENT_MIGRATION_REPORT.json)。

最终人工复核排除一处行内公式伪链接，LOSS_REVIEW_AGENT.md 已还原为迁移前完整原字节。链接工具现屏蔽行内/围栏代码及 LaTeX/math 区间，并仅改既有文件或迁移映射内的目标；未知目标原样保留。19份候选改写文档均完成反向哈希还原证明，最终18份保留115处实际引用改写。见 [反向哈希与解析检查](DOCUMENT_MIGRATION_VERIFICATION.json)。

相关测试5通过、1失败；失败是旧测试仍要求当前README保留历史G2术语/数字，由主审按新入口职责重构。paper两文件读取与V2 queue检查均通过，本子任务未改README或测试。
