# LLM＋DLM 晶体生成研究

C³FD 提供化学动作支持，Typed Llama 生成组成、软计划与物种程序，周期 DLM 生成和修订晶体几何；固定 model494 refiner 单独报告。

最近完成的 **V3 H-P33 G** 评测40066：raw Strict/Meta SUN为 **9/61**，固定tau800后为 **13/116**（全部256请求），未达到目标且未改善旧K4/K8。真实输出链检查40069全256无错配，坐标去噪验证基本没有优于零预测；[实际失败诊断](docs/v3_scientific_audit_20260906/V3_FAILURE_CROSS_STRATEGY_PHYSICS.md)记录逐结构证据。40071正在测试同40064权重的T construction-only及匹配V2 constructor→tau800；后续重点分析K4/K8哪些稳定、哪些失败，若V2/V3最终无效则采用K4或K8。具体状态以[当前记录](docs/CURRENT_STATE.md)为准。

## 项目入口

- [当前状态、结果与后续工作](docs/CURRENT_STATE.md)
- [当前架构及已证实的边界](docs/CURRENT_ARCHITECTURE.md)
- [统一审计目录：论文对照、失败与成功原因、证据、交叉审查](docs/v3_scientific_audit_20260906/README.md)
- [V2 执行说明](docs/periodic_self_repair_v1/V2_EXECUTION_PACKAGE_20260906.md)
- [训练和评测工作流](WORKFLOW.md)
- [独立 1000 配对补评操作](operations/supplement_1000/README.md)

## 结果阅读

同时报告 raw 和固定 tau800 refined，以完整请求为分母；1000 子集使用事先冻结的同一索引，并另列所有 1200 请求。Strict/Meta SUN 的目标为同一端至少 10%/50%。N/U、hull 覆盖、终态验证率及不确定性分别列出。

旧 K8 独立 1000 更正后 raw SUN 为 **3.9%/24.4%**，refined 为 **6.5%/48.4%**。这次更正来自官方 hull 缓存覆盖修复，未重采样或重弛豫；不能算作模型能力提升。完整来源见 [原始评测汇总](docs/v3_scientific_audit_20260906/evidence/EVALUATION_SUMMARIES_20260906.json)。

## 版本与历史

V2 执行 SHA：`2e904c260bafb6750c9a5dbb0d920c4ff8c3a868`。执行归档保持冻结，审计与文档清理在当前工作树进行。

过时的根目录长篇入口已保存在 [历史入口快照](docs/v3_scientific_audit_20260906/historical/root_entrypoints/MIGRATION_MANIFEST.json)，逐文件保留 SHA256。历史计划中的“active/current/approved”描述的是当时状态；当前状态统一由上述入口维护。
