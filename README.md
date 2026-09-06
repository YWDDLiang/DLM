# LLM＋DLM 晶体生成研究

C³FD 提供化学动作支持，Typed Llama 生成组成、软计划与物种程序，周期 DLM 生成和修订晶体几何；固定 model494 refiner 单独报告。

最新完整阶段对照40083：K4构造终点 raw Strict/Meta SUN为 **5/48**，tau800为 **15/126**；K8为 **5/54→16/119**，均256且未达同端10%/50%。当前40088正在评估统一合作修复终点；K4 raw已完成8/48，tau800与K8尚在执行。[构造阶段完整结果](docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/CONSTRUCTION_EVAL_FINAL_40083.json)、[所有当前结果与实际状态](docs/CURRENT_STATE.md)。

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
