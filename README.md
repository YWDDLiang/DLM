# LLM＋DLM 晶体生成研究

C³FD 提供化学动作支持，Typed Llama 生成组成、软计划与物种程序，周期 DLM 生成和修订晶体几何；固定 model494 refiner 单独报告。

最近完整评测40074：K4短接触软约束 raw Strict/Meta SUN为 **6/56**、tau800为 **20/122**；K8同规则为 **11/68→15/123**，均256请求，均未达同端10%/50%。[完整证据](docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/CONTACT_EVAL_FINAL_40074.json)与[K4/K8实际稳定和失败结构](docs/v3_scientific_audit_20260906/K4_K8_STABILITY_AND_CONTACT_DIAGNOSIS.md)保留。当前**40076**在评估固定K4/K8等权同态概率混合，真实8前缀/16次模型计算已验收通过，尚无新SUN；[方法与边界](docs/v3_scientific_audit_20260906/K4_K8_EQUAL_MIXTURE_BOUNDED_REVIEW.md)。具体状态以[当前记录](docs/CURRENT_STATE.md)为准。

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
