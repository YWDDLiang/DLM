# 复现入口

当前 V2 的代码版本、输入、训练终点和评测接续见 [V2 执行包](docs/periodic_self_repair_v1/V2_EXECUTION_PACKAGE_20260906.md) 与 [当前状态](docs/CURRENT_STATE.md)。远端每个 run 保留代码归档、manifest、输入 hash、作业身份及完成标记。复现必须匹配这些实际资产，而不能只匹配名义 epoch、seed 或 tau800。

[审计原始汇总](docs/v3_scientific_audit_20260906/evidence/EVALUATION_SUMMARIES_20260906.json) 保留最新 K8/raw-v1 评测来源和原文件 SHA256。[运行快照](docs/v3_scientific_audit_20260906/EVIDENCE_RUNTIME_SNAPSHOT.json) 记录 V2 训练已经开始的证据，不代替最终 checkpoint 验收。

历史 H1-A2/R03 资产与旧环境说明见 [原复现指南](docs/v3_scientific_audit_20260906/historical/root_entrypoints/REPRODUCTION.md)。其中的占位资产、旧默认选择规则和已失效状态不得当作当前设置。
