# 实验入口

当前实验及结果只在 [当前状态](docs/CURRENT_STATE.md) 维护。全链路实验回顾、失败分类和成功归因集中在 [审计目录](docs/v3_scientific_audit_20260906/README.md)。

V2 训练 39993 已完成，评测 39998 已提交；使用原始 LLaDA、完整 MP20、冻结预测 Planner 条件及显式 fallback，两 epoch、最终 4524 更新。具体冻结执行约定见 [V2 执行包](docs/periodic_self_repair_v1/V2_EXECUTION_PACKAGE_20260906.md)。不得以历史 240 raw-v1 入口重复启动本版本。

V3 尚未定案。用户授权有论据的小改动试验，以及通过反复攻击性审计后的 V3 实现、1–2 epoch 训练、raw/refined 采样和 SUN 评估；每个实际新实验须先登记版本、假设、样本、预算和成败标准。
