# 自主晶体编辑实验结果（2026-09-08）

raw 与 refined 两端均已完成。按照本轮“至少一个端点 promising”的标准，raw 达到阶段目标；refined 的 SUN/MSUN 下降。结论限定于预先冻结的 1200 条回溯面板。

- [完整方法、结果与限制](METHOD_AND_EVIDENCE.md)
- [精确汇总与来源哈希](final_results/RESULTS.json)；[7200 条逐请求指标](final_results/per_request_metrics.csv)
- [选定策略、检查点与推理参数](SELECTED_POLICY.json)
- [训练曲线](training_curves.png)；[开发集独立 R 观测](development_reliability.png)。同名 SVG 可用于排版。

主评测模型在查看主面板结果之前固定为第二阶段 step 800。实际梯度训练累计 38.95 分钟，可训练参数约 2248 万。主面板的实际操作以 G 的完整晶格及坐标修复为主；局部编辑能力、S 阶段独立贡献与多 seed 稳健性尚未得到证明。

目录内的运行清单保留原始源码身份、数据路径、资源时窗和参数；模型权重仍保存在 HPC 的原始检查点目录，具体路径与回执 SHA-256 见 `SELECTED_POLICY.json`。结果包包含小型证据、配置与分析脚本，不包含模型权重。

本地 CSV SHA-256：`10343ef6a12b32a4d59e8e849f2a59619db354d222a945937a7f6313465b84a1`。
本地汇总 JSON SHA-256：`793bd76d150ef1390a6425398cc4e618e0225dab440386008ce2819ee933c7bc`。
