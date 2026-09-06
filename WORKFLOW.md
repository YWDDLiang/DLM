# 当前工作流

1. 从 [当前状态](docs/CURRENT_STATE.md) 核对在运行的作业、版本和已完成结果，避免重复提交。
2. 模型执行归档保持冻结；新改动先在独立版本完成实现、必要验证和实验登记，再按可用资源接续。
3. 只使用完整终点进行正式采样。V2 为两 epoch、4524 更新，训练成功后按既定 250 入口评固定 256 的 raw/refined，并诊断 construction 与 repair 的净作用。
4. 同一端 Strict SUN ≥10% 且 Meta SUN ≥50% 后，按 [独立补评流程](operations/supplement_1000/README.md) 追加配对 1000，并报告全部 1200 请求。使用与该 cohort 匹配的官方 hull 缓存。
5. [统一审计目录](docs/v3_scientific_audit_20260906/README.md) 保存文献、全链路推导、历史实验、反驳、实际修复和验证结果；过时文档不作为运行指令。
6. 当前自动接续任务每十分钟报告阶段与结果；等待时只做轻量检查。研究和必要诊断与健康训练并行。

资源上限为本项目 6 张 A800、24 CPU、同时两个作业。远端操作继续通过已经存在的 starteam5090 → tmux ssha800:1.0 通道，主审统一负责调度。Materials Project 凭证复用已配置的私有提供文件，不进入源码或结果。

历史 H1-A2/R03 通用默认配方保存在 [原工作流快照](docs/v3_scientific_audit_20260906/historical/root_entrypoints/WORKFLOW.md)，仅用于复核旧实验。
