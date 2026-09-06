# V2 实现验收记录

实现版本：2e904c260bafb6750c9a5dbb0d920c4ff8c3a868。2026-09-06。后续仅更新说明、结果报告和启动manifest；实际训练/推理消费该固定代码版本。

已完成102项不同的相关CPU测试：88项集成回归、13项已有结果回归诊断测试、1项新增完整V2模块保存/重建精确logit往返检查。新增往返检查所在文件两项测试均通过；对preprocessor新增的事后软注释相符率记录也重新跑过10项测试。249/250在本机Git Bash和A800登录环境分别通过bash -n；git diff --check通过。

核验包括：

- 旧程序compiler/训练及推理位置、exact 7+4N与canonical alias保持一致。
- compact/original token-ID支持处理在FP32/BF16下等价，梯度与alias语义一致。
- prefix累计可达性、repair old入场、密集风险/空mask/padding分母正确，严格JSON可序列化。
- 全部来源保留、缺元数据明确fallback、Planner/pointer forward不读teacher结构字段，按实际sampled softIDs生成程序。
- 多头真实attention中的active-logit梯度、周期特征、缩放可辨识性、未知几何、N1、重计算一致性。
- V2保存legacy geometry、新head、数值adapter、state conditioner及新token行；重建logits精确一致。
- global24/4524updates及逐source/view两epoch覆盖，validation短尾batch统计正确。
- 中间checkpoint、未完成final、错误policy路径均不能成为V2采样模型。
- 原两worker批次先冻结再分派六worker，原有默认采样行为保持。
- 队列即使无活动job，也不能绕过尚未提交或尚未完成的后续阶段；错误分母及未ready版本被拒绝。
- 同一attempt的construction诊断保留生成失败，不再采样，不读取能量作选择。

还读取了实际original LLaDA配置和forward：32heads、32layers、block_group_size1；head维未被reshape压成1，bias显式传入每层及whole-layer checkpoint调用。该核验针对真正部署源代码，CPU注意力测试则使用可检查的小模型。

六A800、真实8B forward/backward及长期优化尚未执行，将在所有前置任务结束后的正式allocation内验证。Windows本机Gloo不能代替该验证；没有将未运行的DDP报告为通过。训练器保存optimizer/scheduler状态，但当前没有resume入口，只认完整正式final。

原K8两个端点的实际已完成数据已由同一版本CPU诊断脚本运行成功，配对、协议、计数与能量分解均通过；见 [结果及原因分析](../v3_scientific_audit_20260906/historical/docs/periodic_self_repair_v1/K8_RESULTS_AND_REGRESSION_ANALYSIS_20260906.md)。这验证诊断入口适配现有产物，不意味着证明V2的物理收益。
