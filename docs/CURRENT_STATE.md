# 当前项目状态

更新日期：2026-09-06。此文件统一维护当前状态；历史材料中的旧状态不覆盖本页。

## 当前阶段

**全面科学审计进行中，V3 具体规格进入第二轮独立复核，尚未实施或训练。** V2 训练、完整评测和冻结权重诊断均已完成。审计覆盖整链路、近期变更、全部可追溯失败与成功证据，并扩大晶体 diffusion、DLM 与 GNN 融合研究。最终使用 SUN 判断改动是否有效。

统一目录：[全链路审计](v3_scientific_audit_20260906/README.md)。

## 已完成的正式对照

V2 训练作业 **39993** 已完成，Slurm `COMPLETED 0:0`，用时 01:11:28；两 epoch、4524 更新、108544 有效 states，source/view 覆盖 min=max=2。真实零增量 delta=0，前 64 更新各模块梯度非零。正式采样只使用这个完整终点。

固定 256 评测作业 **39998** 已完整成功，raw/refined SUN、对照与 construction→repair 诊断均完成。`V2_EVAL_JOB` 和提交记录已登记，不重复提交 249/250。两端未达到同端 26/128 的补评门槛，本版本不追加 1000。

执行代码固定为 `2e904c260bafb6750c9a5dbb0d920c4ff8c3a868`。输入为原始 LLaDA 和完整 MP20 27136/9047，冻结 C³FD/Llama/pointer；训练直接预测条件 24558 条，显式 teacher-soft fallback 2578 条。验证为 8147 预测与 900 fallback。这不是全量预测条件，也不代表预测软标签与目标几何严格一致。

正式外部 `LAUNCH_MANIFEST.json` 已部署在远端 V2 实验目录；执行 commit 内的同名早期文档不是启动依据。训练、manifest 与唯一提交 claim 保持不变，不重复提交 249。

## 已完成的最近 SUN

以下均为 Strict / Meta SUN；不同 cohort 或样本数之间不作直接配对结论。

| 方法 / cohort | raw | 固定 tau800 refined |
|---|---:|---:|
| 参考 / 开发 256 | 6 / 55 | 19 / 121 |
| 旧 K4 / 同开发 256 | 7 / 57 | 19 / 126 |
| 旧 K8 / 同开发 256 | 11 / 66 | 17 / 123 |
| raw-v1 / 同开发 256 | 3 / 45 | 16 / 113 |
| V2 / 同开发 256 | 8 / 50 | 16 / 113 |
| 旧 K8 / 独立条件 1000 | 39 / 244 | 65 / 484 |
| 旧 K8 / 对应全部 1200 请求 | 47 / 285 | 79 / 570 |

旧 K8 独立评测此前误用了开发小 cohort 的 hull 缓存。官方覆盖已补齐，CPU 作业 39990 重评成功，生成、弛豫、能量和选择索引均未改变。更正后 1200 请求中 hull 已知 1172、官方明确 unresolved 20、不可重建 8。旧覆盖不足的 0/9、1/17 不再作为完整结果。补统计不是模型提升。

原始评测文件与 hash：[评测证据](v3_scientific_audit_20260906/evidence/EVALUATION_SUMMARIES_20260906.json)。历史小幅差异、共同 verified 子集及外部论文分数须按实际口径解释，不能混用来声称提升。

V2 构造终点原生 SUN 为 12/44，一次 full-cell repair 后为 8/50；Strict −4、Meta +6，是这组配对请求的权衡，不能称修订普遍改善。V2 refined 与 raw-v1 同计数，未实现目标提点。[V2两端](v3_scientific_audit_20260906/evidence/V2_EVALUATION_SUMMARIES_39998.json)、[同次构造与修订](v3_scientific_audit_20260906/evidence/V2_CONSTRUCTION_AND_REPAIR_39998.json)。

冻结诊断 **40004** 已 `COMPLETED 0:0`，4A800/16CPU、00:02:24。100来源/200状态、5变体，0训练、新晶体与MLIP；原目标风险复算最大差5.25e−7。旧几何遮蔽主要损害repair风险，真实噪声相对unknown没有清晰平均收益，真实soft降低部分风险但不是可用oracle或SUN证据。真实prefix logits的float64 exp未出现溢出。[完整证据](v3_scientific_audit_20260906/evidence/V2_CONDITIONING_PROBE_40004.json)。

## V3 审查中的具体对象

[H-P33 规格](v3_scientific_audit_20260906/V3_H_P33_SPECIFICATION.md) 复用 V2 final 与共享 LLaDA，保留 T construction、以 G 全几何 v/torus 去噪替代旧 repair CE；拟再训练两epoch。主采样为32个Euler区间加一次明确末端读出，float CIF为主终点，同样样本Q raw为精度诊断。P在G中仅作条件/rank，不新增rank噪声时钟；这是混合表示模型。当前正在核验原CIF全量身份、独立第二轮数学/物理审稿，之后仍须实际计算图与DDP验收。没有SUN收益保证，也未提交新训练。

## 已授权的接续

1. V2 完整终点的固定256 raw/refined及construction→repair评测已完成，保留为对照，不重复提交。
2. 任一端同时达到 Strict SUN ≥10% 和 Meta SUN ≥50%（256 中至少 26/128），追加同索引 1000 raw/refined，并另报所有 1200 请求。
3. 当前先完成全面大审计、竞争解释和攻击性复核。证据支持的小改动可进行受控尝试；必要 V3 在反复审查、实现验证和登记后训练 1–2 epoch，再采样算 SUN。
4. 后续新实验按可用资源使用 4–6 A800；本项目总上限仍为 6 A800、24 CPU、两个作业。旧新 K4/K8、自生成能量 teacher 与旧连续联合路线保持暂停。
5. 根目录入口与142份历史/组件文件迁移、6份完全重复文档去重已完成；唯一历史证据与hash/迁移映射保留。当前收尾新稿引用、状态入口和提交，不再把历史计划当成运行状态。

自动任务 `llm-dlm-sun-24h` 每十分钟汇报当前阶段或 SUN 结果。等待阶段只做轻量检查；最终用户主汇报只列 raw/refined Strict、Meta SUN。详细研究与物理诊断保存在审计目录。
