# 当前执行清单：原任务完成后运行 V2

2026-09-06 用户最新授权覆盖此前暂停交接及条件式新版 K4/K8 安排。本轮新版仅使用原始 LLaDA 与原始 MP20；不提交 243、245、246，不恢复连续联合 diffusion。当前旧 K4/K8 流程独立完成，其路径和 teacher 不进入 V2。

- [x] 完整核实旧 C³FD→Typed Llama→species pointer→编译程序→DLM 顺序控制。
- [x] 修订 loss、状态、噪声、几何 attention 和论文表述；见 [正式设计](LOSS_GEOMETRY_AND_V2_PROPOSAL_20260906.md)。
- [x] 原 K8 train39938、native39945、tau80039948 完成，两个 matched comparison 完成。
- [x] 记录混合结果：native SUN11/66；tau SUN17/123，低于 K4 的19/126；共同 verified A 未改善。
- [x] 按预登记最终 K8 step7977 冻结方法；独立1200 Planner39949 已提交，种子27。
- [x] 39949 完成；已提交233作业39951，4卡/16CPU，K8_MAIN_EVALUATION_JOB已写39951。
- [ ] 原 raw-v1 39942 自然完成6784updates，不恢复错误39934/39937。
- [ ] raw-v1 顺序运行241 native、242 tau800；分别写 RAW_BASE_NATIVE_EVAL_JOB 和 RAW_BASE_TAU800_EVAL_JOB。
- [ ] 完成原独立主评测：源序 parser-only first1000、1200总分母、raw/tau同索引；失败不补样本。

V2 本地准备：

- [x] 原始来源的 frozen Planner 预测与明确 fallback；保留全27136/9047来源、原clean answer/hash、逐源条件证据。
- [x] 精确合法 prefix CE、dense typed CE、对象归一化与逆mask包含概率，T=.7、无平滑。
- [x] log-SPD体积/形状与Cartesian扰动；量化后支持核验、重试与fallback日志。
- [x] 真正 head-specific/layer-shared 周期attention，包含六晶格槽↔site交互及known/noise gate。
- [x] 原始LLaDA初始化、新token行、LoRA及V2保存/加载；拒绝中间checkpoint作为推理政策。
- [x] 六卡global24/4524updates、完整view覆盖、padding校正、正确validation计数、梯度与state audit。
- [x] 249训练、250 native/tau及同trace construction→repair诊断入口。
- [x] 冻结原两worker采样批次后分派六worker，保持配对membership。
- [x] 已完成102项不同的相关CPU检查及两个Slurm入口本地/远端bash语法检查；真实六卡DDP尚未运行。
- [x] 实现固定为2e904c260bafb6750c9a5dbb0d920c4ff8c3a868，两个旧K8端点已实际运行完整回归分析；manifest已标ready。
- [ ] 所有八项前置阶段正式完成、项目额度六卡全部释放后，queue helper一次性提交249。
- [ ] 249完整成功后同版本提交250；native/tau、原参考/raw-v1/repair净作用分开报告。
- [ ] 任一效果低于预期，完成支持/状态/梯度/物理/尾部/N-U/终态验证分析，再考虑必要修改。

调度最多6张A800、24CPU、2个项目jobs。新版按 [V2_LAUNCH_MANIFEST.json](V2_LAUNCH_MANIFEST.json) 校验阶段完整性与实现版本，不仅看squeue是否为空。正常运行只查完成/失败/正式checkpoint，不反复读取普通loss。旧18:49/19:19节点不撤销新增排队授权，也不擅自延长已运行作业walltime。

V2当前保存optimizer/scheduler状态，但没有断点续训入口；不得把中间checkpoint直接投入评测，或声称可自动从任意中断位置恢复。
