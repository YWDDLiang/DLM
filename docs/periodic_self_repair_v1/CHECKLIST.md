# V2 已完成执行清单与审计交接

更新日期：2026-09-06。**raw-v1、V2 训练与评测均已完成；本清单不再是待提交队列，不重复提交 240/241/242 或 249/250。** 当前阶段、授权与后续工作以 [CURRENT_STATE](../CURRENT_STATE.md) 为准。本文记录这些版本的完成事实；旧阶段的暂停或接续安排不覆盖当前科学审计及经过审查、验证和登记后的新 V3 工作。

## 已完成的前置版本

- [x] raw-v1 训练 39942 完成：每 source 两个 view、两 epoch、global batch 16、6784 updates；未恢复早期错误作业。
- [x] raw-v1 native 39975、tau800 39984 及对应比较完成，固定 256 的 Strict/Meta SUN 分别为 3/45、16/113。
- [x] 旧 K8 train39938、native39945、tau80039948、独立 Planner39949 与主评测39951 完成；独立评测 hull 覆盖修正由 39990 完成。当前更正结果见 [评测证据](../v3_scientific_audit_20260906/evidence/EVALUATION_SUMMARIES_20260906.json) 与 CURRENT_STATE，旧覆盖不足的数值不再作为完整结果。
- [x] 原前置阶段已验收并完成 V2 一次性提交；其 claim 和 job pointers 保留，不重置后重提。

## V2 实现与正式运行

- [x] 原始 LLaDA 与完整 MP20 27136/9047 sources；冻结上游预测及显式 fallback，保留原 clean answer/hash 和逐源证据。
- [x] exact prefix/dense CE、log-SPD/Cartesian corruption、量化支持验收和周期 geometry attention；设计见 [正式设计](LOSS_GEOMETRY_AND_V2_PROPOSAL_20260906.md)，实现检查见 [验收记录](V2_IMPLEMENTATION_REVIEW_20260906.md)。
- [x] 执行代码固定为 `2e904c260bafb6750c9a5dbb0d920c4ff8c3a868`；原两 worker 的采样批次 membership 在六 worker 执行时保留。
- [x] 六卡训练 39993 完成，`COMPLETED 0:0`：两 epoch、global batch 24、4524 updates、108544 有效 states，source/view 覆盖 min=max=2；初始输出 delta=0，前 64 更新完成实际模块梯度检查。[训练证据](../v3_scientific_audit_20260906/evidence/V2_FINAL_TRAIN_AND_METADATA.json)。
- [x] 评测 39998 的 native、tau800、参考/raw-v1 比较及同 trace construction→repair 诊断全部成功。正式推理使用完整 step-4524，未选择中间 checkpoint。
- [x] 固定 256 的 construction Strict/Meta SUN 为 12/44；full-cell repair 后 native 为 8/50；固定 tau800 为 16/113。[两端证据](../v3_scientific_audit_20260906/evidence/V2_EVALUATION_SUMMARIES_39998.json)、[构造与修订证据](../v3_scientific_audit_20260906/evidence/V2_CONSTRUCTION_AND_REPAIR_39998.json)。Strict −4、Meta +6 是配对权衡，不是统一改善。
- [x] 两个正式端点均未达到同端 Strict/Meta SUN 至少 26/128 的门槛，本版本不触发追加 1000。
- [x] 只读条件/几何 forward probe 40004 完成，`COMPLETED 0:0`、用时 2分24秒；四张 A800，未训练权重、未生成新样本、未调用 MLIP。[probe 证据](../v3_scientific_audit_20260906/evidence/V2_CONDITIONING_PROBE_40004.json)。

## 保留的入口与后续边界

[V2 执行包](V2_EXECUTION_PACKAGE_20260906.md) 保留原始布局、参数、只读检查和 249/250 入口；[V2_LAUNCH_MANIFEST.json](V2_LAUNCH_MANIFEST.json) 是该次运行的启动合同，其运行前状态字段不是当前待执行指令。已有训练、评测与提交记录不因文档刷新而改变。

当前工作是 [统一科学审计](../v3_scientific_audit_20260906/README.md) 和新规格的交叉审查。具体 V3 是否进入实现/训练按当前授权与新登记决定；不自动复用本清单的旧作业提交安排，也不因历史 raw-v1 暂停文字停止现有审计。项目资源上限仍为六张 A800、24 CPU、两个作业。

V2 虽保存 optimizer/scheduler 状态，但该版本没有 resume 入口；保存状态不等于可自动从任意中断位置续训。刷新前的三份完整字节与 SHA256 已保存在 [历史快照映射](../v3_scientific_audit_20260906/historical/current_entrypoints_before_status_refresh/manifest_aa5d826c7e39cd442817d49c7605444332b95a5fbe5f45c28b80f7fead43da78.json)。
