# 个人repo构建状态

## Proposal/Realization双候选

- 冻结H1-A2保持只读fallback；
- 历史难度分析使用纯Python实现并去除evaluator replay；
- Candidate A/B在独立branch开发，默认均关闭；
- 远端训练和fixed-256 screen尚未产生结果。

2026-08-26远端执行：

- Candidate A：Slurm `34700`，4×A800，control/grounding同job；
- Candidate B：Slurm `34697`，4×A800，two-seed control/candidate同job；
- 两项均固定每GPU 4 CPU，标准H1-A2 schedule，不使用MP网络；
- `34693/34694`为2秒启动路径失败，`34695`为模型启动前的Bash兼容失败，
  `34696`因慢donor builder主动取消，`34698`因sidecar显式plan_state缺失而
  在训练前取消；均不产生科学结果。

当前决定：

- Candidate B完成`34697`训练和`34704` Plan-256后停止；两个seed均少1个
  parsed Plan，projected Strict/Meta mix均下降，不进入downstream；
- Candidate A继续`34700`，step500 factual val CE为1.6240，对照为1.9153。

## 已完成

- 独立Git repo与`codex/personal-research`分支；
- 与公开repo物理隔离的完整源码副本；
- JSON个人配置入口；
- H1-A2、R03 D1和R03 D2内部组成结果；
- A800资产迁移台账；
- 训练、H1-A2推理与256×4快速复现Slurm骨架；
- confirmed/default/unrecorded三类seed记录；
- 相对路径和缺失资产处理逻辑。

## 等待A800

- Conda环境与依赖版本；
- 全部checkpoint与完整MP-20数据；
- H1-A2/R03 Plan文件和逐ordinal seed ledger；
- B0与model_494训练seed证据；
- 最终评价环境与相对路径适配器；
- 在真实A800上完成端到端smoke与256×4验证。

所有A800绝对源路径只填写在`ASSET_TRANSFER_LEDGER.md`，不会同步到公开repo。
