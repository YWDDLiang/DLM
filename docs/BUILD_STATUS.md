# 个人repo构建状态

## Proposal/Realization双候选

- 冻结H1-A2保持只读fallback；
- 历史难度分析使用纯Python实现并去除evaluator replay；
- Candidate A/B在独立branch开发，默认均关闭；
- 远端训练和fixed-256 screen尚未产生结果。

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
