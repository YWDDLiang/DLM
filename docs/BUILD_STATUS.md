# 个人repo构建状态

## Proposal/Realization双候选

- 冻结H1-A2保持只读fallback；
- 历史难度分析使用纯Python实现并去除evaluator replay；
- Candidate A/B在独立branch开发，默认均关闭；
- public headline继续保持`105/1000 Strict、488/1000 Meta`，未被screen静默替换。

2026-08-26远端执行：

- Candidate A：`34700`完成control/grounding训练，`34714/34721`完成四次
  fixed-256 generation与统一评价；结论是有用但有边界的realization/Strict改进，
  Meta存在trade-off，保留在Candidate-A个人branch；
- Candidate B V1：`34697/34704`为未归一化负结果，已冻结；
- Candidate B normalized V2：`34734`完成两个Planner seed，`34739`完成保留
  sparse Planner ordinal的真实下游，`34744`完成四cell Direct/N/U/CHGNet，随后
  fresh official MP GGA/GGA+U与终态评价完成；
- `34693/34694`为2秒启动路径失败，`34695`为模型启动前的Bash兼容失败，
  `34696`因慢donor builder主动取消，`34698`因sidecar显式plan_state缺失而
  在训练前取消；`34737`因循环替换missing Planner ordinal而主动取消，`34743`
  为2秒环境变量错误；这些输出均不进入科学结果。

当前决定：

- Candidate A保留为“counterfactual Plan grounding改善realization/Strict”的候选贡献，
  但不把Meta trade-off隐藏，也暂不替换public headline；
- normalized Candidate B不保留为正方法：seed17的Strict/Meta为正，seed18均反转；
  pooled 512 attempts中Direct joint `+1.17pp`，但Strict `-0.20pp`、Meta
  `-0.98pp`、novel rate `-1.20pp`，未通过预设screen；
- post-hoc训练审计发现V2在`batch_size=1`下逐batch归一化权重，导致difficulty
  sample weight在每个microbatch内完全约掉。因此V2只能解释为“加入buffer rows”的
  pilot，不能作为正确difficulty weighting的负证据；
- strong20 V3使用独立`difficulty_sampling_weight`做replacement weighted sampling，
  self-improvement实际抽样19.875%/20.063%，control/candidate均为800 matched updates；
  `34766/34771/34776`及fresh official MP终态均已完成；
- V3 pooled body `+2`、Direct joint `+8`、novel `+6`、N∩U `+5`、Strict
  `+3/512 = +0.59pp`；Meta all-attempt `-3/512 = -0.59pp`，但Meta hull-known
  `-2.50pp`，因此仅失败冻结的known-Meta gate；归类为promising scoped signal，
  不是完整通过；
- 完整证据见
  [`PLANNER_DIFFICULTY_V2_FINAL.md`](../results/remote_screens/PLANNER_DIFFICULTY_V2_FINAL.md)
  和[`PLANNER_DIFFICULTY_V3_STRONG20_FINAL.md`](../results/remote_screens/PLANNER_DIFFICULTY_V3_STRONG20_FINAL.md)
  及同名JSON/CSV。H1-A2继续作为fallback，public headline不变。

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
