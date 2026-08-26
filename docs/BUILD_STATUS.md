# 个人repo构建状态

## Proposal/Realization双候选

- 冻结H1-A2保持只读fallback；
- 历史难度分析使用纯Python实现并去除evaluator replay；
- Candidate A/B在独立branch开发，默认均关闭；
- 两个候选的远端screen均已完成并判负，公开H1-A2结果保持不变。

2026-08-26 Candidate B：

- `34697`完成Planner训练，`34704`完成two-seed Plan-256；
- 两个seed均少1个parsed Plan，projected Strict/Meta chemistry mix均下降；
- 按预注册选择规则停止，不进入DLM/refiner downstream，不重跑。

2026-08-26 Candidate A：

- `34700`完成control/counterfactual-grounding训练；final factual CE为
  `1.289004`与`1.288558`，candidate true-vs-counterfactual mean margin为`+0.7592`；
- `34711` fixed-256 screen两臂均为`256 parsed / 253 body / 253 refined`；
- `34714`完成4次独立fixed-256、逐sample_idx配对body/refine；
- `34721`完成8-cell Direct、N/U、CHGNet，随后在登录节点完成fresh official
  `GGA_GGA+U` hull；
- pooled Strict known差`+0.171 pp`，但Meta known差`-1.360 pp`，低于
  `-1.0 pp`非劣门，因此Candidate A最终判负；
- 完整证据见
  [`GROUNDING_FINAL_REPEAT4.md`](../results/remote_screens/GROUNDING_FINAL_REPEAT4.md)。

工程谱系：`34700`训练后导入失败、`34710`环境预检失败、`34719`因冻结V3只接受
1000/1200分母而在科学评价前失败；它们均被最小恢复，成功阶段未重跑。fixed-256
adapter只放宽active denominator，不改变Direct、N/U、CHGNet、hull或S.U.N.阈值。

2026-08-27 DLM训练时长与固定requested-1000复核：

- raw-256同Plan扫描覆盖约`0.295/0.590/1.000 epoch`。相对各自matched control，
  counterfactual-grounding在step500的Strict/Meta差为`+3/-1`，step1000为`+3/+4`，
  step1696为`-5/-2`（分母均为256）；
- step1000的body、Direct、novelty、Strict/Meta方向和stable→S.U.N. retention均通过
  downstream门，但candidate validation CE比control高`0.07209`，且所有McNemar检验
  均不显著，因此预注册筛选没有合格checkpoint；这只支持非单调的中间训练窗口信号；
- 独立固定requested-1000复核不做survivor过滤。control/candidate分别得到
  `994/990` body、`877/874` Direct joint、`89/86` Strict S.U.N.和`487/467`
  Meta S.U.N.；Strict差`-3/1000`、Meta差`-20/1000`，完整贡献门失败；
- stable→S.U.N. retention仅小幅变化（Strict `81.65%→81.13%`，Meta
  `82.68%→82.07%`），主要退化来自stable本身（Strict `109→106`，Meta
  `589→569`），而不是novelty或retention崩塌；
- 完整证据见
  [`GROUNDING_CHECKPOINT_SWEEP_FINAL.md`](../results/remote_screens/GROUNDING_CHECKPOINT_SWEEP_FINAL.md)
  和
  [`GROUNDING_FIXED1000_FINAL.md`](../results/remote_screens/GROUNDING_FIXED1000_FINAL.md)。

当前决定：

- Candidate A四重复中的小幅Strict信号保留为历史诊断，但固定requested-1000没有复现，
  不再作为完整或scoped正向训练贡献；
- Candidate B旧Plan-only预筛仍保留为负证据，但现按用户新决定进入一次最小真实下游验证；
- 第二个训练侧贡献目前未成立；下一条非RL候选是固定Plan的energy-contrastive DLM
  supervision，并显式锁定中间训练窗口，直接优化stable而非只优化CE或novelty；
- 标准H1-A2继续作为论文fallback；
- public headline继续是`105/1000 Strict、488/1000 Meta`；
- 所有checkpoint和新requested-1000结果只作内部机制证据，不替换headline结果。

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

- 对外Conda环境文件与依赖版本的最终清理；
- 全部checkpoint与完整MP-20数据；
- H1-A2/R03 Plan文件和逐ordinal seed ledger；
- B0与model_494训练seed证据；
- 将内部fixed-256评价adapter整理成对外相对路径版本；
- 发布资产安装后再做公开repo的一键端到端smoke。

所有A800绝对源路径只填写在`ASSET_TRANSFER_LEDGER.md`，不会同步到公开repo。
