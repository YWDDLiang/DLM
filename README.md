# H1-A2个人研究仓库

> The final C3FD–Llama / Compact-V2 / G2 paper pipeline is documented in
> [`PAPER_PIPELINE.md`](PAPER_PIPELINE.md). Its portable contracts can be
> validated with `PYTHONPATH=src python -m crystal_dlm.paper_pipeline validate`.

这是H1-A2的可配置研究版本。它与论文repo物理独立，不import论文repo，
也不依赖当前历史项目的绝对路径。

保留的主流程为：

```text
Planner → DLM body → Diffusion refiner → Direct/N/U/S.U.N.
```

## 内部结果记录

| 结果视图 | 条目数 | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|---:|
| 对外H1-A2方法族展示值 | 1,000 | 105/1000 = 10.50% | 488/1000 = 48.80% |
| H1-A2 epoch 2 | 1,000 | 94/1000 = 9.40% | 474/1000 = 47.40% |
| R03 D1 control | 1,024 | 99/1024 = 9.67% | 523/1024 = 51.07% |
| R03 D2 Safe-axis | 1,024 | 117/1024 = 11.43% | 496/1024 = 48.44% |

B3和Legacy adjusted不进入当前主结果集合。R5-C作为Gold Plan
conditional executor reference进入机制分析，但不作为fully de novo headline。

`105/1000`与`488/1000`继续作为未来论文S.U.N.主表值；exact all-attempt和
historical frozen结果保持独立审计口径，不替换主表。

## 使用方式

```bash
cp configs/personal.example.json configs/personal.local.json
python -m h1a2_personal.config configs/personal.local.json
bash scripts/submit_personal.sh configs/personal.local.json
```

完整流程见[WORKFLOW.md](WORKFLOW.md)。未复制的A800环境、数据、
checkpoints和seed ledger记录在
[ASSET_TRANSFER_LEDGER.md](docs/ASSET_TRANSFER_LEDGER.md)。

当前历史项目继续作为完整runs/archive档案；本repo只保留H1-A2/R03相关的
干净代码和可配置入口。

当前构建状态见[`docs/BUILD_STATUS.md`](docs/BUILD_STATUS.md)。
严格ICLR红队分析见
[`docs/ICLR_REVIEW_STRATEGY.md`](docs/ICLR_REVIEW_STRATEGY.md)。
只关注论文故事和概念贡献的内部版本见
[`docs/PAPER_STORY_INTERNAL.md`](docs/PAPER_STORY_INTERNAL.md)。
Proposal–Realization故事与现有分析、最小新增推理的优先级见
[`docs/EXPERIMENT_PRIORITIES_INTERNAL.md`](docs/EXPERIMENT_PRIORITIES_INTERNAL.md)。
双候选方法、选择规则和fallback见
[`docs/DUAL_CANDIDATE_PROGRAM_INTERNAL.md`](docs/DUAL_CANDIDATE_PROGRAM_INTERNAL.md)。
去重后的H1-A2化学难度表见
[`results/historical_difficulty/analysis.md`](results/historical_difficulty/analysis.md)。
最新顶会相关工作、CrysLLMGen最近邻分析与故事缺口见
[`docs/RELATED_WORK_INTERNAL.md`](docs/RELATED_WORK_INTERNAL.md)。
Fully de novo边界、learned Planner与R5C/Plan replay控制的区别见
[`docs/DE_NOVO_SCOPE_INTERNAL.md`](docs/DE_NOVO_SCOPE_INTERNAL.md)。
下一轮独立对话的研究故事压力测试prompt见
[`docs/NEXT_CONVERSATION_PROMPT.md`](docs/NEXT_CONVERSATION_PROMPT.md)。
对应的concept-only reviewer结论见
[`docs/STORY_REVIEW_INTERNAL.md`](docs/STORY_REVIEW_INTERNAL.md)。
