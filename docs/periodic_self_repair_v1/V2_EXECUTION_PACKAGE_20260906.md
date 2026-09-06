# V2 六卡执行包：已完成版本的复现记录

更新日期：2026-09-06。**V2 train39993 与 eval39998 均已完整成功，本包不再排队或提交该版本。保留已有 claim、job pointers 和正式终点，不重复提交 249/250。** 当前科学审计与后续经审查、验证和登记的新 V3 工作见 [CURRENT_STATE](../CURRENT_STATE.md)；本包的历史前置条件不覆盖当前授权。

## 完成结果

训练 39993 为 `COMPLETED 0:0`，两 epoch、global batch 24、4524 updates、108544 有效 states；source/view 覆盖 min=max=2，实际初始 delta=0，前 64 更新完成模块梯度检查。评测 39998 的两端、对照和 construction→repair 诊断全部成功。

| 固定 256 请求的终点 | Strict SUN | Meta SUN |
|---|---:|---:|
| 同次 trace 的 construction | 12 | 44 |
| full-cell repair 后 native | 8 | 50 |
| 固定 model494 tau800 | 16 | 113 |

两端均未达到同端 Strict/Meta SUN 至少 26/128（10%/50%）的追加条件，**本版本不追加 1000**。原补评机制见 [补评运行包](../../operations/supplement_1000/README.md)，它的存在不表示本次已触发。只读 probe40004 也已 `COMPLETED 0:0`，用时 2分24秒、四张 A800；未训练权重、未生成新样本、未调用 MLIP。

证据：[训练与 metadata](../v3_scientific_audit_20260906/evidence/V2_FINAL_TRAIN_AND_METADATA.json)、[V2 两端](../v3_scientific_audit_20260906/evidence/V2_EVALUATION_SUMMARIES_39998.json)、[构造与修订](../v3_scientific_audit_20260906/evidence/V2_CONSTRUCTION_AND_REPAIR_39998.json)、[只读 probe](../v3_scientific_audit_20260906/evidence/V2_CONDITIONING_PROBE_40004.json)。两端证据中的早期顶层快照不能覆盖随后完整成功的作业状态。

## 不可变代码、合同与部署布局

- 执行代码：`2e904c260bafb6750c9a5dbb0d920c4ff8c3a868`。
- 启动合同：[V2_LAUNCH_MANIFEST.json](V2_LAUNCH_MANIFEST.json)；其 validation 中的运行前字段保留历史含义，不表示当前尚未执行。
- 方法：[完整论证](LOSS_GEOMETRY_AND_V2_PROPOSAL_20260906.md)；实现：[验收记录](V2_IMPLEMENTATION_REVIEW_20260906.md)。
- 实际入口：[249 train](../../slurm/249_train_periodic_dlm_v2.sbatch)、[250 eval](../../slurm/250_evaluate_periodic_dlm_v2.sbatch)、[前置条件 checker](../../scripts/check_periodic_v2_queue.py)。

原部署变量如下，用于定位和复现，不在本次文档更新中执行：

```bash
ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/proposal_realization_candidates_20260826/grounding
SOURCE=/public/home/jiaosz/ywliang/ai4s/.sscd_periodic_self_repair_20260906_v1
EXPERIMENT="$ROOT/experiments/periodic_self_repair_20260906/v2_from_original_llada"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
```

正式 `LAUNCH_MANIFEST.json` 单独保存在 EXPERIMENT，并在 `runs/train_39993/LAUNCH_MANIFEST.json` 留存。它由后续文档提交提供，指向上述不可变执行代码；执行 commit 内同名早期草稿不是正式启动依据。这种分离避免 manifest 中的 commit 指向自身。

训练和评测各自保存 `CODE_COMMIT` 与 run 内的 `code/` 归档；复现以这些实际归档和正式 manifest 为准，不以当前文档状态重写原运行代码。

## 原一次性提交机制与只读核验入口

原 checker 的只读调用为：

```bash
"$PY" "$SOURCE/scripts/check_periodic_v2_queue.py" \
  --root "$ROOT" --source "$SOURCE" --experiment "$EXPERIMENT" \
  --manifest "$EXPERIMENT/LAUNCH_MANIFEST.json"
```

该入口检查 raw-v1 train/native/tau、旧 K8 train/native/tau、独立 Planner 和主评测八个原前置阶段，以及项目资源与 `_SUCCESS`。历史返回码 2/`ready=false` 表示当时未满足前置条件，不是模型失败。原提交方式是在同一参数上增加 `--submit`；独占 `V2_TRAIN_SUBMISSION.json` 及 `V2_TRAIN_JOB=39993` 已记录这次提交，不能删除 claim 后重提。一次性提交机制现在仅作可复现合同保留。

249 当时使用六张 A800、24 CPU，先准备完整 MP20 的 frozen Planner 条件与显式 fallback，再训练两 epoch。正式 policy 为 `runs/train_39993/train/checkpoints/step-4524`，同时由 `POLICY_PATH`、`TRAIN_FINAL.json` 与最终标记校验。该版本没有 resume 入口，保存 optimizer/scheduler 状态不等于可自动从任意中间 checkpoint 续训。

## 已执行的 250 参数与评测流程

250 使用与训练相同的 SOURCE SHA，实际环境合同为：

```bash
H1A2_CANDIDATE_ROOT="$ROOT"
PERIODIC_REPAIR_SOURCE="$SOURCE"
PERIODIC_V2_ROOT="$EXPERIMENT"
PERIODIC_V2_TRAIN_RUN="$EXPERIMENT/runs/train_39993"
```

`EXPERIMENT/V2_EVAL_JOB=39998`，对应 `runs/eval_39998`。已完成流程是：

1. 固定开发 256 条件；保留原两 worker 批次 layout，在六 worker 执行 construction→full-cell repair。
2. native 共同物理评测，以及固定 model494、800 步、同 graph seed 的 tau800 评测。
3. 两端分别与原参考及 raw-v1 final 配对比较，保留失败和原请求分母。
4. 从同次 trace 提取 construction endpoint，增加共同物理评估以诊断 repair 净作用；没有另采一组 construction 候选。

Strict −4、Meta +6 是 construction→repair 的权衡；refined SUN 与 raw-v1 同计数，不能称该组合普遍改善。现有各 `*-analysis`、support/state/coverage/gradient 记录和 probe 进入 [统一审计](../v3_scientific_audit_20260906/README.md)，不凭这些结果自动增加 repair 轮数或另选 checkpoint。

## 已完成前置版本的指针

| 对象 | 已完成作业/指针 |
|---|---|
| raw-v1 train | 39942，终点 6784；两 view、两 epoch、global batch 16 |
| raw-v1 native / tau800 | `ROOT/runs/RAW_BASE_NATIVE_EVAL_JOB=39975`；`RAW_BASE_TAU800_EVAL_JOB=39984` |
| 原 K8 train / native / tau800 | 39938 / 39945 / 39948 |
| 原独立 Planner / 主评测 | `K8_MAIN_PLANS_JOB=39949`；`K8_MAIN_EVALUATION_JOB=39951`；hull 覆盖修正由 39990 完成 |

raw-v1 固定 256 两端 SUN 为 3/45、16/113；当前 K8 更正结果统一见 CURRENT_STATE。旧 K8 开发结果的 [历史分析](../v3_scientific_audit_20260906/historical/docs/periodic_self_repair_v1/K8_RESULTS_AND_REGRESSION_ANALYSIS_20260906.md) 保留为当时证据，不再作为运行中任务。

当前项目上限仍为六张 A800、24 CPU、两个作业。未来 V3 按当前授权、独立审查、实现验证和新登记执行，不能将本版本的旧等待/暂停条款当成当前审计的状态。刷新前的完整字节和 SHA256 见 [历史快照映射](../v3_scientific_audit_20260906/historical/current_entrypoints_before_status_refresh/manifest_aa5d826c7e39cd442817d49c7605444332b95a5fbe5f45c28b80f7fead43da78.json)。
