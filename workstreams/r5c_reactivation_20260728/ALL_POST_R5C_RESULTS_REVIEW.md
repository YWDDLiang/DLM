# R5-C 之后全部成果复盘与恢复点

日期：2026-07-28

## 结论

ICLR 主线重新确定为：

```text
goal
  -> H1-A2 epoch-2 Planner
  -> one immutable shared Plan
       -> frozen R5-C exact-length diffusion language model -> draft
       -> Plan-conditioned CrysLLMGen diffusion refiner      -> crystal
  -> CrysLLMGen direct metrics
  -> original A100-script / CHGNet S.U.N.
```

恢复不是从“时间上最新”的 PlanBridge checkpoint 开始。最新的 PlanV2
directionality gate 已经给出科学负结果。恢复点由两部分组成：

1. **端到端离散主干**：H1-A2 epoch-2 Planner + 冻结 R5-C exact-length
   body DLM；
2. **可移植的连续机制**：R5-C shared-Plan S2 refiner 的权重和训练数据，
   但只作为条件式初始化与机制证据。S2 的 exact-null 完整轨迹不等价，
   必须先修复 null 接口，不能直接晋升为主模型。

这同时保留了已验证最强的 fully-de-novo 生成能力和后续最有价值的
`Plan -> diffusion` 因果信号。

## 评价口径

所有结果按四类归档，避免把不同问题混为一谈：

- **de novo headline 候选**：Plan 来自 Planner，完整一发生成，分母包括所有
  attempts；
- **conditional/oracle 机制证据**：Plan 或 draft 来自真实结构或冻结 R5-C
  条件式结果，只能证明接口或上界；
- **科学负结果**：作业正常完成，但预注册机制或质量 gate 未通过；
- **纯运行证据**：只解决环境、导入或 Slurm 问题，不包含可用于论文的科学结果。

S.U.N. 统一采用原 A100 脚本语义、CHGNet、all-attempt denominator：

- strict：`E_hull <= 0.0 eV/atom`；
- meta：`E_hull <= 0.1 eV/atom`；
- coverage-adjusted 只作补充；
- 不在训练、筛选、重排或重试中使用 MLIP、MatterSim、S.U.N. 或 MP 查询。

## 第一阶段：原始 R5-C 与 de novo Planner/DLM

| 分支 | 主要结果 | 证据等级 | 处理 |
|---|---|---|---|
| canonical R5-C | comp 90.7%，struct 99.8%，strict adjusted 10.61%，meta adjusted 74.38% | conditional gold-Plan 上界 | 保留为 body/refiner anchor，不冒充 de novo |
| R5-C2 exact-length | parse 100%，graph 96.88%，comp 91.02% | body 表示机制 | 冻结 executor 设计 |
| R5-D11/D12b/D13 | SMACT 64.45% / 69.53% / 75.39%；all-metal 仍约 41–45% | Plan 表示诊断 | 只复用 count/chemistry 表示经验 |
| DN1 | plan/body/graph 29.30% | 科学负结果 | 不重跑 |
| DN2 | plan/body/graph 51.17% | 科学负结果 | 不重跑 |
| DN3 | gate 256 全过；full-1000 comp 84.2%，strict 2.99%，meta 34.02% | 完整负结果 | 不从此恢复 |
| DN4 | plan/body/graph 98.83%，但 4+ 元素 73.44%，`N>=12` 55.47% | 分布病理 | 不重跑 |
| DN5 | parse 95.7%，all-metal 63.67%，长结构消失 | collapse | 不重跑 |
| **H1-A2 epoch 2** | comp 87.8%，struct 99.9%，COV-R 94.96%，strict 9.71%，meta 48.94%，novel+unique 89.0% | **最佳 fully-de-novo 主干** | **恢复 Planner checkpoint** |
| H1-A3 epoch 1/2 | strict 7.00/8.11%，meta 47.8/45.03%；epoch 3 已取消 | 科学负结果 | 不补 epoch 3 |
| free-geometry default | graph 1173/1200，comp 88.2%，strict 9.60%，meta 50.39% | 同一 R5-C executor 的 ablation | 证明默认约束稳健；不是独立 checkpoint |
| H1-G1 | strict 8.67–9.19%，meta 48.26–49.61% | no-promotion | 不续训 |
| H2-P1 | graph yield 约 4%，strict 5.06%，meta 43.02% | 接口 collapse | 不重跑 |
| H1-A4 epoch 1/2 | epoch 2 comp 86.4%，strict 8.35%，meta 47.32%；hybrid gate 过但 S.U.N. 未升 | 完整负结果 | 不续训 |

H1-A2 epoch 2 是最后一个没有被后续结果推翻的 fully-de-novo 主干。它并未达到
conditional R5-C 的 meta 上界，但在 Planner 生成 Plan 的条件下，同时保住了
graph/validity/diversity 和接近 CrysLLMGen 的 strict S.U.N.。

恢复工件：

```text
Planner
runs/20260603_034533-h1a2-epoch2-3-fullmetrics/
  outputs/h1a2_epoch2_llama_rich_sft/final

R5-C exact-length body DLM
runs/20260529_212834-r5c-exactlen-256/
  outputs/r5c_exact_sft/final

CrysLLMGen parent refiner
/public/home/jiaosz/hengzhang/Code/crysllmgen-main/
  out/mp_20/22042026/203930/model_494.pt
```

## 第二阶段：连续 diffusion 与 shared-Plan 后续

| 实验 | 已建立的结果 | 未建立/失败之处 | 处理 |
|---|---|---|---|
| MP20 Plan executor | full Plan 对 target-only 的 lattice relative error 改善 0.000180；对 shuffled 改善 0.000287 | oracle schedule-correct reconstruction，绝对效应很小 | 机制证据 |
| Shared-Plan 1024 factorial | LLM joint strategy +14.336pp；删除效应 +17.188pp；direct diffusion +1.239pp | exact composition -3.906pp，struct -2.246pp；总 gate 失败 | 证明双路径可读，不复用旧 body |
| Plan-conditional benchmark | direct geometry effect +3.7168pp | strict 1.6→1.4%，meta 13.7→13.4% | 低质量 draft basin 负结果 |
| Minimal Plan stability | Crys 分布基本保持 | strict 2.7% 不变，meta 18.4→17.3% | volume-only 路线停止 |
| Crys-origin terminal bridge | matched volume-bin hit 比 shuffled +63.28pp | 连续 volume error 出现灾难性长尾 | 只保留“adapter 会读条件”的证据 |
| **R5-C shared-Plan S2** | lattice-family matched-shuffled +21.678pp；Crys 质量非劣；post-hoc strict 8.20→10.55%，meta 65.62→67.19% | exact-null 完整 reverse 轨迹漂移：coord 0.49997，lattice 3.78357 | **保留权重/数据，先修 null** |
| PlanBridge v1 body | matched-shuffled lattice hit +12.925pp | comp-valid CI 和 density W gate 失败；并非严格同 realized noise | 不作为主干 |
| PlanBridge v2 | 无科学样本 | Pydantic 导入前失败 | 纯运行证据 |
| post-v2 sandbox | 环境/源码 sandbox 通过 | 没有科学结果 | 仅复用工程修正 |
| PlanBridge successor | 两秒 batch 完成并写出 identity receipts | sacct 延迟触发 write-once failure；无群体科学结果 | 纯运行证据 |
| **PlanV2 directionality gate** | validity gate 通过 | `F11-F1S=0.00787`, 95% CI `[-0.01714,0.03426]`；shuffled following 方向相反 | **最新科学负结果，不续训** |

### S2 为什么值得保留但不能直接恢复

S2 是所有后续实验里最强的连续 diffusion 信号：

- 训练只用了 1,840/208 个 same-source R5-C draft pair；
- 5,000 adapter-only updates，best step 250；
- matched Plan 对 shuffled Plan 的 lattice-family effect 为 +21.678pp；
- composition 保持 89.4531%，structure 99.6094%；
- post-hoc CHGNet S.U.N. point estimates 同时提高 strict 和 meta。

但 S2 的 null 条件只在单状态训练诊断中接近 parent，在旧 gold gate 的两次独立
800-step replay 中不等价。源码复查进一步定位了原因：

- `force_null` 已经直接调用 frozen parent decoder；
- FiLM 与 lattice residual 已经在 learned transform 之后乘 active mask；
- 旧 gold gate 却分别执行了 R0-parent 和 R1-null 两条完整 CUDA 轨迹；
- CSPNet 内的 scatter/reduction 不保证两次独立 replay 位级一致，微小差异经过
  800 个 predictor-corrector steps 被放大。

因此旧失败是**null-control 执行协议错误**，不是已证明的 adapter 泄漏，也不是
Plan 信号不存在。正确修复是：

1. 保留已有 post-transform hard mask；
2. 在 denoiser boundary 验证 `force_null` 确实直达 parent；
3. R0 与 exact-null R1 只执行一次 parent reverse trajectory，并把同一不可变
   state 作为两个标签的共同结果；
4. matched/shuffled arms 才分别执行 conditional wrapper。

S2 冻结工件：

```text
/public/home/jiaosz/ywliang/ai4s/llm_plan_diff_runs/
  r5c-shared-plan/s2/20260724T035025Z-093ce0b6a951/
    training/best_plan_adapter.pt

adapter SHA256
533fd02fcc9ac5b77594350ee901fab77dc75c8ddef18a7b333edcf2183e097d
```

## 恢复决定

### 立即恢复

1. H1-A2 epoch-2 Planner；
2. frozen R5-C exact-length constrained DLM body；
3. frozen CrysLLMGen parent；
4. S2 的 PlanLite encoder/adapter、same-source training pairs 和权重，限于
   null-repair 初始化与条件式机制 gate。

### 只移植经验，不移植 checkpoint

- R5-D 的显式 formula/count 表示；
- free-geometry default 的 duplicate-coordinate 与 lattice-volume 约束；
- Shared-Plan 的 same-draft/same-noise matched/null/shuffled 设计；
- PlanBridge 的严格 source sandbox 与 no-outcome-feedback guard。

### 明确停止

- DN1–DN5；
- H1-A3 epoch 3；
- H1-A4 continuation；
- H1-G1/H2-P1；
- volume-only/minimal-Plan adapter；
- PlanBridge v1/v2/successor 续跑；
- PlanV2 方向性 gate 扩样或续训；
- MatterSim/MLIP guidance、候选池筛选、repair、retry 或 rerank。

## 最小下一步

先不训练 Planner、body 或 refiner。第一步只做共享 Plan refiner 的
single-trajectory exact-null 修复和一轮冻结 256 条条件式机制确认：

```text
same frozen R5-C draft + same reverse noise
  R0 stock parent
  R1 repaired adapter + exact-null
  R2 repaired adapter + matched Plan
  R3 repaired adapter + shuffled Plan
```

只有以下条件同时满足，才进入 H1-A2 fully-de-novo paired-256：

- residual null 在位级上为零且 `force_null` 直达 parent decoder；
- parent/null 共用同一次完整 reverse trajectory，输出 hash 必须完全相同；
- matched-shuffled Plan identity effect 的方向和置信区间为正；
- composition、structure、coverage 和 Wasserstein 指标对 parent 非劣。

这一步修的是已知接口缺陷，不读取 S.U.N. 来调参，也不重新生成训练数据。
