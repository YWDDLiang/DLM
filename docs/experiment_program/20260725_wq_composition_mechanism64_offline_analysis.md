# WQ Mechanism-64 离线分析

日期：2026-07-25  
Run：`20260720_0401-crysllmgen-wq-final-v3`

## 1. 结论先行

`22/36` 可以作为**值得继续验证的探索性机制信号**。它不是随机凑出的宽松
结果：

- 22 个成功投影中，15 个只改 1 个 orbit，7 个改 2 个 orbit；
- 19/22 只影响 1–3 个 primitive atoms；
- 22/22 通过冻结 SMACT 默认表和 ICSD16 表，21/22 同时通过所检查的
  四套氧化态表；
- 16/16 valid controls 和 12/12 Pauling-only controls 均保持
  byte-identical。

但这不把原预注册结果改成 PASS。正式记录仍是 `22/36 < 24/36`，Gate
FAIL；用户对 `22/36` 的接受只决定“是否值得设计下一道新 gate”，不能
事后替换阈值或主分母。

更重要的是，22 个当前只证明了 **SMACT composition-valid**。它们尚未证明：

- 物种换位后仍通过 CrysLLMGen structural validity；
- CHGNet 可以成功 relaxation；
- hull 稳定；
- strict/meta S.U.N 提升。

因此现在不应直接训练 sampler。最低成本、最有信息量的下一步，是先对现有
22 个投影做不生成新样本的 survival audit。

## 2. 数据与分析边界

输入身份：

- frozen panel SHA：
  `3da00d574fcf23416d4464b937480d2609f50a04a98d4c81f719c73c938d63f2`
- panel manifest SHA：
  `9a3db1dc1bfa5cbf2cb8b0ecc8158098d3813adf6eae770c5c85922b850b69a7`
- projections SHA：
  `5f2054c52f9fe8db25f272a9ecedac0af9f454f560686e1f81856538b0244a79`
- report SHA：
  `255eba7d9aa862706caa3754ad41a14c111b76480b707ac0d0d3f265df075d2f`
- terminal audit SHA：
  `9b044cbdca5960d03548d651f45e53a7e85f0e8ae3b1c94bbfcc3b46f34a885e`

分析在 A800 登录节点的
`/public/home/jiaosz/miniconda3/envs/diff_meets_diff` 中以 CPU-only
方式完成：

- 无 Slurm；
- 无 GPU；
- 无 MP API 或其他外部 API；
- 无 CHGNet/MLIP 调用；
- 无新生成；
- 无训练；
- 无 projector 重跑或候选替换。

本次新增的只是对既有不可变结果做穷举与敏感性分析。

## 3. `22/36` 应该怎样理解

主结果：

- 预注册分母：22/36 = 61.11%；
- Wilson 95% 区间：44.86%–75.22%；
- 将 4 个 Pm 数据库覆盖病例描述性剔除后：22/32 = 68.75%；
- 22/32 的 Wilson 95% 区间：51.43%–82.05%。

22/32 只能作机制解释，不能替换正式 22/36。样本量仍小，区间较宽，所以
它支持的是：

> 固定 topology 下，whole-orbit species reassignment 经常能修复
> composition validity。

它还不支持：

> 该修复已经提高完整材料生成的 S.U.N。

## 4. 成功投影的结构

### 4.1 修改规模

| 指标 | 分布 |
|---|---|
| changed orbits | 1 个：15；2 个：7 |
| affected primitive atoms | 1：6；2：10；3：3；6：2；8：1 |
| composition-count L1 | 2：9；4：12；8：1 |
| candidate assignments | min 6；median 15.5；max 162 |
| classifier evaluations | min 2；median 3.5；max 20 |

这说明大多数成功不是大范围重写。投影器找到的通常是很局部的离散修复，
并且成功解靠近原 proposal。

### 4.2 22 个 before → after

| ordinal | 原式 | 投影后 | changed orbits | affected atoms |
|---:|---|---|---:|---:|
| 262 | LiO12P4Cu2 | Li2O12P4Cu | 1 | 1 |
| 275 | O4Bi3 | OBi | 1 | 1 |
| 276 | KCu2Se2 | K2CuSe2 | 1 | 2 |
| 285 | LiSi2La2 | Li2SiLa2 | 2 | 6 |
| 292 | Br4Mo3 | Br6Mo | 2 | 8 |
| 295 | CdI5 | CdI2 | 1 | 2 |
| 298 | Cl2Br | Cl3Br | 1 | 1 |
| 316 | O6AlW2 | O5Al2W2 | 2 | 6 |
| 322 | O12Sr4Mo3 | O12Sr5Mo2 | 1 | 1 |
| 323 | Li2O10Ni5 | Li2O8Ni7 | 1 | 2 |
| 324 | S5Er4 | S2Er | 2 | 3 |
| 328 | Li6O9Mn5 | LiO2Mn | 1 | 1 |
| 354 | Li2O8Ta | Li4O6Ta | 1 | 2 |
| 380 | O6NaGe2 | O5NaGe3 | 1 | 2 |
| 383 | O6Na3P2 | O7NaP3 | 2 | 2 |
| 397 | O11Ba3Yb | O11BaYb3 | 1 | 2 |
| 442 | Li3O8Mn2Ni2 | Li3O8MnNi3 | 1 | 1 |
| 445 | P2SrCd3 | P3SrCd2 | 1 | 2 |
| 457 | Li5O8Mn4 | Li6O7Mn4 | 2 | 3 |
| 479 | O4Na3Ca | O3Na4Ca | 2 | 3 |
| 502 | F5Cu4 | F2Cu | 1 | 2 |
| 508 | LiF5Zn | Li2F4Zn | 1 | 2 |

### 4.3 氧化态表敏感性

对 22 个 after-formula 使用同一 charge-neutrality + Pauling 流程，只替换
SMACT 自带氧化态表：

| 表 | 通过 |
|---|---:|
| frozen default | 22/22 |
| ICSD16 | 22/22 |
| SMACT14 | 21/22 |
| Wiki | 22/22 |
| 四表同时通过 | 21/22 |

唯一敏感项是 ordinal 285 `Li2SiLa2`，它未通过 SMACT14。其余 21 个不是
依赖单一默认氧化态表才成立。即便如此，SMACT 仍是 composition-level
necessary screen，不能取代结构、relaxation、hull 或 S.U.N。

## 5. 10 个 no-solution：不是搜索预算问题

对每个 no-solution 状态，分析移除了 `max_changed_orbits=6` 限制，并穷举：

1. 原元素集合不变；
2. 每个原元素至少出现一次；
3. 所有 whole-orbit species assignments；
4. 精确 primitive atom total；
5. 精确 Wyckoff orbit multiplicity partition。

结果：10 个状态中，**可通过冻结 SMACT 的 composition 数仍为 0**。
所以增加 candidate budget、把 6 改成更大、或者重跑 projector 都没有用。

失败恰好分为两类，各 5 个：

### 5.1 总原子数的化学计量同余不允许中和

| ordinal | 原式 | atom total | 最近可行关系 |
|---:|---|---:|---|
| 301 | Cl3Rb2 | 5 | ClRb 需要 total 4 或 6 |
| 337 | F6MgCd | 8 | F6Mg2Cd / F6MgCd2 需要 total 9 |
| 359 | O6K3Rb2 | 11 | 可行式出现在 total 10 或 12 |
| 483 | O9V4 | 13 | 可行 O/V 比例出现在 total 12 或 14 |
| 490 | H2MgCe2 | 5 | H4MgCe 需要 total 6 |

这五个不是 orbit search 不够，而是固定 topology 的 primitive atom total
本身与可中和整数比例冲突。

### 5.2 同 atom total 有有效比例，但 Wyckoff partition 拼不出来

| ordinal | 原式 | multiplicities | 同 total 有效比例示例 |
|---:|---|---|---|
| 267 | O2ClCa | 2,2,2,2 | O3Cl4Ca / O5ClCa2 |
| 336 | O6Ti | 2,4,8 | raw 7:7 的 OTi |
| 384 | B6Co12Dy | 1,6,6,6 | B10Co2Dy7 等，共 49 个 |
| 449 | Ni2ILa2 | 2,4,4 | NiI3La / Ni2I5La3 等 |
| 488 | Li2P5 | 2,2,2,2,2,2,2 | raw 7:7 的 LiP |

这些元素集合和总原子数本来存在有效 composition，但固定 orbit
multiplicity 只能形成更粗的整数分区，因此无法到达有效比例。

## 6. 4 个 taxonomy/classifier 分歧

四个 ordinal 为 282、331、381、443：

- `CSiPm3`
- `LiRhPm2`
- `PmTmPt2`
- `LiPmAu2`

共同点是全部含 Pm。冻结 default/ICSD24 表中 Pm 的 oxidation states 为空，
所以细分类器返回 `oxidation_state_missing`：

| Pm 表 | oxidation states |
|---|---|
| default | [] |
| ICSD24 | [] |
| ICSD16 | [3] |
| SMACT14 | [2,3] |
| Wiki | [2,3] |

因此这不是 projector 搜索失败，而是旧 taxonomy 将“氧化态数据库无覆盖”
包含在宽泛的 no-neutral 桶中。后续 panel 冻结应预先把
`oxidation_state_missing` 单列为 unscorable/coverage stratum，不能与真正
的 charge-neutrality failure 混为一个主分母。

## 7. 对实验设计的直接启示

当前 fixed-topology projector 的正确定位是：

> 一个便宜、确定、MLIP-free 的局部 composition repair operator。

它不应该被扩成盲目更大的搜索。无解病例已经证明：

- 仅增大 search budget 无收益；
- 仅放宽 changed-orbit 数无收益；
- 真正需要的是在 proposal/decode 前检查 topology feasibility。

建议下一版本加入确定性的两层 mask：

1. **atom-total feasibility**：给定元素集合与 primitive atom total，是否存在
   任一 charge-neutral integer stoichiometry；
2. **orbit-partition feasibility**：该 stoichiometry 是否能由当前 Wyckoff
   primitive multiplicities 的子集和/多分区实现。

两层均不需要 MLIP、训练或外部 API。它们能在解码前识别当前 10 个必然无解
状态，避免模型把概率质量放到 topology–chemistry 不相容区域。

## 8. 推荐的下一道 gate

不要立刻训练。先冻结一个 `existing-22 projection survival audit`：

1. 输入只允许现有 22 个唯一投影，不生成新样本；
2. exact all-22 denominator，无 retry/replacement；
3. 渲染投影后的结构；
4. 先测 CrysLLMGen composition + structural validity；
5. 只有重新预注册后，才用 CHGNet 做 relaxation/hull；
6. S.U.N 仍按 all-22 denominator，失败和 unknown 均计 0；
7. 不用结果回改 projector objective 或选择候选。

若 22 个在结构/CHGNet/S.U.N 阶段大面积消失，说明 composition repair
破坏了局部化学环境，路线应停止。若能保留显著比例，才值得把
topology-feasibility mask 接入 proposal/decode，并做新的 held-out 256，
而不是现在就训练。

### 8.1 Existing-22 survival gate 已预注册

在查看 22 个投影态的 CrysLLMGen 结构结果之前，已冻结：

`configs/experiments/wyckoff_codiffusion/wq_existing22_projection_survival_v1.json`

其 SHA256 为：

`4743a2129597b0f68483aa0feb9e2dd286089da3926bbb9ab2b119bf783e540b`

协议边界：

- 只读取既有 `projections.jsonl` 中身份固定的 22 个 `status=projected`
  状态；
- exact all-22 denominator；render、metric 或 fingerprint failure 都计 0；
- 不重跑 projector、不重选候选、不生成、不训练、不使用 Slurm/GPU；
- A800 登录节点 CPU-only，固定 `diff_meets_diff` 环境；
- 用 `PyXtalChartCatalog(hall_style=spglib)` 和唯一 `expand_state` 路径
  重建 primitive structure；
- 使用未修改的 CrysLLMGen `compute_metrics.Crystal`；
- 不调用 CHGNet、MP API 或 S.U.N。

通过阈值也在结果可见前冻结：

- rendered = 22/22；
- composition-valid = 22/22；
- structural-valid ≥ 20/22；
- joint-valid（含 fingerprint）≥ 20/22。

`20/22=90.9%` 对齐既有 confirmatory 设计的 `joint ≥88%`（离散后需
ceil 到 20），并高于 parent 256 panel 的 `207/256=80.86%` joint-valid。
若本 gate FAIL，不进入 CHGNet/S.U.N 或训练；若 PASS，也只允许另行冻结
all-22 CHGNet/S.U.N 协议，不能自动执行或用结果改 projector。

## 9. 审计产物

机器可读离线分析：

`runs/remote_audit/20260725_wq_composition_mechanism64_offline_analysis_v1.json`

原终态审计：

`runs/remote_audit/20260725_wq_composition_mechanism64_v1_terminal.json`

## 10. Existing-22 survival audit 终态

后续唯一 CPU-only all-22 audit 已完成。正式结果：

- rendered：22/22；
- composition-valid：22/22；
- structural-valid：17/22；
- joint-valid：17/22；
- 冻结门槛：structural/joint 均至少 20/22；
- terminal：**FAIL**。

五个失败均为既有坐标产生的小于 0.5 Å 原子碰撞；物种投影没有改变
lattice、orbit coordinates 或 topology。由此可见 composition repair 信号
保留，但固定 topology/geometry 没有达到完整 survival 要求。按冻结停止规则，
未进入 CHGNet、S.U.N 或训练。

详细解释与下一步边界：

`docs/experiment_program/20260725_wq_existing22_projection_survival_outcome.md`

机器可读终态审计：

`runs/remote_audit/20260725_wq_existing22_projection_survival_v2_order_alignment/terminal_audit.json`
