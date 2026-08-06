# Existing-22 composition projection survival audit 结果

日期：2026-07-25  
Run：`20260720_0401-crysllmgen-wq-final-v3`

## 1. 结论

本 gate 的正式结果为 **FAIL**：

| 指标 | 结果 | 冻结阈值 |
|---|---:|---:|
| rendered | 22/22（100%） | 22/22 |
| composition-valid | 22/22（100%） | 22/22 |
| structural-valid | 17/22（77.27%） | ≥20/22 |
| joint-valid | 17/22（77.27%） | ≥20/22 |

这两个结论必须同时保留：

1. composition repair 的目标实现了：22 个既有投影全部保持
   CrysLLMGen composition-valid；
2. 完整 survival gate 没有通过：5 个结构存在小于 0.5 Å 的原子碰撞，
   structural/joint-valid 只有 17/22。

因此不能把本次结果写成 composition mechanism 的整体 PASS，也不能自动进入
CHGNet、S.U.N 或训练。但失败并不是投影器把化学式重新变成无效：22/22 的
化学有效性完整保留。

## 2. 实验边界

本次只读取已存在的 22 个 `status=projected` 状态：

- 没有新生成；
- 没有重跑 projector；
- 没有重新选择候选；
- 没有 retry/replacement；
- 没有 Slurm 或 GPU；
- 没有 CHGNet、其他 MLIP、MP API 或 S.U.N；
- 没有训练。

唯一科学调用在 A800 登录节点以 CPU-only 方式运行，环境为：

`/public/home/jiaosz/miniconda3/envs/diff_meets_diff`

运行时间约 6.25 秒，all-existing-projected-states 分母固定为 22。

## 3. v1 预检与 v2 表示修正

v1 在 claim 创建和任何科学调用之前 fail-closed：

`SurvivalAuditError: projected state changed topology/geometry of orbit o4`

只读诊断证明这不是 topology 或 geometry 改变，而是序列化表示问题：

- 22 个投影中有 13 个 orbit 列表顺序改变；
- 22/22 的 orbit-ID 集合不变；
- 22/22 的逐 orbit-ID geometry 不变；
- species change 与 projector 声明完全一致。

原因是 `canonical_storage=True` 的存储排序键包含 species；whole-orbit species
assignment 变化后，JSON 中的 orbit 列表可能重新排序。v2 在任何科学结果
出现前冻结，只按 immutable `orbit_id` 验证 topology/geometry，然后恢复源
panel 的 orbit-list 顺序再渲染。样本、阈值、metric backend 与选择规则均未
改变。

## 4. 五个结构失败

五个失败均满足：

- `comp_valid=true`；
- `fingerprint_valid=true`；
- `reason=structure_invalid`；
- volume 远大于 0.1 Å³；
- 最短原子距离低于 CrysLLMGen 的 0.5 Å cutoff。

| ordinal | 投影后化学式 | changed orbit | atoms | volume (Å³) | min distance (Å) | redetected SG |
|---:|---|---|---:|---:|---:|---:|
| 262 | Li2O12P4Cu | o10 | 19 | 158.2551 | 0.3576 | 2 |
| 295 | CdI2 | o1 | 12 | 656.1312 | 0.4870 | 12 |
| 323 | Li2O8Ni7 | o3 | 17 | 148.3937 | 0.2104 | 2 |
| 328 | LiO2Mn | o12 | 20 | 211.2162 | 0.0360 | 0 |
| 445 | P3SrCd2 | o6 | 12 | 234.2940 | 0.0000 | 166 |

17 个有效结构的最短原子距离范围为 0.7138–2.3691 Å。

whole-orbit species reassignment 不改变 lattice、Wyckoff coordinates 或
orbit geometry。因此这五个碰撞暴露的是输入固定 topology/geometry 的可渲染
性缺陷，而不是 composition repair 丢失电荷中性。当前 gate 把完整
composition+structure survival 作为联合要求，所以仍必须判 FAIL。

## 5. 对 ICLR 主线的含义

当前证据支持把问题拆成两个层次：

1. **化学层**：局部、确定性的 whole-orbit species reassignment 对既有
   no-neutral 状态具有真实信号；22/36 可修复，且本次 22/22 仍
   composition-valid。
2. **几何层**：对这 22 个固定 topology 状态，5 个在任何 species-only
   修复之前就受坐标碰撞限制；不能依赖 composition projector 解决。

因此下一版不应扩大 projector search，也不应马上长训。更保守且 MLIP-free
的设计是把确定性的 geometry feasibility 放在 chemistry repair 之前：

- 先渲染 proposal topology；
- fail-closed 检查 volume 与 0.5 Å minimum-distance；
- 只对 geometry-feasible topology 进入 atom-total/orbit-partition
  composition feasibility；
- 保持单 proposal、无 best-of、无 retry/replacement；
- 重新冻结 held-out 分母和阈值后，才决定是否做短训练。

这不是对本次 22 个结果做 survivor-only 重算；本次正式分母和 FAIL 均保持
不变。它只是一项下一阶段、需要重新预注册的模型设计建议。

## 6. 冻结决策

根据 v2 合同：

- 不对本 22 个自动运行 CHGNet relaxation/hull；
- 不运行 S.U.N；
- 不启动 sampler 或 bridge 训练；
- 不修改阈值，不剔除五个失败病例，不重试；
- 若继续，只能先冻结新的 MLIP-free topology/geometry feasibility
  诊断或机制 gate。

## 7. 关键身份

- v1 contract SHA256：
  `4743a2129597b0f68483aa0feb9e2dd286089da3926bbb9ab2b119bf783e540b`
- v2 contract SHA256：
  `fe296ad119ed066b8c70af633a3eafc50e0971d29b52278068cabeeac701f4fe`
- v2 runner SHA256：
  `c8b66b3fa7faad340a49cf584db8f425dac4cf7bb69f248470de78fa1cd8a31d`
- report SHA256：
  `03cd5f5e8c95391775e2b129296c309113c8e54aef29b3975eab7e82bd70f209`
- structures SHA256：
  `47c4cf0b858bb846a5f9dc4df6dafa31b4ce4f20bdfc40be63d5226aac6e475e`
- attempt metrics SHA256：
  `360514adaf189db5ec0f6618cfae84068f8919df048535e17397c24c55fd4f69`
- terminal acceptance SHA256：
  `a9903b8e8012b5992117e4872735c5e5b46d67e009805e0dd7588c7b0ed4c7cc`

机器可读终态审计：

`runs/remote_audit/20260725_wq_existing22_projection_survival_v2_order_alignment/terminal_audit.json`
