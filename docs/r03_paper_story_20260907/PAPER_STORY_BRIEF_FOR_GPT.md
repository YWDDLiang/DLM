# 上一轮 TRAIN/FIT 实验事实底稿

这份底稿只记录上一轮已经完成的 TRAIN/FIT 实验。它不包含 MAIN，也不使用当前 `ranked_train_rsi_20260909` 的失败 G1 结果。请 GPT 只根据下面的事实自行提炼科学问题、科学方法和贡献点，不要把这些结论预先写死。

## 实验身份

- 实验版本：`post_refine_sun_gated_20260909`；
- 运行：`run_20260908T161205Z/rsi_training_v1`；
- 配置记录：[RUN_SPEC.json](execution/post_refine_v08/RUN_SPEC.json)；
- 完成记录：[RSI_FINAL_RESULTS.json](execution/post_refine_v08/rsi/RSI_FINAL_RESULTS.json)；
- 本底稿只看 `FIT` 训练 cohort，固定分母为 256 条 TRAIN 请求；
- FIT Plan cohort SHA-256：`05e633ee4550d5acc539ce19e72acdbaddd827a666e54e363bb144d1c417c486`；
- 共完成 3 轮联合 G/E 权重更新，6 个训练凭据均通过最终审计。

## TRAIN 输入和模型链

训练输入是固定 TRAIN Plan 及其对应 prompt。每条请求保留 Plan、body、请求顺序和随机种子。训练和后续检查都围绕同一条 TRAIN 请求链展开：

```text
TRAIN Plan / prompt
        ↓
G body generation
        ↓
raw construction result
        ↓
frozen model494 F800 refinement
        ↓
tokenized F endpoint
        ↓
E KEEP/EDIT editor
        ↓
edited TRAIN endpoint
```

使用的主要固定资产：

- B0/R5-C exact-length DLM body checkpoint；
- R03 safe-axis generation runtime；
- 冻结的 MP20 `model494` 连续晶体 refiner；
- 已初始化的 E 专家编辑器；
- 固定的结构有效性、凸包、N/U 和物理标签流程；
- 固定的 TRAIN Plan 和 seed ledger。

## 上一轮采用的具体方案

上一轮不是重新训练 F，也不是让 CHGNet 在线替学生选择结构。方案是交替更新两个 DLM 分支：

### G 更新

G 根据同一 TRAIN Plan 下的结构端点反馈构造训练记录。raw 结构经过同一个 F800 和 token roundtrip 后再比较质量；训练目标保留 Plan 条件和请求身份，不跨 prompt 拼接结构。

G1、G2、G3 的训练凭据如下：

| 更新 | optimizer steps | local preference examples | parameter delta squared |
|---|---:|---:|---:|
| G1 | 128 | 505 | 0.0808741463 |
| G2 | 128 | 499 | 0.0736000571 |
| G3 | 128 | 502 | 0.0438716692 |

### E 更新

E 接收本轮 TRAIN 的 token-F current state，学习是否 KEEP、是否 EDIT、编辑范围、数值 token 填充和完整 proposal 的接受／拒绝。编辑保留 Plan 的原子数和元素条件；实际 proposal、拒绝、接受和 unknown 都留在轨迹中。

E1、E2、E3 的训练凭据如下：

| 更新 | optimizer steps | local preference examples | local decision examples | parameter delta squared |
|---|---:|---:|---:|---:|
| E1 | 96 | 366 | 384 | 2.0691473523 |
| E2 | 96 | 347 | 384 | 1.5871357604 |
| E3 | 96 | 359 | 384 | 1.5211779262 |

### 固定策略设置

这些是上一轮 TRAIN 使用的原始设置，不能与本轮修复分支的新设置混写：

- `draft_sun_gate=true`；
- `meta_still_refined=true`；
- 偏好顺序：`SUN > Stable > MSUN > MetaStable > other`；
- `construction_recoveries=1`；
- 每轮 F refinement 为 800 steps；
- `editor_max_calls=16`；
- 单次编辑最多 4 个 sites；
- `max_numeric_bin_delta=1`；
- `prefer_raw_if_refiner_degrades=true`；
- `physical_preference_is_online_reranking=false`；
- `DPO_training_enabled=false`。

上一轮物理偏好使用固定的 `0.01 eV/atom` 工程分辨阈值。不能比较的物理结果保留为 unknown，不能自动写成负例。

## TRAIN/FIT 端点结果

以下只列 FIT 训练 cohort，每一行都是 256 条 TRAIN 请求；不含 MAIN，也不代表独立测试集泛化。

| 轮次 | 端点 | comp_valid | Struct_valid | SUN | MSUN |
|---|---|---:|---:|---:|---:|
| theta0 | construction | 184 | 229 | 4 | 36 |
| theta0 | refined | 184 | 229 | 25 | 110 |
| theta0 | tokenized | 184 | 229 | 26 | 106 |
| theta0 | edited | 184 | 229 | 26 | 106 |
| theta1 | construction | 201 | 253 | 7 | 32 |
| theta1 | refined | 201 | 253 | 29 | 120 |
| theta1 | tokenized | 201 | 253 | 31 | 119 |
| theta1 | edited | 201 | 253 | 32 | 121 |
| theta2 | construction | 204 | 256 | 4 | 29 |
| theta2 | refined | 204 | 256 | 29 | 117 |
| theta2 | tokenized | 204 | 256 | 30 | 114 |
| theta2 | edited | 204 | 256 | 30 | 119 |
| theta3 | construction | 204 | 256 | 4 | 33 |
| theta3 | refined | 204 | 256 | 32 | 117 |
| theta3 | tokenized | 204 | 256 | 32 | 114 |
| theta3 | edited | 204 | 256 | 32 | 116 |

这些数字描述的是训练 cohort 在各轮和各端点的观测结果。不要把 theta3 FIT 的结果写成 held-out、MAIN 或全量 MP20 测试结果。

## 需要 GPT 自己提炼的内容

请根据以上事实自行完成：

1. 科学问题：从 Plan-conditioned DLM、物理端点和 TRAIN 反馈之间的关系提炼；
2. 科学方法：从 G→F800→token-F→E 的实际链路、固定条件和三轮联合更新提炼；
3. 贡献点：只提出能被上述上一轮 TRAIN 事实支持的表述。

写作时不要引用或混入：

- 当前 `ranked_train_rsi_20260909` 的失败 G1；
- 当前修复分支的新 batch/KL/曝光限制，除非另行标为后续修复；
- MAIN cohort 的结果；
- 没有在 TRAIN 记录中出现的 DFT、额外 seed 或泛化结论；
- 把 frozen model494 或物理教师的结果写成学生模型独立获得的结果。
