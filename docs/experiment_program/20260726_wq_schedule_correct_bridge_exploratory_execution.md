# WQ schedule-correct bridge：探索性继续执行说明

日期：2026-07-26  
运行：`20260720_0401-crysllmgen-wq-final-v3`

## 1. 现在的实验重心

正式 all-22 composition-projection S.U.N. gate 保持原结论：

- strict：`2/22`；
- meta：`6/22`；
- MP unknown：`1/22`；
- 即使 unknown 乐观计为成功，meta 上限也只有 `7/22`；
- 冻结阈值为 `11/22`；
- 因此正式结论仍是 **FAIL**。

用户的“我觉得问题不大，可以继续推进实验”被记录为一条新的探索性继续授权，
而不是对上述 gate 的追溯性改写。当前不再扩大 composition projector，也不恢复
已经停止的 constrained-decode 分支。

当前唯一科学问题改为：

> 先前几何失败中，有多少来自 CrysLLMGen 训练/推理日程不一致，而不是 WQ
> proposal 本身不可用？

## 2. 已确认的训练/推理错配

released parent 的训练时前向过程是：

```text
x_t = (x_0 + sigma_t * epsilon_x) mod 1
L_t = sqrt(alpha_bar_t) * L_0 + sqrt(1-alpha_bar_t) * epsilon_L
```

历史 matched 32-step parent sampler 则把 clean proposal 直接作为反向状态。这不
等价于任何 `t>0` 的父模型训练分布。新的 preflight 因此强制区分：

- `clean proposal condition`：只读、不可变的 proposal 证据；
- `forward-noised geometry state`：真正送入父 decoder 的 `x_t/L_t`；
- 两者不得共享内存，也不得把 clean condition 再次冒充 reverse state。

## 3. 新的最小实验

身份：`wq_schedule_correct_bridge_parity_v1`

输入不重新生成。只读复用：

```text
runs/20260720_0401-crysllmgen-wq-final-v3/
  outputs/wq_parent_csp_sun256_v1/generation.jsonl
```

冻结 SHA256：

```text
b6eb7f80a29da699407d8d19bbedeb2d657f5d7940cd767d6d71aecb6c58a598
```

从 256 行中仅按 `SHA256(selection_salt:attempt_id)` 选择最小的 8 个
attempt ID。选择不读取 composition、structure、S.U.N.、CHGNet 或任何结果。

矩阵：

| 维度 | 冻结值 |
|---|---:|
| parent timestep | `100, 200, 400, 800` |
| proposal / timestep | `8` |
| 总 cell | `32` |
| reverse steps / cell | `32` |
| parent decoder calls / cell | `64` |
| proposal generation | `0` |
| retry / replacement | `0` |

同一 proposal 在四个 timestep 使用配对的 forward/reverse noise。每个失败 cell
保留在 32 的分母中。

## 4. Gate

进入任何 bridge 短训前，必须同时满足：

1. released `model_494.pt` 完整 `strict=True` load；
2. checkpoint SHA256 为
   `573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e`；
3. scheduler 为 1000 steps、buffer length 为 1001、`run_type=train`；
4. derived schedule 与 strict-loaded buffers 的最大绝对误差不超过
   `1e-6`，相对容差为 0；
5. 已知 forward noise 的 coordinate/lattice reconstruction 最大误差均不超过
   `1e-6`；
6. first reverse step invalid lattice 为 `0/32`；
7. non-finite trajectory 为 `0/32`；
8. 32 个 cell 全部终态；
9. finite positive-volume raw output 为 `32/32`。这是“不低于已冻结 parent
   handoff 4/4”的直接比例门；
10. 无 retry、replacement、best-of、MLIP、外部 API 或新生成。

任一条件失败即停止 bridge training。PASS 也不自动授权训练。

## 5. MLIP-free 边界

本阶段及未来 bridge checkpoint selection 均不得读取：

- MatterSim；
- CHGNet；
- relaxation energy；
- hull distance；
- S.U.N.；
- final test 指标。

后续若获得独立短训授权，checkpoint/early stop 只能使用：

- denoising/recovery loss；
- schedule reconstruction；
- finite trajectory；
- minimum-distance support；
- positive volume / density support；
- proposal recovery。

CHGNet 仍只可在方法和 checkpoint 完全冻结之后作 held-out evaluation。

## 6. 当前完成状态

已完成：

- 用户探索性继续授权的不可覆盖记录；
- 正式 all-22 FAIL 的 non-rewrite boundary；
- exact 4×8 cell 与跨 timestep 配对噪声；
- clean condition / noisy state 分离和 input immutability；
- NumPy 独立 parent schedule 与 forward-noise/reconstruction audit；
- Torch A800 路径的 strict schedule-length 检查；
- correctly forward-noised 32-step parent reverse 入口；
- hash-fixed 8-proposal selection；
- 每 cell append-only evidence 和 terminal report；
- 1×A800 最多 8 CPU 的未来资源 envelope；
- 14 项本地 contract/mutation/static-safety tests：PASS；
- Python compile checks：PASS。

冻结文件：

| 文件 | SHA256 |
|---|---|
| contract | `d4f18bf74a1814d7de6d7a4d4934c615857edef364a039f371723aa1763b4c6b` |
| bridge implementation | `7268b573892359737c713f75d2efb19fda14dc5999b5c79de04f48932d08a434` |
| A800 runner | `1f8fb8716d1b375a64d71715cc31db4e2df301075bc121169ae37fedac0bba64` |
| mathematical tests | `1b653733f1c7448077f52fa6405fa7b2b733d3da7c1359875ac4f4f9ce1acb69` |
| runner safety tests | `de72e16e3f8b422e99ee65c6d5866bd2f4c0cabffe44cb14498496a29d053444` |
| authorization record | `0129966f77c86e02cc4263a0b0a0af2fa75b7068d9a853539b441d771fcfd7f5` |

本地 Python 没有 PyTorch，因此当前结果只证明数学合同、身份、immutability 和
runner safety；它**不**是 checkpoint strict-load 或真实 A800 轨迹 PASS。

## 7. 后续任务

- [x] `B500` clean proposal 与 noisy state 分离。
- [x] `B501-local` 复现 released parent 前向日程和已知噪声重建。
- [ ] `B501-A800` 与 strict-loaded parent schedule 做精确 parity。
- [ ] `B502-A800` strict-load parent decoder/time embedding/checkpoint。
- [x] `B503` 冻结 `t={100,200,400,800}` × 8。
- [ ] `B504-A800` 完成 32-cell schedule reconstruction 与 first-step lattice。
- [ ] `B505-A800` 完成 finite/raw-success/failure accounting。
- [x] `B506-local` 冻结 MLIP-free gate 与 no-selection-leakage。
- [ ] `B507` 只有全部 A800 gate PASS 后，另行设计并授权一次短 bridge-only
  training；本合同不授权训练。

下一动作不是长训，而是把本地冻结实现做成唯一归档，在取得新的传输和 A800
preflight 授权后，只提交一个 1×A800、8 CPU 的 32-cell diagnostic。其终态
决定短 bridge-only training 是继续还是停止。

## 8. 终态补记（2026-07-26）

第 7 节是结果可见前的原始执行计划，继续保留为历史。实际执行发生了两步：

1. 原 diagnostic job28054 在科学计算前因 result schema binding 失败；
   没有可解释的 diffusion cell，禁止把它计为科学失败。
2. 唯一 supersession job28081
   `wq_schedule_correct_bridge_parity_sup28054_v1` 已
   `COMPLETED 0:0`。

job28081 终态：

- A800 strict parent load PASS；
- 32/32 cells terminal；
- first-step invalid lattice `0/32`；
- non-finite trajectory `0/32`；
- finite positive-volume raw output `32/32`；
- schedule max absolute error `0`；
- retry/replacement/new generation/MLIP/API 均为 0。

不可覆盖审计：

```text
runs/remote_audit/
  20260726_wq_schedule_correct_bridge_parity_sup28054_v1/
  terminal_audit.json
```

SHA256：

```text
4b77a10d632c33b53b2208d49db19f542081a7fd91d52f038ebe7e5280e2cf41
```

该 PASS 只关闭 schedule correctness 风险，不授权短训。新的 active 方案是：

```text
docs/experiment_program/
  20260726_wyckoff_tangent_bridge_iclr_plan.md
```

即先复用现有 WQ charts、orbit Jacobians 和 released parent，开发
training-free every-step tangent projection；先完成本地实现与 invariant
tests，再由新的独立授权决定是否运行 WTB-32。
