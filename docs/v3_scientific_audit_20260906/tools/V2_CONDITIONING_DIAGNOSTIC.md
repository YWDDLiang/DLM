# 冻结 V2 条件与几何使用诊断

入口：[diagnose_v2_conditioning.py](diagnose_v2_conditioning.py)。它只运行已有 V2 checkpoint 的前向，回答实际条件敏感性及 validation 目标风险；不训练、不生成晶体/训练数据、不调用 MLIP，不评价或选择 SUN。主审审查后另行安排 GPU，本工具不 SSH、不提交作业。

默认用独立 identity-hash seed `202609061` 从 prepared validation 中固定选 100 个原 source，再用原 V2 maker 的 `state_seed=20260906, epoch=1` 为每源生成两个**诊断输入状态**（construction、repair）。prefix/dense 和位置来自原分布，不挑较好状态，不强行平衡字段；200 状态的实际分支、空监督、各字段位置与覆盖全部保留。这些是对已有 validation 的 corruption/masking 前向诊断输入，不作为训练输出数据。

## 五个输入变体，三个主要配对

| Variant | 改变 | 保持 | 解释边界 |
|---|---|---|---|
| baseline | 无 | 原 maker 的整数 current/old、P、实际 dropout 后噪声标签 | 原 teacher prefix/dense 状态 |
| noise_true | 三分量改为该 corruption 的 declared 参数 | legacy scalar 仍 −1，其他输入不变 | 参数不是量化后实测误差；construction 参数为零 |
| noise_unknown | 三分量全 −1 | 所有几何和其他输入 | 与 true 一起覆盖已有 metadata-dropout 的两个条件 |
| target_soft_same_program | 用原 validation 的三个 soft annotation 替换 Plan/prompt | 固定 P、N/E、所有整数 current/old、监督位置与目标 | 非重编 program；可能不在训练条件联合分布内，原 annotation 不保证在粗 codec 上保真 |
| old_geometry_masked | 仅 old 的六个 lattice 与 3N 个 coordinate token 置原 MASK | current、old N/E、P、phase、sigma、监督全部不变 | 明示缺少 old 几何的敏感性；repair OOS，且同时改变 conditioner/GEM/完整性状态 |

主要配对是 `noise_true − noise_unknown`、`target_soft_same_program − baseline`、`old_geometry_masked − baseline`。没有 clean-old oracle，没有新 P 采样，也没有连续几何预测。

所有风险都使用**预先固定的 baseline 公共支持**：prefix 为真实 runtime policy support，dense 为原 canonical typed support；另对两种分支都报告 canonical typed/schema risk。alias 用原函数在 raw dtype 中合并，随后以原 T=0.7、float64 log-softmax 诊断。prefix admission 使用原 baseline 的累计可达性/old admission，绝不因 masked-old 不合法而将其风险删掉；因此干预风险是公共支持下的敏感性量，不冒称被干预状态的部署 likelihood。

soft 替换会重新 tokenize prompt，可能改变 body 的绝对位置。记录两个 prompt 和长度，并分 `prompt_length_equal/changed` 报告。后者是“soft 文本及其位置后果”的联合干预，不能仅从它证明纯语义控制。完全相同的模型输入可复用同一 batch 结果，`computed_as` 明确注明；这些零差不是独立重复前向的一致性实验。

## 模块读出

只读 hooks 记录 conditioner cell/site 输出、task projection、GEM 的 body/body 与 target-query/body bias、最终 target hidden、typed numeric-adapter 增量、new-token-output 增量。hooks 不替换输出、不改权重，结束后移除。输出 RMS/最大值与配对差值不能跨参数化直接比较成贡献百分比。

从当前代码可预期：noise 三分量仅进入 V2 geometry attention；固定 P 的 soft prompt 改动不直接改变 old conditioner 或 body 对齐后的 GEM。实际前向会检验这些路径；即便某层激活变化，也不单独证明该模块必要或提高物理性能。masked-old 保留 current 可见数字，不能解释成模型完全没有几何信息。

另对 baseline 与 unknown-noise 的 **teacher-prefix** 状态记录实际 policy logits dtype、合法 logit min/max，以及真实 sampler `exp` 的 dtype。脚本从当前 `_transaction_candidate_tokens` 的显式 cast-and-exp 表达式读取该 dtype（当前为 float64），直接在同一设备计算 legal-vector exp 的 overflow、zero/subnormal 和“全合法值下溢为零”计数；排除 hard-mask 的 `finfo_min` 哨兵，不先除 T。没有识别到该表达式时明确报告未测量，不猜 dtype。这些是本批 validation teacher 状态的实测，**不能据零计数排除所有真实生成 trajectory 的极端 logits**；不采样 token、不增加第六个变体。

## 运行

CPU 入口只做现有 codec fixture 的解析、变体/支持、shape、hook indexing 和 objective 一致性检查，不加载模型：

```text
python docs/v3_scientific_audit_20260906/tools/diagnose_v2_conditioning.py --self-check
```

真实前向复用 `load_periodic_v2_model(..., trainable=False)`；该既有 loader 自身的 final-checkpoint 校验保留，脚本没有 job id/step/path 的硬编码。四卡示例，路径由调用方填入：

```bash
torchrun --standalone --nproc_per_node=4 \
  docs/v3_scientific_audit_20260906/tools/diagnose_v2_conditioning.py \
  --model-path "$RAW_LLADA_MODEL" \
  --checkpoint-path "$V2_CHECKPOINT" \
  --prepared-val "$V2_PREPARED_VAL" \
  --original-val "$ORIGINAL_SFT_VAL" \
  --output-dir "$NEW_DIAGNOSTIC_OUTPUT" \
  --source-count 100 --selection-seed 202609061 \
  --state-seed 20260906 --state-epoch 1 --batch-size 2
```

可单 GPU 运行，或按 `torchrun` world size 分源；不使用 DDP 梯度同步。模型 `eval()`、`requires_grad=False`、`inference_mode()`，参数 version 前后核对，无 optimizer。默认 max length=382，不截断；改变上限须记录为调用配置。输出目录必须是新目录。

## 产物与读取顺序

- `CONFIG.json`：100 source IDs 的固定选择、seeds、输入文件 SHA、checkpoint 文件 SHA、核心源码 SHA、全部干预语义和具体配置。
- `TOKEN_SUPPORTS.json`：原 typed vocabulary 的 canonical 顺序。
- `paired.rank*.jsonl`：每源两个完整状态及五 variants、三配对；位置/字段、baseline 公共支持、sigma、原/预测 S、固定 P、finite、CE/TV/KL/JS、模块激活与变化。
- `typed_logits.rank*.npz`：每个监督位置的完整 canonical typed logits；key 给 source、view、variant、body position，可与支持表和配对记录复算。
- `execution.rank*.json`：实际 forward 数、时间、冻结检查、baseline 与原 `PeriodicV2Objective` 的 batch-loss 对照。
- `SUMMARY.json`：全状态与按 source 两 view 的 macro 风险、分支/字段/长度/OOS/相同输入分层、实际监督覆盖和模块响应；missing/nonfinite 不默默从计数消失。
- `_SUCCESS`：全部选择 source 各两个 view 写完并汇总。它表示诊断完成，不表示模型改善。

先确认来源、分母、finite 和 objective parity，再看三个配对的 schema/common risk 与 TV。显著敏感性说明模型使用某输入通路；零响应应结合几何 known 状态、sigma 标签、prompt 是否改变和模块输出定位。这里没有 free-generation、energy、SUN、checkpoint/seed 选择结论。
