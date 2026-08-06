# Existing-22 用户接受继续门与 CHGNet R5-C S.U.N. 评测合同

日期：2026-07-25  
run：`20260720_0401-crysllmgen-wq-final-v3`

## 1. 历史结果不改写

原 `existing-22 projection survival` 预注册门仍是正式 **FAIL**：

- rendered：22/22；
- composition-valid：22/22；
- structural-valid：17/22；
- joint-valid：17/22；
- 冻结阈值：structural/joint 均至少 20/22。

用户随后明确把 17/22 接受为足以继续的探索性结构信号。该决定新建一个
`user-accepted exploratory continuation gate`，不把原 20/22 阈值改成
17/22，也不删除五个失败病例。

## 2. 当前实验重心

当前不训练、不生成新样本，也不把 MLIP 放进训练。唯一问题是：

> 22 个已经冻结的 composition projection，在保留全部失败分母后，是否仍有
> 足够的 exact CHGNet R5-C strict/meta S.U.N. 信号，值得继续开发
> MLIP-free feasibility mask？

因此下一步先执行一次独立、不可覆盖的 all-22 held-out evaluator gate。
geometry、atom-total 和 orbit-partition feasibility mask 均推迟到评测结果
之后；这避免先花时间改 sampler，最后才发现该机制没有稳定性迁移。

## 3. 固定输入与失败处理

输入只允许已完成 v2 survival audit 的：

- `structures.jsonl`，SHA256
  `47c4cf0b858bb846a5f9dc4df6dafa31b4ce4f20bdfc40be63d5226aac6e475e`；
- `attempt_metrics.jsonl`，SHA256
  `360514adaf189db5ec0f6618cfae84068f8919df048535e17397c24c55fd4f69`。

主分母固定为 22，禁止 survivor-only 重算。17 个 joint-valid 结构进入
CHGNet；以下五个已知 structural-invalid ordinal 写成显式 failed
placeholder：

`262, 295, 323, 328, 445`

它们在 strict/meta S.U.N. 中都计 0，并且不交给 CHGNet relaxation。
这样可以防止把 relaxation 当作事后 geometry repair，从而虚增本机制的
结构 survival。

## 4. 评测定义

- 环境：`/public/home/jiaosz/miniconda3/envs/diff_meets_diff`；
- CHGNet package：0.4.2；
- CHGNet model semantics：0.3.0；
- model SHA256：
  `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`；
- evaluator：exact R5-C A100 protocol on A800；
- strict：`E_above_hull <= 0.0 eV/atom`；
- meta-like：`E_above_hull <= 0.1 eV/atom`；
- novelty、uniqueness、relaxation、hull 与 S.U.N. 均使用冻结实现和缓存；
- Slurm 内不调用 MP API；hull unknown 在主 lower bound 中计 0；
- coverage-adjusted 只报告，不能决定是否通过；
- 无 generation、best-of、retry、replacement、candidate reselection 或训练。

## 5. 方向性阈值

历史 CrysLLMGen 1000-sample、同一 exact CHGNet R5-C 口径为：

- strict：90/1000 = 9.0%；
- meta-like：461/1000 = 46.1%。

22 条离散化后的继续阈值冻结为：

- strict S.U.N. 至少 `ceil(22 × 0.09) = 2`；
- meta S.U.N. 至少 `ceil(22 × 0.461) = 11`。

该 22-panel 是机制富集小样本，因此这个比较只作“是否值得继续”的方向门，
不是总体性能等价检验，也不能成为 ICLR headline 数字。

决策是三态：

1. **PASS**：lower-bound strict≥2 且 meta≥11；
2. **FAIL**：即使把全部 hull unknown 都算成功，strict 或 meta 仍达不到阈值；
3. **INCONCLUSIVE_MP_COVERAGE**：lower bound 不足，但 unknown 可能改变结论。

第三种情况必须停止，另行授权 A800 登录节点 MP cache completion；不得在
Slurm 内查询，也不得训练。

## 6. 资源与执行边界

唯一评测 job：

- 普通 `gpu` 分区；
- 1×A800；
- 8 CPU；
- 96 GiB；
- 4 h；
- 严格满足 `CPU <= 8 × A800`。

用户本次只授权同一个最终哈希归档：

1. local workspace → starteam5090 staging，一次；
2. starteam5090 staging → A800 staging，一次。

该 SCP 权限不可复用于后续归档。安装、submission claim、sbatch submission、
scientific output 和 terminal acceptance 均使用唯一 identity，禁止覆盖与
静默重试。

## 7. 结果后的动作

- PASS：只允许设计并重新预注册 MLIP-free geometry +
  atom-total/orbit-partition feasibility-mask 实验；不自动训练。
- FAIL：停止 composition-projection escalation。
- INCONCLUSIVE：只允许另行授权的登录节点 MP cache completion。

冻结 contract：

`configs/experiments/wyckoff_codiffusion/wq_existing22_chgnet_sun_v1.json`

contract SHA256：

`43ed097b3af67f211a968e9f4e73f25ed127174fdb50cd9470beb12a2086a823`

用户授权记录：

`runs/remote_audit/20260725_wq_existing22_chgnet_sun_v1/authorization_record.json`
