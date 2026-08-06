# WQ LLM 主线与 charge-aware STOP pilot

状态：`ACTIVE_PAIRED64_PILOT_AUTHORIZED`  
日期：2026-07-27  
run：`20260720_0401-crysllmgen-wq-final-v3`

## 1. 当前重心

论文主线回到已完成三轮训练的 WQ LLM。近期目标依次是：

1. 提升 proposal composition validity；
2. 保持公式、空间群、Wyckoff topology 与原子数分布不塌缩；
3. 只对冻结冠军运行 CrysLLMGen direct metrics 与 CHGNet S.U.N.；
4. WTB bridge 降为限时侧线，不启动训练。

WTB 的 permutation-safe identity 已经 256/256 通过，但 T 臂仍有 2/32
单斜晶格超出注册 chart。该问题不影响 WQ LLM 主线；在主线有空余后，最多再做
一次无训练的 periodic-beta chart 修复。

## 2. 首个实验

执行 identity：

`wq_llm_charge_stop_paired64_v1`

在同一个 epoch-3 WQ LLM 上，对 ordinals 1024--1087 运行同 seed 配对：

- baseline：现有有限 Wyckoff grammar；
- charge-stop：当前 composition 为 `charge_neutrality_fail` 且仍可增加合法
  Wyckoff orbit 时，暂时禁止 `STOP`。

该约束：

- 不修改已生成 token；
- 不重试、不替换、不 best-of、不 rerank；
- Pauling-only、all-metal、single-element、oxidation-state-missing 均为软项；
- atom-count support 耗尽时允许终止并把 invalid 保留在分母；
- 不调用 parent diffusion、MLIP、S.U.N.、MP API 或训练代码。

## 3. Paired-64 promotion gate

全部 64 对进入分母。只有同时满足下列条件才扩展到 3×256 proposal-only：

- masked generation success 不低于 baseline；
- composition-valid 至少增加 3/64；
- charge-neutrality failure 至少减少 3/64；
- baseline-valid → masked-invalid 不超过 1 对；
- unique-formula/all-attempt 下降不超过 5 pp；
- 成功样本平均原子数增加不超过 4。

FAIL 后不重跑该 identity，转而准备 formula-plan / chemistry-aware SFT；PASS
也不自动运行 parent、S.U.N. 或训练。

## 4. 精简审计规则

从本实验开始只保留四类不可替代记录：

1. frozen contract / execution patch identity；
2. submission claim 与完整 sbatch command；
3. attempt JSONL 与 terminal report；
4. 失败时的 Slurm/stdout/stderr。

不再为无科学影响的命令拼写、只读路径修正、重复状态查询创建独立侧车。完整
源码 hash 仍由一个累计 patch manifest 管理，资源继续满足
`1×A800 <= 8 CPU`。

## 5. 后续分支

- PASS：扩到 3 seeds × 256 proposal-only；再次 PASS 后才选冻结冠军进入
  parent + CrysLLMGen + S.U.N.
- FAIL：停止 STOP mask，使用 MP20 train-only formula-plan 数据做一次短 SFT
  设计；不把 held-out CHGNet/S.U.N. 回流训练。
- WTB：仅在不阻塞上述主线时修 periodic beta chart，先 32/32 mechanics，
  不自动训练。
