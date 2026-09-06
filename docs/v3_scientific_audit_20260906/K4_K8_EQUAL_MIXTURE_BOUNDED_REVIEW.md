# K4/K8 等权同态概率混合：唯一后备及最小探针

2026-09-07。只在 40074 两个固定 contact 候选均未满足原评价要求、且留有完整评价时间时使用。本文件不改变该作业，不增加训练、MLIP、contact 叠加、权重搜索或按 case 路由。

**裁决：值得做一次 8 前缀、16 次模型计算的小探针；目前没有理由宣称它提高稳定性。** 旧 K4 的 refined SUN 19/126、K8 的 raw SUN 11/66 只说明不同终点的既有表现，不能证明两个条件分布在同一状态互补。小探针可以关闭“混合是否真正定义正确、有实际分布差异”的缺口。如果数值通过，是否花剩余时间做完整 256 raw/refined 仍由 root 根据总时限决定。JS、熵、TV 不作为 SUN 质量门。

## 1. 唯一概率定义与实际能力边界

对同一状态 `s=(current,old,P,phase,active,transaction,attention)`，各自完成原 schema、alias 与 hard support 变换，取得同一 canonical 支持 A 上的 raw logits `l4,l8`。固定 T=.7：

`logpk = log_softmax(lk/.7)`；`logq = logaddexp(logp4,logp8)-log2`；交给原 sampler 的 raw logits 为 `.7 logq`。

因此 `q=(p4+p8)/2`。直接平均 raw logits 会得到几何平均式分布，属于另一策略。对每个固定状态有 `q≥pk/2`，所以 `KL(pk||q)≤log2`；这只是局部相对分布界。逐动作混合不是先掷一次硬币选整条 K4 或 K8 路径。两个生成器可以各自产生完整一致的模式，而动作混合会拼成未训练过的组合；旧终点各有优点不能排除这个风险。严格正支持下，给定长度 D 的尝试路径最多只能由局部界得到宽松的 `q(path)≥2^-D pk(path)`，并不构成有用的 SUN 保持保证。

原相同 CUDA Gumbel 数组下，若两个 component 都选择同一 token a，则混合也选择 a：将 `p4(a)/E_a≥p4(b)/E_b` 与 `p8(a)/E_a≥p8(b)/E_b` 相加即可。若二者 winner 不同，混合可能选第三个 token，例如概率向量 (.60,.39,.01) 与 (.01,.39,.60) 在相同 E 时混合选中间类别。这个性质只描述固定前缀的耦合，不说明晶体质量。

## 2. 最小运行接口

独立 sampler 在原 `processed_logits(x,old,positions,transaction_positions,attention_mask,phase=...)` 接口内，让两份冻结完整旧 policy 读取上述完全相同的值。两者各自复用原 `ProgrammedPathSampler.processed_logits`，不能一个读取 K4 自己的旧路径、另一个读取 K8 自己的旧路径。模型初始条件、token ABI、P、phase id、mask id、scaffold、合法约束与完整 logical batch 必须相同。

原 dtype 下先识别 `isfinite(v) & (v > finfo(v.dtype).min)`，再提取共同合法向量转 FP64。BF16 sentinel 不能在转 FP64 后按新 sentinel 重新判合法。两边支持必须逐 ID 一致；不取并集/交集，也不恢复 alias100。输出合法 `.7 logq`，非合法 FP64 sentinel。保留原 `_draw`，因而实际 token 与事件 `log_probability` 来自同一个 q；回放必须重新计算两模型混合。单 token 未改变不代表该 token 的 logp 未改变。

一份模型 NaN/Inf、不可用或支持不一致时，不退回另一份：探针直接失败；正式采样沿用原 unavailable → construction failure 或修订 rollback 的语义，增加 component 诊断，保留所有尝试、失败、rollback 和请求数量。不能让模型加载/ABI错误伪装成普通样本失败。P 和原阶段调度不变，原 seed、salt、vocabulary 随机数组长度以及 CUDA RNG 设备不变。

唯一机器规格是 [candidate_spec.json](evidence_v3_failure_20260907/k4k8_mixture/candidate_spec.json)。生产类由另一实现代理负责；本探针复用旧 component sampler，并对真实8向量在 CUDA 上调用生产 `mix_post_hard_vectors` 与独立公式核对≤1e-12。完整新 sampler 的运行/回放另由固定 pilot 验收，不混称为已验证。

## 3. 已实现的 8 前缀探针与接受门

[probe_same_state_mixture.py](evidence_v3_failure_20260907/k4k8_mixture/probe_same_state_mixture.py) 对 K4、K8 两份原 ledger 分别固定：sample 0 首个 construct cell 与首个 construct Z、sample 1 首个 cooperative Z、sample 2 首个 closure Z。总计 8 状态，各供两模型评分；第一组 cell 状态可能相同，仍保留来源绑定。选择只用 ordinal、阶段和标量类型，不读能量或 SUN。实际原记录三请求均为 batch=1；脚本由完整 256 ledger 重建原有效 layout=2/batch cap=4，并证明它们各是完整 singleton。原 metadata 缺 layout 时保持 null，不补写成旧记录已有2。缺任何固定状态即失败，不替换。

以下同时通过才写 `PROBE.json: status=PASS` 和 `_SUCCESS`：

- 原 source ledger SHA、完成标记、完整训练 policy 与 tokenizer/scaffold/输出 ABI 绑定；原完整 logical batch 成员可重建。
- 两份模型对全部 8 状态读取同一 current/old/P/phase/transaction，实际计数各8，总16；无新轨迹、训练或能量计算。
- 每个状态的来源 component 同 seed/salt 重现原 CUDA token，原所选动作 logp 误差≤1e-6。
- 两边有效支持相同非空、全部合法数值有限；归一化、算术概率公式、dominance 与 trace logp 误差≤1e-12；局部 KL≤log2。
- 生产 helper 对同向量在CUDA输出为FP64，非法位逐值保持 sentinel，合法 raw logits 与独立参考计算最大误差≤1e-12。
- 原相同 CUDA Gumbel 下共同 component winner 的保持无违例。真实结果保存两个原 dtype 全词表向量、合法 IDs、三者 logits/logp、source、state、seed/salt 和 ABI/源代码 hash。

`PROBE.json` 独立报告 `distribution_difference_above_numerical_floor_1e_8`、`maximum_tv_components`、各 winner 差异及 `distribution_difference_is_stability_gate=false`。全为相同分布意味着这几态缺少进一步试验动机；有差异仍只是有必要的作用空间，不是稳定性收益证据。没有额外调阈或重选前缀。

精确入口（`CODE` 为已上传冻结 checkout；`PREFIX_MANIFEST` 为已有 k4k8_prefix_inputs.json；`RAW_MODEL` 为原 base 路径；`OUT` 必须尚不存在）：

```text
python CODE/docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/k4k8_mixture/probe_same_state_mixture.py --source-root CODE --input-manifest PREFIX_MANIFEST --model-path RAW_MODEL --output-dir OUT --device cuda:0
```

输出 `PROBE.json`、`PREFIX_METRICS.json`、`STATE_BINDINGS.json`、`full_original_dtype_vectors.pt` 和成功标记。顺序加载 K4、释放后再加载 K8，严格16次 forward；只需一张可加载原 policy 的 GPU。实际计量两次加载耗时、评分/回放耗时和显存峰值，不事先承诺分钟数。双模型完整生成大致是旧路径两倍模型计算，但新轨迹的失败/rollback 会改变实际步数；加载和同一后处理成本另计，原独立 refiner 评价不能省略。

本地 CPU 已完成8项验收，含生产 helper 对独立公式的一致性、原 helper 的100组随机耦合、40次共同 winner 均保留，固定8真实前缀也已从旧 ledger 重建，见 [CPU_CHECKS.json](evidence_v3_failure_20260907/k4k8_mixture/cpu_acceptance/CPU_CHECKS.json)。真实 8B、CUDA、双模型 pilot 与 SUN 仍未在本代理运行；本次没有提交任何作业。
