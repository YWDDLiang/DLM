# 历史证据对下一版 DLM 设计与重训决策的约束

本文件只读审查已有本地报告。未修改代码、训练目标、GPU 作业、调度或 Git 提交；未读取远端状态。K8 最终学生 SUN 和当前 raw 初始化模型的终态效果不在本次已核实证据中，因此不能写成已知结果。下文明确区分实测、解释和下一轮待验假设。

审查结论：支持继续保留化学 Planner、精确组成、canonical `7+4N`、实际合法支持和可回滚的非因果修订。历史失败较一致地指向「模型见到什么状态、目标是否能从该状态实现、监督是否包含近平衡/低能支持、学生是否学到部署分布」；没有证据支持仅凭 loss 分三类或样本/权重改为 `2:1` 就断言下一版合理，也没有证据要求再从头重训一遍整个系统。若本次终态仍缺少真实自身错误上的修复能力，下一次参数训练应明确补那个能力；其数据与目标改变需要训练，但是否从当前 raw 模型续训，须由本次终态诊断决定。

## 1. 先确认正在评审哪一版

当前 [RAW_LLADA_DESIGN.md:3](../RAW_LLADA_DESIGN.md#L3) 明确覆盖旧 K4/K8 warm-start 方案：只加载原始 LLaDA-8B-Instruct/tokenizer，晶体输入/输出行增量和 LoRA 为新训练参数。因此 [DESIGN_AND_AUDIT.md](../DESIGN_AND_AUDIT.md) 中 K4/K8 终态修复数据、旧 LoRA 初始化和参考 KL 不能再当作当前 raw 模型已使用的训练机制。

当前文档实际定义的是：

- MP20 teacher-only 数据，train/validation 为 `27136/9047` 个 source；每个 source 每 epoch 有一个随机 mask 构造视图和一个 scalar full-cell repair 视图。[当前数据定义](../RAW_LLADA_DESIGN.md#L9)
- `family-balanced schema denoising CE + 0.25 × actual-legal-support CE`；各项内部 `90% exact CE + 10% numerical CE`，数值带宽从 `0.75` 到 `0.25` bins；schema/部署概率温度分别为 `1/0.7`，目标不在合法支持时记录冲突并使合法项为零。[当前目标定义](../RAW_LLADA_DESIGN.md#L11)
- 修复输入是结构化合成扰动及正确的 clean prefix；自提升已暂停，没有当前模型自身生成错误或 K4/K8 反馈进入当前训练。[输入状态](../RAW_LLADA_DESIGN.md#L7)、[范围边界](../RAW_LLADA_DESIGN.md#L19)

这些是设计定义，不是效果证据。上述多组系数不能简化为「三类 loss，按 2:1 配比」。下一版评审首先要固定每项的目标、采样分布、归一化分母和实际代码公式。

## 2. 应保留的实证资产及其边界

| 已核实证据 | 支持保留什么 | 不支持什么 |
|---|---|---|
| Fused Planner 在一个无 MP20/开发集 exact-composition 重合的 prospective 队列上 `256/256` composition-valid，无重试/替换；同一报告固定所有结果分母。[prospective 报告:13](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/36H_FINAL_REPORT_C3FD_G2_20260901.md#L13)、[队列:22](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/36H_FINAL_REPORT_C3FD_G2_20260901.md#L22) | C3FD 可达支持与 typed Llama 在同一合法集上的偏好，精确化学与几何执行的职责分离。 | composition-valid 不能代替 structural-valid、稳定性或 SUN；这也不是独立多 seed 的鲁棒性结论。 |
| 同一固定 Plan、两 stream 的 B0/BC/BP/BR/BS raw Direct 为 `80.27/99.22/98.05/98.83/99.80%`；BC 相对 B0 的 candidate-only/control-only 配对为 `48/0` 与 `51/2`。[06:199–218](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md#L199) | 精确组成预填、事务化条件生成、PBC 合法支持和可修订 runtime 是真实有效的资产。 | 不能把全部增益归功于 pointer；BP 单独的 Direct 比 BC 更低。整套流程有效不等于每个历史阶段都必须永久保留。 |
| Pointer 在 MP20 validation 上 exact permutation/root/pairwise-order 为 `73.50/80.41/82.63%`，canonical exact/root 为 `14.48/19.22%`；部署队列 `229/256` programs 非 canonical。[06:187–195](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md#L187) | Llama 学到了可执行的结构程序，程序通过语义/位置编译接 DLM，不需要 AR/DLM token ID 强行混用。 | Teacher 是 MP20 contact-tree 顺序，不是能量排序；没有独立证明 learned order 降低 A/B 或提高 SUN。 |
| 真 checkpoint 中，后来站点 Z 改一个 bin，使较早 masked anchor 的 X/Y/Z logits 最大变化 `0.125/0.09375/0.15625`。[06:143–146](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md#L143) | 双向 Transformer 确实能利用未来可见上下文做非因果修订；显式旧周期几何值得作为输入机制保留。 | logit 对上下文敏感只证明信息通道存在，不证明敏感方向物理正确或能改善最终结构。 |
| 现有架构审计中 C3FD 只管化学 ledger/支持和 coarse Plan，DLM 负责六晶格与 XYZ，CHGNet 只离线标注/共同比较，model494 单列。[19:23–33](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/19_RESUMED_ARCHITECTURE_AND_EXECUTION.md#L23) | 保持可解释职责边界；下一版评估冻结 Planner 分布，才能把改进归因于几何执行。 | 把 Planner 称为 fine-geometry logits 模型，或用换组成分布掩盖几何失败。 |

合法集合的归纳证明应只声称保持所声明的有限支持：精确组成、数值 token 类型、非退化晶格及规定周期像集合下的距离条件。它不证明低能、近平衡，亦不自动证明任意 unreduced 晶格的全局最近周期像；当前新设计也明确保留该边界。[新设计的支持声明](../DESIGN_AND_AUDIT.md#L43)、[有限周期像边界](../RAW_LLADA_DESIGN.md#L17)

## 3. 失败更像目标/状态错配，不能简化为 LR 或分辨率

### 3.1 正确 MP20 后缀未必是自身错误状态的可执行目标

固定 `128` 个真实 exact-axis rollout 的 oracle continuation：BASE Direct `59/128`，用 MP20 填剩余 mask 的 lattice/X/Y/Z 阶段结果为 `122/102/54/43`；Z 阶段 `19` 个 valid→invalid、只有 `3` 个 invalid→valid。按 canonical site order 重排 target 后仍为 `122/95/57/43`。这发生在训练前，表明后续 MP20 坐标与既有错误前缀不相容；改变 LR、epoch 或例数比例不会使这个错误配对自动成立。[rollout checklist:91–115](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/ROLLOUT_MATCHED_DLM_24H_CHECKLIST_V1.md#L91)

这不是反对 MP20，也不是反对全胞重建。完整修订允许一起改变不相容的 lattice/XYZ，比「保留错误前缀、监督孤立后缀」更有理由；但 full-cell teacher forcing 仍需检验自身采样 prefix 上的 rollout transfer，不能只报告 clean-prefix CE。

### 3.2 合成状态 NLL 进步已多次未转移到真实采样

B3 的 frozen-panel token NLL：synthetic safe-axis `2.1251216630→1.7615879962`，actual B0 rollout `4.6503425199→4.8122329398`；IID、D1 也下降。四个固定 Plan 进程复现的 Strict SUN 变化均为负，分别 `-2/-5/-5/-6`，层次配对区间为 `[-3.22265625,-0.390625]` percentage points。[B3 原始结果:59–70](D:/codex_work/ai4s/diffsion_language_model_meets_diffusion/workstreams/final_method_development_20260808/experiments/dlm_b3/RESULT.md#L59)、[重复结果:118–132](D:/codex_work/ai4s/diffsion_language_model_meets_diffusion/workstreams/final_method_development_20260808/experiments/dlm_b3/RESULT.md#L118)

因此当前 raw 模型的合成 repair 是有待真实自身错误验证的能力假设，不能从「有 repair 数据」直接得出会自修复。四个 repeat 复用同一固定 Plan 队列，不能说是四个独立训练 seed。

### 3.3 对齐序列语义能改善 Direct，但不足以解决低能

旧 MP20 teacher 与部署 hard-prefill 的元素槽位 exact-order 匹配仅 train `4069/27136`、validation `1296/9047`。canonical 重排不丢 source、组成与物理结构不变。固定 prospective 队列 raw Direct 由 OLD-G2 `128` 到 canonical-DLM `143`；但 `249` 个共同能量配对均值 `+22.95 meV/atom`，中位数 `-0.30 meV/atom`，区间跨零。canonical-DLM 加 G2 后 Direct 为 `133`，仍无能量优势。[canonical 审计:140–187](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/ROLLOUT_MATCHED_DLM_24H_CHECKLIST_V1.md#L140)

这个结果同时支持保留 canonical 语义，以及避免把可用的旧模块机械叠加进下一版。目标语义先正确，再考察物理目标。

### 3.4 分辨率不是当前全部失败的解释，局部量化损失仍真实存在

- G0 的 `256` 请求中 parse `8`、PBC `<0.5 Å` collision `142`、pass `106`，其余主要失效类为零；`248/248` CIF-bearing rows exact token round-trip 没有 Direct 翻转。[G0:243–252](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/DLM_POST_FM_STRUCTURAL_LEARNING_AND_REFINER_FEEDBACK_PLAN_V2.md#L243) 这是该集合的执行/codec 证据，不能当作所有连续 MP20 结构的能量上限。
- 后来的独立 Phase 0 有 `9047/9047` exact codec round-trip；固定 `512` 个连续/量化能量对全部可评估，量化能差中位数 `+2.846 meV/atom`、q95 `+19.031 meV/atom`，Direct `458→458`，Meta proxy retention `505/512`。Strict exact-zero proxy `25/207` 只作诊断，少量 meV 已会翻动零阈值；proxy 不是 SUN。[Phase 0:34–39](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/09_EFFICIENCY_FIRST_POTENTIAL_CLOSURE_PLAN.md#L34)
- Force teacher 的连续一步 `76.8%` 降能，精确量化后只 `54.3%`；near-threshold `59/64` 状态即使降能仍 valid→invalid。修订 teacher 后，最终候选能通过预先定义的机制门槛，但 microstudent 在 `128` holdout 上 Direct flips 为 `6` 正/`6` 负，净零。[force 前检:41–46](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/FORCE_SCORE_DLM_CHECKLIST_V1.md#L41)、[有效候选:103–108](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/FORCE_SCORE_DLM_CHECKLIST_V1.md#L103)、[student 终态:65–66](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/ROLLOUT_MATCHED_DLM_24H_CHECKLIST_V1.md#L65)

解释：目前没有必须扩词表/减 bin 的普遍证据。若下一版追求原生极低力/应力或 Strict 零阈值，应先测同一新目标的 `R(x)→Q(R(x))→R(Q(R(x)))`，分别报告能差、force/stress 和盆地变化；不能用旧 codec 有效性检查替代，也不能把 force 候选成功当学生成功。

### 3.5 梯度有限、甚至尺度接近，仍不能说明目标有效

Potential-Closure train-only probe 中 cell/CE、site/CE 梯度 norm 比值为 `0.109/0.160`，cosine 为 `0.0058/0.0572`，全部有限/非零；最终原生能量平均差 `-0.236 eV/atom` 的区间为 `[-0.599,+0.131]`，仍跨零。[probe 与结果:55–83](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/09_EFFICIENCY_FIRST_POTENTIAL_CLOSURE_PLAN.md#L55)

K10 full-data probe 的 scaled posterior norm `32.55`、clean norm `39.37`、median cosine `0.034`。追加一遍同目标训练后，另一个新 prospective stream 的 raw Strict/Meta SUN 为 `5/49`、tau800 为 `14/123`，未达到登记目标；与上一 stream 不同队列，不能把跨 stream 数值差当严格配对训练效应，但足以说明「梯度健康 + 再一遍」尚未建立稳定原生提升。[K10 probe:59–74](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/13_STREAM19_DIAGNOSIS_AND_FINAL_ITERATION.md#L59)、[末轮:88–112](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/13_STREAM19_DIAGNOSIS_AND_FINAL_ITERATION.md#L88)

PMTR 的非有限梯度发生在第一个 optimizer update 之前，它是实现失败，不是物理机制的学生负结果。[工程失败边界:416–443](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/GPT6_AUDIT_HANDOFF_20260905.md#L416) 下一版仍应检查数值和梯度，但不能把「有梯度」或「观察了 norm」写成权重校准成功。

## 4. K4 本轮究竟支持什么

全部比例以 `256` 请求为分母，失败保留。native 参考/方法 Strict SUN `6→7`，Meta SUN `55→57`；共同终态完整验证只有 `37` 对，平均 `ΔA=+0.125768`、`ΔB=-0.055748 eV/atom`、实际弛豫步数 `+12.594595`。tau800 Strict `19→19`、Meta `121→126`；`63` 个验证交集的 `ΔA=+0.00626629`、`ΔB=+0.0254841`，均上升。[25:13–58](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/25_ROUND0_MATCHED_EVALUATION.md#L13)、[tau800:64–76](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/25_ROUND0_MATCHED_EVALUATION.md#L64)

同一 `37` 对的 A 分解为 `Δe0=+0.0700202633` 减去 `ΔeR=-0.0557481824`，得到 `+0.1257684456 eV/atom`。降低 B 本身会扩大 A，除非原生能量跟着下降；这只是恒等式，不能当作主动进入更深盆地的因果证明。[26:9–25](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/26_ROUND0_GAP_INCREASE_DIAGNOSIS.md#L9)

过密实例很直接：In6Yb2 的 VPA `46.4117→12.7358 Å³`，最小周期距离 `2.1242→1.4250 Å`，原生能量 `0.0024→9.4845 eV/atom`，终态 `-2.7885→-2.7822`，A 增加 `9.4758`。前三个 A 大增样本在协同前已分别相差 `7/5/9` 个构造 token，均含 `2` 个晶格 token，所以不能单独归咎协同事务。[几何实例:40–60](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/26_ROUND0_GAP_INCREASE_DIAGNOSIS.md#L40)

这与旧 SPAD 分析一致：生成结构 force RMS 中位数约 `22 eV/Å`，量化 MP20 约 `0.176`；stress norm 约 `100` 对 `1.85 GPa`。合法动作集明显大于近平衡域。[旧物理差距:26–32](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/12_LLAMA_PROGRAMMED_BASIN_CLOSURE.md#L26)

K4 的数据/teacher 限制同样有实测支持：

- `1024` 组成中 `526` 有验证路径，只有 `273` 至少两条，`253` 为单验证路径；mean TV(q,u) 为 `0.025613`，teacher A/B 均值收益各约 `22.47 meV/atom`。那 `37` 个评估组成与能量训练条件按完整计数或约化比例均无重合，测的是跨组成迁移。[26:71–90](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/26_ROUND0_GAP_INCREASE_DIAGNOSIS.md#L71)
- `972` verified 路径中，等组成均匀 A `6.538571`，teacher A `6.516099 eV/atom`；忽略 B/KL、每组选最小 A 的池内下界仍为 `5.484187`。仅 `13/526` 组成存在 `A≤0.1` 候选。这个下界只限已有 verified 候选池，不是模型空间不可行证明。[27:9–29](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/27_SELF_IMPROVEMENT_REPAIR_PLAN.md#L9)
- 原 teacher 允许 `98` 组 A 均值退步、`38` 组 B 均值退步。把任一目标退步的 `136` 组恢复均匀 u，得到只读诊断 `ΔA=-50.851 meV`、`ΔB=-3.706 meV`、KL `0.002943`；这不是新最优解、未用于训练、更不是学生结果。[27:21–49](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/27_SELF_IMPROVEMENT_REPAIR_PLAN.md#L21)

因此需要补「好候选支持/同起点有效修订」和学生迁移，而不只是更强重加权。K8 增加同一组成的可比候选，不增加组成覆盖。teacher KL 约束的是经验 q，相对于 verified-path 均匀 u；它不是 student 相对采样策略的 KL 上限。[27:28–31](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/27_SELF_IMPROVEMENT_REPAIR_PLAN.md#L28)、[20:119–145](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/20_DATA_SUFFICIENCY_AND_DELIVERY_20260906.md#L119)

## 5. 为什么「三类 loss / 2:1」不能直接成为结论

1. **分类不决定统计目标。** 将 length/angle/coordinate 归为三族，是避免坐标数随 N 增长而支配平均；并未证明晶格与坐标对力、应力、低能盆地的贡献相等。把构造、修复、稳定偏好分为任务也一样：任务名不说明监督的条件分布和目标是否合法可达。
2. **例数比例不等于梯度比例。** dense random-mask 与单 scalar 状态的监督位数不同；family 平均、source/组成平均、路径长度、抽样概率修正、合法目标失配而置零、padding、梯度裁剪和 optimizer 状态都会改变有效贡献。应报告每项加权前后 norm、cosine、实际更新占比及有效 source/状态覆盖，再讨论一个固定比例是否失衡。该工程检查仍不等于物理收益证明。
3. **CE 是数据条件分布目标，不自动是稳定性目标。** 三项 CE 即使都下降，也可能只更好地拟合合成/teacher-prefix 状态。历史 B3、oracle continuation 与 ordinary CE 的执行/稳定性分歧已经给出反例。后者从总两 epoch 到三 epoch body `985→992`、Direct `871→878`，Strict SUN `81→79`、Meta `489→477`。[CE 历史:88–96](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/DLM_SUN_STABILITY_MECHANISM_DEEP_DIVE_V2.md#L88)
4. **良好 teacher 与良好 student 必须分开。** 无低 A 支持时，再加权不产生新支持；有候选改进时，soft CE/NLL 仍可能学不全、多模态混合或跨组成失效。`L(q)=L(u)+L(q-u)` 中偏好部分较弱，u 的拟合、共享参数和原 CE 可以稀释其效果；且旧 q/u 都已条件化在 verified 子集。[20:131–145](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/20_DATA_SUFFICIENCY_AND_DELIVERY_20260906.md#L131)
5. **重新命名不能解决 A/B 耦合。** 即使另加 force/stress 或几何项，也需给出作用状态、单位/归一化、与 CE 的梯度关系，以及合法支持和量化后收益。force、stress、A、B 不可互相代替；所有目标是否取均值约束或逐条件约束必须写明。

建议把任意 `2:1` 当作尚待检验的固定初始设计，而不是已被实验支持的优选。最终方案应注明「这个比例试图平衡什么」与「哪些日志/终态结果会否定它」，不按本轮开发 SUN 做比例搜索。

## 6. 决定下一版是否必要的本次终态诊断

这些是报告需求，不授权修改正在运行的训练或新增 GPU 实验。优先从既有终态、训练日志和保存的 phase/attempt trace 计算；缺失则在最终报告明确列出。

| 必须区分的问题 | 本次需要的证据 | 对下一版的含义 |
|---|---|---|
| raw 基础训练是否真的学到了当前表征/任务 | 全 source/view 覆盖；按 length/angle/coordinate、construction/repair 分的 heldout schema/实际合法 CE；support conflict 与置零率；输入/输出新行、LoRA、数值模块、geometry bias 的梯度及 reload/replay 一致性。 | 若失败在表征行、mask/alias/target mapping 或 loss 归一化，先修实现；不能据此宣告需要新物理架构。 |
| 合成恢复能否迁移到模型自身错误 | 同一个自身生成起点的 repair 前后对比，保留失败、unchanged、rollbacks；分构造终点、修订终点报告体积/最小距离/能量/force/stress。不能只用 admitted-target 或 clean-prefix CE 作分母。 | 若合成 heldout 很好而自身错误 repair 无效/恶化，优先定义与真实输入匹配的训练目标；不能只增加同类 synthetic epoch。 |
| K8 是否真正补强了自提升信号 | 完整 K4/K8 的 verified 覆盖、多验证路径/多终态组成、低 A 候选分布、共同改善支持、TV/KL/ESS、逐条件 A/B 退步数和收益集中度；学生实际偏好变化另报。 | 支持仍不足时先补有效候选/同起点修订，不是调 loss 比例；teacher 好但 student 不学时才定位拟合、漂移和部署状态错配。 |
| 改进来自原生近平衡还是公共后处理 | 固定 Plan/program/组成/seed 的 raw 与 tau800 分开；e0/F/stress 对有限交集，A/B/步数对双方 verified 交集并报告排除数；同一交集做 `ΔA=Δe0−ΔeR`。看均值、中位数、尾部和 paired win/loss。 | 只有 tau800 或 B 改善而 e0/A 恶化，不足以说原生修复已经成功。 |
| 集合有效与 SUN 是否保住 | 全请求分母的构造/完整验证/Stable/N/U/Strict SUN/Meta SUN、全部标签状态、未知 hull；同一队列配对命中变化。新队列只作独立结果，不与旧 stream 直接构成因果差值。 | 如果原生指标和目标均可靠达到，没有仅为三类 loss 而立即重训的理由；若未达，按上述定位选择最小必要目标改变。 |
| 数值分辨率是否真限制本次候选 | 若有新的低残差目标，检查量化前后及重新弛豫后的能量、force/stress、几何与盆地；不把旧 codec 通过当作该目标的验证。 | 仅当这一步给出实质上限，才有证据考虑连续残差/更细量化；目前历史证据不要求普遍扩词表。 |

若下一版新增真实自身错误、自身完整修订轨迹、可靠终态反馈或 student/reference 约束，就需要新的参数优化；更换推理开关不能让当前 raw checkpoint 自动获得未监督过的能力。现有证据没有证明必须丢弃本次 raw 训练结果重开底座，也没有理由重训已经验证可用的化学 Planner。实际下一版应在本次唯一终态报告之后，固定一个能解释剩余错误的训练对象，并分别验收「候选更好→teacher 偏好正确→student 学到→独立原生/系统 SUN」；前一项数字不能替代后一项。

## 7. Crystalite 启发下，保留整体 LLM+DLM 的具体接口

Crystalite v2 的 GEM 将当前 noisy coordinates/lattice 的周期最小像几何变为 attention 的加性 bias，组合距离项与 learned edge，并用随噪声变化的 head-wise gate 调节。因此可迁移的重点是「周期几何直接改变 Transformer 内部交互」，不是给文本多加几个几何词。该论文处理连续 noisy 几何，不能把其始终可用的数值状态假设直接复制到本项目的 MASK canvas。[Crystalite v2 §3.4](https://arxiv.org/html/2604.02270v2#S3.SS4)

### 7.1 上游可用信号与真值边界

| 模块/字段 | 部署时真实可用的内容 | 新版应如何使用 |
|---|---|---|
| C³FD | 当前 typed 化学 ledger、满足已声明原子数/计数/价态族/可达规则的动作支持、基础动作分布。 | 化学硬支持；证书是对所建规则的可执行性，不是该晶体的能量/真实氧化态证明。它没有精细晶格或 XYZ logits。 |
| Typed Llama Planner | 在同一化学合法支持内的残差偏好；已采样 N、元素与计数；预测的 lattice-system、SG-bucket、VPA-bin。 | Llama 决定组成与软计划分布；DLM 接收语义条件，不把 AR token ID 或 AR logits 冒充数值几何。保持 Planner 队列冻结以判定 DLM 的贡献。 |
| Species pointer/program | 由 terminal Llama hidden、已确定元素/计数和 soft Plan 预测的物种排列；编译为 canonical slots、分组/anchor/修订顺序。 | 可执行离散程序。MP20 contact-tree 只在训练时给排列 label；推理没有目标接触图、真实配位数、最低能原子顺序或目标坐标。 |
| `N/elements/counts` | 上游已生成的精确组成条件。 | 硬预填，修订不改变它；若改变元素则已换成另一个科学问题，不能继续称固定组成修复。 |
| `lattice_system/spacegroup_bucket/volume_per_atom_bin` | Planner 的预测/抽样值，而非将生成晶体的已知真值。 | 软先验或可 mask 的条件；不能强制真实 cell 长度/角度遵守单一 SG 映射。预测先验与动态实际几何应是不同通道。 |
| Wyckoff occupancy、具体配位图/配位数 | 当前 Compact-V2 schema 没有这些字段；它们也不是 species permutation 的隐含输出。 | 默认未知。若未来由模型预测，须带来源/可缺失状态，仍是未经验证的软提议；若由用户给定精确结构约束，则另行定义相容性与可达性，不能从 SG bucket 自动推出。 |
| 当前/旧晶体、task、mask、noise | 已提交整数 token 与实际旧几何、哪些位置可见/活动；controlled corruption 才有已知噪声，真实生成错误通常未知。 | GEM-like 特征只能读这些当前可观测量。未来 target geometry、R(x)、energy/force 或目标配位不能作为部署输入。 |

模块职责来源：[19:23–37](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/19_RESUMED_ARCHITECTURE_AND_EXECUTION.md#L23)、[pointer teacher:101–109](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/teacher_feedback_unified_v1/06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md#L101)。当前 schema 仅将 lattice/SG/VPA 列为 soft fields，prompt 也明确该边界：[c3fd_native_plan.py:12](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/c3fd_native_plan.py:12)、[149](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/c3fd_native_plan.py:149)。

不能将 metric lattice 与 SG 当作一对一物理真值。历史 rich-field 审计中，原标签的一对一 metric/SG map agreement 仅 `42.60%`，而旧 compiler「内部一致」并不代表符合标签；单独监督的 SG head accuracy 为 `60.61/60.66%`，旧 compiler 只有 `26.78/27.50%`。这些数字是旧 checkpoint 的 validation 审计，不是当前 Planner 的性能估计，但足以否定那个一对一规则。primitive cell 的 metric 类与 conventional SG 元数据本来就是不同语义。[rich-field 审计:23–65](D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/docs/C3FD_RICH_FIELD_SEMANTICS_AUDIT_V2.md#L23)

### 7.2 当前 bias 已进入交互，但几何覆盖不足必须单列

代码确实将 `attention_bias` 作为显式 forward 参数传入底层 Transformer，因此不能写成「现在只加 embedding」。当前实现是所有层复用的单个 shared bias；pair 只有在旧 lattice 与两个旧 site 完整时才有效，输出 shape 的 head 维为单一广播维。prompt 和六个 lattice slots 不在 `token_sites` 的有效区间，所以没有直接 pair bias；修复时 pair geometry 由显式旧状态读取。[periodic_repair_model.py:58](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_repair_model.py:58)、[95–158](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_repair_model.py:95)、[227–244](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_repair_model.py:227)

由此可作结构性推断：all-mask construction 没有可计算的真实 pair；一个尚未完整的 query site 对应 bias 行也被关闭。完整可见站点之间仍可经多层普通 attention 间接影响它，不能说所有几何通道完全失效。应报告不同构造/修复阶段的有效 pair/query 覆盖、bias 非零比例与量级、对 active logits 的实际影响，再判断 GEM-like 机制是否在决定待采样坐标时工作；不能用「有 attention bias 类」替代这项诊断。

### 7.3 下一版连接方式的审查建议，尚非已验证实现

保留 `C³FD 支持 → Llama 软计划/物种程序 → DLM 条件分布 → 合法支持采样/整体验收`。GEM-like 部分放在 DLM 内：精确 species、program rank 和软 Plan 属于条件通道；实际周期几何属于动态状态通道；最后仍由 DLM 输出各合法数值 token 的概率。两类通道应共同影响交互，但不能把软 Plan 编译成虚构距离或目标对称性。

如果当前终态诊断证实构造时几何通道长期关闭，下一版需要明确选择可观测、可训练的状态形成机制，例如在构造获得完整提议后重点训练其全胞重建，或在同一随机状态定义下给未知几何提供明确的联合 noisy numerical state。后者会改变 mask/noise 过程，不能在推理临时补一个 softmax 坐标平均值就宣称已复现 Crystalite；尤其本项目历史上已经出现几何均值/残差模块叠加后 Direct 退步的结果（本报告 §3.3）。

几何模块可考虑区分 head/层的交互偏好，以及 lattice↔site 的显式全局通道，并在修订过程中区分「固定错误起点」与「当前已接受/已生成几何」的语义。它们是待检验的架构选择，不是论文模块名带来的自动收益；SG/Wyckoff 不应被用来填补缺失的动态几何。任何新增通道都须与训练状态、实际修订因子分解及概率回放同步，并沿用本报告 §6 的原生物理/SUN 验收。保留 LLM+DLM 核心不要求保留所有旧修订阶段，也不等于把 Crystalite 当成第二个生成器接在末端。
