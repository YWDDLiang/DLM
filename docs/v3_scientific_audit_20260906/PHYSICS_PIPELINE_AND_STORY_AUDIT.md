# 物理评价、LLM→DLM 衔接与论文主张：首轮科学审计

审查日期：2026-09-06。被审执行代码固定为 `2e904c260bafb6750c9a5dbb0d920c4ff8c3a868`。V2 的正式训练已经开始；本报告不改变它的模型、数据、阈值、采样、评价或调度，也不启动任何新 trick、GNN、V3、K4/K8、自教师或连续联合训练。冻结代码和作者评价代码的只读副本、版本与 SHA256 见 [来源台账](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/SOURCE_MANIFEST.json)。

**结论：当前最坚实的贡献是可执行化学支持和真实的 Llama 程序控制；原生低能近平衡生成仍未被证明。失败不能归结为单一 loss、学习率或“DLM 不适合晶体”。** 已观察到的生成偏差、教师支持不足、软条件语义、公共弛豫停止标准和评价覆盖问题处于不同位置。必须先分别归因，否则可能把缓存错误当模型崩溃，把 refiner 的收益当 DLM 学会稳定，把局部几何修复当更低能盆地发现。

本报告将证据分成三类：原始汇总可复算的结果；冻结代码或 CPU 反例能确定的机制边界；尚待逐结构产物验证的主因假设。第三类不写成已查明的唯一原因。

## 1. 首先撤销错误的“主实验崩溃”前提

K8 独立主实验经官方 hull 缓存覆盖修复后，条件 1000 的 native Strict/Meta SUN 为 **39/244**，tau800 为 **65/484**；全部 1200 请求为 **47/285** 与 **79/570**。原先极低的 `0/9`、`1/17` 来自把开发 256 的小缓存用于新组成，没有完整覆盖所需化学体系。修复复用了生成与物理标签，没有重采样；不能再用旧计数证明模型整体性能崩塌。[原始修复与评价汇总](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence/EVALUATION_SUMMARIES_20260906.json)、[独立统计复算](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/SUMMARY_RECALCULATION.json:36)

修复后仍未达到目标，但这是另一个事实：全部 1200 中 hull 已知 **1172**、官方明确 unresolved **20**、不可重建 **8**。即使那 20 个 unresolved 全部有利于 SUN，tau800 的宽松上界也只有 Strict `99/1200=8.25%`、Meta `590/1200=49.17%`。因此剩余未知本身不能救回 10%/50% 的全请求目标。这个上界故意忽略未知样本可能不满足 N/U，是保守的最有利边界，不是填补标签。[复算:119](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/SUMMARY_RECALCULATION.json:119)

冻结评价器会把没有覆盖但也不在 unresolved 清单的体系记为 `official_cache_not_covered`，最后仍写评价 `_SUCCESS`。这能表示完成记账，却不能表示完成参考覆盖。未来评价入口的最小改进应是先检查 `required_chemsys ⊆ covered ∪ explicit_unresolved`，将意外的 not-covered 与物理失败分开；这不需要改变任何稳定阈值。当前 V2 开发评测仍用既有固定队列，不能未经授权换缓存或重命名已冻结结果。[冻结评价源码:137](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/frozen/scripts/evaluate_programmed_paths.py.txt:137)

## 2. 实际数据揭示的是两种不同瓶颈

下表来自已下载的七份原始 JSON 汇总，保留远端路径和原始 bytes SHA。这里的“输入”是公共 CHGNet 弛豫前的 native 或 tau800 晶体；“verified”在公共弛豫后判断。不同队列之间不构成模型的配对因果比较。

| 队列 / 输入端点 | 请求 | 可重建 | 输入 N∩U | 公共终态 verified | Strict / Meta stable | Strict / Meta SUN | Verified Strict / Meta SUN |
|---|---:|---:|---:|---:|---:|---:|---:|
| K8 独立 native | 1200 | 1192 | 1188 | 294 | 47 / 288 | 47 / 285 | 19 / 136 |
| 同队列 K8 tau800 | 1200 | 1192 | 1025 | 428 | 113 / 731 | 79 / 570 | 36 / 224 |
| raw-v1 开发 native | 256 | 255 | 253 | 60 | 4 / 47 | 3 / 45 | 0 / 20 |
| 同队列 raw-v1 tau800 | 256 | 255 | 219 | 91 | 23 / 146 | 16 / 113 | 8 / 40 |

来源：[K8 native](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/SUMMARY_RECALCULATION.json:36)、[K8 tau800](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/SUMMARY_RECALCULATION.json:119)、[raw-v1 native](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/SUMMARY_RECALCULATION.json:201)、[raw-v1 tau800](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/SUMMARY_RECALCULATION.json:266)。

**原生结构明显远离近平衡。** K8 独立 native 输入的最大原子力中位数为 `40.497 eV/Å`、最大应力分量中位数 `87.920 GPa`；进入 tau800 后分别是 `0.414`、`2.857`。两端中位数之比约 `97.7` 和 `30.8`，不是“每个样本都改善这么多”，但同队列的分布变化清楚显示 model494 的实际作用。原生 verified 子集 A 均值仍约 `6.523 eV/atom`；tau800 verified 子集约 `0.0554`，二者子集不同，不能当作逐样本 ΔA。[同源复算](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/SUMMARY_RECALCULATION.json:813)

**refiner 同时提高 S、损失部分 N/U。** K8 的 S 增加 Strict 66、Meta 443，SUN 净增加 32、285，而输入 N∩U 减少 163。tau800 中有 34 个 Strict-stable 和 161 个 Meta-stable 因不同时满足 N/U 而没有进入 SUN。因此“refiner 只洗掉信号”和“refiner 已解决所有问题”都不成立。不能用 aggregate stable 增益替代发现率，也不能把 N/U 损失误当全部来自重复——需要逐标记分解 novelty 与 unique。[同源复算:813](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/SUMMARY_RECALCULATION.json:813)

raw-v1 的 native headline Strict SUN 为 3，但 verified Strict SUN 为 0。这不是把 3 改判为 0；它说明两种标签的物理含义不同，不能写成“已发现 3 个经完整验证的原生稳定极小值”。

## 3. 真正被测量的对象，以及 A/B 不可互相替代的原因

令 `c` 为精确组成，`H` 为预测软计划，`P` 为 Llama 物种程序，`x` 为 DLM 完整构造/修复后的晶体，`F800` 为固定 model494 映射，`R` 为公共 CHGNet 变胞弛豫。当前流程实际是：

```text
c,H,P → DLM construction → DLM full-cell repair → x
         native: y=x                 tau800: y=F800(x)
         原生/输入物理量 e(y), force(y), stress(y)
         公共 R(y) → terminal eR、停止状态、验证状态
         N(y), U(y) 与 eR−h(c) 的稳定阈值求交
```

因此 native SUN 的“native”只排除了 model494，并未排除公共 `R`。当前 headline 是

`N(y) ∩ U(y) ∩ [e(R(y))−h(c) ≤ threshold]`，

并不是 `[原生 y 已为低能极小值]`。`verified_*` 另外要求公共终态的 optimizer stop、force/stress、有限周期支持、首末能量一致性和周期表示能量检查。[冻结分类器:30](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/frozen/scripts/evaluate_programmed_paths.py.txt:30)、[N/U 输入与终态阈值:117](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/frozen/scripts/evaluate_programmed_paths.py.txt:117)

`A=e(y)−e(R(y))` 测量同一势和有限弛豫算子下的能量松弛量；`B=e(R(y))−h(c)` 描述该公共终态相对已知竞争相的能量。配对组成相同时，`ΔB=ΔeR`，并有 `ΔA=Δe0−ΔeR`。降低 B 而原生 e0 不变，反而会让 A 增大；只压 A 也可能停在一个浅而较高能的盆地。这是代数和对象定义，不是模型因果机制。

raw-v1 与参考的共同 verified native **28 对**：`ΔA=-0.8350016`、`ΔB=+0.0117954 eV/atom`；共同 verified tau800 **57 对**：`ΔA=+0.00437537`、`ΔB=+0.00277834`。前者支持一部分共同验证样本的松弛负担下降，同时 B 变差；后者两项均值都没有改善。分别排除 228 和 199 个请求，不可扩大成全 256 的 A/B 收益。[原始配对字段复算:331](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/SUMMARY_RECALCULATION.json:331)

力、应力、A、B 是相关但不同的对象。小 A 不保证小力（局部高曲率/不充分 R），小力不保证低 B（局部极值或 MLIP 平坦错误区），平均 B 下降也不保证跨过 Strict 零阈值或保留 N/U。当前论文不能把这些作为可互换的“稳定性”。

## 4. 物理代码：正确的单位路径与需要警惕的边界

### 4.1 暂无证据支持一个简单的单位倍率错误

冻结代码直接采用 `CHGNet.predict_structure(...)["e"]` 的 eV/atom，trajectory 总能量除以原子数一次；直接预测应力按 GPa 读取，ASE trajectory 应力乘 `160.21766208` 转成 GPa；StructOptimizer 则用其倒数把模型应力送给 ASE。首帧 per-atom energy 与直接预测以 1 meV/atom 容差核对。没有看到 eV/atom 再除 N 或 GPa 重复转换这类确定错误。[label:114–156](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/frozen/scripts/label_programmed_paths.py.txt:114)、[optimizer:199](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/frozen/scripts/label_programmed_paths.py.txt:199)

作者文档说明 CHGNet 预训练能量已来自 MP2020-compatible 数据。因此不能因为另一个模型的评价器显式执行 MP2020 correction，就给当前 CHGNet 输出再盲加一次同类 correction。应核对势的训练能量约定、参考条目 correction 和 hull 数据源；“流程长得不同”不是单位错的证据。[CHGNet 官方说明](https://github.com/CederGroupHub/chgnet/blob/48a6bada7f42603cf0133b24470c7924587effaa/README.md#notes-for-training)

定义仍须写清：当前 `force_rms` 是 `sqrt(mean_i ||F_i||²)`，不是对全部 Cartesian 分量取均方根；`stress_max` 是矩阵/Voigt 表示的最大绝对分量，不是压力、Frobenius norm 或最大特征值。跨报告比较“力 RMS / 应力 norm”时不能省略该定义。[label:47–62](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/frozen/scripts/label_programmed_paths.py.txt:47)

### 4.2 verified 覆盖下降可能混入停止准则不等价

冻结 R 为 FIRE + FrechetCellFilter，原子/晶胞同时优化，`fmax=.1`、最多 500 步，外部另要求 `stress_max≤.5 GPa`。ASE 3.28 的默认 `exp_cell_factor=N`，cell 广义力除以 N；FIRE 的收敛检查使用广义 gradient norm，未独立比较本项目的 0.5 GPa 阈值。[固定协议](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/frozen/scripts/label_programmed_paths.py.txt:194)、[ASE 3.28 原始发行源码:599](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/authors/ase-3.28.0/ase/filters.py.txt:599)、[cell gradient:669](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/authors/ase-3.28.0/ase/filters.py.txt:669)、[optimizer predicate:520](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/authors/ase-3.28.0/ase/optimize/optimize.py.txt:520)

在参考构型、各向同性应力下，cell 梯度量级为 `(V/N)σ`。`V/N=18 Å³`、`σ=.6 GPa` 给出 `0.067408 eV/Å`，低于 `.1`，但外部应力仍不合格。这个 **NumPy 解析反例**已核算；没有执行 ASE、MLIP 或真实弛豫。它证明两种停止条件不等价，不证明实际多少样本受影响。[CPU 证据](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/CPU_PHYSICS_PROBES.json:29)

现有开发 K8 报告中，tau800 丢失的 40 个 verified 个例里，33 个是力通过而应力失败。不能仅据此决定“延长到更多步”——若它们已因 FIRE 条件提前停止，多给步数上限不会自动让它继续。先交叉查看 `optimizer_converged / actual_steps / force / stress`。即使该问题被证实，也不能由此解释原生输入几十 eV/Å 的力和数 eV/atom 的 A；两种问题可同时存在。[K8 开发诊断:41–52](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/periodic_self_repair_v1/K8_RESULTS_AND_REGRESSION_ANALYSIS_20260906.md:41)

**本轮存量数据审计已经证实它实际发生，且在 tau800 端非常普遍：**

| 数据端点 | `not_converged` 记录 | 其中 optimizer 已返回 true、未耗尽 500 步 | 其中仅应力超标、原子力达标 | 仅应力失败且 0 步即停止 |
|---|---:|---:|---:|---:|
| K8 independent native | 504 | 399 | 295 | 0 |
| K8 independent tau800 | 763 | 762 | 712 | 19 |
| raw-v1 development native | 105 | 77 | 59 | 0 |
| raw-v1 development tau800 | 164 | 164 | 140 | 1 |

表中第三、四列均限定 optimizer=true。K8 tau800 的 763 个 `not_converged` 中只有 1 个是实际耗尽 500 步且力/应力仍超标；另有 1 个耗尽步数的 invalid-terminal，未混入该分母。raw-v1 tau800 没有任何一个 `not_converged` 是因为耗尽上限。因此把这两个端点的低 verified 覆盖主要解释成“500 步不够”已经被本次记录反驳。也存在 optimizer=true 但 Cartesian 原子力超标；Frechet 的原子广义力使用 deformation-gradient 变换，不能把 filter 的单个 fmax 与外部原子/应力双阈值画等号。[完整实际交叉计数](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/ACTUAL_SAVED_PHYSICS_AUDIT.json)、[独立解释复算](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/ACTUAL_AUDIT_INTERPRETATION.json)、[ASE 3.28 原子力与 cell packing:663](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/authors/ase-3.28.0/ase/filters.py.txt:663)

这是 **R 的停止/验证语义不一致的实证**，不是重新获得任何 SUN 命中，也不是证明更长/更严格 R 会给出更低能终态。对 headline 的具体影响还应按 `terminal_status × strict/meta_stable × N/U` 汇总；当前不重标冻结结果。

最小未来修复候选是让停止谓词与公开的联合验收条件一致，而非放宽应力阈值；需要对全部配对方法统一重新登记 R，并保留原冻结结果。它属于评价/标注协议修复，不能冒充 DLM 学习收益。

### 4.3 有限 image shell 不能支撑“所有晶体严格有效”

本地调用冻结 validator 的反例：晶格参数 `(1,3,4;90°,90°,8°)`、一个原子时，半径 2 shell 的最小周期像长度为 `1.0 Å`；`(-3,1,0)` 周期像长度却为 `0.418539 Å`。这些长度/角度都能由现有 token 表示。故当前声明的有限支持确实可能漏掉晶格倾斜造成的短周期像；不能从“125 个像 + 三斜胞”推导普遍的精确 MIC。[CPU 反例](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/CPU_PHYSICS_PROBES.json:6)

已进一步读取四个端点的保存晶胞，用完备的**阈值内**搜索方法审查：若 `r=(δ+n)L` 且 `||r||<d`，则每个分量满足 `|δ_k+n_k|≤d||column_k(L⁻¹)||`；对 centered δ 构造有限整数 box。正常胞若整个 box 已在半径 2 内，直接确认没有遗漏；极端胞才额外搜索，超过预算明确记 uncertified。不能把额外有限半径 5 又称为普遍精确算法。[只读工具](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/tools/audit_saved_physics_lane.py:93)

**实际结果：5,788 个可用 raw/terminal 几何记录全部完成阈值审查，半径 2 漏掉 `<0.5 Å` 近邻的记录为 0；没有 image-budget 未完成项。** 另外 36 个几何缺失对应 18 条已记录生成失败的两个端点，保留为 missing。故 MIC 漏邻不是这批低 SUN 的已知主因，应从根因优先级中降级。它仍约束论文能够声称的数学保证。

另一项真实几何结果不应混淆：所有可用 raw 输入都通过原有距离底线，R 后的终态却分别有 K8 native 393、K8 tau800 1、raw-v1 native 90、raw-v1 tau800 0 条违反底线；这些是原半径 2 就能检测的失败，不是新增近邻算法“发现”的漏判。它支持“原生坏结构与无约束 R/势外推相互作用”这个竞争解释，不能单独归咎 sampler 或单独归咎 optimizer。[实际原/终态支持计数](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/ACTUAL_AUDIT_INTERPRETATION.json)

### 4.4 可复算能量仍不是物理可信性证书

旧 Gd2 的异常 `eR≈-499.128 eV/atom` 在同势下基本可复算，周期平移跨度只约 `1.007 meV/atom`，略超过统一 1 meV 门槛。不能说已经找到约 485 eV 的记账错误，也不能说该小数值跨度解释了极深能量。原 teacher 约 96.27% 的 B 正收益来自该组成，说明 surrogate 异常能支配全局偏好。[旧统一验证:12–40](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/21_TERMINAL_REPRODUCIBILITY_AUDIT_20260906.md:12)

当前 `check_terminal_energy` 检查三表示能量一致、fresh F/stress 有限；它不把 fresh 三表示 F/stress 再与外部阈值逐项比较。可先读取已存 `terminal_consistency.scores` 审查，不必重跑模型。即便全部通过，也只提高同一势下的可复算性；需异势/DFT 才能检验异常低能区域的物理可信性，当前大审计不启动这些计算。[冻结检查:25](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/frozen/src/crystal_dlm/terminal_energy_consistency.py.txt:25)

本轮也已读取全部 873 个 labelled-verified 记录的三表示 fresh 检查：只有 raw-v1 native `eval:44:0:0` 的 shifted Fmax 从 trajectory 的 `0.0998855` 到 `0.1000361 eV/Å`，略跨 `.1`；同记录原表示 fresh 为 `0.0998486`，三表示能量跨度仅 `9.5367e-7 eV/atom`。其他 fresh F/stress 都在原阈值内。因此该 fresh-force 门槛敏感性真实存在，但幅度很小，不支持“广泛的数值不一致导致弱 SUN”。保留原 verified 标签，不事后改变容差。[具体例与全部统计](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/ACTUAL_AUDIT_INTERPRETATION.json)

## 5. LLM→DLM 链路没有断，但“条件对齐”仍有科学边界

### 5.1 哪些资产有证据，哪些不是稳定性保证

C³FD 为 typed 化学动作提供可达支持；Llama 在同一支持内通过 unit-weight PoE 改变概率，之后预测 soft Plan 与物种程序。PoE 保证支持相容，不保证更低的 eR；Planner 的训练数据实际是 MP20 near-stable 正先验，不能称为学到了高/低 hull 的对比分类器。[Planner 数据边界:80–85](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/reference/component_specs/ADR_C3FD_LLAMA_FUSED_TYPED_PLANNER_20260831.md:80)

Species pointer 控制 unique-species 顺序、anchor、后续 native slots 的访问与 program rank；它不选择每个数值 token，也不在每次 DLM 修订前重新阅读结构作在线决定。teacher 为 contact-tree 排列，不是最低能排列。旧 BC→BP 中 Direct 从 `99.22%` 到 `98.05%`，因此“Llama learned order 已优于 canonical”没有证据；真正支持的是学到非平凡、可执行的程序，以及后续 DLM 能吸收它。[程序实测边界](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md:187)、[完整代码追踪](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/periodic_self_repair_v1/review_notes/OLD_LLM_DLM_ORDER_TRACE.md:5)

“组成有效”也没有消除组成选择的上限：在给定 c 下没有可达低能结构、或模型对该化学域严重外推，几何恢复不能靠规则创造一个稳定组成。反过来，metadata 缺失并不证明该组成物理不可行或超出 C³FD 真正支持；V2 的 2578 fallback 原因实际记录为 `missing_typed_metadata`，不应改称 2578 个“无解组成”。

### 5.2 V2 已解决的接口问题

V2 对可用 typed metadata 使用真实 frozen Planner forward，并采用部署同类的 PoE→temperature→top-p→sampling，再把**实际 sampled soft IDs**送进 pointer；不再把 MAP soft IDs 当随机部署分布。program 到 scalar positions、old/current/active 的编译与回放明确共享。V2 的多头周期 bias 确实进入 Transformer 交互；旧数值状态、绝对尺度和 cell↔site 交互都存在。这些是接口和可训练性成果，不能仅由梯度 marker 推出物理收益。[冻结条件代码:294](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/frozen/src/crystal_dlm/periodic_v2_plan_data.py.txt:294)、[V2 实现验收](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/periodic_self_repair_v1/V2_IMPLEMENTATION_REVIEW_20260906.md:9)

当前真实准备结果：24558/27136 train sources 直接预测，2578 行 teacher-soft/canonical fallback；直接预测行中 program 与原顺序相同 18197 行，soft annotation agreement 为 lattice 14103、SG 13987、volume 16217 行。它们的分母均为 24558，而不是全部训练集，也不是 deployment 生成准确率。[真实准备快照:165–190](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/EVIDENCE_RUNTIME_SNAPSHOT.json:165)

### 5.3 低 annotation agreement 不足以证明伤害，但暴露另一种条件目标

训练把一个真实 MP20 晶体 y 与根据其化学 metadata 随机生成的软计划 H 配对，而不按 H 选择另一个符合它的真实 polymorph。若理想化地 `H ⟂ y | c`，则 `p(y|c,H)=p(y|c)`：最佳条件模型可能学会忽略 H，而不是服从每次随机 SG/VPA 提议。这是一个条件独立假设下的推导，实际 typed history、重复组成、稳定标签和固定 source seed 可能破坏该假设，需要验证。

因此至少有三种竞争解释：模型学会对不可靠 H 保持鲁棒；模型过度服从错误 H 而破坏 geometry；模型忽略 H，软计划对发现无增益。单个 agreement 比例不能区分它们。重复组成可能本来就有多个合理 lattice/SG/VPA，primitive metric 与 conventional SG 标签也不是一对一。

最早偏差位于“条件与 clean target 配对”而非 GEM 末层。必要的有界诊断是同 composition/program 下，teacher-soft、predicted-soft 与 soft-masked 的**条件敏感性及目标可达性**，并按实际 disagreement 类型和 fallback 分层。先用已有预处理行和记录计算条件/annotation 关系；没有相应模型对照时，不宣称这些关系造成 SUN 损害。不得把 teacher SG、Wyckoff occupancy 或真实配位图补进部署输入。

## 6. 为什么不能拿别人的“成功指标”直接证明我们缺一个模块

| 原论文 / 作者实现 | 输入条件和实际评价 | 本项目可借鉴什么；不能比较什么 |
|---|---|---|
| DiffCSP++ CSP | 作者论文区分 GT SG/Wyckoff 与 CSPML 模板，MP20 Match Rate 分别 `80.27%` 与 `70.58%`；衡量对指定真值结构的重建。 | 说明正确的低维结构条件有用；不能将该重建率与 de novo SUN 横比，也不能把我们预测 SG bucket 当作 GT Wyckoff 条件。[论文 §5.2](https://arxiv.org/html/2402.03992v2#S5.SS2) |
| DiffCSP++ DNG | 作者从训练结构抽模板；主指标为 validity/COV，性质分布用 valid 子集的 Wasserstein。代码不是本项目的 hull-SUN 管线。 | 对称性约束缩小几何空间是机制启发；模板支持与未条件化生成不同，低 formation-energy Wasserstein 不是低 B 或高 SUN 的同义词。[作者 generation](https://github.com/jiaor17/DiffCSP-PP/blob/e82e7ffa7cb2a383bde69b067a343ca137d73e47/scripts/generation.py#L54)、[指标实现](https://github.com/jiaor17/DiffCSP-PP/blob/e82e7ffa7cb2a383bde69b067a343ca137d73e47/scripts/compute_metrics.py#L160) |
| Crystalite v2 / 作者当前代码 | N/U 从生成的 finite-geometry 集合计算；先筛 novel 再 deduplicate 得 UN，再对 UN 做热力学并乘率。matcher 默认为 `.3/.5/10°`，不同于我方 `.2/.3/5°`。 | 支持几何直接进入 attention 的机制。它与我方同样可采用 pre-R N/U，但有效分母、UN 运算顺序、matcher、热力学势与采样量不同，SUN 名称不能代替协议同一。[作者 N/U](https://github.com/joshrosie/crystalite/blob/3b2d4eacf3f0b17a04851b7bed1fcedc9733cda9/src/eval/uniqueness_novelty.py#L114)、[作者 SUN](https://github.com/joshrosie/crystalite/blob/3b2d4eacf3f0b17a04851b7bed1fcedc9733cda9/src/eval/dng_eval.py#L242) |
| Crystalite 的热力学说明 | 论文 Appendix E 说明 fixed 200-step NequIP/FIRE/Fréchet、MP2020 entry correction；主文常把 0.1 eV/atom 称 stable。 | 不能把其主要 SUN 数字自动映射为我方 Strict≤0；也不能把固定步评价叫收敛终态证明。[论文 Appendix E](https://arxiv.org/html/2604.02270v2#A5) |
| MatterGen 作者评价器 | 默认先以 MatterSim 弛豫，然后把 relaxed structures 同时传给 N/U 与能量评价；默认稳定阈值 0.1。作者另明确论文采用 DFT，发布代码的 MLIP 数值可能不同。 | 它的 SUN 对象更接近最终 relaxed discovery，而我方 frozen hybrid 指标强调原生成结构的 N/U。两者都可定义，但不可混称同一 benchmark。[作者评价入口](https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/mattergen/evaluation/evaluate.py#L18)、[作者说明](https://github.com/microsoft/mattergen/blob/92423660a8bd70e83679086e88f88596d484dc16/README.md#evaluation) |

外部作者仓库固定到了本次审查的 SHA；这不是声称仓库 HEAD 恰等于论文当年的实验归档。论文版本与当前作者实现分别注明。

### 当前 N/U 的另两个边界

我方 novelty 只相对固定 MP20 **train** structures，而 hull 来源覆盖更广的官方 MP 竞争相。故“novel”意为相对这份训练参考的新结构，不能翻译成“MP 从未收录/全世界从未发现”。完整源码 hash 与 training CSV 已核：[冻结 N/U 函数](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence/FROZEN_NU_FUNCTIONS.json)。

conditional1000 的 unique 代表继承疑点已经关闭：冻结 `compute_uniqueness` 只依次比较更早的同 formula 结构，后来的样本不改写以前的 class。因此 native first-1000-parseable 源序前缀可以继承全 1200 的 first representatives；它不是全局 union-find。不过条件 1000 仍是**给定 parseable 的条件队列**，不等于独立的 1000 次无条件请求。不同方法若各取自己的 first-1000 成功项，成员可能不同，不能叫逐请求配对；本轮 native/tau800 复用同一选择清单是正确的。

## 7. 过去结果的成功原因与过度外推

下面审查与当前链路直接相关的实验族；完整 run/版本清单由主审统一整理。保留下来的汇总不能补造已经删除的逐样本资产。

| 实验族 | 成功或失败提供的实际信息 | 必须撤回或限定的外推 |
|---|---|---|
| C³FD v2.5 / typed Planner | 在线可达 witness 使两个 requested1000 的 composition validity 达到 2000/2000；是规则支持与执行的真实贡献。[原始证据台账:222](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/DLM_ORIGINAL_ARTIFACT_EVIDENCE_LEDGER_36H_V1.md:222) | 不是 DLM 几何结果，也不是热力学最优组成保证；typed positive prior 不是 hull oracle。 |
| SPAD canonical / pointer / backfill / schedule training | B0→BC Direct 大幅增加，BS 达到 511/512；后缀干预确实改变早期 masked logits。[06:143](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md:143)、[199](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md:199) | 证明可执行性和非因果信息通道，不证明 pointer 顺序最优或近 100% 物理稳定。 |
| Canonical slot 对齐 / G2 交互 | canonical Direct `128→143`；再加 G2 为 `133`，无可靠 raw 能量优势。[旧结果:41](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/FAILED_METHODS.md:41) | 语义对齐有效不等于每个旧模块都应叠加；不能以几何 bias 的合理性代替完整效果。 |
| G2 prospective | raw 能量区间跨零，refined hull 有利，但 binary SUN 未建立多 seed 鲁棒性。[完整报告:45](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/36H_FINAL_REPORT_C3FD_G2_20260901.md:45) | 系统层收益不能重命名为原生 DLM 学得热力学稳定。 |
| B3 / 更多普通 CE / SGTC | B3 synthetic NLL 降而真实 rollout NLL 升；更多 CE 常保住/提高 body，未同步改善 Strict。SGTC L7 Strict 从 60 到 53。[rollout 汇总:49](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/ROLLOUT_MATCHED_DLM_24H_CHECKLIST_V1.md:49)、[SGTC 台账:273](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/DLM_ORIGINAL_ARTIFACT_EVIDENCE_LEDGER_36H_V1.md:273) | 不能仅因 loss 曲线下降或多训练一遍就许诺 stable discovery。也不能由一个负结果否定所有 CE 或课程学习。 |
| Rollout→MP20 后缀 oracle | BASE 59/128，Z-stage teacher continuation 43/128；训练前目标就与既有错误前缀不相容。[实测:91](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/ROLLOUT_MATCHED_DLM_24H_CHECKLIST_V1.md:91) | 这类错误不是调 LR 可修复；也不意味着所有完整 old→target reconstruction 都不可能。 |
| Force-Score / projected microstudent | 连续降能常在量化后丢失，硬 species margin 可过约束；修正后 teacher 成功仍未转成 student 净 Direct 收益。[力教师:41](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/FORCE_SCORE_DLM_CHECKLIST_V1.md:41)、[student:34](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/FAILED_METHODS.md:34) | energy downhill 不保证 geometric validity；teacher 局部成功不等于学生会自修复。 |
| BTRD endpoint distillation | 两次 residual-only 有限实验提高部分 Direct，却降低 Strict，能量方向不利/不确定；相应资产部分已删除。[保留记录:3–24](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/FAILED_METHODS.md:3) | 只能否定已测数据/实现/预算的稳定迁移；不能说已经彻底证明所有 endpoint distillation 无效。 |
| CTV token value / D3PO | 单 token value 不能可靠预测完整 continuation；post-refiner preference 的原生收益未被建立。D3PO 还有原始 artifact 缺口。[台账:293](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/DLM_ORIGINAL_ARTIFACT_EVIDENCE_LEDGER_36H_V1.md:293) | 不能把单 token credit 或仅 refiner 后能量改善当全路径因果价值；工程失败和未完成官方评测不是物理负结果。 |
| Potential-Closure / K10 | 有真实局部 candidate headroom 与可训练梯度；更大支持/重复暴露仍未建立严格低能尾的可靠完整晶体迁移。[K10 末轮:88](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/13_STREAM19_DIAGNOSIS_AND_FINAL_ITERATION.md:88) | 两个不同 prospective stream 不是严格配对的 epoch 因果实验；不能把 early UNDERTRAINED 当最终唯一解释。 |
| K4/K8 完整路径偏好 | K4 verified pool 的最低 A 支持稀少；teacher 全局平均双降允许条件内退步。K8 已纠正主实验有实际 native/system 收益范围，但未完成 A/B 普遍双改善。[27:9](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/27_SELF_IMPROVEMENT_REPAIR_PLAN.md:9)、[21:45](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/21_TERMINAL_REPRODUCIBILITY_AUDIT_20260906.md:45) | K 增加不增加 composition coverage；有限候选重加权不能产生池中不存在的近平衡样本。teacher KL 不是 student 漂移上界。 |
| PMTR | 合成 corruption certificate 与梯度实现是两关；原实现停在首个更新前的非有限梯度。[工程事实:416](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/GPT6_AUDIT_HANDOFF_20260905.md:416) | 不能把 510/512 合成 certificate 当 student 物理提升，也不能把未训练过的模型称为科学负结果。 |
| raw-v1 / 当前 V2 | raw-v1 工程与表征可训练但最终 SUN 低于参考；V2 已修多项来源/目标/交互接口，真实梯度 marker 存在。 | 两者同时改变多因素，不能归因于一个 ratio 或一个 GEM；V2 尚无终态物理结果时，不能说已修好稳定性。 |

## 8. 排序后的根因与竞争解释

| 优先级 / 判断强度 | 最早可能产生偏差的位置 | 已有证据与竞争解释 | 必要的下一项验证 | 最小修复方向及可能反效果 |
|---|---|---|---|---|
| 高：原生非平衡确实存在 | DLM 初始 construction，可能早于 repair | 高 force/stress/A、过密晶胞；旧 K4 大退步样本在协同前已变动 lattice。并非全由 R 引入。 | V2 同一 trace 的 construction→repair 物理差，按 VPA/最小距离/步骤分层，保留失败。 | 若 repair 有效保留它；若无效，先定位旧状态、实际 prefix 与 teacher 覆盖，不能靠更多同类 CE 或更硬阈值掩盖。 |
| 高：teacher/support 与目标有上限 | 候选生成和反馈 admission | K4 均匀 A 约 6.54，池内每组最小 A 均值仍约 5.48；但存在局部 Pareto 信号。 | 比较候选支持、可用反馈与 student 行为三层，避免互代。 | 将来若重启自提升，需真正同起点可达改进；仅放大权重会放大 surrogate 异常/跨条件补偿。当前不重启。 |
| 高：外部验收与 optimizer 停止不等价已实证 | 公共 R 停止 | K8 tau 的 762/763、raw-v1 tau 的 164/164 个 not_converged 都是 optimizer 已返回 true；多数仅应力失败。 | 已完成交叉计数；下一步只需把现有 S/SUN 按这些停止类别分解。 | 若将来修 R，改联合停止而非放宽标准；统一协议可能改 A/B/S，必须给所有对照同等处理。 |
| 中：软计划不是对应 target 的 oracle | frozen Planner 预测与 clean target 配对 | 条件来源改得更接近部署；低 agreement 也可能被鲁棒模型正确忽略，而非必然伤害。 | 条件敏感性、同组成多型、fallback 与非 fallback 的分层结果。 | 将来可做来源/可靠性显式化或更精确的条件目标，不能硬化 SG/Wyckoff；硬化可能排除真实结构、退回立方坍缩。 |
| 中：refiner 转移分布及 N/U tradeoff | `F800(x)` | S 大幅增加、N/U 减少，旧 native 偏好训练没有直接优化固定 F800 映射。 | 固定输入配对的 raw→F800→R 三端转移，不能比较不同队列 median 归因。 | 确定是 mode loss 后才考虑最小 bridge；减 tau 可能降低 S，更多修复可能损失 N/U。旧 tau sweep 不能为新 proposal 保证最优。 |
| 低：有限 MIC 的普遍保证不成立，但本批未受影响 | 生成支持与 label geometry validation | 反例成立；5,788 个可用原/终态几何记录无阈值内漏邻、全部完成 box 审查。 | 本批疑点已降级；未来分布改变仍应保留同类 audit。 | 不为未观察到的本批问题重训；若未来修验证器，必须给共同对照同等处理。 |
| 中低：势外推支配能量偏好 | raw 或 terminal 的 MLIP 预测 | Gd2 极端低能基本可复算；高能/力尾也可能出于真实生成坏结构。 | 存量量纲/一致性/物种/几何联合诊断；必要时未来另登记异势/DFT。 | 仅数值 gate 不足以物理认证；事后删 outlier 会制造收益，跨势分歧也不能靠选最有利的势处理。 |
| 已确定的评价故障，已纠正主结果 | hull cache 与新队列对接 | 主 1000 的旧“近零”结果被补覆盖纠正。 | required/covered/unresolved 三者完备及源 hash。 | 加覆盖契约即可；不需为缓存故障重训模型。 |

## 9. 本轮只读验证与交付工具

已完成：原始汇总计数/比例/unknown 上界复算；单位与 A/B 恒等式检查；有限 MIC 反例；应力分量旋转反例；Frechet 停止条件解析反例；作者评价实现的固定 SHA 对照；conditional1000 首代表继承的冻结源码确认。没有执行新的 DLM/MLIP/弛豫。

用于真实存量产物的 [audit_saved_physics_lane.py](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/tools/audit_saved_physics_lane.py) 只需 NumPy，输入为同端点 `labels.jsonl` 和可选的 exported `paths.jsonl`，输出逐例审计、停止交叉表、fresh F/stress 检查和阈值 MIC 完备检查。工具已通过漏邻反例、立方对照、60 个随机几何 box 核验及完整两行合成 CLI 检查。[工具 CPU 证据](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/SAVED_AUDIT_TOOL_CPU_TEST.json)

四组真实存量输入（1200、1200、256、256 条）已在本地完成审计，OMP/MKL/OpenBLAS 均限制为 1，最多两个 CPU 进程并行。导出的原字节 SHA 已由主审核对，扫描保留新的输入 SHA，不修改原字段。全部输出位于 `evidence_physics/actual_{k8_native,k8_tau800,raw_native,raw_tau800}`，跨端点统计见 [ACTUAL_SAVED_PHYSICS_AUDIT.json](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_physics/ACTUAL_SAVED_PHYSICS_AUDIT.json)。模型调用、新样本、新弛豫均为 0。

## 10. 对论文与潜在 V3 的最小结论

当前可辩护的故事是：**化学可达支持与 Llama 物种程序提供精确条件和执行分解；DLM 在同一周期晶体上进行条件生成/联合修订；几何可执行性、原生近平衡和低能新颖发现是三个需要分别证明的结果。** 程序控制和周期几何交互是真实机制，是否提升低能发现仍是实验问题。

不应再承诺“合法支持已解决物理有效性”“GEM 存在/有梯度就会稳定”“每组成候选更多就必然自提升”“小 A 等于低 B”“model494 可以直接去掉”，或把 engineering success、局部 certificate、teacher 最优解、普通 val loss 当作最终 student SUN。

存量审计已经把“停止/验证不等价”升级为实证，同时把“本批 MIC 漏邻”和“广泛 fresh 数值不一致”降级。V2 自身终态仍需等待既定评测；这些结果不构成启动又一个架构的授权或证据。最小的当前工作是清楚分离指标和数据来源、完成现有 construction/repair 对照，并将 S/SUN 按停止状态分解。若未来需要 V3，应只针对被证实的失败：R 的停止问题修 R 的合同；软条件问题修条件语义；自身错误迁移问题修训练状态；低 B 支持不足才考虑新的物理反馈。它们不是一个可以由统一 `2:1` loss ratio 或外接 GNN 一次解决的问题。
