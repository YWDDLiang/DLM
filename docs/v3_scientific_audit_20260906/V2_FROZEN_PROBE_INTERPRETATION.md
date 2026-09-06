# V2 冻结前向诊断 40004：结果解释与有限决策

2026-09-06。只读分析 [V2_CONDITIONING_PROBE_40004.json](evidence/V2_CONDITIONING_PROBE_40004.json)，证据 SHA256：`8d3818ac6ba4c5f13614d89d65b1d424701847edba08bed94c86f609d3c706a6`。本轮只新增本报告，没有训练、生成、SSH、MLIP 或模型修改。另读 [V3 数学第一轮审稿](V3_REVIEW_R1_MATH.md)、[V3 物理第一轮审稿](V3_REVIEW_R1_PHYSICS.md)，不修改它们或候选稿。

## 1. 结论

**V2 的旧几何与 soft prompt 已实际影响预测；“模型完全没用这些信息”可以退出当前根因清单。** 遮蔽旧几何提高了这批目标风险，影响集中于 repair；将预测 soft 换成原目标注释，在固定 P 下平均降低风险。真实 sigma 标签也有实际输出通路，但相对 unknown 没有表现出平均风险优势。

这些结果证明的是已实现通路的使用及这批 teacher 状态上的预测作用。它们**不证明**旧几何模块提高 SUN、不识别某个单模块的因果贡献，也不证明 soft SG 服从、posterior 校准或新连续头会成功。当前结果足以结束这次 frozen probe；没有必要为维护某个故事再扩充更多遮蔽臂。

## 2. 分母、冻结与计量口径可接受

- 固定 identity-hash selection seed `202609061`，100 个原 validation source；state seed `20260906`、epoch=1，每源 construction/repair 各一 view，共 **200 状态**。没有 outcome 筛选。
- construction prefix/dense 为 **73/27**，repair prefix/dense 为 **74/26**；空监督 **0**，共 **1,062 个目标 scalar**。
- 四张 A800；各 rank 50 状态。共 **425 个实际 forward batch、850 个实际 forward row**；逻辑上五变体×200状态=1,000，差额来自相同输入复用。记录的单 rank 循环时间为约 42.45–62.20 秒，**不含模型加载和全部准备成本**，不能直接当成新训练吞吐。
- 所有 variant 的 200 个 state risk 有限；汇总的模块激活/差值和 1,062 个位置的 TV 也均有限。所有 rank 参数 version 未变、可训练参数为0、model/dropout 均非 training 状态。
- baseline 与原 `PeriodicV2Objective` 的最大 batch-risk 差为 **5.24×10⁻⁷**；prefix objective conflict 为 **0**。本次测量没有发现风险实现与原目标不一致的迹象。

公共支持在干预前固定：prefix 用原实际 policy support，dense 用 canonical typed support；另报 schema risk。mask-old 后没有重新用 old admission 将损失置零。alias 先在原 dtype 合并，再以 T=0.7 计算风险，因此下面的差异不能归因于换了合法集合或删掉难状态。

对 source i，以两个 view 的风险差平均定义 `Δr_i=(ΔR_i,construct+ΔR_i,repair)/2`，再对100 source取宏平均。R 继承原对象权重与 dense inverse-inclusion 系数，单位是**加权条件分类风险**，不是最终结构 NLL、能量或 SUN。因为每源恰有两 view，source 宏均值与200状态均值相等；但中位数和分位数不同，不能混用。

## 3. Source 宏结果：几何有作用，噪声标签未显示平均收益

负的 ΔR 表示干预后目标风险降低。baseline 的 common/schema 平均风险分别是 **2.009258 / 2.030197**。

| 配对，method − reference | source 宏均值 ΔR | source 中位 ΔR | source q10 / q90 | 全位置平均 TV |
|---|---:|---:|---:|---:|
| true noise − unknown noise | +0.000559 | +0.000030 | −0.009824 / +0.014332 | 0.004724 |
| target soft − predicted soft，同 P | −0.035770 | 0 | −0.219729 / +0.084932 | 0.055773 |
| masked old − available old | +0.062878 | +0.017508 | −0.080264 / +0.301145 | 0.052670 |

q10/q90 是100个 source 配对效应的分布分位数，**不是置信区间**。最后一列按1,062个目标位置计权，由各字段 `count×mean(TV)` 汇总；它不是 source 宏 TV，不能与 source 风险列当成同一种平均。快照没有完整逐源风险数组，本文不虚构显著性检验、胜负 source 数或 bootstrap 区间。

schema 风险均值差依次为 **+0.001026、−0.035588、+0.062372**，与 common 风险方向一致。因而两个主要现象并非只在几何截断后的较小支持上出现。

## 4. 任务分层比一个全局均值更有解释力

下表每个单元格为“该状态组平均 ΔR / 该组目标位置平均 TV”。dense 一状态可有多个目标，两个数仍采用不同计权。

| 任务 | 状态 / 目标数 | true − unknown | target soft − predicted soft | masked old − available old |
|---|---:|---:|---:|---:|
| construction prefix | 73 / 73 | +0.000519 / 0.00415 | −0.043044 / 0.04551 | +0.007257 / 0.03141 |
| construction dense | 27 / 444 | +0.000348 / 0.00083 | −0.014644 / 0.05609 | +0.002053 / 0.00871 |
| repair prefix | 74 / 74 | +0.001105 / 0.00632 | −0.045487 / 0.05439 | +0.067599 / 0.08236 |
| repair dense | 26 / 471 | −0.000666 / 0.00824 | −0.009633 / 0.05728 | **+0.268774 / 0.09274** |

旧几何遮蔽的 construction/repair 各100状态平均效应分别为 **+0.005852 / +0.119904**。repair dense 中位风险差也为 **+0.115174**，并非只有均值变化。这支持“完整旧几何在 synthetic repair 预测中有用”的判断；不能外推成“依靠它执行 repair 就会改善自由生成结构”。

true sigma 的四个任务组效应都接近零，且符号不统一。当前没有证据支持仅把 runtime 的 unknown 改为某个声称正确的 sigma 就会提点；对自由生成错误本来也没有这份 corruption 的真实参数。它也不意味着未来新 score 模型不需要正确时间/噪声定义。

不同字段的 prefix 样本很少，例如 repair prefix 的某些 length/angle 只有3–5个目标。可以定位响应，不能据最大字段均值排出稳定的“最重要轴”或调整训练权重。

## 5. 模块响应支持哪些机制结论

**噪声通路实际存在。** true/unknown 间 conditioner cell/site 和 task projection 的差为0；GEM body bias 有变化，repair prefix/dense 的平均差值 RMS 分别约 **0.000611 / 0.000668**，最终 target hidden 的差值 RMS 为 **0.04835 / 0.03341**。这与 sigma 三分量进入 V2 GEM 的代码路径相符，排除了“新 noise channel 完全不影响输出”的简单说法。非零响应不等于 calibration 已有收益，宏风险已给出其限制。

**soft 通过 prompt/Transformer 路径产生响应。** 同 P 的 soft 替换下，conditioner、task projection、body 对齐后的 GEM 差值在所有任务组都是0；最终 target hidden 的平均差值 RMS 约 **0.129–0.218**，数值输出也改变。不能因原 population 独立性反例，直接宣布有限 V2 一定忽略 S；该反例限制的是随机标签配原目标的可识别保证，不是已经观测到的网络行为。

**mask-old 是多通道干预。** 它改变 conditioner 与 GEM，repair 的 task projection 也改变；repair prefix/dense 的最终 target hidden 平均差值 RMS 约 **0.4854 / 0.3707**。因此风险升高不能被归因为“单独关掉某层 GEM”，更不能把 conditioner/GEM/head 的 RMS 比值当作贡献比例。current 里原本可见的数值仍在，mask-old 也不等于完全移除所有几何信息。

**construction 的直接 query 瓶颈得到真实权重下的验证，但不是完全断开。** 100个 construction 状态的 baseline target-query geometry bias 均为0；repair prefix/dense 则分别有约 **0.01598 / 0.02154** 的平均 RMS。与此同时，construction hidden 仍会响应 old/noise 干预。已知上下文之间的几何 bias 和全局 conditioner 通路可以传到目标位置，不能将“直接 query 边为0”扩大为“construction 没有几何计算”。

上述是输入干预的实际传播与风险证据；没有单独移除模块、没有训练匹配对照，不提供某模块必要性或物理贡献的独立估计。

## 6. Soft 的长度、标签与复用边界

target-soft 配对中，**70个 source 的模型输入改变，30个不变**。不变者不等于30个 teacher fallback：其中也可以包含原预测已与注释相同的 source，本快照未提供可分解计数。

32个 source 的 prompt 长度改变，组均值 ΔR 为 **−0.059117**；68个长度不变，均值为 **−0.024784**。后者包含30个完全相同输入。扣除这60个零差状态，可以由现有分组和均值直接复算：**38个“文本改变、长度不变”的 source，平均 ΔR≈−0.044350**。因此，全部 soft 效应不能仅用 body 绝对位置变化解释。

这仍是固定 P 的原注释干预，可能离开原 S/P 联合分布；原注释也未必等于粗 codec 后的实际 SG/metric class。不能把它改称 SG 控制成功，不能在部署时提供未知的 target S，也不能只保留 annotation 匹配的 source。

完全相同的整 batch 输入会复用前向，并记录 `computed_as`。总共1,000条逻辑变体状态只计算850条实际 forward row；这些复用的零差不是独立重复前向的确定性实验证据。mask-old 另有3个状态输入本来就相同；全部仍留在分母中。

## 7. Float64 exp：本批没有触发此前的数值反例

真正的 seeded-sampler 路径先将 policy vector 转为 **float64** 再 exp，没有先除 T。baseline 与 unknown-noise 各检查 **147个 teacher-prefix vector、30,645个合法值**：

| 项目 | baseline | unknown noise |
|---|---:|---:|
| 原 policy logits dtype | bfloat16 | bfloat16 |
| exp 输入 dtype | float64 | float64 |
| 合法 logit 范围 | **[−19.25, 17.125]** | **[−19.25, 17.0]** |
| exp overflow / zero / subnormal 值 | 0 / 0 / 0 | 0 / 0 / 0 |
| 全合法值 exp 为0的 vector | 0 | 0 |
| policy-unavailable vector | 0 | 0 |

hard-mask 的 `finfo_min` 已排除。float64 的 log(max finite) 约709.78、log(min normal)约−708.40，本批真实值远离这些范围。**用 exp 溢出/全下溢解释当前低 SUN，应退出优先根因队列。** 原 ±1000 反例仍是合法的工程健壮性边界，但不是本批实测故障。

这里测的是 validation teacher 状态，未覆盖39998的所有实际生成 trajectory，也没有测完整随机 Gumbel 比值。零计数不构成所有未来 logits 的全域保证；它已经足够支持当前不要围绕此问题追加科学实验或宣称物理提点。

## 8. 监督密度是有限预算事实，不是效果倍数

本批平均 N=9.5，共有每个 source 一份完整几何的 **3,450个 scalar**。两 view 若都监督全几何，共6,900；现有1,062个目标相当于每状态5.31个，数量比为 **6.50**。去重后，两个 view 一共覆盖 **936个 source-position**，平均每源9.36个。这个计数描述本次抽取的 validation views，不能当作实际两 epoch 的训练覆盖结果。

原 construction/repair view 分别给出517/545个目标。若在完全相同的这批 T 状态上采用“一 T、一全几何 G”，目标分量计数会是 `517+3450=3967`，约为现计数的 **3.74倍**；这是条件性的预算算术，未运行 H。两个 full-G view 与一T一G不是相同几何曝光。

这些 target 相关，高噪声 torus 目标甚至可接近零；不同 CE/MSE/scaled-score 的一个分量也不等价。密集 token 监督同样可增加目标数，所以这份结果不构成“只有连续 head 才能解决”的论证。

## 9. 与真实 SUN 结果合并后，对候选的判断

冻结开发256仍是 **construction 12/44 → 一次 repair 8/50 → 固定 tau800 16/113**（Strict/Meta）。第一段 Strict 失去8、新增4；Meta 失去26、新增32。完整 verified SUN 又从4/21降至2/18；共同 verified 仅33。raw E 均值虽降0.17806 eV/atom，中位差却升0.19127。来源：[construction/repair 配对](evidence/V2_CONSTRUCTION_AND_REPAIR_39998.json)、[两端结果](evidence/V2_EVALUATION_SUMMARIES_39998.json)。没有 construction→tau800 对照，不推算它。

这正是“条件预测确有用，但执行后的物理结果有权衡”的实例；两种观察并不矛盾。

| 候选 | 本 probe 提供的有限支持 | 可操作判断与不能跳过的边界 |
|---|---|---|
| A：条件/目标联合校正 | 同 P 的 target annotation 平均降低风险，且存在等长度响应；S 并未被完全忽略 | 保留为有依据的条件语义假说。先固定输出属性/unknown/冻结 prior 的联合支持；不直接把GT-S喂入部署，不把这次CE差当SUN收益，也不自动混入几何头实验 |
| B：同核 C_mix→D | repair 预测确实依赖 old；unknown 通道也真实工作，未见 true-sigma 的平均优势 | 如果做零训练比较，仍只登记H=1与原三档mixture/unknown；必须击败现成construction及原repair权衡。probe没有检验新C→D、stationarity或novelty，不支持盲加H或只取最低sigma |
| C：epsilon/DDPM + rank-clock 连续候选 | 全几何可见与密集监督有明确接口动机；现有主干可传播几何 | 不能用probe越过第一轮发现的固定反向方差偏差、rank相关末端噪声/codec混杂。没有证据要求新增species clock；不应作为默认最小方案维护 |
| H：共享V2、v/torus头与明确ODE | V2已有可用的条件/几何通路，复用它们有实证基础 | H-P的统一clock可作为下一份最小规格的优先写法；这不是H已胜出的性能证据。暖启动要计入既有两epoch，新增float API、真实共享梯度、时间尺度与主终点仍需明确；H-C的额外construction与proposal失配不设为必需 |

两份第一轮审稿已经给出可结束的修订事项：C的oracle Gaussian variance仍可收缩、不同rank会改变最终量化误差；H的ODE到有限epsilon仍不等于clean端点；原连续CIF要核查身份，或者明确选择全量decoded targets并放弃sub-bin监督。这些不是40004已验证的对象，不能用“旧几何有效”替代。

**建议就此关闭这次探针，把下一步收敛为一份最小 H-P 规格，而非继续添加诊断臂或模块。** 保留现有construction/V2作为真实竞争基线；在新实现后用一次有界的float输入/共享前后向与吞吐核验确认接口，再依据预先确定的native float或Q主终点做效果实验。若主线仍要求token-native，则在规格阶段调整选择，不等结果后挑输出。A/B保持独立假说，最终SUN不优时接受简化或否决。
