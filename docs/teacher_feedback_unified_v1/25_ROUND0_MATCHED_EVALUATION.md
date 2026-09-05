# 首轮方法：固定256配对评估

2026-09-06 04:43 Asia/Shanghai。原生已完成，固定model494 tau800正在运行。
阶段为`round0_diagnostic`，这是首轮方法与冻结参考的比较，不能称为最终K8模型
结果，也不能独立归因于某一个架构模块或单独的后训练步骤。

## 已完成的原生结果

所有比例分母均为256请求，失败不替换。这里的原生SUN仍包含共同CHGNet评价弛豫，
只是没有先经过model494。N/U在进入公共R之前的输入晶体上计算。

| 指标 | 匹配参考39893 | 首轮方法39910 | 净变化 |
|---|---:|---:|---:|
| 可重建 | 255 | 254 | -1 |
| 终态通过完整验证 | 63 | 61 | -2 |
| Strict SUN | 6 / 2.34375% | 7 / 2.734375% | +1 / +0.390625百分点 |
| Meta SUN | 55 / 21.484375% | 57 / 22.265625% | +2 / +0.78125百分点 |
| 完整验证Strict SUN | 1 / 0.390625% | 3 / 1.171875% | +2 |
| 完整验证Meta SUN | 26 / 10.15625% | 30 / 11.71875% | +4 |

方法的标签状态：61 verified、91 not_converged、101 invalid_terminal、
2 generation_failure、1 relaxation_energy_increased。所有状态均保留。
246条hull已知、8条官方缓存未解决、2条输入不可重建；未知不填零。

Strict的配对计数为双方命中5、仅参考1、仅方法2；Meta为双方36、仅参考19、
仅方法21。净增加不代表所有原来命中的样本都保持命中。

## 配对物理指标

脚本已核对全部256个请求的组成、Plan、program、prompt、N、seed、batch与候选
索引；公共R、末态一致性、N/U源码hash及hull缓存协议也匹配。
以下delta均为“方法减参考”，负值表示该量下降。

| 指标 | 有效配对数 | 参考均值 | 方法均值 | 平均delta | delta中位数 |
|---|---:|---:|---:|---:|---:|
| 原始能量 eV/atom | 253 | 2.940417 | 2.982739 | +0.042321 | +0.177143 |
| 原始最大力 eV/Å | 253 | 104.088832 | 64.315225 | -39.773606 | 约0 |
| 原始最大应力 GPa | 253 | 225.619776 | 112.058878 | -113.560897 | +5.370413 |
| A：原生—终态能差 eV/atom | 37 | 5.613625 | 5.739394 | +0.125768 | 约0 |
| B：同组成终态能量差 eV/atom | 37 | -4.222666 | -4.278415 | -0.055748 | 约0 |
| 实际弛豫步数 | 37 | 57.459459 | 70.054054 | +12.594595 | +2 |

最后三行只用**双方终态都通过完整验证的37例**，排除其余219请求；不能泛化成
整个256的无条件A/B估计。B行前两列显示的是eR均值，不是绝对hull差；
同组成配对时delta B=delta eR，因此最后两列可作为B差。

原始力/应力均值下降并不表示普遍改善：力配对delta中位数约0；应力155例升高、
97例降低、1例不变，delta中位数反而为正。不能只选均值降幅报告。
在37例验证交集中，A为17降/19升/1平，B为19降/17升/1平；
弛豫步数为10降/19升/8平。

## 当前结论与下一步

SUN净增加1和2例，变化很小；原生仍明显未到10%/50%。验证交集上B均值降低，
A和步数均值升高，**尚无学生A/B共同改善的证据**。教师训练池中的双目标可行性
没有自动转化为开发集上的双目标改进。这不作为挑中间checkpoint、改阈值或改分母
的理由；按已登记的K8完整标注和最后两遍继续，随后独立报告最终结果。

tau800仍在运行，本页不会用参考tau结果冒充新方法结果。连续diffusion扩展
保持下一轮范围，不改变本轮model494。

## 可复核产物

远端ROOT为`/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/proposal_realization_candidates_20260826/grounding`。

- 方法：`ROOT/runs/spad_state_eval_method_39910/evaluation/EVALUATION_FINAL.json`，
  以及同目录`attempt_results.jsonl`与`_SUCCESS`。
- 参考：`ROOT/runs/spad_state_reference_verification_39893/native-evaluation`。
- 已完成的提前配对报告：`ROOT/runs/spad_state_round0_eval_39910/native-early-comparison/COMPARISON_FINAL.json`、
  `comparison.md`及`_SUCCESS`。这只读取同一批完整评估数据，没有新增生成或物理标签。
- 整个39910结束后，wrapper会另存`native-comparison`和`tau800-comparison`，
  提前报告使用独立目录以免与wrapper写入冲突。
