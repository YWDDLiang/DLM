# K4/K8 有界短接触项：快速概率与执行复核

2026-09-07。仅复核 [唯一规格](K4_K8_SOFT_CONTACT_CANDIDATE.md) 与既有 runtime，没有改模型、采样生产代码或启动作业。机器规格 SHA256 `5789ed98a16627dbc0ad052f18bbd5c992604b339a006d2475e8231b6d8d9336`；冻结半径表 SHA256 `74bf9289545246bead1fa2b2a70bd5cbd73498ae3cef74f5d7ec24007c78bec7`。

**裁决：定义是明确的支持内推理分布；没有发现需要再开理论路线的阻断。** 应先完成少量真实 prefix 的数值/回放验收，再以同256 raw/refined结果判断效用。这里没有 SUN 保证。

## 1. 概率、温度与 alias

对原合法 canonical 集合 A(s)，原策略已含 schema、0.5 Å hard mask、alias 折叠及 T=.7。指定 `0≤b(a)≤1` 后

`q(a|s) = p(a|s) exp[-b(a)] / E_p exp[-b]`

合法且归一。实际 raw post-processing logits 应减 **T·b=.7b**，随后仍用原 T=.7 和原 Gumbel；直接减 b 会得到另一强度。原 alias 的 `logaddexp` 必须先做，几何 cost 只加到 canonical Z=0一份物理类别；不能分别处罚0与100再合并。

固定状态下，`log(q/p)=-b-log E_p exp(-b)`，因而其绝对值不超过1。且 `E_q[b]≤E_p[b]`；但某个 b>0 的 action 仍可能因归一化而增加绝对概率。不能把这个局部成本关系升级为整轨迹、能量或 SUN 改善。

只Z、已知cell/XY、其它当前完整XYZ站点，以及 max而非邻站求和的定义足够明确。未来尚未重掩码的站点若在当前canvas完整，也属于可见邻站；不能另改为“仅本phase已写站点”。不借 old 填 MASK，不把新软阈加进 hard support，不改 P、active位置、阶段或salt规则。

## 2. 零惩罚保持论证成立，但必须按路径状态陈述

原带seed sampler等价比较 `exp(l_a)/(-log U_a)^T`。新分数为原分数乘 `exp(-T b_a)`。在**同prefix、同Gumbel数组**下，原 winner 的 b=0则分数不变，竞争者只会下降，因此该draw保留（包括稳定的原tie规则）。若整条原路径每个已选动作均满足此条件，可以逐步归纳保持路径。

最终晶体无短接触、verified或SUN不推出上述逐步条件：中途接触可能被后续修订消除。不能按最终结果承诺保留，也不能按case ID关闭惩罚。全合法vector b=0时按规格直接返回原tensor，保留原dtype与随机行为。

## 3. 唯一需要特别防止的实现陷阱

- **先取原dtype的legal mask，再计算soft项。** BF16 `finfo.min` 升到FP32仍是很负的有限值，却大于FP32的 `finfo.min`；cast后重新用新dtype sentinel识别合法，会把原masked/alias100误认成合法。
- 当前 `_draw` 的float64 exp/Gumbel与log_softmax仍把这个旧极小值处理成零质量，因此“保留原屏蔽数值”对这条确切路径足够；audit/replay必须沿用cast前legal mask，不能依赖新的dtype_min判断。必要时显式携带原支持位图。原unavailable集合始终保留。
- 为避免小惩罚被BF16舍入抹掉，可将半精度合法向量升FP32；本已FP64的输入不应降精度。b=0 action的logit不得上升或下降；其它logit只降，实际数值偏差按运算dtype记录。
- 半径缺失按唯一规格在条件构造时明确失败，不换表/猜值/静默跳过。应验证本次256和声明生产元素词表覆盖；缺配置不能靠删请求或补样解决。disabled旧入口不能新增这项条件拒绝。

这些是小型接口/数值检查，不需要改变所选公式。

## 4. 最小可执行验收与回放证据

独立子类在 `super().processed_logits` 后仅修改当前合法Z向量，保留原 `_draw`，是合适的接入点；旧路径默认off。无需第二模型、额外DLM forward、MLIP或训练。

CPU及既定前4请求的真实prefix应覆盖：

1. b边界/归一化/精确温度位置，alias0/100同物理距离；半径表及单位一致。
2. X/Y/cell与缺cell/XY/完整邻站状态零作用；原0.5 hard拒绝、unavailable及rollback不变。
3. all-b-zero逐值与dtype相同；原选中b=0时同seed/salt选中token相同；非零项只减少对应logit。
4. 同一修改后的vector产生采样和 `trace.log_probability`。fresh replay使用同一子类/config/radii hash，不能用旧无惩罚的 `process_scalar_path_logits` 重算新trace。

需保存原trace和checkpoint标识、P/phase/decision、current/old/active、合法canonical IDs、候选Z与b/最坏pair、原新logits/logp或可核数组、温度与seed/salt、batch membership、dtype和半径hash。失败与回滚中的尝试动作同样保留。单个token选中不变时logp仍可能因normalizer改变，必须记录新值；终态结构概率仍不等于一条trace的logp。

如果上述验收失败，修实现与本规格的偏差即可；如果通过而SUN无增益，保留这一实测结论，不以更多惩罚、改共同R或逐请求择优掩盖。
