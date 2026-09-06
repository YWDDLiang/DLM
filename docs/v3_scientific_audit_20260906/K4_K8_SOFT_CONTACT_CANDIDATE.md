# K4/K8 唯一短接触候选：有界坐标 action 惩罚

2026-09-07。候选定义先固定，再做真实 prefix 验收与完整 SUN；不按已知 case 或 SUN 切换策略，不扫惩罚系数。它适用于既有 K4/K8，权重、P、hard support、共同 R/hull/N-U 和全部请求分母保持原合同。新增项是生成策略中的便宜几何偏好，无推理期 MLIP、重新训练、整晶体筛选或第二模型。

## 1. 数据动机与不使用硬过滤的原因

已有 K4/K8 native 的 invalid_terminal 分别为101/77。按下述弱元素尺度检查，其原生结构有176/164条存在短接触，覆盖100/75条 invalid_terminal。但被触发的请求中，分别有93/87条在原τ800流程达到Meta SUN，且各有2条native verified Strict SUN。直接拒绝这些请求、删除旧轨迹或把软阈值改为新的硬 support，可能破坏有用初态；本候选仅对 action 作有上限的概率修正。

这只是接触与结果的关联。真实低能晶体、离子/金属键和不同配位不能由一个半径表判定；规则不声称化学正确性或 SUN 保证。

## 2. 唯一公式与单位

半径采用 [冻结96元素表](evidence_v3_failure_20260907/k4k8_physics/covalent_radii.json)，单位 Å，SHA256：

`74bf9289545246bead1fa2b2a70bd5cbd73498ae3cef74f5d7ec24007c78bec7`。

它来自当前 pymatgen `CovalentRadius.radius`，其文档注明 [Cordero 等，2008](https://pubs.rsc.org/en/content/articlepdf/2008/dt/b801115j)。精确数值以冻结JSON为准，例如 H=.31、C=.73、O=.66、K=2.03、Zn=1.22、Pu=1.87；不切换到另一半径表或估算缺值。下面的二分之一系数是本候选的固定启发式，不是该论文给出的普适非重叠下界。来源细节见 [radii_source.json](evidence_v3_failure_20260907/k4k8_physics/radii_source.json)。

对当前补全站点 i 与另一当前完整站点 j，定义：

`dsoft_ij = max(0.5 Å, 0.5 * (r_i + r_j))`。

候选 a 使用其给定 Z，加上当前已写 X/Y。距离 `d_ij(a)` 使用当前已完整晶格与已有125像 PBC距离实现。只考虑不同站点；固定晶格的自像距离不随Z改变，不加入本action偏好。定义：

`b(a) = min(1, max_j max(0, ln(dsoft_ij / d_ij(a))))`。

无其它完整站点时 `b(a)=0`。对原规则留下的合法 canonical action：

`p_new(a | s) = p_old(a | s) * exp(-b(a)) / sum_c[p_old(c | s)*exp(-b(c))]`。

固定强度 λ=1、最大惩罚1 natural-log unit。这里 p_old 已含原 alias/hard support 与温度 T=0.7；实现等价于在原 `process_path_logits` 后，对合法 action 的原logit减 `T*b(a)`。不在alias合并前各罚一次，不再改温度，不添加第二个λ。masked/非schema/alias100位置的原屏蔽值完全保留，惩罚不创建或删除合法action。

## 3. 作用范围

- **仅Z补全 action**：当前站的 X/Y 已知、Z待写，且当前六晶格字段均有效。X/Y、N/E、晶格动作均不改。
- 对其它站只用当前 canvas 中 XYZ 全部已知者；不借 immutable old geometry 补 MASK，不猜未知坐标或候选的未来环境。
- 在 K4/K8 原有 construct、cooperative、closure 中统一生效；若原路线本就启用 full_cell_repair，其Z action同样定义。候选不额外打开任何阶段。
- 缺完整晶格、缺X/Y或没有完整其它站时返回零惩罚。当前 hard checker 已判无合法动作的状态继续原失败/回滚行为。
- 半径在构造本次化学条件时查表，并在开始采样前核对本cohort及声明词表的覆盖。发现缺值是配置验证失败，不能默默套半径、跳过source或补样；不能仅凭表有96项推断未来cohort覆盖。

已有实现位置：[ProgrammedPathSampler.processed_logits](../../src/crystal_dlm/programmed_path_runtime.py) 返回原支持/alias变换后的logits；[_apply_pbc_min_distance_mask](../../src/crystal_dlm/llada_generation.py) 已提供“当前cell、XY、Z候选、完整邻站”的取值顺序。实现应以显式候选开关/子类接入，旧路径默认行为不变，不修改共享的硬几何验收语义。

## 4. 什么能保持，什么不能保证

**零项保持。** 整个合法vector的b都为0时，直接返回原tensor，不经过额外dtype转换或重新归一化；原分布与同seed抽样完全不变。

合法vector的b若严格为同一正数，常数也会在归一化中抵消；实现同样保留原policy，并记录原b和 `constant_penalty_removed`。不把“近似常数”视为常数，也不因此改变惩罚尺度。

**同噪声的动作保持。** 给定相同prefix与原Gumbel噪声，如果原选中action的b=0，它的score不降，竞争action只可能降分，所以该次draw保留。若一条原轨迹每一步被选action都满足零惩罚，则可逐步保持整条轨迹。数值实现须检验此单调性。

归一化后并非每个b>0 action的绝对概率都下降；准确说法是相对odds按惩罚差改变、固定状态的 `E_pnew[b] <= E_pold[b]`，每个合法action的log概率变化在[-1,1]内。这个局部关系不能外推成整条轨迹或SUN改善。

**没有已知稳定性 oracle。** 最终晶体无短接触不代表轨迹中每一步都无短接触；中途旧几何可能被后续修订纠正。因此不能仅凭原终态verified或SUN，承诺这个病例会保留。惩罚非零的可靠旧成功是风险对象，必须报告，但不能据此按ID关闭规则。使用max而非邻站求和避免仅因已知站数更多而线性放大惩罚；仍不能保证不同N的策略漂移相同。

## 5. 最小真实 prefix 探针

先使用ROOT已固定的原请求序前4条及其真实prefix包；失败请求也保留，不换到成功源。所有原有phase取早/中/末Z状态，去重。K4/K8均用同一来源序选择，不从“损失病例”挑选，不在探针后扫参数。如必要分支没有覆盖，仅补事先按N分层/hash选择的状态，不按SUN选择。

每个状态保存：checkpoint/源trace摘要、group与phase/decision_index、current body/old body及已知mask、原P位置、N与元素、当前L和完整邻站索引；合法canonical token ID和候选Z；各候选 `min_j(d_ij/dsoft_ij)`、最坏pair、b；原/新logit与log probability（或可复核数组摘要和误差）；原/新惩罚区域总概率、KL/TV、normalizer；温度、salt/seed、实际batch元数据；同噪声原/新选中token及其b。

验收只检查：support与alias不变、概率归一化、0≤b≤1、logit单调不增、全零状态逐值相同、选中零惩罚action的coupled draw保持、所有trace log_probability使用新分布并能fresh replay。不是要求loss或SUN在探针上提升。没有完整邻站或晶格的状态应实际测到零作用。无需MLIP或新弛豫。

旧运行的 `sampling_layout_world_size=null` 应照实保存；复现原2卡逻辑分桶时不要把旧记录改填2。继续保留原请求序、seed、P、固定batch会员和原hard支持。若CPU/真实prefix验收失败，只修实现与本定义的偏差，不改公式来迎合个别样本。

## 6. 完整验证与成本

完成上述验收后，先使用既有K4完整256的raw/τ800作为对照，在同一全部256请求上测这一固定候选；对K8应用同一公式时也不另调尺度。K4/K8原策略已完成结果可复用。报告新增/丢失的可靠S、SUN、invalid终态、N/U、unknown与耗时；如果总体未改善，就保留原K4/K8最好结果，不事后逐请求拼接端点。

每个Z补全增加小型PBC候选距离计算，没有额外DLM forward、能量排序或候选重采；真实CPU/GPU吞吐仍需测量。这个候选改善的是生成概率偏好，不改model494、共同R、hull或N-U，也不把工程验收当成稳定性收益。
