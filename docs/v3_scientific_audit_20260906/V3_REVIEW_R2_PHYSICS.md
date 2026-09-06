# H-P33 第二轮独立物理与终点审稿

2026-09-06。对象为 [H-P33具体规格](V3_H_P33_SPECIFICATION.md)，本轮阅读版本 SHA256：`fe2fc38d57a2f2c6ff1333c78648a4fe7318266c7ba88f7b7865e3139dd8533d`。只审这一已收敛对象，不再扩展A/B/C或其他架构。本轮未调用模型、GPU、MLIP或弛豫，也未修改模型、被审规格或已冻结来源工具。

**裁决：未发现要求更换H-P33目标或solver的物理定义致命错误，可以进入已列出的有限实现验收。** 原连续CIF身份阻断已关闭，normalizer和基础数学CPU检查也已有真实完成证据。第33次读出、float主终点、Q失败规则、实际种子和float评测通路已经明确；这些不等于共享8B计算图、真实结构/graph回环或SUN已通过。后者是接下来的实现与实证工作，不应再变成无限期理论阻断。

## 1. 可关闭的事项与尚未完成的事项

| 事项 | 当前裁决 | 证据或下一步 |
|---|---|---|
| 原连续源是否能全量唯一对应 | **已关闭阻断** | 40009：train27136、val9047全部完整Q唯一匹配，零歧义、零丢行 |
| train-only normalizer与基本chart/score/readout数学 | **CPU实现检查已通过** | 40013：19项测试通过，未调用模型/GPU |
| H-P33生成对象与第33次读出 | **定义成立，明确近似** | 32步有限Euler加一次真实终端forward；不是精确posterior采样 |
| float/Q/tau的身份、裁剪和种子口径 | **定义已收口，代码回环待验** | 结构来源、Q独立record、原native_structure、失败不salvage、实际request种子 |
| 共享H/LoRA梯度、T/G/DDP/padding、显存/吞吐 | **尚待真实实现验收** | 按规格做一次有界的4–6卡短验收；CPU数学通过不替代它 |
| SUN、程序净价值、语言预训练净价值 | **未验证** | 完整冻结输出与既定比较；无事先收益保证 |

## 2. 来源身份已完成，不再要求重新证明一个不存在的缺口

[CONTINUOUS_SOURCE_IDENTITY_40009.json](evidence/CONTINUOUS_SOURCE_IDENTITY_40009.json)显示：

| split | CSV行 | 当前来源 | 核验通过 | 完整Q重复组 | 解析/编码错误 | 未分配CSV行 |
|---|---:|---:|---:|---:|---:|---:|
| train | 27136 | 27136 | 27136 | 0 | 0 | 0 |
| val | 9047 | 9047 | 9047 | 0 | 0 | 0 |

每项使用`cif`、完整`source_answer`、既有species-stable排列与全部晶格/坐标token完成匹配；未用formula、`cif.conv`、条件donor或裸ordinal消歧。原CSV哈希、prepared哈希、CIF哈希、material_id、精确排列、解析版本和连续数组均有记录。实际parser是pymatgen2025.6.14，`primitive=False, sort=False, merge_tol=0`；工具SHA为`1a097e25af007089dab1a14fa7da3820653cc6245b668c3844af936c4e75f08e`。

因此`original_cif`标签精度可按规格使用；不应继续以“原CIF没拿到”“27316/27136数字不一致”阻止实现，也无需恢复decoded混合或补训练子集。这个结论是给定两个已指纹化CSV中的来源身份/数值对齐，不是热力学、训练/验证物理独立性或SUN证书。

规格中一般性的“直接ordinal”应始终理解为有原文件绑定证明的ordinal；**本次实际通过方式全部是完整Q唯一匹配**。后续消费者应绑定40009的逐源文件SHA和全量gate，不能重新用未证明的行号替代这些映射。

## 3. 40013关闭的是数值基础，不是新神经模型

[GEOMETRY_NUMERICS_40013.json](evidence/GEOMETRY_NUMERICS_40013.json)记录19项CPU测试通过，`model_forwards=0, GPU_calls=0`。包括chart单位/回环、VP与wrapped目标、mask/源权重、逆向符号、实际NFE、点目标与Gaussian读出的差别，以及多模态反例。

normalizer只拟合27136个train来源，`ddof=0`，六个std均未命中`1e-6` floor；train标准化通道均值约0、总体std约1。train/val最大相对Gram回环误差约为`1.45e-14/1.23e-14`，全部chart有限。val没有参与拟合。它们是可复用的已完成基础产物。

尚未关闭的是：新hidden-only adapter是否数值等价于实际LLaDA的ln_f通路，G是否使用真正float状态，真实LoRA/conditioner是否获得有限梯度，以及T/G交替与padding是否正确归约。40013的预测张量可微检查不等于这些8B模型检查。

## 4. 第33次读出确实修正了旧反例，但不是p0抽样

当前注册的是在实际`(z_ε,F_ε,ε)`上新增一次forward，再取

\[
z_{out}=\alpha z_\epsilon-\sigma_L\hat v_\epsilon,
\qquad F_{out}=\operatorname{wrap}(F_\epsilon+\epsilon\hat u_\epsilon).
\]

这里`u=εs`，所以坐标增量是`ε²s`，尺度与单位正确。不能复用第32次、较早时间/不同状态的输出来声称仍为H-P33。

**点目标：** 本轮独立CPU例对`f0=.0049`、ε=.002和多个噪声值（含跨周期边界）计算正确wrapped目标，第33次读出都回到`.0049`、Q为0；六维lattice点目标也在浮点误差内恢复。R1针对“直接输出pε”的约48%换bin例，不能再直接当成H-P33最终失败率。该修改解决了那一具体反例。

**Gaussian目标：** 若clean z标准正态、score精确且概率流精确，`z_ε`仍方差1，但条件均值读出变为`α(ε)z_ε`，方差为`0.9999901304`。差别很小，却说明终端均值不是posterior sample。R1针对旧C的`beta_tilde`大幅方差收缩反例也不属于当前H-P33。

**多模态目标：** 两个等权clean坐标为`.494/.506`，在观测`.5`处wrapped score为0，读出仍为`.5`；Q从两端的49/51变成中间50。给出明确toy势

\[
E(f)=0.05\left[1-\left(\frac{f-.5}{.006}\right)^2\right]^2
\]

时，两模式能量为0，中间点能量`.05`、一阶导数0、二阶导数负。均值可以落在能量峰或不合适的几何区；小力本身也不证明极小值。此toy在真实加噪混合分布下有约`.1696%`的质量被均值读出带到两模式中间半区。这是该toy的概率，不是晶体或本模型失败率，更不提供真实CHGNet能量。

这些现象已经符合规格对“确定性local-lift读出”的限定，**不是必须再换solver的阻断**。实际误差和SUN决定它是否有用。实现宜保存终端前后状态/位移和失败原因，不能在读出失败时悄悄改用pε状态、Q结构或另一次随机样本。

低噪声也不使网络精度要求消失：局部点目标下`F_out−F0≈ε(u_hat−u*)`。距Q边界`.0001`的例子，方向误差`.05`就可改变bin；全时间、全通道平均MSE不能保证每个请求都安全。规格已有小σ敏感度和分时间验证，按这些有限检查执行即可，不增加新的物理loss。

## 5. 有限125-image shell：已声明的近似，应保留真实边界

现conditioner先计算`delta−round(delta)`，通常消除了普通周期wrap带来的整数偏移；不应误报为所有平移都会改变相对差。随后125-image枚举仍不能覆盖任意非约简cell的全部近邻或精确最短像。[实际计算](../../src/crystal_dlm/periodic_state_conditioning.py:118)

解析例`(a,b,c;α,β,γ)=(1,3,4;90°,90°,8°)`中，radius2最短自像为1 Å，`(-3,1,0)`却只有`.418539 Å`。在6 Å cutoff内，radius2只包含48个自像，完整有界枚举得到570个。该例说明无全局保证，不代表它常见于实际先验或输出。

本轮进一步使用**40013真实train normalizer**，固定256个Gaussian z，在N=1/10/20上只检查cell自像：

| N | cell先验样本 | 6 Å内存在被radius2遗漏的自像 | radius2对0.5 Å自像门假通过 |
|---|---:|---:|---:|
| 1 | 256 | 136 | 0 |
| 10 | 256 | 0 | 0 |
| 20 | 256 | 0 | 0 |

所有距离box均完成，没有budget未完成项。它是自像检查，不覆盖不同原子对；同一组z用于三个N，也不是将来256个生成请求的N分布或SUN测量。结果表明小N的6 Å特征确有实际截断；没有证据说明本小样本发生0.5 Å门假通过。

这支持按N记录实际失败/特征范围，**不要求临时加入新GNN、改变先验或扩大每步搜索**。现有有限算法仍定义明确生成器。共同几何验证的radius2也不是全局证明，不能将其标签写成严格普遍MIC证书；如增加完整距离诊断，宜对保存终态作CPU检查并保持原headline/共同R口径，不能只为新模型换有利评测。

## 6. float主通路与Q/tau：定义已修订，实际数值回环仍必需

旧exporter raw分支无条件parse`body`，会覆盖已有float `structure`；旧`graph_from_arrays`还先以0.01-bin检查重复坐标。这两项不能直接用于H float主端点。[旧exporter](../../scripts/export_programmed_path_artifacts.py:58)、[旧graph guard](../../src/scripts/sample_llada_dynamic_crystals.py:436)

独立例中，`L=30I Å`、两个站分别`.9951³/.0049³`，真实周期距离为`.509223 Å`，但旧bin keys都为0。它可通过0.5 Å几何底线却被旧bin guard拒绝；这只是几何接口反例，不是稳定材料例子。

目前规格已要求显式structure来源，直接调用同一`process_one(cif,True,False,'crystalnn',False,.01)`，保留既有refiner图预处理、N/E与全请求检查。这一refiner阶段的Niggli操作属于既有流程，与训练来源/反向过程中“不重新约简”的规则不同，不能为统一措辞而改掉它。

需要实际完成的回环很有限：

1. 以sub-bin sentinel确认primary导出的`structure`、label和N/U所读几何保持float；不经Q后再命名float。
2. Q secondary构造独立record，`structure`显式为decoded-Q；不能只改body却留下优先读取的float `structure`。
3. tau record的`structure`是实际refined终点，`native_structure`是真实float上游；regression不能因`native_body_only`而读取Q诊断。
4. 新H33的⊥必须传到底：主`body/structure`不提供可被旧逻辑复活的候选，预览另字段保存；不能被“body可解析就success=True”的legacy salvage复活。旧默认body模式行为可保持。

`label/evaluate`已有structure优先通路可复用，所以这些是可修复的数据通路验收，不是新研究机制。refiner数据入口转FP32是现有计算约定；该数值转换与强制0.01量化不能混为一谈。

Q的失败规则现已明确：length/angle clipping非零统一secondary失败，非预期coord clipping和Q后不可用cell也失败；坐标wrap及100→0 alias合法。失败保留分母，不回选clipped投影结构；primary仍是事前固定的float native。

## 7. 实际种子与触发规则已更正

[V2_ACTUAL_EVALUATION_SEEDS.json](evidence/V2_ACTUAL_EVALUATION_SEEDS.json)记录实际base seed为`20260905`，round0/candidate0、layout2，request seed由`path_seed`产生。第一条记录的实际batch size为1，故“batch cap4”不等于所有记录batch size都为4；按原分组记录实际值即可。64位seed保留整数，不经过浮点转写。

当前规格已采用这些实际键，并保持refiner的`20260905+sample_idx mod 2³²`。同request seed保证身份/复现约定；不同算法使用不同分布与随机数消费方式，不能称为完全相同的几何噪声路径。

primary两端中同一端达到整数counts`Strict≥26、Meta≥128`才触发已授权独立补评；Q secondary不是另一个可择优的触发端。规格已避免直接使用旧`strictly_exceeds_10_and_50`布尔，因其`>50%`与`Meta=128`边界不同。旧字段含义不改，新增触发器按注册整数阈值判断。

## 8. 额外两epoch与故事：系统比较允许，单因素归因尚未得到

H-P33先有V2两epoch，再有新增两epochT/G；新增T、G各54272个真实example，共108544，不能称与fresh两epoch同总训练量。独立结构来源仍是27136个；dense通道和更多noise view不是增加新的晶体知识。

目前报告37,570,905个T+G并集参数、G整tensor上界26,720,601，并区分unused rows，方向正确。fresh optimizer、weight_decay0和逐microbatch同步使行为较明确，但真实DDP/尾padding窗口还需实测；CPU19项不替代它。特别要保留每个来源和原源权重，不能用drop_last或临时按剩余有效数改写已登记风险。

G没有程序控制的同步写入顺序，T保留的是程序接口与训练任务；不能仅因T存在就宣称新增两epoch后其SUN保持不变。H-P33若整体变好，可以作为一个额外训练、连续监督、读出/表示共同改变的系统结果；这不分别证明P、语言预训练、dense监督或readout各自贡献。无需为第一轮系统判定马上增加更多分支，额外因果主张等有相应对照再写。

33次LLaDA forward与小N原V2的18次相比仍更贵，词表投影省时也须实际profile。Planner、refiner、共同R和Q诊断评测分别计费；R更严、缓存补全或换分母均不能算H-P33模型提点。

## 9. 最终处置

**来源与基础数值阻断已关闭，设计层面可通过第二轮并进入已列的实现验收。** 多模态均值、有限shell、有限32步和两epoch质量是已明确的近似/风险，不构成要求另造路线的理由。当前还不能称为“已通过模型验收”或“已取得SUN改善”。

接下来只完成已经列清的事项：真实hidden/G输入与共享梯度、T/G/source-keyed RNG及padding、float/Q/tau结构回环和失败账本、实际资源测量。通过后在已授权预算内运行唯一final policy及完整冻结评测；是否采用由真实SUN与成本决定。

本轮独立 [CPU检查脚本](evidence_v3_physics_r2/readout_and_shell_checks.py) 和[结果](evidence_v3_physics_r2/readout_and_shell_checks.json)保留点/Gaussian/多模态读出、真实normalizer下的cell-prior自像检查、旧bin guard反例及源证据SHA。所有数值为有限值；没有新模型样本、GPU或物理模型调用。
