# 当前架构、科学对象与恢复执行

2026-09-05 22:30 Asia/Shanghai。新任务获准继续。工作树为
`D:\codex_work\ai4s\DLM_llama_programmed_basin_closure`，分支为
`codex/llama-programmed-basin-closure`；恢复时本地和 GitHub 均为 `a5fb9cf`。
旧任务已中断、heartbeat 为 PAUSED，避免两个任务同时改代码/提交。原截止
2026-09-06 19:19 不延期，资源仍为最多6 A800、每卡4 CPU、最多两个job。

## 初心和实际架构

科学问题是：科学 LLM 如何编排 DLM 的生成与非因果修订，让精确化学、未来
晶体上下文与周期几何共同决定最终结构。当前瓶颈是可执行有效域远大于物理
低能近平衡域；不能把解析成功、碰撞底线或后处理成功当成原生稳定性学习。

| 模块 | 已核对的实际职责 | 本轮状态 |
|---|---|---|
| C3FD | 化学ledger/可达支持，proposal、species/count及soft Plan基础分布；无精细几何logits | 冻结 |
| Typed Planner-Llama | typed embedding进Llama，在同一合法支持上以unit-weight PoE提供残差偏好 | 冻结 |
| Species pointer | terminal hidden、元素/数量及soft IDs→物种排列；teacher来自MP20 contact tree，不是能量顺序 | 冻结 |
| Plan/compiler | 语义字段接DLM自己的tokenizer；排列映射canonical slots、anchors和构造/闭合顺序 | 确定性 |
| DLM/SPAD | 双向Transformer，exact 7+4N，预填N/E，预测六晶格和XYZ token | LoRA训练，大embedding/head冻结 |
| Periodic conditioner | FP32 metric/体积、旧周期坐标、物种/program rank和邻域，显式embedding增量 | 与LoRA一起训练 |
| 联合事务 | 同时mask cell+程序/几何选出的约半胞XYZ，逐scalar采样，整体验收/回滚，再反向物种闭合 | 保留完整attempt事件 |
| CHGNet | 原生e/F/stress及同一势的变胞终态 | 仅离线标注/共同评价 |
| model494 tau800 | 完整原生晶体之后的固定可选保底 | 单列结果 |

对应实现：`c3fd_llama_typed_planner.py`、`species_program_pointer.py`、
`c3fd_native_plan.py`、`spad_program.py`、`programmed_path_runtime.py`、
`state_conditioned_model.py`、`periodic_state_conditioning.py`。
历史PMTR/K10/SPAD-E不作为当前执行入口；不宣称在线LLM控制或端到端联合训练。

## 科学对象和训练目标

对象是在固定组成与程序下，晶格和周期多原子坐标的联合生成分布。
物理标签在原生终点及其公共弛豫终点上定义，实际拟合变量是完整执行路径的
概率。不同尝试可能因回滚落到同一结构，所以保留尝试中的全部随机决策，
不把最终结构的概率误当成最后一次接受动作的概率。CHGNet只提供离线监督，
不在原生部署时替模型挑选结构。
实际warmup每source抽一个协同/闭合scalar状态，以同一MP20晶体为clean目标做
全词表CE；旧数值输入保留corrupted source。它不是能量后训练。

下一阶段每条件四条完整独立路径。A=e0−eR，B=eR−h(c)；组内中心化精确消去
hull常数。固定条件等权，先求KL≤0.2下最大共同改善，再取半程收益求最小KL
teacher。学生目标是 `−mean_condition sum_path w sum_decision log pθ`，
概率分母为实际完整合法token支持；不能对K4路径再做分类softmax或使用A+B。
每遍每path分层六决策，按抽样概率校正且不再除路径长度；每四个路径更新插
一个真实MP20 CE anchor。两遍后最多一次train-only刷新，再两遍。
teacher双降不保证学生/SUN双升；force/stress、实际步数、Stable/N/U/SUN另报。

## 恢复后的真实状态

- 39853成功marker/TRAIN_FINAL/LoRA和conditioner文件已读取核实：27136 source、
  1696 updates；LoRA14680064、conditioner1267200参数，训练记录1298.49秒。
- 39857成功marker/manifest已核实：typed可用24558条、metadata排除0，选择1024
  个不同组成；没有重采组成/读取outcome，预测soft Plan用于pointer。
- 严格canonical预填、完整路径CLI、occurrence汇总和新进程全probe决策回放已实现。
- 标注草稿import已修复，固定零外压FIRE/FrechetCellFilter、全晶胞自由度，保存
  真实optimizer返回、原/终态单位和压缩轨迹；重复终点只共享物理计算，路径重数保留。
- 74项组合CPU测试通过。集群实际CHGNet包0.4.2、ASE3.28.0；势固定CHGNet0.3.0。
- 39867在模型加载前因旧Git不支持-C退出；15dc3d9修复快照命令。
- 39869实际完成两条全路径和新进程base+LoRA+conditioner回放：51+102=153个
  decision全部误差0，非零state residual；label端将非法终态误分类导致总job失败。
- 0840154修复label分类并保留其能量/停止状态。39871复用39869路径，33秒完成
  标注协议检查：1 verified、1 invalid_terminal；二者都完整记账，不是性能样本。
- 四卡39872正在采固定前128组×K4。完整1024池、刷新及新SUN仍未完成。
- 全路径trainer已接通，正确HT/minibatch条件均值、零权padding、4:1 MP20 CE、
  optimizer续训和不可选用的工程checkpoint；需真实数据梯度检查后才正式训练。

新run保存明确commit的代码快照。真实检查通过后先固定0..127组，再完成同一池
128..1023组；不根据能量更换条件。原有`apply_pmtr_fixed_bodies.py`及测试原样保留。

## 路径训练实际短检查与等价优化

39872已完成：512请求、509成功、3失败，9分21秒。39873正在标注全部512条。
39874在模型加载前受旧Bash `nounset` 空数组行为影响，aba489c修复；39875从同一
warmup和真实train轨迹完成4个路径更新+1个CE更新，参数梯度有限，`eligible_policy=false`。
这次只用均匀工程权重，没有能量teacher。训练部分58.38秒（80 scalar），包括保存75.37秒。
微批2+不同长度padding时，初始16 scalar与采样记录的最大logp差0.07679 nat，不能写成0。

现在只对实际active logits保留autograd；离散schema/几何支持仍调用同一实现，alias
logaddexp保持梯度。FP32/BF16的dense/scalar logits和梯度逐元素一致，LoRA/conditioner
端到端梯度测试通过。按无padding的相同序列长度组batch、微批4×4卡、accumulation1，
全局batch仍16；不足的bucket仅补零权重，loss使用相应总slot数保持condition均值。
这项等价工程优化需下一次真实短检查，不按SUN调参。

评估参考入口直接调用原来的 `revise_spad_cell` 和 `revise_spad_species_blocks`，
保留独立cell拒绝边界；用共同的构造流程和对应随机数地址，避免错误地把
`cooperative=False`当成已经执行cell的参考。参考ledger仅作评价，不冒充完整训练trace。

## 首128组诊断完成，继续完整池

39877优化后真实四卡短检查完成2分48秒：4path+1CE、梯度有限；训练27.93秒，
含保存39.16秒。初始16决策最大logp差9.999e-7，rank0峰值19.21GiB。
这仍是`eligible_policy=false`的均匀工程检查，没有能量后训练效果。

39873标注31分20秒完成：512请求，111verified、205not_converged、193invalid_terminal、
3generation_failure，无软件异常。65/128条件至少一条验证路径，30条件有多条；
只有4条件四条都验证。65个验证条件上的诊断teacher为optimal：rho_max1.08126，
半程目标0.054063eV/atom，平均A/B分别-0.054063/-0.054063eV/atom，KL0.006744，
ESS87.94。该128组teacher明确diagnostic_only，不能充当完整1024条件的正式teacher。
509成功路径里491个联合事务实际接受且改变结构。39878已开始其余896条件×4；
与首128的末段标注重叠只改变调度，所有条件和参数保持冻结。

后续标签入口支持按实际额度使用1..6卡、或两个不重叠的4+2卡分片；不修改FIRE/
收敛标准来提高验证率。CPU组合测试最近一套87项通过。新CIF/refiner接入与固定评价
CLI已实现，实际参考运行待验证。原Stable/SUN保持终态能量阈值定义，同时单列
满足optimizer/force/stress/几何完整验证的子集；不能把低能但未验证终态当作已经证明
低能极小值的证据。未知hull/能量不填零，输入N/U仍在共同CHGNet弛豫前计算。

## 全量路径完成，参考原生结果冻结

2026-09-06 00:25：39878完成58分47秒，剩余896条件的3584请求里3561成功，
23失败。与39872合计1024条件、4096请求、4070成功、26失败。39885用六卡
标注余下3584条，之后才求完整1024条件teacher、启动正式两遍训练。

共同协议参考39884完成28分35秒：256请求、255可重建，终态63verified、
98not_converged、94invalid_terminal、1generation_failure。原定义的Strict/Meta
SUN分别6/256（2.34375%）和55/256（21.484375%）；另报完整验证子集1/256和
26/256。247个输入有已知hull，8个缓存明确未解决，1个不可重建。它是旧closure
参考，新双目标学生还没有SUN。参考tau800将在正式四卡训练时用另两卡执行。

最终配对报告会校验相同Plan/program、组成、源序号和随机种子，并按请求匹配。
原生e/F/stress取双方有限值交集；A/B和终态物理量另列双方验证交集及排除数。
同组成的delta B可直接由delta eR得到，不需要给未知hull填值；表中绝对能量均值
明确写为eR。SUN仍以全部256请求为分母，不能用验证交集替代分母。

00:35：旧任务仍为interrupted/notLoaded。原有十分钟heartbeat
`llm-dlm-sun-24h`已迁到当前任务`01a071d2-ebe3-79b1-a311-14a5107cb6c6`
并激活，保留原截止19:19。它会根据真实markers接续完整teacher、两轮训练和
注册评估，状态无变化时安静，仅阶段终态、失败或需要处理时通知；不恢复旧任务。

01:02：根据用户最新数据充分性与提效要求，首轮保留K4，唯一刷新扩大为同1024
条件K8；相关代码已推送并部署，自动化已更新。当前标签累计2624/4096、其中630
verified。实际覆盖评估、修订后的训练预算和交付检查点见
[20数据充分性与交付](20_DATA_SUFFICIENCY_AND_DELIVERY_20260906.md)，其K4/K8预算
覆盖本文前面的原K4/K4安排。不再追加小数据消融。
