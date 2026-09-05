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
