# KEEP/EDIT主线进度

当前执行计划见[固定G/F的KEEP/EDIT尝试与后续1000条计划](docs/r03_paper_story_20260907/EDITOR_T2T_TRIAL_PLAN_20260910.md)。详细实验讨论见[EXPERT_REPAIR_DISCUSSION.md](docs/r03_paper_story_20260907/EXPERT_REPAIR_DISCUSSION.md)。

2026-09-10更新：固定G/F的KEEP/EDIT试验完整结束，未通过采用条件。按用户约定冻结回退到原已完成方法，1000条S0＋三次G/E更新已正式启动。后续仅修阻断执行的小bug，主判断仍为Stable和SUN。

固定G3/F输入为Stable 32、SUN 28；旧E3为Stable 33、SUN 32。四个新候选在24个预定DEV策略下全部KEEP；冻结选择在52条FINAL上为Stable 5、SUN 5，与KEEP及旧E3持平，未获得新增Stable。全256条新输出为32/28，低于旧E3的33/32。本次划分只保证不参与此次E更新，不能称为旧RSI从未见过的独立测试。

四组训练实际完成内容更新2/8/0/8次，均完成552次头更新。关闭KL硬停增加了覆盖，但未带来FINAL Stable晋升。详细分析发现动作从旧E3的单原子局部编辑大幅转向整胞编辑；所有新模型的最高接受概率都低于0.5，而且FINAL没有可供接受的Stable晋升提案。仅训练头也出现范围变化，说明KL不是唯一瓶颈。完整对照、逐条原因、源码/路径小bug修复和原失败回执见[试验证据与结果](docs/r03_paper_story_20260907/EDITOR_T2T_TRIAL_RESULTS_20260910.md)。

1000条检查确认1000个不同TRAIN来源、987个不同约化组成、与MAIN验证组成零重叠，保留原256条Plan/prompt/seed前缀。G从B0开始、E使用新初始化，不导入试验权重或验证反馈；采用旧偏好/局部提案/接受规则，关闭试验性历史教师、模式归一化、头续训和稠密T2T路径。固定配方和三份配置哈希见[启动回执](docs/r03_paper_story_20260907/receipts/CLEAN_1000_FROZEN_METHOD_AND_LAUNCH_20260910.json)。

正式调度于北京时间2026-09-10 06:01:15启动，绝对截止18:01:15；最多6张A800及3个并发/排队作业。S0主路与第二RAW对照均已完成1000条生成记录，各有998/997条可用结构，其余几何生成失败保留在分母中。两个作业均正常结束，合计分配3.7925 GPU小时；两路RAW物理标注也已完整结束。评分因旧参考缓存未覆盖扩展面板而暂停，正在按原协议补齐覆盖；尚无完整S0评分或权重更新结果。前256条新生成RAW与旧实验在body、生成状态及调用数上全部逐条一致，见[前缀复核](docs/r03_paper_story_20260907/receipts/CLEAN_1000_S0_FRESH_PREFIX_AUDIT_20260910.json)。

1000条的逐阶段指标、同Plan获得/损失、六次真实训练和资源记录见[逐轮结果报告](docs/r03_paper_story_20260907/CLEAN_1000_RESULTS_20260910.md)。已完成固定G/F试验的48个作业共分配8.0283 GPU小时，含两次失败开销，见[试验资源回执](docs/r03_paper_story_20260907/receipts/EDITOR_T2T_TRIAL_RESOURCE_USAGE_20260910.json)。

本文件将在固定输入比较、成功或失败论证、方法冻结和1000条每轮结果完成后补充相应回执与证据。
