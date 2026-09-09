# KEEP/EDIT主线进度

当前执行计划见[固定G/F的KEEP/EDIT尝试与后续1000条计划](docs/r03_paper_story_20260907/EDITOR_T2T_TRIAL_PLAN_20260910.md)。详细实验讨论见[EXPERT_REPAIR_DISCUSSION.md](docs/r03_paper_story_20260907/EXPERT_REPAIR_DISCUSSION.md)。

2026-09-10更新：固定G/F的KEEP/EDIT试验完整结束，未通过采用条件。按用户约定冻结回退到原已完成方法，1000条S0＋三次G/E更新已正式启动。后续仅修阻断执行的小bug，主判断仍为Stable和SUN。

固定G3/F输入为Stable 32、SUN 28；旧E3为Stable 33、SUN 32。四个新候选在24个预定DEV策略下全部KEEP；冻结选择在52条FINAL上为Stable 5、SUN 5，与KEEP及旧E3持平，未获得新增Stable。全256条新输出为32/28，低于旧E3的33/32。本次划分只保证不参与此次E更新，不能称为旧RSI从未见过的独立测试。

四组训练实际完成内容更新2/8/0/8次，均完成552次头更新。关闭KL硬停增加了覆盖，但未带来FINAL Stable晋升。详细分析发现动作从旧E3的单原子局部编辑大幅转向整胞编辑；所有新模型的最高接受概率都低于0.5，而且FINAL没有可供接受的Stable晋升提案。仅训练头也出现范围变化，说明KL不是唯一瓶颈。完整对照、逐条原因、源码/路径小bug修复和原失败回执见[试验证据与结果](docs/r03_paper_story_20260907/EDITOR_T2T_TRIAL_RESULTS_20260910.md)。

1000条检查确认1000个不同TRAIN来源、987个不同约化组成、与MAIN验证组成零重叠，保留原256条Plan/prompt/seed前缀。G从B0开始、E使用新初始化，不导入试验权重或验证反馈；采用旧偏好/局部提案/接受规则，关闭试验性历史教师、模式归一化、头续训和稠密T2T路径。固定配方和三份配置哈希见[启动回执](docs/r03_paper_story_20260907/receipts/CLEAN_1000_FROZEN_METHOD_AND_LAUNCH_20260910.json)。

正式调度于北京时间2026-09-10 06:01:15启动，绝对截止18:01:15；最多6张A800及3个并发/排队作业。两路RAW生成、物理标注和评分均已完成：S0主路Stable/SUN为25/25（各2.5%），第二RAW对照为19/19（各1.9%），分母均为1000。当前继续F800，尚未完成S0的KEEP/EDIT或权重更新。前256条新生成RAW与旧实验在body、生成状态及调用数上全部逐条一致，见[前缀复核](docs/r03_paper_story_20260907/receipts/CLEAN_1000_S0_FRESH_PREFIX_AUDIT_20260910.json)。

扩容时遗漏的参考缓存覆盖已修复：原242体系缓存保留原路径，按相同2026.04.13版本与原协议扩展至925体系（895可评分、30官方参考缺失）。原输入、物理标签和配置字节未变；失败评分已归档，仅重试评分，科学源码仍固定657985c。37项相关检查及真实覆盖检查通过，恢复后的两路评分均正常完成，见[修复回执](docs/r03_paper_story_20260907/receipts/CLEAN_1000_HULL_COVERAGE_RECOVERY_20260910.json)。

1000条的逐阶段指标、同Plan获得/损失、六次真实训练和资源记录见[逐轮结果报告](docs/r03_paper_story_20260907/CLEAN_1000_RESULTS_20260910.md)。已完成固定G/F试验的48个作业共分配8.0283 GPU小时，含两次失败开销，见[试验资源回执](docs/r03_paper_story_20260907/receipts/EDITOR_T2T_TRIAL_RESOURCE_USAGE_20260910.json)。

本文件将在固定输入比较、成功或失败论证、方法冻结和1000条每轮结果完成后补充相应回执与证据。
