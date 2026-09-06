# V3 失效后的有界 fallback 准备

2026-09-06创建，09-07更新。**仅作条件触发准备，fallback尚未改代码、训练或提交 GPU。** 40064训练已完整完成，既定 G float raw/refined 与 secondary Q raw 由40066评测继续执行。待两主端 SUN 出齐并按用户条件判断效果不足后，才启用下列顺序；不根据中间 loss 或 Q 的较好结果提前切换主端点。

本次只保留两个值得优先验证的改动，**首选①：G 辅助训练后，以 T construction 部署**。旧 K4 作为保底参照保留；复用既有 K4/K8 checkpoint 不等于重新采集或训练自生成能量 teacher，后者没有被本条件自动解禁。

## 决策依据

同一开发256的 Strict/Meta SUN：K4 raw **7/57**、refined **19/126**；K8 **11/66、17/123**；V2 construction raw **12/44**，full-cell repair 后 **8/50**，refined **16/113**。K8 native 增益没有完整保留到 refiner；V2 修订是 Strict−4、Meta+6 的权衡。V2 construction→tau 尚未测，不能填成其他分支的 refined 结果。[casebook](EXPERIMENT_CHANGE_CASEBOOK.md)、[K4/K8 配对分析](historical/docs/periodic_self_repair_v1/K8_RESULTS_AND_REGRESSION_ANALYSIS_20260906.md)、[V2 构造/修订](evidence/V2_CONSTRUCTION_AND_REPAIR_39998.json)。

这些结果支持恢复可比较的旧路径并限制改动数量；它们没有证明某个稳定性 trick 必然有效。

## ① 首选：G 只提供训练辅助，部署完整 T 构造

**作用点。** 使用40064按登记完成的同一最终权重及冻结上游 C/S/P；调用其保留的 T construction，完整输出原 `7+4N` 晶体。终点取构造结果，不追加 full-cell repair；随后对这一端点执行同一固定 model494/refined 评估。这里改变部署读出，保留当前已经付出的联合训练成本；接口映射另有记录，本文不重复。

**支持与反证。** T 的 CE 锚点、数值词表和真实程序路径一直保留；G 的密集几何监督可能改善共享表征，而 G 的先验输运、有限积分或末端均值读出可能成为另一层瓶颈。V2 的同次构造端 Strict 更高，是避免默认追加整胞修订的直接线索。反面是 G/T 可能互相干扰，旧离散精度限制仍在，V2 构造的 Meta 也较低；工程上保留 T 不证明其 SUN 已改善。

**与旧失败的区别。** 不重新做更多局部修订、扩大 K、训练偏好 critic 或从随机8B再来一次；把“几何辅助学习是否能改善 token 生成”与“连续 sampler 是否有效”分开。效果仍可能失败，不能因换了终点就宣称规避了全部旧问题。

**最小配对测试／训练预算。** 无新增训练。固定已有256条件与 T 对照的逐请求随机接口，比较“V3-final T construction”与“V2 T construction”，复用可严格对应的旧对照；两者均评 raw/refined，失败保留原分母。G/Q 原结果完整保留。若 T 有实质 SUN 信号，再登记同初始化、同总 forward 预算的 token-only continuation 作论文归因对照；在此之前，不能把额外 T 训练和 G 辅助的收益单独归给 G。

**论文关系。** 最能维持 LLM 程序→DLM 数值生成的主问题；可研究周期几何辅助训练对离散生成的迁移。若只有 T 有效，应诚实降低连续主采样器的主张，而不是把 T 的成绩写到 G 名下。

## ② 次选：对既有 K4 学生使用固定参考策略混合

**作用点。** 从 K4 运行记录找出其真实采样父 checkpoint `π₀`，先核对相同 tokenizer、输入/程序与可执行 action support；不把表中的39893参考自动当成训练父版本。在每个实际访问的同一 `(C,S,P,U,prefix)` 状态，分别得到规范化的原策略和 K4 action 概率，固定

\[
\pi_{\rm mix}(a\mid s)=\tfrac12\pi_0(a\mid s)+\tfrac12\pi_{K4}(a\mid s).
\]

各自完成一致的 alias folding、合法支持和原温度后再混合；保留原事务失败/回滚语义。不直接平均 LoRA A/B 权重，也不把两份已生成晶体的分数当成当前 prefix 的 action 分布。

**支持与反证。** K4 的 refined 表现优于增加候选后的 K8；旧审计发现 teacher 的 KL 并不约束学生的实际策略漂移，且严重缩胞已可出现在构造阶段。混合把参考保留落实到部署 action 本身，逐状态有 `TV(πmix,π₀)=½TV(πK4,π₀)`。但这个局部关系不是整条路径的漂移界或能量保证；参考本身也有坏结构。K10 已有局部 KL 约束仍未达到 SUN 目标，说明“约束漂移”本身远不足够。[K4 池与学生漂移证据](historical/docs/teacher_feedback_unified_v1/27_SELF_IMPROVEMENT_REPAIR_PLAN.md)、[K10 完整负结果](historical/docs/teacher_feedback_unified_v1/13_STREAM19_DIAGNOSIS_AND_FINAL_ITERATION.md)。

**与旧失败的区别。** 复用已完成模型，在真实部署状态直接改变概率；不继续弱 teacher 的加权续训、不以更多 K 冒充跨组成覆盖，也不假设多次 repair 会单调改善。它仍可能只是稀释 K4 的有效更新，失败后不扫描混合系数补救。

**最小配对测试／训练预算。** 无新增训练、无新能量标签；只登记 `λ=½` 一点。一次固定256，比较 `π₀`、原 K4、混合策略的 raw/refined SUN 与逐请求得失；同版本可配对旧结果可复用。实际逐状态同时计算两分布，DLM forward 预算约为单模型同流程的两倍，另记录模型驻留内存；不是免费 ensemble。先看严重缩胞/高力尾部是否减少，再以 SUN 决定是否保留，不按几何或能量筛掉请求。

**论文关系。** 仍是冻结 LLM 条件和程序下的 DLM 生成，但贡献应称参考策略约束的推理集成；不能宣称单一学生已经学会稳定，也不替代①所需的辅助学习归因。

## 触发后的一次选择规则

先完成①，保留既有 K4 原策略作为明确参照；只有①不值得保留且错误分解仍支持策略漂移时，再考虑②。每次只改变一个登记对象，不同时换温度、P、共同 R 或 hull/N-U 口径。任一主端达到原定26/128后，按既定独立1000规则继续；旧开发集已经多次参与设计，不能当作新的独立确认。

本轮不将“再增 K／更多相同 CE／重启旧能量 teacher／多轮局部修订”列作第三候选；casebook 已有对应弱信号或负结果。公共 FIRE 停止与 force/stress 验收不等价是另一项真实问题，但改变共同 R 必须对所有基线统一处理，不能充当 DLM 模型提点。[物理审计](PHYSICS_PIPELINE_AND_STORY_AUDIT.md)。
