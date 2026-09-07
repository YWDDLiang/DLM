# H1-A2 到当前的连续几何与 refiner 谱系

2026-09-07。独立只读复核；只新增本文。读取旧 DLM 仓库、当前 `DLM_periodic_self_repair` 和被旧报告直接引用的 `llm_plan_diff` 源码，没有执行历史文档里的训练、恢复、提交或自动化指令，也没有 SSH、GPU、NN 或 MLIP 调用。

**核心结论：R03 safe-axis、Shared-Plan/S2 和当前 V3 是三种不同干预。R03 的旧高点不是“训练了一个零初始化 model494 residual”的结果。成熟 model494 仍在当前 refined 端保留；真正没有迁入 V3 新 G 分支的是其已经学会的连续去噪场。** 同时，旧 B0 的晶体权重、计划分布和全局 safe-axis 调度后来确实经历了替换，但现有证据不足以把长期未提点单独归因于其中一项。

本文区分三种证据：**运行记录**指已保存的终态/原报告及其hash；**源码事实**指实际读取的代码或明确git版本；**解释**指由这些事实支持、尚未完成单因子因果隔离的判断。旧报告的“下一步”不当作已执行；历史权重hash来自对应记录，本次没有重新加载大型权重。

## 1. 先拆开三个容易混淆的“保留父模型”

| 路线 | 实际改变 | 起点保留什么 | 不应误读为 |
|---|---|---|---|
| H1-A2 / R5-C → R03 safe-axis | 只改 body token reveal order | 完整 P0、B0、model494 权重与当时配对条件 | 在model494上训练了Plan residual |
| Shared-Plan S1/S2 | 在冻结model494六层CSPNet中增加Plan encoder、FiLM和晶格残差 | S1零初始化时的父denoiser函数；训练后可显式选择null父路径 | S2匹配Plan时仍逐点等于父模型，或已证明de novo SUN优势 |
| 当前 V3 H-P33 | V2 LLaDA共享骨干上增加连续v/u头，从联合prior生成完整几何 | 继承V2权重；T模式起点保留V2输出 | G模式继承model494连续场，或从零头就有一个成熟晶体先验 |

R03报告明确限定“只改变揭示顺序”，且refiner就是冻结 `model_494`；该R03的有效refiner batch已经是1。[R03系统边界](D:/codex_work/ai4s/diffsion_language_model_meets_diffusion/workstreams/plangraph_dlm_iclr_20260731/H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md:59)、[R03 refiner合同](D:/codex_work/ai4s/diffsion_language_model_meets_diffusion/workstreams/plangraph_dlm_iclr_20260731/H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md:152)

## 2. 早期真正的权重和条件链

H1-A2的已登记主干为：H1-A2 epoch2七行Planner → R5-C exact-length B0 → model494 refine800。P0的adapter SHA为 `65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a`；B0的adapter SHA为 `5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d`；model494 SHA为 `573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e`。[P0/B0身份](D:/codex_work/ai4s/diffsion_language_model_meets_diffusion/workstreams/plangraph_dlm_iclr_20260731/H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md:96)

- **Planner**：由MP20的27,136训练行学习formula、anion、charge、lattice、spacegroup、volume等七行目标；epoch2是从epoch1 adapter再训练一遍，重新开始optimizer/scheduler。它不是现在的C³FD合法支持＋typed Llama残差策略。[数据与训练记录](D:/codex_work/ai4s/diffsion_language_model_meets_diffusion/workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/H1A2_EPOCH2_BASELINE_DOSSIER.md:77)
- **B0**：学习 `7+4N` 晶格/元素/坐标特殊token；报告记载一遍MP20、global batch16（27,136行对应1696更新）、LR5e-5、r8 LoRA、answer-only IID masked CE。它还保存已训练的 `model.transformer.wte` 和 `ff_out`，并非仅一个很小的LoRA文件，adapter约6.39GB。[权重与SFT边界](D:/codex_work/ai4s/diffsion_language_model_meets_diffusion/workstreams/plangraph_dlm_iclr_20260731/H1_BODY_DLM_COMPLETE_PROTOCOL_AND_RL_DESIGN_V1.md:159)
- **model494**：已经训练好的连续晶体denoiser，本路线不再训练它。原H1-A2历史评测取前1000个可构图proposal，refiner batch128、world2、800步；六月运行没有记录可证明执行源码逐字节身份的git commit，原dossier明确保留这个缺口。[历史refinement与分母](D:/codex_work/ai4s/diffsion_language_model_meets_diffusion/workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/H1A2_EPOCH2_BASELINE_DOSSIER.md:180)

R03完整保留P0和B0，只把坐标组改成全局 `lattice → grouped X → grouped Y → grouped Z`。其实现明确检查 `all_xy_precede_all_z`，防止在XY未知时先提交Z，绕过duplicate-coordinate mask。原mixed-axis D2在32例中产生18个新的duplicate失败；safe-axis的32/64/256阶梯修复了这类故障。这是有直接接口和实验支持的早期有效机制。[实际schedule代码](D:/codex_work/ai4s/diffsion_language_model_meets_diffusion/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1/safe_axis_schedule.py:131)

但不能把这项机制扩大为对任意新Plan分布都更稳定：后续保留同一P0、B0、model494的连续1280重复和exact-Plan重放，R03相对H1-A2并未出现稳定优势。旧高点与具体Plan cohort、下游随机实现、分母有关，不只由方法名称决定。[后续完整重放报告](D:/codex_work/ai4s/diffsion_language_model_meets_diffusion/workstreams/final_method_development_20260808/evidence/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v6_final/FINAL_REPORT.md:3)

## 3. Shared-Plan S1/S2：确实保留了成熟连续场，但高值是条件式诊断

### 已执行的模型和目标

这条分支在另一个 `llm_plan_diff` 仓库实现。`PlanConditionedCSPNet`直接持有原 `base_decoder`，冻结其参数；每个CSP block后增加FiLM，最后增加9维lattice residual。FiLM末层和lattice adapter末层均零初始化。全连接边、原coord_out、原lattice_out、parent schedules继续使用。[FiLM和残差实现](D:/codex_work/ai4s/llm_plan_diff/generation/models/plan_conditioned_cspnet.py:93)、[冻结父decoder](D:/codex_work/ai4s/llm_plan_diff/generation/models/plan_conditioned_cspnet.py:154)

因此S1起点在理想算术下是“已有denoiser＋零增量”，没有要求新adapter先学会整个晶体场。**S2却不是fresh zero起点**：它从已训练S1 best继续，匹配Plan时允许改变父输出。后续null路径能回到parent，也不意味着匹配Plan的完整分布始终等于parent。

| 阶段 | 运行与初始化 | 训练状态/目标 | 实际训练量与选择 |
|---|---|---|---|
| S1 | job27340，model494＋fresh zero adapter | 干净MP20 x0按parent前向过程加噪；lattice ε与wrapped coordinate-score MSE | adapter-only 5000步，batch128；选validation step4750 |
| G0 | job27350，冻结R5-C body | 2048个train-source各生成一次draft，并配回同一MP20结构 | 2048有效；固定1840 train /208 validation |
| S2 | job27397，从S1 best初始化 | 原始draft直接作为t=800输入；按同源x0反算lattice target及wrapped coordinate target，不再次加噪 | adapter-only 5000步，batch128；matched validation选step250 |

S2 best adapter SHA为 `533fd02fcc9ac5b77594350ee901fab77dc75c8ddef18a7b333edcf2183e097d`。5000是实际run更新数，250是最终选中的S2更新位置；不能将“best250”误写成只训练250步，也不能把S1与S2忽略后称其小量从零学习。[已接受训练链及manifest](D:/codex_work/ai4s/llm_plan_diff/docs/MP20_R5C_SHARED_PLAN_REFINER_RESULTS_2026-07-24.md:122)、[原S2固定配置](D:/codex_work/ai4s/llm_plan_diff/configs/experiments/mp20_r5c_shared_plan_s2_v1.yaml:113)、[实际paired-draft目标](D:/codex_work/ai4s/llm_plan_diff/generation/models/plan_conditioned_diffusion.py:316)

其PlanLite只向refiner暴露formula/elements/counts/N和一个lattice-family hint；没有volume、坐标、energy/hull字段。模型学习“共享粗计划如何修改一个既有物理去噪器”，不同于新G分支的完整连续场拟合。

### 10.55% /67.19%到底是什么

原gold gate的输入不是H1-A2生成的新Plan。它从 `20260531_0040-r5c-full1000-sun` 的1200条held-out R5-C记录中，先得到1167个已有可解析draft，再按固定hash选256；匹配hint来自held-out **gold lattice-family label**。每臂一组256、gold gate batch16，不是多seed独立复现。[原报告的条件来源](D:/codex_work/ai4s/llm_plan_diff/docs/MP20_R5C_SHARED_PLAN_REFINER_RESULTS_2026-07-24.md:93)、[oracle声明](D:/codex_work/ai4s/llm_plan_diff/docs/MP20_R5C_SHARED_PLAN_REFINER_RESULTS_2026-07-24.md:339)

门失败后job27411才做post-hoc SUN，复用现成R1-null和R2-matched张量，未生成新样本：Strict为21→27/256，Meta为168→172/256，即8.20→10.55%、65.62→67.19%。它是**同adapter的Plan开关诊断**，没有对untouched R0重新做这个SUN比较。分母是已选生成的256；未知hull的34/36例计false；使用旧R5C/A100 CHGNet0.3＋冻结MP cache。这里的“all256”不能抹去上游只从1167个有效draft选择的条件性。[post-hoc来源、分母和结果](D:/codex_work/ai4s/llm_plan_diff/docs/MP20_R5C_SHARED_PLAN_REFINER_RESULTS_2026-07-24.md:358)

这确实是有价值的连续Plan信号，但不是H1-A2 de novo的67.19% Meta基线。修复null协议后的H1-A2同draft F10/F11实验，报告Strict SUN两边均15/256；Plan-family matched−shuffled约+0.1988仍在，终态收益未出现。说明条件可读与端点收益分离，且gold/same-source draft上的收益没有直接转移。[已完成Stage2记录](D:/codex_work/ai4s/llm_plan_diff/docs/ICLR_PLAN_DLM_RESTART_2026-07-28.md:72)

### 原null失败必须按真实源码时间解释

本次直接读取了原S2训练commit `093ce0b6a9514502befe544cc220ad4952024292` 和gold-gate commit `fd74474d18e318944dd96f4cf67939e4202eded0` 的 `generation/models/plan_conditioned_cspnet.py`：

- 原FiLM与lattice residual已经在learned transform后乘active mask；force_null将active置零。
- **原forward没有 `force_null → return base(...)` 快捷分支**，仍走完整wrapper。
- 直接返回parent的分支直到 **2026-07-27 commit `16e6401b4abb012fe95c28d6b216257abcd0e5de`** 才加入，提交标题为 `Make refiner exact-null path bitwise parent-identical`；当前文件231行可见该分支。[当前明确parent路由](D:/codex_work/ai4s/llm_plan_diff/generation/models/plan_conditioned_cspnet.py:231)

两个7月24日commit中的该文件逐字节相同，SHA256为 `f19b7e1cfb77f379fae3cea93cf77e0d534296f235e9218260a460de545c09ec`：108/109与141/142行为zero init，230行取得base后直接重跑wrapper。7月27日文件SHA为 `508e9dd5184213ed052dd327f3140174864743fb325027e224767501daf14ae6`，236行新增 `return base(...)`。这些是本次 `git show <commit>:generation/models/plan_conditioned_cspnet.py` 的实际读取与hash结果。

这修正了后写综述中“force_null已经直达parent”被回投到7月24日的风险。原gate以两个独立800步CUDA轨迹的终态byte误差作为adapter污染证据，原结果coord≈0.49997/lattice≈3.78357未通过；这个判据无法单独区分wrapper路径差异、父模型非确定归约及其迭代放大。后来同parent自身重放也出现漂移，所以旧失败**不能直接确证非零Plan泄漏**；反过来，也不能仅凭后写解释确证原全部差异都只有scatter一种原因。single-trajectory parent/null alias解决的是控制臂身份，未凭空证明更高de novo SUN。

## 4. 现在K4/K8与V3到底继承了什么

### K4/K8不是旧R5-C B0的小续训

当前可追的body链为：

```text
原始LLaDA
  → fresh canonical Compact-V2 39282 / step3392
  → SPAD schedule 39520 / step1696
  → closure CE 39700
  → periodic state warmup 39853 / 1696 updates
  → K4 path posttraining 39892 / step1020
  → K8 refresh continuation 39938 / cumulative step7977
```

153入口明确 `checkpoint_path is None`、fresh LoRA，代码从原始LLaDA加载并重建晶体词表；不是加载5月B0的 `5c39976...`。164入口显式从39282继续；196从39520；223从39700的POLICY_PATH；完成的39853和后续K4/K8状态则有历史终态记录。[39282实际合同与检查](D:/codex_work/ai4s/DLM_periodic_self_repair/slurm/153_train_canonical_compact_v2_dlm.sbatch:67)、[加载分支](D:/codex_work/ai4s/DLM_periodic_self_repair/src/scripts/llada_sft.py:1290)、[SPAD父checkpoint](D:/codex_work/ai4s/DLM_periodic_self_repair/slurm/164_train_spad_schedule_dlm.sbatch:17)、[closure父checkpoint](D:/codex_work/ai4s/DLM_periodic_self_repair/slurm/196_train_spad_basin_closure_ce.sbatch:18)、[warmup父checkpoint](D:/codex_work/ai4s/DLM_periodic_self_repair/slurm/223_state_programmed_warmup.sbatch:18)、[已完成warmup记录](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/19_RESUMED_ARCHITECTURE_AND_EXECUTION.md:64)

这个替换保留了表示和预训练语言骨干，却没有原封不动保留旧B0的已训练晶体embedding/head/LoRA。另一方面，canonical训练修正了真实的物种-坐标槽位对齐问题；不能因为它是fresh初始化就断言该修正无益。[canonical实际数据/运行记录](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/ROLLOUT_MATCHED_DLM_24H_CHECKLIST_V1.md:140)

K4/K8采用C³FD＋typed Llama的条件与species program，保留精确N/E；Plan prompt仍输入DLM，旧几何与active事务通过conditioner输入双向Transformer。它们的目标是原生完整attempted-path概率，经离线CHGNet原生/弛豫双目标形成经验teacher，加MP20 CE anchor；没有训练model494，也没有把tau800映射放入这个path目标。[实际状态/目标](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/historical/docs/teacher_feedback_unified_v1/19_RESUMED_ARCHITECTURE_AND_EXECUTION.md:20)、[输入代码](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/programmed_path_runtime.py:175)

已读取终态metadata：K4固定1024条件×4，972个verified候选覆盖526条件；实际1020更新=816 path＋204 CE。K8在同1024条件上刷新×8，1843 verified覆盖662条件；新增6957更新，累计7977=6382 path＋1595 CE。未验证请求均保留但teacher权重为0。增加K没有增加训练组成集合，也没有使teacher的有限池改善成为student分布的保证。[保存的TRAIN_FINAL元数据](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/k4k8_prefix_inputs.json)

SPAD调度已经替换R03的“全局X→全局Y→全局Z”：它是cell scalars、各物种anchor的逐site XYZ、再future sites，外加cooperative修订和逆物种closure。**保留的是每个Z前本site XY已知这个安全前提，未保留R03的完整排序。** 当前代码明确声明这一点；不能把不同调度的表现都叫作R03机制仍原样存在，也不能因排序不同直接认定现在重新引入了旧mixed-axis bug。[SPAD实际调度](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/spad_program.py:195)

### raw-v1/V2再次重置；V3继承V2，但完整G场重新学习

raw-v1与V2明确放弃K4/K8 weights和反馈，只用原始LLaDA＋新晶体行/LoRA/周期模块，原MP20 teacher数据训练。raw-v1两epoch6784步；V2另从原始LLaDA独立训练两epoch4524步，不能把V2称为K4/K8连续继承。[raw实际合同](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/periodic_self_repair_v1/RAW_LLADA_DESIGN.md:3)、[V2初始化](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_v2_initialization.py:63)

V3用严格V2-final loader继承LLaDA、LoRA、token行、conditioner等，再加零初始化v/u输出头；T模式直接调用原token_model，所以它确实保留了一个现成token分支，**不是重训整个随机8B**。但G模式没有 `model494(state)+delta`：它读取浮点noisy geometry，经共享LLaDA直接输出完整lattice-v与coordinate-u。几何numeric token槽为MASK，Plan prompt、N/E、program rank仍在；P在这个并行场中提供条件/特征，而不是逐site执行顺序。[零头与V2加载](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/mixed_geometry_model.py:71)、[浮点输入与T/G分支](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/mixed_geometry_model.py:306)、[严格V2 warm start](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/mixed_geometry_model.py:378)

G训练是 `.5 mean6(v−v*)² + .5 mean3N(u−u*)²`，目标来自teacher连续几何的已定义加噪过程；生成从Gaussian lattice chart和uniform torus开始。零v/u意味着初始场没有学好的坐标transport，不等于model494的非零物理denoiser。与S1“成熟场上的零增量”相比，新G分支承担了完整连续分布的学习负担。V3实际两epoch4524更新、108544 T/G状态，其中54272个G状态；没有S1/S2的model494权重作为G输出的基底。[损失与prior](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/mixed_geometry_diffusion.py:402)、[原始训练终态](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence/MIXED_TRAIN_FINAL_40064.json)

这不是仅凭设计猜测的担忧：本轮100来源真实噪声零头基准、256输出和两次冻结路径重放一致指向坐标场有效学习不足；u在低/中t风险相对零头几乎无改善，而v在高t有约25.5%/32.4%改善。没有发现简单σ/符号或最终readout错误。因此“整个8B没加载”和“仅多积分几步即可”都不是目前最有据的解释。[实际G诊断结论](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/G_DIAGNOSIS_CLOSURE.md:5)

## 5. model494与batch：保留、改变和缺证据必须分别说

在H1-A2/R03、K4/K8、raw-v1/V2及V3的**refined端**，同一model494/checkpoint和tau800机制继续存在；当前wrapper仍从完整生成结构进入parent sample，没有Plan-conditioned S2 adapter。故“后续完全丢掉成熟连续prior”不成立；“新G分支本身没继承成熟连续场”成立。生成输入从B0 draft变为SPAD、synthetic-repair或H-P33 draft后，进入refiner的分布已经变了；相同refiner不会自动保持每一种proposal策略的端点排序。

| 已有证据 | 能支持什么 | 不能支持什么 |
|---|---|---|
| H1-A2六月batch128/world2；R03后来batch1、per-ordinal seed、4次进程实现 | 执行合同确实改变；R03高值不依赖继续使用128batch | batch128优于1、或batch变化解释全部长期下降 |
| 旧零变化控制：同节点、同body、246图精确相同、parent batch64、同noise，连续终态不逐位同 | 早在8月已有same-input数值重复差异 | 一个128↔1纯batch因子对照；报告将其归因scatter也不替代首分叉记录 |
| 本次40074与旧K4/K8：每组171图输入/seed完全同；1011个输出映射全通过；world2/4→6，batch都1 | 排除简单seed、graph或保存错位；相同输入的refiner实现波动真实存在 | 将变化唯一归给world，或把输出差异全部当内禀结构差 |

本次针对已列的历史报告、恢复源码和当前证据搜索，**未找到固定同输入、同权重、同实现，仅切batch128↔1的实测SUN对照**。这应作为明确证据缺口，不能升格为退化原因。[旧同输入控制](D:/codex_work/ai4s/diffsion_language_model_meets_diffusion/workstreams/plangraph_dlm_iclr_20260731/H1_REFINER_REPRODUCIBILITY_AMENDMENT_V1.md:11)、[旧通用/R03路径差异](D:/codex_work/ai4s/diffsion_language_model_meets_diffusion/workstreams/final_method_development_20260808/main_experiment_freeze_20260815_v1/training/provenance/DLM_reproducibility_audit_20260812.md:467)、[本次实际重复性结论](D:/codex_work/ai4s/DLM_periodic_self_repair/docs/v3_scientific_audit_20260906/REFINER_REPEATABILITY_DIAGNOSIS.md)

## 6. 对“早期有效机制是否丢失”的最终判断

1. **确实保留**：精确N/E与7+4N表示；单site Z前XY已知的安全前提；冻结model494作为refined端的成熟连续模型；Planner条件仍存在。不能以“Plan/几何全没了”概括后续路线。
2. **确实替换**：旧P0的输出分布和七行接口；旧B0实际晶体权重；R03完整全局safe-axis排序；K4/K8到raw/V2的初始化与数据来源。V3另外把新G前端改成完整连续场学习。这些是需要单独归因的组合干预，不是一条持续保留父函数的小residual链。
3. **值得保留的早期原则**：已训练的几何先验上做可控增量、明确Plan输入、同draft/同noise的matched-null-shuffled控制、严格保护有效schedule与checkpoint身份。S2本身提供的是条件机制证据，尚不是可直接恢复的de novo冠军。
4. **当前失败最具体的证据**：新G坐标场没有显示相对零场的有效预测；K4/K8经验teacher覆盖有限且优化native路径，不直接优化model494后的SUN；早期历史高点也对cohort、抽样拓扑和评价分母敏感。仅说“训练不够”“模型不懂物理”或“refiner batch改坏了”都超出证据。

本文没有把R03官方28/128、S2 gold27/172、历史survivor1000及新的all-attempt指标排成同一榜单。它们的输入与分母不同。后期1000规模基准应按其原始cohort和评价记录另行统一核算；机制谱系不能替代该统计核对，本文也不启动恢复训练或新实验。
