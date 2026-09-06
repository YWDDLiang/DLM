# 达标版本的1000样本 raw/refined 补评

用户最新要求：任意达标版本都补一套1000样本的raw/refined，不只给某一版补；已在运行或已完成的同policy补评不重复。

当前默认达标定义为原最终SUN目标：固定开发256中，native或tau800任一端Strict SUN≥10%且Meta SUN≥50%，即Strict≥26、Meta≥128；同端两项AND、两端OR。旧K4/K8后训练准入门native6/55或tau18/124不是最终达标目标，不能混用。用户如明确选择旧门，则将registry的qualification_rule改为posttrain_admission；当前使用final_sun_goal。这一区分已经写入自动任务，并保留了阈值澄清入口。

原K4/K8目前未达到10%/50%目标，因此不因旧低门槛新增K4补评。原K8的39951属于此前已安排的独立主评测，继续并复用。raw-v1训练39942已经完成，native256作业39975已接续，随后tau800再检查是否触发1000。

## 独立运行包

本目录与冻结训练实现分开。SOURCE_NEW保持执行commit 2e904c260bafb6750c9a5dbb0d920c4ff8c3a868，不能为了运行本目录把模型实现改成未登记版本。将本目录三个文件从发布提交提取到：

ROOT/experiments/periodic_self_repair_20260906/supplement_1000/operations

SUPPLEMENT_ROOT为上面的supplement_1000目录，REGISTRY.json放在其根目录。按原K4、原K8、raw-v1、V2的实际policy、256目录、补评job/run、方法锁更新登记。原K8现有39951的锁可引用39949/METHOD_LOCK.json，不能改写它。

在原V2八阶段检查之前，再运行：

    "$PY" "$SUPPLEMENT_ROOT/operations/check_before_v2.py" \
      --registry "$SUPPLEMENT_ROOT/REGISTRY.json"

该检查按当前规则重新计算达标，不信任手写“未达标”状态；所有before_v2候选的256结果必须齐全，达标者必须完成原policy的两端1000、同一selection与1200总分母。返回2表示有待办，不能先开V2。V2自己达标后的1000在V2固定256评测后进行，不形成自依赖。

## 触发后的提交

为每个达标policy创建独立METHOD_LOCK.json，包含method_frozen=true、policy_path、qualified_for_1000=true、qualification_rule、通过端点与完整256指标/文件hash、原阶段身份、planner_seed=27、dlm_seed=20260906、cohort_requested=1200、parser_only_target=1000。确认policy的TRAIN_FINAL eligible和路径一致。按policy身份去重后提交evaluate_qualifying_policy.sbatch，环境为：

- H1A2_CANDIDATE_ROOT=ROOT
- PERIODIC_REPAIR_SOURCE=SOURCE_NEW
- SPAD_MAIN_COHORT_RUN=ROOT/runs/spad_state_main_plans_39949
- SUPPLEMENT_ROOT=上述补评根目录
- SUPPLEMENT_VARIANT=登记的版本ID
- SUPPLEMENT_METHOD_LOCK=该版本独立方法锁
- SUPPLEMENT_WORLD_SIZE=2、4或6，与SBATCH GPU/CPU资源匹配

脚本默认2A800/8CPU/180G，按可用额度可覆盖4卡16CPU或6卡24CPU；仍遵守6卡/24CPU/最多2项目jobs，不抢占健康任务。RUN在SUPPLEMENT_ROOT/runs/<variant>_<job>。记录job id、CODE_COMMIT、操作脚本快照及_SUCCESS/_FAILED。

脚本复用233的原始1200请求/first1000流程，保持同一Planner条件与种子；以逻辑4-worker layout固定批次，再分配实际worker。它依据checkpoint原始初始化marker启用raw-v1/V2的full-cell repair，旧K4/K8继续旧cooperative/closure。按raw CIF源序冻结前1000，refined只处理对应来源；失败不替换，同时报告完整1200。model494、800步、逐graph seed、N/U/hull/终态协议不变。

不会在这里新增K4/K8训练、选择中间checkpoint、重采更好的样本、按能量筛1000或用256报告替代1000。效果不及预期仍进行完整原因分析。自动任务须等所有已触发补评和报告完成后才能停止。
