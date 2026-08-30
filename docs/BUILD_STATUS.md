# 个人repo构建状态

## Proposal/Realization双候选

- 冻结H1-A2保持只读fallback；
- 历史难度分析使用纯Python实现并去除evaluator replay；
- Candidate A/B在独立branch开发，默认均关闭；
- 两个候选的远端screen均已完成并判负，公开H1-A2结果保持不变。

2026-08-26 Candidate B：

- `34697`完成Planner训练，`34704`完成two-seed Plan-256；
- 两个seed均少1个parsed Plan，projected Strict/Meta chemistry mix均下降；
- 按预注册选择规则停止，不进入DLM/refiner downstream，不重跑。

2026-08-26 Candidate A：

- `34700`完成control/counterfactual-grounding训练；final factual CE为
  `1.289004`与`1.288558`，candidate true-vs-counterfactual mean margin为`+0.7592`；
- `34711` fixed-256 screen两臂均为`256 parsed / 253 body / 253 refined`；
- `34714`完成4次独立fixed-256、逐sample_idx配对body/refine；
- `34721`完成8-cell Direct、N/U、CHGNet，随后在登录节点完成fresh official
  `GGA_GGA+U` hull；
- pooled Strict known差`+0.171 pp`，但Meta known差`-1.360 pp`，低于
  `-1.0 pp`非劣门，因此Candidate A最终判负；
- 完整证据见
  [`GROUNDING_FINAL_REPEAT4.md`](../results/remote_screens/GROUNDING_FINAL_REPEAT4.md)。

工程谱系：`34700`训练后导入失败、`34710`环境预检失败、`34719`因冻结V3只接受
1000/1200分母而在科学评价前失败；它们均被最小恢复，成功阶段未重跑。fixed-256
 adapter只放宽active denominator，不改变Direct、N/U、CHGNet、hull或S.U.N.阈值。

2026-08-27 DLM训练时长与固定requested-1000复核：

- raw-256同Plan扫描覆盖约`0.295/0.590/1.000 epoch`。相对各自matched control，
  counterfactual-grounding在step500的Strict/Meta差为`+3/-1`，step1000为`+3/+4`，
  step1696为`-5/-2`（分母均为256）；
- step1000的body、Direct、novelty、Strict/Meta方向和stable→S.U.N. retention均通过
  downstream门，但candidate validation CE比control高`0.07209`，且所有McNemar检验
  均不显著，因此预注册筛选没有合格checkpoint；这只支持非单调的中间训练窗口信号；
- 独立固定requested-1000复核不做survivor过滤。control/candidate分别得到
  `994/990` body、`877/874` Direct joint、`89/86` Strict S.U.N.和`487/467`
  Meta S.U.N.；Strict差`-3/1000`、Meta差`-20/1000`，完整贡献门失败；
- stable→S.U.N. retention仅小幅变化（Strict `81.65%→81.13%`，Meta
  `82.68%→82.07%`），主要退化来自stable本身（Strict `109→106`，Meta
  `589→569`），而不是novelty或retention崩塌；
- 完整证据见
  [`GROUNDING_CHECKPOINT_SWEEP_FINAL.md`](../results/remote_screens/GROUNDING_CHECKPOINT_SWEEP_FINAL.md)
  和
  [`GROUNDING_FIXED1000_FINAL.md`](../results/remote_screens/GROUNDING_FIXED1000_FINAL.md)。

2026-08-27 sufficient-DLM与count-valence Planner复核：

- 同一冻结raw1000 rich-Plan cohort上，总2 epoch得到`985` body、`871` Direct joint、
  `102/81` Strict stable/S.U.N.与`587/489` Meta stable/S.U.N.；总3 epoch得到
  `992` body、`878` Direct joint、`100/79` Strict与`578/477` Meta；
- 更长CE训练继续改善body，但Strict/Meta stable及S.U.N.均下降；冻结Pareto规则选择
  总2 epoch。其S.U.N.为`8.1%/48.9%`，没有达到`10%/50%`绝对门；
- count-valence数据审计本身通过：train/val/raw1000可精确分配并电中性的比例为
  `96.66%/96.32%/94.10%`，composition和soft-field roundtrip均为100%；
- 但单一text Planner的生成端失败：P0/countfields/countvalence pooled parse为
  `509/498/491`（各请求512），countvalence emitted-neutral仅`247/491=50.31%`，
  all-metal从P0的`29.47%`升至`45.82%`，lattice--space-group一致率仅`41.14%`；
- 因此该text-token count-valence arm不进入DLM/refiner downstream。其结果说明
  “teacher有物理标签”不足以让普通BPE自回归模型可靠执行耦合电荷算术；若重访Planner，
  必须使用显式species/count head或约束生成，而不是继续普通SFT；
- 完整证据见
  [`DLM_SUFFICIENT_RAW1000_FINAL.md`](../results/remote_screens/DLM_SUFFICIENT_RAW1000_FINAL.md)、
  [`PLANNER_COUNTVALENCE_AUDIT.md`](../results/remote_screens/PLANNER_COUNTVALENCE_AUDIT.md)
  和
  [`PLANNER_COUNTVALENCE_FACTORIAL_FINAL.md`](../results/remote_screens/PLANNER_COUNTVALENCE_FACTORIAL_FINAL.md)。

2026-08-27 same-Plan energy-pair可行性终态：

- outcome-blind地从train-only rich Plans冻结256个unique-formula Plan，其中192 train、
  64 validation，并与raw1000 exact formula/Plan identity隔离；
- 4个独立DLM/model494 stream在`0.06 eV/atom`固定gap下产生`67/22`
  train/validation pairs；按预案扩到8 streams后为`95/27`；
- 8-stream energy-gap q10/q25/q50/q75/q90为
  `0.0702/0.0854/0.1185/0.1649/0.2513 eV/atom`，说明same-Plan结构间确有
  可学习的能量跨度；
- 但冻结最低pair-yield门为`96/24`，train恰少1对，因此
  `preference_training_authorized=false`。没有降gap、改split、追加stream或启动训练；
- 该结果把当前energy-contrastive候选终止在数据可行性阶段，不产生新的模型权重，也不
  进入L6/CFG/L7；完整证据见
  [`DLM_STABILITY_PAIR_DATA_4STREAM_FINAL.md`](../results/remote_screens/DLM_STABILITY_PAIR_DATA_4STREAM_FINAL.md)
  与
  [`DLM_STABILITY_PAIR_DATA_8STREAM_FINAL.md`](../results/remote_screens/DLM_STABILITY_PAIR_DATA_8STREAM_FINAL.md)。
- 失败权重已清理：删除total3 duplicate final、step1392/2088/2392、四个判负
  countfields/countvalence Planner adapters，并在energy路线终止后删除仅具诊断意义的
  total2 `step-696`，合计约释放`30.7 GB`。本轮失败路线不保留任何模型权重；日志、
  metrics、Plans、结构、final reports及从原始resume重训所需合同均保留。

当前决定：

- Candidate A四重复中的小幅Strict信号保留为历史诊断，但固定requested-1000没有复现，
  不再作为完整或scoped正向训练贡献；
- Candidate B旧Plan-only预筛仍保留为负证据，但现按用户新决定进入一次最小真实下游验证；
- 第二个训练侧贡献目前未成立；下一条非RL候选是固定Plan的energy-contrastive DLM
  supervision，但其首个冻结256-Plan/8-stream pair-yield门也已失败，当前不得训练或
  通过放宽阈值补救；如需新候选，必须重新提出并预注册独立数据/目标机制；
- 标准H1-A2继续作为论文fallback；
- public headline继续是`105/1000 Strict、488/1000 Meta`；
- 所有checkpoint和新requested-1000结果只作内部机制证据，不替换headline结果。

2026-08-28 第二贡献点复审：

- formula-only H1-B历史结果已重新核对：尽管body执行为100%，Strict/Meta adjusted仅
  `5.54%/43.13%`，并出现`81.57%`全90度晶格角模板化；不得重跑后包装成新稳定性方法；
- safe-axis/H1-A2并非XYZ联合提交。D1为全X→全Y→全Z；safe-axis保持所有X/Y早于Z。
  本轮joint arm重现旧mixed-axis D2 duplicate失败，未启动的atom-major job35556已在
  dependency阶段取消，无科学计算，代码与Slurm已清理；
- MP-20真实train/val/test在冻结legacy Direct `comp_valid`下本身仅
  `90.50/90.24/90.95%`；H1-A2 refined1000为`87.8%`，model494不改变atom multiset；
- CCFD CPU Phase0已通过：train/val/test/raw1000可赋价覆盖为
  `96.74/96.46/96.76/94.10%`（含显式elemental-unary分支），所有可表示公式round-trip为`100%`；legacy SMACT在
  CCFD可表示集合上的false reject约为`8.25–9.67%`。冻结证据见
  [`CCFD_PHASE0_MANIFEST.md`](../results/remote_screens/CCFD_PHASE0_MANIFEST.md)；
- 冻结Planner tokenizer接口审计也已通过：train/val/test/raw1000的syntax、token
  round-trip、incremental prefix、prefix safety、formula-prefill boundary与UNK-free均为`100%`，因此Phase1可在
  不扩词表、不改权重的条件下实现。证据见
  [`CCFD_TOKENIZER_INTERFACE_AUDIT.md`](../results/remote_screens/CCFD_TOKENIZER_INTERFACE_AUDIT.md)；
- CCFD只解决composition correctness，不宣称稳定性；F0/F1已按正式分母完成并失败，
  特殊tokenizer/BPE Phase2不再触发，RL仍未授权；
- composition correctness与固定composition稳定转化的统一顺序、门控和终态解释已冻结于
  [`DUAL_TRACK_COMPOSITION_STABILITY_PLAN_V1.md`](DUAL_TRACK_COMPOSITION_STABILITY_PLAN_V1.md)。
- 用户随后冻结贡献点1与主故事并排除外部方法比较；后续仅推进内部matched CCFD
  与固定composition稳定转化，不再重开贡献点1。
- 两seed L6 conditioning/schedule终态已完成：full-axis pooled512为body/DirectJ
  `505/457`、Strict/Meta S.U.N. `48/230`；hard-axis为`512/463`、`47/230`，
  Strict下降且seed/retention门失败，因此继续选full-axis。mixed-joint两臂继续因body/Direct
  大幅下降而失败；证据见
  [`DLM_CONDITION_SCHEDULE_L6_FINAL.md`](../results/remote_screens/DLM_CONDITION_SCHEDULE_L6_FINAL.md)；
- raw→model494@800诊断表明refiner是当前stable提升的主要来源：full-axis DirectJ
  `188→457`、Strict S.U.N. `10→48`、Meta `66→230`；但novelty rate下降`10.69pp`，
  Strict/Meta retention分别下降`21.31/18.15pp`，未通过冻结Pareto门。因而已触发且仅触发
  `tau={0,200,500,800}`粗校准；0/800复用，只新算200/500。证据见
  [`DLM_REFINER_EFFECT_L6_DIAGNOSTIC.md`](../results/remote_screens/DLM_REFINER_EFFECT_L6_DIAGNOSTIC.md)。
- 固定tau L6扫描已终态：tau0/200/500/800的Strict S.U.N.为`10/29/39/48`，Meta为
  `66/171/222/230`（均/512）；novel分别`505/488/466/451`。较短tau恢复novelty与
  retention，但Strict/Meta均下降；200与500都未过双正向/双seed门，selected tau仍为
  `800`。因此不做raw1000短tau确认，下一步按冻结顺序进入noisy-state critic的独立评价
  可预测性审计。完整证据见
  [`DLM_REFINER_TAU_L6_FINAL.md`](../results/remote_screens/DLM_REFINER_TAU_L6_FINAL.md)。
- noisy-state critic可行性数据已严格组装：8 streams共`1752`条Direct-valid、CHGNet-known
  结构，train/validation=`1291/461`，`222/221`个Plan分别有至少2/3条结构；证据见
  [`NOISY_CRITIC_FEASIBILITY_DATA_MANIFEST.md`](../results/remote_screens/NOISY_CRITIC_FEASIBILITY_DATA_MANIFEST.md)。
- 独立MatterSim审计当前为工程阻塞而非科学结果：隔离Python3.12环境成功，但官方
  `mattersim==1.2.5`完整依赖在phonopy构建时要求源码NumPy 2.5.2，集群GCC4.8/Cython
  无法生成metadata；一次二进制scikit-learn恢复后仍同样失败。未运行MatterSim、未训练
  critic、未用CHGNet或official冒充独立评价。下一步需要用户明确授权inference-only
  `--no-deps`安装及最小运行时依赖，否则B3按合同停止。

2026-08-28 CCFD Phase1终态：

- 同checkpoint/tokenizer的F0 free与F1 CCFD均完成两seed×requested1000；F1把冻结内部
  assignment从`1898/2000=94.90%`提高到`1983/2000=99.15%`，说明在线守恒机制按设计执行；
- 但独立legacy composition validity仅`1724→1725`，seed17 `+0.5pp`、seed18
  `-0.4pp`，paired 95% CI `[-2.05,+2.15]pp`，McNemar discordant `229/230`、`p=1.0`；
- N分布TVD=`0.064>0.05`，且两seed正向与CI门失败；formal `phase1_pass=false`。因此
  CCFD冻结为“内部可赋价保证但未改善独立comp_valid”的负证据，不进入贡献列表，也不触发
  特殊tokenizer Phase2。完整证据见
  [`CCFD_PHASE1_FINAL.md`](../results/remote_screens/CCFD_PHASE1_FINAL.md)。

## 已完成

- 独立Git repo与`codex/personal-research`分支；
- 与公开repo物理隔离的完整源码副本；
- JSON个人配置入口；
- H1-A2、R03 D1和R03 D2内部组成结果；
- A800资产迁移台账；
- 训练、H1-A2推理与256×4快速复现Slurm骨架；
- confirmed/default/unrecorded三类seed记录；
- 相对路径和缺失资产处理逻辑。

## 等待A800

- 对外Conda环境文件与依赖版本的最终清理；
- 全部checkpoint与完整MP-20数据；
- H1-A2/R03 Plan文件和逐ordinal seed ledger；
- B0与model_494训练seed证据；
- 将内部fixed-256评价adapter整理成对外相对路径版本；
- 发布资产安装后再做公开repo的一键端到端smoke。

所有A800绝对源路径只填写在`ASSET_TRANSFER_LEDGER.md`，不会同步到公开repo。

## 2026-08-30 Rich-Planner recovery canary终态

- seed19冻结development cohort上的M0/RCF/R0六cell generation和十二cell
  raw/refined offline evaluation均已终态；requested分母始终为256，body缺失行按
  sample index保留；
- raw Direct在stream17/18由M0的`150/167`下降到aligned-rich R0的`111/110`；
  paired raw `R0-M0`为`+0.911996 eV/atom`，95% CI
  `[+0.552966,+1.281389]`，明确朝更差方向；
- model494 tau800把三臂refined Direct恢复到`246--254/256`，但refined
  `R0-M0=+1.292meV/atom`、CI`[-6.631,+9.993]`，没有复制性稳定优势；
- 因而旧rich checkpoint+prompt package不能作为本轮Stable-DLM主初始化。corrected
  rich Planner继续作为接口机制诊断；唯一prospective路线切换到预登记的minimal-DLM
  same-composition continuous listwise + raw-safety训练，不用canary outcome调整test；
- eval job38420耗时7134秒、11.8900 A800-hours。首个finalizer仅因Python3.10动态
  dataclass注册错误在写输出前失败，commit `b11c2c7`参数不变修复后成功；没有重复
  generation/evaluation。终态见`RICH_RECOVERY_CANARY_OFFLINE_FINAL_20260830.md`。

## 2026-08-30 C3FD-native DLM目标合同

- active checklist升级为`C3FD_NATIVE_DLM_SUN50_CHECKLIST_36H_V3.md/json`；最终
  prospective固定分母目标为Strict S.U.N.`>=10%`、Meta S.U.N.`>=50%`，但不作为
  事后删结果或选择seed/checkpoint/cohort的门；
- 旧H1-A2 rich JSON只保留为compatibility diagnostic，不再作为权重初始化。
  production接口`C3FD_NATIVE_PLAN_V2`沿用rich-Plan可读JSON风格，只包含exact N/counts、
  composition-derived family与soft lattice/SG/volume；Planner内部certificate不暴露给DLM，
  动态`7+4N` body不变；
- 正式Stage-1使用MP20-train的teacher-native与冻结C3FD-predicted-native双视图做
  Planner-interface SFT；soft fields按train/validation可靠性dropout，不硬约束geometry；
- native DLM从预训练LLaDA-8B新建LoRA，不继续旧rich/minimal adapter。按两个训练seed
  在完整MP20标准train `27136`行各跑两个source epochs：epoch1/1696步使用LR`5e-5`，
  epoch2再1696步使用LR`1e-5`，最终只保留step3392；step1696仅监控。五view由
  source-balanced sampler
  每source每epoch取一个并跨epoch轮换，不把expanded rows误算成五倍epoch。C3FD现有
  checkpoint已训练10 epochs，保持冻结，不追加epoch；
- 现有3614个历史candidate因混合retired L6/L7/D3PO谱系、raw-invalid上游选择，只作
  development evidence。若native SFT恢复execution但仍未达到10/50，只允许从冻结SFT
  checkpoint构建一次fresh on-policy、MP20-train-only同composition pool再做安全排序；
- 当前energy-only listwise wrapper会把raw-invalid但post-refiner低能量candidate选为anchor，
  正式训练前必须改为raw-validity-gated rank与显式best-valid anchor。

## 2026-08-28 C³FD-v2/v2.1 阶段终态

- C³FD-v2用typed `N→element-valence-count`、在线atom/charge reachability、physics
  species features和benchmark双证书替换formula BPE约束；CPU audit对全部
  benchmark-valid行100%召回，N/Q/composition/rich round-trip均100%；
- 正式v2两seed×requested1000把独立comp-valid从`1724/2000`提高到`1972/2000`
  （双seed均`+12.4pp`，CI`[+10.81,+13.99]pp`），Novel和NU提高；但原冻结gate因
  arity/family/support等漂移NO-GO；
- v2.1按小步门控推进：Step1 proposal/ledger与可达质量、Step2 exact-arity/ledger
  model、Step3 per-head LBFGS/no-top-k/pair0、Step4 100k outcome-blind proposal
  simulation全部通过，并分别保留首次失败、最小修复和MD/JSON记录；
- requested256 pilot job36514终态NO-GO。P0/C3FD-v2.1 pooled comp-valid为`438/460`，
  两seed分别`+5.08/+3.52pp`且CI下界为正；ionic为`80.75%→100%`，NU`388→417`，
  N/arity/family到full-train距离均优于P0；但`52/512` semantic dead ends使parse
  `510→460`，parsed all-metal `39.57%`距full-train `34.91%`超过3pp门。严格停止，
  不进入requested1000；
- 完整合同与证据见
  [`C3FD_V21_CORRECTION_CONTRACT.md`](C3FD_V21_CORRECTION_CONTRACT.md)、
  [`C3FD_PLANNER_FINAL.md`](../results/remote_screens/C3FD_PLANNER_FINAL.md)和
  [`C3FD_V21_PILOT_FINAL.md`](../results/remote_screens/C3FD_V21_PILOT_FINAL.md)。

## 2026-08-28 C³FD-v2.5 requested-1000 确认终态

- v2.1的`52/512` semantic dead ends已被定位为两个必要条件的错误交集：通用
  N/charge/arity continuation与family-prefix分别可达，并不保证存在同一个后缀同时满足
  二者。v2.5没有调temperature、top-k或pair prior，而是把family、exact N/charge/arity、
  alloy/ionic branch与Pauling电负性顺序编译为constructive witness bitset；每个被采样的
  动作都保留至少一个可证终态；
- 冻结teacher审计为train `24558/24558`、validation `8159/8159` witness-valid，且不加载
  Planner权重或任何outcome/stability标签。双seed在线canary各`32/32` parsed和独立
  comp-valid、零失败，采样耗时`31.6/32.0s`，才自动扩至requested256；
- requested256 pooled对照/候选：parsed `510→512`，独立comp-valid `438→512`，
  Novel/Unique/NU `388/508/388→463/512/463`；双seed comp-valid分别`+14.84/+14.06pp`，
  paired 95% CI `[+11.40,+17.50]pp`，全部预注册门通过；
- requested1000确认复用相同seed17/18 checkpoints并只补global sample indices
  `256..999`，每个请求恰好一次。pooled 2000对照/候选为：parsed `1989→2000`，独立
  comp-valid `1724→2000`（`86.2%→100%`），Novel/Unique/NU
  `1538/1961/1530→1763/1985/1756`；双seed增益`+13.9/+13.7pp`，paired 95% CI
  `[+12.29,+15.31]pp`，McNemar discordance=`0/276`、`p=1.65e-83`；
- ionic独立comp-valid由`1147/1412=81.23%`提高到`1255/1255=100%`；semantic dead end
  为`0/2000`。候选all-metal为`37.25%`，距full-train `34.91%`仅`2.34pp`；候选到
  full-train的N/arity/family TVD为`0.0349/0.0185/0.0204`，均优于P0的
  `0.1017/0.0319/0.0948`；requested1000全部冻结门通过；
- 该结果建立一个独立的composition-correctness贡献候选：typed semantic Planner以
  constructive certificate进行单轨迹约束解码，而不是formula BPE、后过滤、repair、
  replacement、rerank或RL。它不宣称稳定性、可合成性或改变public S.U.N.
  `105/488`；稳定性Track仍独立；
- v2.2--v2.4只保留为工程演化记录，均在GPU前因穷举runtime门停止；没有将失败请求删除或
  放宽科学门。正式证据见
  [`C3FD_V25_TEACHER_WITNESS_AUDIT.md`](../results/remote_screens/C3FD_V25_TEACHER_WITNESS_AUDIT.md)、
  [`C3FD_V25_PILOT_FINAL.md`](../results/remote_screens/C3FD_V25_PILOT_FINAL.md)和
  [`C3FD_V25_REQUESTED1000_FINAL.md`](../results/remote_screens/C3FD_V25_REQUESTED1000_FINAL.md)。

## 2026-08-28 CTV-DLM-v1终态与SGTC fallback

- CTV formal Branch完成train `2048`与validation `1024`条forced-action终态，全部
  CHGNet-known；冻结hidden/action feature复现320 states和202个legal geometry token，
  base probability最大误差仅`8.88e-16`；
- 正确的within-state结果为Spearman `0.0353`、95% LCB `-0.0563`，pairwise AUC
  `0.5053`，两continuation action-sign agreement `0.4915`；mean supported mass仅
  `0.1613`，guided coverage `0.0781`。因此gamma未设置、L6未授权；
- 初版absolute-energy Spearman被composition baseline抬高，commit `3261893`改为
  state-centered estimand；旧输出只作统计实现错误保留。完整边界见
  [`CTV_DLM_V1_FINAL_NO_GO.md`](CTV_DLM_V1_FINAL_NO_GO.md)；
- 按冻结fallback启动SGTC-DLM-v1：训练时N/element特殊token保持可见，只mask并监督
  lattice/angle/XYZ；G0使用全部C3FD-certified MP20，G1只用source
  `e_above_hull<=1e-8`的strict-stable结构。训练JSON递归移除能量/稳定字段，两个arm
  从同一base固定续训348步，随后进入matched L6；合同见
  [`SGTC_DLM_V1_CONTRACT.md`](SGTC_DLM_V1_CONTRACT.md)。

## 2026-08-29 SGTC-DLM-v1 official L7终态

- matched requested1000使用同一C³FD seed18 Plan、DLM seed、temperature、exact-axis、
  model494 tau800和refiner seed。generation `37617`与eval `37807`均成功；fresh
  official query只执行一次，union `932`个chemsys中`913` resolved、`19` unresolved，
  unknown始终按missing处理；
- base/G0/G1的body为`998/1000/1000`，Direct joint为`996/997/996`，N/U/NU为
  `922/995/922`、`933/999/933`、`930/998/930`。composition/execution已经接近饱和，
  不是本轮稳定性失败的原因；
- official Strict stable/S.U.N.为`81/60`、`78/55`、`73/53`，Meta为
  `486/412`、`486/421`、`485/417`。G1相对base的Strict/Meta S.U.N.为
  `-0.7/+0.5pp`，相对G0为`-0.2/-0.4pp`；全部paired CI跨0且McNemar不显著，
  formal `sgtc_l7_pass=false`，public `105/488`不变；
- all-known official e_hull q10/q50/q90为base `0.0082/0.1014/0.3417`、G0
  `0.0102/0.1012/0.3548`、G1 `0.0092/0.1016/0.3476`。G1-base matched mean
  e_hull为`+4.525meV/atom`、fraction lower `0.4801`；matched CHGNet为
  `+4.238meV/atom`、`0.4850`，均未左移；
- 相对G1 continuation训练chemsys，L7每臂仅约`320/1000` reconstructed处于seen
  chemsys。G1 seen Meta S.U.N.为`167/320`，unseen为`250/680`；优势未外推到
  unseen support。exact composition seen仅`92/1000`，说明粗粒度MP20分布接近不等于
  fixed-composition realization处于训练局部支持；
- v2补充报告明确披露L7合同没有运行raw CHGNet/hull，不能事后推断raw→refined连续效应。
  immutable结果位于A800
  `runs/sgtc_l7_official_final_20260829_v2`；报告扩展代码commit `31ea2ee`；
- SGTC因此作为正式阴性保留：positive-only strict-stable continuation改善teacher-forced
  NLL，但没有提供same-composition能量边界。下一候选已经收敛为用户确认的低资源
  [`D3PO_256_MIN_CONTRACT_V1.md`](D3PO_256_MIN_CONTRACT_V1.md)：保持C³FD composition与
  model494 tau800不变，用post-refiner相对能量训练full-sequence shared-noise masked-D3PO，
  不引入外部rich Plan、rerank或额外tau/checkpoint搜索。

## 2026-08-30 D3PO-256-Min 零GPU冻结

- 完整历史复盘纳入R5C、107-token fixed-slot、H1-A2换seed崩塌、R03 D1/D2以及后续
  CTV/CCFD/SGTC阴性；因此动态`7+4N`、full-axis、base step696、双训练seed和双共同
  sampling/refiner stream全部写入合同，单seed阈值高点不再算成功；
- L6/L7内same-composition CHGNet排序与official hull排序在非tie pair上完全一致；L7
  K1 `56/416`到K3 oracle `74/502`、L6 K1 `13.2/89.5`到K6 `23/123`，证明候选集合存在
  可学习上限；但手工scalar几何probe AUC仅`0.507`，所以不再训练独立critic，而把偏好
  直接写进DLM序列概率；
- production pair v3为train `5007` pairs/`885` compositions、validation `850/166`，
  chemsys零交叉。4205 outcomes先平均并删除`335`个exact repeats，再严格PBC去掉`164`
  个物理等价体，零parse失败，避免把refiner随机走运当DLM监督；
- fresh seed17 test256已按sample_idx冻结，先对L6/L7/noisy pool做reduced-composition排除，
  Plan SHA为`de27b5c066cfd3e1068bf48cbe617f5ae5e216ead51d5676d206290c2764e9fb`，
  选择过程未读取任何稳定/能量/validity结果；
- one-backbone/two-adapter trainer、shared-mask数学core和一张A800双seed串行wrapper已实现。
  local `176` tests通过（`33` dependency skips）；remote完整D3PO trainer tests `12/12`
  通过，真实tokenizer逐行载入`5857` pairs成功，最大长度`152`，PEFT named-adapter
  load/switch/delete/save canary通过。base与pair数据SHA均硬冻结；当前仍无GPU job。

## 2026-08-30 D3PO执行与fallback预登记

- 首次训练job37966在任何optimizer update前因LLaDA不支持通用gradient-checkpointing
  API失败，已保留`_FAILED.json`与独立root-cause；commit `6546a1b`只把该非科学内存
  优化改为model支持时启用，未改data/seed/loss/steps。唯一恢复job37974已通过step0
  policy/reference equality canary并进入有限loss/gradient训练；
- 主test的6-cell generation/refine与12-cell raw/refined evaluation wrapper已冻结，使用
  `6 A800/48 CPU`分别一次完成，不增加arm或搜索；generation-facing Plan SHA为
  `21a20c8...d94d3f5`；
- 为避免fallback在主test上自适应，另冻结第二个outcome-blind seed17 holdout256，与
  L6/L7/noisy-pair/main-test reduced composition全不相交。raw/certified SHA分别为
  `ac321dea...077094`与`ba061122...f4e08`，主结果分类前不得读取其任何科学outcome；
- fallback按[`D3PO_FALLBACK_DECISION_TREE_V1.md`](D3PO_FALLBACK_DECISION_TREE_V1.md)
  预登记：replicated continuous positive但阈值弱才允许late-only单点guidance；无raw
  能量信号才考虑DLM内部self-predicted structural intent；raw有效/refiner抹平则只做
  bridge attribution。任何情况都禁止seed选择、rich Plan、rerank与同test调参。

## 2026-08-31 C3FD-native fresh-retrain执行修正

- 用户确认paper方法必须从共享预训练LLaDA-8B新建LoRA重训，不从旧rich或minimal
  adapter继续。两训练seed在full MP20 train `27136`行上各跑两个source epochs：
  `1696+1696=3392` updates，仅step3392可进入prospective；step1696只监控；
- 原始`mp_20_r5_exact_length`已逐行核对：train `27136`、validation `9047`，与full
  semantic及双C3FD prediction的composition/N mismatch均为0。原始MP20标准split本身有
  `3469`个train/validation chemsys重叠；过滤派生数据中的`3089`不是五view构造引入；
- 按用户决定保留MP20标准material-level split，不另做chemsys重分。文档与manifest只称
  `MP20 standard validation`并披露重叠，禁止继续称chemsys-held-out；builder仍保留独立
  fail-closed held-out模式供未来实验；
- job38668因把MP20标准split误当chemsys-held-out而pre-science失败；full-source job38673
  又因把Planner内部composition certificate误设为DLM数据hard gate而在6秒内失败。两者均
  negative归档。用户随后明确简化接口：MP20真实结构无需Planner证书授权；正式V2采用
  H1-A2-like rich JSON，仅N/elements/counts/family/LS/SG/VP，使用full `27136/9047`；
  teacher/pred17/pred18/masked/minimal五view、答案一致、每source总weight1、outcome-blind
  合同不变；
- faithful offline job38603仍为development诊断，不再参与paper方法初始化选择；主方法
  fresh初始化已冻结。预计无alignment时于2026-08-31 `14:00--18:00`完成全部SUN与论文
  主线；若唯一fresh on-policy alignment被启用，预计`20:00--23:00`完成；
- MP凭证只允许进入最终唯一query的临时nonambient process env，启动后立即unset；不得
  写入git、docs、automation、命令行、日志、hash或manifest，本任务结束前不再向用户索取。
