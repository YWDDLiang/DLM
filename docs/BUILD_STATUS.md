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
