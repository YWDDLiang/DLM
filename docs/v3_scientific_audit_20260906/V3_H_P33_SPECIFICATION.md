# V3 具体规格：共享 LLaDA 的周期几何去噪（第二轮审稿版）

2026-09-06。状态：**主审选作第二轮攻击对象，尚未实现、训练或认定有效。**
本文替代两份候选稿中的未定选择，唯一正式候选记为 H-P33；第一轮审稿见 [数学](V3_REVIEW_R1_MATH.md) 与 [物理](V3_REVIEW_R1_PHYSICS.md)。原提案和反例保留，不把一般审计的 ROUND2 计入本规格复核。
可修复的定义/实现问题关闭、两位非候选作者独立复核后，按用户已授权预算执行。只有真实 SUN 可以支持采用；没有事先保证。

## 1. 选择及论文主张

选择一个系统候选：冻结 C³FD／Typed Llama／pointer 的化学与条件生成；从完成的 V2 权重继续训练共享 LLaDA 的 construction token 模式 T 和周期几何模式 G。G 从先验联合生成连续晶格与坐标。它是**共享预训练 DLM 主干的混合表示生成器**，不称纯 token diffusion，也不把新头叫成独立 GNN。

理由是可验证的机制差别：全时刻完整几何观测、每个 G forward 的全几何监督、不经中间 codec 的联合反向更新。它们直接改变原 V2 的训练和生成对象。已有1-hop周期消息、全局 Transformer 与 old 条件仍复用；没有“补上原先缺失的 GNN”主张。

P 在 G 中提供固定 rank／条件，不控制同步几何更新顺序。不采用 rank 噪声时钟，不因论文需要加入 H-C constructor、Bayes外循环或更多 repair。T 保留实际程序 construction 路径及其辅助训练，但首个 G 主端点不经过 T constructor。程序与语言预训练的净收益仍须单独实验，当前不能宣称。

A 的真实软标签改善 frozen 风险，是存在条件信号的线索；原生成时没有对应 oracle，改变监督语义还需新的 joint 条件合同。B 的 corrupt→D 也没有由本次 construction12/44→repair8/50 建立收益。首轮先推进一个明确候选，不同时扫 A/B/C/H。

## 2. 固定数据、条件与来源身份

用 V2 完整 prepared train27136、val9047；不重新采样 C/S/P，保留预测／显式fallback及其来源标记。source_split/source_row_idx、prompt、plan_state、species_program 和 canonical source_answer 均逐条携带。训练 C/S/P 在给定 C 后可能与实际 G 独立；新几何损失不解决这个可识别性限制。

原始 CSV 只取 cif 列，**不取 cif.conv**。已核路径和 SHA 见 [输入证据](evidence/CONTINUOUS_SOURCE_INPUT_PATHS.json)。实际40009全量核验已完成：27136/9047全部唯一完整Q匹配原CSV、一一对应，0解析/编码错误、0重复Q歧义、0未核验或丢行；精确site permutation后Q全部等于source_answer，[来源门已通过](evidence/CONTINUOUS_SOURCE_IDENTITY_40009.json)。计数相同本身不作为行身份证明。逐源audit保存：

- split、CSV SHA、CSV ordinal 和 material_id、原 CIF SHA、pymatgen 版本；
- 原 row lattice／fractional／species、只做 species-stable 排列的精确 permutation；
- 对齐连续结构经原 structure_to_dynamic_answer／canonicalization 得到的完整 token string，与 prepared source_answer 的相等性；
- 直接 ordinal／stable ID／完整量化answer唯一匹配等连接证据，重复完整answer造成的身份歧义单列。

不按 formula 配对，不忽略原失败行，不重新 primitive／Niggli／conventional 化，不在 noisy 时刻重排或做同元素 Hungarian 匹配。全量身份核验通过后，primary label_precision 固定为 original_cif。若不能完成全量映射，本文不自动混入 decoded 标签；可以修订为完整 decoded_codec 版本并重新由第二轮确认，届时放弃 sub-bin 监督主张。不能用子集掩盖缺失。

连续源仅由已有 MP20 取得，不读取能量、SUN、generated paths 作训练目标；CSV 中这些列存在不构成模型输入。固定源排列沿所有时刻携带；F 仅 wrap 到 [0,1)，不减均值或平移投影。训练与验证不混用，normalizer只从 train 得出。

## 3. 状态空间与晶格 chart

行向量约定 R=FL，L 的三行为晶格矢量，单位 ℓ₀=1 Å。设
S=½ log(LLᵀ/ℓ₀²)。固定 Frobenius 正交基：

E1=diag(1,-1,0)/√2，E2=diag(1,1,-2)/√6；
E3=(e12+e21)/√2，E4=(e13+e31)/√2，E5=(e23+e32)/√2；
E6=I/√3。

y_j=〈S,E_j〉（j≤5），y6=(tr S−log N)/√3=log[V/(Nℓ₀³)]/√3。
从全部 train source 按总体方差 ddof=0 计算 μ_j、d_j；实际用 d_j=max(std_j,10⁻⁶)，记录所有是否命中 floor 的通道。z=(y−μ)/d。

逆变换 k_j=μ_j+d_jz_j（j≤5），k6=μ6+d6z6+log N/√3；
L=chol_lower{ℓ₀² exp(2∑k_jE_j)}。F不另旋转，此操作保留 ΔF LLᵀ ΔFᵀ。

chart／normalizer预处理 float64。训练 noisy z/F 及 conditioner 输入保留FP32，谱运算只用于输入数据转换，无预测谱损失反传。采样的 z/F、时间积分和最终 chart 解码使用FP64；输入网络的FP32副本保留，不把BF16 hidden当状态存储。

这个 chart 消除 Cartesian 旋转代表冗余，未实现 GL(3,Z)、平移、置换或 SG 商空间等变性。有限 z 的数学 SPD 不保证有限精度 exp/cholesky 可用；失败计入请求，不clip、不投影到新密度范围、不重抽。

## 4. 正向过程与监督

每个 G view 独立采 t∼LogUniform[ε,1]，ε=0.002。所有原子使用同一个 t；“统一时钟”不是 uniform-t 抽样。
α=cos(πt/2)，σL=sin(πt/2)，σF=t。
z_t=αz0+σL ξL，F_t=wrap(F0+t ξF)，ξL∼N(0,I6)、ξF∼N(0,I3N) 条件独立。

晶格 v*=αξL−σLz0。单个坐标 δ=f_t−f0，n∈Z：
w_n∝exp[-(δ+n)²/(2t²)]；
u*=−∑w_n(δ+n)/t = t∂f log q_t(f_t|f0)。

targets以FP64 logsumexp计算 n=-8…8，转FP32作loss；在δ∈(-1,1)、t≤1上用尾界和扩大像窗复算验证截断误差。不是MIC/t²替代，也不把随机−ξ与条件均值误认为逐样本相同。

L_G=½||v_hat−v*||²/6+½||u_hat−u*||²/(3N)。
每个 source 权重相等，两几何类别各半。这是指定风险，不宣称无权ELBO、能量梯度或SUN目标。高t的u近零、同胞通道相关，密集targets数不等于独立信息数。

最优输出条件在完整(z_t,F_t,C,S,P,t)上，隐式联合场允许晶格／站点相关。log-uniform t给最高t邻域较少曝光是已知取舍；验证按固定时间段单列，不用低噪声平均loss替代先验生成质量。正式抽样分布与范围不据开发SUN再调整。

## 5. 两模式显式输入与共享计算

新API接受 mode='token' 或 mode='geometry'，G另带 ContinuousGeometryState(z,F,t,species,N,program_rank)；整数context只提供scaffold／位置／padding。保持7+4N body：N/E真实，六晶格和全部XYZ为MASK。MASK只标槽位，G的当前几何全部已知。

G直接从L_t/F_t构建共享 state_conditioner 和 geometry_attention 的浮点 geometry；不得经 old token codec。全部实际站known/active，晶格known，padding不参与。复用现有模块权重及125-image shell，不宣称任意极端胞下全局精确MIC。原整数V2入口不改变。

G embedding=原token embedding+input_delta+当前几何cell/site residual+G的time/mode residual。
几何residual使用现有六cell槽、每站E/XYZ映射。G time/mode residual加入body全部有效槽位，prompt/padding不加。T完全走原V2语义。

G中不调用旧 task_features 推断mask ratio／噪声；legacy 9维task向量明确置零，typed旧噪声4维输入为unknown所得零向量，跳过 repair_task_projection。旧geometry_attention中的常数输入列可无G梯度，time由新模块提供，不能把零当成V2已有任务。

time用64维sin/cos特征：ω_k=exp[-log(10000)k/31]（k=0…31），输入1000tω_k。Linear64→128、SiLU、Linear128→H，另加可训练G向量（初始零）；MLP用标准Linear初始化。v/u最后线性层权重与bias全零，禁止再乘另一个零门。

从共享Transformer最终ln_f后的hidden读取：
v_hat=Linear6(mean(h_六cell槽))；
u_hat_i=Linear3(h_Ei)，各站共享同一头。
新头／time模块保持FP32，hidden转FP32读出，loss FP32。G hidden不detach；base权重冻结但LoRA、已有conditioner/bias、相关input rows接受梯度。source噪声/目标构造不需梯度。

G应直接返回几何输出并跳过旧 numeric_adapter/output_delta 和词表投影。已读实际 [LLaDA core forward](evidence/LLADA_ACTUAL_CORE_FORWARD.json)：词表ff_out在最终ln_f之后，可在项目本地的显式hidden-only adapter中复用原block、mask/bias合并、checkpoint与ln_f路径。保留相同层的LoRA对象，不复制冻结/断梯度主干。不得临时替换ff_out、注册跨forward可变hook或修改原下载模型目录。
这个adapter必须和原forward最终hidden做数值一致性及梯度验收。若初版仍计算整词表，必须明示并计实际显存／吞吐，不能虚称已省开销；这不允许通过detach“优化”。

## 6. 初始化、真实活参数及训练配方

完整载入V2 final step4524（adapter、输入输出增量、conditioner、V2 bias），路径和hash在 [probe配置](evidence/V2_CONDITIONING_PROBE_40004.json)。不走fresh原始LLaDA初始化，不恢复旧optimizer momentum。8B base继续冻结；这是V2两epoch后再训两epoch，累计曝光必须报告。

每source每新增epoch：一个T construction view（V2原0.75prefix／0.25dense，原目标与support），一个G full-noise view。替代旧repair CE，不声称所有旧任务均保留。两epoch T/G各54272真实examples，共108544。T loss与G loss各按view权重1；不把CE+MSE称联合log-likelihood。

新模块577673参数（H=4096）；旧V2总可训练库存36993232；T+G optimizer并集37570905。G路径整tensor上界26720601，不是每一步有效自由度。G不更新的旧output_delta10162176、numeric_adapter651264、task_projection36864由T更新；input table只统计实际索引行的直接梯度。

正式默认6A800、24CPU、global batch24。40045真实4卡microbatch1×accumulation6验收已经通过，峰值显存16.12GiB；据此在正式训练前追加一次6卡microbatch2×accumulation2的吞吐/累计验收，若通过则使用该配置。它保持相同12-source T/G全局窗口、噪声和损失系数，减少同步与forward启动开销；不承诺不同微批的浮点优化轨迹逐bit相同。6卡microbatch1×accumulation4、4卡microbatch1×accumulation6保留为容量配置，不按SUN选择。实际资源和预检结果写launch manifest，不在模型内硬编码GPU数。max_length沿用382，padding/尾batch用真实source/view权重，不能drop_last丢源。

每个优化窗口把T和G在所有rank上同步安排，各占半数；单个microbatch同模式，避免一次forward两套输出/多余loss。保留DDP find_unused_parameters=True；不用static graph或0×unused参数伪loss。正式implementation需验证梯度累计策略；没有证明no_sync对交替未用参数正确前，用每microbatch同步的明确路径。

AdamW（betas .9/.999，eps1e-8，weight_decay0），clip_norm1；fresh optimizer。沿V2两阶段LR：第一新增epoch基准5e-5、warmup100、cosine最低ratio.2；第二新增epoch基准1e-5、warmup100、cosine最低ratio.1。阶段长度根据source数/global batch推导，当前每epoch2262、总4524更新。初始化seed82018，数据seed202609062；t/噪声独立由source/view/epoch决定，与rank/批次无关。

每epoch保存checkpoint；两epoch final是唯一正式policy，不按validation／SUN挑中间点。验证用事先hash选的100 val sources，selection、T掩码和G噪声的validation seed均为202609063；每源固定1份T和6份G，G时间分别取t∈[.002,.01),[.01,.05),[.05,.2),[.2,.5),[.5,.9),[.9,1]各区间几何均值。各G噪声由source与band独立固定，共700个monitor states/epoch，T/G及各band分报；这是诊断，不是早停或选policy。此处只落实原定验证分层，不改变训练时每源每epoch各1份T/G及LogUniform时间测度。

## 7. 唯一主采样器 H-P33：先验、积分、显式末端读出

初始z1∼N(0,I6)，F1∼Uniform(T3N)，条件N/A固定。
晶格先验精确；坐标σmax=1的uniform近似TV上界≤1.606×10⁻⁷（N≤20）。这个界不限制网络、solver或SUN误差。

形式probability-flow：
dz/dt=(π/2)v_hat；
dF/dt=−u_hat。
反向网格 t_j=ε+(1−ε)(j/32)²，j=32…0。
每个区间一遍联合forward，Δ=t_j−t_(j−1)>0：
z←z−(π/2)Δv_hat；F←wrap(F+Δu_hat)。

第33次forward在实际(z_ε,F_ε,ε)上，输出确定性读出：
z_out=α(ε)z_ε−σL(ε)v_hat_ε；
F_out=wrap(F_ε+ε u_hat_ε)。
输出CIF来自chart(z_out)、F_out、固定species。没有新随机末步、CG/FIRE/能量选择、连续clip或拒绝重抽。

**这是第一轮H的32-NFE、p_ε端点的明确修订，不是等价改写。** z_out是v回归对应的clean条件均值表达；F_out是局部lift的wrapped Tweedie读出，不声称全局torus posterior sample。多模态下均值可落入低概率/碰撞区；极小ε也不使有限模型误差消失。该读出给出单一注册端点并消除点目标反例的主要残余blur，但不能保证普遍提升。第二轮必须专门攻击它。

整体模型是33次可测离散映射对初始分布的pushforward；非有限、解码失败和后续不可重建都进入⊥。实际conditioner含finite-shell／mask分段，不能无条件援引全局光滑ODE唯一流定理；连续probability-flow只说明目标，实施对象是此有限算法。

primary endpoint固定为float native。对每个相同样本另外做一次原Q roundtrip，作为secondary输出精度诊断；不据结果选float或Q。若原EncodeDiagnostics的length_clips或angle_clips非零，secondary Q统一记失败并保留请求，不把内部clip的投影结构当成功Q；coords先wrap到[0,1)，periodic100→0 alias合法，非预期coord_clips也记失败。量化后不可用cell同样失败，不clip后挑结果。float输出改了系统表示，未来成绩不能归为纯离散DLM收益。

## 8. 实现验收与停止条件

以下是有界必要验收，不是重复增加的模型机制：

1. 全量来源身份和normalizer产物；chart回环、wrapped score符号/尾窗、噪声分布、point/oracle Gaussian采样器CPU检查。包括33rd readout多模态反例，不只测实现自己。
2. T step0与载入V2的相同输入输出；hidden adapter与原ln_f输出、bias/padding/checkpoint一致。
3. 真正4或6卡小规模T/G forward/backward/update：head首步与打开后步骤；LoRA/conditioner/bias有效有限梯度，T-only在G预期无梯度；梯度累计/padding/unseen rows/DDP归约正确。只测新图，不重跑旧全库测试。
4. 固定val子样本小σ状态扰动经过FP32 features→BF16 hidden→FP32 heads的输入敏感度，不因入口float就假定模型保留全部精度。新头零时应另看encoder/hidden和head打开后的输出。
5. 正式训练前固定完整launch manifest、代码SHA、数据hash、epoch与种子；记录显存/吞吐，4–6卡满足项目总上限6A800/24CPU/2jobs。只按实际测量确定walltime，不任意延长提交任务。

源缺失／计算图断开／非有限优化／实现与数学不符需要修复后重验；“SUN尚未知”不是无限阻断定义完善模型的理由。若短验收表明机制无法在此预算工作，应记录具体证据并作一次有界修订，不装作正式训练成功。

## 9. SUN验证、对照与费用

完成两epoch后，沿用同开发256化学条件、Planner outputs和request顺序。实际V2开发运行的base seed是20260905，collection_round=0/candidate=0，per-request seed由现有path_seed(base_seed,group_id,round,candidate)产生；不是先前稿中误写的独立cohort种子20260906。[实际运行证据](evidence/V2_ACTUAL_EVALUATION_SEEDS.json)保留原64位整数。H复用相同per-request seed、batch cap4和原layout_world_size2的分组/顺序，记录实际batch size；执行worker数可改变。几何噪声由request seed派生，与worker调度无关。refiner继续使用现有export默认base20260905加原sample_idx并模2³²，不另改seed。

primary float native和固定model494 tau800两个端点均用原共同评估、N/U/hull版本与全256分母；secondary Q raw对同样256额外评估。三端全部报告，不择优，不按SUN／能量过滤输入。

导出器必须有显式native geometry source：旧body路径保持默认，H使用structure路径。不能让旧export_programmed_path_artifacts重新parse body覆盖float结构，也不走graph_from_arrays的0.01-bin重复坐标guard。H直接将同一continuous CIF交给原process_one(cif,True,False,'crystalnn',False,.01)，保留相同refiner图预处理及精确composition检查，不在此前Q；这不改变共同物理0.5Å验收规则。tau记录保留真实native_structure，regression的上游几何读取它，不能把Q诊断body冒充float上游。label/evaluate原有structure优先入口可以复用。需要一次实际CIF/graph/endpoint数值回环验证，不能仅凭配置名称声明保留连续精度。

既有参照：V2 construction raw12/44、repair raw8/50、repair→tau16/113；原参考raw6/55、tau19/121；raw-v1 raw3/45、tau16/113。未测construction→tau，不能假设其等于其他refined分数。H-P33 warm-start和额外两epoch、连续监督、loss、表示/solver共同改变，首轮只判断完整系统是否改善，不声称单因素因果。

NFE只计几何LLaDA时为33／请求；N=1/10/20的V2 construction+repair为18/72/132（不含失败/重试），H对小N可能更贵。实际记录forward和walltime；Planner、物理评估和refiner另记，不能用NFE代替全部成本。Q不新增DLM forward。

primary同一端达到Strict≥26/256、Meta≥128/256才按已授权规则追加独立1200条件、固定源序1000 raw/refined，并另报1200。触发器从整数counts和登记阈值判断；不直接复用legacy strictly_exceeds_10_and_50布尔（其严格>50%会在Meta=128时为false），也不改写原评测字段含义。未达目标仍完整记录实际得失，不靠改共同R、hull筛选或换分母提点。现开发集已用于设计；若未来独立1000用于选择也不得继续称未见确认。

## 10. 第二轮需要回答的具体问题

审稿人分别判断“定义可实现”“明确近似/代价”“实际验收未完成”，不把规范化生成器、精确逆过程和SUN混成一个判词。

- joint v/u目标与反向符号、σ单位、先验和chart是否一致；33rd local-lift readout是否引入未声明的目标或失败路径？
- CIF身份与精度合同能否按实际产物关闭；Q roundtrip/float主端点是否事前明确且可公平解释？
- hidden-only、T/G数据调度、参数并集与梯度是否真正走共享预训练LLaDA；有没有重演PMTR冻结head-only或保留死词表开销的虚假说法？
- frozen条件统计、程序职责和一至两epoch可学性是否被夸大；最小实证如何在已有预算内给出可否证结果？
- 是否存在需先修的实质错误；若没有，列清尚待实现验收的事项，允许向实际实验推进，SUN判定留给完整结果。
