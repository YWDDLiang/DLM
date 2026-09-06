# V2 数学与接口反审（只读）

2026-09-06。审阅 `LOSS_GEOMETRY_AND_V2_PROPOSAL_20260906.md` 第6–8节，行号对应本次读取的246行版本。仅检查定义与落地契约，附两项小型CPU数值核验；未修改实现、自动化或作业，未使用SSH/GPU。

**结论：保留token ABI、exact条件分类、完整旧整数几何条件、log-volume/shape corruption这条主线可以成立，没有发现必须推倒重来的数学矛盾。但主方案还需补齐以下六项契约，才能称为可执行设计。它们是定义修订，不要求再增加一组新方法。**

## 1. 在合法集合内归一化，并区分合法风险与全源似然

方案140–150行的 `softmax(... )_{S(s)}` 有“softmax后截取而未重新归一化”的实现歧义。建议明确写成

```
g = CanonicalFold(SchemaMask(z))
p(a|s) = 1[a∈S(s)] exp(g_a/T) / Σ_{b∈S(s)} exp(g_b/T)
T = 0.7
```

顺序继续复用当前公共部署处理：typed schema → 在raw dtype中合并000/100 → 对canonical物理动作施加几何支持 → 在剩余集合内归一化。dense辅助省略几何支持，但仍使用canonical typed集合与同一T。当前的alias语义是“先合物理类别，再温度化物理类别”，不是“温度化101个token后再相加”。不需要改alias规则，只需把它定义准确并统一loss、采样和trace。

方案178行“target不可达记冲突”还缺少目标与分母定义。若target不在S，exact NLL本来为无穷；实现上不能计算无穷后再乘0，也不能静默剔除再重算batch mean。应明确采用eligible-gated prefix risk，原source/state分母固定，另报eligible conditional NLL与coverage。

eligibility不能只检查当前target。令π为实际scalar动作顺序，最小定义为：

```
construction Z_j = 1[teacher动作π_1,…,π_j依次在各自S(s_k)中]
repair Z_j = 1[x_t满足runtime修复入场支持] ×
             1[teacher动作π_1,…,π_j依次在各自S(s_k)中]
L_prefix = E_{source,x_t,j}[Z_j * NLL(y_j|s_j)]
```

只有Z=1时才计算NLL。若某个早期teacher动作非法，却被teacher forcing写入，后续target即使合法，该prefix仍不是实际策略可到达的prefix。当前源码保留schema-valid但不满足deployment support的source；corruption失败回退clean也不能自动保证repair旧状态合格。可缓存同一source/program的teacher-prefix可达性，避免每个样本重复完整检查。不可达source保留在数据/typed辅助及ledger，不能伪造legal动作或声称已拟合它的完整legal似然。

方案150行的 `H(p*)+KL` 结论正确，但p*必须指**所声明训练状态、支持与admission之下的条件后验**。存在Z=0时，该门控均值不是全原始MP20源的无条件NLL。轨迹上实际draw的log-probability仍无family权重；其总和是一次attempted trace的概率，带rollback的多个trace可对应同一个最终结构，不能把它当最终结构的唯一概率。

## 2. 固定四种状态的采样测度，避免重复权重与预算翻倍

方案158–168行的field risk合理：

```
R_field = (1/2)(1/6) Σ_lattice ell_j
        + (1/2)(1/(3N)) Σ_coord ell_j
```

scalar按 `P(j∈lattice)=1/2`、family内均匀采样后直接取ell，确实估计此对象风险；不要再乘一次1/2对象权重。

方案172行与226行组合后，建议把训练样本协议明确为：每source/epoch仍恰有construction和repair两个view；每个view以3/4选prefix、1/4选dense。这样原两epoch仍为 `27136×2×2=108544` 个state、global16共6784次更新。不是每source物化四种state后仍声称相同步数/来源覆盖，也不是采样后再把prefix loss乘3/4、dense乘1/4。

该协议对应状态混合权重

```
(construction-prefix, construction-dense, repair-prefix, repair-dense)
    = (3/8, 1/8, 3/8, 1/8).
```

这仅说明抽样期望；有限运行需记录实际分支计数。若改成确定性四分支采样，也可以，但必须用同一测度的权重并重新核对state预算，不能混用两种算法。

对独立p-mask，dense状态估计式应完整写为

```
L_dense(M) = (1/2) Σ_{j∈lattice} M_j ell_j(s_M)/(6 p)
           + (1/2) Σ_{j∈coord}   M_j ell_j(s_M)/(3N p).
```

block mask的分母替换为真实条件inclusion probability `π_j`。若强制至少一个mask、只选发生变化的位置、按noise自适应选mask，都要重新计算π_j，不能继续使用原p。空mask保留零贡献，不再另按非空family重归一化。

由于ell依赖整个mask后的上下文，以上期望为

```
Σ_j alpha_j E[ell_j(s_M) | M_j=1]
```

而非prefix风险，也非clean state所有位置损失。方案168行的“同一目标风险”宜改为“同一对象/位置权重下的dense条件去噪风险”。两位置的CPU枚举验证：一个构造例中inclusion-corrected值为7，强制目标mask条件风险同为7；masked-mean/空项置0仅5.125，clean-state位置均值为5.5。辅助branch和prefix共享对象权重即可，不应宣称其condition相同。

## 3. 把old/current/program/active四元组写死；teacher forcing仅匹配顺序

方案176–179行方向正确。需把“prefix/suffix”明确成**program顺序π的前后缀**，不是native token位置数值上的连续切片。

当前 `spad_program.py:196-213` 的顺序是六个lattice scalar、各species anchor、各species剩余site；`programmed_path_runtime.py:255-269,321-331`复用该顺序。若实现用`range(position)`或本地slot连续顺序，便重新引入历史program错配。必须使用同一个compiled `full_cell_transaction_positions(program)`。

精确状态应为：

- construction prefix：current为所有N/E已知、π之前的teacher numerical prefix可见、π及之后mask；old=current。active scope为全部剩余mask numerical，匹配fresh sampler。
- repair prefix：old为完整受扰且已量化的x_t；current的π之前为teacher prefix，其余numeric mask；active scope是完整full-cell transaction，包含已经填好的prefix，匹配现有runtime。
- construction dense：old=current masked canvas；任何mask的clean值不得通过precomputed geometry进入模型。
- repair dense：old=x_t；current在M内mask、M外为同一个x_t的受扰token。目标是clean y，仅用于loss。需要在协议中进一步固定task id与active scope（现有`structured_denoise`可表达辅助任务）；不要把active silently设为“与clean发生变化的位置”。

repair dense里old可能包含与target相同的token，这不是泄漏，而是正常的identity/denoising条件。需要分报零改变与fallback，不能把它们的低CE当作纠错成功。

“prefix训练部署一致”只能指动作顺序、张量构造与可用信息一致：训练prefix来自teacher，部署prefix来自自身抽样。新方案没有消除exposure差异，也没有解决synthetic old到self-generated old的分布差异；文中176行已有提醒，最终执行稿不要把它缩写为完全distribution-matched。无需因此引入任意同组成多晶型作为逐原子teacher，方案213行的限制应保留。

所有新几何特征必须从**解码后的整数x_t**计算，不能从生成corruption时未量化的L_t/F_t计算；否则训练拥有部署没有的sub-bin信息。门控中的已知噪声只能是声明的corruption参数，不得读取相对于clean target的实际误差、changed-position mask、true volume差或future reconstruction误差。unknown/dropout必须同时遮蔽所有新增noise channels，而非仅把旧单scalar设成-1。

## 4. log-SPD公式正确；补充采样分布与坐标噪声参照物

方案185–207行对行向量晶格是正确的：`G=LL^T`，`S=(1/2)log(G/l0²)`，`trS=log(V/l0³)`；`chol(G_t)`必须是lower Cholesky，满足`L_t L_t^T=G_t`。不应误改成`L^T L`、upper factor或右乘错置。

volume/shape分解没有数学冲突；必要补充是η_v、五个shape系数和ΔR的分布、独立性，以及sigma是std还是hard bound。沿原稿的Gaussian记号，一个完整定义可为：`η_v,ξ_1,…,ξ_5`独立标准正态，`Xi_dev=Σξ_r B_r`；ΔR每个Cartesian component为独立`Normal(0,sigma_F²)`。若实际选择bounded分布，则直接写该分布及界，不把截断后的参数继续称作实际standard deviation。

一套固定Frobenius正交无迹基可以是：

```
B1=diag(1,-1,0)/sqrt(2)
B2=diag(1,1,-2)/sqrt(6)
B3=(E12+E21)/sqrt(2)
B4=(E13+E31)/sqrt(2)
B5=(E23+E32)/sqrt(2).
```

于是shape perturbation的Frobenius RMS为`sqrt(5)*sigma_s`，单原子Cartesian RMS为`sqrt(3)*sigma_F`，volume ratio恰为`exp(sigma_v*eta_v)`。这些量不能互称为同一个“noise level”。范围仍按train-only诊断冻结，失败处理和admission按第1项声明；不需要增添新的corruption方法。

`F_t=wrap(F_0+ΔR L_t^{-1})`的物理含义必须写清：**先让原子随cell仿射变形，保持fractional F0，再加Cartesian ΔR**。未wrap时

```
F_t L_t = F_0 L_t + ΔR,
R_t-R_0 = F_0(L_t-L_0)+ΔR.
```

因此sigma_F控制的是相对于仿射变形参考的displacement，不是相对于原clean Cartesian结构的总位移。若方案本就意图联合晶胞/原子噪声，公式无需改动，只补此语义。求ΔR L_t^{-1}时可解线性系统而非显式逆；这不改变模型。

CPU斜晶胞核验（无模型）：`trS-logV`误差8.9e-16，Cholesky重构误差3.6e-15；Δv=.2时volume ratio=1.221402758，与exp(.2)一致；仿射+ΔR恒等式误差4.4e-16。例中ΔR norm约0.126/0.119 Å，总Cartesian变化约0.350/0.325 Å，展示了两种位移不能混报。

## 5. GEM迁移必须有identity路径、逐层路由和非退化cell↔site定义

方案119行“旧scalar广播、新增残差从0开始”应落实成

```
B_new^(layer,head) = B_old + Delta_B^(layer,head),
Delta_B_initial = 0.
```

旧feature顺序、旧edge网络、旧task channels和旧bias路径应在同一输入下保持等价；新增feature不能先替换旧输入再声称广播旧权重便能保持初始输出。可扩展输入并让新列初始为0，或保留legacy residual路径，但必须用固定旧输入证明等价。

不要让“新增残差为0”与“乘在该残差上的门控也为0”串联，两者会使整条新分支初始梯度同时为0。简单一致的约定是新残差最后投影为0、其层门控初始1；新可靠性只作用于新残差，旧bias门控保持1。迁移验收应覆盖实际active logits和teacher-forced trace probability，不能只确认state_dict键已加载。

按head输出可通过`[B,H,L,L]` bias实现；**按层不同还需要backbone显式路由**。当前 `periodic_repair_model.py:238-245` 向base传一个共享`attention_bias`，仅把head维从1扩成H不会自动产生layer-specific bias。执行稿必须点名checkpoint-safe的per-layer输入/选择位置；不能依赖mutable hook在activation recomputation时猜当前层。若这一接口未落实，C只能称head-specific，不能报告“head/layer GEM完成”。

方案122行cell↔site pooling需要保留每个site的环境：

```
u_i = Σ_j valid_ij * phi_ij / max(1, Σ_j valid_ij)
b_(cell_slot k -> site i,h) = f_(k,h)(u_i, cell_features, known_flags)
```

reverse方向单独定义映射，随后广播到该site的E/XYZ槽位。不能先全局pool成一个u，再对同一query的全部key加同一常数；后者在softmax中抵消，或至多区分site-block与prompt-block，不能实现所声称的局部环境选择。无有效pair时保留count=0/known flag，并令不存在的环境为0；padding、unknown geometry不能通过identity placeholder制造“物理特征”。原子数1的输入尤其需有明确的zero-neighbor行为，无需为此另添一套邻居模型。

方案120行与130行还需统一“冻结”的对象。可靠性门控若读取current mask ratio，它会随scalar步骤变化。建议固定：事务内缓存**物理特征phi(x_t)**；当前mask/task门控及最终B(s_j)随每一步重算。不能一处缓存整个GEM bias，另一处训练时用该prefix的当前mask ratio。仅完整事务验收后才替换x_t和重算物理几何；这一选择保留原稿的旧状态语义。

## 6. A/B/C验收措辞与数据分母

方案222–226行A/B/C可作为有限的递进比较，但需固定：A旧方案，B等更新预算的新state/loss/corruption方案且shared GEM，C只在B上新增GEM交互；三个正式起点均为同一raw final checkpoint。短比较后的权重不能自动升级为正式起点，否则来源/预算不再与声明一致。若最终使用短训权重续训，应把其更新与source覆盖计入正式预算，并让对照同步处理。

“等预算”至少记录有效state数、optimizer updates、source/目标覆盖和GPU-hours。B的dense token较多、C有head/layer bias，不应把等步数说成等计算成本；也不需要为硬凑相同GPU-hour改成未声明的不同source覆盖。B相对A同时改变state、CE、对象权重和corruption，能归因于这个组合，不能单独证明某一项CE/shape机制有效；C-B才直接检验GEM增量。

support coverage应按全source/state分母报告，并用共同eligible集合比较action NLL；否则不同corruption admission率会让gated loss变低而非模型更好。原生修复、unchanged、rollback、入场失败继续用全请求分母，方案217行方向正确。

第8/10节原有“仅分析、暂不启动”的操作边界已被用户最新排队授权更新；本次审阅不操作排队，执行稿应由主审统一替换旧边界，避免并存冲突指令。

## 保留不动的主设计

- exact CE为主，ordinal soft weight=0；不对概率均值/逐轴argmax伪结构反传force/MSE。
- 原LLM化学/soft-plan职责、program/N/E硬接口和native数值token ABI。
- construction接受最初无几何；repair读取完整旧整数晶体并冻结物理状态到事务结束。
- row-vector log-SPD的volume/shape构造、同一cell basis/站点身份，不独立规约/重排target与old。
- canonical alias规则、T=.7实际动作概率、现有支持协议和native/tau800分开验收。

以上修订足以把目前方案变成可复算、可迁移的实现合同；其物理收益仍需原稿的有限比较证明。本审阅没有增加新版K4/K8、改变上游LLM或引入新输出测度。
