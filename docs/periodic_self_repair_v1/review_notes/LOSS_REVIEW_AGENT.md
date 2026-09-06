# Loss / state / architecture read-only review

日期：2026-09-06。范围：当前 `DLM_periodic_self_repair` 工作区中的 fresh LLaDA + original MP20 分支。只读源码和少量 CPU 数学计算；没有访问 SSH/GPU、修改实现、启动训练或提交代码。本文不判断运行中的 job39942 已达到何种性能。

## 最先需要纠正的结论

1. 当前代码实现了随机 mask 条件重建，加上旧结构条件下的 teacher-forced scalar reconstruction。部署是固定顺序逐标量构造，再执行一次逐标量 full-cell repair。它可以构成条件生成模型，但没有实现与训练随机 mask 过程对应的迭代离散扩散采样器。
2. 当前 objective 中的 Gaussian `q` 是围绕 clean token 的标签平滑，不是 structured corruption 的 forward kernel/posterior，也不是 score matching。较低的这个 loss 不能直接证明自生成结构的几何误差、力、能量或 SUN 改善。
3. 可直接确认的实现错误是 validation monitor 的 batch mean 被再次除以状态数。job39942 的脚本明确使用 microbatch2；这里记录的 val loss 恰好比正确的 200-state mean 小一半。它影响指标，不影响训练梯度。
4. 下文中 family 权重、覆盖率、温度冲突、几何分支屏蔽是可证明事实；它们造成多大性能损失仍需有限的针对性比较，不能把设计取舍全部写成已确认训练失败。

## 真实优化目标与归一化

源码：`src/crystal_dlm/periodic_base_objective.py:33-48,50-89,113-122`；`src/crystal_dlm/periodic_base_training_data.py:254-288,307-309`。

设每源长度/角度/坐标位置数为 `m=(3,3,3N)`，随机 mask 为 M，`K_f=|M∩f|`。数值单 token loss 为：

```
r = 0.9 delta_y + 0.1 q_sigma
ell(z,y;S,T) = CE(r, softmax(z_S/T))
q_sigma(k|y,S) ∝ exp(-d_f(k,y)^2 / (2 (step_f sigma_bins)^2)), k∈S
```

长度/角度用原始标量距离，坐标用分数坐标圆周距离。schema support 是整个对应轴数值词表，坐标 100 alias 先折叠到 000；legal support 则经过真实当前整数状态的部署约束。

construction `L_c=(1/3)Σ_f 1[K_f>0](1/K_f)Σ_{j∈M∩f}ell_j`。repair 先均匀抽一个 family，再在该 family 的所有位置中均匀抽一个 j，因此 `E[L_r]=(1/3)Σ_f(1/m_f)Σ_j ell_j`，每个 j 的 conditioning 是其 teacher-forced 前缀。

总 loss 是两个 view 的状态均值：`mean_i w_i (L_schema_i + 0.25 L_legal_i)`。weight 不再除以 weight 总和；空 construction、padding 为 0 并保留 batch 分母。这一行为是明确写入 data protocol 的风险定义，不能孤立称为代码 bug。训练数据 54272 状态刚好整除16，无尾部 padding。

construction 的 legal probe 同样先抽 family；该 family 没有 masked token 时 legal loss=0。target 不在实际 legal support 时不把它强行加回，记 conflict，legal contribution=0。这是合理区分 schema 可编码性与部署可达性，但使 legal auxiliary 的实际权重还依赖 family、mask 与约束触发频率。

### 空 family 使严格的等 family 声明不成立

`p~Uniform(0.1,1)`，一个 m-token family 全未被 mask 的概率：

```
P(K_f=0) = E[(1-p)^m] = 0.9^m/(m+1)
```

长度与角度均为 0.18225。坐标 family：N=1 为0.18225，N=2 为0.0759201，N=5 为0.0128682，N=10 为0.00136746，N=20 为0.0000294592。

对“所有 token loss 恰为同一常数”的诊断情形，dense view 的总有效系数由 N=1 的0.81775上升到 N=20 的0.8784902。N=20时 dense coord family 相对 length family 权重为1.22283；两个 view 等量混合后仍为1.10025。实际 loss 与 mask 上下文相关，不能把这些系数误写成对任意 loss 的完整期望公式；但空项造成的差异是真实存在的。

如果希望严格定义为“每源、每 family、每目标位置等权的条件去噪风险”，可选一个清晰的估计式：

```
E_p,M [ (1/3) Σ_f (1/(m_f p)) Σ_{j∈f} M_j ell_j ]
```

它等价于均匀选 family/位置、强制该目标被 mask、其余位置独立按 p mask 的条件风险。这里 `1/p` 是所声明风险对应的采样校正；不应在未先定义目标的情况下声称任何 DLM 都必须添加它，也不应仅凭这个式子声称取得完整轨迹似然。

### N 与 repair 抽样

family averaging 有意使整个坐标 family 与整个长度 family 处于同一量级。由此每个 length scalar 被作为 repair 目标的概率为1/9，每个 coord scalar 为1/(9N)。N=20时前者是后者20倍。这是 family risk，而非所有数值位置等权的 path NLL。

每个源每 epoch 只有一个 repair scalar，两 epoch 总共两个。按采样分布，源的两个 repair 样本都没有 coord family 的概率为4/9；N=20的任意指定 coord 两次都未被选中的概率为 `(179/180)^2=0.9889198`。dense construction 每 epoch 平均 mask 55%的数值位置，但这不能等同于获得同样多的 old-error-conditioned repair 监督。覆盖率事实不等于共享模型不能泛化，不能直接据此宣告训练失败。

## 90/10 的实际强度、熵下界与温度

源码：objective `:39-48,76-78,105-108`；trainer `:133,189`。

对远离边界的数值 token，或100-bin坐标圆周，CPU直接求和得到：

| sigma_bins | q(y) | r(y) | r(每个相邻bin) | CE 最小值 H(r), nat | 最优 exact CE |
|---:|---:|---:|---:|---:|---:|
| 0.75 | 0.5319070 | 0.9531907 | 0.02186735 | 0.2329953 | 0.0479403 |
| 0.50（函数默认，但当前训练未采用） | 0.7865707 | 0.9786571 | 0.01064508 | 0.1183838 | 0.0215740 |
| 0.25 | 0.9993295 | 0.9999330 | 0.0000335238 | 0.000757853 | 0.0000670498 |

因此 epoch2 几乎是 hard CE。epoch 切换改变目标分布及不可消除的熵部分；train loss 跨阶段的下降不能全归于学习。边界或 legal support 的孔洞会改变 q 归一化及下界。固定 val monitor 每次始终用 sigma=.25，故其跨 epoch 比较没有这个目标切换问题（但要修正批大小缩放）。

`sigma_bins` 只随 epoch 改变，与 mask p、structured noise level、晶胞尺度都无关；长度宽度分别为0.075/0.025 Å，角度0.75/0.25度，坐标0.0075/0.0025 fractional。坐标相同 bin 距离在不同晶胞对应不同 Å 距离；triclinic cell 的真实二次误差为 `delta_f G delta_f^T`，包含轴交叉项。独立 scalar CE 可以通过条件模型学习联合结构，但当前 q 本身没有这个物理度量，也不是 Cartesian perturbation 的 posterior。

schema 在 T=1、legal 在 T=0.7。即使 support 完全相同，非均匀 soft target r 的两个 CE 一般不能同时达到各自 H(r)：需要同时满足 `softmax(z)=r` 与 `softmax(z/0.7)=r`。例如sigma=.75，当schema恰好拟合 r(y)=0.9531907，部署 T=.7 给出的中心概率已为0.9907835，而 legal loss 仍要求0.9531907。这是明确的两个目标之间的折中，不能同时解释为对同一个校准分布的 proper fitting；但混合两个不同温度的风险在数学上仍是可优化目标，不能称为必然不收敛。

即便温度统一，`r_full` 在 S 上条件化，与“q先在S归一化，再按90/10混合”也一般不同。sigma=.75，去掉一个相邻 bin 后，前者中心为0.9745004，后者为0.9680774。若最终目标是同一个带约束分布的校准，应统一 support/温度与 target 条件化顺序；若保留两个风险，应明确它是辅助正则而非两个一致的似然项。

## alias：没有发现训练和部署折叠次序不一致

schema objective `:66-75`；scalar legal transform `programmed_path_runtime.py:378-394`；deployment `llada_generation.py:457-483,596-602` 都在原 logits dtype 中使用 `logaddexp(z_000,z_100)`，屏蔽100，随后在 loss/sampler 中应用温度。clean target 在 `periodic_base_training_data.py:86-92,128-132` 预先规范化为000。

因此不能把 alias 说成已有的 train/runtime mismatch。需要明确的是其数学语义：代码定义的是“先在T=1形成物理类别logit，再温度化物理类别”。它不等价于“先以T=.7温度化101个原token，再把000/100概率相加”。相等 raw alias logits 时，两种对其他bin的赔率分别为2^(1/.7)=2.6918与2。fresh所有新output row同均值初始化，因此部署初始 canonical0 mass约0.02647，而普通coord bin约0.009833；这是所选表示的初始偏置，可学习，并非目标泄漏或数值异常。

## 训练与部署的状态分布

construction：data `:254-262` 在clean结构任意独立mask，old=current，避免把被隐藏clean几何泄漏给条件器。runtime `:255-269` 按species program固定顺序，每步只有一个scalar draw，old=x.clone()，可见prefix来自模型自己。active scope 为全部剩余masked numerical，与训练的 transaction mask 一致；unknown noise=-1在训练50% dropout中确实被覆盖，不能称为未训练metadata。

repair：data `:264-275` 先抽family/目标位置，old为一次结构扰动；current把全部numeric mask，再揭示该目标之前的clean前缀。runtime `:321-338` 使用相同full-cell事务和固定old快照，按同一顺序采样，因此“给定同一个旧结构与前缀”的状态语义匹配。真正差异在于旧结构分布和prefix：训练old来自clean MP20附近的三档 bounded strain/Cartesian perturbation，部署old来自自身construction；训练prefix全部正确，部署prefix自身采样。teacher forcing本身合法，但off-policy条件重建loss低不等价于on-policy长序列修复成功。

data `:165-223` 的扰动是三个离散level的一次bounded perturbation；不合支持/编码裁剪时回退clean，未产生量化token变化时noise=0。runtime `:276-280` 先丢弃不满足complete support的construction结果，只有alive进入repair。因此当前repair根本不会尝试拯救未通过构造支持检查的结果。即使支持可行，0.5 Å距离阈值也不是低能/低力判据。需要分别报告construction失败、repair回滚、改变大小与最终结构质量，不能将rollback通过视作修复成功。

## 几何模块在 construction 中的直接可用性

state `state_conditioned_model.py:112,143`：6个cell scalar必须全已知，某site XYZ必须全已知；`periodic_repair_model.py:117,146-158` 只对双端已知、cell已知的site pair施加几何attention bias。

由此：

- 任意被监督construction coord所属site至少有一个mask，该site所有query/key的几何bias均为0。已知site之间的几何bias仍可能通过Transformer中间层间接影响目标，故不能说整个几何分支没有梯度。
- 任意被监督construction lattice scalar使lattice_known=False，整cell所有几何pair bias为0。
- p~U(.1,1)时，6个cell scalar全可见概率 `0.9^6/7=0.0759201`；某两个不同site加cell共12个scalar全可见概率 `0.9^12/13=0.0217253`。这说明该分支在construction有强可见性瓶颈，在完整old的repair中才直接启用。
- 对部分已知XYZ，explicit conditioner把整个site当unknown；剩余可见数字仍在base token输入，但不能通过该site的显式几何特征使用。这里有必要区分当前masked state、noisy visible state与旧快照，不能只增加一个task bit便声称三者等价。

结构层面，attention bias是一个对所有heads/layers共享的scalar；conditioner含绝对坐标sin/cos与program rank；base Transformer仍使用有序token位置。numeric adapter有周期basis，但同时存在独立full-rank new output row residual（`periodic_repair_initialization.py:44-73,94-102`）。这些是可学习的几何归纳偏置，不构成整个模型的平移/置换/晶胞基变换或空间群等变证明。

## 验证与日志

`train_periodic_dlm_from_base.py:189` objective返回batch mean，`:190`仅把mean加入分子、batch size加入分母，`:194`再相除。正确应累加`value*len(examples)`，或者objective直接返回逐状态sum+count。`slurm/240_train_periodic_dlm_from_base.sbatch:30`固定microbatch2；200state分为每rank100state，完全整除2，所以当前reported val loss乘2可恢复正确state mean。microbatch变化/尾batch时不再只是统一比例。

training loss反向的microbatch/accumulation/DDP平均未发现该问题。`:137-138,172-180`的train日志只含rank0：schema_loss_sum/legal_loss_sum是该rank8个state之和，而loss是该rank均值；val是all-rank固定100sources/200states。监控只取val前100sources，未宣称全9047sources评估；仍不足以单独判断跨composition/N泛化。validation阶段丢弃objective metrics和conflicts（`:189`），使eligible coverage及unsupported target变化在一个总loss中不可见。

## 建议交给执行者的最小方案决策

1. 先定义要优化并部署的随机过程。若主目标是真正离散DLM去噪，应让训练state/noise和迭代采样一致，dense numeric监督按明确的family/source风险归一化；若保留scalar事务，就明确它是old-conditioned scalar factorization，用匹配的prefix状态训练，并单独处理on-policy误差累积。
2. 把 construction 学习、局部denoising、模型自身错误修复分为可解释的训练分布与指标。当前original-MP20-only规则下，可先改善扰动覆盖与dense目标；只有另行明确允许自生成/物理teacher数据，才加入那类监督，不能暗中重新引入旧K4/K8。
3. 以hard CE或明确的物理邻域target作为primary risk；平滑强度应有一个可解释的用途。若保持label smoothing，记录H(r)、exact CE及excess CE，不能把不同sigma的raw CE当同尺度。schema/legal选择统一概率语义，或承认并量化它们的折中。
4. 几何分支应根据明确的noisy/current/old state工作；保留完整噪声几何时才有直接pair支持。若引入lattice/symmetry-aware表示，应先说明其对现有六scalar/100-bin编码、支持约束和采样器的影响，不能只把embedding扩展称作几何扩散架构完成。
5. 在新训练前优先修正val求均值；以固定small validation集合同时报告construction/repair、family、N、mask/noise层级的exact CE与legal coverage，并进行一次匹配部署状态的target可达性及修复效果评估。这里需要的是回答建模问题的少量比较，不是穷举防御性测试。

本审阅没有决定哪一种新方案必定提高SUN，也没有执行任何上述改动。主要建议是先把随机过程、几何表示与目标概率定义统一，再让执行模型实施一次可归因的改动。

## 给主方案的补充：exact CE 为主，soft ordinal 仅作消融

建议 primary objective 使用纯 exact CE，并把 ordinal soft target 的权重固定为0作为主基线。理由是：对已声明的 conditioning state s，exact CE 在期望上的最优解为真实离散条件分布 `p_data(y|s)`；它是 proper scoring rule。90/10的期望最优解则变成

```
0.9 p_data(y|s) + 0.1 Σ_x p_data(x|s) q_sigma(y|x,s)
```

即主动把真实后验与人为邻域kernel平滑后的后验混合。它可以是有价值的有限样本正则，但不能不经对照就视作更物理、更正确的生成目标。当前kernel不对应Cartesian/log-strain corruption或量化误差模型，末阶段又几乎关闭；因此没有充分理由让它成为主目标的必需组成。

exact CE 不把相邻错误和远距离错误直接赋不同损失，也不保证学到低能结构。这一局限应由可解释的物理corruption、conditioner、联合采样及评价处理，而非把多个模式的坐标概率平均成一个非真实结构。soft ordinal可以作为一次受控消融：固定其他训练分布、权重、temperature、sampler与预算，只比较一个固定强度，并同时评估exact NLL与真实采样结果；不以其较低/较高raw soft CE裁决优劣。

### 温度和 alias 的两种合法概率定义

令raw logits为z，canonical物理coord类别0代表tokens000与100。

1. **先合物理类别，再温度化物理类别（当前代码）**：`g_0=logsumexp(z_000,z_100)`，其余`g_k=z_k`；`p_T(k)=softmax(g/T)_k`。这等价于先取T=1的token分布并合并aliases，再对100个物理类别的概率做power tempering。
2. **先温度化tokens，再合物理类别**：`p_T(0)=p_token,T(000)+p_token,T(100)`；对应合并logit为`g_0,T=T logsumexp(z_000/T,z_100/T)`。除T=1或某alias完全占优外，它与1不同。

两者都可定义有效分布；现在schema/legal/deployment的fold次序属于1，legal与deployment一致。新方案若保留这个行为，应在公共概率函数中明确“support mask + canonical folding + temperature”的次序，用同一个函数计算训练NLL、采样及trace log-probability。不能把1与2的变化当成无语义影响的数值重构。

建议primary在唯一声明的T_train下拟合exact legal likelihood，并在同温度给出主要校准/采样结果。若仍采用T=.7作为部署解码控制，可以固定它，并让所称actual-policy NLL准确使用该分布；若训练采用T=1而推理改T=.7，应把后者明确列为decoder setting并单独比较，不再把两种温度的NLL合成一个声称校准于同一分布的主loss。

去掉soft target后，单个确定标签的T=1与T=.7 CE都能在无穷logit margin处趋近0，所以前述每样本soft target的正熵最小值冲突消失；但对具有多个真实可能target的相同condition，期望exact CE依旧要求拟合非退化条件分布，不同temperature的双重目标仍不是同一个校准风险。不能据此保留温度混合而声称问题全部自动消失。
