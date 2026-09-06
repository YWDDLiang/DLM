# 第二轮交叉攻击：晶体机制报告与主张审计

2026-09-06。审查对象是 [CRYSTAL_DIFFUSION_MECHANISMS](CRYSTAL_DIFFUSION_MECHANISMS.md) 与 [CLAIM_AUDIT](CLAIM_AUDIT.md)，不重写任一首轮报告。本轮只写本文件；没有修改模型、训练、评估协议或其他人的文件，没有 SSH、GPU、安装依赖或派子代理。

初次完整核对时的文件 SHA256：

- CRYSTAL：`d94a30e209bfb650c8f2115b07d1d37ca0716f3bf1c6a7bc9c3c830ba84e1f35`。本轮指出的基归一化问题后来已由原作者在其报告中修订，不能把这里的旧快照问题误称仍未响应。
- CLAIM：`f9758f56092357bbf9459b36fff99fc9d799ca25019dd704c85c37f543a21441`。

本报告的“阻断”是**阻断某项理论保证、尺度换算或直接复用候选**，不是要求停止正在运行的 V2。当前仍是全面审计阶段，trick/V3 未定案。

## 1. 结论概览

CLAIM 的主要限定经得住攻击：正样本 CE 不必有显式负能量边界才能学习好分布；joint 分布不要求 simultaneous joint updates；fixed-old 可以是正确的后验观测；soft 条件的独立性必须对齐 DLM 实际输入。没有找到推翻这些限定的反例。

CRYSTAL 有一处实质的**论文记号与作者代码混同**，已经反馈并由该 lane 修订：DiffCSP++ 所引代码将六个 basis 单位化，不能对那份代码使用未单位化的 `v=3k6` 或 shape-noise 归因。此外，逐原子量化与群作用不交换、SGWN 的 ASU fold 测度、以及 SGEquiDiff 某个 ODE 导数 helper，构成需要进一步限制迁移声明的具体问题。

我接受对自己首轮的两项反驳：**经验分布版 C→D 的理想极限可以没有任何 novelty**；**active attention-query 的几何边为零不意味着 active hidden 只能经 Transformer 多跳获得几何**，已有 conditioner 的 global cell 通道也能传递。以下明确其成立条件。

## 2. 阻断问题与已修订项

### B1. DiffCSP++ 的 basis 归一化：不能据错误映射决定 V3 噪声尺度

审查快照 CRYSTAL §4.1 将 DiffCSP++ 概括为非单位化六基、`v=3k6`，进而对比其 k 噪声与 V2 Frobenius 正交 shape 噪声。**这只能用于明确的论文未单位化记号，不能描述它引用的作者代码。**

实际作者快照 `e82e7ffa7cb2a383bde69b067a343ca137d73e47` 的 [get_basis:33](https://github.com/jiaor17/DiffCSP-PP/blob/e82e7ffa7cb2a383bde69b067a343ca137d73e47/diffcsp/pl_modules/lattice/crystal_family.py#L33) 除以每个 basis 的 Frobenius norm；[hexagonal offset:50](https://github.com/jiaor17/DiffCSP-PP/blob/e82e7ffa7cb2a383bde69b067a343ca137d73e47/diffcsp/pl_modules/lattice/crystal_family.py#L50) 也补了相应的 sqrt(2)。作者原始文件与本地 cached 文件逐 byte hash 相同：`fb62ddafa7587ba2a05aac0d11f8c912dc54cf28901ed44ed08ae3a251b79c35`。

仅抽取 `get_basis` / `get_spacegroup_constraint` 做 CPU 算术，得到：

| 检查 | 数值 |
|---|---:|
| 六个 basis 的 norm | 均约 0.99999994（FP32） |
| basis Gram 与 I 最大差 | 5.96e-8 |
| trace(B6) | 1.732050776，而不是 3 |
| 代码 hexagonal 第一项 offset | −0.388418108 |
| 论文未单位化对应 offset | −0.274653072 |

因此代码的体积坐标关系是 `v=sqrt(3)*k6`。五个 shape basis 在代码中也是正交单位基，与 V2 的 shape basis 只存在顺序等表示差异；V2 **独立设置 volume/shape σ** 仍是不同选择，但不能把差异归为“作者代码 shape 基未归一化”。

**状态：已由 CRYSTAL lane 接受并修订。** V2 自己的 `S=.5 log(LLᵀ)` 与正交无迹基计算不因此出错。这个发现否定的是迁移尺度的错误理由，不是 DiffCSP++ 或 V2 的总体方法。

### B2. “100-grid 对群作用闭合”仍不足以保证独立量化后的空间群

CRYSTAL 对 1/3 不能在 100-grid 精确表示的警告成立；但还有更强的遗漏：**即使群操作把这个网格映回自身，逐原子 round 也未必与群作用交换。**

取三方平面的整数操作

\[
R=\begin{pmatrix}0&-1\\1&-1\end{pmatrix},\qquad R^3=I.
\]

R 对 `Z_100²` 是可逆映射。连续 seed `f=(0.2349,0.3551)` 的三点 orbit，分别按本项目 half-up / periodic codec 量化，得到整数 bins

\[
Q(\{f,Rf,R^2f\})=\{(23,36),(64,88),(12,77)\}.
\]

R 将第一点映到 `(64,87)`，不在该集合中。反之，**先量化自由 seed，再用整数群作用展开**得到

\[
\{Qf,RQf,R^2Qf\}=\{(23,36),(64,87),(13,77)\},
\]

才对群闭合。CPU 已逐集合检查：前者 false，后者 true。

所以 V3-S 必须同时说明：群动作/平移分母、自由坐标的离散域、orbit 编译、量化顺序、碰撞与特殊位置、数值容差。不能只把现有连续 orbit 展开后的每个坐标送入 `arrays_to_dynamic_tokens`，便承诺保持 SG。

这也不是禁止离散 SG 编译。可选路径是精确离散自由参数与群展开、受限可表示子群/设置，或明确降低为容差意义的对称性；后者不能重新包装成严格全 230 群保证。

### B3. SGWN：canonical ASU 不会自动救回错误的 forward 密度

CRYSTAL §8.2 的原反例通过复算：非正交 fractional R 保持 hexagonal metric H，却不保持 Euclidean metric；等方 Euclidean orbit-center Gaussian sum 的全空间群不变性失败。这个判断不能被“只在 ASU 训练”直接撤销。

对一般位置、无 stabilizer 的有限群示例，先从 `mu+epsilon` 加 Euclidean noise，再 fold 到 fundamental domain，正确 push-forward 密度包含**逆像**：

\[
q_{\rm fold}(x\mid\mu)\propto
\sum_{g,n}\varphi_\sigma(gx+n-\mu)|\det R_g|.
\]

而公开 helper 使用的是**orbit centers**：

\[
q_{\rm centers}(x\mid\mu)\propto
\sum_{g,n}\varphi_\sigma(x-g\mu+n).
\]

在所用 Euclidean metric 下为 isometry 时两者可重排相等；非正交 fractional 操作一般不等。考虑同一 C3 例子，选择 orbit 中字典序最小的代表定义一个合法 ASU。`mu=(.11,.87)`、`x=(.08,.82)` 都在其内部，sigma=.2、integer images 每轴 −8…8：

| 项 | centers helper 形式 | fold push-forward 形式 |
|---|---:|---:|
| 未归一密度 | 1.344164723 | 1.289778893 |
| log-density 梯度 x 分量 | −0.872308194 | −1.395314212 |
| log-density 梯度 y 分量 | 1.913959336 | 3.260141394 |

这是本轮独立 NumPy + 有限差分结果，可嵌入三维一般位置。点位不在 ASU 边界；差别不由边界歧义或舍入解释。

作者代码 [compute_loss:148](https://github.com/rees-c/sgequidiff/blob/2231031fdabbf6b372031f3f233f8638922fa91e/src/model/diffusion/diffusion_model.py#L148) 先加 projected fractional Gaussian，再 `wrap_frac_coords_into_asu`，随后调用 [orbit-center helper](https://github.com/rees-c/sgequidiff/blob/2231031fdabbf6b372031f3f233f8638922fa91e/src/model/diffusion/diffusion_utils.py#L452)。所以“helper 在 ASU 限制上的某种有效性”仍需证明它对应这个 forward kernel，不能把 domain restriction 作为默认免责。

**成立条件与最强反证边界：**

- 若噪声 covariance、metric、群作用和参考测度配套，centers 表达可以成立；不能只给 loss 加 metric、却继续用不匹配的 Euclidean forward noise。
- 若作者使用另一明确定义的 ASU density/extension，它仍可能产生有效生成器；此处阻断的是直接继承全空间 SG-invariant score / 指定 forward 的 exact reverse 主张。
- 对低维 Wyckoff 子空间还要核其 induced measure 与 Jacobian；本反例使用一般位置，未借低维特例制造问题。
- 没有运行作者 checkpoint，也没有复算它的 SUN。不能把这项数学/实现疑点推成“论文方法实证无效”。

### B4. SGEquiDiff ODE 导数 helper 的明确链式法则遗漏

作者固定 SHA `2231031fdabbf6b372031f3f233f8638922fa91e` 的 [NoiseScheduler.d_sigma_sq_dt:932](https://github.com/rees-c/sgequidiff/blob/2231031fdabbf6b372031f3f233f8638922fa91e/src/model/diffusion/diffusion_model.py#L932) 对

\[
\sigma(t)=\sigma_{\min}r^{(t-1)/(T-1)}
\]

返回的是 `2*sigma'(t)`，而

\[
\frac{d\sigma(t)^2}{dt}
=\frac{2\sigma(t)^2\log r}{T-1}
=2\sigma(t)\sigma'(t).
\]

只截取该方法的 AST，在 CPU 上注入其声明参数，并用中心有限差分核对。`T=1000, sigma_min=.002, sigma_max=.5` 的例子：

| t | 代码返回 | 正确解析值 | code / correct |
|---|---:|---:|---:|
| 1 | 2.21079516e-5 | 4.42159032e-8 | 500 |
| 500 | 3.48592741e-4 | 1.09930491e-5 | 31.7103 |
| 1000 | 5.52698791e-3 | 2.76349395e-3 | 2 |

有限差分与正确式一致。作者 raw URL 与 cached 文件 byte hash 相同：`9481ea64af79b647b759089e2a368c3354976c66c98e2362c44d06e3b5414b64`。

**必须分开调用路径：** `forward()` 的 ODE field / divergence 调用此 helper，`log_prob` 使用对应 ODE；PC `sample()` 用离散 `sigma[t+1]²−sigma[t]²`，并不调用它。故它阻断直接复制该 ODE/likelihood helper，**不构成否定该论文 PC 采样/SUN 的证据，更与本项目 V2 无直接运行关系**。

### B5. C→D 的 empirical-data 极限不能支持 SUN 改善优先级

接受 crystal lane 的反驳，但补充精确前提：如果 `p_*` 是有限训练经验分布，且 novelty reference 包含或匹配其全部支撑，则完美 Bayes posterior D 的输出只在这些训练支撑上。即使 C→D 保持 `p_*`，Novelty 仍为 0，SUN 也为 0。

这是对“stationarity ⇒ discovery/SUN 好”的直接反例，比“未必单调”更强。它不要求训练点能量差；即使所有点稳定也成立。

不能扩大成未经检查的本项目数值结论：本项目离散 targets 与原始 reference CIF 之间存在 codec/matcher；精确的 Novelty=0 需要它们被 reference 匹配。对训练经验分布没有支撑的**未见条件 C**，其 conditional `p_*(G|C)` 甚至没有定义。需要一个可泛化的未知 population，而不是把经验分布 posterior 的极限当成新组成发现保证。

因此 C→D 仍可保留为比盲目 re-denoise 更有依据的**kernel 诊断**，但不是 SUN 改善的定理，也不能单凭这一证明排到所有 V3 候选之前。还须检查同 kernel、正确 noise channel、完整支持和有限模型拟合；首轮已列出的 prefix admission 反例仍有效。

## 3. 非阻断修订与应避免的假警报

### N1. Affine 状态投影出现在 noise 代码里，不自动就是 bug

DiffCSP++ 的公开代码确实把 random noise 也送入 affine `P(z)=Mz+b`。但 `M²=M, Mb=0`，于是

\[
P(aP(x)+cP(\epsilon))=aMx+cM\epsilon+b,
\quad \|P(v)-P(\epsilon)\|^2=\|M(v-\epsilon)\|^2.
\]

forward 末端与 reverse [L274](https://github.com/jiaor17/DiffCSP-PP/blob/e82e7ffa7cb2a383bde69b067a343ca137d73e47/diffcsp/pl_modules/diffusion.py#L274) 都重新施加 P。因此 fixed-offset 不会在这些自由坐标里累积；不能只看到一个 affine-noise 调用便宣布非零均值错误。状态 affine 投影与 tangent/noise linear 投影在理论上应区分，但必须审计**合成后的实际操作**。

原 lane 已接受这一补充并修订。

### N2. Fractional covector / velocity 公式通过；joint score 还需参考测度

固定行向量 L、R=FL 下，CRYSTAL 的

\[
v_F=v_RL^{-1},\qquad s_F=s_RL^T
\]

成立。CPU 对 `log p_R=-||R||²/2` 用非正交 L 核对 fractional score，有限差分最大误差 `4.50e-11`。空间群 fractional covector 按逆转置变换，metric-raised vector 才按 tangent vector 规则变换，也成立。

更完整的 joint-score 迁移还需明确测度。若在全 9 维可逆 L chart 下比较 Lebesgue densities，则

\[
p_{F,L}(F,L)=p_{R,L}(FL,L)|\det L|^N,
\]

\[
s_L^{(F,L)}=s_L^{(R,L)}+F^Ts_R+N L^{-T}.
\]

在六参数或 log-SPD gauge 下，还要用其相应 Jacobian / induced measure。此式不是让 V2 加一个力项，而是防止未来把 displacement、force、density score 与 lattice token 混成同一个对象。

### N3. 离散极限推导通过，但不覆盖任意 finite-grid / hard-support sampler

最近邻 rate `D/h² * exp((log pi(y)−log pi(x))/2)` 在正、足够光滑的 pi、固定维数规则网格、常 D、适当边界和极限下，具有声明的 drift/variance 极限及 detailed balance。这是合法的“离散并非不可能”的反例。

它不是对现有有限100-grid、状态相关合法集、variable-dimension Wyckoff strata、独立逐轴 token、截断及 rollback 的自动证明。state-dependent mobility、边界条件与参考质量需重新写出；采样 dt 与 h² 的稳定性/计算成本也不能忽略。

### N4. fixed-old 与 working canvas 的时钟区分通过

`p(Y|U)=prod p(Y_j|U,Y_<j)` 中 U 是观测，应该保留。`H_j=h(U,Y_<j)` 不增加给定观测下的信息。CRYSTAL 明确把 H 定位成有限模型的特征候选，而非每 scalar 必须更新物理状态，这一修订成立。

H 仍可帮助有限模型，也可能是无效 hybrid：例如已经改 lattice、尚保留 old fractional sites 会导致暂时过密。应同时保留 U，并训练同一 H 定义；不应把未训练 H 在推理时强塞进去或把它当真实下一 noise-time。

### N5. 已有 GNN 通路：我方首轮也需限缩解释

沿实际代码确认：[conditioner:234–244](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/periodic_state_conditioning.py:234) 将 known-site neighbor 环境汇总到 global cell，再经 `site_update` 广播到所有 present site；[state_embeddings:174](D:/codex_work/ai4s/DLM_periodic_self_repair/src/crystal_dlm/state_conditioned_model.py:174) 加入 E/XYZ hidden。因此我方首轮只提 Transformer 多跳作为绕路是不完整的。

正确表述是：**active query 的该 attention-bias 边为零；active hidden 后期仍可直接得到 conditioner 的全局几何。** 初始六 lattice scalar 没有完整可解码 cell/site 供该图通路使用的判断仍成立，但 conditioner 此时依然可处理物种/program/known flags 等非几何信息。当前已提交的 lattice tokens 和 positive-volume logit mask 也仍可提供几何约束；不能把 GEM 不可见扩大为“前六步不存在任何几何计算”。

新增 GNN 还需更严的新颖性区分：当前 pair mask 在 known sites 上本质是全连接，额外 hops 不自动扩展空间感受野；主要增量是更高阶组合/迭代环境计算。现有 relative fractional sine/cos 也不是纯径向输入，不能把“首次有方向信息”作为无需验证的新贡献。显式 Cartesian 方向/三体或 equivariant features 可以是不同归纳偏置，但需要同输入、同预算的实证。

### N6. “6.47 sigma”是两方法跨度，不是已测的训练 corruption OOD

CRYSTAL 用 In6Yb2 的 46.4117→12.7358 VPA 得到 log 差 −1.293，并除 V2 最大 sigma_v=.2。这一算术正确；但两端是既有参考与方法生成结构，不是同源 clean MP20 与其 noisy state。

应称“相对两方法跨度的尺度对照”，不能用它估计真实 self-error 的标准差、Gaussian tail probability 或据此选择新的 sigma。参考也可能不是应恢复的 GT，多晶型差异也未必是错误。若要校准 corruption，需要同源有意义的对应，或比较条件生成分布与实际 synthetic-old 分布的统计支持，而非把方法间差值当训练噪声。

### N7. 小的数学措辞修订

- `E(x)=(x²−1)²` 在一维 x=0 是不稳定局部极大值，不是严格意义的 saddle；加 `+y²` 才给 Hessian `diag(-4,2)` 的 saddle。原结论“强对称不保证稳定”不受影响。
- Wyckoff 自由变量那一段在全篇行向量记号中用了 `A u+b` 的列向量写法；局部声明为列坐标或给 F 转置即可，无需改算法。
- “确定性对称 field 无法跨过 fixed point”应注明通常需要 ODE 解唯一性/局部 Lipschitz 等条件；离散数值误差或非光滑/非唯一解不在同一个保证内。

## 4. 通过项：不为了攻击制造问题

1. **硬 SG/Wyckoff 与 soft hints 不同。** 轨道同物种、multiplicity/count 相容、晶胞设置、0D 重复占据限制、特殊位置和更高意外对称都需要处理。严格保持某群作用不等于检测到的最大 SG 恰好等于输入群。
2. **联合分布不强制同一步联合更新。** 先 L 后 F 的精确 conditional factorization 可以表示同一个 joint；是否容易训练及误差传播是实验问题。SGEquiDiff 的分阶段设计可反驳“joint diffusion 是普遍必要条件”，但其消融不证明本项目应无条件照搬。
3. **forward 独立不代表 reverse 条件独立。** 多个头读同一个完整几何状态可以学习耦合；反过来，正确 marginals 一次独立并行采样也不等于 joint posterior。
4. **CE 与能量的限定合理。** 学稳定的 population 并不需要显式负能量样本作为逻辑必要条件；但归一 CE、低 loss、对称性和几何有效性都不构成本项目 SUN 的现成保证。
5. **soft 条件独立命题通过，作为条件命题而非实际数据结论。** CLAIM 已明确 Planner 全输入与 DLM 可见输入的区别；M 相关而未直接给 DLM 时，S/P 可传 M 信息。应按真实模型输入审计，不从低 annotation agreement 推定 bug。
6. **历史负结果与证据缺失的限缩合理。** 一次有限预算失败不否定整个机制家族；被删除资产不能声称已逐样本复算；缓存修复不是模型改进；工程通过与物理实现分开。

## 5. 新物理证据对 teacher-selection 归因的限制

本轮只读取 [ACTUAL_SAVED_PHYSICS_AUDIT](evidence_physics/ACTUAL_SAVED_PHYSICS_AUDIT.json) 并从其 `stop_crossings` 重算计数，没有调用 MLIP：

| endpoint / 全标签数 | not_converged | 其中 optimizer=true 且 <500（含0步） |
|---|---:|---:|
| K8 native / 1200 | 504 | 399 |
| K8 tau800 / 1200 | 763 | 762 |
| raw-v1 native / 256 | 105 | 77 |
| raw-v1 tau800 / 256 | 164 | 164 |

这些是 `not_converged` 子类的计数，不等于所有未 verified 失败；不能据此抹掉 native 中 `invalid_terminal`、跑满步数等其他问题。

特别对 tau，低 verified 不能普遍解释成“最大500步不够”。优化器已经按自己的判据停止，外部 force/stress/其他验收仍未通过。单纯提高最大步数在同一提前停止判据下不必增加任何计算。它也不自动说明外部阈值错：优化器广义坐标停止与独立物理阈值可以是不同契约。

如果旧 teacher 只在 verified 候选上分配权重，先发生的是选择

\[
p_{\rm selected}(G\mid C)
\propto p_{\rm proposal}(G\mid C)\mathbf1[V(G,C)=1],
\]

然后才有能量/路径 reweight。V 包含停止、force/stress、geometry 与一致性条件；因此它不是“稳定样本”的同义词，而是协议选出的可验证子分布。某些条件没有候选，某些条件仅一个候选；teacher 偏好大小不能代表原始全请求分布。

**目前不能进一步确定的方向：** 当前提供的是这些 evaluation endpoints 的交叉计数，不是原 K4/K8 training teacher 的全源分层表。不能直接声称该筛选必定偏向某个 VPA、N 或化学类别，也不能说这已解释 student 的退化。需要在已有 train labels 上比较 admission 与 VPA/N/组成/原始力应力、目标覆盖及权重；仍应保留未入选分母，不为审计重新放宽标签或筛新训练数据。

## 6. 仍需实证的候选，不从本轮直接定 V3

| 候选/主张 | 当前结论 | 最小必要证伪 |
|---|---|---|
| C→D 比一次/多次 D 提高 SUN | 理论只给特定目标分布不变性；经验支撑可 Novelty=0 | 固定起点/条件、同 noise kernel/channel、noise-only 与 rollback 对照、全请求 SUN/NU |
| working-canvas 更有效 | 它是确定性特征，不是新统计信息或必须的 clock | 保留 U，训练同一 H；按 prefix 改 cell 程度分层与同端 SUN |
| 多跳/方向 GNN 有新价值 | 已有 learned message/global hidden 通路；新颖性尚未成立 | 相同输入/状态与参数预算，区分表示、action 消费、最终 SUN 三层 |
| SG/Wyckoff 是主要瓶颈 | 有低维合法性的优点，但不能保证稳定/novelty且信息条件改变 | 明确可表示子集与 codec-群交换性，再做同条件对照；不拿 oracle G/W 当 de novo |
| 预测 soft 改善控制 | 输入独立性/metadata 信息边界已正确提出，实际效用未知 | S/P 正交干预、实际 adherence、fallback/goal/重复组成分层 |
| 当前门槛筛掉的 teacher 是坏结构 | verified 是协议门，不是完整物理真值排序 | 已有标签上的原因/停止交叉与分层覆盖；不把失败状态名当唯一原因 |

本轮最重要的成果是**缩小可声称的保证并找出可复核的迁移错误**，不是宣布某个新架构必胜。V2 的既定完整终点评测仍应继续，trick/V3 的采用由后续实证和交叉审计共同决定。

## 7. 本轮复核收据

- 重新执行了 crystal lane `mechanism_checks.py` 的纯算术部分，移除文件写入和 print 语句后执行；log-SPD 与原 SGWN 数值一致。未改其 `.py/.json`。
- 从两份已读作者源码的 AST 只提取 `get_basis` / constraint 与 `d_sigma_sq_dt` 方法做 CPU 算术；没有 import 作者模型、数据集、checkpoint 或训练框架。
- 使用 NumPy 独立计算 ASU preimage-density 反例与 100-grid orbit 反例；使用有限差分检查 score pullback 和噪声导数。数值已列正文。
- 已通过原作者固定 raw URL 核对两个代码缓存的完整 byte hash。原论文来源为 [DiffCSP++ v2](https://arxiv.org/html/2402.03992v2) 与 [SGEquiDiff v3](https://arxiv.org/html/2505.10994v3)，没有把第三方博客当理论证据。
- 本文不声称复现外部模型指标；本项目物理停止计数来自已保存审计 JSON，V2 live 状态由主审管理。
