# 连续 diffusion 的训练本质、SDE 条件与双目标后训练审查

2026-09-06。针对“先看它如何训练、学习的数学本质是什么，再考虑与现有故事和优化结合”的独立审查。
范围限于本地源码、原始论文、推导和 CPU 数值反例；没有训练模型、使用 GPU、访问远端或修改本轮实验。
本文修正 23 初稿中过早使用“数学上可以做”及优先推荐实际 PC 路径训练的判断。

## 1. 当前可以作出的判断

**同一 A/B 终态目标可以定义连续结构分布上的教师；这尚不证明现有 model494 能通过某个简单后训练损失实现该教师，更不证明连续 SDE 或 SUN 收益。**
这里存在三个不同的问题：

1. 教师分布是否存在、有限候选是否有共同收益；
2. 该教师是否对应可容许的连续漂移控制，或现有有限步采样器可表达的转移；
3. 有限数据训练是否学到这个控制/分布，并在相同评价协议下保留收益。

第一项在有限 verified 池中可直接计算。第二项需要噪声、初始分布、边界和模型表达能力等条件。第三项目前没有新实验支持。
因此，**本轮继续完成 K4/K8；连续扩展值得保留为下一轮假设，先验证数据中是否存在可学习的 B 对比，再选择训练路线。**
当前没有依据预先指定“完整 PC 路径 NLL”一定优于“加权终点去噪”，也没有依据因 SDE 很难就断言方向不可能。

## 2. 源码核验：它究竟在训练什么

本地审查文件为 [diffusion.py](../../src/vendor/crysllmgen/models_ddpm/diffusion.py)、
[diff_utils.py](../../src/vendor/crysllmgen/models_ddpm/diff_utils.py)、
[dataset.py](../../src/vendor/crysllmgen/models_ddpm/dataset.py)、
[diff_train.py](../../src/vendor/crysllmgen/diff_train.py)、
[cspnet.py](../../src/vendor/crysllmgen/models_ddpm/cspnet.py)。
前四个文件 SHA256 分别为：

| 文件 | SHA256 |
|---|---|
| diffusion.py | 88c38d3fc237001e163c01adeb6296c795497f15a54067a487a9527b3859d208 |
| diff_utils.py | 91dd347d9f20932f6505269fa247ad4ada1276f7ad51610eac0efe373143ff2e |
| dataset.py | 987b60e1329ae35078d1d626404c36340c232243a684cdfdbdd44b591265dfd0 |
| diff_train.py | be4ac65bde0f7bb900f20130290636269afe8e37b64e465adc80b3883a56a86f |

主任务此前已核对这些 hash 与运行端一致；本审查只重新计算本地 hash，没有访问运行端。
官方 [CrysLLMGen 训练说明](https://github.com/kdmsit/crysllmgen#usage-guide)给出 MP20 的 1000 个噪声时刻、500 epochs 训练命令。
代码与默认命令不是 model494 的完整训练履历，不能仅由“494”推断精确训练步数、验证选择、数据快照或物理目标。

### 2.1 数据与实际损失

训练入口读取选定数据集的 train.csv；MP20 路径为 data/mp_20/train.csv。
MaterialDataset 采用 Niggli 化、非 primitive 转换，返回晶格参数、分数坐标、原子类型等。
虽然预处理缓存 heat_ref，返回的 Data、CSPDiffusion.forward 及 loss 均没有使用该能量作为监督。

固定原子类型序列及原子数，记干净晶体为 \(Y=(L_0,U_0)\)，
\(L_0\in\mathbb R^{3\times3}\)，\(U_0\in\mathbb T^{3N}\)，其中 \(\mathbb T=\mathbb R/\mathbb Z\)。
每个 batch 中，每个晶体均匀抽取 \(t\in\{1,\ldots,1000\}\)，独立生成晶格和坐标噪声：

\[
L_t=\sqrt{\bar\alpha_t}L_0+\sqrt{1-\bar\alpha_t}\epsilon_L,\qquad
U_t=(U_0+\sigma_t\epsilon_U)\bmod1.
\tag{1}
\]

decoder 读取 \(t\)、原子类型、当前 \(L_t,U_t\)，输出 \(f_\theta,g_\theta\)。
目标是晶格噪声 \(\epsilon_L\) 及坐标的归一化负 wrapped score：

\[
\ell(\theta)=
\operatorname{MSE}(f_\theta,\epsilon_L)+
\operatorname{MSE}\!\left(
g_\theta,-\frac{\nabla_{U_t}\log q_t(U_t\mid U_0)}
{\sqrt{n_t}}\right).
\tag{2}
\]

这里 \(n_t\) 对应 sigmas_norm，是按各噪声尺度 Monte Carlo 估计的单坐标 score 二阶矩。
实现的 d_log_p_wrapped_normal 虽以 d_log 命名，实际返回负 score：

\[
\frac{\sum_{k}(u+k)\sigma^{-2}e^{-(u+k)^2/(2\sigma^2)}}
{\sum_{k}e^{-(u+k)^2/(2\sigma^2)}}
=-\partial_u\log\sum_k e^{-(u+k)^2/(2\sigma^2)}.
\tag{3}
\]

源码对 \(k=-10,\ldots,10\) 截断。后面的 score 恒等式针对精确周期核；数值实现仍有截断、浮点和归一化估计误差。
采样时先乘回 \(\sqrt{n_t}\) 再减去预测，因此现有符号相配。

训练使用 Adam、初始学习率 0.001、梯度逐值裁剪 1；学习率调度与 best checkpoint 判断读取训练 loss。
这些选择使“best”表示该训练准则下的选择，不表示最低能量、最优 SUN 或最好的实际精修链。

### 2.2 MSE 的总体最优解

设 \(c\) 包含原子类型与数目，\(r_0(Y\mid c)\) 为数据分布，
\(r_t(z\mid c)=\int q_t(z\mid Y)r_0(dY\mid c)\)。
平方损失的条件期望恒等式给出，在函数类不受限且相关积分可交换时：

\[
f^*(z,t,c)=\mathbb E[\epsilon_L\mid Z_t=z,c],\qquad
\nabla_{L_t}\log r_t(z\mid c)
=-\frac{f^*(z,t,c)}{\sqrt{1-\bar\alpha_t}},
\tag{4}
\]

\[
g^*(z,t,c)=
-\frac{\mathbb E[\nabla_{U_t}\log q_t(U_t\mid U_0)\mid Z_t=z,c]}
{\sqrt{n_t}}
=-\frac{\nabla_{U_t}\log r_t(z\mid c)}{\sqrt{n_t}}.
\tag{5}
\]

两个分量均以**联合的**带噪晶格和坐标为条件，所以不是分别拟合两个独立的晶体分布。
原始 MSE 等价于按 \(1-\bar\alpha_t\) 和 \(1/n_t\) 加权的 score 回归；
任意正的纯时间权重不改变逐时刻、不受限函数类中的最优 score，有限网络共享参数时则会改变拟合取舍。
这一训练解释与 [Score SDE 的 §2–3](https://arxiv.org/html/2011.13456v2#S3)一致；以上符号转换来自本地代码。

还需区分数据权重：源码晶格 MSE 按晶体矩阵元素平均，坐标 MSE 按 batch 中所有原子坐标平均。
后者并不自动等于当前按组成固定分布的教师平均。若做加权 DSM，必须明确先怎样归约每个晶体，再施加组成/提议/候选权重。
这个归约选择是 DSM 的统计目标，不应冒称现有完整 token 路径 NLL 的等价写法。

### 2.3 学到的 score 不是能量、力或弛豫算子

模型学习的是“哪些带噪数值更像训练晶体分布”的梯度场。它并未监督以下对象：

- 势能 \(e(y)\) 或物理力 \(-\nabla e(y)\)；
- 公共弛豫映射 \(R(y)\) 或到某个吸引盆地的确定性修复；
- A/B、hull、SUN 或弛豫步数；
- 独立的原始 DLM 提议、typed Plan 或 program。

只有另行证明数据按同一温度、同一能量、同一参考测度服从
\(r_0(y)\propto e^{-\beta e(y)}\)，才可把干净分布 score 与能量梯度关联。
MP20 的收录、筛选和结构计算过程不提供这种 Gibbs 采样假设。
即使该假设成立，带噪 \(r_t=q_t*r_0\) 的 score 也通常不是 \(-\beta\nabla e(z_t)\)。
因此，不能从“数据多为合理晶体”推导“模型已学会 CHGNet 力场”。

噪声也不是物理热浴：晶格在固定矩阵参数化中加噪，坐标在 fractional 环面加噪，
同一 fractional 位移在不同晶胞中对应不同笛卡尔距离，\(t\) 是噪声刻度。
clean 晶格转换采用固定矩阵表示，带噪矩阵可离开有效晶胞区域；晶胞基、同元素置换和原点平移仍需单独处理。

## 3. SDE 是正确背景，但现有 PC 链并非已证明的精确反演

### 3.1 理想的正向与反向过程

对离散 schedule 作合适的连续插值，式 (1) 对应的自然形式为：

\[
\begin{aligned}
dL_t&=-\tfrac12\beta(t)L_t\,dt+\sqrt{\beta(t)}\,dW_t^L,\\
dU_t&=\sqrt{\tfrac{d}{dt}\sigma^2(t)}\,dW_t^U\quad\bmod1.
\end{aligned}
\tag{6}
\]

状态空间可先取固定表示下的 \(\mathbb R^9\times\mathbb T^{3N}\)。
这只是参数化空间，并非已经构造了对所有晶胞基变换取商后的物理晶体流形。
在正则性、密度和初始边缘条件成立时，令 \(f\) 为正向漂移、\(a=\Sigma\Sigma^\top\)，
将生成时间写作 \(s=T-t\)：

\[
dZ_s=\{-f(T-s,Z_s)+a(T-s)\nabla\log r_{T-s}(Z_s\mid c)\}\,ds
+\Sigma(T-s)\,dW_s.
\tag{7}
\]

这里扩散矩阵与状态无关；若改成状态相关矩阵，反向漂移还需相应散度项。
精确恢复 \(r_0\) 要求从 \(r_T\) 出发、使用精确 score、正确解 SDE，并处理 \(t\downarrow0\) 的边界。
晶体联合去噪和流形扩散的相关基础见
[DiffCSP](https://arxiv.org/abs/2309.04475)、
[Riemannian Score-Based Generative Modelling](https://arxiv.org/abs/2202.02763)；
它们不替本项目证明当前改动后的入口。

### 3.2 实际执行的过程

sample 把原始 DLM 坐标及晶格直接标为 \(t=800\)，没有执行式 (1) 的正向扰动。
按当前 schedule，训练在该时刻看到的晶格约为
\(0.306610L_0+0.951835\epsilon_L\)，坐标噪声 \(\sigma_{800}=0.198870\)；
第一步 corrector 的坐标噪声标准差为 0.177875 fractional。
所以不能把 tau800 简称为“只做微小局部修复”，也不能用标准逆扩散定理证明该入口恢复了任何指定数据分布。

这并不否定冻结 model494 的已测效果。它仍然定义一个有效的有限随机算法，只是其输出分布要从这个算法本身定义和测量。
即使改成先把 \(x_D\) 加噪到 t800，得到的也是“DLM 提议经正向核后的分布”，未必等于 MP20 的 \(r_{800}\)；
“补一次加噪”也不会自动消除所有分布失配。

每一整数时刻实际包含坐标 corrector、联合 predictor。corrector 的中间坐标不立即 wrap；
predictor 输出再 wrap；\(t=1\) 所有随机项强制为零，整个末步成为共享网络参数相关的确定性映射。
轨迹输出只保存整数状态，没有 corrector 中间值和 pre-wrap 随机动作。

PC corrector 是额外的 score-MCMC 更新，不能把两个子步随意合并成式 (7) 的一个 Euler 步。
尤其本实现 corrector 步长
\(\eta_t=10^{-5}(\sigma_t/0.005)^2\) 没有随积分网格间隔缩放。
若增加噪声时刻数却保持该规则，则总 corrector 时间随时刻数增加；
这不是对同一个有限时域 SDE 的自动网格细化。
固定步长 Langevin 更新也不精确保持目标边缘，需要单独控制离散误差。

CSPNet 的 SinusoidsEmbedding 返回 emb.detach()，sample 还有 no_grad。
单步参数 score 回归或固定已采集状态下的似然梯度仍能训练参数；
直接删除 no_grad 后做全链 pathwise 反传会漏掉经过坐标 embedding 的状态导数，不能声称得到完整链梯度。

## 4. 终态双目标在分布层面如何成立

固定 \(b=(c,P,x_D)\)、公共 \(R\)、能量标尺和验证协议，令连续输出为 \(Y\)：

\[
A_F(Y)=e(Y)-e(R(Y)),\qquad B_F(Y)=e(R(Y))-h(c).
\tag{8}
\]

先对组成等权，再对该组成的预注册提议取固定分布 \(\nu(db)\)。
原生 DLM 的 \(A_D,B_D\) 仍单独在 \(x_D\) 上计算。
以下不把 A 等同于优化器步数，也不假设实际有限步 \(R\) 必使 \(A\ge0\)。

对参考路径测度 \(P_{\mathrm{ref}}(d\omega\mid b)\)，终点 \(Y=G(\omega,b)\)，
分开约束两项平均收益的 KL 投影，在对偶乘子存在且归一化有限时满足：

\[
\frac{dQ^*}{dP_{\mathrm{ref}}}(\omega\mid b)
=\frac{W_b(Y)}{Z_b},\quad
W_b(y)=\exp[-\lambda_A\widetilde A_F(y)-\lambda_B\widetilde B_F(y)].
\tag{9}
\]

\(\lambda_A,\lambda_B\ge0\) 来自两个约束，而不是手工选定固定 A+B；
各 \(b\) 的归一化保证条件分布仍归一，且不改变外层 \(\nu\)。
R 可以不可微，但须是固定、可测、协议明确的标签生成器，且相关指数矩、成本矩存在。
有限 verified 池没有无限尾部，并不证明总体指数矩也有限。

当前 K4/K8 教师只给出 verified 候选索引的 \(w^*\) 与 \(\mathrm{KL}(w^*\|u)\le0.2\)；
离散经验点质量对连续模型通常没有有限的测度 KL。
不能将这个 0.2 写成学生、未筛选总体或连续路径的信赖域保证。
验证覆盖率、空组和失败请求必须保持可见。

## 5. 真正的连续路径控制还需要什么

### 5.1 Girsanov 比较的是相对路径测度

连续路径空间没有供“整条轨迹普通密度”使用的有限维 Lebesgue 测度。
设参考过程为
\(dZ_s=b_0(s,Z_s)ds+\Sigma_s dW_s\)，
训练只改漂移为 \(b_\phi=b_0+\Sigma_s v_\phi\)，初始分布相同。
在解存在、不爆炸、随机指数为真鞅且控制能量可积等条件下：

\[
\frac{dP_\phi}{dP_0}
=\exp\left(\int_0^T v_\phi^\top dW_s^{0}
-\tfrac12\int_0^T\|v_\phi\|^2ds\right),
\quad
\mathrm{KL}(P_\phi\|P_0)
=\tfrac12\mathbb E_{P_\phi}\int_0^T\|v_\phi\|^2ds.
\tag{10}
\]

显示的等式采用可识别的噪声坐标；对非退化 \(\Sigma\)，
\(v_\phi=\Sigma^{-1}(b_\phi-b_0)\)。
若有退化噪声，漂移改变必须位于噪声可作用的方向，伪逆写法还需相应的测度条件。
零噪声方向不能无条件改成不同的确定性动作，再套用有限 KL。
Novikov 型指数可积条件是常用充分条件；“神经网络输出有限”并不自动核验全过程条件。
严谨的熵控制论述可参照
[Tang，§3.2–3.3](https://arxiv.org/html/2403.06279v2#S3.SS2)。

固定非零噪声的 Euler 链，单步漂移改变量为 \(O(\Delta s)\)，方差为 \(O(\Delta s)\)，
故 KL 累加趋向式 (10)。改变扩散协方差则不同：不同二次变差可使连续路径互相奇异。
CPU 例子中，相同 \(\sigma=0.6\)、漂移改变 0.4、时域 1，16 至 1024 步的总 KL 都是 0.222222；
只把 \(\sigma\) 改为 0.7，KL 随步数由 0.422478 增至 27.038593。
这个反例否定“有限步 Gaussian 都可评分，所以连续极限任意可训”，不否定固定噪声的漂移控制。

### 5.2 最优漂移依赖未来盆地概率

在固定 \(b\)、固定起点 \(Z_0=z_D\)、正则参考扩散下，定义：

\[
H(s,z,b)=\mathbb E_{P_0}[W_b(Y)\mid Z_s=z,b],\qquad
\mathcal L_0=b_0\cdot\nabla+\tfrac12 a:\nabla^2.
\tag{11}
\]

它满足后向 Kolmogorov 方程
\[
\partial_sH+\mathcal L_0H=0,\qquad H(T,z,b)=W_b(z).
\tag{12}
\]

若 \(H>0\)、具有所需正则性且得到的控制可容许，Doob 变换的漂移为：
\[
b^*=b_0+a\nabla\log H.
\tag{13}
\]

令 \(V=-\log H\)，等价 HJB 为
\[
\partial_sV+\mathcal L_0V-\tfrac12\nabla V^\top a\nabla V=0,\qquad
V(T,z)=\lambda_A\widetilde A_F(z)+\lambda_B\widetilde B_F(z).
\tag{14}
\]

这里的难处是学习/求解高维 \(H\)，并检验网络与采样器能否表达它。
知道 A/B、R 或 \(w^*\)，并不等于已得到 \(\nabla\log H\)。
连续控制基础也不要求参考漂移必须是某个真实数据过程的精确 reverse score：
只要它定义了合格的参考扩散，熵控制问题便有自己的含义；
但现有 PC 链还没有被证明对应这一固定 SDE。

### 5.3 固定随机初始分布时，不能漏掉边界条件

若初始分布为随机 \(\rho\)，式 (9) 的全局路径倾斜会使它变为：
\[
\rho^*(dz)=\rho(dz)\frac{H(0,z)}{Z}.
\tag{15}
\]

若必须保留原初始分布 \(\rho\)，同一终态成本下的条件最优控制改为：
\[
\frac{dQ_{\mathrm{fixed}}}{dP_0}
=\frac{W(Y)}{H(0,Z_0)}.
\tag{16}
\]

其终点分布一般不再是 \(P_{0,T}(dy)W(y)/Z\)。
只有固定 Dirac 起点，或 \(H(0,z)\) 在 \(\rho\)-几乎处处为常数等特殊情况，两者一致。
要在随机先验不变时指定另一终点分布，需要另解相应的两端边缘约束问题，不能把式 (9) 原样套上。
[Uehara 等的连续熵控制论文，§5–6](https://arxiv.org/html/2402.15194v1#S6)明确把最优初始分布也作为学习对象。

CPU 的两起点例子：两种初始状态各占一半，优质终点概率分别 0.9、0.1，优质/普通权重 2、1。
全局倾斜得到优质率 \(2/3\)，但初始比例变为 \(0.633333/0.366667\)；
固定初始比例时，条件最优优质率只有 0.564593。
现行固定同一 \(x_D\) 的 refiner 起点是 Dirac，因此本例不是对它的否决；
它是对“换成标准随机先验也完全同理”的反例。

### 5.4 R 不可微为何既不是否决，也不是小问题

若确定性 R 在吸引盆地内部到达同一个极小值，则 B 在盆地内为常数。
直接 \(\nabla B\) 几乎处处为零，边界处可能不连续；局部降力/降 A 没有自动降低 B 的机制。
实际有限步 R 可能只近似该情形，需由终态数据确认。

然而，对剩余噪声为 \(k=\sigma\sqrt{T-s}\) 的一维 Brownian 参考过程，
取终态正半轴权重 2、负半轴权重 1，虽终态成本是阶跃函数，仍有：
\[
H(s,z)=1+\Phi(z/k),\quad
\nabla\log H(s,z)=\frac{\varphi(z/k)}{k[1+\Phi(z/k)]}.
\tag{17}
\]

在终点之前它是平滑且非零的。控制来自跨边界的未来概率，而不是终态阶跃函数的普通导数。
这给出不可微盆地反馈可进入 SDE 的明确例子；高维晶体的稀有盆地、长轨迹和终点附近陡峭的 H 仍会使估计困难。
不能把式 (17) 当作已有晶体模型可低成本求解 H 的证据。

## 6. 从真实训练出发，最直接的结合是改变干净终点分布

设 \(r_0(y\mid b)\) 为明确选择的参考终点分布，
\[
r_0^*(dy\mid b)=\frac{W_b(y)}{Z_b}r_0(dy\mid b),\qquad
r_t^*(z\mid b)=\int q_t(z\mid y)r_0^*(dy\mid b).
\tag{18}
\]

Bayes 恒等式直接给出：
\[
r_t^*(z\mid b)=\frac{r_t(z\mid b)}{Z_b}
\underbrace{\mathbb E_{r_0q_t}[W_b(Y)\mid Z_t=z,b]}_{H_{\mathrm{noise}}(t,z,b)},
\quad
\nabla\log r_t^*
=\nabla\log r_t+\nabla\log H_{\mathrm{noise}}.
\tag{19}
\]

相应加权 denoising loss 是：
\[
\mathbb E_{b\sim\nu,\,Y\sim r_0,\,t,\,Z_t\sim q_t(\cdot\mid Y)}
\left[\frac{W_b(Y)}{Z_b}\,
\ell_{\mathrm{denoise}}(\theta;Y,Z_t,t,b)\right].
\tag{20}
\]

无限数据、充分表达能力和可积条件下，它的最优解就是 \(r_t^*\) 的 score。
这个结论由平方损失条件期望直接导出，不用将去噪 MSE 称为精确的实际路径似然，也不穿过 R 反传。
有限候选实现可写为
\(\sum_jw_{bj}^*\mathbb E_{q_t(\cdot\mid y_{bj})}\ell_{\mathrm{denoise}}\)；
它定义了候选经验分布的加噪训练目标，不能据此认证总体收益。
有限候选的干净目标仍是点质量混合；正噪声时 score 平滑，不表示 \(t=0\) 也有平滑密度或有界漂移。
若以正的小噪声截断采样，评价的是截断后的实际输出，不能把原候选标签原样当作其新能量标签。

三个必须保留的区别：

1. \(H_{\mathrm{noise}}\) 是“将干净参考终点正向加噪后，对原终点的后验权重”；
   式 (11) 的 H 是“从实际生成中间状态继续采样的未来权重”。
   只有参考路径确为该正向过程的匹配时间反演，且初始边缘和条件一致时，二者才可对应。
2. 若 \(r_0\) 选为现行 refiner 的输出分布，其重新加噪得到的 \(r_t\) 不等于现有 MP20 模型原来拟合的边缘。
   式 (19) 的第一项不能自动替换成 model494 的原 score；后训练还需要学习这个参考分布的变化。
3. 正确拟合 \(r_t^*\) 后，仍须使用相应的 reverse process 和初始 \(r_T^*\) 或有界误差的近似先验。
   保留未加噪 \(x_D\) 从 t800 起跑，没有式 (18) 的终点保证。
   单坐标 wrapped normal 在有限 \(\sigma_T=0.5\) 时也不是严格均匀；它的第一 Fourier 模幅度约 0.00719，
   高维联合先验误差不能仅凭这个单坐标数字省略。

因此，加权 DSM 是**训练统计目标最清楚**的一支，不是对旧 refiner 部署分布的免费 KL 改善。
任意正的 anchor/MP20 混合也会改变目标，必须记为额外取舍，不能同时宣称仍精确恢复式 (18)。

### 6.1 真正把 score 训练误差连接到生成分布的条件

为避开可能奇异的干净终点，先取 \(t\in[\varepsilon,T]\)、\(\varepsilon>0\)。
令理想模型使用 \(s_t^*=\nabla\log r_t^*\)，学生使用 \(s_{\phi,t}\)；
两者严格采用式 (7) 的同一噪声和正向漂移，理想起点是 \(r_T^*\)，学生起点是 \(\rho\)。
在相应 Girsanov 条件下，目标到学生方向的路径 KL 及终点数据处理给出：

\[
\begin{aligned}
\mathrm{KL}(r_\varepsilon^*\|r_{\phi,\varepsilon})
&\le \mathrm{KL}(r_T^*\|\rho)\\
&\quad+\frac12\int_\varepsilon^T
\mathbb E_{r_t^*}
\left[\|\Sigma_t^\top(s_{\phi,t}-s_t^*)\|^2\right]dt.
\end{aligned}
\tag{20a}
\]

这是训练与生成真正有关的一个有条件连接：右边的误差在目标带噪分布下度量。
它与当前 verified 索引教师的 \(\mathrm{KL}(w\|u)\) 是两个对象、两个方向，不能共用数值 0.2。
还有初始边缘误差；用实际 PC 算法时还缺它与式 (7) 的采样误差论证。

将两项 denoising 平方误差暂写成未归约平方和，其超出 Bayes 最优值的 excess risk 满足：

\[
\mathcal E_L(t)=(1-\bar\alpha_t)\,
\mathbb E_{r_t^*}\|s_{\phi,L}-s_L^*\|^2,\qquad
\mathcal E_U(t)=n_t^{-1}\,
\mathbb E_{r_t^*}\|s_{\phi,U}-s_U^*\|^2.
\tag{20b}
\]

所以式 (20a) 的 score 积分对应
\[
\frac12\int_\varepsilon^T
\left[\frac{\beta(t)}{1-\bar\alpha_t}\mathcal E_L(t)
+\frac{d\sigma^2(t)}{dt}\,n_t\,\mathcal E_U(t)\right]dt.
\tag{20c}
\]

实现中的晶格/坐标维度归约常数还须显式保留。
原始均匀 t、两项同权 MSE 并不是这个熵权重积分，且原始 loss 含不可约的去噪方差；
训练 loss 下降不能直接读成式 (20a) 右端下降多少。
即便得到小的总体 KL，对无界 A/B 的期望改善仍需尾部控制；对 SUN 等有界事件可用分布距离界，
但当前没有测得这里的总体 score excess risk。

## 7. 选择什么终点数据，决定了故事能讲到哪里

| 监督与条件 | 理想目标能够表达什么 | 主要限制 |
|---|---|---|
| 固定 \(b=(c,P,x_D)\)，唯一终点 \(R(x_D)\) | 摊销该提议的几何弛豫，理想 \(A_F\approx0\) | R 近幂等时 \(B_F=B_D\)，没有该提议下更低 B 的标签 |
| 条件只含 \(c,P\)，混合多个 \(R(x_D)\) 并重加权 | 重新分配组成内盆地概率，B 可以下降 | 更像条件连续生成器，原始提议作为逐样本输入的角色可能变弱 |
| 条件含同一个 \(x_D\)，收集多个实际连续输出 \(y_j\) 及 \(R(y_j)\) | 学习提议条件下、更优的连续输出分布，可同时监督 A/B | 要新采样、新标签、新匹配采样入口；原 DLM K8 不能替代 |
| 条件含同一个 \(x_D\)，保存完整 PC 路径并重加权 | 对原算法附近的随机转移作拟合 | 需补轨迹记录、正确周期密度、冻结确定性末步，且教师可能超出转移族 |

只训练 \(R(x_D)\) 不会降低该 x_D 已选定盆地的 B，但不能扩大为“所有终点蒸馏都不能降 B”。
按组成混合多个已弛豫终点并改变其概率，可以改变 B。
同样，DSM 学习的是分布 score，不是把不同晶体的坐标逐元素平均。
跨提议训练仍须保证物种、原子数、周期表示以及条件中原子身份的一致性；
不能给不相干的提议与目标任意配对，再把它解释为有物理依据的修复算子。

## 8. 实际 PC 路径训练：保留为另一支，不能预先指定为主线

固定方差 Gaussian 子步骤的完整动作似然确实可算。
对采集的反向动作，加权 NLL 是按协方差缩放的回归；
周期输出须用 wrapped density，或保存 pre-wrap 动作并在覆盖空间评分。
corrector 与 predictor 均须记录；HT 子步骤抽样只减少估计成本，不免除完整目标定义。

若最后 \(Y=g_\phi(Z)\) 为参数相关的确定性映射，不同 g 的条件 Dirac 可不绝对连续。
一种有清楚测度含义的有限步方案是：t>=2 训练适配器，整个 t=1 corrector 与 predictor 切回冻结模型；
最后评价是共同固定 pushforward。只在 loss 中忽略 t=1、执行时仍更新共享网络，不满足该条件。
这能合法定义有限步目标，仍不能认证它是某个连续 SDE 的一致离散化。

更根本的是，有限步最优 Doob 核
\[
k_s^*(z'\mid z)=k_s^0(z'\mid z)\frac{H(s+1,z')}{H(s,z)}
\tag{21}
\]
通常不是固定方差 Gaussian。
即使完整教师相对参考 KL 很小，也可能无法由只调 Gaussian 均值的学生实现：
CPU 例子中参考 \(N(0,1)\)、教师
\(\tfrac12N(-0.6,0.4^2)+\tfrac12N(0.6,0.4^2)\)，
教师 KL=0.149513，成本 \((x^2-0.6^2)^2\) 期望由 2.4096 降至 0.3072；
但最小化 \(\mathrm{KL}(Q^*\|N(\mu,1))\) 的最优 \(\mu=0\)，学生完全没有收益。

该反例只否定有限步转移族的无条件可实现性。
若 H 足够平滑、步长趋于零、噪声按 \(\sqrt{\Delta s}\) 缩放，
倾斜核的均值改变在首阶为 \(a\nabla\log H\,\Delta s\)，
方差仍为 \(a\Delta s+O(\Delta s^2)\)，所以不反驳式 (13) 的连续控制结果。
现有大噪声 PC 子步骤、固定 corrector 规则和有限 LoRA 容量没有自动满足这些极限条件。

任意路径拟合改变的向量场也未必继续是同一个正向加噪家族的精确 score；
它可以定义新的随机生成算法，但“保留了原 denoising 数学解释”需要另外证明。
[DDPO](https://arxiv.org/html/2305.13301v3)支持按实际随机转移求似然梯度的接口；
[Diffusion-DPO](https://arxiv.org/html/2311.12908v1#S4)使用 ELBO、界与 forward posterior 近似。
这些工作没有替当前周期 PC 链、确定性末步或 Gaussian 表达能力提供现成保证。

## 9. 与本项目故事真正契合的部分

可以统一的命题是：**在冻结科学条件、固定组成和公共终态评价下，学习改变结构分布的方式；离散完整 token 路径与连续 score/漂移使用各自正确的训练对象。**
它不要求把两者写成相同的 DPO 或 token likelihood，也不要求推理时调用 MLIP。

C3FD、typed Llama Plan/program、物种顺序继续定义化学与可执行构造条件。
DLM 的 7+4N 离散几何、周期旧状态 conditioner、联合 cell/半胞 XYZ 事务与反向闭合，是本轮已有方法的主体。
连续网络当前只看到原子类型及动态几何；它没有独立 Plan/program/x_D 通道。
新增条件通道可以让提议在全程可见，但其科学价值须由匹配消融证明，不能仅凭信息进入网络就宣称程序指导有效。

“DLM 只跨盆地、连续扩散只修盆地内误差”不是现有代码保证。
更可检验的问题是：同一 DLM 提议经过连续过程后，A 改善来自几何残差减少，还是 B 也因盆地概率变化而改善；
这种变化是否仍保留有效性、新颖性、唯一性与程序条件。

若最终分数主要由 refiner 提升，就报告完整系统提升；原生 DLM 能力仍按原生端点评估。
目前 DLM 教师在 \(x_D\) 打标签，连续教师未来在 \(y\) 打标签，两者不是已经存在的同一个联合教师。
先冻结最终 DLM 可保持输入分布及归因明确，不能把独立模块各自优化直接写成联合全局最优。

## 10. 下一轮最小可证方案及停止条件

这里“可证”只指统计目标与代码对象能对应，效果仍需独立样本验证。

1. **先检验标签支持。**固定 train-only 的 \(c,P,x_D\)，对同一提议收集多个连续随机输出及公共 R 标签。
   检验 B 的不同值/盆地是否存在、A/B 共同收益是否大于零，报告 verified 覆盖率与异常能量。
   若所有候选同盆地同 B，则这批数据没有双目标连续后训练依据；去噪状态数再多也不能补出盆地对比。
2. **先确定科学问题再定训练法。**要改变条件终点分布，选择式 (20) 的加权 DSM，并注册与之匹配的起点及反向采样；
   要保持旧 t800 算法，选择完整实际 PC 路径记录与固定末步方案，只作有限步主张。
   要研究完整 SDE 控制，则必须另外定义连续时钟、噪声矩阵、末端边界、H 的估计和 admissibility；它不是上述代码修改的别名。
3. **先做数学接口复核。**DSM 检验噪声目标符号、权重归约、周期不变性以及匹配起点；
   PC 检验相同参考模型的动作重放密度、全部随机子步、pre-wrap 动作和冻结末步。
   不在同一实验中同时改输入 DLM、MLIP、参考池和连续采样协议。
4. **再比较学生。**固定输入分布和资源口径，用独立候选、配对噪声评价冻结与训练后模型。
   原生输出、连续输出、公共 R 后结果分别报告 A/B、稳定率与完整 SUN，不用训练池 teacher gain 代替学生结果。

小 K 主要限制的是盆地发现，不是连续动作打分能否完成。
假设某优质盆地在参考过程中的概率仅 0.001，K8 至少见一次的概率约 0.007972；
这个数字是数学例子，不是晶体实测。
即使有真正总体 \(\mathrm{KL}\le0.2\) 约束，二元事件的数据处理界也只容许把该盆地概率提高到约 0.06277，
并非随意集中到优质盆地；当前 verified 索引 KL 更不能据此给部署保证。

A/B 平均值同时下降也不蕴含 stable 或 SUN 上升，23 与 CPU 脚本已给出阈值反例。
同一 MLIP 同时监督两段还会传播其系统误差；统计与数值一致性不能代替物理真实性。
尚未解决的问题是：足够的跨盆地标签、score/控制表达能力、有限样本误差、采样失配与真实评价收益。

## 11. CPU 核验与证据边界

[continuous_math_checks.py](continuous_math_checks.py) 与 [JSON 结果](continuous_math_checks.json)保留原有检查，并补充：

- 加权终点加噪后的 score 与 \(\nabla\log H_{\mathrm{noise}}\) 恒等式；
- 不可微终态阶跃成本仍产生平滑未来价值梯度；
- 全局终态倾斜改变随机初始分布的反例；
- 不缩放 corrector 步长时，总 corrector 时间随网格时刻数增长。

此前已有周期密度、联合 KL 链式法则、固定终点映射收缩、确定性末步极限、
固定/改变噪声的 SDE KL 极限、有限 Gaussian 投影丢失收益和小 K 稀有盆地例子。
本次用项目 Python 环境执行，退出码 0，全部断言通过；
加权 score 恒等式残差约 \(1.52\times10^{-9}\)，未来 H 梯度残差约 \(1.50\times10^{-10}\)。
同一 corrector 规则在 1000 与 4000 个噪声时刻下的步长总和分别为 10.8955 与 43.4643，
说明增加时刻数时没有保持相同的 corrector 时间预算。
这些检查仅用于核对公式与反例。它们没有证明神经网络已学得 score/H、当前 PC 收敛到某个 SDE，或任何晶体性能改善。
