# K4/K8：什么稳定、什么失败，以及唯一便宜候选

2026-09-07。中心是 K4、K8 与同cohort参考策略的实际256请求，不以V3为判断基准。**当前K4的τ800整体SUN较好，K8的raw较好；两者的可靠成功重叠有限，也存在相同的正常几何高能末态。** 主要可廉价干预的问题是极短接触进入原生生成；拟统一使用有界soft action惩罚，保持共同R及原hard support，不按ID筛样本。

## 1. 输入、真实终态与复核产物

读取 [old_cases.json](evidence_v3_failure_20260907/old_cases.json)（SHA `f2f07653920cc2b1304a0e13f40f53ceeaae93ee0089701e2a2dc0ce86f98a82`）与新补 [四端终态标签](evidence_v3_failure_20260907/k4k8_labels_compact.json)（SHA `1dd63a7812db564c147d5565ac6c4a05aa8bed1d0b7e20ebdb13ed26155be28a`）。后者保留原float的R后晶格/坐标、验证、optimizer和能量字段；逐trajectory与原attempt的group、状态、verified及终态能量完全对齐。

[CPU脚本](evidence_v3_failure_20260907/analyze_k4_k8_physical_outcomes.py) 约5秒完成；[数值摘要](evidence_v3_failure_20260907/k4k8_physics/K4_K8_ANALYSIS.json)、[完整病例/几何](evidence_v3_failure_20260907/k4k8_physics/K4_K8_CASES.json)、[全部分组ID表](evidence_v3_failure_20260907/k4k8_physics/K4_K8_OUTCOME_GROUPS.csv)、[逐请求结果矩阵](evidence_v3_failure_20260907/k4k8_physics/K4_K8_OUTCOME_MATRIX.csv)可复查。没有新能量计算、弛豫、SSH、GPU或生产算法修改。参考端仅使用其原结果和输入几何，没有虚构参考的R后坐标。

距离分析包含完整周期像及非零晶格自像；自适应盒覆盖认证无缺失。近邻壳层数CN*定义为每原子距离≤其最近邻距离1.1倍的周期邻居数，包含重复晶胞中的真实邻居。它是几何壳层描述，不是键阶、氧化态或普适配位数判定；不同原子的局部截距可以不同。

## 2. 先分开总SUN和可靠成功

| 策略 | raw Strict/Meta SUN | raw verified SUN | τ800 Strict/Meta SUN | τ800 verified SUN |
|---|---:|---:|---:|---:|
| Reference | 6/55 | 1/26 | 19/121 | 10/49 |
| K4 | 7/57 | 3/30 | **19/126** | **9/53** |
| K8 | **11/66** | **4/30** | 17/123 | 8/47 |

分母始终256。以下“verified S”要求共同R终态验证和能量阈值，“verified SUN”再要求原全cohort N/U；unknown不作为高能负例。

| 端点与指标 | K4/K8共同 | 仅K4 | 仅K8 |
|---|---:|---:|---:|
| raw verified Strict S / SUN | 2 / 2 | 1 / 1 | 2 / 2 |
| raw verified Meta S / SUN | 13 / 13 | 17 / 17 | 17 / 17 |
| τ800 verified Strict S / SUN | 9 / 7 | 2 / 2 | 1 / 1 |
| τ800 verified Meta S / SUN | 43 / 33 | 25 / 20 | 16 / 14 |

- raw共同verified Strict SUN：**158、250**；共同verified Meta SUN：58、96、105、108、116、122、134、158、164、175、190、214、250。
- τ800共同verified Strict SUN：**35、66、68、88、137、214、250**；共同verified Meta SUN的33条见分组ID表。
- raw仅K4 verified Strict为82，仅K8为159、255；另一方均未通过完整验证，因此没有双方verified支持的Strict纯能量翻转。τ800仅K4为46、55（另一方未验证），仅K8为190（双方verified，能量真正跨Strict阈值）。
- 双方均verified的Meta能量胜负：raw K4胜74、174、192、213，K8胜90、218；τ800 K4胜41，K8胜218。其余许多单方verified命中还混有验证状态变化，不能全部解释成能量改善。

raw的17个K4-only verified Meta SUN中，另一方为11 not_converged、2 invalid_terminal、4双方verified能量翻转；K8-only的17个对应12、3、2。τ800 K4-only20个对应18 not_converged、1生成失败、1能量翻转；K8-only14个为13 not_converged、1能量翻转。完整ID、能量与状态保留在产物中。

## 3. 共同真实高能失败、unknown和NU损失

**双方verified且已知hull、e_hull>0.1 eV/atom：** raw17条：1、60、65、68、71、75、85、94、100、112、128、138、144、180、221、233、241；τ80011条：1、39、49、74、81、116、139、165、169、180、201。这仅界定已经观察到的末态失败，不能推出该组成的其他结构不存在稳定相。

两端各有8个official hull unknown请求；K4/K8重建失败的并集均为5，二者没有混计为13个“unknown化学体系”。raw任一方invalid_terminal并集116，τ800为0；τ800另有K4一条invalid_raw。两边均能量稳定、但至少一边N/U失败的请求，raw Strict/Meta为1/1，τ800为7/36。这些是多样性问题，不是高能结构。

K4 τ800的153条not_converged全部已optimizer-stop，145条是力达标/应力超限；K8对应170/160。不能把它们全部解释成耗尽最大步数，或通过改共同R来冒充策略提升。

## 4. 原子排布说明了两类不同失败

下面的距离和CN*均来自真实**R后**结构，能量是原协议的e_hull。

| 病例/组成 | K4与K8的实际排列 | 物理解释与边界 |
|---|---|---|
| **158 / N2Br2，raw共同verified Strict** | 两边均有N–N最近对约1.115 Å，N各有1个N近邻，Br各有1个Br近邻；e_hull −0.148/−0.134。K4经τ后变成最近N–Br对约1.728 Å，e_hull +1.683，仍verified。 | 正常短键和极端重叠不同；后处理改变了局部连接类型，不能用“距离更长/更光滑”替代稳定性。编号是158，不是157。 |
| **88 / AlAgLu2，τ共同verified Strict** | Ag/Al各有8个Lu近邻，Lu壳层由Al/Ag组成；两边最近距3.063 Å、V/N约22.342 Å³，e_hull均约−5.28 meV/atom。 | 这是重复出现的有序环境；raw能量也为负，但未通过完整验证，不能把raw一起称verified成功。 |
| **250 / Ba6Pt2，共同可靠成功** | raw R后Pt有6–7个Ba/Pt近邻；τ后K4为Pt–Ba六近邻，K8九近邻，两者均verified Strict。原输入最短距/半径和仅0.302/0.349，R后升至约0.887/0.896。 | 严重不佳的初态仍可能被共同R/精修救回，是反对硬接触拒绝的直接案例；不能保证新采样偏好逐例保留它。 |
| **74 / CaCl2，raw K4真实胜出** | K4 Ca有6个Cl近邻，K8为4个；最近Ca–Cl约2.752/2.628 Å，V/N 30.71/32.45，e_hull .024/.153；双方verified。 | 不同局部环境与能量差同时存在，不是简单总体缩胞。τ后二者都到CN*约6的另一结构且e_hull≈.102，说明单个CN*也不能决定能量。 |
| **218 / KS2La，K8两端真实胜出** | raw K8 K/La各6个S近邻，S近壳层主要为3个La，e_hull .0307；K4 K/La为8近邻，e_hull .1565。τ后K8保留较低能环境，K4的S壳层含K/La而e_hull .1785。 | 相同组成可进入不同配位环境；此例并不支持“配位数越大越稳定”。 |
| **41 / In3Pr3Tl3，τ K4真实胜出** | raw两边发生In–Tl约.194/.424 Å的重叠，均invalid_terminal。τ后均恢复正常距离，K4 Pr壳层9/9/11，K8为7/5/5；e_hull .0244/.1020且双方verified。 | 排除重叠是必要的几何修复方向，但之后的能量差仍涉及整体环境；不能从消除重叠直接保证跨阈值。 |
| **156 / NiGd3，伪“始终稳定”** | raw终态Gd–Gd约.359/.361 Å，部分Gd只有一个极近邻，均invalid_terminal，尽管headline判为Strict；τ后距离约3.27–3.29 Å、壳层12，但不达Meta。 | 这类低能数字不能当可靠成功目标或恢复对象。 |
| **1 / PrPu，共同真实高能失败** | raw两边近壳层10（异种8＋同种2）；τ后壳层12，最近距约3.54 Å、V/N约32.19，双方verified，e_hull仍约.269。 | 几何正常、配位完整仍可高于竞争相；本候选短接触惩罚基本不作用于它。不能推断不存在稳定Pr–Pu结构。 |
| **245 / In6Yb2** | raw的过密输入经R恢复到最近距约2.95 Å；τ后趋近12近邻环境，均verified。 | hull仍unknown，不能列入持续高能失败。 |

用半径和归一化的最短接触ρ统计：K4/K8 invalid_terminal真实末态的ρ中位约**0.113/0.100**；verified Meta成功约**0.902/0.936**。但verified高于Meta的正常末态约**0.900/0.883**，与成功高度重叠。短接触能识别灾难性几何，不能单独识别所有稳定结构。

## 5. 唯一可执行候选与小探针

采用 [固定soft-contact规范](K4_K8_SOFT_CONTACT_CANDIDATE.md) 和 [机器JSON](evidence_v3_failure_20260907/k4k8_physics/candidate_spec.json)：

`dsoft_ij=max(0.5Å,0.5*(r_i+r_j))`，`b(a)=min(1,max_j max(0,ln(dsoft_ij/d_ij(a))))`，`p_new ∝ p_old exp(-b)`。

λ=1、上限1 natural-log unit；仅当前cell与XY已知的Z补全，对当前其它XYZ完整站；不借old、不动XY/cell、不新增hard support或fallback。所有原phase统一适用。原schema/alias后、温度前减`T*b`，保留原P/Gumbel及新策略trace logp；无需MLIP、第二模型或额外DLM forward。

**静态支持证据及风险：** K4/K8完整raw结构触发176/164条，覆盖100/75个invalid_terminal。但触发请求中原τ800已有93/87条Meta SUN，原raw各有2条verified Strict SUN，因此禁止硬拒绝整个晶体。真实R后还有一个K4 verified Meta例82/MgSr3触发该软尺度：Sr–Sr=1.861 Å，b仅.0465；这个反例再次说明半径表不是物理禁区定理。

**准确的125像边界：** 上述完整raw静态计数，已分别按候选125像和自适应完整周期距离复算，K4/K8触发ID均0差异；其它参考/τ输入同样0差异。这不证明任意未来胞或实际prefix也相同。完整末态有全部邻站，真实Z action只看到当前完整邻站，**176/164不是实际action触发覆盖率**。

**保持原结果的条件：** 整个vector b=0时原分布逐值保持；同prefix/同Gumbel下，原选中action若b=0，其score不降而竞争者只降，故该draw保持。整轨迹需要每一步都满足，不能仅凭终态稳定或最终无短接触保证。归一化后并非所有b>0的绝对概率都下降，可靠主张仅为固定状态的惩罚期望不增。

ROOT已固定原序前4个真实prefix包供K4/K8小探针，失败请求保留。还需验证support/alias不变、概率归一化、b范围/单调性、零项精确保留、同噪声draw保持和fresh replay，并记录各状态受罚质量与实际耗时；具体保存字段见规范。该探针不使用SUN调尺度。先K4、再K8用同一公式做完整256 raw/τ评估，原结果可复用；若没有实测增益，保留原最好SUN，不按病例拼接输出。

此候选只针对进入灾难性接触的生成偏好；它不能解决PrPu类正常高能末态、未知hull或所有多样性损失。现有完整trace未纳入本物理包，不能宣称已定位全部坏接触首次出现的token；当前小probe负责确认动作层作用，避免在18:15UTC可执行目标前再扩展候选或理论。
