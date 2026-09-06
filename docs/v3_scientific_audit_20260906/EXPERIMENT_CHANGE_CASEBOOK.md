# 全链路实验、变更与成功/失败归因台账

2026-09-06，主审整合。状态：历史来源已分族整理，V2完整评测、construction与冻结模型诊断已完成；H-P33具体规格进入第二轮审查，尚未实施或训练。本台账不把一次工程失败、未执行提案、教师可行性、代理指标改善和最终SUN改善混成同一种结果。

## 1. 覆盖与证据强度

近期 Git 清点覆盖自 2026-08-26 起的 **739 个 first-parent commit**，逐项变更路径已保存于 [变更库存](evidence/inventory_before_cleanup/RECENT_CHANGE_INVENTORY.json)。[失败文本索引](evidence/inventory_before_cleanup/FAILURE_DISCOVERY_INDEX.json) 的 559 条是查找线索，经过引用/日志去重后才可能对应实验；它们不是 559 个独立失败。

更早的 R5C/H1-A2/R03、C³FD 和稳定性路线按[原始证据台账的18族](historical/docs/DLM_ORIGINAL_ARTIFACT_EVIDENCE_LEDGER_36H_V1.json)逐项保留原 artifact、分母、seed、缺口和 G0–G4 分级。旧台账的“当前找不到”描述其当时工作树；本轮未重新取得原始行的条目继续标为来源受限，不借再次引用提升证据等级。

- **直接证据**：本轮真实文件/源码/逐行统计可复核，或已登记完整的历史终态 JSON。
- **历史汇总证据**：保留报告及来源，但本轮未逐样本复算；表中数字只在原 cohort/协议内成立。
- **提案或机制假设**：不能记为已训练失败，也不能记为已成功。

## 2. 基础模型与条件接口：真实收益和未能转移的收益

| 实验族 / 改动 | 观察到什么 | 可以成立的原因或机制 | 不能据此断言 |
|---|---|---|---|
| R5C fixed-slot → exact dynamic | 动态 `7+4N` 执行成为后续基础；早期效果部分仅存历史汇总 | 消除人工 padding、明确 N 和字段位置 | padding 修复本身带来稳定性，或早期重建率等于 de novo SUN |
| H1-B formula-only | 历史汇总显示可完整执行，但常见几何模板集中、SUN 不足 | 组成确定后仍需学多模态几何 | 所有 formula-only 模型必然失败；原始分母缺口不可补造 |
| H1-A2 rich Plan 与 R03 safe-axis | 存在有用的旧结果；不同 cache、process/seed 与全请求重算明显改变高点 | 条件、顺序、refiner 与抽样共同影响结果 | 把历史最好点固定为当前架构 headline，或把 replay 当独立新组成复制 |
| Rich SFT epoch2 → epoch3 | 同一历史1000中 body/Direct 提高，Strict/Meta SUN 81/489→79/477 | 拟合/执行与稳定新颖结构不是同一目标 | 更多 CE 永远无用；这次小差异也不是充分单因果证据 |
| Counterfactual rich-field grounding | 真/反事实似然 margin 提高；固定1000 SUN 89/487→86/467，四重复亦有权衡 | 改变了条件敏感性 | 条件敏感性已经转为物理效用或计划遵从必然降低 hull |
| Text count/valence Planner | 文本标签看似物理正确，实际发射/耦合约束仍大量失效 | 标签学习不能替代可执行的联合支持 | 化学知识本身无价值；问题是该实现与表达方式 |
| CCFD 内部偏好 | 内部 assignment 指标提高，外部 comp-valid 变化极小 | 内部约束准确性可改善 | 该内部指标是外部物理终点 |
| C³FD v2 → v2.1–v2.4 | 部分化学指标改善，部分版本在穷举/解析/时限处停止 | 可达性计算与实际效率都重要 | CPU 超时或未执行 GPU 是晶体学习失败 |
| C³FD v2.5 bitset witness | 原始两seed2000请求有100% benchmark comp-valid、N∩U供应改善 | 精确计数、电荷与声明见证支持被放入解码 | 支持是所有真实化学的完备集合，或任一输出组成必有低hull结构 |
| C³FD rich-field 语义修复 | metric-cell class 与 conventional SG bucket 非一一对应；原先“100%一致”测了编译器自身规则 | 修正了标签语义和不合理硬映射 | lattice/SG标签不一致必为错误；实际数据允许两种分类不同 |
| Typed Llama / local PoE | 真实预训练骨干和 typed 残差头参与动作分布 | 支持内归一化有明确数学定义 | 各expert是独立物理证据，或语言化学知识的增益已隔离 |
| Compact-V2 fresh SFT | 重叠开发 cohort refined Meta 达到约55%，原生仍低 | 接口与拟合在这组条件下可工作 | 可以宣称未见组成泛化，或借此覆盖新的 prospective 失败 |

早期18族的精确来源及限制见上面的原始台账；当前上游实现及其独立攻击见 [C³FD/Llama 审计](C3FD_LLM_UPSTREAM_AUDIT.md)。

## 3. 稳定性学习路线：负结果具体否定了什么

| 实验族 / 改动 | 结果类型与观察 | 最有依据的解释 | 仍未排除的竞争解释 |
|---|---|---|---|
| Same-Plan extreme energy pairs | 8 streams 得95/27 train/val pairs，train门为96；训练未运行 | 固定门禁/数据产率限制了该实验 | 差一对不是能量对比学习原理失效；不能记为训练阴性 |
| CTV 单token价值头 | 3072分支完整；state-centered Spearman≈.035、AUC≈.505，未进入引导生成 | 该冻结特征/头没有可泛化的动作优势；原绝对能量相关受组成基线支配 | 所有token学习或所有critic均无效 |
| SGTC 稳定正样本续训 | 1000基线/两续训 Strict 60/55/53、Meta412/421/417；未通过既定方向门 | 正样本子集和更多拟合没有给出目标收益 | 能量负样本损失是成功生成的数学必要条件 |
| D3PO 全序列偏好 | 现存历史汇总有小的refined能量/Meta信号，raw可能恶化；部分原始终态缺失 | 后refiner偏好可能容许原生几何变差，源于不匹配的目标/轨迹 | 只凭小均值确定唯一原因，或把缺失原始结果当已复算 |
| SI-LWA / self-intent | 全MP20意图标签覆盖已完成；predictability/endpoint效用当时未测 | 数据标签可构造 | 存在一份 FINAL 数据文件就代表生成方法已训练成功或失败 |
| G2 periodic relation 与 uncertainty gate | full-epoch A/B中B无稳定增益，raw A/B为7/41与6/41，tau为23/117与23/115（256） | 该新增gate未提供所需增量 | 所有图特征或周期几何输入没有价值 |
| BTRD tau200 / tau800 endpoint imitation | 两次 Direct 小幅提升，但 Strict 分别降低6/4；部分资产已在旧清理中删除 | 端点模仿目标不等于识别 raw→terminal 修复关系 | 所有分布输运或同组成配对均不成立；该失败不覆盖其他正确 coupling |
| Projected-force microstudent | 128中有效性净变化0 | 该小头/投影/预算未取得预定收益 | 任何dense几何监督或score head必失败 |
| Generated prefix＋原始teacher suffix | 固定128中早期补全好，Y/Z补全反而差；未训练该路线 | 原始坐标在已错晶格/前缀下未必是合理条件动作 | 所有self-state训练都不成立；需要目标与状态对应 |
| Canonical site alignment | 发现约85%的物种槽位次序不匹配；修正后raw Direct143，旧G2为128；再加G2降至133 | 物种和坐标必须成对置换，修复了真实接口错误 | 已解决原生稳定性或再次添加同类G2必提升 |
| SPAD BC/BP/BR/BS | raw Direct约99.22/98.05/98.83/99.80%；BS refined SUN35/234（512） | 事务执行及匹配训练有明确有效性收益 | learned pointer优于canonical，或执行完整等于低能几何 |
| Potential closure pilot | 同256 tau SUN20/125，control17/115；raw10/58，control12/55 | 有小的后处理端正信号与明显端间权衡 | 原生稳定性或新cohort效用已确立 |
| K10 局部transaction posterior、再一轮 | stream19 tau14/107；后续新stream21 tau14/123，raw结果不同且未达门 | 两次规定实验均没达到目标 | 从不同cohort的变化唯一判定“欠训”或“局部控制不足” |
| PCTP | 终态reward提案在实施前被拒绝，无GPU/checkpoint | 是设计取舍与归因/成本问题 | 是实际训练失败 |
| PMTR | 原continuous head/transport实现与设计存在，未形成正式成功终点证据 | 同一修复目标、几何表示与可微连接必须一起定义 | 可直接恢复旧head-only/detached路径或换名成为新V3 |

这些族的详细来源保存在 [历史文档镜像](historical/docs/)；最常用定位器为 `FAILED_METHODS.md`、`CTV_DLM_V1_FINAL_NO_GO.md`、`DLM_DIRECT_STABILITY_DECISION_20260902.md` 与 `teacher_feedback_unified_v1/`。原始JSON继续优先于叙述，旧状态说明已退出当前入口。

## 4. 最近的路径学习、raw-v1 与 V2

| 改动 | 已完成证据 | 解释强度 |
|---|---|---|
| State-conditioned warmup / joint transaction / 完整attempted trace | 真实输入和梯度连接、支持及回放修复，教师与轨迹似然对象更明确 | 工程/数学对象改善，不是 SUN 保证 |
| K4 双目标经验teacher | 有限verified池内可构造均值改善；独立学生有A/B权衡 | 有限候选的可行优化不自动迁移到新组成或新轨迹；池选择依赖R及验收 |
| K8增加候选 | 开发256raw11/66、tau17/123；独立1000更正后raw39/244、tau65/484 | 候选覆盖增加没有实现目标；K增加不扩大训练组成集合 |
| 原K8独立hull覆盖修复 | 同一生成/能量/1000索引，只重算缺覆盖hull，CPU39990成功 | 纠正了虚假巨大退化；属于统计修复，不是模型改善 |
| raw-v1 新原始LLaDA适配 | 完整两epoch、终点6784；256 raw3/45、tau16/113 | 新初始化、状态与目标是组合变化；未达目标 |
| V2 冻结预测条件＋exact prefix/dense CE＋log-SPD/GEM | 两epoch4524、完整source/view覆盖、0初始化delta、真实梯度；无prefix support conflict文件、0corruption fallback | 排除若干简单实现断开/过滤假说；不能把82个empty-supervision直接叫support冲突 |
| V2 SUN | 256 raw8/50、tau16/113；refined与raw-v1同计数；reference为6/55、19/121 | 组合改动未实现refined提点。raw小幅差异不是单模块归因 |
| V2 construction→repair | 同次trace构造终点SUN12/44，修订后8/50；N∩U同为251 | Strict−4、Meta+6是配对权衡，不能据此承诺多次修订改善，也不能推出完全不应修订 |
| 当前实测软条件与metadata | 全clean/组成未改；pred三soft共同匹配train6613/24558、val1946/8147；pooled29249 exactC组的M唯一 | 当前随机soft是否有可学习控制语义值得检验；不能把抽样agreement低直接叫bug |
| V2冻结模型probe40004 | 100来源/200状态：true−unknown风险差约+.000559、target−pred soft约−.03577、masked−available old约+.06288；147真实prefix向量的float64 exp均有限 | 证明实际输入使用及有限风险变化，不把OOS变体、宏平均/模块响应或target-soft oracle替代SUN因果验证 |

最新直接来源：[评测原始汇总](evidence/EVALUATION_SUMMARIES_20260906.json)、[V2最终训练/metadata](evidence/V2_FINAL_TRAIN_AND_METADATA.json)、[V2评测](evidence/V2_EVALUATION_SUMMARIES_39998.json)。

## 5. 本次新核验将哪些原因升级、哪些原因降级

**升级为真实发生的机制：** 更严格的终态验收与FIRE广义力停止并不相同。K8tau的763个not-converged中762个实际上已提前optimizer-stop，raw-v1tau的164个全部如此。单增500步上限不改变已主动停止的过程。它影响“验证不足”的解释，也可能影响旧teacher入池；但评测频率不能外推成旧训练池的排除率。

**升级为物理解释问题：** 原冻结energy-threshold SUN包含已知无效终态。K8raw的47/285中有7/12，raw-v1的3/45中有2/3。原H口径与终态几何有效G、完整verified V应分开，且需对所有基线同口径才可比较。改变这个口径的数字不是模型收益。[逐状态交叉表](evidence/TERMINAL_STATUS_SUN_CROSSTAB.json)

**降级为理论边界而非本批根因：** radius2搜索确实存在构造反例，但5788个实际可用几何中没有发现漏掉0.5Å以下近邻；V2 prefix选择性截断反例正确，但本轮没有实际conflict记录。不能因为找到漂亮反例就宣称它解释了SUN。

**已撤回的架构误诊：** 现有conditioner已经有一跳周期消息聚合进入hidden；fixed-old是合法Bayes条件观测；后期active-query没有直接attention边不等于没有global hidden几何通路。修正这些描述避免把已有功能重新包装为新机制。

## 6. 尚待回答，不能靠库存假装完成

1. 已取得V2同次construction与repair的SUN权衡；constructor→tau800尚未评测，不推测该分支结果。
2. 条件联合分布、几何监督密度/误差分布、全几何反向过程及GNN增量，各自能否用最小实验区分。
3. 旧缺失资产能否从登记远端或Git对象进一步取得；拿不到时保留证据等级，不补造结果。
4. [V3具体规格](V3_H_P33_SPECIFICATION.md)已进入第二轮独立反驳；数据身份、实际计算图验收与SUN仍须执行，严谨stationarity/ELBO不能代替实验。

最终采用与否按可比 SUN 判断。历史成功说明某些部件确实有用，历史失败说明某些实现/目标/预算未取得收益；两者都不应被扩大成无条件保证。
