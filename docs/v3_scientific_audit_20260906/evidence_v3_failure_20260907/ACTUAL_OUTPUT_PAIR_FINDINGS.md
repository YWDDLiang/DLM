# 40066 实际 float／Q／refined 输出检查

2026-09-07。来源为 [v3_cases.json](v3_cases.json) 中三端完整256请求；[CPU复算脚本](analyze_casepack_output_pairs.py)、[统计](output_pair_cpu/SUMMARY.json) 和 [逐请求数据](output_pair_cpu/per_request.jsonl) 保存可复现结果。未调用模型、能量或优化器。包中seed有字符串与整数两种编码，比较时用Python精确整数，没有经浮点转换。

**目前最强线索是坐标输运很弱，输出仍接近初始均匀坐标。** 这比“codec错了”或“晶格全面塌缩”更符合本批数据；仅凭净位移仍不能区分场幅度小与轨迹中往返抵消，需要冻结轨迹probe补证据。

## 全请求及物理失败分层

| 端点 | Strict／Meta SUN | verified Strict／Meta | invalid raw | invalid terminal | not converged | verified |
|---|---:|---:|---:|---:|---:|---:|
| G float native | 9／61 | 6／28 | 40 | 64 | 92 | 60 |
| 同生成结果Q | 6／58 | 2／26 | 38 | 64 | 99 | 54 |
| G→τ800 | 13／116 | 6／47 | 0 | 0 | 161 | 95 |

Q还包含1条`optimizer_stop_unverified`。三端都记录256个可解析输出；**可解析不等于通过共同几何支持**。native的40条invalid_raw均由本地直接复算的周期最近距离<0.5 Å对应；其余216条最近距离与原label逐条完全相等。native另外有64个无效终态，其中仍有1／4进入headline SUN；不能把headline9／61全说成物理verified。

native可评的216条原始能量中位数3.294 eV/atom，最大力中位数47.157 eV/Å，最大应力中位数63.861 GPa。τ800的全部256条对应最大力中位数0.446 eV/Å、应力3.304 GPa，说明refiner确实大幅修复了几何；它同时把NU从256降为224。τ800 Strict stable=20、SUN=13，Meta stable=143、SUN=116；即使只找回新颖性，现有Strict稳定结构总数20仍不到26。

## 坐标与晶格的实际变化

- 将存储的CPU FP64 prior按同一normalizer解码，初始prior有35条最近距离<0.5 Å；最终native为40条。最近距离中位数从0.9463 Å到0.9118 Å，没有表现出整体碰撞清除。
- 按固定站点顺序计算 `wrap(F_final−F_prior)` 每分量RMS：全256中位数 **0.0119183**，p10／p90为0.007574／0.020633，最大0.033360。这里比较的是实际记录的相同prior，不是另采噪声；它不等于全轨迹累计步长。
- 晶格没有全面失控：最终／初始prior体积比中位数0.998914、均值1.05231、范围0.484–2.102；最终V/N中位数19.099 Å³。最小晶格奇异值最低1.280 Å，condition number最高8.181。本批的严重近邻问题不能简单替换为全体晶格数值坍缩。
- N分层很突出：N=1–5共66条，SUN8／33；N=6–10共85条，1／26；N=11–15共45条，0／2；N=16–20共60条，**0／0**。这是描述性分层，不能只挑小N重报总体成绩。它提示后续场检查须保留大N，避免只用单原子或少原子验收证明多站点学习。

## Q是否改变了应有的结构

从每条native float重新调用原 `arrays_to_dynamic_tokens`、100→0 alias规范化、strict parser和`arrays_to_structure`，**256／256与包中Q一致**：fractional最大误差0，Gram最大相对误差3.58×10⁻¹⁶。没有发现Q另取了别的几何或产生错误解码。

实际量化扰动每分量最大不超过0.00499966，分量RMS中位数0.002874；Gram相对变化中位数0.007858。它会改变具体终态：Strict SUN有4条共有、5条float独有、2条Q独有；Meta有49共有、12条float独有、9条Q独有。当前净结果是9／61降到6／58，没有给出把Q改为主端的依据。

## 当前结论的边界与下一项检查

casepack没有sample原始structure、真实CIF、proposal_graphs或pre-ε／v/u，所以本次**没有验证整个导出／图链，也没有定位第33次readout**。原raw label直接读取native `structure`，图/refiner独有错误不能解释raw的40个近邻冲突和高力。

根代理可执行 [实际输出链CPU脚本](audit_actual_output_chain.py) 验证全256 sample→native→CIF→Niggli图→ProposalDataset FP32。该脚本只读40066并在外部诊断目录输出，不重跑图构建、模型、能量或优化器。机制诊断随后应优先比较u的实际范数／累计步长与上面的净位移，并结合分时段zero-u基准判断；不要仅因最终坐标净移动小就预先选择某种solver修补。
