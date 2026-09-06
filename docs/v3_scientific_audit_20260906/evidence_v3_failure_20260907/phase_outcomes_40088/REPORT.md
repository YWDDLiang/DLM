# 原 K4/K8 三阶段配对结果

所有格均为同一256请求；沿用原S/N-U/SUN与verified标记，没有重算N/U。

| 模型／终点 | construct Strict/Meta | cooperative Strict/Meta | closure Strict/Meta |
|---|---:|---:|---:|
| k4/native | 5/48（verified 2/20） | 8/48（verified 3/27） | 7/57（verified 3/30） |
| k4/tau800 | 15/126（verified 6/49） | 18/114（verified 10/44） | 19/126（verified 9/53） |
| k8/native | 5/54（verified 2/24） | 10/57（verified 2/30） | 11/66（verified 4/30） |
| k8/tau800 | 16/119（verified 9/48） | 15/122（verified 9/49） | 17/123（verified 8/47） |

共同三阶段均verified及均verified+known却原Meta标记均为False的请求，完整ID及实际e_above_hull见 COMMON_VERIFIED_IDS.json。
- k4/native：三阶段均verified 19；均verified+known 18；按原Meta标记三阶段均未达标 5。
- k4/tau800：三阶段均verified 46；均verified+known 43；按原Meta标记三阶段均未达标 7。
- k8/native：三阶段均verified 20；均verified+known 19；按原Meta标记三阶段均未达标 7。
- k8/tau800：三阶段均verified 49；均verified+known 46；按原Meta标记三阶段均未达标 9。

相邻阶段的both/lost/gained/neither与完整ID见 SUMMARY.json；逐请求变化见 adjacent_changes.csv/JSONL。
变化的互斥证据桶按生成/解析不可用、未知hull、缺终态能量、verified资格变化/不足、双方verified+known后的阈值与N-U变化依次分类；它不是单一因果归因。
not_converged、invalid_terminal及unknown不会被计作可信高能。NU但未标为稳定的计数包含未验证或不可用情况，不能改称高能失败。
同一原始几何可能在重复共同R或独立refiner/R执行中产生S或verified差异；但相同token的SUN变化并非都属于重复误差。整个cohort中其它结构变化也可能改变U代表，造成仅N-U变化。本表分别列出相同上游token时S/verified变化与仅N-U变化；tau800还包含refiner端点变化，不能单独归因于DLM阶段。原有refiner站点/晶格表示与重复性边界继续适用。
阶段几何只作辅助对照，不填补未verified的能量证据，不用于逐请求选择阶段，也不由几何变化宣布SUN改善。
