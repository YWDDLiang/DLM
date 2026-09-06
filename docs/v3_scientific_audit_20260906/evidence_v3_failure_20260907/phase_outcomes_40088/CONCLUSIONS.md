# 三阶段真实结果：简短结论

12格原始结果全部通过SHA、256请求身份、原bool计数及共同R/N-U协议核对。输出保留3072条阶段结果和2048条相邻阶段配对；没有重算N/U。完整表见 [REPORT.md](REPORT.md)，逐请求见 [adjacent_changes.csv](adjacent_changes.csv)。

统一省去后续阶段没有带来跨raw/τ800一致的收益。完整循环的τ800仍为K4 **19/126**、K8 **17/123**；停在cooperative分别为18/114、15/122。保留closure后，raw Meta两模型均增加9，但K4 raw Strict少1。所有12格均未达到同端26/128，没有独立1000补评资格。

headline与verified也不总同向：K4 τ800从cooperative到closure，headline Strict 18→19，但verified Strict 10→9；K8相应headline 15/122→17/123，verified却9/49→8/47。每个相邻阶段对有56–73条verified状态变化，双方同时verified+known的比较范围只有27–63条。比如K8 raw closure的Meta净增9，对应双方verified+known的阈值翻转却是失去5、增加2；因此不能把headline净增直接称为已验证的能量改善。证据桶表示比较资格及共同出现的变化，**不表示验证状态直接造成headline公式变化**。

N-U损失也实质存在。完整循环τ800：K4 Strict/Meta稳定计数28/160，SUN为19/126；K8稳定24/157，SUN为17/123。这些差额与未verified、未知hull是不同层次，不能统称高能失败。每格确有8条官方未覆盖/未解决的hull；另有K4的2条、K8的4条输入未重建，已单列。

三阶段均verified+known、且原Meta标记均False的ID为：

- K4 raw：1、71、112、138、144；K4 τ800：1、39、74、139、165、169、201。
- K8 raw：1、6、49、85、112、138、144；K8 τ800：1、39、74、81、139、140、165、169、201。

各阶段实际e_above_hull与共同verified成功ID均在 [COMMON_VERIFIED_IDS.json](COMMON_VERIFIED_IDS.json)。not_converged和unknown未进入上述可信失败集合。

这些是同一开发cohort的配对描述，不是模型因果增益的独立确认。相同上游token下仍观测到S/verified变化；脚本另将仅N-U变化分开，避免把cohort其它结构导致的U代表变化误归因于R。τ800还包含既有refiner站点/晶格表示与重复运行差异。阶段几何只作辅助，不替代稳定性证据；本次不据这些结果再选择逐请求阶段或提出新候选。
