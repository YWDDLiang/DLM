# Roadmap Release-Consistency Audit V1

状态：`release_audit_hold_no_execution_authorization`  
审计日期：2026-08-04  
范围：只审计发布一致性；不提出新研究路线、不修改主路线图、不授权实验。

## 1. 审计快照

| 文档 | SHA-256 |
|---|---|
| `ICLR2027_TIMEBOXED_FULL_IMPROVEMENT_ROADMAP_V1.md` | `fea45bf9042cf720b7b96120169c2f2327fcd319740e898bbc0e04d916f05956` |
| `ROOT_EVIDENCE_AND_DEADLINE_FRAME_V1.md` | `02747c6f58c8d38a104cdb56de90b9d0bf94712f64d578dc642b612c3c0974b6` |
| `CROSS_REVIEW_BODY_ON_PLANNER_RL_V1.md` | `06cd3b5a3df666a2c3a17aceb12101199c29052227b41aba945926fffda6e46f` |
| `CROSS_REVIEW_PLANNER_ON_BODY_RL_V1.md` | `55d3c356ebbb8bed8f57910cbdb1207aa860cc65c9abff2747697b706abbdc9c` |
| `CROSS_REVIEW_RL_ON_PLANNER_BODY_V1.md` | `e46071b8ad784252998fe602244a5b3fe03d98ef0d1c2c46342351b6fef49a98` |
| `INDEPENDENT_RED_TEAM_REVIEW_V1.md` | `5ac3b0a7ad85d64d9da0a9908f60a9e933e1d72fd662fa9dfc30e331cf5a6364` |

注：独立红队审计的是更早的 roadmap SHA `6151a4ad…`；本文件重新绑定当前
`fea45bf…` 快照。

## 2. 发布结论

**HOLD：科学组合共识通过，但当前快照尚不能作为无歧义的执行合同发布。**

已一致：

- P0 是 primary causal anchor；
- CR-Plan 是唯一 scientific main line；
- PILS-L 仅为 cold backup，且只允许在 08-08 novelty / 08-10 engineering
  早停后一次性接棒；
- RL 本投稿 `0 GPUh / CUT`；
- chemistry compiler `CUT`；
- 08-15、08-22、08-31、09-05 四个硬日期均已保留。

以下 BLOCKER 清零并重新绑定 SHA 后，才可把文档标记为 release-ready。

严重度定义：

- `BLOCKER`：会使是否运行、是否晋级或 claim 等级出现不同合法解释；
- `MAJOR`：不改变主组合，但会造成共识、预算或表述冲突；
- `MINOR`：发布质量或 Markdown 问题。

## 3. BLOCKER

### B1. Fatal novelty gate 仍不可机械裁决

位置：路线图 422–434、659–673 行。

四臂 Plan-only 消融已经加入，但没有冻结 panel 大小、主 estimand、数值
pass/fail 阈值以及 latency/DP-state 资源上限；“作用基本等价”“机制证据不能
支持增量”仍依赖事后判断。该门直接决定 CR-Plan 是否进入标题、摘要和
end-to-end 64，因此当前不是可执行的预注册 gate。

修正要求：在不改变四臂设计的前提下，冻结样本/ordinal ledger、主 estimand、
数值裁决阈值、资源上限和不晚于 08-10 的裁决日期；禁止 endpoint 后调阈值。

### B2. PILS-L 被授权接棒，但接棒后的执行合同与预算缺失

位置：489–491、1318–1322、1384–1387 行。

文档允许 08-08/10 一次性切换，但资源表只定义 CR-Plan active 时
PILS-L `0 GPUh`，未定义切换后的 conversion/SFT/64/256/confirmation caps、
绝对日期及 owner。当前既“允许接棒”，又没有可执行的接棒合同。

修正要求：二选一并冻结：发布前补齐 PILS-L 被提升后继承的绝对日期、阶段门
和 hard cap；或明确切换只在另行冻结并授权的 PILS-L execution annex 生效后
才可发生。不得默认借用 CR-Plan 预算。

### B3. Independent evaluator 的门槛语义互相冲突

位置：732–740、752–767、1393–1402 行。

737、763 行将 independent evaluator 写入 all-metric / confirmation gate；
1399–1402 行却把它定义为可缺席的 robustness audit，缺席只降级
stability/SOTA claim。当前无法唯一判断“未提供”或“方向反转”时，方法确认、
broad-stability claim 和 SOTA claim 分别如何处置。

修正要求：明确区分 historical common evaluator 的硬门与额外 independent
evaluator 的角色，并分别冻结 `missing` 和 `reverse` 对方法确认及各类 claim
的后果；三处使用同一语义。

### B4. 最终发布物尚未绑定最终 SHA

位置：全文发布元数据；当前文件没有 input/release manifest。

当前快照综合多份评审，但正文未绑定输入版本；且本审计后的任何修正都会改变
`fea45bf…`。红队旧 SHA 不能替代最终 release binding。

修正要求：修正完成后生成外部 release manifest，列出最终 roadmap SHA、五份
输入 SHA 和最终审计 SHA；对新 SHA 做一次窄范围复核。不要在被哈希正文内写
自身 SHA。

## 4. MAJOR

### M1. 统一评分表不等于三份交叉评审共识

位置：454–469 行。

- PILS-L 当前为 `3/3/2/2/3/3`；三份评审对 Innovation 均为 2，对
  Feasibility 均为 3，Expected effect 多数为 2。
- RL 当前 Innovation=4；三份评审均为 3。
- “RL 的算法新颖性最高”因此是过度 claim。

修正要求：按交叉评审共识修正相应格和解释，或显式给出冻结的聚合规则及偏离
理由；不得把方法复杂度当新颖性证据。

### M2. P-control 的“必须执行”与 `0 GPUh / 非必需` 冲突

位置：257–264、531–556、1384–1385 行。

262 行称“必须用新 1,024 …确认”，后文却明确该 run 不在投稿必需队列、需新
授权且 hard cap 为 0。

修正要求：把 262 行限定为“若要进入 confirmed supporting table，则必须……”；
不得形成当前投稿的执行义务。

### M3. 顶层 08-10 日历吞并了 08-08 novelty deadline

位置：126–133、489–491、604–605、1321–1322 行。

顶表把 CR-Plan novelty 写在 08-10 才“有结论”，细则和三方共识则要求
novelty 08-08、engineering 最迟 08-10。顶表可被误读为延后两天。

修正要求：在顶表拆开或同格明确写成 `08-08 novelty / 08-10 engineering`。

### M4. paired-64 的 evaluator-incomplete 结果没有唯一终态

位置：679–699 行。

681–682 行只规定 meta 标 `incomplete` 且不得选方法，但 683–699 行没有说明
这对 64 promotion 是 `PASS`、`HOLD` 还是 `CUT`。这会改变 08-15 的唯一
candidate 裁决。

修正要求：明确 evaluator-incomplete 时 64 gate 的唯一状态，以及最迟在哪个
后续阶段必须补齐；不能把 incomplete 当 safety evidence。

### M5. `preliminary` 与标题/摘要权限未完全锁死

位置：130–133、752–769 行。

08-31 未完成确认可降为 preliminary，但没有在该处明确 preliminary 是否仍可
作为标题/摘要的 confirmed CR-Plan contribution。

修正要求：统一 claim ladder，明确 preliminary 只能出现在哪些论文位置，不得
与 confirmed 主张混用。

## 5. MINOR

### m1. Markdown 列表缺少空格

位置：911 行。

`-真实 behavior trace；` 应改为 `- 真实 behavior trace；`。

### m2. “proof-carrying”仍需受 novelty gate 条件约束

位置：433–434、588–590、1238 附近、1591–1593 行。

正文已正确限定为 table-relative certificate，也已规定 novelty gate 失败后
撤回对应表述；发布时仍需确保标题/摘要模板和结论同步执行该条件，避免把
oxidation-table witness 写成真实氧化态证明。

### m3. Gate 命名字符不完全统一

全文存在 `Gate-1`、`Gate −1` 等连字符/负号变体。发布前统一，以免脚本检索和
交叉引用漏项。

## 6. 数值、预算与格式核验

以下项目在当前快照中通过：

- Direct raw/conditional counts、比率和 S.U.N. 百分比/百分点差均可复算；
- transition decomposition 与 failure taxonomy 总数闭合：
  `24+4+7=35`，`1044+98+37+7=1186`；
- P-control `434/512=84.77%`、`456/512=89.06%`、
  `442/512=86.33%` 及百分点差正确；
- `8/256=3.125 pp`、`5/256=1.953 pp` 与“约 +3/+2 pp”一致；
- CR-Plan target cap `48+48=96 GPUh`，hard cap `72+64=136 GPUh`，
  无第二个可选 96 hard cap；
- 共同 evaluator 与最多 16 GPUh robustness audit 已被写入 confirmation
  hard cap；
- Markdown code fences 成对；表格结构未发现断裂；
- 1445、1449、1453、1456、1460 行末双空格为 Markdown hard break，可保留。

## 7. Release checklist

- [ ] B1：novelty gate 可机械裁决；
- [ ] B2：PILS-L 接棒合同或授权前置条件闭合；
- [ ] B3：independent evaluator 语义统一；
- [ ] B4：最终 release manifest 绑定新 SHA；
- [ ] M1–M5 已修正或显式接受并记录；
- [ ] Markdown 小问题清理；
- [ ] 对最终 roadmap SHA 复核 P0 / CR-Plan / PILS-L / RL / compiler / 日期；
- [ ] 最终审计状态改为 `release_ready` 后才授权执行。

当前结论仍是：**HOLD；本审计不构成任何实验执行授权。**
