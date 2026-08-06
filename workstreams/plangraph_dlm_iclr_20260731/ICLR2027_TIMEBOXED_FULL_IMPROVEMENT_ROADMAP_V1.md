# PlanGraph-DLM ICLR 2027 时间约束后续改进路线图 V1

状态：`decision_only_no_execution_authorization`

日期：2026-08-04

适用系统：

```text
Planner
  -> rich Plan / PlanGraph
  -> exact-length special-token Body-DLM
  -> frozen CrysLLMGen model_494 exact800
  -> Direct + strict/meta S.U.N.
```

本文综合当前 H1-A2、R03、Planner、特殊 token 和 DLM-RL 的冻结证据，并采用
三条路线独立提案、交叉评审、独立红队的流程，给出 ICLR 2027 截止日前的唯一
主线、备线、停止条件和投稿叙事。本文只做研究决策，不授权训练、生成、
refinement、Materials Project 查询、checkpoint promotion 或任何自动下游。

---

## 0. 执行摘要

### 0.1 研究判断

目前最重要的事实不是“每个模块都还能优化”，而是各模块的可改进空间极不
对称：

- composition/joint 的最大已证实瓶颈在 Planner formula chemistry；
- Body-DLM 已有高 completion，连续 refiner 后的 conditional
  `struct_valid` 已约 `99.7%–99.9%`，不应再把结构有效性当首要攻关目标；
- strict S.U.N. 已有竞争力，但 safe-axis 证明 strict 和 meta 可以明显反向；
- meta 的剩余空间属于完整的离散 proposal、连续 refiner basin 与稳定性分布，
  不能只靠 charge neutrality 或 strict-only reward 解决；
- 2,481 个特殊 token 中大量 held-out 未出现是明确审计信号，但 full-train
  coverage 和实际 legal probability mass 尚未证明它是当前性能瓶颈；
- principled Body-DLM RL 需要新的 joint token-position policy、完整 trace、
  exact resume、独立 evaluator 和昂贵 post-refiner label，当前不是低风险
  快修。

因此本轮不能把 Planner、token 表示和 RL 同时组合。ICLR 路径必须保持：

```text
one frozen anchor
  -> one principal treatment
  -> 32 engineering
  -> 64 mechanism
  -> 256 scientific screen
  -> independent confirmation
```

### 0.2 最终排序

本节的最终标签由三份独立提案、三份交叉评审与独立红队共同决定。所有数值
预期都是 planning prior，不是结果承诺。

| 方向 | 角色 | 当前标签 | 原因摘要 |
|---|---|---|---|
| P-control 复核 | supporting audit | `APPENDIX_ONLY / SUPPORT` | 旧 `+4.30 pp` 有信号但属 post-selected discovery；不改变 P0 primary anchor |
| CR-Plan prefix reachability | ICLR 方法候选 | `KEEP_MAIN, CONDITIONAL` | 直接命中 charge failure，单因素、最快形成机制证据；新颖性受 CrysVCD 压缩 |
| Formula-derived compiler | 投稿后独立消融 | `CUT FOR ICLR` | 与 CR-Plan 首测会混入第二个 Planner factor；当前没有独立 meta 证据 |
| PILS-L length token sharing | 预备路线 | `KEEP_BACKUP, GATE −1 ONLY FIRST` | 可能改善 length calibration/refiner basin/meta，但必须先证明 train/legal-mass 问题 |
| Mask-aware Body RL | 截止日前非关键路径 | `CUT FOR ICLR / POST-ICLR` | 方法潜力高，但 trace、reward、独立 judge、算力和时间风险过高 |
| 两个长期 RL 模型 | 不执行 | `CUT` | fidelity 不是两个产品；增加一倍选择自由度与标签成本 |
| safe-axis 继续扩展 | 不执行 | `CUT` | strict 上升但 meta 显著下降，已有 scientific stop |

### 0.3 推荐的一句话论文主张

> PlanGraph-DLM 把晶体生成拆成可验证的化学规划、exact-length 离散结构生成
> 与冻结连续精修；其中 table-relative certified formula-prefix reachability 在单次
> all-attempt 采样、无 repair/retry/rerank 的条件下减少化学无效 Plan，并在
> 冻结 Body/refiner 管线中保持结构、稳定性和多样性非劣。

这句话只有在 CR-Plan 的 64、256 和独立确认门通过后才成立。若失败，论文
退回已有 PlanGraph-DLM 系统、exact-length interface 和模块化瓶颈诊断，
不得保留选择性小样本结果来制造方法成功。

### 0.4 最重要的资源决定

- 模型创新只允许一条主线占用确认性 GPU；
- common evaluator、baseline source ledger、论文骨架和复现包与主线并行，
  它们不是可选工作；
- token 路线先只做 read-only Gate −1；未证明 legal probability mass 问题前
  不训练；
- RL 在 ICLR 截止前只保留设计文档，不执行 R0、训练或评测；
- 2026-09-05 后不再增加 principal factor。

### 0.5 三类改进目标的分期归属

用户希望继续提升 `comp_valid`、`struct_valid` 和 meta S.U.N.，但现有证据不
支持用一个新 treatment 同时解决三者：

| 目标 | 当前证据 | ICLR 截止日前归属 | 不能预先承诺 |
|---|---|---|---|
| `comp_valid` / raw joint | Planner charge failure 是明确瓶颈 | CR-Plan 主线 | meta/strict 会随之上升 |
| conditional `struct_valid` | 已约 99.7%–99.9%，接近天花板 | 非劣与 failure ownership | 还能取得大幅绝对增益 |
| meta S.U.N. | safe-axis 显示与 strict 可反向；token 根因未证实 | common evaluator + 非劣；PILS-L 仅冷备 | CR-Plan 有直接稳定性机制 |
| strict/meta 的进一步联合优化 | Body proposal/refiner basin/RL 可能相关 | 投稿后 mask-aware 单-policy RL | 今年在不完整 trace 上快速训练有效 |

因此完整研究方向是分层的：今年先用 Planner 单因素解决已经定位的
composition/joint；同时守住 structure/meta；特殊-token 表示和 RL 在证据门
通过后依次接手 meta，而不是为了“全指标都改”把三种变量同时塞进一次实验。

---

## 1. 官方截止与内部冻结线

ICLR 2027 官方时间：

- 摘要截止：2026-09-18 AOE；
- 全文截止：2026-09-25 AOE；
- 主文最多 9 页；
- 论文与 submission form 必须披露生成式 AI 的使用。

官方来源：

- https://iclr.cc/Conferences/2027/index.html
- https://iclr.cc/Conferences/2027/AuthorGuidelines
- https://iclr.cc/Conferences/2027/AIPolicyForAuthors

从 2026-08-04 起，摘要/全文分别只剩 45/52 个自然日。项目采用更早的内部
冻结线：

| 日期 | 硬交付 | 逾期处置 |
|---|---|---|
| 08-07 | 证据审计、候选定义、论文一句话主张冻结 | 不再增加新方法族 |
| 08-08 | CR-Plan 文献/claim novelty 有结论 | 不能防守相对 CrysVCD/grammar decoding 的增量则退出 |
| 08-10 | P0/CR-Plan 工程与四臂 Plan-only 机制门、token Gate −1 有结论 | 未完成项目退出对应路径 |
| 08-15 | 唯一主候选必须已有 paired-64 机制信号 | 无信号则退出 ICLR 主线 |
| 08-22 | paired-256 完成，方法、threshold、evaluator 冻结 | 不再调参或换候选 |
| 08-31 | 独立 seed、共同评测、核心消融完成 | 未完成则降为 preliminary，CR-Plan 不进题目/摘要 |
| 09-05 | 科学结果冻结 | 原则上不启动新实验 |
| 09-12 | 9 页主文、附录、复现包只修不扩 | 不再改变 claim |
| 09-18 | 提交摘要、锁定作者 | 官方 AOE 截止 |
| 09-25 | 提交全文和 supplement | 官方 AOE 截止 |

排队、网络/API 或工程延迟不能通过缩小 denominator、取消独立确认、放宽
meta/多样性门槛来补偿。

---

## 2. 冻结证据与真实瓶颈

### 2.1 系统锚点

当前可复现锚点是：

```text
P0, H1-A2 epoch-2 Planner
  -> rich seven-line Plan
  -> B0/R5-C exact-length masked DLM, D1 schedule
  -> frozen model_494, exact 800 reverse steps, batch1
  -> frozen Direct evaluator
  -> strict E_hull <= 0.0 / meta E_hull <= 0.1 S.U.N.
```

Body 的关键合同：

- exact semantic length `7+4N`；
- `N`、elements/counts 与 Plan composition 预填；
- stochastic action slots `6+3N`；
- schema/dynamic legal support、duplicate-coordinate 和 positive-volume
  约束；
- 每个成功 Body proposal 都必须通过同一 frozen refiner 后才可评测；
- raw all-attempt 是主分母；
- 无 sample ID、retry、replacement、repair、filter 或 rerank。

### 2.2 Direct 指标

R03E 四个 CUDA process repeats 每臂共有 1,024 raw attempts，但它们复用了
同一个 scientific ledger，不能称为 1,024 个独立科学样本。

| Arm | generation | composition | structure | joint |
|---|---:|---:|---:|---:|
| D1 control | 984/1024 | 848/1024 | 982/1024 | 846/1024 |
| safe-axis | 992/1024 | 852/1024 | 989/1024 | 851/1024 |
| delta | +8 | +4 | +7 | +5 |

只在 refine-success denominator 上重聚合：

| Arm | comp_valid | struct_valid | joint_valid |
|---|---:|---:|---:|
| D1 | 848/984 = 86.1789% | 982/984 = 99.7967% | 846/984 = 85.9756% |
| safe-axis | 852/992 = 85.8871% | 989/992 = 99.6976% | 851/992 = 85.7863% |

冻结的历史参考：

| System | comp_valid | struct_valid | joint_valid |
|---|---:|---:|---:|
| historical CrysLLMGen, 1,000 | 89.2% | 99.9% | 89.1% |
| H1-A2 epoch 2, 1,000 | 87.8% | 99.9% | 87.7% |

这些历史表与 raw all-attempt 表不能混为同一个主表。最终投稿必须在共同
evaluator 下重算或明确标为 reported-only。

### 2.3 S.U.N. 与 strict/meta 极化

R03G completed-snapshot lower-bound：

| Arm | strict S.U.N. | meta S.U.N. |
|---|---:|---:|
| D1 | 99/1024 = 9.67% | 523/1024 = 51.07% |
| safe-axis | 117/1024 = 11.43% | 496/1024 = 48.44% |
| delta | +1.7578 pp | -2.6367 pp |

R03H 归因：

```text
strict       +18
meta-only    -45
above-meta   +36
ineligible    -9
unknown        0
```

结论：

- strict 增加不是 broad stability improvement；
- meta 的下降来自 finite `0.1 eV/atom` crossing，而非 hull unknown；
- 后续 checkpoint gate 必须先满足 meta/validity/diversity 非劣，再比较 strict；
- strict-only reward 或只汇报 strict 是不可接受的。

### 2.4 Planner chemistry

safe-axis 248 个成功 formula 中 35 个 composition-invalid：

| failure class | count |
|---|---:|
| charge failure | 24 |
| Pauling failure | 4 |
| Planner tag plausible but Direct-invalid | 7 |

更早的 1,186 parsed Plans：

| class | count |
|---|---:|
| valid | 1,044 |
| charge invalid | 98 |
| Pauling invalid | 37 |
| oxidation state missing | 7 |

这使 composition 成为当前最明确、最可定位的性能空间。只消除 charge
failure 的理论上限不等于预期收益，因为 constrained decoding 可能转移到
罕见 composition、shortcut 或高-hull 区域。

### 2.5 P-control

同一 discovery 512：

| Planner | comp_valid | relative to P0 |
|---|---:|---:|
| P0 | 434/512 = 84.77% | — |
| P-control | 456/512 = 89.06% | +4.30 pp |
| P* | 442/512 = 86.33% | +1.56 pp |

P-control 只可视为待确认的 supporting baseline：

- 它是同一 discovery panel 上的 post-selected 信号；
- 需要 all-metal、unary、formula duplication、train overlap 和 distribution
  drift 审计；
- 若要进入 confirmed supporting table，必须用新 1,024 Plan-only ledger
  独立确认；它不是当前投稿的必需执行义务；
- 即使通过，也不是足够的新算法贡献，且不替换 P0 作为 CR-Plan primary
  causal anchor。

### 2.6 特殊 token

冻结 tokenizer 定义 2,481 个晶体 special token。held-out 9,046 rows：

| item | count |
|---|---:|
| seen special token | 1,437 |
| unseen special token | 1,044 |
| schema numeric stochastic union | 2,343 |
| numeric held-out-unseen | 1,013 |
| length-family unseen | 953 |

当前不能把它写成 train-unseen 或性能根因。必须先测：

```text
train_count(v)
validation_count(v)
test_count(v)
legal_probability_mass_B0(v | state)
real emission and endpoint attribution
```

token identity 数量大本身不等于每个位置 branching factor 大，也不等于删除
token 会改善结果。

### 2.7 RL readiness

历史 TraceRL 的 formal use 为 NO-GO，原因包括：

- 未按真实 rollout state 保存 action；
- legal support 后没有重新归一化；
- 缺少真实 temperature；
- reveal position 没有概率；
- old log-prob 不是 online behavior probability；
- 只保存最终 token，未保存影响 position 的所有 candidates；
- 不具备 exact replay/resume；
- reward 仍面向旧 composition generation，而当前 Body composition 已冻结。

fixed Plan 下 composition reward 在同 Plan rollout group 内是常量，group
baseline 后 advantage 为零。Body RL 只能优化 proposal geometry、completion、
refiner compatibility、stability 与 diversity，不能声称提升 composition。

---

## 3. 研究原则与不可跨越的边界

### 3.1 单因素

每个正式 comparison 只改变一个 principal factor：

- Planner checkpoint；
- Planner decoding support；
- Body token representation/checkpoint；
- Body decoding policy；
- RL objective；
- evaluation coverage。

以下组合不得首测：

```text
P-control + CR-Plan + compiler
CR-Plan + PILS-L
PILS-L + new schedule
token sharing + support pruning + RL
safe-axis + RL
new Planner + new Body checkpoint
```

组合候选只能在组成因素各自以独立实验通过后出现，而且本投稿时间表原则上不
为组合候选预留主线预算。

### 3.2 all-attempt

- raw ordinal 预先冻结；
- failure 原样保留；
- 不替换失败样本；
- 不以 accepted/refined survivor 作为唯一分母；
- conditional 结果只作诊断和与历史表的 secondary reaggregation；
- 每个 treatment 记录 paired discordance、失败 taxonomy 和 stage ownership。

### 3.3 非劣优先

候选的选择顺序：

1. protocol/identity 完整；
2. completion、raw structure、meta、novelty、uniqueness 非劣；
3. comp/joint 或 strict 的目标改善；
4. 独立 seed 方向稳定；
5. 共同 evaluator 与独立 audit 不反向。

任何 strict-positive/meta-negative 的重复都判 scientific stop。

### 3.4 约束与 reward 分工

硬约束负责：

- exact length；
- Plan composition 保持；
- legal token；
- positive volume；
- no duplicate coordinate；
- formula grammar 和已冻结的 reachability；
- no retry/repair/replacement/filter/rerank。

reward 只允许在合法 proposal 之间排序。不能用奖励让模型重新学习可精确
执行的 schema，也不能用后处理把失败移出分母。

### 3.5 evaluator firewall

Planner/Body 的训练或约束实现不得读取 final S.U.N. panel。RL 若未来启动：

- `E_train`：冻结 reward judge；
- `E_gate`：不同 checkpoint 或不同 MLIP family；
- `E_final`：历史口径 + 独立 MLIP/DFT audit；
- 最终 panel 只打开一次；
- reward proxy 上升但 independent evaluator 下降即 reward hacking。

---

## 4. 新颖性边界

### 4.1 已被近邻覆盖的宽泛主张

以下表述不能作为本项目独有贡献：

- “先生成 composition，再生成 structure”；
- “使用 valence/charge constraint 提高化学有效性”；
- “对 crystal language model 做 RL”；
- “用 preference optimization 改善 stability/novelty”；
- “用 masked diffusion 生成晶体”。

必须正面对照：

- CrysVCD: https://arxiv.org/abs/2507.19799
- PLaID++: https://arxiv.org/abs/2509.07150
- CRYSTAL: https://openreview.net/pdf/94d95333b625bc19463eca098ff60038d639d590.pdf
- Mask-Aware Policy Gradients for DLMs:
  https://arxiv.org/abs/2607.15200
- CrysTune: https://openreview.net/forum?id=Oe5iihLiiV

### 4.2 可防守的组合差异

当前最有希望的差异不是某一个已有组件，而是一个可验证的层级接口：

1. rich Plan，而不只是 formula label；
2. formula prefix 上的 grammar × finite-budget charge-reachability product
   automaton；
3. terminal oxidation witness，可复算、可审计；
4. exact-length masked DLM 根据 Plan 编译离散晶体；
5. frozen continuous refiner；
6. single-pass all-attempt，无 repair/retry/filter/rerank；
7. 对每一层的 failure ownership 和 before/after-refiner 因果拆分。

如果 CR-Plan 最终只剩“换一种 valence mask”，相对 CrysVCD 新颖性不足，应
退出标题和摘要。

新颖性必须由预注册的三臂机制消融证明，而不是由命名证明：

```text
grammar-only
vs terminal-only fail-closed charge gate
vs full prefix finite-budget reachability
```

至少报告 terminal 换行前的 prefix affected-rate、blocked prefix、
removed probability mass、dead-end、all-attempt yield、formula diversity 和
decoder latency。若 full prefix treatment 几乎只在 terminal 阻止换行，或不
优于 terminal-only gate，则“prefix reachability”“proof-carrying”与对应标题
全部撤回；只保留系统/诊断结论。

### 4.3 论文应该强调的科学问题

论文的核心问题应是：

> 晶体生成的不同约束应该由哪个生成阶段承担，怎样在不靠后处理和 survivor
> denominator 的条件下验证每一层真的改善了 end-to-end 结果？

它比“我们又提高了一个 validity 数字”更适合解释 PlanGraph、exact-length
DLM、refiner 和严格评测为什么必须同时存在。

---

## 5. 候选路线比较与选择逻辑

### 5.1 统一评分

评分为 1–5，分数只是强迫显式比较，不替代硬门。

| 方向 | Innovation | Expected effect | Evidence | Feasibility | Deadline fit | Paper coherence |
|---|---:|---:|---:|---:|---:|---:|
| frozen P-control supporting confirmation | 1 | 4 | 4 | 5 | 5 | 3 |
| CR-Plan | 3 | 4 | 4 | 4 | 4 | 5 |
| chemistry compiler | 2 | 2 | 2 | 4 | 4 | 3 |
| reachability-mass SFT | 3 | 3 | 2 | 3 | 2 | 4 |
| PILS-L | 2 | 2 | 2 | 3 | 3 | 3 |
| mask-aware multi-fidelity RL | 3 | 2 | 2 | 1 | 1 | 3 |

解释：

- P-control 总分看似高，但主要来自容易做和旧信号，不来自创新；
- CR-Plan 是唯一同时命中已证实瓶颈、保持单变量并能服务论文一句话的方法；
- PILS-L 和 RL 主要作用于 meta，正好是需要提升的指标，但目前根因证据与
  时间可达性弱；
- RL 有长期研究价值，但近期 mask-aware PG/CRYSTAL 已压缩宽泛算法主张；
  复杂度不能当作新颖性证据，且 deadline fit 最低；
- 硬门失败时，不能用总分挽救路线。

### 5.2 “主线、备线、长期线”不是三线并跑

定义：

- **主线：** 获得正式 64/256/independent confirmation 预算；
- **备线：** 只做不会污染主线的 read-only/CPU 前置，未经新决策不训练；
- **长期线：** 设计文档和单元测试可以保留，但不占用投稿结果承诺。

因此默认资源状态是：

```text
Main scientific GPU: CR-Plan only
Backup preflight: token Gate −1 only
Long-term readiness: RL design only; no execution in this submission cycle
Always-on: common evaluation + paper + reproducibility
```

PILS-L 不是并行竞赛者。CR-Plan 在 08-08 novelty gate、最迟 08-10 工程
gate 早停，只会让 PILS-L 成为唯一**有资格被提议**的 backup；它不会自动
切换。实际接棒还需在 08-10 前冻结单独 execution annex、预算与授权。若该
annex 不存在，或到 08-15 scientific 64 才失败，直接使用 frozen H1 fallback。

### 5.3 两层成功定义

**方法成功：**

- CR-Plan 显著减少 charge-invalid；
- raw comp/joint 改善；
- completion、structure、strict、meta、novelty、uniqueness 非劣；
- 独立 panels 方向稳定。

**全指标 Pareto 成功：**

- 在上述条件外，meta 也有正向独立证据；
- strict 不通过牺牲 meta-only mass 获得；
- common evaluator 下仍领先 H1-A2/CrysLLMGen；
- 外部 baseline 的 reported 与 recomputed 口径都准确。

不能因为方法只实现第一层就称其失败，也不能因为 strict 上升就跳过第二层的
meta 审计。

### 5.4 Claim ladder

| 证据层级 | 允许出现的位置 | 禁止 |
|---|---|---|
| design/R0/32 | internal roadmap、appendix implementation note | 结果、方法成功、SOTA |
| paired-64 mechanism screen | appendix/limitations，明确 pilot | 标题、摘要、主结论 |
| paired-256 pass、`4×256` 未完成 | 主文 exploratory analysis 或 appendix，必须标 preliminary | CR-Plan 方法题目、摘要结果、confirmed/SOTA |
| `4×256` confirmation 全门通过 | CR-Plan 可进入题目、摘要和主结论 | 未通过 endpoint 的宽泛主张 |
| all-metric + extra robustness 同向 | 可主张 broad stability，但须限于实际 evaluator | “guaranteed stable” |
| common external baseline table 也通过 | 可按具体 endpoint 写 SOTA | 跨不同 denominator 的宽泛 SOTA |

08-31 未完成确认时，论文自动回退为 frozen H1/PlanGraph-DLM 系统与诊断主线；
CR-Plan 只能以 preliminary 消融出现，不能继续占用方法题目或摘要结果槽。

---

## 6. ICLR 主线：frozen P0 → CR-Plan

### 6.1 Supporting A0：P-control 旧证据只读审计

最迟 08-07 完成：

- 512 ordinals 全量 paired discordance；
- all-metal、unary、charge、Pauling、oxidation-missing；
- formula exact/reduced-composition duplication；
- 训练集 exact/reduced formula overlap；
- element、arity、N、anion family、top-frequency drift；
- raw parse/completion；
- P0/P-control/P* checkpoint、tokenizer、prompt、seed identity。

旧 `456/512` 若主要来自 unary/all-metal shortcut，或原始 ledger/identity
不闭合，P-control 不进入 supporting result。

### 6.2 Out-of-critical-path A1：新 1,024 Plan-only confirmation

冻结：

- 新 scientific ledger；
- P0 与 frozen P-control；
- same prompts、tokenizer、chat template、ordinal roles；
- no body/refiner/S.U.N.；
- raw all-attempt；
- 不重新训练或选择 checkpoint。

P-control 可进入 supporting table 的门：

- raw composition `>= P0 +3 pp`；
- paired CI/discordance 方向支持提升；
- parse/completion 非劣；
- all-metal、unary 不膨胀；
- formula diversity 和 element coverage 非劣；
- 旧 512 与新 1,024 方向一致。

该 supporting run 不属于本次投稿的必需队列。只有 CR-Plan paired-256 已
完成、完整 `4×256` confirmation 资源已预留、且获得新的独立授权后，才可
考虑执行；否则延期到投稿后。它不得阻塞 CR-Plan，也不得占用其
64/256/confirmation 预算。无论结果如何，CR-Plan primary comparison 都固定
为 `P0 vs P0+CR-Plan`。本次主文可以如实披露旧 512 的 post-selected
supporting signal，但不能把未确认的 `+4.30 pp` 当正式 baseline 结论。

### 6.3 CR-Plan 定义

工作名称：

> Chemistry-Reachable Plan Decoding, CR-Plan

只改变 formula 行的 legal support。状态为：

```text
state =
  formula grammar state
  × tokenizer-prefix state
  × selected element set
  × partial counts and atom budget
  × reachable total-charge set
```

核心操作：

1. 冻结 formula grammar 和 `1 <= N <= 20`；
2. 使用冻结 SMACT oxidation-state table；
3. memoized DP 判断在剩余 atom budget 下是否存在 charge-neutral suffix；
4. 当前模型 logits 与 grammar/reachability mask 取交；
5. 在交集上重新归一化并只采一个 token；
6. charge-applicable terminal formula 必须保存一个可复算的
   table-relative oxidation witness；
7. 无合法 token 时该 attempt fail closed。

V1 的 witness 写入 immutable Plan record/audit sidecar，不新增模型可见的第八行，
也不改变 Body prompt；否则会把 constrained decoding 与 rich-Plan schema
同时变成两个 principal factors。它只能称为“相对冻结 oxidation table 的
可行性证书”，不能称真实氧化态证明，也不表示 Body 已使用 oxidation state
作为新条件。

在 novelty/R0 前必须冻结 applicability 语义：

- 不能静默采用“一种元素只能取一个 oxidation state”的简化；
- mixed-valence 必须由 integer allocation 明确支持，或在固定 oracle panel
  上证明没有错误排除；任何已知有效 mixed-valence class 被错误屏蔽，工程门
  fail closed；
- unary、all-metal、covalent/oxidation-table-missing formula 分成明确 strata；
- charge-applicable stratum 要求 neutral witness；
- charge-not-applicable stratum 只保存 applicability certificate，不计入
  charge-primary gain，也不得靠数量膨胀制造 comp 提升；
- oracle/applicability 分类必须与冻结 composition evaluator 的语义逐例对齐。

如果这套语义在 08-08/10 前不能闭合，CR-Plan 退出主线。不能在看到 endpoint
后再决定哪些 chemistry strata 适用约束。

第一版只加 charge reachability：

- 不同时加入 Pauling；
- 不改 lattice、space group、volume；
- 不改 prompt、weights、Body、refiner 或 evaluator；
- unary/all-metal/charge-not-applicable 不视为 oxidation witness，也不计入
  primary charge gain；
- 不 beam search；
- 不生成多个候选再选；
- 不 fallback 到 unconstrained；
- 不 retry、repair、filter、replacement 或 rerank。

必须记录：

- 每步 grammar/reachability support size；
- removed probability mass；
- mask entropy；
- blocked newline；
- dead-end state；
- DP cache hit；
- terminal oxidation witness；
- charge/Pauling/shortcut taxonomy；
- formula distribution drift。

### 6.4 R0/32 工程门

R0：

- formula tokenizer prefix 能无歧义恢复；
- reachability DP 与离线 brute-force fixture 一致；
- terminal witness 可复算；
- oracle 与冻结 composition evaluator 的 charge semantics 对齐；
- mixed-valence、unary、all-metal、covalent/table-missing 固定 fixture 的
  applicability 与 false-exclusion 审计通过；
- same legal support 下 constrained/unconstrained probability parity；
- rank、batch、resume、ordinal RNG 独立；
- empty support fail closed。

paired-32：

- raw attempts 32/arm；
- 0 tokenizer/FSM error；
- 0 silent fallback；
- 0 retry/replacement；
- charge-applicable candidate 的 terminal charge failure 为 0；
- charge-not-applicable strata 单独报告、不得计入 primary charge gain；
- parse/completion 不下降超过 1/32；
- formula 七行与 canonical identity 完整；
- 不用 32 的 S.U.N. 调 mask。

任何工程门失败先修实现，不将失败结果解释为科学信号。

在进入 end-to-end 64 前，还必须在 08-10 前用新冻结、与 discovery 独立的
`512 raw attempts/arm` Plan-only ledger 完成：

```text
original frozen P0
grammar-only
terminal-only fail-closed charge gate
full prefix reachability
```

四臂使用相同 ordinal roles，只报告 raw all-attempt Plan endpoint，不运行
Body/refiner/S.U.N.，也不根据结果修改 oxidation table。冻结主 estimands 和
pass gate：

- full-prefix 相对 terminal-only 的 raw composition-valid Plan yield
  `>= +11/512`（至少约 `+2 pp`）；
- 在 charge-applicable attempts 中，至少 5% 的 attempt 在 terminal token
  之前出现真实 support difference；
- full-prefix 的 charge-applicable terminal charge failure 为 0；
- full-prefix 的 raw parse/completion 不低于 terminal-only；
- unique-formula rate 与 element coverage 相对 terminal-only 各不下降超过
  2 pp；
- candidate shortcut-valid count 不高于 terminal-only，且 shortcut 不计入
  yield gain；
- blocked prefix、blocked newline、removed mass、dead-end 与 transition
  taxonomy 全量保存。

工程上要求 brute-force/DP parity 100%，每 attempt memoized DP states
`<=100,000`，decoder median latency `<=1.5×` original P0、p95
`<=2×`，且无 OOM/timeout/silent fallback。这里的相对 latency 在同一节点、
相同 batch/长度合同下测量。

任一主门未通过、或 08-10 前未形成冻结 terminal report，即判 empirical
novelty/engineering CUT；CR-Plan 不进入标题、摘要或 end-to-end 64。阈值在
ordinal 结果打开后不得修改。

### 6.5 paired-64 机制门

最迟 08-15，目标 08-14：

- 两臂均走 frozen B0/D1、model_494 exact800、Direct 和 treatment/control
  union 的同一 frozen S.U.N. snapshot；64 的 S.U.N. 只作预注册 safety gate，
  不作 efficacy claim。若 historical common snapshot/unknown accounting
  不完整，64 gate 的唯一状态为 `HOLD_EVALUATOR_INCOMPLETE`，不能据此调方法、
  选择或晋级 candidate；必须在 08-15 硬截止前补齐同一 frozen contract，
  否则该主线按未通过处理并回退 H1；
- raw comp 增加至少 `3/64`；
- charge-applicable candidate 的 terminal charge failure 为 0；control
  charge failure 为 0 时 reduction ratio 记为 `N/A`，不能伪造百分比；
- Planner parse/completion 不多损失超过 1；
- candidate 的 unary/all-metal/charge-not-applicable count 不高于 control；
- primary comp gain 在排除这些 shortcut strata 后仍为正；
- unique formula 至少为 control 的 95%；
- mean N 漂移不超过 0.5；
- element/arity/anion family 完整报告；64 只作预注册分布诊断，不用模糊的
  “无明显 collapse”作事后硬门；
- removed probability mass 与成功机制一致；
- strict/meta 只作冻结 safety 描述；64 上的稀疏符号不作为 efficacy、promotion
  或调 mask 的依据；
- 无新 failure class。

64 只决定是否值得扩展，不作为最终论文数字。08-15 仍无正信号即从 ICLR
主线 CUT，不临时加入 Pauling、compiler、PILS-L 或 RL 追结果。

### 6.6 paired-256 科学门

最迟 08-22，不设 late grace。主比较是：

```text
frozen P0
vs
same frozen P0 + CR-Plan
```

两臂使用 D1、B0、model_494 exact800、共同 evaluator 和冻结 paired-ordinal
ledger。

“paired”表示同一预注册 ordinal、prompt、seed role 和 uniform counter；
constraint 改变 support 后，token-level sample 自然会分叉，不能声称两臂仍有
相同 formula 或完全相同 token trajectory。统计配对单位始终是 raw ordinal。

进入独立确认的最低门：

- historical common Direct/S.U.N. evaluator、union snapshot 与 unknown
  accounting 完整；不完整则 `HOLD_EVALUATOR_INCOMPLETE`，08-22 仍未补齐即
  不进入 confirmation；
- raw comp `>= +8/256`，即至少约 +3 pp；
- raw joint `>= +5/256`，即至少约 +2 pp；
- charge-applicable candidate 的 terminal charge failure 为 0；
- completion/raw structure 不下降超过 1 pp；
- strict point delta `>=0`；
- meta point delta `>=0`，作为继续 confirmation 的 safety screen；正式
  `-2 pp` noninferiority 在 pooled `4×256` 上裁决；
- formula/structure novelty、uniqueness 各不下降超过 2 pp；
- 两个 128 blocks 的 comp/joint 方向只作诊断，不作低功效硬门；
- candidate unary/all-metal/charge-not-applicable count 不高于 control，且
  排除这些 strata 后 primary comp gain 仍通过；
- element coverage 和 unique-formula rate 各不下降超过 2 pp、mean N 漂移
  不超过 0.5、top-1 formula frequency 不增加超过 2 pp；arity/density 分布
  全量报告；
- no new failure class。

更强的 “all-metric” 门：

- meta `>= +2 pp` 或 paired interval 明确支持正向；
- strict 正向；
- comp/joint 保持上述门；
- 额外 independent robustness evaluator 已可用且不反向。

如果 comp/joint 通过而 meta 仅非劣，CR-Plan 仍可作为化学规划方法；不能宣称
它改善 broad stability。若 meta 为负，则 scientific stop。

### 6.7 独立确认

最终确认不是对同一 256 做四次 CUDA repeat，而是：

```text
4 independent scientific panels
× 256 raw attempts
× 2 paired arms
```

确认门：

- comp 至少 3/4 panels 正，mean `>= +3 pp`；
- joint 至少 3/4 正，mean `>0`；
- hierarchical paired 95% interval 是 primary；raw comp 的 interval lower
  必须 `>0`，否则方法结论降为 preliminary；
- meta 至少 3/4 非负，mean `>=0`，且 pooled noninferiority lower
  `>-2 pp`；
- strict 至少 3/4 非负，mean `>=0`；
- structure/completion/diversity 非劣；
- 无 shortcut inflation 或新 failure class；
- historical common snapshot/evaluator 完整且通过上述硬门。

3/4 sign stability 是支持性诊断，不能替代 hierarchical paired interval。
如果资源只能完成一个 independent panel，结果必须写作 preliminary，不能把
一个 panel 重复运行或 process repeat 称为确认。

额外 independent evaluator 的语义固定如下：

| 状态 | comp/joint 方法确认 | broad-stability claim | SOTA claim |
|---|---|---|---|
| historical common evaluator missing/incomplete | `HOLD`；硬日期后不确认 | 禁止 | 禁止 |
| historical common evaluator 通过，extra evaluator missing | 可按共同口径确认 | 仅写 common-evaluator 非劣，不写跨 judge 稳定 | 禁止宽泛 SOTA |
| historical common evaluator 通过，extra evaluator 同方向 | 可确认 | 可按证据等级主张 | 仍需外部 baseline common table |
| historical common evaluator 通过，extra evaluator reverse | comp/joint 化学规划结论可保留并完整披露冲突 | 禁止 broad stability/all-metric | 禁止 |

因此 extra evaluator 缺席或反向都不会反向改写 raw comp/joint 的共同口径因果
比较，但会严格降低 stability/SOTA claim；historical common evaluator 缺失
则直接阻止 scientific promotion。

08-31 是独立确认硬截止；09-05 是全项目 science/table freeze。

### 6.8 Planning prior

基于现有 failure taxonomy 的合理预期，不是 power calculation：

| Endpoint | CR-Plan planning prior |
|---|---|
| charge failures | -50% 至 -80% |
| raw comp_valid | +2 至 +5 pp |
| raw joint_valid | +1.5 至 +4 pp |
| conditional struct_valid | 基本不变 |
| strict S.U.N. | 无直接先验，以非劣为门 |
| meta S.U.N. | 0 至 +2 pp 的弱先验 |
| diversity | 有下降风险，必须硬审计 |

最大风险是 reachable suffix 的原始概率质量过低，mask 把概率推到罕见
element/count tail，形成“charge-valid 但不稳定”或 diversity collapse。

### 6.9 Chemistry compiler：本投稿 CUT

compiler 不是 CR-Plan 的组成部分。本投稿不运行 compiler treatment。其
投稿后 read-only headroom 可包括：

- formula-derived anion 与 Planner self-report mismatch；
- formula-derived charge bucket 与 self-report mismatch；
- oxidation candidate 空缺；
- mismatch 与 Body/refiner/meta outcome 的关联。

只有投稿后 mismatch headroom 足够、且预先注册独立 comparison 后才运行：

```text
fixed formula/Planner
  + self-reported metadata
vs
fixed formula/Planner
  + formula-derived metadata
```

它必须通过独立 64/256、meta 正向且其他指标非劣后，才可在未来作为方法；
本次只保留为 future work。不得与 CR-Plan 首次组合。

---

## 7. 备线：特殊 token/support

### 7.1 Gate −1 必须先于任何 token 训练

恢复并冻结：

- train 27,136；
- validation 9,047；
- test 9,046；
- R03D/E/G/H emission；
- B0 fixed-256 constrained logits/trajectory states。

对每个 special token：

```text
count by split
position/family/axis
legal support membership
legal probability mass at B0 states
real emission
completion/refiner/Direct/S.U.N. attribution
```

PILS-L 只有在以下全部条件满足时才可继续：

- length 占 numeric train-unseen + rare identity 至少 50%；
- `LA/LB/LC -> L` 后 median positive target count 至少提高 2 倍；
- B0 length rare/unseen legal mass mean `>=0.5%`，或 p95 `>=2%`；
- 数据/tokenizer/checkpoint identity 全闭合；
- sharing 范围不由 S.U.N. 反向选择。

未通过即停止 token ICLR 路线。审计本身仍可作为负结果或 appendix。

### 7.2 唯一候选 PILS-L

```text
<LA_k>, <LB_k>, <LC_k> -> <L_k>
```

axis 语义由 exact position 给出。保持：

- 0.1 Å 分辨率；
- exact `7+4N`；
- stochastic step 数；
- angle、coordinate、count、element token；
- D1、Plan、refiner、evaluator。

需要 matched control：

| Arm | representation | continuation |
|---|---|---|
| C0 | original LA/LB/LC | same one epoch |
| L1 | shared L, averaged rows | same one epoch |

不能只把 L1 与未经 continuation 的历史 B0 比较。两臂各 exactly 1,696
updates，同数据、corruption、optimizer、LR、trainable boundary，final
checkpoint 固定。

### 7.3 作用边界

PILS-L 的合理机制链：

```text
supervision sharing
  -> length NLL/calibration
  -> extreme lattice/density tail
  -> refiner displacement/basin
  -> meta S.U.N.
```

它不能直接改善 fixed-Plan composition。其 primary 应是 meta 与 refiner
burden，strict、completion、structure 和 diversity 为非劣门。

### 7.4 截止日前的默认决定

因为完整 PILS-L 需要 tokenizer remap、两臂 matched SFT、32/64/256 和独立
确认，它不能在 CR-Plan 08-15 失败后才启动并仍保持严谨。默认策略是：

- 现在完成 Gate −1；
- 未获单独资源授权前不训练；
- 只有 CR-Plan 在 08-08 novelty gate 或最迟 08-10 工程 gate 早停时，
  PILS-L 才成为唯一可提议 backup；
- 第一次 conversion/SFT 写操作前，必须另行冻结并授权
  `PILS_L_EXECUTION_ANNEX`：包含 tokenizer/checkpoint conversion parity、
  C0/L1 matched SFT、32/64/256、`4×256`、common evaluator、owner 和预算；
- annex 继承 08-15/08-22/08-31/09-05 日期，且 CR-Plan 已消耗算力与 PILS
  预算之和不得超过 `136 A800 GPUh`；不得默认借用或追加 CR-Plan 预算；
- annex 未在 08-10 前 release-ready 时，backup 失效并回退 frozen H1；
- CR-Plan 仍 active 时，不并行做 C0/L1 matched SFT 或 endpoint；
- 否则转为 post-ICLR 第一优先级。

长期更有新颖性的版本是 family/value 因子化或 ordinal head，而不是继续增加
大量独立 bin token；但这需要新表示和完整训练，不属于当前 52 天窗口。

---

## 8. 投稿后高潜路线：mask-aware Body-DLM RL

### 8.1 为什么不作为当前关键路径

RL 需要同时补齐：

- legal-support renormalized token probability；
- reveal-position probability；
- 真实 behavior trace；
- exact replay/resume；
- rare-token exploit gate；
- pre/post-refiner reward calibration；
- independent evaluator；
- 约 96 A800 GPUh 的完整路径；
- 4 个真正独立的 256 panels。

这些是算法成立条件，不是可以省略的工程细节。当前 Body/refiner 已强，
Planner 又是明确瓶颈，因而 RL 的 deadline-adjusted expected value 低于
CR-Plan。

### 8.2 推荐 policy

保持 D1 宏观 group：

```text
lattice -> all X -> all Y -> all Z
```

每一步在当前 group：

1. 对每个 masked position 从 legal masked logits 采一个 token candidate；
2. 用 sampled-token confidence 构造 K=1 Plackett–Luce position policy；
3. 采一个 reveal position；
4. 提交该位置，其他位置仍 mask；
5. 保存当前 group 全部 token candidates、legal support、token log-prob 和
   position log-prob。

扩展 action：

```text
a_t = ({Y_i for every current masked position}, selected position J)

log pi(a_t|s_t)
  = sum_i log q_token(Y_i|s_t,i)
  + log p_position(J|Y,s_t)
```

只记录 committed token 会漏掉影响 position decision 的 candidates，因此不
是完整 policy likelihood。

### 8.3 一个模型，而不是两个长期模型

推荐一个 RL-only LoRA：

```text
B0
  -> pre-refiner proxy warm-up
  -> randomized pre/post mixed fidelity
  -> post-refiner-only final
```

如果部署一定经过 model_494 exact800，pre/post 是 label fidelity，不是两个
产品。两个长期 LoRA：

- 近似翻倍标签、验证和 selection 成本；
- 增加 checkpoint 自由度；
- 混淆 fidelity 与训练阶段；
- 最终仍需选择 post-refiner 最优 policy。

只有未来存在真正不经过 refiner 的独立部署产品，才保留第二个模型。

### 8.4 多保真 reward

对每个 proposal 在看到 proxy score 前冻结：

```text
Z ~ Bernoulli(p=0.5)

R_MF = R_pre + Z/p * (R_post - R_pre)
```

这使 mixed reward 对 post-refiner target 保持无偏。禁止按 pre score 选择是否
refine。

reward vector 至少保存：

- completion/exact/Plan match；
- pre/post structure；
- refiner displacement；
- continuous `E_hull`；
- strict/meta；
- novelty；
- symmetric cluster uniqueness；
- judge identity。

composition 只记为 Planner quality，不进入同 Plan group advantage。

### 8.5 Reward gate

不能只用 strict binary。推荐 meta-aware utility：

```text
E_hull <= 0        -> 1.00
0 < E_hull <= 0.1 -> 0.50 ... 0.75 continuous
E_hull > 0.1       -> 0
unknown/failure    -> 0
```

再与 validity、novelty 和 symmetric uniqueness 结合。checkpoint selection
先过 meta/validity/diversity 非劣，再比较 strict。

### 8.6 投稿后重新注册的条件

ICLR 投稿后按以下次序：

1. Gate −1 support/coverage；
2. R0 exact likelihood/replay/resume；
3. paired-32 PL sampler safety；
4. first-64 pre/post calibration；
5. 一个 LoRA 最小训练；
6. held-out 256；
7. 4×256 independent panels；
8. independent MLIP/DFT audit。

任一以下条件成立即停止：

- support 仍变化；
- position entropy 为零；
- reward group 无方差；
- pre/post correlation 不足；
- proxy 与 independent evaluator 反向；
- rare token exploit；
- strict 正而 meta 负；
- completion/structure/diversity 下降；
- exact resume 不闭合。

当前投稿中，RL 最多作为严谨的 future direction；没有 held-out 256 和独立
confirmation 不进入摘要。

---

## 9. 共同评测与 baseline 修复

### 9.1 为什么它和方法同等重要

用户已发现外部论文指标曾被错误摘录。若 baseline 数值、分母、refiner 或
hull snapshot 不一致，即使模型真实领先，SOTA claim 也会被审稿人直接否定。

因此建立逐值 source ledger：

| field | 必须记录 |
|---|---|
| paper/model | 全称、版本、checkpoint |
| value | 原始数值 |
| source | page/table/row/column |
| sample cohort | test split、generated count |
| denominator | raw/accepted/refined/valid-only |
| Direct semantics | comp/struct/joint 的 evaluator 和阈值 |
| refinement | none/MLIP/steps/checkpoint |
| S.U.N. | strict/meta threshold、novelty database、unique rule |
| hull | MP snapshot/API/unknown handling |
| status | reported-only 或 locally recomputed |

任何无法准确定位的值不进入主表。

### 9.2 两套表必须分开

**Table A: reported literature values**

- 逐篇按原论文定义；
- 不宣称完全同口径；
- 每个数值附 source ledger。

**Table B: common-evaluator recomputation**

- H1-A2、CrysLLMGen、final candidate；
- 尽可能加入可获得 checkpoint 的强 baseline；
- same raw attempt count；
- same Direct evaluator；
- same refiner/no-refiner stage；
- same novelty database；
- same hull snapshot；
- same unknown handling；
- same all-attempt denominator。

reported value 不得用来填补 common-evaluator 缺口。

### 9.3 before/after refiner

每个系统至少报告：

```text
proposal before model_494
proposal after model_494 exact800
```

以回答：

- Planner/Body 自身产生了什么；
- frozen refiner 修复了什么；
- stability 改善是否只是 refiner 贡献；
- candidate 是否对特定 refiner 过拟合。

### 9.4 主指标

Primary：

- generation completion；
- raw all-attempt comp/struct/joint；
- strict/meta S.U.N.；
- novel、unique、novel_unique；
- paired discordance/McNemar；
- paired/hierarchical bootstrap；
- independent-panel sign stability。

Secondary：

- successful/refined conditional Direct；
- charge/Pauling/shortcut taxonomy；
- refiner displacement；
- element/arity/N/density drift；
- valid-stable-novel per GPU-hour。

### 9.5 SOTA claim 门

只有同时满足以下条件才能写 “state of the art”：

- common-evaluator 表支持；
- baseline checkpoint/implementation 没有明显降级；
- denominator 与 threshold 一致；
- reported-only 表没有被当作同口径；
- candidate 不是靠 retry/rerank/survivor filtering；
- confidence interval 和 independent panels 支持；
- comp/strict 改善没有以 meta/diversity 显著损失为代价。

否则使用：

```text
strongest under our controlled all-attempt protocol
```

或只写具体指标，不写宽泛 SOTA。

### 9.6 统计合同

64 是机制早停，不做正式显著性结论。256 与独立 panels：

- binary paired endpoint：discordance table + exact McNemar；
- rate delta：ordinal paired bootstrap；
- 多 panel：panel-aware/hierarchical paired bootstrap；
- continuous hull/refiner displacement：paired distribution summary，不只报均值；
- primary estimands 在看结果前固定为 raw comp 与 raw joint；
- strict/meta、novelty、uniqueness 全部报告，不用多指标选择性隐藏；
- noninferiority margin 预先冻结：completion/structure 1 pp，meta 2 pp，
  novelty/uniqueness 2 pp；任何修改都需要新实验而不是回写旧结果；
- 4 个 panel 的 sign stability 与 pooled interval 同时报告；
- CUDA process variation 只作 refiner noise，不进入 independent panel 数。

如果 256 point estimate 正但区间宽，结论写为 promising/preliminary；不能用
单个 `p<0.05` 替代独立确认，也不能把多个 endpoint 中偶然显著的一个改成
primary。

---

## 10. 论文定位与 9 页结构

### 10.1 推荐题目

以下方法题目只有在 08-08 novelty gate、terminal-only 消融和 prefix
affected-rate 通过后才解锁；此前论文工作题目保持中性的
`PlanGraph-DLM: Stage-Specific Crystal Generation and Evaluation`。

条件主候选：

> Chemistry-Reachable Planning for Exact-Length Diffusion Crystal Generation

条件备选：

> Plan Before You Diffuse: Proof-Carrying Crystal Planning for Exact-Length
> Diffusion Language Models

避免：

- “first chemically valid crystal generator”；
- “guaranteed stable crystal generation”；
- 在 title 同时放 Planner、token compression 和 RL；
- 在没有 common evaluator 前写 SOTA。

### 10.2 三个贡献

如果 CR-Plan 完整通过，主文只保留三个贡献：

1. **Hierarchical constraint allocation.** rich Plan、exact-length Body-DLM
   和 frozen continuous refiner 分别承担化学、离散结构和连续几何；
2. **Table-relative certified reachability.** formula prefix 上的 grammar ×
   finite-budget charge reachability，一次采样且 terminal 有可复算 sidecar
   witness；仅在三臂消融证明 prefix 增量后使用 “proof-carrying” 简称；
3. **Causal all-attempt evaluation.** failure ownership、before/after-refiner、
   strict/meta、common snapshot、independent panels，无后处理 survivor bias。

P-control、special-token audit、safe-axis polarization 和 RL design 放在消融、
诊断或 future work，不能与三个核心贡献争夺 9 页空间。

### 10.3 建议页数

| Section | pages | 内容 |
|---|---:|---|
| 1 Introduction | 0.9 | 问题、现有 generator 的 stage ambiguity、三贡献 |
| 2 System and problem | 1.3 | rich Plan、exact-length DLM、refiner、all-attempt 定义 |
| 3 CR-Plan method | 1.8 | automaton、DP、witness、single-pass algorithm |
| 4 Experimental protocol | 1.0 | datasets、baselines、Direct/S.U.N.、independent panels |
| 5 Main results | 1.8 | common table、comp/joint、strict/meta、before/after |
| 6 Mechanism and ablation | 1.1 | charge taxonomy、removed mass、drift、failure cases |
| 7 Related work | 0.6 | CrysVCD/PLaID++/CRYSTAL/DLM-RL 精确边界 |
| 8 Limitations and conclusion | 0.5 | oxidation assumptions、MP coverage、RL future |
| Total | 9.0 | 不用附录掩盖核心定义 |

### 10.4 必须准备的图表

Figures：

1. 整体 pipeline 与每层 constraint/failure ownership；
2. CR-Plan product automaton 与 oxidation witness 示例；
3. control/candidate 在 comp–meta–strict 空间的 paired effect/Pareto 图；
4. charge failure、removed probability mass、formula drift 的机制图。

Main tables：

1. common-evaluator baselines；
2. P0 vs P0+CR-Plan primary endpoints；
3. before/after-refiner 与 failure taxonomy；
4. independent-panel statistics/compute。

Appendix：

- 完整 algorithm/pseudocode；
- tokenizer/formula DFA；
- oxidation table/version；
- 所有 32/64/256/panel gates；
- ordinal/seed/SHA manifests；
- per-panel metrics 和 failure rows；
- special-token coverage；
- negative/stopped results；
- compute、API、hull coverage；
- AI use statement。

### 10.5 Abstract 只能在结果冻结后填的槽

摘要模板：

```text
Problem:
  Existing crystal generators conflate chemistry, discrete structure and
  continuous refinement, obscuring where validity gains arise.

Method:
  We introduce a rich-plan/exact-length-DLM hierarchy and a table-relative
  certified formula-prefix reachability decoder.

Protocol:
  We evaluate every registered attempt without repair, retry, filtering or
  reranking, before and after a frozen continuous refiner.

Result:
  [Only insert independently confirmed common-evaluator numbers.]

Conclusion:
  Stage-specific verified constraints improve [only confirmed endpoints]
  while preserving [only confirmed noninferiority endpoints].
```

在 08-31 前不把 provisional 64/256 数字写进摘要。

### 10.6 AI 使用披露

ICLR 要求披露生成式 AI 在 hypothesis/method design、implementation、
analysis、interpretation 和 writing 中的使用。项目从现在开始保存：

- AI 辅助的文献整理与方案提议；
- subagent propose/cross-review/red-team 流程；
- AI 生成或修改的代码/测试；
- 人工核验人和核验结果；
- AI 未参与的最终实验决定、作者责任和数据真实性确认。

披露必须真实、简洁，不能因为担心评价而隐藏当前已发生的协作。

---

## 11. 倒排执行日历

### 11.1 2026-08-04 至 08-07：只读证据与设计冻结

允许：

- CrysVCD/等价 constrained-decoding claim-by-claim novelty audit；
- P-control 旧 ledger 只读审计；
- CR-Plan DFA/DP/witness specification 与 fixtures；
- special-token full-corpus Gate −1；
- common evaluator/cache/baseline source inventory；
- paper skeleton、figures mock 和 AI-use log。

不允许：

- 同时训练 Planner/Body/RL；
- 根据 S.U.N. 选择 constraint；
- 继续设计第四个候选；
- 把 P-control、safe-axis 或 compiler 合并到 CR-Plan。

### 11.2 08-08 至 08-12：唯一候选工程闭环

默认：

- P0 冻结为 primary causal anchor；
- CR-Plan R0、32；
- common evaluator contract/MP cache path freeze；
- PILS-L 只保留 cold-backup readiness；
- RL 不执行。

如果 CR-Plan 在 novelty 或基础工程上不晚于 08-08/10 失败，已通过
Gate −1 的 PILS-L 只获得 backup eligibility。只有独立 execution annex 在
08-10 前 release-ready 且获得新授权，才可成为唯一 active candidate；否则
回退 frozen H1。Planner 与 PILS-L 不能两线竞争结果。

### 11.3 08-13 至 08-15：64 机制裁决

08-15 只允许一个 active candidate terminal：

- pass：成为唯一 256 主线；
- fail：不再晚切新候选，回到 frozen H1 paper fallback；
- partial/queue delay：按 fail 处理，不降低 gate。

### 11.4 08-16 至 08-22：256 科学裁决

- exact800、Direct、common-snapshot S.U.N.；
- paired effects、两个 128 blocks；
- shortcut/diversity/distribution drift；
- no parameter/threshold/checkpoint selection。

08-22 后不能通过新的小试验修改方法。

### 11.5 08-23 至 08-31：独立确认

- 4 个真正独立 panels；
- shared evaluation contract；
- hierarchical paired bootstrap；
- external/independent evaluator audit；
- core ablation；
- baseline common table。

只有一个候选获得这段预算。

### 11.6 09-01 至 09-05：科学冻结

- final source/result/figure/table SHA；
- claim-by-claim evidence map；
- limitations；
- reproducibility checklist；
- author review。

09-05 后不启动 principal factor。

### 11.7 09-06 至 09-25：只写作和修错

- 09-12 进入 only-fix；
- 09-18 摘要；
- 09-25 全文；
- 只允许修复文字、图表、引用、匿名和复现包错误；
- 不因故事缺口补选模型、seed 或 metric。

---

## 12. 资源与责任分离

### 12.1 A800 预算

规划 cap，不是默认花满：

| Stage | target cap | hard cap |
|---|---:|---:|
| CR-Plan 到 paired-64 | 8 GPUh | 12 GPUh |
| 累计到 paired-256 | 48 GPUh | 72 GPUh |
| `4×256` confirmation、共同评测与冻结 robustness audit 追加 | 48 GPUh | 64 GPUh |
| CR-Plan 投稿路径总计 | 96 GPUh | 136 GPUh |
| P-control 旧 ledger / special-token Gate −1 | 0 GPUh，CPU/read-only | 12 GPUh diagnostic-only |
| P-control 新 1,024 | 不在必需队列 | 0 |
| PILS-L matched SFT（CR-Plan active 时） | 0 | 0 |
| RL 本投稿 | 0 | 0 |

预算采用 `GPU count × wall hours`，不能把“两天 2×A800”写成“两天”而漏计
GPU 数。`96 GPUh` 是 planning target，`136 GPUh` 是绝对 hard cap，不再存在
第二个“可选 96 hard cap”。

进入 confirmation 前必须已经预留完整 `4×256`、共同 evaluator 和对应 owner
的资源。资源不足时降低 claim 为 preliminary，而不是只保一个 panel 后仍称
confirmed，也不能牺牲 denominator、独立 panel 定义或写作冻结。special-token
Gate −1 的最多 12 GPUh 只允许读取冻结 B0 state 做 legal-mass 诊断，禁止 SFT、
reward label 或 endpoint；未使用不结转给 PILS-L/RL。

历史 common Direct/S.U.N. 是正式结果硬门。额外 independent evaluator 只作
robustness audit：使用 08-22 前冻结且已经可运行的 evaluator，最多 16 GPUh，
包含在 confirmation hard cap 内；若资产/owner/预算未在 08-22 前闭合，则降级
稳定性/SOTA claim，不在最后一周临时接入新 judge；若方向反转，保留共同口径
raw comp/joint 因果结论并披露冲突，但 broad-stability/all-metric/SOTA claim
全部失败。

### 12.2 四类 owner

最好分开：

- method owner：CR-Plan implementation 和 unit tests；
- evidence owner：common evaluator、baseline source ledger、statistics；
- paper owner：9 页骨架、图表、related work、AI disclosure；
- red-team owner：每个 gate 前检查 shortcut、confounding、deadline。

method owner 不得单独决定 final evaluator 或 promotion。

### 12.3 网络/API

- novelty/common-evaluator 路径尽早列出全部 chemsys union；
- treatment/control 使用同一 snapshot；
- unknown policy 预注册；
- API secret 只通过 runtime secret carrier；
- 不在 source、logs、manifest、TODO 或论文保存凭据；
- API failure 不通过筛掉 candidate-only systems 修复。

---

## 13. 风险登记

| Risk | 概率 | 影响 | 早期信号 | 缓解/停止 |
|---|---|---|---|---|
| CrysVCD/其他工作已有等价 prefix reachability | 中 | 致命 | claim 对照无法找到实质差异 | 08-07/08 novelty CUT |
| neutral suffix probability mass 太低 | 中 | 高 | removed mass 大、dead-end、rare tail | 32/64 停止，不加 fallback |
| comp 上升但 meta 下降 | 中 | 高 | 64/256 hull distribution 右移 | meta 非劣硬门，scientific stop |
| unary/all-metal/charge-not-applicable shortcut | 中 | 高 | candidate count 高于 control | 单独分层，candidate 不得膨胀，且排除后 primary gain 仍通过 |
| diversity collapse | 中 | 高 | formula top frequency/coverage 异常 | unique/novel hard gate |
| new chemsys hull coverage 不完整 | 中 | 高 | candidate-only unknown | common union snapshot；不删样本 |
| baseline 口径错误 | 高 | 致命 | page/denominator 找不到 | source ledger，reported/recomputed 分表 |
| process repeat 冒充独立 seed | 中 | 高 | scientific ledger SHA 相同 | 4 independent panels |
| tokenizer/DFA parity 错 | 中 | 高 | fixture/source mismatch | 32 前 fail closed |
| queue/GPU 超时 | 中 | 高 | 预计 64/256 滑出日期 | 绝对日期 CUT，不压缩证据 |
| 论文故事过载 | 高 | 高 | title 同时含 Planner/token/RL | 一主一备一砍 |
| AI use 披露不足 | 低到中 | 高 | 无 usage log/人工核验 | 即日起记录并由作者确认 |

### 13.1 五个最像 ICLR 拒稿意见的问题

1. **“方法只是 CrysVCD valence constraint 的变体。”**  
   修复：prefix product automaton、witness、single-pass/no-filter、rich-Plan
   interface 的 claim-by-claim 证据；找不到差异就停止。

2. **“指标来自 frozen refiner，不是提出的方法。”**  
   修复：before/after-refiner 双表、Body/refiner 完全冻结、paired failure
   ownership。

3. **“baseline 数值和分母不一致。”**  
   修复：reported/recomputed 分表、source ledger、共同 evaluator。

4. **“只提高 comp，不能说明更稳定。”**  
   修复：把 comp/joint 设 primary，把 strict/meta 当非劣；只有独立正结果才
   声称 stability。

5. **“多次 pilot 和 post-selection 造成 winner's curse。”**  
   修复：P0 primary anchor、唯一 candidate、预注册 64/256、independent
   panels、不从 stopped routes 选赢家。

---

## 14. 未来 72 小时的唯一清单

不涉及实验授权的前提下，最有价值的 72 小时交付：

1. 完成 CrysVCD 与等价 prefix constrained-decoding 的 claim matrix；
2. 冻结 P0 作为 CR-Plan primary anchor；
3. 完成 formula tokenizer DFA、charge-reachability DP 和 witness 书面合同；
4. 建 100% 可枚举的 synthetic/brute-force oracle fixtures；
5. 完成 P-control 旧 512 的 shortcut/drift 只读审计；
6. 完成 special-token full train/val/test/generated coverage 与 legal-mass
   Gate −1；
7. 完成 H1-A2/CrysLLMGen/external papers 的 baseline source ledger；
8. 冻结 common Direct/S.U.N./MP snapshot/unknown contract；
9. 建立 9 页论文骨架、四张图 mock 和表格空槽；
10. 建立 AI-use、author/OpenReview、匿名复现包 checklist。

完成这十项后才决定是否授权 32。不要把“代码已经能跑”当作 novelty 和
evaluation contract 已通过。

---

## 15. 投稿后的优先研究方向

### 15.1 Planner 2.0：typed proof-carrying Plan

在 CR-Plan 独立验证后，再逐项扩展：

1. charge witness；
2. Pauling compatibility；
3. explicit oxidation-state candidates；
4. density/volume envelope；
5. symmetry/space-group compatibility；
6. learned reachability-mass alignment。

每一项先独立测，再做 factorial。目标不是把所有规则塞入 hard mask，而是将：

```text
hard impossible
soft chemically implausible
learned distribution preference
```

分成不同层。

### 15.2 Body representation 2.0

特殊 token “很多没用上”应按三步处理：

1. **测量：** full-train count + legal probability mass + endpoint attribution；
2. **因子化：** PILS-L；
3. **重表示：** family token + ordinal value、monotonic/ordinal head、或
   coarse-to-fine value code。

不能直接删除所有当前未出现 token，因为它们可能代表合法但稀有的晶格区域。
长期目标是让数值邻近关系进入参数共享，而不是为每个 bin 永久维护互不相关的
分类 row。

### 15.3 Mask-aware RL

投稿后按已完成的设计只训练一个长期 LoRA：

- support-aware token probability；
- group-local K=1 PL reveal policy；
- joint token-position trace；
- exact resume；
- pre/post-refiner randomized multi-fidelity reward；
- meta-aware constrained optimization；
- independent MLIP/DFT evaluator。

两个模型只保留为“是否部署 refiner”的产品分叉，不因为两个 reward fidelity
就训练两个长期模型。

### 15.4 分层 credit assignment

最终可以分别优化：

- Planner：composition feasibility、chemical diversity；
- Body：completion、geometry、refiner basin；
- Refiner：保持冻结或独立研究；
- global selector：只在正式设计允许时做多目标约束。

在三个单因素都通过前，不做 end-to-end joint RL。否则 composition reward 会
被错误归因给 Body，refiner reward 会被错误归因给 Planner。

---

## 16. 复现与决策防火墙

每个新 workstream 必须：

- 新 source/run root，不覆盖历史；
- source/config/data/tokenizer/checkpoint/evaluator SHA；
- authorization record；
- immutable ordinal/seed ledger；
- all-attempt attempt_results；
- stage reports 与 terminal；
- explicit `formal_g3=false`、`automatic_promotion=false`、
  `automatic_training=false`、`automatic_downstream=false`，除非有新的明确
  授权；
- 失败证据保留，不通过改 seed、阈值、分母或 checkpoint 修复科学 stop。

本文提出的是路线，不是 execution authorization。

---

## 17. 综合结论

在 52 天窗口中，最合理的研究组合不是“Planner、特殊 token 和 RL 都做一点”，
而是：

```text
P0 primary anchor
  -> CR-Plan as the only scientific main line
  -> exact-length B0/D1 and model_494 frozen
  -> common all-attempt evaluation

PILS-L
  -> Gate −1 cold backup only
  -> early CR-Plan stop grants eligibility, not automatic execution
  -> replacement requires a release-ready annex and new authorization

mask-aware RL
  -> post-ICLR
  -> zero new ICLR GPU budget
```

这套选择把条件创新性放在 prefix reachability 和 table-relative Plan
certificate；只有 terminal-only 消融通过后才简称 proof-carrying Plan，
把效果目标放在已证实的 composition/joint bottleneck，把 strict/meta 设为
不可牺牲的安全门，并把 common evaluator、独立确认和写作时间当作与模型同等
重要的资源。

最关键的诚实边界：

- CR-Plan 可以合理期待改善 comp/joint，不能预先承诺 meta；
- special-token sharing 可能改善 meta，但根因尚未证明；
- RL 长期值得做，但今年现在启动最可能留下不完整 pilot；
- common evaluator 未完成前，不能因为已有表面数字领先就写 SOTA；
- 08-15 无机制信号时，应停止新模型线，而不是叠加变量。
