# DLM 原始实验 artifact 证据台账（36H V1）

审计日期：2026-08-30  
工作树：D:/codex_work/ai4s/DLM_rich_planner_audit  
审计时 HEAD：79ac47d453b58e163e2e9950ca3ccbdf03a0df07  
只读证据分支：codex/evidence-first-sun-msun@5fca5c9cd06bdbdda7c83c2cf2b37a3b891f9ee0；main@f78b38f1166d2037ea6bce825db9a705e6a70fac

本文件只把逐 attempt JSONL、冻结 manifest、机器可读终态 JSON/CSV、成功/失败标记和对应源码视为原始证据。总结文档只能作为定位器或 secondary_only 数字来源，不能补造 numerator、seed、process independence 或因果解释。机器可读同源台账见 DLM_ORIGINAL_ARTIFACT_EVIDENCE_LEDGER_36H_V1.json。

## 证据等级

- G4：逐 attempt 行、不可变 manifest/hash、终态报告和对应源码齐全。
- G3：终态 JSON/CSV 与源码/contract 齐全，但本工作树没有完整逐 attempt 行。
- G2：只有部分原始行，或 aggregate report 加冻结 manifest/source；不能本地完整重算。
- G1：只有总结、源码常量或远端 locator；结果数字一律 secondary_only。
- G0：没有执行后的 metric-bearing artifact。

## 全局核验结论

- 当前 runs 目录只有 README.md；历史 run root 没有签出。
- R03 有 Git 对象中的 first256 逐 attempt 证据和机器可读 replay terminal。R5C、H1-B、H1-A2 exact/continuous、SGTC L6、D3PO fixed256 终态以及 tau900/1000 均存在原始 artifact 缺口。
- R03 旧 frozen-cache D2 为 117/496；clean-cache replay 为 120/500。二者是不同 evaluator/cache 版本，禁止混写。
- C3FD-v2.5 requested1000 JSON 的 claim_boundary 字符串误留 requested256；cells 和 requested1000 gate 才是可核字段。
- L6 原始 contract 把 full_axis 标为 full_plan_state、hard_axis 标为 hard_anchor_only；不能把 full_axis 直接改写成纯 minimal prompt。
- SGTC L6 终态没有复制回来；L7 contract 中“L6 authorized L7”的一句话不能替代 L6 结果。
- 没有发现 D3PO_TRAIN_FINAL、D3PO_GENERATION_FINAL 或 D3PO_FIXED256_OFFICIAL_FINAL。

## 逐项台账

### 1. R5C fixed-slot / exact-dynamic

- Artifact path：
  - remote-only：/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260529_212834-r5c-exactlen-256
  - source：git:main@f78b38f1166d2037ea6bce825db9a705e6a70fac:legacy_dlm_r5c/crystal_dlm/r5_dynamic_length.py
  - source：git:main@f78b38f1166d2037ea6bce825db9a705e6a70fac:legacy_dlm_r5c/launchers/pre_wyckoff/a800/run_r5c_de_novo_1000_sun.sh
  - secondary_only：results/remote_screens/DLM_CAPABILITY_REGRESSION_36H.json
- Denominator：原始 manifest missing。总结记录 requested1200、graphs1167；refined view 为 1000。fixed-slot 107-token 也只有 secondary 记录。
- Seed/process unit：missing。没有历史 global train seed、inference seed、rank/process manifest 或逐 ordinal ledger。
- Exact metric：secondary_only 为 1167/1200=97.25%，refined1000 composition-valid=90.7%，structure-valid=99.8%。
- 成功原因：源码可核 exact dynamic 7+4N 和结构化 Plan-conditioned body/refiner 执行路径。
- 失败原因：原始 run manifest、logs、sample metrics、proposal graphs、refined outputs、attempt ledger 均 missing；该视图也不是干净 de-novo S.U.N.
- 可支持 claim：R5C 提供 dynamic exact-length executor 设计与 checkpoint locator；可淘汰 padded fixed-slot 目标。
- 禁止 claim：不得把 1167/1200 或 90.7/99.8 写成当前工作树可重算的 primary evidence；不得宣称 R5C 已验证 de-novo stability，或把效果归因于 padding/某个未知 seed。
- 证据等级：G1。

### 2. H1-B formula-only

- Artifact path：
  - source：src/crystal_dlm/h1_formula_only_body.py
  - source：git:main@f78b38f1166d2037ea6bce825db9a705e6a70fac:legacy_dlm_r5c/launchers/pre_wyckoff/a800/run_h1_formula_only_dlm_body.sh
  - secondary_only：results/remote_screens/DLM_CAPABILITY_REGRESSION_36H.json
- Denominator：missing；总结只有 rate，没有 requested count 或 numerator。
- Seed/process unit：missing；没有 H1-B run manifest、seed ledger 或 attempt rows。
- Exact metric：secondary_only：body success 100%（分母未知）、Strict=5.54%、Meta=43.13%、all-90-degree=81.57%、repeated-two-lengths=53.33%。
- 成功原因：源码确认 formula/elements/counts/N 精确锚定和 dynamic body 生成。
- 失败原因：secondary 几何统计指向模板塌缩，但原始行和分母缺失，百分比不能独立复算。
- 可支持 claim：formula-only 接口能执行；body success 不等于几何或热力学质量。
- 禁止 claim：不得把四个百分比升级成 primary artifact；不得宣称 seed-robust 因果塌缩率或“所有 formula-only 方法都失败”。
- 证据等级：G1。

### 3. H1-A2 historical compatibility

- Artifact path：
  - results/internal_results.json
  - results/reference_results.json
  - remote-only raw Plans：/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_planner1200
  - source defaults：src/h1a2_repro/science.py
- Denominator：historical compatibility=1000；本地缺 frozen1000 选择 manifest。
- Seed/process unit：单一历史 process view。Planner train seed17 有配置级记录；DLM global train seed、DLM/refiner inference seeds 均 missing；不是独立 seed replication。
- Exact metric：secondary aggregate：Strict 94/1000=9.40%，Meta 474/1000=47.40%，N∩U 890/1000=89.0%。
- 成功原因：两份 curated JSON 对 94/474 一致，且能定位 rich-Plan epoch2 checkpoint/Plan lineage。
- 失败原因：没有 frozen1000 attempt ledger、input/evaluator manifest 或 outcome rows；也不能从该 view 重建独立 public 105/488。
- 可支持 claim：94/474 是历史兼容参考；一个 rich-Plan 历史 process 曾进入有用 basin。
- 禁止 claim：不得称为 replicated architecture effect；不得把 105/488 归入该 frozen1000；不得归因于未知 DLM/refiner seed。
- 证据等级：G1。

### 4. H1-A2 exact all-attempt replay

- Artifact path：results/reference_results.json；src/h1a2_repro/science.py。
- Denominator：requested1200；hull known/unknown=1132/32。原始 attempt/reconstruction ledger missing。
- Seed/process unit：missing；没有 replay seed、world/rank 映射或 attempt seed ledger。
- Exact metric：secondary aggregate：Strict 103/1200=8.583333%，Meta 553/1200=46.083333%。
- 成功原因：使用 requested1200 分母，避免 frozen-success survivor denominator。
- 失败原因：attempt labels、generation failures、manifest 和 evaluator cache 均 missing。
- 可支持 claim：all-attempt accounting 明显低于 historical compatibility，说明选择/记账会移动 headline。
- 禁止 claim：不得称为独立 training-seed replication；不得说它完全解释历史到当前的 regression；不得把 curated count 叫原始 artifact。
- 证据等级：G1。

### 5. H1-A2 continuous replay

- Artifact path：secondary_only results/remote_screens/DLM_CAPABILITY_REGRESSION_36H.json。
- Denominator：secondary pooled3840；构成 repeat、reconstructed 和 hull-known 分母 missing。
- Seed/process unit：missing；不能假设 3840 条独立。
- Exact metric：rounded secondary_only：Strict=7.63%，Meta=45.47%；numerator missing。
- 成功原因：保留了一个 pooled sensitivity reference。
- 失败原因：终态 JSON/CSV、per-repeat rows、numerators、cache manifest、process ledger 均 missing。
- 可支持 claim：仅可说存在一个低于 historical compatibility 的 secondary continuous reference。
- 禁止 claim：不得反推 numerator/置信区间、不得称 primary evidence、不得称 seed-robust。
- 证据等级：G1。

### 6. R03 D1/D2 historical four-process panel

- Artifact path：
  - aggregate/hash report：git:main@f78b38f1166d2037ea6bce825db9a705e6a70fac:workstreams/plangraph_dlm_iclr_20260731/H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md
  - execution manifest：git:main@f78b38f1166d2037ea6bce825db9a705e6a70fac:workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1/EXECUTION_MANIFEST.json
  - first256 D1 rows：git:codex/evidence-first-sun-msun@5fca5c9cd06bdbdda7c83c2cf2b37a3b891f9ee0:workstreams/final_method_development_20260808/evidence/h1_r03_h1a2_archived_first256_downstream_repair_v4/arms/control/generation/generation.jsonl
  - first256 D2 rows：同目录 arms/candidate/generation/generation.jsonl
  - schedule source：src/r03/safe_axis_schedule.py
- Denominator：每 arm 4 process repeats × 256 frozen Plans=1024；upstream failures 仍占分母。
- Seed/process unit：四个独立 CUDA process realization 复用同一 first256 Plan cohort 和 ordinal ledger；不是四个独立 Planner sample/training seed。
- Exact metric：
  - D1 Strict/Meta=99/1024（9.667969%）/523/1024（51.074219%）。
  - D2 Strict/Meta=117/1024（11.425781%）/496/1024（48.4375%）。
  - D1 generation/comp/struct/joint=984/848/982/846。
  - D2=992/852/989/851。
- 成功原因：safe-axis 源码强制 exact coverage、无 mixed-axis group、所有 X/Y 早于 Z；历史 panel 中 completion 和 Strict 增加。
- 失败原因：Meta 减少27；Plan cohort 非独立；continuous refiner 有 process sensitivity；完整四 repeat attempt rows 未复制。
- 可支持 claim：coordinate commitment order 是可执行设计变量；在冻结 cohort/process panel 上 D2 呈现 Strict↑、Meta↓。
- 禁止 claim：不得说 D2 广泛提高 stability；不得把四 repeats 当独立 Planner seeds；不得外推 +18 Strict；不得混 old/clean counts。
- 证据等级：G2。

### 7. R03 D2 clean-cache replay

- Artifact path：
  - git:codex/evidence-first-sun-msun@5fca5c9cd06bdbdda7c83c2cf2b37a3b891f9ee0:workstreams/final_method_development_20260808/evidence/h1_r03_refined256_current_sun_cache_replay_v2/terminal_report.json
  - 同目录 repeats/0..3/attempt_summary.json 与 repeat_validation.json
  - git:codex/evidence-first-sun-msun@5fca5c9cd06bdbdda7c83c2cf2b37a3b891f9ee0:workstreams/final_method_development_20260808/evidence/h1_sun_official_gga_u_skip_unknown_reeval_v2/terminal_report.json
- Denominator：4×256=1024 all attempts；每 repeat 9 hull unknown、247 all-attempt-skip-unknown。
- Seed/process unit：同一 frozen first256；terminal 明确 repeat_role=independent_cuda_process_realization_on_same_frozen_first256_cohort，且 pooled_1024_independence_assumed=false。
- Exact metric：clean per-repeat Strict=[28,32,30,30]，pooled120/1024=11.71875%；Meta=[122,125,126,127]，pooled500/1024=48.828125%。同一 terminal 的 old fields 为117/496。
- 成功原因：保持 byte-frozen historical refined trajectories，用 clean compatible-entry cache 补全评估并显式保留 hull unknown。
- 失败原因：仅 evaluation replay，未重跑 generation/refinement；repeats 共用 cohort；clean 与 old count 不同。
- 可支持 claim：archived D2 trajectory 在 clean cache 下仍是历史高点；cache/evaluator version 会改变 exact count。
- 禁止 claim：不得称新 generation replication；不得称独立 Planner seed；不得把120/500与117/496互换。
- 证据等级：G3。

### 8. R03 corrected exact / continuous references

- Artifact path：secondary_only results/remote_screens/DLM_CAPABILITY_REGRESSION_36H.json。
- Denominator：exact requested1200；continuous pooled3840。
- Seed/process unit：missing；没有 constituent process manifest，禁止 independence assumption。
- Exact metric：rounded secondary_only：exact Strict/Meta=8.42%/46.58%；continuous=7.47%/45.26%；numerators missing。
- 成功原因：保留 corrected sensitivity reference。
- 失败原因：没有 original terminal、numerators、attempt rows、cache manifest 或 seed ledger。
- 可支持 claim：仅可说 corrected references 低于 historical D2。
- 禁止 claim：不得反推 numerator/uncertainty、不得叫 primary evidence、不得称 R03 seed-robust。
- 证据等级：G1。

### 9. Rich SFT epoch2 / epoch3

- Artifact path：
  - results/remote_screens/DLM_SUFFICIENT_RAW1000_FINAL.json
  - results/remote_screens/DLM_SUFFICIENT_RAW1000_FINAL.csv
  - slurm/33_dlm_sufficient_raw1000_two500.sbatch
  - scripts/finalize_dlm_sufficient_raw1000.py
- Denominator：每 checkpoint requested1000，冻结 first500/last500；S.U.N. all-attempt denominator=1000。
- Seed/process unit：同一 DLM seed17117、refiner seed27117、seed-by-sample-index、同一 Plan cohort/stream。epoch2=step696，epoch3=step2392；不是两个独立 training seeds。
- Exact metric：
  - epoch2：body985，Direct joint871，Strict81，Meta489，hull K/U=958/27。
  - epoch3：body992，Direct joint878，Strict79，Meta477，hull K/U=965/27。
  - known-both=956；Strict McNemar 19/21，p=0.874629；Meta 81/96，p=0.292637。
- 成功原因：多一个 CE epoch 让 body 和 Direct 各增7。
- 失败原因：Strict -2、Meta -12；frozen epoch3 gate 的 strict/meta noninferior 均 false，最终 selected_epoch=2。
- 可支持 claim：ordinary CE 可改善 execution 而不改善 thermodynamics；epoch2 是序列化 Pareto gate 的选择。
- 禁止 claim：不得称 epoch3 stability improvement、不得称 multi-seed replication、不得按最低 val CE 选 stability checkpoint。
- 证据等级：G3。

### 10. Counterfactual grounding

- Artifact path：
  - results/remote_screens/GROUNDING_FINAL_REPEAT4.json/.csv
  - results/remote_screens/GROUNDING_FIXED1000_FINAL.json/.csv
  - results/remote_screens/GROUNDING_CHECKPOINT_SWEEP_FINAL.json/.csv
  - results/remote_screens/GROUNDING_OFFICIAL_INPUT_MANIFEST.json
  - results/remote_screens/GROUNDING_OFFICIAL_COMPLETION_MANIFEST.json
  - slurm/23_grounding_repeat4.sbatch
- Denominator：repeat4 每 arm 4×256=1024；fixed confirmation 每 arm1000；sweep 在 step500/1000/1696 各每 arm256。
- Seed/process unit：repeat4 DLM seeds=[17117,17217,17317,17417]、refiner=[27117,27217,27317,27417]，同一 frozen256 Plans，arms paired。fixed1000 使用 DLM18117/refiner28117，first1000 parsed、无 survivor filter。
- Exact metric：
  - mechanism margin=0.7591561057；candidate-control factual val CE=-0.0004459551。
  - repeat4 control body/direct/Strict/Meta=1017/920/103/472；candidate=1020/923/105/460，即 Strict +2、Meta -12。
  - fixed1000 control=89/487；candidate=86/467，即 -3/-20。
  - sweep delta：step500 +3/-1；step1000 +3/+4；step1696 -5/-2；qualifying_steps=[]。
- 成功原因：candidate 学到 factual-vs-counterfactual separation，repeat4 body/Direct 非劣。
- 失败原因：checkpoint direction 非单调、repeat4 Meta 下降、fresh fixed1000 Strict/Meta 都下降；contribution_pass=false。
- 可支持 claim：mechanism 有变化，但没有转化为可复制 S.U.N. 改善；fixed1000 是冻结 cohort 下的正式负结果。
- 禁止 claim：不得 post-hoc 选择 step500/1000；不得用 NLL/margin 代替 stability；不得把 +2/1024 包装成贡献成功。
- 证据等级：G3。

### 11. C3FD-v2

- Artifact path：results/remote_screens/C3FD_PLANNER_FINAL.json/.csv；scripts/finalize_c3fd_planner.py。
- Denominator：seed17/18 各 requested1000；每 arm pooled2000。
- Seed/process unit：两个 Planner sample seeds17/18；单位是 requested formula/Plan attempt，不包含 structure/refiner。
- Exact metric：P0 comp-valid1724/2000=86.2%；C3FD-v2=1972/2000=98.6%；paired delta=0.124，CI95 [0.108096,0.139904]，两 seed 均+0.124。drift：all-metal abs7.45pp、arity TVD0.1955、max-family abs5.2pp、N TVD0.162；unique1919/2000 对 P0 1961/2000。
- 成功原因：两个 seeds 都有相同 +12.4pp composition-validity 增益。
- 失败原因：distribution drift 和 unique-noninferiority gate 失败，c3fd_v2_pass=false。
- 可支持 claim：v2 在两 seed 上提高 composition correctness，但需要分布修正。
- 禁止 claim：不得说 v2 full gate pass、不得说改进 structure/S.U.N.、不得把 composition gain 等同 discovery gain。
- 证据等级：G3。

### 12. C3FD-v2.1 至 v2.4

- Artifact path：
  - results/remote_screens/C3FD_V21_PILOT_FINAL.json
  - results/remote_screens/C3FD_V22_AUDIT_V1_ENGINEERING_FAILURE.json
  - results/remote_screens/C3FD_V22_AUDIT_V2_ENGINEERING_FAILURE.json
  - results/remote_screens/C3FD_V23_AUDIT_ENGINEERING_FAILURE.json
  - results/remote_screens/C3FD_V24_AUDIT_ENGINEERING_FAILURE.json
- Denominator：v2.1 为 seed17/18×requested256，pooled512/arm；v2.2-v2.4 是 runtime gate，不是 scientific sample denominator。
- Seed/process unit：v2.1 两个 Planner seeds；v2.2-v2.4 在 GPU/scientific sampling 前被冻结 CPU runtime gate 终止。
- Exact metric：
  - v2.1 P0 comp438/512，对 candidate460/512；delta0.042969，CI95 [0.002954,0.082984]。
  - candidate parse460/512 对 P0 510/512；all-metal parsed39.565% 对 full-train34.906%；step5_pass=false。
  - v2.2 312s、317s 超300s；v2.3 120s gate；v2.4 126s 超60s、SIGTERM；v2.4 teacher witness train24558/24558、val8159/8159。
- 成功原因：v2.1 保持正 composition delta；v2.4 证明 teacher witness 全覆盖并定位 runtime bottleneck。
- 失败原因：v2.1 parse/all-metal gate 失败；v2.2-v2.4 超时，均显式 scientific_result=false、gpu_authorized=false。
- 可支持 claim：该 lineage 区分 scientific pilot failure 与 engineering failure；teacher witness existence 在 v2.5 前已完整。
- 禁止 claim：不得把 v2.2-v2.4 当科学结果、不得由 timeout 推断质量、不得说为 v2.5 放宽 gate。
- 证据等级：G2。

### 13. C3FD-v2.5 requested1000

- Artifact path：
  - results/remote_screens/C3FD_V25_REQUESTED1000_FINAL.json/.csv
  - results/remote_screens/C3FD_V25_TEACHER_WITNESS_AUDIT.json
  - archives/successful_contributions_20260828/c3fd_v2_5/MANIFEST.json
  - archives/successful_contributions_20260828/c3fd_v2_5/CHECKPOINT_POINTERS.json
- Denominator：seed17/18 各 requested1000；pooled2000/arm。
- Seed/process unit：两 Planner sampling seeds17/18；checkpoint pointer 记录各自 path/bytes/SHA。单位仍是 formula/Plan attempt，不含 downstream structure。
- Exact metric：
  - candidate parse/formula/legacy-comp=2000/2000；P0 legacy-comp=1724/2000。
  - paired delta0.138，CI95 [0.122881,0.153119]；McNemar f1-only/f0-only=276/0，p=1.6472e-83。
  - N∩U candidate1756/2000=87.8%，P0 1530/2000=76.5%。
  - semantic dead ends seed17/18=0/0；teacher witness train24558/24558、val8159/8159。
- 成功原因：exact online witness decoding 消除 semantic dead end，并在两个 requested1000 seeds 达到100% parse/formula/composition。
- 失败原因：仅 composition evidence，无 structure/energy/hull/Direct/S.U.N.；claim_boundary 字符串的 requested256 是 stale 文本。
- 可支持 claim：C3FD-v2.5 是 two-seed composition-correctness contribution；checkpoint/result 有冻结 hash manifest。
- 禁止 claim：不得宣称 structure stability/S.U.N./external SOTA；不得让 stale requested256 覆盖真实 cells；不得用 composition correctness 代替 realization quality。
- 证据等级：G3。

### 14. Minimal-executor L6 bridge

- Artifact path：results/remote_screens/DLM_CONDITION_SCHEDULE_L6_FINAL.json/.csv；slurm/38_dlm_condition_schedule_l6.sbatch。
- Denominator：seed17/18×256/arm；full_axis、full_joint、hard_axis、hard_joint 各 pooled512。
- Seed/process unit：DLM seeds17117/18117；refiner27117/28117；同一 L6 Plans、public H1-A2 checkpoint、tau800；无 RL/rerank/CFG。
- Exact metric：
  - full_axis body/Direct/Strict/Meta=505/457/48/230。
  - hard_axis=512/463/47/230。
  - full_joint=319/295/29/146。
  - hard_joint=417/376/35/181，promote_hard_joint=false。
- 成功原因：exact-axis arms 保持近饱和 execution，明显优于 joint-coordinate commitment；artifact 选择 full_axis。
- 失败原因：factorial joint schedule 平均 body -27.4414pp、Direct -24.3164pp；所有 candidate promotion 都 ineligible。源码标签表明 full_axis 是 full_plan_state，不能擅自改成纯 minimal。
- 可支持 claim：exact-axis 是受支持的 L6 schedule；joint-coordinate 是正式 no-go；hard/full axis pooled S.U.N. 接近但不能宣称等价。
- 禁止 claim：不得说 full_axis 已证明是纯 composition+N prompt；不得说 hard_axis 是 stability improvement；不得忽略两个 seed block 的依赖。
- 证据等级：G3。

### 15. Minimal-spec L7 base

- Artifact path：
  - results/remote_screens/sgtc_l7_official_final_20260829_v2/SGTC_L7_OFFICIAL_FINAL.json/.csv
  - 同目录 _SUCCESS
  - docs/SGTC_DLM_L7_CONTRACT_V1.md
- Denominator：requested1000 seed18；base reconstructed998；hull K/U=979/19；S.U.N. denominator=1000。
- Seed/process unit：one downstream holdout seed18；DLM92117、refiner102117、temperature0.7、exact-axis、model494 tau800、每 Plan 一次；不是 multi-seed 或 globally fresh。
- Exact metric：parsed/body/refined=998/998/998；Direct comp/struct/joint=998/996/996；N/U/NU=922/995/922；Strict stable/S.U.N.=81/60；Meta=486/412。
- 成功原因：minimal-spec execution 和 post-refiner Direct 基本饱和。
- 失败原因：stable-to-N/U conversion 低，Strict60/1000、Meta412/1000，低于 preregistered100/500。
- 可支持 claim：minimal executor 可可靠实现 C3FD composition；瓶颈不在基本 parse/body。
- 禁止 claim：不得说恢复 historical H1-A2/R03 stability；不得称 multi-seed confirmation；不得由 post-refiner Direct 推断 raw DLM geometry。
- 证据等级：G3。

### 16. SGTC-DLM-v1 L7

- Artifact path：
  - results/remote_screens/sgtc_l7_official_final_20260829_v2/SGTC_L7_OFFICIAL_FINAL.json/.csv
  - docs/SGTC_DLM_V1_CONTRACT.md
  - slurm/63_sgtc_l7_generation.sbatch
- Denominator：base、G0-all、G1-strict 各1000 attempts；base pairing known-both=979。
- Seed/process unit：同一 seed18 cohort、DLM92117/refiner102117/tau800；G0/G1 training contract 均 seed81017。不是 two-policy-seed replication。
- Exact metric：
  - base/G0/G1 Strict=60/55/53，Meta=412/421/417，body=998/1000/1000，Direct joint=996/997/996。
  - G1-base known-both979：Strict delta=-0.007150，p=0.264931；Meta +0.004086，p=0.789396。
  - G1-base refined CHGNet mean delta=+0.004238 eV/atom；official E_hull +0.004525，95% [-0.000339,+0.009390]，正值为不利。
  - sgtc_l7_pass=false。
- 成功原因：geometry-only continuation 保住 body/Direct/diversity floor；G0 Meta 点数小幅增加。
- 失败原因：预声明 G1 Strict 60→53、Meta仅417，mean energy/hull 轻微上升；absolute 与 G1-vs-G0 gates 均失败。
- 可支持 claim：在该 contract 下，positive-only strict-stable continuation 未证明 fixed-composition energy boundary；base 仍是 reference。
- 禁止 claim：不得说 SGTC 改善 stability；不得由单 seed 否定所有 curriculum；不得声称知道 L6 结果；不得用 training NLL 选 arm。
- 证据等级：G3。
- Missing：SGTC_L6_OFFLINE_FINAL、SGTC_L6_OFFICIAL_FINAL、L7 raw attempt JSONL/run hash manifest。

### 17. D3PO fixed256

- Artifact path：
  - design：slurm/65_d3po_train.sbatch、slurm/66_d3po_fixed256_generation.sbatch、slurm/67_d3po_fixed256_eval.sbatch
  - engineering reports：docs/D3PO_37966_ENGINEERING_FAILURE.md、D3PO_38034_ENGINEERING_FAILURE.md、D3PO_38233_ENGINEERING_FAILURE.md
  - secondary_only：results/remote_screens/DLM_CAPABILITY_REGRESSION_36H.json
- Denominator：仅 design 可核：每 cell256，arms=base/seed81017/seed81018 × streams17/18；official denominator 因终态 missing 而不可核。
- Seed/process unit：training seeds81017/81018 顺序跑至 step348；evaluation streams17/18 使用 DLM91117/92117、refiner101117/102117，共享 per-stream randomness。Outcome manifests missing。
- Exact metric：secondary_only：refined mean delta=-0.002591 eV/atom，CI crosses zero；raw mean delta=+0.200266，fraction_lower=0.366142；refined Direct 254–255/256；raw Direct base=[151,164]、seed81017=[124,137]、seed81018=[150,144]；official=pending。
- 成功原因：源码证明 two-policy-seed、two-stream、six-cell paired design；summary 仅提示 refined mean 略有利。
- 失败原因：所有 metric-bearing D3PO terminal missing；secondary raw energy 不利/不复制，refined CI 跨0，official pending。37966、38034、38233 分别在 update、sampling、model load 前工程失败。
- 可支持 claim：设计区分两 policy seeds 和两 common streams；三个 job 是 engineering failures，不是 scientific negatives。
- 禁止 claim：不得说 D3PO 改善 stability/energy/Direct/S.U.N.；不得把 secondary 数字称 original artifact；不得说 official fixed256 完成；不得把38233当 late-guidance result。
- 证据等级：G1。
- Missing：D3PO_TRAIN_FINAL、D3PO_GENERATION_FINAL、D3PO_FIXED256_OFFICIAL_FINAL、raw/refined attempt rows，以及三个 failed job 的原始 _FAILED.json/ENGINEERING_FAILURE.tsv 副本。

### 18. model494 tau

- Artifact path：
  - results/remote_screens/DLM_REFINER_EFFECT_L6_DIAGNOSTIC.json/.csv
  - results/remote_screens/DLM_REFINER_TAU_L6_FINAL.json/.csv
  - slurm/41_dlm_refiner_tau_l6.sbatch
  - tau900/1000 source only：scripts/finalize_refiner_tau_high_l6.py
  - remote checkpoint locator：/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
- Denominator：seed17/18×256=512/tau；full_axis reconstructed505，所有 tau hull K/U=489/16。
- Seed/process unit：同一 frozen full_axis bodies、sample-index pairing、refiner seeds27117/28117。tau0/800 为复用 endpoint，tau200/500 为新 matched refinements。
- Exact metric：
  - tau0：Direct188、Strict10、Meta66、E_hull q50=2.176715。
  - tau200：456/29/171，q50=0.133699。
  - tau500：458/39/222，q50=0.101886。
  - tau800：457/48/230，q50=0.091306，selected_tau=800。
  - raw→tau800：Direct188→457、Strict10→48、Meta66→230；novelty505→451；Strict/Meta retention 1→0.786885/0.818505。
  - tau900/1000：original terminal missing。
- 成功原因：完整 matched sweep 证明 model494 大幅修复 Direct/S.U.N. 并降低 median hull；frozen progress rule 选择 tau800。
- 失败原因：refinement 降低 novelty/retention，可能洗掉 DLM 差异，是 attribution confound；tau900/1000 不完整，不能改变选择。
- 可支持 claim：model494 tau800 是该 L6 cohort 的主要绝对 downstream contributor；必须同时报告 raw/refined；完整证据只支持 tau800。
- 禁止 claim：不得把 post-refiner 改善归于 DLM；不得说 tau900/1000 优于800；不得称 model494 binary/training seed 在本工作树完全可复现。
- 证据等级：G3。
- Missing：DLM_REFINER_TAU_HIGH_L6_FINAL、model494 binary/training manifest、逐 attempt tau rows。

## Claim 使用总则

1. G1/G0 数字只能以“secondary summary reports”形式出现，不能作为论文 primary table 或复现验收值。
2. historical、exact、continuous、old-cache、clean-cache 必须保留各自 denominator 和 evaluator version。
3. process repeat、DLM sampling seed、Planner sampling seed、training seed 是不同统计单位，禁止互换。
4. composition correctness、body/Direct execution、raw energy、post-model494 energy、Strict S.U.N.、Meta S.U.N. 是不同 endpoint，禁止跨 endpoint 归因。
5. artifact missing 时必须继续写 missing；不得用本文件、decision log、capability summary 或其他综述反向生成“原始证据”。
