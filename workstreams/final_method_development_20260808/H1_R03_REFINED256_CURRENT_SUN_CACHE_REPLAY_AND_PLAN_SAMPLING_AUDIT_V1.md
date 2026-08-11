# H1 R03 refined256 当前 S.U.N./MP cache 重评与 Plan 采样审计 V1

Updated: 2026-08-11 (Asia/Shanghai)

## 结论

当前低 S.U.N. **不是**由 Plan prompt 文本变化，也不是由当前 S.U.N.
实现或 MP cache 把旧结果打低造成的。

- H1A2 与 Plan1200 使用同一 P0 adapter、同一 `h1_rich_plan_v1` 七行
  prompt 分支；system/user/formatted prompt 以及 token IDs 均通过字节级
  审计。prompt 模块文件 SHA 的变化来自后来加入但本次未走到的
  `h1_rich_nocharge` 分支。
- 真正改变的是采样与 cohort 构造：历史 H1A2 使用 seed `17029` 的单一
  全局 RNG 流并冻结 first256；Plan1200 使用 stateless ordinal 采样、三个
  base seed，并从各自 raw1200 中冻结前 1000 个 parse-success plan。
- 将历史四组 byte-frozen R03 refined256 用当前 exact S.U.N. 与当前
  cohort-complete MP cache 重评后，四组 strict/meta 判定逐样本完全不变：
  两套 endpoint 都是 old-only=0、current-only=0、McNemar p=1。
- model_494 对当前 Plan1200 有巨大作用：pre 的 strict 约 1.12–2.68%、
  meta 约 12.07–15.06%，post 提升到 strict 5.93–7.22%、meta
  42.12–46.34%。因此此前看到的极低值大部分是 pre-refine 阶段效应。
- 但当前 R03 post-model494 仍低于历史 R03 refined256 的 strict
  11.29–12.50%、meta 49.19–50.81%。在 evaluator/cache、P0 权重和 active
  prompt 已排除后，最强剩余解释是不同 RNG/cohort 及其输入分布；这不是
  “P0 模型权重变差”，而是 P0 的采样协议和实际抽到的 Plan 集合变了。

## Prompt 与采样协议审计

| 项目 | 历史 H1A2 | 当前 Plan1200 | 结论 |
|---|---|---|---|
| P0 adapter | SHA `65766c74...aa3a` | 同一 SHA | 未变 |
| prompt style | `h1_rich_plan_v1` | `h1_rich_plan_v1` | 未变 |
| `include_sample_id` | false | false | 未变 |
| active seven-line branch | frozen historical bytes | 当前 bytes | byte-equal |
| generation knobs | temp 0.9, top-p 0.95, top-k 50, max-new 96, max-atoms 20 | 相同 | 未变 |
| RNG | 单一 global stream，seed 17029 | stateless ordinal，三个 base seed | **改变** |
| cohort | 同一 first256 被四个 CUDA process realization 使用 | 三个互异 raw1200→first1000 parse-success cohort | **改变** |

当前 prompt 身份：system SHA `4b00f195...9050`、user SHA
`79e7e3bf...d44e`、formatted SHA `ec0aed37...c3f`、token-ID SHA
`d4b78b8b...17cb`。单独的 cohort identity 审计还显示，历史 first256 中
254 个可比 formula identity 只有 19 个出现在当前 3000-plan union；加入
prototype 后只有 4/254 重合。四个历史 repeat 是**同一 Plan cohort 上的
进程实现重复**，并非四个独立 Plan 样本，因此历史高值不能视为 planner
总体分布的四次独立验证。

## 历史 refined256 的当前 S.U.N./cache 重评

Replay run:
`20260811_h1_r03_refined256_current_sun_cache_replay_env_repair_v2`。
Repeat array `31650` 与 assembly `31651` 全部 `COMPLETED 0:0`。它不运行
Planner、R03、model_494 或 CHGNet，只重用 byte-frozen generation 与
relax-energy cache，离线重算 novelty/uniqueness/hull。

| Repeat | Gen/recon | Novel | Unique | Novel∩unique | Hull eval/unknown | strict /248 | strict /256 | meta /248 | meta /256 | old→current |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 248/256 | 227 | 248 | 227 | 218/9 | 28/248 = 11.29% | 10.94% | 122/248 = 49.19% | 47.66% | identical |
| 1 | 248/256 | 224 | 248 | 224 | 215/9 | 31/248 = 12.50% | 12.11% | 123/248 = 49.60% | 48.05% | identical |
| 2 | 248/256 | 226 | 248 | 226 | 217/9 | 29/248 = 11.69% | 11.33% | 125/248 = 50.40% | 48.83% | identical |
| 3 | 248/256 | 227 | 248 | 227 | 218/9 | 29/248 = 11.69% | 11.33% | 126/248 = 50.81% | 49.22% | identical |

MP cache 对历史 cohort 的 224 个 chemical systems 全覆盖：已有 132，唯一
一次补齐查询 92/92，transport retry=0，最终 cache SHA
`ede6921b...140d`。headline 分母严格保持 legacy reconstructed=248；
all256 只作保守 secondary，evaluated-only 只作诊断。pooled1024 仅描述，
不假设四组为独立 Plan 样本。

## 当前 V4 完整 pre/post stage 数据

以下 12 份 stage report 均为 `_SUCCESS`。`strict`/`meta` 单元均写成
`numerator / reconstructed = headline；/1000 = secondary`。

| Arm | Rep | Stage | Gen/recon | Comp/struct/joint | COV-P/R | Novel/unique/N∩U | Hull eval/unk | strict | meta |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| R03 | 0 | pre | 969/969 | 838/461/405 | 94.9/26.88 | 968/969/968 | 939/29 | 26/969=2.68%; 2.60% | 130/969=13.42%; 13.00% |
| R03 | 0 | post | 969/969 | 838/968/837 | 96.6/95.40 | 874/963/870 | 846/24 | 70/969=7.22%; 7.00% | 449/969=46.34%; 44.90% |
| R03 | 1 | pre | 978/978 | 835/470/410 | 95.2/26.91 | 978/978/978 | 950/28 | 22/978=2.25%; 2.20% | 122/978=12.47%; 12.20% |
| R03 | 1 | post | 978/978 | 835/977/834 | 97.4/94.88 | 865/975/864 | 842/22 | 65/978=6.65%; 6.50% | 435/978=44.48%; 43.50% |
| R03 | 2 | pre | 977/977 | 818/484/413 | 96.3/38.44 | 976/977/976 | 945/31 | 22/977=2.25%; 2.20% | 138/977=14.12%; 13.80% |
| R03 | 2 | post | 977/977 | 818/976/816 | 97.1/95.04 | 876/976/875 | 851/24 | 60/977=6.14%; 6.00% | 441/977=45.14%; 44.10% |
| B3 | 0 | pre | 981/981 | 853/497/434 | 96.2/53.57 | 975/981/975 | 946/29 | 20/981=2.04%; 2.00% | 129/981=13.15%; 12.90% |
| B3 | 0 | post | 981/981 | 853/979/851 | 98.0/95.15 | 877/976/873 | 847/26 | 70/981=7.14%; 7.00% | 450/981=45.87%; 45.00% |
| B3 | 1 | pre | 978/978 | 832/507/441 | 96.3/47.49 | 973/978/973 | 945/28 | 11/978=1.12%; 1.10% | 118/978=12.07%; 11.80% |
| B3 | 1 | post | 978/978 | 832/977/831 | 97.5/95.42 | 865/974/865 | 840/25 | 58/978=5.93%; 5.80% | 444/978=45.40%; 44.40% |
| B3 | 2 | pre | 983/983 | 821/534/453 | 96.4/55.12 | 977/983/977 | 948/29 | 21/983=2.14%; 2.10% | 148/983=15.06%; 14.80% |
| B3 | 2 | post | 983/983 | 821/981/819 | 97.8/95.51 | 884/983/884 | 860/24 | 60/983=6.10%; 6.00% | 414/983=42.12%; 41.40% |

model_494 几乎把 structure-valid 从 46.1–53.4% 提高到 96.8–98.1%，
也把 meta S.U.N. 提高约 29–33 个百分点。R03 与 B3 post 的范围高度
重叠，所以当前残余下降不能简单归因于 B3。

## V4 工程状态与可解释边界

R03 array `31583` 与 B3 array `31584` 的六个任务及 12 个 stage 均
`COMPLETED 0:0`/`_SUCCESS`。Assembly `31585` 在生成正式统计表时因
`OverflowError: int too large to convert to float` 失败 `3:0`；因此上表是
逐 stage 的完整 point estimate，但 V4 没有通过正式 assembly，也没有
三重复 bootstrap/McNemar 终态。按 fail-closed 顺序门，native1000 未提交，
本次审计没有修复、重投或触发它。

这组证据能排除 evaluator/cache 与 active prompt 文本，但不能仅凭观察性
cohort 差异把全部 post-refine gap 因果归给某一个 Plan 字段。若要进一步
闭合因果，下一实验应固定 P0、R03、model_494 与所有 downstream seeds，
只交叉切换 legacy global-stream first256 与 stateless/parse-success cohort；
本报告不授权该新生成实验。

## 证据

- Replay terminal report SHA: `b4f449c4...9ea2`。
- Replay markdown SHA: `bd2319e3...94e`。
- Source manifest SHA: `2626de50...dae`。
- 当前 exact S.U.N. runner SHA: `e952795c...de`；MP cache SHA:
  `ede6921b...140d`。
- 最小回传证据位于
  `evidence/h1_r03_refined256_current_sun_cache_replay_v2/`；V4 的 12 份
  stage report 位于其 `v4_context/stages/` 子目录。
