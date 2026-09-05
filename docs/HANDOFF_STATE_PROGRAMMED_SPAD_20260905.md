# 新 session 交接：当前 Llama–DLM 双目标 SPAD

日期：2026-09-05。最后远端只读状态快照约为 **21:21 Asia/Shanghai**。

## 0. 先读这一页：当前停止边界与最重要的事实

用户因为当前 session 卡顿要求：**停止推进、写交接文档；A800 及其上已有任务不用干预。**

- 已暂停十分钟自动化 `llm-dlm-sun-24h`，没有删除它。
- **没有执行 scancel，没有取消/重启/干预远端任务，也不主动关闭现有 SSH/tmux。**
- 最后查询时，本轮 `spad-state-*` 没有 active/pending job：39853 warmup 和39857训练条件准备均已自然完成。查询只覆盖本轮名称，不代表集群其他任务或资源状态。
- **新方案还没有 SUN 结果。**已完成的是接口预检、完整 MP20 warmup 和训练条件准备；双目标能量后训练与正式生成/评估尚未完成。
- 新 session 不要重跑已完成的 39853/39857，不要恢复旧 PMTR、K10、G2、BTRD。
- 只有用户明确要求继续后，才恢复实施或自动化；旧24小时目标不会因换 session 自动重新计时或延期。

当前最短阅读顺序：本文件 → [执行清单](teacher_feedback_unified_v1/04_EXECUTION_CHECKLIST.md)顶部 → [17当前方案](teacher_feedback_unified_v1/17_STATE_CONDITIONED_TERMINAL_BASIN_PLAN.md) → [18审计记录](teacher_feedback_unified_v1/18_DUAL_OBJECTIVE_REVIEW_AND_DECISIONS.md)。

## 1. 工作树与代码位置：不要用错仓库

| 对象 | 实际位置/状态 |
|---|---|
| **当前本机工作树** | `D:\codex_work\ai4s\DLM_llama_programmed_basin_closure` |
| 当前分支 | `codex/llama-programmed-basin-closure` |
| 截止停止时最后已推送的代码版本 | `a5fb9cf` |
| GitHub | `https://github.com/YWDDLiang/DLM/tree/codex/llama-programmed-basin-closure` |
| **当前远端 checkout** | `/public/home/jiaosz/ywliang/ai4s/.sscd_state_programmed_20260905_v1` |
| 旧远端 checkout，保留不要覆盖 | `/public/home/jiaosz/ywliang/ai4s/.sscd_basin_closure_ab9ff21` |
| App 原始 cwd，**不是本轮工作树** | `D:\codex_work\ai4s\diffsion_language_model_meets_diffusion` |

本机每次执行命令应显式指定当前工作树，不依赖 App 默认 cwd。旧 `DLM_unified_scientific_decoding`、G2、Force-Score 等 worktree 都是历史资产，不是本轮写代码的位置。

远端统一产物根目录，拼写中的 `diffsion` 不要修正：

```bash
ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/proposal_realization_candidates_20260826/grounding
SOURCE=/public/home/jiaosz/ywliang/ai4s/.sscd_state_programmed_20260905_v1
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
```

代码版本演进：

| Commit | 本轮内容 |
|---|---|
| `6dc53b5` | 周期状态输入、程序化完整路径、双目标求解器、预检入口与测试 |
| `a8c6c60` | full-MP20 source-matched 协同/闭合训练状态 |
| `bb9eebf` | warmup trainer/Slurm；修复预检必填 `min_lattice_rad` 参数 |
| `a5fb9cf` | 冻结 Planner 对训练组成预测 soft Plan 和程序，不重新采样化学组成 |

以上代码均从**本机** commit/push。远端只是 fetch/checkout，没有从远端 push。39853启动代码为 `bb9eebf`；运行中后续 checkout 只新增准备入口/文档，没有改它正在使用的训练实现。

本交接文档及清单的停止状态更新先保留在本机，没有继续做GitHub上传。只clone最后已推送版本不会自动获得这份交接或§6.1的未提交草稿；跨机器接续时应由用户一并传递。

## 2. 如何正确使用 A800

### 2.1 两层连接，先分清当前 prompt

本机 Windows PowerShell：

```powershell
ssh -tt -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o ConnectTimeout=15 starteam5090
```

到达5090后，prompt通常是 `ywliang@victory:~$`。**此时不是A800。**先查看既有会话：

```bash
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} dead=#{pane_dead} cmd=#{pane_current_command}'
tmux capture-pane -pt ssha800:1.0 -S -8
```

确认 `ssha800:1.0 dead=0 cmd=ssh`，而且 capture 中已有清晰内层 shell prompt、没有前台命令尚未结束，再进入：

```bash
tmux select-window -t ssha800:1
TERM=xterm-256color tmux a -t ssha800
```

进入后应看到类似 `(base) [jiaosz@mgt .sscd_state_programmed_20260905_v1]$`。这是 A800 集群登录端；GPU计算必须通过 Slurm。

- 优先使用既有 `ssha800:1.0`。`ssha800_2` 不重连，也不要擅自操作备用会话。
- 不新建内层 A800 SSH，不因为外层连接失效就重连内层或重提 job。
- 本轮外层持久连接曾使用工具 session 8297；这类编号不保证跨 session 可用。`Unknown process id` 不表示 A800 作业失败。重新建立**外层** starteam5090 连接后，仍 attach 既有 tmux。
- 若出现 `open terminal failed: terminal does not support clear`，使用上面的 `TERM=xterm-256color`。
- tmux 屏幕可能保留旧 PMTR traceback，不能把屏幕残留当作当前失败；以 job ID、sacct、对应 run 的日志/marker 为准。

### 2.2 资源与操作边界

- 最新额度：**最多6张A800，每卡4CPU，总24CPU；最多2个job**。
- 本轮采用4卡训练、global batch16；另外2卡可以做数据/评价。独立生成/标注可六卡分片，也允许奇数可用卡。
- 不能为了凑六卡悄悄把 global batch16 改成18；4卡训练已经验证，微批大小与累积必须正确记账。
- 暂停 MatterSim 和独立 AR 路线；`public105/488` 只读。
- 禁止无关 job、`nvidia-smi`、独立 CUDA 探针；不要在登录节点做 GPU 测试。真实模型预检与训练都放进对应 Slurm allocation。
- 不用泛化的 `scancel -u` 等操作。用户当前已明确“不用管A800及在跑的”，这次交接没有取消任何任务。

获准继续后，优先只读核对本轮相关任务：

```bash
sacct -j 39850,39852,39853,39857 --format=JobID,JobName%30,State,Elapsed,ExitCode -P
squeue -h -n spad-state-preflight,spad-state-warmup,spad-state-conditions
```

### 2.3 环境、部署与常见坑

- 始终用 `PY` 的绝对路径。正确位置是 `/public/home/jiaosz/miniconda3/...`，**不是** `/public/home/jiaosz/ywliang/miniconda3/...`。prompt中的 `(base)` 不代表训练环境。
- 远端系统 Git 较旧：`git remote get-url` 不支持，不能假定 `git worktree` 可用；用 `git config --get remote.origin.url` / `git remote -v`。本轮新 checkout 通过 `git clone --shared --no-checkout OLD NEW` 创建，再 checkout明确commit。
- GitHub上传只能在本机。远端可在已有tmux里使用 HTTPS fetch：`git fetch https://github.com/YWDDLiang/DLM.git codex/llama-programmed-basin-closure`。本轮fetch已成功。
- 非Git传输若需要，本机SCP统一打包，最多五分钟一次；本轮没有用SCP，没有base64传输。
- 不在正在跑任务时随意修改它依赖的模块。新run用明确代码版本；今后可按run保存小型代码快照，避免运行时checkout漂移。
- 远端环境没有 pytest，但有 unittest；不要因此误判模型失败或盲目升级整个Conda环境。
- 交互命令要等prompt回来；有依赖的短命令可用 `&&` 一次发送。读日志尽量只输出十余行，tmux重绘/终端高度会让长输出难以观察。
- `run` 不覆盖；工程恢复使用新job/run，并保留旧失败记录。

## 3. 我们当前到底保留什么、要优化什么

唯一主线：

```text
C3FD chemical support
→ 冻结的 Planner-Llama：composition + compact Plan + species program
→ canonical 7+4N SPAD 构造
→ 完整旧周期状态输入 + cell/多原子协同事务
→ 反向物种闭合
→ 单条原生晶体
→ 可选固定 model494 tau800（系统保底，单列）
```

- Llama输出的是**物种排列**。编译器映射到原子槽位、锚点、非连续构造顺序和反向闭合；不是实时在线Llama控制器，不是端到端同时训练Llama与DLM。
- DLM使用双向上下文保留未来原子、修订早期部分；新状态输入还保留被mask变量的旧数值，避免只看到MASK。
- 联合事务修改cell六分量与按程序/周期邻域选出的约半胞XYZ，最终整体验收；替换单独cell闭合，不再额外叠加一次旧cell修复。
- 保留有效性基础，**不恢复G2、BTRD、PMTR连续头、旧局部K10分类loss或在线MLIP选优**。
- 原生推理没有CHGNet/MLIP，没有rerank或best-of-K；K4只用于训练数据。

科学目标：

\[
A=e(x)-e(R(x)),\qquad B=e(R(x))-h(c).
\]

A是原生—弛豫终态能差，B是终态盆地能量相对hull。新经验teacher要求固定组成权重下平均A/B都改善，并最小化分布改动；不是固定0.5:0.5相加。

双目标求解器已实现，但尚未用本轮真实完整路径数据训练。经验teacher双降不是学生或SUN必然双升的保证；A也不直接等于force/stress/优化步数。

**指标口径：**仓库所谓 raw SUN 仍包含公共CHGNet评价弛豫，只是没有先经过model494。旧评估先在输入结构上算N/U，再用弛豫能量判断稳定，不能写成已经评估了非重复的低能极小值集合。

## 4. 本轮实际完成到哪里

### 4.1 最后已核实的Slurm终态

| Job | 内容 | 资源 | Slurm终态 | 用时 | 含义 |
|---|---|---|---|---|---|
| 39850 | 第一次真实模型预检 | 2GPU/8CPU | FAILED 1:0 | 00:01:50 | 五步梯度已有限；采样约束漏传必填参数，工程失败 |
| 39852 | 参数原值恢复后的预检 | 2GPU/8CPU | COMPLETED 0:0 | 00:01:57 | 真实模型注入/反传/有限回放检查完成 |
| **39853** | **full-MP20 state warmup** | **4GPU/16CPU** | **COMPLETED 0:0** | **00:23:18** | **不用重训；后续从其完成产物接续** |
| **39857** | **冻结Planner训练条件准备** | **2GPU/8CPU** | **COMPLETED 0:0** | **00:00:52** | **不用重新采组成；优先核对并复用产物** |

这四项按主job耗时折算约 **1.71 allocated A800-hours**，不是硬件利用率统计。没有测出或声称全程70% GPU利用率。

### 4.2 预检的准确覆盖范围

39852报告确认：

- 两rank各五步loss/gradient有限；大多数单步约0.30秒。
- 同一输入下零初始化conditioner的最大logit差为0。
- trainable LoRA：14,680,064参数；conditioner：1,267,200参数；embedding/head等大表冻结。
- 启用的native checkpoint模块为 `base_model.model.model`。
- 完成一次真实联合/闭合trace，记录51个sampled scalar decisions。
- **只对前3个真实decision做了回放比较，误差0；不是51个都逐项验过。**
- 保存后在同一个模型实例重载了conditioner并比较；**尚未完成新进程重新加载整套base+LoRA+conditioner后的数值对照**。未来正式采样入口必须正确加载两部分。

预检产物仅是工程资产：`eligible_policy=false`，不得当最终模型选用。

### 4.3 Warmup实际设置

- 初始化：已完成的 `spad_basin_closure_ce_39700`，不是失败PMTR。
- 全MP20 train：27136条；一source一修订状态、一轮，global batch16、预定1696 updates。
- 4 ranks，每rank batch2，gradient accumulation2。
- 只训练已有LoRA和conditioner；LR1e-5、warmup100、cosine、grad clip1。
- teacher Plan及原数据记录的program来源保留；warmup中不少program是contact-tree teacher，不能说它们全是Llama预测。
- XYZ_100周期别名在warmup目标中规范为XYZ_000；化学组成不变。
- 协同源由小幅cell strain和活动区坐标扰动产生，旧快照不替换为clean target；若坐标扰动改变选区，记录并仅退回坐标扰动，不丢source。

成功Slurm及脚本约定指向完整1696步终点。由于用户随后要求不再操作A800，**停止后没有逐文件读取TRAIN_FINAL、manifest或完整验证新进程加载**；接续先只读核实，不重复训练。

### 4.4 训练条件准备实际设置

旧full-MP20 pointer导出器复制teacher soft Plan，只预测物种程序。为避免把teacher字段当predicted，本轮新增准备入口：

- 在已有typed train记录中，按固定seed20260905选1024个不同精确组成条件；数据可用性/stratum检查不读能量。
- 化学组成及已有typed化学transcript固定，不进行composition采样。
- 冻结C3FD + Llama PoE预测LS/SG/VPA，使用terminal MAP值。
- Pointer使用**这些预测soft IDs**，不是teacher soft IDs。
- 输出规范native prompt，保留结尾换行；标记train、outcomes_read=false、composition_resampled=false。

该池是1024个train条件，不是全部27136组成的on-policy训练。具体typed覆盖和排除数量请读实际manifest；没有在本次交接中猜测。MAP补齐也不等于复现生产Planner的全部随机采样分布。

## 5. 关键产物与模型路径

以下路径来自已成功job的脚本约定；接手时核对文件与对应marker：

```text
$ROOT/runs/spad_state_preflight_39852/
  _SUCCESS
  preflight.out / preflight.err
  check/PREFLIGHT.json
  check/reload_check/                 # 工程检查，非eligible模型

$ROOT/runs/spad_state_warmup_39853/
  _SUCCESS
  POLICY_PATH
  train.out / train.err
  train/TRAIN_FINAL.json
  train/training_config.json
  train/training_log.jsonl
  train/checkpoints/step-1696/
    adapter_config.json
    adapter_model.safetensors
    tokenizer files
    periodic_state_config.json
    periodic_state.pt

$ROOT/runs/spad_state_train_conditions_39857/
  _SUCCESS
  prepare.out / prepare.err
  conditions/manifest.json
  conditions/plans_for_dlm.jsonl
  conditions/conditions.rank0.jsonl
  conditions/conditions.rank1.jsonl
```

其他保留资产：

| 资产 | 路径 |
|---|---|
| DLM共享底座 | `/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct` |
| 原closure参考 | `$ROOT/runs/spad_basin_closure_ce_39700/train/checkpoints/step-1696` |
| fullMP20 warmup源 | `$ROOT/data/spad_basin_closure_sft_v1_20260904/train.jsonl` |
| pointer训练记录 | `$ROOT/data/spad_species_pointer_v1_20260903/train.jsonl` |
| 冻结Llama底座 | `/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B` |
| 冻结Planner | `$ROOT/runs/c3fd_llama_typed_planner_39051/train_seed85017/final` |
| 冻结pointer | `$ROOT/runs/spad_species_pointer_39511/train_seed86017/pointer_state.pt` |
| C3FD | `$ROOT/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt` |
| C3FD词表 | `$ROOT/data/c3fd_semantic_v21_step1b_20260828/vocabulary.json` |
| 旧固定256开发Plan | `$ROOT/cohorts/spad_prospective_seed23_256_v1_20260903/plans_for_dlm.jsonl` |
| 旧4104池，仅历史诊断/条件资产 | `$ROOT/cohorts/spad_basin_scale4104_train_v1_20260905` |

旧4104局部候选及K10 label **不是**本轮联合事务的完整路径teacher，不能直接改名复用。

## 6. 已实现代码与未完成部分

| 文件 | 当前功能/状态 |
|---|---|
| [periodic_state_conditioning.py](../src/crystal_dlm/periodic_state_conditioning.py) | 真实metric/周期邻域/物种程序的FP32小型编码器；两输出投影零初始化 |
| [state_conditioned_model.py](../src/crystal_dlm/state_conditioned_model.py) | 显式 `inputs_embeds` 注入；旧数值状态解码；保存/加载conditioner；冻结大表 |
| [programmed_path_runtime.py](../src/crystal_dlm/programmed_path_runtime.py) | 等N批处理构造、联合事务、反向物种闭合、完整attempt事件和scalar回放 |
| [state_revision_data.py](../src/crystal_dlm/state_revision_data.py) | source-keyed warmup状态和可复现corruption |
| [state_training.py](../src/crystal_dlm/state_training.py) | 混长训练batch/context和native checkpoint设置 |
| [basin_path_objective.py](../src/crystal_dlm/basin_path_objective.py) | 双目标经验teacher凸求解和lazy Torch HT NLL；**不是完整trainer** |
| [preflight入口](../src/scripts/preflight_state_programmed_spad.py) / [222](../slurm/222_state_programmed_preflight.sbatch) | 已运行39850/39852，不要重复 |
| [warmup入口](../src/scripts/train_state_conditioned_spad.py) / [223](../slurm/223_state_programmed_warmup.sbatch) | 已完成39853，不要重复 |
| [条件准备入口](../src/scripts/prepare_state_train_conditions.py) / [224](../slurm/224_state_train_conditions.sbatch) | 已完成39857，冻结Planner，无新Planner训练 |

**尚未完成：**

1. 批量完整路径生成CLI/Slurm/汇总入口：核心sampler有了，但尚未接成1024×K4生产任务。
2. 离线终态标注入口的测试与真实包验证：本地有草稿，见下一节。
3. 将完整trace、teacher权重、每phase抽样决策接成真正的双目标训练器；包含正确HT/minibatch归一化和CE anchor。
4. 一次train-only数据刷新与第二轮训练。
5. 新模型固定256、tau800及最终1000有效CIF的SUN结算。

### 6.1 未提交草稿，换 session 时不要丢

本机 `git status` 中有四个untracked文件：

- `scripts/label_programmed_paths.py`：主agent刚写的离线CHGNet标注草稿。
- `tests/test_label_programmed_paths.py`：对应测试，**目前collection失败，未通过测试**。
- `src/scripts/apply_pmtr_fixed_bodies.py`：此前已经存在的未提交文件，原样保留。
- `tests/test_apply_pmtr_fixed_bodies.py`：其测试，原样保留。

labeler测试目前的明确问题：`from scripts.label_programmed_paths ...` 被 `src/scripts` 包遮蔽，无法导入根目录 `scripts/` 下的文件。下一步用现有工程的 importlib 文件加载模式修复测试入口，再真正跑mock测试；**不能把这次collection错误当作labeler功能已验证**。

草稿不在已推送的 `a5fb9cf` 中。新session应继续使用本机现有工作树；只clone GitHub会缺失这些文件。它们本次没有部署或提交Slurm。

## 7. 已验证到什么程度

- 本机一个明确的组合suite：105 tests PASS；之后必填参数AST测试PASS、条件选择2 tests PASS。不要把多次重叠suite相加宣称更多独立测试。
- 远端18个conditioner unittest、6个runtime直接调用测试、6个objective unittest PASS；条件准备的2个选择测试和必填参数1个测试也PASS。
- 真实GPU39852的有限覆盖见§4.2；full warmup39853和准备39857的Slurm终态已核实。
- **labeler没有通过测试，完整生成CLI和全路径训练器没有实现，新SUN没有计算。**

本机已准备好隔离CPU测试环境，不要再从头安装：

```powershell
# 在 D:\codex_work\ai4s\DLM_llama_programmed_basin_closure
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='1'
& '.\.venv\Scripts\python.exe' -m pytest tests/test_state_programmed_runtime.py tests/test_periodic_state_conditioning.py tests/test_basin_path_objective.py -q
```

该venv为Python3.12.14 / Torch2.8.0+cpu / NumPy2.3.5 / SciPy1.16.2 / pytest8.4.2。默认 `C:\Python314\python.exe` 没有Torch，别用错。远端真实训练用既有Python3.10环境，不能以本机版本代替远端验证。

## 8. 接手后最容易踩的接口坑

1. **必须加载conditioner。**普通 `load_model_and_tokenizer` 只加载base+LoRA；新采样还必须创建 `StateConditionedDLM`、读取 `periodic_state_config.json` 并调用 `load_state_conditioner(checkpoint)`，且forward真正传入 `geometry_context`。否则会静默丢掉新方法。
2. **新进程加载检查还没做。**不要把预检同实例reload当成整个新checkpoint已端到端验证。
3. **旧状态不是target。**完整source快照与当前masked/staging canvas必须分开；梯度不能沿用PMTR的detach；不能用随下一forward变化的全局hook上下文。
4. **冻结embedding/head。**加载完整权重后调用 `set_state_lora_trainable`；不要让旧PEFT `modules_to_save` 又把十亿级大表放进optimizer。
5. **已验证native checkpoint方式。**`enable_native_checkpointing` 调用LLaDA的 `set_activation_checkpointing("whole_layer")`。不要假定泛用HF方法与该底座一致。
6. **联合验收和回滚。**整体事务不可顺序拼旧cell/site函数冒充；实际无持久几何cache，状态由快照重建。后续不要加跨batch的陈旧缓存。
7. **参考路径陷阱。**当前 `ProgrammedPathSampler.run(cooperative=False)` 会跳过cell步骤，并不等于旧 `cell→reverse-species` closure参考。不要直接把它当原参考；应调用原参考路径或明确补齐cell-only分支并验证。
8. **采样CLI尚需严格prefill。**按program编译的canonical slots填N/E，验证body与Plan；不要按可能非canonical的元素出现顺序随意填槽。
9. **真实概率规则。**温度0.7、schema/alias/PBC处理顺序、逐scalar条件必须一致。回滚前X/Y已采样则计概率；未采样Z和确定性rollback不额外计概率。
10. **不能复用旧K4分类loss。**学生概率分母是完整合法token支持。旧 `potential_closure.py` 的组内log-softmax和3/6长度除数不属于新目标。
11. **零权重。**旧SFT collator的 `value or 1.0` 会把0变1；路径训练不能用它表达未验证/零权样本。
12. **HT与batch尺度。**每path每遍分层抽六个真实决策，必须按包含概率校正，不能再除轨迹长度；dataset均值与每condition目标之间的系数也要一致。整法预算约150016 scalar状态/9376 updates，不是348-step小头训练。
13. **prompt格式。**当前训练与新准备文件都以 `prompt.rstrip()+"\n"` 作为body前缀；不要在推理时无意删掉末尾换行。词表是101编码坐标值/0.01 fractional网格，不是1000-bin；000/100为周期别名。
14. **Plan来源。**warmup的teacher Plan/contact-tree程序、39857的预测soft Plan/Llama程序必须分开记。只允许把schema相同说成接口相同，不能说两阶段分布完全一致。
15. **CHGNet单位。**`predict_structure()['e']`是每原子能量；trajectory energies为总能，需除N。raw stress通常GPa，ASE trajectory stress为eV/Å³，转换常数约160.21766208。
16. **终态验证。**有限能量或K10不等于收敛；labeler拟用FIRE/FrechetCellFilter、变胞、Fmax≤0.1 eV/Å、最大stress分量≤0.5GPa、max500。记录真实optimizer停止状态，缺失不能编造为true。
17. **目标与证据。**输入N/U、终态能量、Stable、SUN、force/stress分别报告；不能只凭loss或SUN一项变化声称已经学会稳定。

## 9. 用户确认继续之后的最短工作流程

1. 只读核实39853/39857的marker、TRAIN_FINAL/manifest及新checkpoint文件；确认没有重复任务。
2. 修复并测试labeler草稿，不进行新的方法设计；保持CHGNet仅离线使用。
3. 实现完整路径采样入口，正确加载39853；复用39857的1024冻结条件，每条件4条独立路径，原始失败和重数全部保留。
4. 保存可重建的attempt事件，再做终态标注。先128组用于接口/协议诊断，不按能量换掉它们，继续完成同一1024组。
5. 对完整验证池求A/B经验teacher；若无共同可学收益如实分析，不伪造winner或无限加epoch。
6. 实现真实全路径训练：LoRA+conditioner，冻结大表，部署无dropout的likelihood，正确HT；两遍后最多一次train-only刷新再两遍。
7. 一个closure参考与一个最终方法，固定256配对；原生物理量、共同弛豫Stable/N/U/SUN和tau800保底分开。
8. 方法锁定后才做独立1000有效CIF；按源顺序、仅根据可解析性补取，披露总请求和失败，不能按能量/稳定性挑补。

不再追加多seed、多τ、2×2、多个方法分支。实现、测试、部署、监控、归档由主agent直接做；**只有大规模审计/调研才可使用智能体**。本轮编码子任务已结束/关闭，没有需要等待的回传。

## 10. 自动化、凭证与交接后的权限

- 自动化ID：`llm-dlm-sun-24h`；名称：`LLM–DLM 双目标 SUN 24h`；原频率10分钟；**现在PAUSED**。
- 旧 `h1-a2-stability-ccfd` 已不存在，不要创建重复监控。
- 原努力截止：2026-09-06 19:19 Asia/Shanghai。用户当前停止指令优先；恢复时需明确剩余预算/截止，不自动续期。
- 新session不要把自动化继续绑定旧session并放任两边同时提交；如需迁移到新任务，先暂停旧目标、确认唯一监控归属。
- MP key曾由用户提供，但**不写在本交接、Git、命令、日志或自动化中**。`MP_API_KEY_FROM_USER_THREAD_CONTEXT`只是来源说明，不是实际环境变量或可用key。
- 优先复用兼容的官方phase-diagram缓存。新查询仅在明确授权且安全取得凭证后以临时子进程环境使用，不把凭证带给智能体，不声称可以删除原始聊天。
- 用户允许尽力提点，不允许把历史public结果改名为新结果、伪造10/50、按稳定性挑样、倒灌test或改分母不披露。

**交接后的默认动作是等待用户指示，不是继续运行本文件里的计划。**

## 11. 历史文档如何读

- [17当前计划](teacher_feedback_unified_v1/17_STATE_CONDITIONED_TERMINAL_BASIN_PLAN.md)、[18审计](teacher_feedback_unified_v1/18_DUAL_OBJECTIVE_REVIEW_AND_DECISIONS.md)：本轮方法与工程注意事项。
- [04清单](teacher_feedback_unified_v1/04_EXECUTION_CHECKLIST.md)：只以顶部当前状态为执行入口；其后K10历史勾选不是新任务。
- [GPT6审计交接](GPT6_AUDIT_HANDOFF_20260905.md)：完整历史迭代/失败原因；其中“PMTR待训练”等状态已经过期。
- `WORKFLOW.md`、R03、G2、BTRD、早期SUN冲刺文档是历史；不要直接运行它们的默认采样/训练/凭证门禁。

本轮停止时没有新SUN可汇报；已有历史BS、K10、tau800数字必须保持原来的模型、cohort和评价端点，不能当作39853的效果。
