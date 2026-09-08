# 本轮数据归档索引

归档及失败权重清理已完成。保留清单共 **91,296 个文件**，包含本轮数据、源码快照、模型元数据和实验记录，共 **8585118776 字节**。压缩包为 **3882030876 字节**。原数据目录仍可直接读取。

服务器原始运行目录：

`/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/proposal_realization_candidates_20260826/grounding/experiments/expert_self_edit_20260908/run_20260907T180747Z`

服务器归档目录：

`/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/proposal_realization_candidates_20260826/grounding/experiments/expert_self_edit_20260908/run_20260907T180747Z_data_archive_20260908`

归档目录内的核心文件：

| 文件 | 内容 |
|---|---|
| experiment_data.tar.gz | 全部本轮非模型权重文件的完整压缩归档 |
| DATA_MANIFEST.json | 逐文件相对路径、字节数、SHA256，以及模型权重的保留/移除清单 |
| ARCHIVE_VERIFIED.json | 逐个压缩包成员及当前原文件均已核验的回执 |
| CHECKPOINT_REMOVAL_PLAN.json | 清理前全部模型权重的SHA256和明确保留标记 |
| CHECKPOINT_REMOVAL_LOG.jsonl | 实际逐文件移除记录 |
| CLEANUP_FINAL.json | 删除后数据、保留权重和文件清单再次核验的结果 |

压缩包SHA256：`d3f3d365f7ef6c467cd18858a9f0db3676e8d0bc0ed57bf1187ff6067fcc0f2a`。
逐文件清单SHA256：`d7667b74d988e0cf1478e7814f36cb97bfe54451e897b206b9aa2a14ea709bd0`。
本地可直接查阅[归档核验回执](ARCHIVE_VERIFIED.json)、[清理结果](CLEANUP_FINAL.json)和[权重清单](CHECKPOINT_REMOVAL_PLAN.json)。

## 新idea可以复用的数据

下表路径均相对于服务器原始运行目录；压缩包保留同样的目录结构。

| 类别 | 相对位置 | 用途及状态 |
|---|---|---|
| 原始Planner/B0/教师生成 | collection_retry1、collection_expanded/shard_0至shard_5 | 原始rich Plan、初稿及连续教师输出，保留来源与请求ID |
| 编辑配对与标签 | headroom256/prepared及compiled、data_remaining/prepared及labels、expanded_training_data/prepared及labels | 已构造的旧态/目标配对和标签；可直接训练的部分以DATA_FINAL及_SUCCESS为准 |
| 学生态刷新与辅助数据 | student_proposal_refresh1、training_controls_refresh | 完整提议、原生标签、几何辅助数据及未完成阶段的原回执 |
| 本轮SUN64 | sun_v05_teacher64 | 64来源、固定四种教师动作加KEEP、稀疏教师轨迹、两组F800精修、全部原生/精修标签 |
| 本轮SUN评分 | sun_v05_teacher64_scores、SUN_TEACHER64_COMPACT.json、SUN_TEACHER64_ANALYSIS.json | 15个评分臂、逐来源完整结果、预登记门槛判断与分析源码SHA |
| 根因实验 | mechanism_* | 监督修正、等价目标、OLD画布、FIT8等实际生成、标签、分析和训练元数据 |
| 既有正式对照 | formal_baselines*、formal_editor_run、formal_editor_refined_eval | 原正式评测用途数据，保留其用途与完整请求分母 |
| 实现和运行身份 | code、各MANIFEST、submissions、logs | 不可变源码、实际配置、调度记录和失败日志 |

归档保留成功与失败尝试的原始状态。训练来源/source_split、主面板组成隔离和training_feedback/evaluation用途标记继续适用；目录存在不自动表示它是一份可直接训练的完整数据集。

## 权重保留与清理

实际正式评测使用的参考权重仍在原目录：

`expert_full_stage2/train/checkpoints/step-000800`

已清理其他 **24** 份失败或中间checkpoint中的 **95** 个权重/优化器文件，共 **56,964,189,806 字节（53.05 GiB）**。仅移除了adapter_model.safetensors、periodic_state.pt、expert_edit_modules.pt、training_state.pt。checkpoint配置、tokenizer、roundtrip_probe.pt验证数据和历史回执均保留，并有CHECKPOINT_REMOVED.json说明。

数据压缩包不包含模型权重；保留的第800步权重仍在服务器原路径，外部H1A2/B0/model494父模型未修改。本轮全部原数据在清理后重新核验通过。

本次没有新的DLM梯度训练。[科学结果和简要经验总结](RESULT.md)记录了有用信号与未通过的SUN门槛。后续等待用户的新idea。
