# A800资产迁移台账

本文件只存在于个人repo，可记录集群内部绝对源路径。当前先占位，待实际复制时填写。

| 资产 | A800源路径 | 本repo相对目标 | 状态 |
|---|---|---|---|
| Conda环境导出 | `/public/home/jiaosz/miniconda3/envs/diff_meets_diff` | `environment/` | 已确认，待导出 |
| official-MP环境 | `/public/home/jiaosz/ywliang/ai4s/.venvs/mp_api_0_45_13_emmet0_85_1_py310_v4_system` | `environment/official_mp/` | 已确认，待导出 |
| Planner base | `/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B` | `checkpoints/planner/base/` | 已确认，待复制 |
| Planner checkpoint | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final` | `checkpoints/planner/` | 已确认，待复制 |
| DLM base | `/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct` | `checkpoints/dlm/base/` | 已确认，待复制 |
| DLM checkpoint | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final` | `checkpoints/dlm/` | 已确认，待复制 |
| model_494 | `/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt` | `checkpoints/diffusion/model_494.pt` | 已确认，待复制 |
| CHGNet checkpoint | `/public/home/jiaosz/miniconda3/envs/diff_meets_diff/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar` | `checkpoints/chgnet/` | 已确认，待复制 |
| MP-20 train/val/test | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/reference/crysllmgen/data/mp_20` | `data/mp20/` | 已确认，待复制 |
| Planner训练JSONL | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/data/dlm_sft/mp_20_h1a2_rich_planner_noid_l3base` | `data/planner/` | 已确认，待复制 |
| DLM exact-length JSONL | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/data/dlm_sft/mp_20_r5_exact_length` | `data/dlm_r5_exact_length/` | 已确认，待复制 |
| H1-A2 raw/parsed Plans | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_planner1200` | `data/plans/` | 已确认，待复制 |
| R03 P0 raw/parsed source512 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260729_h1a2c_jointchem_v1/arms/P0/plan512` | `data/plans/` | 已确认，待提取first256 |
| R03 ordinal seed ledger | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1/data/attempt_ledger.jsonl` | `data/plans/r03_seed_ledger_256.jsonl` | 已确认256行，待复制 |
| B0训练日志/config | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260529_212834-r5c-exactlen-256` | `docs/a800_seed_evidence/` | 已审计：仅data_seed，global seed未记录 |
| Planner seed恢复配置 | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exact_retrain_recovery_v1/CONFIG.json` | `docs/a800_seed_evidence/` | 已确认train seed 17 |
| model_494训练代码/目录 | `/public/home/jiaosz/hengzhang/Code/crysllmgen-main` | `docs/a800_seed_evidence/` | 已审计：Torch default 1234，NumPy timestep未设seed |
| official evaluator source | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v6_login_official_resume` | `docs/a800_runtime_evidence/` | 已确认，待相对路径适配 |
| Safe-axis body source | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1` | `docs/a800_runtime_evidence/` | 已确认 |
| 4-repeat refine/eval source | `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1` | `docs/a800_runtime_evidence/` | 已确认 |
