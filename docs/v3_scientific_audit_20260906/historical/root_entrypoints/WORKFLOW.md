# 个人工作流：环境、训练与推理

## 1. 环境

`environment/`当前是占位。待A800确认后，复制Conda导出并在
`docs/ASSET_TRANSFER_LEDGER.md`记录来源。个人环境名和路径可以在配置中修改。

## 2. 数据

完整MP-20 split与论文repo相同；个人repo允许把数据放在其他相对位置，并通过
`paths.mp20`配置。原始R03 256 Plans、parsed Plans和ordinal seed ledger均为
独立资产。

## 3. 训练

个人配置可以覆盖Planner/DLM/Diffusion路径、明确标为release/custom的seed、
batch size、学习率、训练轮数以及Slurm资源。历史未记录seed不会自动伪装成
历史值。

## 4. H1-A2推理

默认采用1,200个Planner attempts并选择前1,000个body success进入refiner。
个人repo允许修改attempt数、refined target以及Safe-axis开关。

这是fully de novo路线：Plan在推理时来自learned Planner。训练阶段从MP-20提取
Plan label不改变这一点；若推理时直接使用MP-20-derived Plan，则该运行应标记为
empirical-Plan control，而非de novo Plan generation。

## 5. R03快速复现

默认使用冻结256 Plans、冻结ordinal ledger、Safe-axis开启、4次process replay。
如果Planner checkpoint存在，可设置`resample_plans=true`，使用seed `17029`
重新采样。

冻结Plan路线只复现`p(B|P)`和refiner，不把四次process replay解释为新的Planner
样本，也不承担fully de novo headline。

## 6. 评估

Materials Project key只通过`MP_API_KEY`环境变量提供。没有key时停止S.U.N.
阶段，但保留生成、refinement、Direct、novelty和uniqueness结果。
