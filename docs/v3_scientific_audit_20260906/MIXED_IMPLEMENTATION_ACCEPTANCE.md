# H-P33 实际实现验收

2026-09-06。数学与物理两轮审稿后，以下必要实现门已经实际关闭。正式新增两轮训练及SUN仍须完成，工程checkpoint全部`eligible_policy=false`。

| 门 | 实际证据 | 裁决 |
|---|---|---|
| 全量原CIF来源 | [40009](evidence/CONTINUOUS_SOURCE_IDENTITY_40009.json)，27136/9047唯一完整Q匹配 | PASS |
| train-only normalizer与数值 | [40013](evidence/GEOMETRY_NUMERICS_40013.json)，19项Torch2.4 CPU检查 | PASS |
| 共享模型CPU接口 | [模型报告](MIXED_GEOMETRY_MODEL_CPU_ACCEPTANCE.md)，14新模型+2旧加载 | PASS；mock边界明确 |
| 来源、全窗口与padding | [独立审查](IMPLEMENTATION_DATA_SCHEDULE_REVIEW.md)，另新增micro2窗口等价测试 | PASS |
| 真实4卡micro1/acc6 | [40045](evidence/MIXED_PREFLIGHT_40045.json)，4分48秒 | PASS |
| 真实6卡micro2/acc2 | [40060](evidence/MIXED_BATCH_PREFLIGHT_40060.json)，4分21秒 | PASS |
| 实际continuous CIF/graph/refiner输入 | [40058](evidence/CONTINUOUS_GRAPHS_40058.json)，8CPU/2分30秒 | PASS |
| float/Q采样及失败账本 | [11项采样测试](../../tests/test_mixed_geometry_sampling.py) | CPU PASS；正式256待运行 |

模型核心为`1111739bc5a590734b02a61f395faa59be605531`。`e1439cbc840d7fd743499ac9654fd20759a9f903`只扩展微批验收、图精度检查与事前规格，没有改变模型/训练数据/损失/正式trainer。实际环境Torch2.4.0+cu121；本机CPU额外检查为Torch2.8.0。

## 真实神经计算与累计

两个GPU作业都做了32个抛弃更新。T step0与加载V2相同，hidden-only与实际LLaDA最终ln_f在head/key相关bias及padding条件下差0。零头首步只有v/u头收到G梯度；打开头之后，G实际进入LoRA、state conditioner、geometry attention、输入增量表和time模块，T-only累计梯度增量0。所有rank训练参数hash一致。真实采样调用33次、最后时间为epsilon，末端晶格解码有限；保存恢复后T/v/u输出最大差均0。

DDP尾批与串行相同人口目标比较包含4真实source×2view和16个零权重padding。4卡整体相对L2误差2.45e−8；6卡整体0.4757%，最大片组geometry_attention为1.9010%，均在事前2%容差内。各组参考范数、误差和绝对floor保留在原始JSON；六卡结果不能改写成逐bit一致。

在N=1/10/20、t=.002的浮点扰动例中，FP32 encoder、BF16 hidden和打开后的FP32头都出现响应；未证明良好条件数、任意sub-bin分辨率或准确score。部分很小encoder变化对应较大hidden差异，保留原量级，不据此声称流光滑或物理校准。

## 精度与评测通路

40058实际调用远端CrysLLMGen `data_utils.process_one`，源码SHA256为`76f1fe865f50a8f6f4382081670bd530dc04c2156bcdff80c8561570994e90ab`，与已读vendor一致。4个例子包括非Q网格fixture及原val N=1/10/20。验收先核原结构到16位CIF，再按真实Niggli流程核graph Gram与按元素的周期坐标双射；进入真实ProposalDataset的FP32坐标误差最大2.911e−8。没有0.01-bin量化。

曾发现仅用全pair距离直方图与volume会漏掉N=1等体积晶格量化，提交前已补严格Gram检查。内置反例Gram误差0.289953 Å²被拒绝。该修复加强检查，没有改变共同物理阈值或refiner。

## 正式配置与时长依据

正式选6A800/24CPU、micro2/acc2/global24；相同12-source窗口各一T/G，完整27136来源新增两epoch，4524更新、108544实际状态，每epoch16个padding。验证seed202609063固定100来源×(1T+6G)，不用于选择中间policy。

六卡实测峰值16.273GiB，32更新benchmark129.45秒，剔除前4更新后的记录为3.2382秒/更新。小来源集含padding/缓存/首次kernel开销，不代表完整MP20吞吐；用该保守平均直接外推约4.07小时，加入初始化/验证、25%余量和半小时开销，取6小时Slurm上限。该上限在提交前冻结，不承诺完成时刻，不延长已提交任务。

首次六卡提交遭`QOSMaxSubmitJobPerUserLimit`拒绝，未创建GPU作业；先完成CPU40058再提交40060成功。后续遵守实际队列限制串行接续，不把提交失败当模型失败。

正式SUN固定主float native与原model494 tau800，同样256另做Q raw；全部失败保留，不择优或补样。达到同一主端Strict≥26/Meta≥128才触发已授权1000/1200独立补评。
