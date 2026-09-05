# 连续扩散阶段的统一终态监督：代码核对与扩展建议

2026-09-06。用户提出：是否也给连续diffusion安排当前方法，从论文整体性和效果上
是否更好。本文是经过代码核对的扩展方案，**不是已经训练/部署的新refiner**。
当前DLM刷新、首轮开发评估和今天的完整结果排程继续，model494在已登记对照中保持冻结。

## 实际接口

当前远端`crysllmgen-main/models_ddpm/diffusion.py`的CSPDiffusion已核对：

- forward读取原子类型、连续fractional坐标、晶格矩阵与时间；晶格训练预测Gaussian
  噪声，坐标训练匹配wrapped-normal score，两项MSE相加。
- sample直接把输入DLM提议的坐标/晶格作为t=800初始状态，并进行predictor/corrector
  更新；每步读取当前几何，预测坐标score和晶格噪声。它没有DLM的MASK旧值丢失问题。
- Llama的soft Plan和species program目前没有作为独立条件传进这个decoder；DLM提议
  只经初始几何及固定原子类型影响精修。原提议没有独立的长期条件通道。
- 坐标有mod1包裹，采样含corrector与predictor两个子步，最后t=1人为将随机项置零。
  所以不能把DLM的离散完整路径log probability直接搬到这里，也不能把最后确定性步
  当作普通有噪声Gaussian转移来计算密度。

本地调用入口为`src/scripts/refine_dlm_with_crysllmgen.py`，固定batch1及每个源样本的
refiner_seed，调用model.sample(...,diff_steps=800)。架构背景可见
[CrysLLMGen官方代码](https://github.com/kdmsit/crysllmgen)。

## 统一什么，而非机械复制模块

建议的论文主线是同一个科学目标下的分阶段学习：

```text
冻结Llama的组成/Plan/程序
          ↓
DLM：精确化学下的离散盆地提议 x_D
          ↓
连续扩散：利用提议条件精修得到 y
          ↓
共同离线R评价：原生—终态能差与终态盆地能量
```

DLM侧保留当前A(x_D)=e(x_D)-e(R(x_D))、B(x_D)=e(R(x_D))-h(c)。如果单独对
refiner输出做双目标后训练，就应明确A_F=e(y)-e(R(y))、B_F=e(R(y))-h(c)，而且
在固定(c,x_D,Plan/program)内采多个refiner噪声结果。现在DLM的K8是不同x_D，不能
直接改称“同一refiner输入的K8”，其标签也不能冒充refiner输出y的标签。

一种较省新增物理计算的扩展，是用已验证的训练对(x_D,R(x_D))做连续终态蒸馏，
保留MP20去噪监督，并增加对原提议及Plan/program的显式条件通道。这利用了同一个
teacher的数据，但其训练目标应诚实标为条件去噪/score拟合；不能称为已实现与DLM
完全相同的精确路径KL优化。新的条件接口需处理周期表示、晶胞基变换和原子对应。

已有[DDPO研究](https://arxiv.org/abs/2305.13301)说明可以对扩散模型优化下游奖励；
[Diffusion-DPO](https://arxiv.org/abs/2311.12908)采用扩散似然的ELBO代理。
这些提供可行性参考，不构成我们晶体任务上的效果保证，也不能替代本机采样器的推导。

## 对论文和效果的判断

如果统一终态目标、说明两阶段职责并验证增量收益，论文会更完整。连续阶段已在
当前参考上把Strict/Meta SUN从2.34%/21.48%提高到7.42%/47.27%，说明它确实是
系统效果的重要组成。但这些是冻结refiner已有的收益，不能据此保证继续微调会提升。
参考精修后novel计数也从254降到221，提示稳定性改善可能伴随新颖性损失。

同时改两边会使归因更难。如果新收益全部来自refiner，就不足以证明DLM原生稳定性
创新；如果两者都拟合同一个有误差的势模型，也可能进一步放大已经观察到的异常能量。
必须保留当前固定refiner链作为对照，新refiner在相同DLM输入、相同噪声种子、相同
评价协议下单列，不能把精修收益写成原生收益。

因此建议把这项纳入明确的连续阶段扩展，先拿正在安排的首轮对照判断瓶颈。当前
不因概念讨论取消已有任务或静默更换model494。今天的完整主线结论先按既定方案
结算；若进一步实施refiner适配，要先完成独立接口/目标和数据核对，再将新增训练
及固定对照共同列入实验计划，不能把它当成无成本的附加改动。
