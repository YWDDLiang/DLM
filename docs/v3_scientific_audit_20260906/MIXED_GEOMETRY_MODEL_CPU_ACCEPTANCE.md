# H-P33 模型接口的 CPU 实现验收

日期：2026-09-06。状态：**本机小型 Torch 模块验收通过；不代表真实 8B、GPU/DDP 或 SUN 验收通过。** 本轮未加载预训练权重、未安装依赖、未提交作业，也未修改下载模型。涉及短小测试模块的 backward/update，不是晶体模型的正式训练。

## 已冻结代码

四文件 SHA256 映射的整体指纹为 `80f00450f6aff00681bf16252d7f990cd253c8d0f7e2fcd667d09bd2802065e2`；这是文件集合指纹，执行 Git commit 由主审另行冻结。提交该指纹后，作者停止修改这些文件。

| 文件 | SHA256 |
|---|---|
| [llada_hidden.py](../../src/crystal_dlm/llada_hidden.py) | `1cf99fbb5e2260012426ba75fdfea6f491cac388b31adaa51b27a9466cbc79ca` |
| [mixed_geometry_model.py](../../src/crystal_dlm/mixed_geometry_model.py) | `f003c151c1e141c005c1c6da3c5f86bd3a2a545706d1c3a8661f2d8c9d00ad31` |
| [periodic_v2_initialization.py](../../src/crystal_dlm/periodic_v2_initialization.py) | `0764c177fe38ff23a5dfff8824f940ad71968f0a8385cf2124d85de19ede8363` |
| [test_mixed_geometry_model.py](../../tests/test_mixed_geometry_model.py) | `a4e952b064e11cec6869fe0787c0bf6e803ed7deae0d61166238e866b43bc8c5` |

V2 的改动只提取 `load_periodic_v2_architecture` 供不同架构共用。旧公开 `load_periodic_v2_model` 仍先执行原 completed-final 校验；没有布尔跳过选项。新 mixed loader 读取自己的 marker、配置、normalizer、heads 与 token 组件哈希，具体方法的最终 policy 资格仍由调用者检查，不绑定 V2 的更新数。

## 实际通过的测试

运行环境为本机 `D:/codex_work/ai4s/DLM_llama_programmed_basin_closure/.venv/Scripts/python.exe`，Python 3.12.14、PyTorch `2.8.0+cpu`，CUDA build 为 `None`。

| 测试集 | 实际结果 | 核对内容 |
|---|---:|---|
| `test_mixed_geometry_model.py` | **14/14 PASS，0 skip** | hidden、T/G 路由、梯度、精度存储与保存加载 |
| `test_periodic_v2_initialization.py` | **2/2 PASS** | 旧 V2 final gate 与原 compact module 保存重构 |
| `test_mixed_geometry_diffusion.py` | **19/19 PASS，0 skip** | 最终 VP 端点、chart、wrapped targets、风险和 P33 数值 oracle |

四个本轮文件的 AST、尾随空白检查与限定范围 `git diff --check` 均通过。数学测试最终本机执行包含严格 VP 0/1 端点分支；先前 [CPU40013](evidence/GEOMETRY_NUMERICS_40013.json) 已通过全来源 chart/normalizer 检查，本报告不重复将它算作新的模型训练证据。

14 项模型测试的关键证据是：

- hidden oracle 直接取自 [实际 LLaDA forward 记录](evidence/LLADA_ACTUAL_CORE_FORWARD.json) 中未改写的 `LLaDAModel.forward`，在小型真实 Torch attention/LoRA 模块上运行。新 hidden-only 路径与其最终 `ln_f` 输出精确一致，包含 head/key 相关 bias、padding、bool/min bias 和 block groups；未用新实现作为自己的参考函数。
- native nonreentrant checkpoint 路径的输入、bias 和 LoRA 梯度与原 forward 一致；没有 hidden detach 或复制第二套 LoRA/block。
- T 路由是同一个 `token_model` 对象，输出精确一致。G 将旧 numeric decoder、task projection、numeric adapter、output-row 和词表投影替换为测试中的禁止调用桩，仍成功输出几何预测。
- G 的完整浮点状态能改变 hidden；改变未使用的 old numeric tokens 不改变 G 输出。time residual 只加入真实 body，未直接加入 prompt/padding。
- 两次小模块更新验证零 head 首步及随后共享 LoRA、conditioner、bias、input rows、time 模块的有限非零梯度；G 下 T-only 参数无梯度。新 heads/time 参数在父模型 dtype 转换后仍为 FP32。
- mixed 保存加载保留 T/G 输出、normalizer 与 parent provenance；normalizer 哈希被改动时拒绝加载。初始化调用严格 V2 parent loader。这里的外部 raw/PEFT 架构重建用小模块 fixture 替代，尚未验证真实 8B checkpoint 的完整重载。

可复现命令如下，在仓库根目录使用上述已配置解释器；`-B` 避免写项目 bytecode cache：

```python
import sys, unittest
sys.path[:0] = ["src", "tests"]
suite = unittest.defaultTestLoader.discover("tests", pattern="test_mixed_geometry_model.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
```

数学测试将 pattern 改为 `test_mixed_geometry_diffusion.py`；旧 V2 回归采用 `pytest -q -p no:cacheprovider tests/test_periodic_v2_initialization.py`，同样设置 `src/tests` 导入路径。

## 供后续脚本使用的接口

`forward(input_ids, attention_mask=None, *, mode='token', geometry_context=None, geometry_state=None, species=None, output_hidden_states=False, **kwargs)`。

G 的 `geometry_state` 是既有 `GeometryState(z, fractional, t, atom_mask)`；`species` 为 `[B,Npad]` atomic Z；`CrystalStateContext` 仅供位置、N、rank 与 scaffold 布局。G 返回 `GeometryModelOutput.v_prediction/u_prediction`，可选 `hidden_states=(ln_f_hidden,)`，没有 logits。T 原样返回 V2 输出。

原模型是 `model.token_model`；`model.base_model` 指向原 PEFT 对象。新参数在 `model.geometry_heads` 下的 `time_mlp`、`geometry_mode`、`v_head`、`u_head`。`model.normalizer` 是 `LatticeNormalizer`。训练使用 `set_mixed_geometry_trainable(model)`；checkpointing 继续通过 `enable_native_checkpointing(model.base_model)`。

初始化 `initialize_mixed_geometry_model(model_path, v2_checkpoint_path, normalizer, device, ...)`；恢复 `load_mixed_geometry_model(model_path, checkpoint_path, device, ...)`，均返回 `(model, tokenizer)`。`save_pretrained` 写入 `mixed_geometry_config.json`、`mixed_geometry_heads.pt`、`mixed_geometry_normalizer.json` 及原 token 组件。

## 尚需真实模型验收的边界

上述测试不证明真实 PEFT/8B 的 BF16 hidden parity、多卡交替未用参数的归约、dropout/checkpoint 行为、实际吞吐/显存、卸载后重载或低噪声输入敏感度。真实 preflight 还需执行完整 33 次场评估及最终 chart 解码，确认数值过程与登记端点一致。小模块往返加载不替代真实 compact checkpoint 重新加载。

本报告只是实现验收记录。候选的有限采样误差、末端均值在多峰分布中的限制、训练可学性及最终 Strict/Meta SUN，仍按 [H-P33 规格](V3_H_P33_SPECIFICATION.md) 分别检查；没有因这些 CPU PASS 提前批准模型效果。
