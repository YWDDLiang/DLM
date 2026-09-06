# 40074 相同输入的精修重复性：收束结论

2026-09-07。仅CPU读取已封存产物；未增加NN、CUDA、MLIP或改动40076。

**已排除图输入改变、sample seed错配和保存产物的请求/原子排序错位。相同输入仍不能逐位重现tau800；具体首个数值分叉算子尚未确认。** 归约顺序是有代码和样本形状证据支持的候选原因，不能将它写成已证实根因，也不能把这些差异归给contact。

## 已证实的范围

两组各171个相同native body都有真实有效且逐字典相同的native结构；实际送入模型的FP32位置/晶格、long元素/N以及eval0 seed全部171/171一致。tau也各有171对有效结构，仅5对逐字典相同。这5例不是两边均缺失形成的假相同。[全部输入与种子检查](evidence_v3_failure_20260907/REFINER_REPEATABILITY_40074.json)

最后检查覆盖四次运行的全部1011个有效tau结果：每rank的sample_indices严格对应原图列表stride，N与有序元素逐图一致；merged的六类张量逐dtype、shape与字节等于按rank拼接；sample_idx指定的payload与既有tau JSON对应。坐标误差为0，晶格矩阵最大误差约2.10×10⁻¹⁵ Å。因此没有发现可复现的输出拼接或请求映射错位。[输出对齐检查](evidence_v3_failure_20260907/REFINER_OUTPUT_ALIGNMENT_40074.json)、[1011条映射](evidence_v3_failure_20260907/REFINER_PAYLOAD_BINDINGS_40074.json)

| 已保存运行 | 有效tau映射 | world size | batch size |
|---|---:|---:|---:|
| 原K4 / 40074 K4 | 254 / 253 | 2 / 6 | 1 / 1 |
| 原K8 / 40074 K8 | 252 / 252 | 4 / 6 | 1 / 1 |

## 同seed实际保证了什么

Exporter写入 `refiner_seed=(20260905+原sample_idx) mod 2^32`。Wrapper虽然先设置 `27017+rank`，但每个图在 `model.sample` 前又按图中seed重置NumPy、Torch及CUDA generator。因此这里的rank分流或前序图数量没有直接改变该图的seed。实际配置也一致：batch=1、timesteps=1000、tau800、num_evals=1、checkpoint路径及seed模式。

本地git对比旧K4 `e70e14b...`、旧K8 tau `d865897...`：refiner wrapper、graph_from_arrays和refined assembler均未改变。Exporter新增float分支和失败字段，但body分支仍用原图函数与seed公式，而且实际171对网络输入已验证相同。[源码比较](evidence_v3_failure_20260907/refiner_repeatability_source/LOCAL_SOURCE_AUDIT.json)

当前实际外部CRYS的六个相关模块均与本地vendor哈希相同；model494当前哈希为 `573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e`。旧运行没有完整外部执行字节hash，不能用今天的文件追认旧环境的全部细节。已核代码中，fc decoder自行生成全连接边，未使用导出的CrystalNN edge字段；SigmaScheduler的随机估计量是checkpoint恢复的buffer，不能单凭构造后才设seed将它认定为原因。

同seed重置的是随机流，不能约束GPU浮点归约的执行顺序。CSPNet的节点与晶体汇总均使用torch_scatter mean；官方文档明确说明其GPU原子归约可产生非确定的浮点结果。[torch_scatter官方说明](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html)

这里还有具体形状证据：两组全部5个N=2样本都逐位重现，包括rank改变的样本；全部N≥3的166例都未逐位重现。fc聚合每节点有N项，两个加数换序不改变结合关系，三个以上可能改变。这与归约顺序的解释吻合，但没有记录首个不同操作，不能据此确认scatter解释了所有输出或SUN变化。

## 位点对应与论文归因边界

原始同site分数坐标RMS约0.28，混有整体平移和等种原子置换，不能直接解释为结构内部变化。最后的有限搜索使用旧cell度量、同元素Hungarian对应和整体平移：

| 搜索达到的残差 | K4 | K8 |
|---|---:|---:|
| RMS中位数 / p90，Å | 0.771 / 1.831 | 0.772 / 1.732 |
| RMS低于10⁻³ Å | 34/171 | 34/171 |
| 前8个固定几何matcher匹配 | 2/8 | 3/8 |

这些是**搜索达到的残差上界**；不能据较大残差全局排除其它平移、晶格基变换或对称对应，也不是能量距离。固定matcher关闭缩放和primitive变换，未按能量选择样本。[全部342个几何对应记录](evidence_v3_failure_20260907/REFINER_SITE_GAUGE_CASES_40074.json)

在171个相同native请求中，tau Strict SUN为K4 15→12、K8 15→11，不能归因contact改变了这些输入。相同native的重复标签本身也有差异：K4 raw Meta SUN49→48；K8 raw SUN10/57保持。因此tau变化不能全部单独归给refiner，SUN中的N/U也有整个cohort的依赖。[修正后的归因统计](evidence_v3_failure_20260907/CONTACT_ATTRIBUTION_V2_40074.json)

各次SUN仍是对应运行的原始实测值；本诊断限制的是小幅差异的机制归因。诊断已收束，不新增GPU复跑或改变现行评价合同。
