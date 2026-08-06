# job28185 晶格 retraction 根因与 v2 计划

状态：`DEVELOPMENT_MECHANICS_GATE_PASS_JOB28187_AUDITED`  
日期：2026-07-26  
适用 run：`20260720_0401-crysllmgen-wq-final-v3`

## 1. 结论先行

job28185 的失败不是 released CrysLLMGen schedule、SG63 centering 或
Wyckoff orbit topology 本身失效。直接原因是 v1 晶格 projector 对一个**绝对
候选晶格**使用了只在无穷小邻域有效的一阶 chart 更新：

\[
\Delta z_{\mathrm{v1}}
\approx J_L^+
\operatorname{vec}(L_{\mathrm{proposal}}-L_{\mathrm{current}}).
\]

晶格长度存储在 log chart 中。若 parent 提议
\(a_{\mathrm{new}}=f a_{\mathrm{old}}\)，正确的有限位移是
\(\log f\)，v1 却得到近似 \(f-1\)，随后 decode 再计算
\(\exp(f-1)\)。因此大步长会被指数放大。

在失败 SG63 source lattice 上做受控复现时，20 倍的 \(a\) 轴提议：

- 正确 chart 位移：`log(20)=2.9957322736`；
- v1 位移：`18.9999999886`；
- v1 primitive lattice 最大绝对元素：`3.3776209769e8 Å`；
- v2 最大绝对元素：`37.8482459854 Å`。

旧的 primitive consistency exception 是爆炸后的浮点症状，不是应当通过放宽
`atol` 接受的正常误差。只把 `1e-8` 改大将保留约 \(10^8\) Å 的非物理晶格，
所以已明确拒绝。

## 2. 不可变失败事实

旧身份：

```text
wq_wyckoff_tangent_bridge_preflight_v1
job28185
```

终态：

- Slurm：`FAILED 2:0`；
- F：`32/32` succeeded；
- T：`31/32` succeeded；
- retry/replacement：`false`；
- training/new generation/MLIP/API：均未发生。

唯一失败：

| 字段 | 值 |
|---|---|
| arm | `T` |
| cell | `b-e2a6902801f056e71703433a` |
| timestep | `800` |
| panel index | `2` |
| source attempt | `a-155dee320a1473935cfeef50` |
| SG | `63` |
| 阶段 | first predictor projection |
| exception | `projected primitive lattice disagrees with WQ expansion` |

不可覆盖终态审计：

```text
runs/remote_audit/
  20260726_wq_wyckoff_tangent_bridge_preflight_v1/
  terminal_audit_job28185.json
```

SHA256：

```text
2f686b881479f12b4abdc4c0ece217947c5aeb99c072f6119137e97887905f22
```

job28185 不取消、不重试、不替换，也不改变其 FAIL 结论。

## 3. SG63 取证

失败 source 的 conventional lattice 为：

\[
L_c=\operatorname{diag}
(3.7848245985,\ 10.7125049331,\ 8.3374172856).
\]

只读证据中的 primitive lattice 为：

\[
L_p =
\begin{bmatrix}
1.8924122993 & -5.3562524665 & 0\\
1.8924122993 &  5.3562524665 & 0\\
\epsilon     & -2\epsilon    & 8.3374172856
\end{bmatrix}.
\]

对应 exact centering transform：

\[
C =
\begin{bmatrix}
0.5 & -0.5 & 0\\
0.5 &  0.5 & 0\\
0   &  0   & 1
\end{bmatrix},
\qquad L_p=C L_c.
\]

重建最大绝对误差为 `8.88e-16`。因此：

- centering transform 与 source lattice 一致；
- SG63 不是因为错误的 primitive basis 才失败；
- 旧代码每步重新由浮点 lattice pair 估计 \(C\)，会把微小误差乘到爆炸尺度上，
  但它只是放大后的触发器。

失败 source 的固定 orbit inventory：

- Si：conventional multiplicity 4，primitive 2，chart dimension 1；
- La：conventional multiplicity 4，primitive 2，chart dimension 1；
- Ru：conventional multiplicity 8，primitive 4，chart dimension 2。

没有 orbit、species、multiplicity 或 atom count 改变证据。

## 4. v1 数值错误

以 orthorhombic 的 \(a\) 轴为例：

\[
z_a=\log a,\qquad
\frac{\partial a}{\partial z_a}=a.
\]

v1 对绝对提议 \(a'=fa\) 做：

\[
\Delta z_a
\approx
\frac{a'-a}{a}
= f-1.
\]

然后：

\[
a_{\mathrm{v1}}
=
\exp(z_a+\Delta z_a)
=
a\exp(f-1).
\]

有限位移的正确结果应为：

\[
\Delta z_a
=
\log a'-\log a
=
\log f,\qquad
\exp(z_a+\log f)=fa.
\]

因此该错误不是正则系数选错，而是把局部 tangent map 用作有限步长 global
retraction。

## 5. 受控最小复现

下面只是在本地 NumPy 路径对失败 source lattice 施加合成轴缩放，不是新的
scientific attempt，也没有调用 parent model：

| scale | v1 chart step | exact `log(scale)` | v1 max \(|L_p|\) | v1 consistency abs | v2 max \(|L_p|\) |
|---:|---:|---:|---:|---:|---:|
| 15 | 14.0000000 | 2.7080502 | 2,275,823 | `5.34e-10` | 28.3862 |
| 18 | 16.99999999 | 2.8903718 | 45,711,129 | `1.07e-8` | 34.0634 |
| 20 | 18.99999999 | 2.9957323 | 337,762,098 | `7.93e-8` | 37.8482 |
| 21 | 19.99999999 | 3.0445224 | 918,132,572 | `2.38e-7` | 39.7407 |

注意 v1 的 relative consistency error 仍约 `2.35e-16`。这正说明：

1. exception 来自巨大尺度上的 absolute floating-point mismatch；
2. 单独改成 relative tolerance 会让 exception 消失；
3. 但 \(10^8\) Å 晶格仍存在，所以 relative tolerance 也不能作为修复。

失败 runner 没有在 throw 前序列化实际 parent-proposed lattice。因此本文不声称
remote cell 的实际缩放恰好为 20；本文声称的是：同一个 source、同一条 v1
映射可以确定性地产生同类 exception，并且新 retraction 消除了该数学机制。

## 6. v2 方法

新身份：

```text
wq_wyckoff_chart_retraction_preflight_sup28185_v2
```

新 lattice rule：

1. 从 `ExpandedState` 读取一次 exact centering transform \(C\)；
2. 用 linear solve 把 proposed primitive lattice 映到 conventional frame；
3. 对**绝对 proposed conventional lattice**调用
   `LatticeChartCodec.encode_matrix`；
4. 对该 chart 调用一次 `decode_matrix`；
5. 用同一个 \(C\) 回到 primitive frame；
6. 重新 expansion 后同时审计 transform、absolute consistency 与 relative
   consistency。

名称冻结为：

```text
global_chart_retraction_v1
```

局部 finite-difference Jacobian仍保留用于记录“若线性化会产生什么更新”，但不再
决定有限步长的输出。lattice Tikhonov `1e-8` 也不再用于 v2 retraction；
orbit projector 的 `1e-8` 保持不变。

## 7. 新增 mechanics Gate

除 v1 全部 topology、round-trip、finite、call-budget Gate 外，v2 增加：

| Gate | 阈值 |
|---|---:|
| projection method | exact `global_chart_retraction_v1` |
| primitive transform consistency | `<=1e-12` |
| primitive lattice absolute consistency | `<=1e-10` |
| primitive lattice relative consistency | `<=1e-12` |
| primitive lattice max absolute entry | `<=100 Å` |
| post-retraction condition number | `<=1e6` |

`100 Å` 是非常宽松的 MP20 数值安全上限，只用于抓住 `exp(19)` 类爆炸；它不是
物理质量、S.U.N. 或模型选择阈值。condition Gate 同样只用于 fail-closed
numerical degeneracy。

每步还记录：

- input ambient update norm；
- exact chart displacement norm；
- local-linearized update norm；
- actual retracted update norm；
- normal residual norm；
- primitive transform/absolute/relative consistency；
- primitive lattice scale。

## 8. 证据边界

v2 若再次运行同一个 `8 sources × 4 timesteps` panel，只能用于：

- mechanics regression；
- 数值安全闭环；
- 证明已定位 failure mode。

因为 job28185 的唯一失败 cell 已经影响修复设计，所以同 panel 的 v2 PASS
**不能**用于：

- confirmatory method claim；
- 宣称 T 优于 U；
- headline validity/S.U.N.；
- adapter 或长训练授权。

只有 development v2 PASS 后，才可另行冻结一个与这 8 个 source 不重叠的
held-out panel。任何 held-out 执行也必须在看到输出前冻结阈值、arms、paired
noise、denominator 与 hashes。

## 9. 当前冻结文件

核心实现：

```text
crystal_dlm/wqcodiff/crysllmgen/tangent_bridge.py
SHA256 127e3c707b1bf79f2fc44d97bccecd8a6de3cb39b8e989a2806c5c7b377bfbaf

crystal_dlm/wqcodiff/runtime.py
SHA256 8b5ba104ee1be25ff7f8a14b703193b33920bfd71abd52b1ba1e0d082e909ea4
```

v2 runner：

```text
scripts/a800/run_wq_wyckoff_chart_retraction_preflight_sup28185_v2.py
SHA256 63925aa0b877914b35240e55459026cf10e660665a14304a427c20253ef57a35
```

v2 contract：

```text
configs/experiments/wyckoff_codiffusion/
  wq_wyckoff_chart_retraction_preflight_sup28185_v2.json
SHA256 518e44cc1a94334f8232ee54f4199a3c01436c0768defb9e61e2628a27324a6a
```

机器可读根因审计：

```text
runs/remote_audit/
  20260726_wq_wyckoff_tangent_bridge_job28185_root_cause_v1/
  root_cause_audit.json
```

## 10. 本地验证

已执行：

```text
python3 -m unittest \
  tests.test_crysllmgen_tangent_bridge \
  tests.test_wqcodiff_runtime \
  tests.test_crysllmgen_chart_retraction_preflight_v2 \
  tests.test_crysllmgen_tangent_preflight_runner \
  tests.test_wq_wyckoff_tangent_submission
```

结果：

- run：45；
- pass：42；
- expected skip：3；
- failure/error：0；
- Python compile：PASS；
- Ruff：PASS。

其中新增回归直接使用失败 SG63 source lattice 和 20× 合成 proposal，确认：

- chart update 等于 `log(20)`；
- primitive lattice 不爆炸；
- exact centering transform 不漂移；
- primitive lattice re-expansion 一致。

## 11. 执行终态

用户随后授权了该确切 v2 身份的修正归档、远端 Gate、一次性 development
mechanics submission 和终态审计。最终使用的累计执行 manifest 为：

```text
227dd62635dc20ac79580810defeea2b5f47e399fb70913535c5809f5e876642
```

远端执行前：

- 138/138 文件原子安装；
- 65/65 targeted tests PASS；
- installed-byte Gate PASS；
- released parent checkpoint CPU strict-load PASS；
- schedule 最大绝对误差 `5.96e-08 <= 1e-6`；
- submission record、claim、output 和同名排队/运行 job 均不存在。

`submit_once.sh` 严格调用一次并得到：

```text
job28187
wq-chart-ret-wtb32-v2
gpu / node99 / 1×A800 / 8 CPU / 64G / 01:00:00
COMPLETED 0:0
elapsed 00:00:42
```

development panel 结果：

| arm | terminal | succeeded | failed | positive volume | retry/replacement |
|---|---:|---:|---:|---:|---:|
| F | 32 | 32 | 0 | 32 | 0 |
| T | 32 | 32 | 0 | 32 | 0 |

全部 30 个 mechanics Gate 为 `true`。T arm 的 first-reverse invalid lattice、
nonfinite trajectory、forward-contract failure 和 input-identity failure 均为
0；projection method 精确为 `global_chart_retraction_v1`。这证明 v1 的有限步长
晶格 chart failure mode 已在同一 development panel 上闭环。

不可覆盖终态审计：

```text
runs/remote_audit/
  20260726_wq_wyckoff_chart_retraction_preflight_sup28185_v2/
  terminal_audit_job28187.json
SHA256 385fa64d40486b600d67a001488b278bb4c6c3a941419b8b6d52d50d7ec120dd
```

证据边界保持不变：该 PASS 不是 confirmatory evidence，不授权训练或新生成，
也不能用于宣称 T 优于 U。下一步若继续，应另行预注册与这 8 个 source 不重叠的
held-out mechanics contract，并在看到输出前冻结阈值、paired noise、denominator
与全部 hashes。
