# 最新相关工作与H1-A2内部定位

检索范围以2024–2026年ICLR、ICML、NeurIPS和AAAI为主，CrysLLMGen作为最近邻；
2026年尚未进顶会的预印本只作为未来边界，不作为已接收SOTA混排。

## 一、顶会方法地图

| 层级 | 代表工作 | 已解决问题 | 对H1-A2的压力 |
|---|---|---|---|
| crystal text generation | [CrystaLLM, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5dc6d1c81754c32008c9667339b3fdd3-Abstract-Conference.html) | CIF/text AR生成、infilling、prompt control | 不能把“晶体也能用LM”当贡献 |
| invariant serialization | [Mat2Seq, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e23133d34964a0a09f6d076fc4b922a4-Abstract.html) | SE(3)与periodic invariant canonical tokenization | 不能把序列表示不自然泛化成我们的独占问题 |
| periodic continuous generation | [FlowMM, ICML 2024](https://proceedings.mlr.press/v235/miller24a.html) | Riemannian flow matching与晶体对称 | 证明geometry-native统一方法可行 |
| hybrid proposal/refinement | [FlowLLM, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/51d317df78eded9eb3c9d3fb1091c279-Abstract-Conference.html) | LLM base distribution＋continuous flow | “离散探索＋连续精修”不新 |
| joint text-guided generation | [TGDMat, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/24e4e3234178a836b70e0aa48827e0ff-Abstract-Conference.html) | 文本知识进入每个joint denoising step | Planner条件本身不足以构成创新 |
| periodic Bayesian generation | [CrysBFN, ICLR 2025 Spotlight](https://proceedings.iclr.cc/paper_files/paper/2025/file/1f09e1ee5035a4c3fe38a5681cae5815-Paper-Conference.pdf) | hyper-torus Bayesian flow、entropy conditioning、高效采样 | 反驳“必须分解才能处理mixed variables” |
| symmetry-preserving generation | [SymmCD, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/3a14ae9951e8153a8fc814b5f506b5b7-Paper-Conference.pdf) | asymmetric unit和space-group-preserving diffusion | H1-A2不能把Plan中的space-group range写成对称保证 |
| permutation-invariant AR | [WyFormer, ICML 2025](https://proceedings.mlr.press/v267/kazeev25a.html) | Wyckoff表示＋permutation-invariant AR | 直接反驳“AR必然受任意atom order伤害” |
| exact symmetry likelihood | [SGEquiDiff, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/697b2f31f99fb79f8a0a16e923b2471d-Abstract-Conference.html) | discrete sampler＋permutation-invariant AR＋space-group-equivariant diffusion | 与H1-A2分解最接近，必须突出接口而非模块数量 |
| hybrid LM＋diffusion | [CrysLLMGen, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f789a628fca473e922c806657512a20f-Abstract-Conference.html) | AR LLM提议species/geometry，固定species后continuous diffusion精修 | 最大威胁：H1-A2可能被视为换decoder |
| unified simple diffusion | [CrystalDiT, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37121) | unified DiT、periodic-table encoding、balanced objective | 反例：简单统一建模可能优于复杂分解 |

## 二、masked DLM文献给我们的支持和限制

| 工作 | 可用结论 | 不能滥用的地方 |
|---|---|---|
| [MDLM, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb0b13cc515724ab8015bc978fdde0ad-Abstract-Conference.html) | masked corruption/reconstruction是principled discrete generator | 不是晶体贡献 |
| [Simplified and Generalized MDM, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html) | 支持state-dependent masking schedule | 不等于domain constraints自动正确 |
| [Scaling MDM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ce1c1ff5d94079dea348a2317a889281-Abstract-Conference.html) | 可扩展、双向条件推理有优势 | 不证明所有任务优于AR |
| [MDM is Time-Agnostic, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9e3b203e72c4e058de26d02a92a81844-Abstract-Conference.html) | 提醒MDM可理解为iterative masked model | 不要用“diffusion time”包装创新 |
| [Diffusion Beats AR in Data-Constrained Settings, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0f705a932553c08ebf0d1bc520b7cbc6-Abstract-Conference.html) | random masking相当于学习多种token order | 与非固定信息顺序契合，但不能替代晶体对照 |
| [Theoretical Benefit and Limitation of DLM, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2318d75a06437eaa257737a5cf3ab83c-Abstract-Conference.html) | 不同目标下MDM复杂度不同 | 不能宣称普遍速度和正确率优势 |

## 三、CrysLLMGen应当如何作为基础

CrysLLMGen不是普通baseline，而是H1-A2故事的起点。其论文已经明确：

- LLM先生成atom types、coordinates和lattice；
- diffusion保留atom types，精修coordinates与lattice；
- composition/geometry分工可解释`E_hull`中的chemical与structural部分；
- 无效atom type/composition仍需采样期检查和丢弃；
- 两模块独立训练，采样时单向串联。

所以最强承接句是：

> CrysLLMGen establishes the value of separating discrete proposal from
> continuous refinement; H1-A2 asks the next question—whether the discrete
> proposal itself should remain autoregressive.

H1-A2必须把差异落在以下hierarchy和合同上：

1. learned `p_phi(P)`先生成underdetermined Plan，而不是回放MP-20 Plan；
2. `N`先决定exact-cardinality state；
3. DLM对partial state上的compatible realizations建模；
4. typed schema、anchors与当前实现的selected lattice/PBC checks进入生成；
5. refiner只改变连续geometry并保持composition/N。

少任何一项，reviewer都可能把它压缩成“CrysLLMGen＋MDLM”。

训练时从MP-20确定性提取Plan label是监督构造；推理时若直接抽取这些Plan，则只得到
Plan-conditioned structural generation。R5C/empirical Plan因此适合做conditional
executor reference，
不能替代H1-A2 learned Planner的fully de novo角色。

## 四、当前故事的八个不足与修复

### 1. “晶体不是句子”太泛

不足：Mat2Seq、WyFormer已经从表示和排列角度讲过类似直觉。

修复：Main RQ不再断言serialization有害，而是检验两件事：selected checks在其
prerequisites可见时介入是否有效，以及grouped confidence-adaptive policy与fixed
positional policy是否产生不同realization yield。

### 2. Planner可能只是structured prompt

不足：CrysLLMGen和TGDMat已有composition/space-group/text condition。

修复：列出Plan固定变量、剩余自由度和由N导出的状态维度；强调它是生成状态的
type/cardinality contract，而不只是自然语言条件。

### 3. DLM可能只是backbone swap

不足：bidirectional context和masking是通用DLM已有能力。

修复：贡献名称必须是Crystal completion interface，DLM只是实现该接口的核心机制；
把Plan、exact cardinality、typed schema、anchors和当前确实实现的局部checks统一呈现。

### 4. Exact length像工程优化

不足：省padding不够顶会贡献。

修复：把它解释为`cardinality before realization`，即全局N改变样本空间维数，避免让
模型同时猜结构和inactive-slot bookkeeping。

### 5. Factorization不是自明优于joint model

不足：CrysBFN、TGDMat和CrystalDiT证明统一建模可行，CrystalDiT甚至以简单统一架构
为主要论点。

修复：不宣称普遍优越；强调原生categorical support、清晰invariants和failure
diagnosis，承认以部分end-to-end coupling换可控接口。

### 6. 非前缀不等于invariance/symmetry

不足：atom-slot serialization仍非permutation invariant，Plan range也非space-group
equivariance。

修复：正文主动将non-prefix information flow、permutation invariance和symmetry
equivariance分为三个正交维度。

### 7. Novel/Unique与稳定性Pareto不是新发现

不足：CrystalDiT、Crys-JEPA和DynaCrys都在处理discovery balance。

修复：把candidate supply / conversion作为H1-A2架构的诊断视角，不把trade-off本身
当贡献，也不把高UN归因成DLM定理。

### 8. “Diffusion”名称可能反受攻击

不足：ICLR 2025指出MDM可被视为time-agnostic masked model。

修复：故事依赖iterative non-prefix completion与probabilistic masking objective，不
依赖continuous diffusion analogy；不宣称parallel speed。

## 五、2026研究前沿

- [Crys-JEPA](https://arxiv.org/abs/2605.14759)：用energy-aware embedding进行screening
  和generative refinement，直接把stability–novelty写成狭窄可用区域。
- [DynaCrys](https://arxiv.org/abs/2608.07401)：让space group、Wyckoff occupation、
  elements和geometry共同演化，是symmetry-aware symbolic diffusion的强边界。

两者目前作为preprint horizon使用。它们提醒我们：H1-A2的故事应集中在base generator
的Plan-to-realization接口，不要与feedback optimization或exact symmetry抢同一贡献。

## 六、最终可占据的空位

> 从learned prior生成一个欠定global Plan，用其cardinality实例化typed partial crystal
> state，再用non-prefix masked generation从compatible realization distribution中采样，
> 最后只把continuous geometry交给physical refiner。

顶会相关工作已经分别解决representation、periodic manifold、symmetry、joint
generation、hybrid refinement和feedback optimization；这个空位足够窄，因而可信，
又可以上升到“serialized scientific object的storage/inference/optimization order解耦”
这一更一般原则。

## 七、评分

- 若摘要写成“we replace AR with DLM”：`5/10`；
- 当前经Proposer–Reviewer冻结的support×commitment framing：concept约`7/10`；
- 该评分以严格Plan-level paired evidence为条件，不要求声称DLM优于AR；
- 若support与policy均无正向或有意义的异质性，方法故事应降至`4.5–5/10`，不能用
  end-to-end S.U.N.替代机制证据。
