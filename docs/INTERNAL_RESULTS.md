# 内部组成实验结果

这些组成结果只在个人repo中保留，不进入对外README的组成实验表。

未来论文主表继续使用`105/1000` Strict与`488/1000` Meta。下表用于内部来源追踪；
exact all-attempt与historical frozen视图均不替换该主表。

| 实验 | 总条目数 | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|---:|
| H1-A2 epoch 2 | 1,000 | 94/1000 = 9.40% | 474/1000 = 47.40% |
| R03 D1 control | 1,024 | 99/1024 = 9.67% | 523/1024 = 51.07% |
| R03 D2 Safe-axis | 1,024 | 117/1024 = 11.43% | 496/1024 = 48.44% |

独立审计视图：

| 审计视图 | Requested | Reconstructed | Hull known/unknown | Strict | Meta |
|---|---:|---:|---:|---:|---:|
| exact all-attempt replay | 1,200 | 1,164 | 1,132/32 | 103/1200 | 553/1200 |
| historical frozen compatibility | 1,000 | 1,000 | legacy contract | 94/1000 | 474/1000 |

R03 D2逐repeat：

| Repeat | Strict | Meta |
|---:|---:|---:|
| 0 | 28/256 | 122/256 |
| 1 | 31/256 | 123/256 |
| 2 | 29/256 | 125/256 |
| 3 | 29/256 | 126/256 |

四次repeat复用同一冻结Plans和scientific seed ledger，是process
realizations而不是四个独立Planner samples。
