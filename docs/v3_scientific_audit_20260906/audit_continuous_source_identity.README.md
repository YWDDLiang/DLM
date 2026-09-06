# 原连续 CIF 来源身份核验

工具：[audit_continuous_source_identity.py](tools/audit_continuous_source_identity.py)。这是 CPU 数据核验，不调用模型、GPU、CHGNet、弛豫或预测器，也不生成筛选后的训练集。

默认使用原 CSV 的 **`cif`** 列和当前 prepared JSONL 的 **`source_answer`**。不回退到 `cif.conv`：本项目的后者可能是不同 N 的常规大胞。`answer` 可能属于 transition/训练状态，不能取代 clean source；只有审计不同数据格式时才显式传 `--target-field answer`。

## 运行

需要现有 pandas、pymatgen 和项目 `src/crystal_dlm`。`--help` 不加载这些运行依赖。工具可放在其他路径，使用 `--project-root` 指向本次实际使用的冻结源码；不写死作业号、远端目录或样本数。

```bash
python audit_continuous_source_identity.py \
  --project-root /path/to/frozen/project \
  --source-csv /path/to/original/train.csv \
  --training-jsonl /path/to/prepared/train.jsonl \
  --output-dir /path/to/new/train-source-audit \
  --split train --expected-sources 27136 --workers 4
```

val 用相应文件、`--split val --expected-sources 9047`，分别写新输出目录。实际输入路径与已取回的 CSV SHA 见 [CONTINUOUS_SOURCE_INPUT_PATHS.json](evidence/CONTINUOUS_SOURCE_INPUT_PATHS.json)；工具按文件重算实际数量，不把示例数字当成内置常量。可加 `--source-csv-sha256` / `--training-sha256` 核对已知输入摘要。

`--output-dir` 必须不存在或为空。`--workers` 默认1；pymatgen解析支持多个CPU进程，`--chunk-size` 默认256，逐chunk输出进度。没有额外安装步骤。

## 身份依据与排列

按以下依据选择原 CSV 行，最后都必须通过完整结构核对：

1. training行明确携带 `csv_row_idx` **和匹配的** `source_csv_sha256`，将行号绑定到具体原文件。
2. training保留稳定 `material_id`/`mp_id`（允许原 `metadata`/`source_metadata` 中相应字段），且原CSV该ID唯一。
3. 可选 `--r5-jsonl` 中的完整answer、稳定ID及明确source index提供桥接证据。中间文件的裸ordinal不会自动升级为CSV行号。
4. 没有上述来源证据时，按**完整量化answer**的唯一匹配选择原CSV行。索引包括全部晶格参数、所有元素和坐标；只按稳定元素分组规范顺序，组内原顺序不变，并归一原有100→0周期alias。它不是formula匹配，也不排序坐标、不旋转/平移拟合、不约简晶胞。

裸 `source_row_idx` 只报告其是否恰好等于匹配CSV序号，不能用它打破多个完整answer匹配的歧义。若存在未成功解析的CSV行，就不能凭其余行的“唯一hash”宣布全CSV唯一；该情况保留为未核实。稳定ID和几何不相符时也不转而挑另一个hash候选。

选中原行后，按当前 `plan_state` 的硬N/E顺序调用已有 `canonicalize_dynamic_answer_to_plan`。排列方向明确为：

```text
training aligned slot -> original parsed CIF site index
```

同元素内部保持原解析顺序。然后使用**原连续** lattice/coordinates 应用该排列，坐标只做周期wrap，重新调用当前codec，并要求 `Q(continuous_aligned)==canonical source_answer`。不会从量化token解码出假连续标签。`species_program` 是访问顺序，`condition_prediction.typed_metadata_source_row_idx` 可能是条件donor；两者都不作为原CIF或站点排列依据。

默认拒绝重复training source key；确实审计展开后的多view文件时，可显式用 `--allow-repeated-source-records`，仍逐记录输出，并分别统计记录数/唯一来源数。多个不同training source key映射到同一CSV行会单列，不假装一一对应。

## 产物和返回码

| 产物 | 内容 |
|---|---|
| `INPUT_RECEIPTS.json` | CSV/JSONL/R5 SHA、选用列、解析调用与包版本、codec配置、项目源码与工具SHA |
| `csv_index.jsonl` | 每个CSV行的CIF文本SHA、稳定ID、解析/编码状态、警告、完整answer签名 |
| `source_identity.jsonl` | 每个非空training输入行的结果，包括坏JSON、缺失、歧义和成功；成功行包含连续对齐结构与精确排列 |
| `DUPLICATE_FULL_ANSWERS.json` | 完整量化结构重复组及原CSV行、ID、CIF SHA |
| `r5_index.jsonl` | 提供R5时的解析记录与错误；不默默丢行 |
| `SUMMARY.json` | 分母、状态计数、匹配方法、原CIF身份gate、未分配CSV行等 |

每个primary结果带原split绑定、CSV SHA、原数据行序号、稳定ID（如有）、CIF文本SHA、pymatgen版本、目标hash及对齐检查。CSV行序号是**数据记录序号**，不是含多行CIF的文本行号，也不是 `Unnamed: 0` 列。

- 返回0：全部当前source记录通过，预期来源数匹配，并写 `_AUDIT_COMPLETE` 和 `_AUDIT_SUCCESS`。
- 返回2：审计完成但有明确未解决项，写 `_AUDIT_COMPLETE` 和 `_AUDIT_ISSUES`。不能仅取成功子集冒充全量original-CIF监督。
- 返回1：输入/程序错误，写 `_AUDIT_ERROR`；部分文件不是完整核验结果。

只有完整成功标记与汇总gate都通过，才可以将本次结果称为完整原CIF身份对齐。身份核验不是物理稳定性证书；codec clipping和parser警告会记录，不能把它们隐藏成模型训练数据修复。

## 已完成的本地检查

[TOOL_SELFTEST.json](evidence_continuous_source_identity/TOOL_SELFTEST.json) 使用真实pymatgen解析和2个CPU worker，检查了唯一完整answer匹配、稳定ID消除重复结构歧义、几何与ID矛盾、坏JSON保留、source_answer优先于transition answer、可选R5明确index/ID桥接。测试是少量合成数据，不代替27136/9047来源的全量执行。
