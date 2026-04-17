# ICTV Agent 批量分类工具

## 概述

`batch_classify.py` 是 ICTV Agent 的命令行批量分类工具。输入一个多序列 FASTA 文件，并行提交到 Agent API 进行分类，输出结构化的 Excel 汇总表和每条序列的详细 TXT 报告。

## 依赖

```bash
pip install httpx openpyxl
```

## 使用前

确保 ICTV Agent 服务已启动：

```bash
cd ictv_agent/
bash run.sh          # 默认端口 18231
# 或指定端口
bash run.sh 9000
```

## 基本用法

```bash
python scripts/batch_classify.py input.fasta -o output_dir/
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入多序列 FASTA 文件（必需） | — |
| `-o, --output` | 输出目录（必需，自动创建） | — |
| `--api` | Agent API 地址 | `http://localhost:18231` |
| `--parallel` | 最大并发数 | 4 |
| `--family` | 可选的科名提示（跳过 BLAST 科鉴定） | 无 |
| `--timeout` | 单条序列超时时间（秒） | 600 |

## 示例

### 1. 基本分类

```bash
python scripts/batch_classify.py my_viruses.fasta -o results/
```

### 2. 指定病毒科（加速）

如果已知所有序列属于同一科：

```bash
python scripts/batch_classify.py corona_seqs.fasta -o results/ --family Coronaviridae
```

### 3. 调整并发数

```bash
# 8路并发（服务器最大支持8）
python scripts/batch_classify.py big_dataset.fasta -o results/ --parallel 8

# 单线程（调试用）
python scripts/batch_classify.py test.fasta -o results/ --parallel 1
```

### 4. 远程服务器

```bash
python scripts/batch_classify.py input.fasta -o results/ --api http://192.168.1.100:18231
```

## 输出文件

运行完成后，输出目录包含：

```
output_dir/
├── results_summary.xlsx     # Excel 汇总表（带颜色标注）
├── batch_classify.log       # 运行日志
├── MG021328.1.txt           # 每条序列的详细报告
├── PQ489644.1.txt
├── ...
```

### Excel 汇总表字段

| 列 | 说明 |
|----|------|
| Accession | 序列 ID |
| Status | done / error |
| Time(s) | 耗时（秒） |
| Cached | 是否命中缓存 |
| Family | 病毒科 |
| Genus | 属 |
| Species | 种（含 novel species 标注） |
| Confidence | High / Medium / Low |
| Novel | 是否为新种候选 |
| Evidence | ICTV 标准依据摘要 |
| Reasoning | 分类推理说明 |
| Steps | Agent 推理步数 |

颜色标注：
- **Confidence**: 绿色=High, 黄色=Medium, 红色=Low
- **Novel**: 红色底色=True
- **Status**: 红色底色=Error

### TXT 详细报告

每条序列生成一个 `.txt` 文件，包含：
- 完整分类层级（Realm → Species）
- 置信度和新种判定
- 使用的 ICTV 标准和计算值
- 分类推理全文
- Agent 推理步骤日志

## 运行日志

实时日志输出到终端和 `batch_classify.log`，格式：

```
14:50:08 [INFO] ICTV Batch Classifier
14:50:08 [INFO]   Input:    hepe_seqs.fasta
14:50:08 [INFO]   Output:   results/
14:50:08 [INFO]   API:      http://localhost:18231
14:50:08 [INFO]   Parallel: 4
14:50:08 [INFO] Server OK: {'status': 'ok', 'families_in_criteria': 32, ...}
14:50:08 [INFO] Parsed 21 sequences from hepe_seqs.fasta
14:50:12 [INFO] [1/21] START  MG021328.1 (6830 nt) [running=1]
14:50:12 [INFO] [2/21] START  PQ489644.1 (7486 nt) [running=2]
14:50:12 [INFO] [3/21] START  PQ489643.1 (7334 nt) [running=3]
14:50:12 [INFO] [4/21] START  PQ489669.1 (6615 nt) [running=4]
14:51:45 [INFO] [1/21] DONE   MG021328.1: Hepeviridae/Rocahepevirus/... conf=Medium (96.2s)
14:51:50 [INFO] [5/21] START  PQ489656.1 (7468 nt) [running=4]
...
14:58:30 [INFO] ============================================================
14:58:30 [INFO] BATCH COMPLETE
14:58:30 [INFO]   Total:    21 sequences
14:58:30 [INFO]   Done:     19
14:58:30 [INFO]   Errors:   2
14:58:30 [INFO]   Time:     502.1s total, 24s avg
14:58:30 [INFO] ============================================================
```

## 性能参考

| 序列类型 | 长度 | 平均耗时 | 说明 |
|----------|------|---------|------|
| Coronaviridae | 27-32 kb | 120-280s | 使用 DEmARC PUD |
| Hepeviridae | 6-7.5 kb | 90-170s | 无数值阈值，启发式判定 |
| Flaviviridae | 7-10 kb | 60-150s | NS3/NS5 p-distance |
| 小基因组（<5 kb） | 3-5 kb | 60-90s | BLAST + 简单判定 |

并发数建议：
- MiniMax-M2.7 API: `--parallel 4`（官方 API 有并发限制）
- GLM-4.7 (火山引擎): `--parallel 4-8`
- 本地部署: 取决于 CPU 核心数

## 错误处理

- **API 529 过载**：自动重试 3 次，间隔递增。持续 529 时记录错误并继续处理其他序列
- **超时**：默认 600 秒，可通过 `--timeout` 调整
- **连接错误**：自动重试，transient 错误不影响其他序列
- **失败序列**：错误信息写入对应的 `.txt` 文件，Excel 中标红

## 重跑失败序列

从 Excel 中筛选 Status=error 的序列，提取 FASTA 后重跑：

```bash
# 从原始 FASTA 中提取失败的序列
grep -A1 "KY370052.1\|OP288968.1" input.fasta > retry.fasta
python scripts/batch_classify.py retry.fasta -o results_retry/
```
