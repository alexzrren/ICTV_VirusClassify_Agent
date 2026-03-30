# ICTV病毒分类智能Agent实施方案

# 项目概述与目标

## 背景

国际病毒分类委员会（ICTV）维护着全球统一的病毒分类体系，最新版本MSL40包含16,213个已确认物种，覆盖从Realm到Species的完整分类层级。ICTV为每个病毒科发布了详细的分类报告（Report Chapter），其中明确定义了不同分类阶元（科、属、种）的界定标准（demarcation criteria），包括序列相似度阈值、系统发育关系、基因组结构、宿主范围等。

然而，**当前的病毒分类实践面临三个核心痛点**：

1. **标准分散且异质**：32个脊椎动物相关病毒科各自有不同的分类标准——有的用氨基酸相似度（如Picornaviridae P1蛋白>66%差异划分属），有的用遗传距离（如Paramyxoviridae L蛋白0.64-0.90 substitutions/site），有的无明确数值阈值（如Coronaviridae基于PPD聚类）。研究人员需要逐科查阅文献，学习成本极高。

2. **计算流程不统一**：不同科要求计算不同基因区域（RdRp、L蛋白、VP1、L1 ORF等）的不同指标（核酸%identity、氨基酸%identity、遗传距离等），使用不同的工具组合，缺乏统一入口。

3. **现有方法的局限性**：系统发育放置方法（如EPA-ng）虽能自动化分类，但本质上是"最近邻"方法，不等价于ICTV官方标准。**目前尚无能够严格遵循ICTV官方界定标准进行自动化分类的智能系统。**

## 核心目标

构建一个基于大语言模型（LLM）的智能代理系统，能够：

- **接受病毒序列（FASTA格式）作为输入**
- **自动识别所属病毒科**（通过BLAST/DIAMOND搜索）
- **查询该科的ICTV官方分类标准**（从结构化知识库中检索）
- **调用相应的生物信息学工具**（MAFFT、HMMER、BLAST等）执行标准要求的计算
- **比对计算结果与官方阈值**，给出分类判定（科/属/种）、置信度和证据链
- **支持新种判定**：当序列低于种级阈值时，标记为疑似新种

核心创新点：**用LLM作为推理引擎，将ICTV文档中的非结构化分类标准转化为结构化知识库，并驱动工具调用流程——不同于传统的固定流程Pipeline，Agent可根据不同科的标准动态选择计算方法和阈值。**

## 学术目标

- 发表一篇方法学论文（目标期刊：Bioinformatics / NAR / Virus Evolution / PLOS Computational Biology）
- 创新点：(1) LLM驱动的标准化病毒分类；(2) ICTV标准的结构化知识库构建；(3) 相比EPA-ng系统发育放置的方法比较

## 应用目标

- 面向病毒学研究者和病毒组学课题组的在线分类工具
- 可与现有病毒发现Pipeline（如ictv_classifier中的EPA-ng流程）互补，提供基于官方标准的"第二意见"

---

# 系统架构设计

## 总体架构

采用 **ReAct Agent + 工具集 + 知识库** 的三层架构。LLM作为推理核心，通过Claude tool_use API循环调用生物信息学工具，逐步完成分类。

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        用户界面层 (Web Application)                      │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  FASTA序列上传  │  实时推理过程展示(SSE)  │  分类结果+证据链展示  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
                                     │ FastAPI
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    LLM ReAct Agent (Claude API, tool_use)                │
│                                                                          │
│  系统提示词：ICTV分类专家角色 + 标准工作流程指引                           │
│                                                                          │
│  推理循环：                                                               │
│    1. blast_search → 识别病毒科                                          │
│    2. get_criteria → 查询该科的ICTV官方分类标准                           │
│    3. extract_hmm_region → 提取标准指定的目标基因区域                     │
│    4. compute_pairwise_identity → 计算序列相似度                          │
│    5. lookup_taxonomy → 查询参考序列的完整分类                            │
│    6. 对照阈值 → 输出分类判断 + 证据 + 置信度                            │
└──────────────────────────────────────────────────────────────────────────┘
         │                      │                        │
         ▼                      ▼                        ▼
┌──────────────────┐ ┌──────────────────────┐ ┌────────────────────────┐
│    工具层         │ │    知识库层           │ │    数据层               │
│                   │ │                       │ │                        │
│ • BLAST/DIAMOND   │ │ • criteria.json       │ │ • taxonomy.db          │
│   (科鉴定)        │ │   (32科分类标准)      │ │   (MSL40, 16213种)     │
│ • HMMER           │ │ • ICTV txt文档        │ │ • 参考序列FASTA        │
│   (区域提取)      │ │   (32科, TF-IDF索引)  │ │   (VMR GenBank下载)    │
│ • MAFFT           │ │ • ICTV原文RAG检索     │ │ • BLAST数据库          │
│   (比对+%id计算)  │ │                       │ │ • HMM profiles         │
│ • EMBOSS getorf   │ │                       │ │                        │
│   (ORF预测)       │ │                       │ │                        │
└──────────────────┘ └──────────────────────┘ └────────────────────────┘
```

## 模块详解

### 用户界面层

- **技术栈**：FastAPI后端 + 原生HTML/JS/Bootstrap前端（单页应用）
- **交互方式**：
  - FASTA文本框输入或文件上传
  - 提交后通过Server-Sent Events（SSE）实时展示Agent推理步骤
  - 结果展示：分类层级（彩色标签）+ 使用的ICTV标准 + 计算数值 + 置信度
- **API端点**：
  - `POST /classify` — 提交FASTA序列，返回job_id
  - `GET /stream/{job_id}` — SSE实时推理步骤
  - `GET /result/{job_id}` — 获取最终分类结果
  - `GET /family/{name}` — 查询某科的分类标准
  - `GET /species?q=` — MSL物种分类查询

### LLM推理引擎

- **模型**：GLM-4.7（火山引擎，Anthropic API兼容模式）或其他兼容模型，通过`ANTHROPIC_BASE_URL`和`CLAUDE_MODEL`环境变量切换
- **Agent模式**：ReAct（Reasoning + Acting），最多20步推理循环
- **系统提示词**注入：ICTV分类专家角色、标准工作流程、输出JSON格式要求
- **工具定义**：8个工具以Claude tool_use schema注册（见下文工具清单）
- **异步执行**：FastAPI BackgroundTasks + asyncio.to_thread，同步API调用和工具执行均通过线程池运行，不阻塞事件循环，支持并发分类请求

### 知识库层

- **分类标准知识库**（`data/criteria.json`）：
  - 32个已整理科的结构化分类标准
  - 每科含subfamily/genus/species三级界定标准
  - 字段：`primary_method`（方法类型）、`regions`（目标基因区域）、`thresholds`（数值阈值）、`description`（ICTV原文摘录）
  - 可通过`extract_criteria.py`调用LLM从ICTV文档自动批量提取
- **ICTV文档RAG**（TF-IDF关键词检索）：
  - 32个科的ICTV Report Chapter，共1,736个文本块
  - 支持按科或全局搜索，返回相关原文段落

### 分类法数据库

- **SQLite数据库**（`data/taxonomy.db`）：
  - 源自ICTV MSL40（Master Species List 2024）
  - 16,213个已确认物种，完整21级分类层级（Realm → Species）
  - 含ICTV_ID、基因组类型（ssRNA(+)、dsDNA等）
  - 支持按物种名、科名、属名的精确/模糊查询

---

# 技术栈选择

| 技术类别 | 采用方案 | 选择理由 |
|----------|----------|----------|
| 后端框架 | **FastAPI** | 异步支持好，自动生成API文档，SSE流式原生支持 |
| 前端 | **原生HTML/JS + Bootstrap 5** | 零构建依赖，单页面足够，开发快 |
| LLM引擎 | **Anthropic兼容API (tool_use)** | 支持Claude/GLM-4.7等兼容模型，原生Function Calling，无需LangChain封装 |
| 工具协议 | **Claude tool_use schema** | JSON Schema定义工具参数，LLM自主选择和调用 |
| 数据库 | **SQLite** | 16K物种轻量级查询，零运维，单文件易备份 |
| 知识库 | **JSON + TF-IDF文本检索** | 避免重量级向量数据库依赖，32科文档量级用关键词检索够用 |
| 序列搜索 | **BLAST+ / DIAMOND** | 业界标准，科鉴定首选 |
| 序列比对 | **MAFFT** | 高精度多序列/双序列比对 |
| 结构域提取 | **HMMER + EMBOSS getorf** | HMM profile搜索提取保守区域（RdRp、L蛋白等） |
| Python环境 | **micromamba base (Python 3.11)** | 复用现有环境，兼容ictv_classifier工具链 |
| 容器化 | 暂不采用 | 单机部署，micromamba管理依赖即可 |

---

# 工具整合与知识库构建

## 核心工具清单

| 工具名称 | Agent中的函数名 | 功能描述 | 输入 | 输出 |
|----------|-----------------|----------|------|------|
| BLAST+MAFFT一体化 | `blast_and_compare` | **核心工具**：BLAST搜索 + 自动获取参考序列 + 全局pairwise identity计算，一步完成 | 原始序列 + 类型 | Top hits（含BLAST %id + 全局%id） |
| BLAST+ (blastn) | `blast_search` | 核酸序列搜索，识别病毒科 | 原始核酸序列 | Top hits（科、accession、%identity、e-value） |
| DIAMOND (blastp) | `blast_search` | 蛋白序列快速搜索 | 原始蛋白序列 | Top hits |
| 参考序列获取 | `fetch_reference_sequence` | 从本地参考库按accession获取完整序列 | accession/关键词 | 原始序列 + header + 科名 |
| HMMER (hmmsearch) | `extract_hmm_region` | 用HMM profile提取目标基因区域（如RdRp） | 核酸序列 + 科名 | 蛋白子序列 |
| EMBOSS (getorf) | （内部调用） | ORF预测，核酸→蛋白翻译 | 核酸序列 | 所有ORF蛋白序列 |
| MAFFT | `compute_pairwise_identity` | 双序列比对 + 相似度计算 | 两条序列 | pairwise %identity |
| SQLite查询 | `lookup_taxonomy` | MSL40物种分类查询 | 物种名/accession | 完整分类层级 |
| JSON查询 | `get_criteria` | ICTV分类标准检索 | 科名 + 分类级别 | 结构化标准（方法、阈值、区域） |
| TF-IDF检索 | `search_ictv_docs` | ICTV原文段落搜索 | 自然语言查询 | 相关文本块 |
| 物种列表 | `list_reference_species` | 列出指定科/属的MSL40物种列表 | 科名/属名 | 物种名列表 |

## 分类标准知识库

### 已提取的32科分类标准

| 科名 | 种级标准方法 | 目标基因区域 | 数值阈值 |
|------|-------------|-------------|----------|
| Coronaviridae | PPD聚类 | replicase ORF1b | 无固定阈值（按属不同） |
| Picornaviridae | aa identity + 系统发育 | P1, 2C, 3C, 3D | 属级：P1 >66%差异, 2C/3C/3D >64%差异 |
| Paramyxoviridae | 遗传距离 | L蛋白（完整） | 亚科级：0.64-0.90 subst/site |
| Flaviviridae | 系统发育 | NS5 RdRp | 无明确数值 |
| Togaviridae | aa identity + 生态 | 全编码区 | 种级：≥10% aa差异 |
| Caliciviridae | aa identity | VP1衣壳蛋白 | 属级：>60% aa差异 |
| Papillomaviridae | nt identity | L1 ORF | 亚科<45% aa; 属<60% nt; 种<90% nt |
| Polyomaviridae | nt identity | 全基因组 | 种级：>81% nt identity为同种 |
| Paramyxoviridae | 遗传距离 | L蛋白 | 见上 |
| Rhabdoviridae | nt/aa identity + 生态 | L蛋白, N蛋白 | 属级因属而异 |
| Adenoviridae | 系统发育 + 生物学 | DNA polymerase, hexon | 无固定数值 |
| Hantaviridae | 系统发育 + 生态 | L (RdRp), S (NP) | 无固定数值 |
| Filoviridae | 遗传距离 + 生态 | L (RdRp), GP | 无固定数值 |
| Arenaviridae | 系统发育 + 生态 | L (RdRp), S (NP) | 无固定数值 |

### LLM自动提取流程

对于未手动整理的18个科，提供`scripts/extract_criteria.py`脚本：
- 输入：`ictv_txt/*.txt`（ICTV Report Chapter纯文本）
- 过程：调用Claude API，使用专用系统提示词提取结构化JSON
- 输出：追加到`data/criteria.json`
- 每科提取约需1次API调用，可增量运行

---

# 智能决策引擎设计

## 推理流程

Agent采用ReAct模式，推理与工具调用交替进行。以一条未知病毒核酸序列为例：

```
用户上传 FASTA 序列
        │
        ▼
Step 1: Agent调用 blast_search(sequence, "nucleotide")
        → 返回Top 10 BLAST hits，最佳命中科: Coronaviridae, 85% identity
        │
        ▼
Step 2: Agent推理: "最佳命中属于Coronaviridae，查询该科分类标准"
        Agent调用 get_criteria("Coronaviridae", "all")
        → 返回: 种级标准基于replicase ORF1b的PPD聚类，无固定阈值
        │
        ▼
Step 3: Agent推理: "需要提取RdRp区域进行比较"
        Agent调用 extract_hmm_region(sequence, "Coronaviridae")
        → 返回: 提取的RdRp蛋白序列（412 aa）
        │
        ▼
Step 4: Agent调用 compute_pairwise_identity(query_rdrp, ref_rdrp, is_protein=true)
        → 返回: 92.3% aa identity with closest reference
        │
        ▼
Step 5: Agent调用 lookup_taxonomy(closest_ref_accession)
        → 返回: Betacoronavirus gravedinis, Coronaviridae, Orthocoronavirinae
        │
        ▼
Step 6: Agent综合判断:
        "查询序列与Betacoronavirus gravedinis的RdRp aa identity为92.3%，
         根据Coronaviridae的PPD分类标准和参考文献，该相似度水平提示
         可能属于同一种或近缘新种。置信度: Medium。"
        │
        ▼
输出结构化JSON分类结果 + 证据链
```

## 动态工具选择

**与传统Pipeline的关键区别**：Agent不执行固定流程，而是根据每个科的分类标准动态决定：

| 科 | Agent的工具调用链 |
|----|-----------------|
| Papillomaviridae | BLAST → get_criteria → **提取L1 ORF** → **计算nt identity** → 与90%阈值比较 |
| Paramyxoviridae | BLAST → get_criteria → **提取完整L蛋白** → **计算遗传距离** → 与0.64-0.90阈值比较 |
| Caliciviridae | BLAST → get_criteria → **提取VP1蛋白** → **计算aa identity** → 与60%阈值比较 |
| Coronaviridae | BLAST → get_criteria → **提取ORF1b** → **计算PPD** → 查询RAG获取更多上下文 |

这种灵活性正是LLM Agent相比固定Pipeline的核心优势。

## 参数选择

工具参数由Agent根据上下文推理选择：
- BLAST e-value阈值：默认1e-5，若初次无命中可放宽
- MAFFT模式：双序列用`--auto`，多序列用`--localpair`
- HMMER域阈值：默认score≥30
- Pairwise identity计算：排除双gap列

---

# 数据资源

## 已有数据

| 数据 | 来源 | 规模 | 状态 |
|------|------|------|------|
| ICTV MSL40 | ICTV_Master_Species_List_2024_MSL40.v2.xlsx | 16,213物种 | ✅ 已导入SQLite |
| ICTV Report Chapters | ictv.global/report/chapter/ | 32科HTML/TXT | ✅ 已下载并清洗 |
| VMR (Virus Metadata Resource) | VMR_MSL40.v2.20251013.xlsx | GenBank accession列表 | ✅ 已获取 |
| HMM Profiles (RdRp) | 自建 | 32科 | ✅ 已有（ictv_classifier共享） |
| 分类标准知识库 | 手动+LLM提取 | 14科结构化JSON | ✅ 已构建 |

## 待构建数据

| 数据 | 构建方式 | 预计规模 | 脚本 | 状态 |
|------|----------|----------|------|------|
| 参考序列FASTA | 从VMR accession下载NCBI | ~2-5 GB | `scripts/download_reference_seqs.py` | ⚠️ Coronaviridae已下载（59条），其余待下载 |
| BLAST核酸数据库 | makeblastdb | ~500 MB | `scripts/build_blast_db.py` | ⚠️ 已构建（仅含Coronaviridae），需扩展 |

---

# 项目结构

```
ictv_agent/
├── backend/
│   ├── main.py                  # FastAPI应用，API路由定义
│   ├── agent.py                 # LLM ReAct Agent（Claude tool_use循环）
│   ├── models.py                # Pydantic数据模型
│   ├── tools/
│   │   ├── blast.py             # BLAST/DIAMOND搜索封装
│   │   ├── hmmer.py             # HMMER区域提取封装
│   │   ├── alignment.py         # MAFFT比对 + pairwise identity计算
│   │   └── taxonomy.py          # SQLite分类查询
│   └── knowledge/
│       ├── criteria.py          # 分类标准知识库加载/查询
│       └── rag.py               # TF-IDF文档检索（32科ICTV文本）
├── frontend/
│   └── index.html               # 单页Web界面（Bootstrap 5 + SSE）
├── scripts/
│   ├── extract_criteria.py      # LLM批量提取分类标准
│   ├── build_taxonomy_db.py     # MSL40 Excel → SQLite
│   ├── download_reference_seqs.py # VMR accession → NCBI FASTA
│   ├── build_blast_db.py        # 构建BLAST/DIAMOND数据库
│   └── build_vectordb.py        # 构建文本向量索引（可选）
├── data/
│   ├── taxonomy.db              # MSL40分类数据库（16,213物种）
│   ├── criteria.json            # 结构化分类标准（32科）
│   ├── references/              # 参考序列FASTA（按科组织）
│   ├── db/                      # BLAST数据库文件
│   └── vectordb/                # 文本向量索引（可选）
├── requirements.txt
├── run.sh                       # 一键启动脚本（含默认API配置）
├── stop.sh                      # 关闭服务脚本
└── ICTV病毒分类智能Agent实施方案.md  # 本文档
```

---

# 环境依赖

## Python包

```
fastapi>=0.115          # Web框架
uvicorn[standard]>=0.30 # ASGI服务器
python-multipart        # 文件上传
anthropic>=0.50         # Claude API客户端
openpyxl>=3.1           # Excel读取
biopython>=1.83         # NCBI序列下载
pydantic>=2.5           # 数据验证
```

## 命令行工具（micromamba base环境已有）

```
mafft                   # 序列比对
hmmer                   # HMM搜索
emboss (getorf)         # ORF预测
```

## 需额外安装

```
blast+ (makeblastdb, blastn)    # conda install -c bioconda blast
diamond                          # conda install -c bioconda diamond（可选）
```

## 环境变量

```bash
ANTHROPIC_API_KEY=...                  # 必需：LLM API密钥
ANTHROPIC_BASE_URL=https://...         # 可选：自定义API地址（兼容Anthropic协议的第三方服务）
CLAUDE_MODEL=glm-4.7                   # 可选：覆盖默认模型（默认glm-4.7）
NCBI_EMAIL=your@email.com              # 参考序列下载用
NCBI_API_KEY=...                       # 可选：提高NCBI限速
```

---

# 快速启动

```bash
# 1. 下载参考序列（首次运行，可按需只下载部分科）
python scripts/download_reference_seqs.py \
    --vmr ../VMR_MSL40.v2.20251013.xlsx \
    --families ../vf.list \
    --email your@email.com \
    --api-key YOUR_NCBI_API_KEY

# 2. 构建BLAST数据库
python scripts/build_blast_db.py

# 3. 启动服务（API配置已内置于run.sh）
bash run.sh          # 默认端口18231
bash run.sh 9000     # 或指定其他端口

# 4. 访问 http://localhost:18231

# 5. 关闭服务
bash stop.sh         # 默认关闭18231端口
bash stop.sh 9000    # 或指定端口
```

---

# 与EPA-ng方法的比较

| 维度 | EPA-ng系统发育放置（ictv_classifier） | ICTV标准Agent（ictv_agent） |
|------|--------------------------------------|---------------------------|
| 分类依据 | 系统发育树上最近邻（LCA） | ICTV官方界定标准（方法+阈值） |
| 优点 | 全自动，无需人工整理标准 | 严格遵循官方标准，分类结果可溯源 |
| 缺点 | 不等价于ICTV标准，某些情况可能误判 | 依赖知识库完整性，部分科标准模糊 |
| 适用场景 | 大批量快速分类筛查 | 需要严格标准依据的正式分类 |
| 新种判定 | pendant length + LWR + species conflict | 低于种级阈值 + LLM综合推理 |
| 可解释性 | 低（仅给出LCA结果） | 高（给出使用的标准、阈值、计算值） |
| 互补性 | 两套方法可互相验证：EPA-ng给出快速初筛，Agent给出标准化终判 |

---

# 实施步骤

### 总体时间线

| 阶段 | 周次 | 主要任务 | 产出 |
|------|------|----------|------|
| **数据准备** | 第1-2周 | 参考序列下载、BLAST库构建、剩余18科标准提取 | 完整数据层 |
| **核心开发** | 第3-4周 | Agent调优、工具联调、端到端测试 | 可运行的分类系统 |
| **验证与评估** | 第5-6周 | 已知序列验证、与EPA-ng对比、边界case分析 | 验证数据集和评估报告 |
| **论文撰写** | 第7-8周 | 方法描述、实验设计、结果分析、图表制作 | 论文初稿 |

### 当前进展

- [x] 分类法数据库构建（MSL40 → SQLite, 16,213物种）
- [x] ICTV文档下载与清洗（32科HTML/TXT）
- [x] 分类标准知识库构建（32科结构化JSON）
- [x] RAG文档检索（32科, 1,736块, TF-IDF）
- [x] 工具层实现（BLAST、HMMER、MAFFT、taxonomy查询、blast_and_compare一体化）
- [x] LLM ReAct Agent实现（Anthropic兼容API tool_use, 8工具, 20步循环）
- [x] FastAPI后端（所有端点已测试通过，异步不阻塞事件循环）
- [x] Web前端（FASTA上传 + SSE实时推理展示 + family_hint）
- [x] Coronaviridae参考序列下载（59条）+ BLAST数据库构建
- [x] 端到端分类测试（Coronaviridae云南样本，BLAST命中 + 全局pairwise identity计算通过）
- [ ] 其余31科参考序列下载与BLAST库扩展
- [ ] 与EPA-ng结果交叉验证
- [ ] 多科测试与分类准确率评估

---

# 风险与对策

| 风险 | 应对措施 |
|------|----------|
| 部分科的ICTV标准无明确数值阈值 | Agent输出"参考标准无明确阈值，建议人工判断"，并附ICTV原文段落供参考 |
| LLM提取的分类标准不准确 | criteria.json中保留ICTV原文摘录（description字段），供用户核对；关键阈值人工审核 |
| BLAST参考库不够全面 | 逐步扩展参考序列；Agent在无命中时提示用户上传参考序列 |
| 不同科的计算方法差异大 | criteria.json的primary_method字段驱动Agent动态选择工具，天然支持异质性 |
| LLM推理过程偶发错误 | 限制最大步数（20步）；输出完整推理日志供审查；结果中标注置信度 |
| API调用成本 | 使用Claude Sonnet（性价比高）；单次分类约需3-8次tool call |

---

# 论文撰写计划

## 创新点

1. **首个严格遵循ICTV官方标准的自动化分类系统**——区别于序列相似度或系统发育放置方法
2. **LLM驱动的非结构化标准→结构化知识库转化**——解决不同科标准异质性问题
3. **动态工具编排**——Agent根据标准要求选择计算方法，而非固定Pipeline
4. **可解释的分类决策**——输出证据链（使用的标准、阈值、计算值），而非黑箱结果

## 实验设计

- **数据集**：从VMR参考序列中抽取已知分类的序列作为测试集（ground truth来自MSL40）
- **对比实验**：
  - Agent分类结果 vs. MSL40官方分类 → 计算准确率
  - Agent分类结果 vs. EPA-ng系统发育放置结果 → 一致性分析
  - Agent分类时间 vs. 人工查阅标准+手动计算 → 效率对比
- **边界案例**：
  - 疑似新种序列（来自ChinaBatVirome项目的未分类蝙蝠病毒序列）
  - 跨属边界序列
  - 标准模糊的科（无数值阈值）
- **消融实验**：
  - 有无criteria.json知识库的分类差异
  - 不同LLM模型的表现（Sonnet vs Opus vs Haiku）

## 目标期刊

Bioinformatics, Nucleic Acids Research, Virus Evolution, PLOS Computational Biology

---

# 任务分工

（待补充）

---

# 进展日志

## 2026年3月30日

端到端分类流程跑通：
- **数据层**：下载Coronaviridae参考序列（59条），构建BLAST核酸数据库
- **新增工具**：`blast_and_compare`（BLAST + 全局pairwise identity一步完成）、`fetch_reference_sequence`（按accession获取参考序列）
- **关键修复**：
  - 同步API调用和工具执行改用`asyncio.to_thread`，解决阻塞事件循环导致请求卡死的问题
  - 修复SSE步骤重复显示（replay/queue竞争条件）
  - BLAST结果去重（同一subject多条HSP只保留最高bitscore）
  - 始终注入完整序列到BLAST工具（防止模型只发前500字符）
  - 新增fallback结果提取（从工具调用结果中自动构建taxonomy和evidence）
- **LLM配置**：支持自定义`ANTHROPIC_BASE_URL`，已接入火山引擎GLM-4.7
- **测试结果**：云南冠状病毒序列 → BLAST命中Murine hepatitis virus (84.77% local, 61.62% global pairwise identity)
- **脚本**：新增`stop.sh`关闭服务脚本

## 2026年3月29日

完成核心系统搭建：
- **数据层**：taxonomy.db（16,213物种），criteria.json（32科），32科TF-IDF文本索引
- **工具层**：BLAST/DIAMOND封装、HMMER区域提取、MAFFT pairwise identity、SQLite taxonomy查询
- **Agent**：Claude tool_use ReAct循环，6个工具，20步上限
- **后端**：FastAPI 6个端点全部测试通过
- **前端**：单页HTML + SSE实时推理展示
- **数据脚本**：参考序列下载、BLAST库构建脚本就绪，待运行
