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
  - 结果展示：分类层级（彩色标签）+ 使用的ICTV标准 + 计算数值 + 置信度 + Reasoning Summary
  - 历史面板：显示已缓存的分类记录，点击可直接加载结果
  - 重复序列自动识别：相同序列（忽略FASTA header差异）直接返回缓存结果
- **API端点**：
  - `POST /classify` — 提交FASTA序列，返回job_id（自动查缓存，命中则即时返回`cached: true`）
  - `GET /stream/{job_id}` — SSE实时推理步骤
  - `GET /result/{job_id}` — 获取最终分类结果
  - `GET /family/{name}` — 查询某科的分类标准
  - `GET /species?q=` — MSL物种分类查询
  - `GET /history?limit=20` — 获取最近的缓存分类记录
  - `GET /cache/{seq_hash}` — 按序列哈希获取缓存结果

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
| DEmARC PUD | `corona_pud_classify` | **Coronaviridae专用**：翻译ORF1ab→提取5个保守结构域→计算PUD→按Table 4阈值分配亚属/属/亚科 | 核苷酸基因组（≥20 kb） | 分类层级 + Top hits + PUD值 |
| HMM区域提取 | `extract_target_region` | **通用多科工具**：用各科专用HMM profile从基因组中提取ICTV标准指定的目标蛋白区域（L蛋白、VP1、NS5 RdRp等），供后续pairwise identity比较 | 核苷酸基因组 + 科名 | 各区域蛋白序列 + 长度 |

## 分类标准知识库

### 分类标准完整性评估（32科）

当前`data/criteria.json`覆盖32个目标科，其中 **21科有明确数值阈值**，11科依赖系统发育/生物学标准无数值阈值。

**有明确数值阈值的21科（可直接自动化分类）：**

| 科名 | 种级方法 | 关键阈值 |
|------|---------|---------|
| Coronaviridae | DEmARC PUD（5结构域） | 物种 ≤7.5%; 亚属 ≤14.2%; 属 ≤36%; 亚科 ≤51% |
| Picornaviridae | aa identity（按属） | P1: Enterovirus <30%差异，Hepatovirus <40% |
| Paramyxoviridae | L蛋白遗传距离 | 分支长度 ≤0.03 sub/site（种级） |
| Papillomaviridae | L1 ORF nt identity | 属 <60% nt; 种 <90% nt |
| Polyomaviridae | 全基因组 nt identity | 种级 >81% nt |
| Arteriviridae | RdRp aa identity | 属 <70% aa |
| Parvoviridae | NS1 aa identity | 同种 ≥85% aa（4属统一阈值） |
| Anelloviridae | ORF1 nt identity | >65% nt同种 |
| Circoviridae | Rep aa identity | >80% aa同种 |
| Hantaviridae | N蛋白 aa identity | 属 <7% aa差异 |
| Nodaviridae | RNA2 nt identity | >80% nt同种 |
| Sedoreoviridae | VP6 aa identity | Rotavirus: >53% aa |
| Arenaviridae | S/L段 nt identity | S段 <80%、L段 <76%为不同种 |
| Bornaviridae | PASC全基因组 | ≥72-75%同种 |
| Phenuiviridae | RdRp aa identity | 属 <50% aa |
| Rhabdoviridae | 按属（N/L蛋白） | Lyssavirus N基因 ≥80% nt同种 |
| Filoviridae | PASC | <77%（种级阈值） |
| Adenoviridae | DNA pol + hexon | pol差异 >10-15%分属 |
| Peribunyaviridae | L蛋白 aa identity | Orthobunyavirus ≥96% aa同种 |
| Flaviviridae | NS3/NS5B aa identity | Hepacivirus: NS3 <75%, NS5B <70%分种 |
| Amnoonviridae | 基因组 nt identity | >80% nt同种 |

**无明确数值阈值的11科（依赖系统发育/生物学标准）：**

| 科名 | 主要原因 |
|------|---------|
| Caliciviridae | VP1序列分歧度与宿主/抗原性联合判定，无固定阈值 |
| Hepeviridae | 亚型内多样性大，需结合宿主和系统发育，无固定阈值 |
| Orthoherpesviridae | 大型DNA病毒，以基因组共线性+系统发育分群，无统一数值 |
| Pneumoviridae | 物种少（仅5种），以系统发育+宿主特异性区分 |
| Poxviridae | 双链DNA病毒，标准复杂（基因含量+系统发育），无单一阈值 |
| Asfarviridae | 单科单属，近缘种少，标准基于比较基因组 |
| Picobirnaviridae | 目前缺乏经过验证的数值阈值 |
| Astroviridae | VP90/VP70 aa identity用于属级，但种级界定标准不统一 |
| Nairoviridae | 综合系统发育+宿主范围，无数值阈值 |
| Spinareoviridae | 分段基因组，各节段标准不一致 |
| Togaviridae | 物种少，以系统发育分群为主 |

### 属级种间界定标准（genus_criteria.json）

对于在科级标准上需要属级细化的病毒科（如Picornaviridae、Paramyxoviridae、Rhabdoviridae等），提供`data/genus_criteria.json`：
- **覆盖98个属**，通过`scripts/fetch_genus_criteria.py`从ICTV网站获取属级Report页面后解析
- **52个属有明确数值阈值**，46个为单型属（无种间比较意义）
- 可通过`backend/knowledge/criteria.py`的`get_genus_criteria(family, genus)`接口查询

### Coronaviridae DEmARC PUD分类（专用子系统）

Coronaviridae采用基于氨基酸PUD（Pairwise Uncorrected Distance）的DEmARC分类框架，实现亚属级分辨率。

**5个保守复制酶结构域**（在pp1ab中的位置，基于SARS-CoV-2 MN908947）：
| 结构域 | Nsp | aa坐标 | 大小 |
|--------|-----|--------|------|
| 3CLpro | Nsp5 | 3264-3569 | 306 aa |
| NiRAN | Nsp12 N端 | 4393-4535 | 143 aa |
| RdRp | Nsp12催化核心 | 4536-4932 | 397 aa |
| ZBD | Nsp13锌指 | 5316-5443 | 128 aa |
| HEL1 | Nsp13解旋酶 | 5444-5836 | 393 aa |

**PUD阈值（Ziebuhr et al. 2021, Table 4）：**
| 分类阶元 | PUD阈值 |
|---------|--------|
| 种 | ≤7.5% |
| 亚属 | ≤14.2% |
| 属 | ≤36.0% |
| 亚科 | ≤51.0% |
| 科 | ≤68.1% |

**工具实现**（`backend/tools/corona_pud.py`）：
1. **基因组方向检测**（`orient_genome()`）：扫描6个reading frame（正链3个 + 反向互补链3个），找最长无终止密码子区��（即ORF1a），一步确��正确链方向��ORF1a大致位置。解决宏基因组contig���能以反向互补方向组装的问题
2. **ORF1a起始定位**（`find_orf1a_start()`）���在`orient_genome`提示位置附近搜索ATG起始密码子，不再限制于基因组前1000 nt，支持5'截断的contig
3. **移码位点检测**（`find_frameshift_site()`）：在ORF1a内枚举所有 `TTTAAAC` 候选位置（搜索窗口 ORF1a起始+8000 至 +20000 nt），对每个候选验证-1框架的ORF1b翻译长度（>1500 aa），选取产生最长ORF1b的位置作为真实移码位点
4. **结构域提取**：以坐标缩放法为主（`extract_domains_by_coords`，基于pp1ab长度��SARS-CoV-2的比例缩放5个结构域坐标），HMM搜��为辅（仅当HMM结果大小更接近���期时才替换）。query和reference统一使用相同提取方法，避免不对称性。HMM结果有大小验���（拒绝>2x预期大小的hit）
5. MAFFT比对 + 计算PUD，与59条参考序列逐一比较
6. 按Table 4阈值划分分类阶元，查SQLite获取完整分类

> **设计演变**：
> - v1���硬编码SARS-CoV-2移码位点坐标 → 对其他冠状病毒失效
> - v2：取搜索区间第一个 `TTTAAAC` → SARS-CoV-1命中假阳性位点，PUD虚高79%
> - v3：枚举所有TTTAAAC + ORF1b长��验证 → 解决假阳性，但仍假设正向链
> - **v4（当前）**：ORF分析自动检测链方向 + 坐标缩放为主的结构域提取 → 支持宏基因组反向互补contig，全部4条测试contig（含2条revcomp）均成功��取5/5结构域

**HMM建库**（`scripts/build_corona_hmms.py`）：
- 使用`corona_pud.translate_orf1ab()`自动检测���条参考序列的移码位点（而非硬编码SARS-CoV-2坐标），坐标缩放法提取结构域种子序列
- 58条种子序列/结构域 → MAFFT多序列比对 → hmmbuild → hmmpress
- 输出：`data/hmm/CoV_5domains.hmm`（可直接用于hmmsearch）

**验证结果：**
- SARS-CoV-1 (AY274119) vs 自身: PUD=0.0% → same_species ✓
- SARS-CoV-2 (MN908947) vs SARS-CoV-1 (AY274119): PUD=3.47% → same_species ✓（均为*Severe acute respiratory syndrome-related coronavirus*，同属 Sarbecovirus 亚属）
- SARS-CoV-2 vs 蝙蝠Beta冠状病毒: PUD=27-35% → same_genus ✓（均为Betacoronavirus属）
- SARS-CoV-2 vs Alpha冠状病毒: PUD=45-60% → same_subfamily ✓（同属Orthocoronavirinae）
- 云南啮齿动物宏基因组contig（含2条反向互补）：4/4成功分类，PUD=0.73%-30.76%

### LLM自动提取流程

对于未手动整理的科，提供`scripts/extract_criteria.py`脚本：
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
| 分类标准知识库 | 手动+LLM提取 | 32科结构化JSON | ✅ 已构建（21科有数值阈值） |
| 参考序列FASTA | 从ictv_classifier复制 | 34科6,476条 | 从`ictv_classifier/reference/`复制 | ✅ 全部34科已就位 |
| BLAST核酸数据库 | makeblastdb | 34科合并 | `scripts/build_blast_db.py` | ✅ 已构建（6,476条序列） |
| 各科HMM Profiles | tblastn+MAFFT+hmmbuild | 24科30个区域 | `scripts/build_family_hmms.py` | ✅ 已构建 |
| HMM目标配置 | 手动整理 | 24科 | `data/hmm_targets.json` | ✅ 已创建 |

---

# 项目结构

```
ictv_agent/
├── backend/
│   ├── main.py                  # FastAPI应用，API路由定义（含缓存和历史端点）
│   ├── agent.py                 # LLM ReAct Agent（Claude tool_use循环，10个工具）
│   ├── models.py                # Pydantic数据模型
│   ├── cache.py                 # SQLite结果缓存（序列SHA-256去重，历史记录查询）
│   ├── tools/
│   │   ├── blast.py             # BLAST/DIAMOND搜索封装
│   │   ├── hmmer.py             # HMMER区域提取封装（24科HMM自动发现 + 多区域提取）
│   │   ├── alignment.py         # MAFFT比对 + pairwise identity计算
│   │   ├── taxonomy.py          # SQLite分类查询
│   │   └── corona_pud.py        # DEmARC PUD分类流水线（Coronaviridae专用）
│   └── knowledge/
│       ├── criteria.py          # 分类标准知识库加载/查询（含属级接口）
│       └── rag.py               # TF-IDF文档检索（32科ICTV文本）
├── frontend/
│   └── index.html               # 单页Web界面（Bootstrap 5 + SSE + 历史面板）
├── scripts/
│   ├── extract_criteria.py      # LLM批量提取分类标准
│   ├── build_taxonomy_db.py     # MSL40 Excel → SQLite
│   ├── download_reference_seqs.py # VMR accession → NCBI FASTA
│   ├── build_blast_db.py        # 构建BLAST/DIAMOND数据库
│   ├── fetch_genus_criteria.py  # 下载ICTV属级Report页面并解析
│   ├── build_corona_hmms.py     # 构建Coronaviridae 5结构域HMM profiles
│   └── build_family_hmms.py     # 通用多科HMM构建（24科30区域，tblastn+MAFFT+hmmbuild）
├── data/
│   ├── taxonomy.db              # MSL40分类数据库（16,213物种）
│   ├── criteria.json            # 结构化分类标准（32科；21科有数值阈值）
│   ├── genus_criteria.json      # 属级种间界定标准（98属；52属有数值阈值）
│   ├── cache.db                 # 分类结果缓存（自动创建）
│   ├── hmm_targets.json          # 各科HMM目标区域配置（24科30区域）
│   ├── hmm/
│   │   ├── CoV_5domains.hmm     # Coronaviridae 5结构域HMM
│   │   └── {Family}_targets.hmm # 各科HMM profiles（24科，由build_family_hmms.py生成）
│   ├── references/              # 参考序列FASTA（34科，按科组织）
│   └── db/                      # BLAST数据库文件（34科6,476条合并）
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
- [x] 分类标准知识库构建（32科结构化JSON；21科有数值阈值，11科暂无）
- [x] 属级分类标准扩展（genus_criteria.json；98属，52属有数值阈值）
- [x] 属级ICTV页面抓取（fetch_genus_criteria.py；98成功，12失败）
- [x] RAG文档检索（32科, 1,736块, TF-IDF）
- [x] 工具层实现（BLAST、HMMER、MAFFT、taxonomy查询、blast_and_compare一体化）
- [x] **Coronaviridae DEmARC PUD子系统**（corona_pud.py；亚属级分辨率）
- [x] **Coronaviridae HMM profiles**（build_corona_hmms.py；5结构域，59参考序列）
- [x] LLM ReAct Agent实现（Anthropic兼容API tool_use, **10工具**, 20步循环）
- [x] FastAPI后端（所有端点已测试通过，异步不阻塞事件循环）
- [x] Web前端（FASTA上传 + SSE实时推理展示 + family_hint + **历史面板**）
- [x] Coronaviridae参考序列下载（59条）+ BLAST数据库构建
- [x] 端到端分类测试（Coronaviridae云南样本，BLAST命中 + 全局pairwise identity计算通过）
- [x] **ORF分析自动链方向检测**（支持反向互补宏基因组contig）
- [x] **坐标缩放为主的结构域提取**（解决HMM profile过宽问题，query/reference对称）
- [x] **结果缓存 + 历史记录**（SQLite，序列SHA-256去重，前端历史面板）
- [x] **Reasoning Summary修复**（前端始终显示，后端自动生成结构化摘要）
- [x] **全部34科参考序列导入 + BLAST库重建**（6,476条序列，从ictv_classifier复制）
- [x] **24科HMM Profile库构建**（30个目标蛋白区域，通用build_family_hmms.py脚本）
- [x] **Agent新增extract_target_region工具**（第10个工具，多科HMM蛋白提取）
- [ ] 多科端到端测试与分类准确率评估
- [ ] 与EPA-ng结果交叉验证

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

## 2026年4月8日（第二次）

**全科扩展：34科参考序列 + 24科HMM Profile库 + Agent第10个工具**

**1. 参考序列全科导入**
- 从`ictv_classifier/reference/`复制34科共6,476条参考序列到`data/references/`
- 重建BLAST核酸数据库（`makeblastdb`，34科合并）
- 无需NCBI联网下载，全部本地已有

**2. 通用HMM构建系统**
- 新增`data/hmm_targets.json`：24科30个目标蛋白区域配置（包括L蛋白、VP1、NS3、NS5 RdRp、DNA聚合酶、N蛋白、L1 ORF、NS1/Rep、3D聚合酶、P1衣壳等）
- 新增`scripts/build_family_hmms.py`（~300行）：通用HMM构建脚本
  - 算法：从参考FASTA取种子ORF → tblastn搜索所有参考序列提取同源蛋白 → MAFFT比对 → hmmbuild
  - 支持单科构建（`--family`）和全部构建，`--dry-run`预览模式
  - MAX_SEEDS=80防止过大科（Rhabdoviridae 654条）构建过慢
- 全部24科30个HMM region构建成功，输出`data/hmm/{Family}_targets.hmm`

**覆盖的科和目标区域：**

| 类型 | 科 | 目标蛋白 | 种子数 |
|------|---|---------|-------|
| RNA（L蛋白） | Paramyxoviridae, Rhabdoviridae, Peribunyaviridae, Pneumoviridae, Filoviridae, Bornaviridae | L protein | 4-80 |
| RNA（RdRp） | Phenuiviridae, Nairoviridae, Hantaviridae, Arenaviridae, Sedoreoviridae, Spinareoviridae | L/RdRp | 5-80 |
| RNA（多区域） | Flaviviridae (NS3+NS5), Picornaviridae (P1+3D), Rhabdoviridae (L+N) | 多蛋白 | 21-80 |
| RNA（其他） | Caliciviridae (VP1), Togaviridae (nsP4) | 衣壳/RdRp | 25-52 |
| DNA | Papillomaviridae (L1), Adenoviridae (DNA pol), Orthoherpesviridae (DNA pol), Polyomaviridae (VP1+LTAg) | 结构/复制蛋白 | 4-80 |
| ssDNA | Anelloviridae (ORF1), Circoviridae (Rep), Parvoviridae (NS1) | 复制蛋白 | 26-80 |
| 其他 | Hepeviridae (ORF1) | 非结构蛋白 | 4 |

**3. Agent集成**
- `backend/tools/hmmer.py`重构：
  - `_resolve_hmm_path()`：自动查找各科HMM profile（`{Family}_targets.hmm` → `CoV_5domains.hmm` → legacy RdRp）
  - `extract_all_regions()`：多区域提取，返回`{region: protein_seq}`字典
  - `list_available_hmms()`：列出所有可用HMM及其区域
  - `_run_getorf()`：修复EMBOSS getorf环境变量问题
- `backend/agent.py`：新增第10个工具`extract_target_region`，系统提示词增加HMM区域提取指导
- 序列自动注入：`extract_target_region`的`genome_nt`字段与`corona_pud_classify`同样自动注入完整序列

**验证**：Paramyxoviridae L蛋白（1600 aa）、Flaviviridae NS3+NS5 RdRp（1959 aa）、Papillomaviridae L1 ORF（642 aa）均成功提取。

---

## 2026年4月8日

**三项重大改进：ORF方向检测 + 结构域提取重构 + 结果缓存/历史**

**1. 基因组方向自动检测（`orient_genome()`）**

问题：宏基因组contig可能以反向互补方向组装，`translate_orf1ab`只检查正向链，导致pp1ab翻译失败（仅26-77 aa），domain extraction返回空，最终PUD分类失败。

修复：新增`orient_genome()`函数，扫描6个reading frame（3 forward + 3 revcomp），利用冠状病毒ORF1a是基因组最长ORF（>4000 aa）的特性，找到包含最长stop-free区域的链方向。`find_orf1a_start()`改为从longest-ORF位置附近搜索ATG，支持5'截断contig。

验证：4条云南啮齿动物冠状病毒contig（2条正向、2条反向互补），全部成功提取5/5结构域并完成PUD分类。

**2. 结构域提取策略重构**

问题：HMM profiles对RdRp/HEL1匹配区域远大于预期（1208 vs 397 aa），query用HMM、reference用坐标缩放的不对称提取导致PUD不准确。根因是`build_corona_hmms.py`用硬编码SARS-CoV-2坐标翻译所有参考序列的ORF1ab。

修复：
- `build_corona_hmms.py`：改用`corona_pud.translate_orf1ab()`自动检测移码位点，坐标缩放法提取种子序列
- `corona_pud.py`：以坐标缩放为主（`extract_domains_by_coords`），HMM仅作为refinement（只在HMM域大小更接近预期时替换）。query/reference统一用相同方法。HMM结果加入大小验证（拒绝>2x预期的hit）

**3. 结果缓存与历史记录**

- 新增`backend/cache.py`：SQLite数据库，以清洗后序列（去header、去空白、大写）的SHA-256为key缓存ClassifyResult
- `POST /classify`先查缓存，相同序列（即使FASTA header不同）直接返回，跳过Agent推理
- 新增`GET /history`和`GET /cache/{hash}`端点
- 前端新增History面板，显示最近分类记录，点击可加载缓存结果
- 缓存命中时显示蓝色"(cached)"标识

**4. Reasoning Summary修复**

- 前端：去掉"有SSE steps就清空reasoning"的逻辑，始终显示；新增`buildReasoningSummary()`从taxonomy/evidence自动生成摘要作为fallback
- 后端：`_build_result_from_logs`生成结构化reasoning（如"Identified as Coronaviridae. global pairwise identity 85%..."）

---

## 2026年4月7日（第二次）

**Bugfix：修复冠状病毒ORF1ab移码位点检测错误导致分类失准**

**问题**：`corona_pud_classify` 对 SARS-CoV-1 (AY274119.3) 返回 PUD=79%（错误，远超科级阈值68.1%），实际应分类为 same_species。

**根因**：`find_frameshift_site()` 在搜索窗口内取第一个 `TTTAAAC` 匹配，SARS-CoV-1 基因组在真实移码位点（nt 13391）之前存在另一个 `TTTAAAC`，导致 pp1ab 从错误位置翻译，产生乱序蛋白，与所有参考序列 PUD 均异常高。

**修复**（`backend/tools/corona_pud.py`，`find_frameshift_site()`）：
- 枚举搜索窗口内**所有** `TTTAAAC` 候选位置（搜索起始从 ORF1a+10000 nt 放宽至 +8000 nt）
- 对每个候选尝试 -1 框架翻译 ORF1b（12000 nt 窗口），仅保留 ORF1b >1500 aa 的候选
- 选取产生最长 ORF1b 的位置（即真实移码位点）
- ORF1b 翻译窗口从 `slip_pos+12000` 扩大至 `slip_pos+15000`（覆盖更长的禽类冠状病毒 ORF1b）

**验证**：
- SARS-CoV-1 slip_pos=13391（修复前使用错误位置）；SARS-CoV-2 slip_pos=13461（正确不变）
- SARS-CoV-1 pp1ab=7073 aa，SARS-CoV-2 pp1ab=7096 aa（均为合理值）
- SARS-CoV-1 vs 自身 PUD=3.8% → same_species ✓
- SARS-CoV-2 vs SARS-CoV-1 PUD=6.2% → same_species ✓

---

## 2026年4月7日

分类标准知识库全面升级 + Coronaviridae亚属级DEmARC PUD子系统完成：

**知识库扩展：**
- `criteria.json` 升级：有明确数值阈值的科从6个增加到**21个**。新增标准覆盖Paramyxoviridae（L蛋白分支长度≤0.03）、Flaviviridae（NS3/NS5B aa identity）、Arenaviridae（S/L段nt identity）、Bornaviridae（PASC 72-75%）、Parvoviridae（NS1 aa identity ≥85%）、Rhabdoviridae（按属，Lyssavirus N基因≥80%）、Peribunyaviridae、Adenoviridae、Filoviridae等
- 新增 `data/genus_criteria.json`：98个属的种间界定标准，52个属有数值阈值，通过`scripts/fetch_genus_criteria.py`从ICTV网站抓取解析
- `backend/knowledge/criteria.py` 新增 `get_genus_criteria()` 接口，`get_demarcation_summary()` 支持可选 `genus` 参数

**Coronaviridae DEmARC PUD子系统：**
- `scripts/build_corona_hmms.py`：从59条参考序列建立5个复制酶结构域的HMMER3 HMM profiles（输出 `data/hmm/CoV_5domains.hmm`）
- `backend/tools/corona_pud.py`：完整DEmARC PUD分类流水线
  - 自动检测ORF1a起始密码子（验证长ORF >3000 aa）和核糖体-1移码位点（TTTAAAC）
  - HMM提取5个结构域（坐标缩放法兜底）
  - MAFFT比对 + PUD计算 + Table 4阈值分类
- `backend/agent.py`：注册第9个工具 `corona_pud_classify`，系统提示词更新
- **验证通过**：SARS-CoV-2 vs SARS-CoV-1 PUD=6.2% (same_species)，Beta vs Alpha CoV PUD=45-60% (same_subfamily)，Beta CoV属内 PUD=27-35% (same_genus)

**已知5个科无明确标准的原因分析：**
- Caliciviridae/Hepeviridae：宿主范围和抗原性联合判定，无单一数值标准
- Orthoherpesviridae：大型dsDNA病毒，以基因组共线性+系统发育分群
- Pneumoviridae/Poxviridae：物种少（5种/100+种），标准涉及多维度生物学特征

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
