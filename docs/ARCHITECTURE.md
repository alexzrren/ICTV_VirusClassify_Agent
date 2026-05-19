# ICTV 病毒分类智能 Agent — 架构与流程图

本文档以 [Mermaid](https://mermaid.js.org/) 流程图形式呈现 ICTV Agent 的系统架构、单序列分类流程、批量分类时序，以及数据库构建管线。配合 `ICTV病毒分类智能Agent实施方案.md`（设计与开发日志）阅读。

> 📌 **GitHub / VS Code / Obsidian 可直接渲染本文档**。如需 PNG/SVG，可执行：
> ```bash
> npx -p @mermaid-js/mermaid-cli mmdc -i docs/ARCHITECTURE.md -o docs/diagrams/diagram.png
> ```

---

## 一、系统架构（分层视图）

```mermaid
graph TB
    subgraph U["前端 / 客户端"]
        WEB["Web UI<br/>frontend/index.html<br/>SSE 实时步骤展示"]
        CLI["batch_classify.py<br/>命令行批量驱动<br/>--parallel / --model"]
    end

    subgraph API["FastAPI 后端 (backend/main.py)"]
        E1["POST /classify<br/>提交分类任务"]
        E2["GET /result/{job_id}<br/>轮询结果"]
        E3["GET /stream/{job_id}<br/>SSE 推流"]
        E4["POST /cancel/{job_id}<br/>中途取消"]
        E5["GET /history /cache<br/>SHA-256 去重缓存"]
        SEM["asyncio.Semaphore(8)<br/>并发限制"]
    end

    subgraph AG["Agent 推理核心 (backend/agent.py)"]
        ALOOP["ReAct 循环<br/>max_steps=20<br/>+ continuation nudges (×2)"]
        SDK["anthropic.Anthropic<br/>httpx pool: 200 conn / 5min keepalive<br/>retry: 8× exp backoff"]
        OVR["三层 Override 防御<br/>L1 模型 nudge / L2 BLAST→VMR / L3 Corona PUD"]
    end

    subgraph TL["11 个工具 (backend/tools/)"]
        T1["blast_and_compare<br/>blast.py + alignment.py"]
        T2["corona_pud_classify<br/>corona_pud.py"]
        T3["compare_query_to_reference<br/>hmmer.py 复合工具"]
        T4["extract_target_region<br/>hmmer.py"]
        T5["lookup_taxonomy / list_reference_species<br/>taxonomy.py"]
        T6["search_ictv_docs<br/>knowledge/rag.py"]
        T7["get_criteria<br/>knowledge/criteria.py"]
        T8["fetch_reference_sequence<br/>BioPython / 本地 fasta"]
        T9["blast_search<br/>blast.py"]
        T10["compute_pairwise_identity<br/>alignment.py (MAFFT)"]
    end

    subgraph DATA["数据层 (data/)"]
        D1["taxonomy.db<br/>16,213 species (MSL40)<br/>21,541 vmr_accessions"]
        D2["criteria.json (32 科)<br/>genus_criteria.json (98 属)"]
        D3["references/ — 34 科 FASTA<br/>db/ — BLAST + Diamond 库"]
        D4["hmm/ — 24 科 30 区域<br/>+ CoV_5domains.hmm"]
        D5["vectordb/ — 32 科 TF-IDF"]
        D6["cache.db — SHA-256 缓存"]
    end

    subgraph EXT["外部 LLM 接入"]
        LLM["SiliconFlow / 火山 Ark / MiniMax<br/>Anthropic-compatible API<br/>当前默认: Pro/zai-org/GLM-5.1"]
    end

    WEB --> E1 & E2 & E3 & E4 & E5
    CLI --> E1 & E2 & E4
    E1 --> SEM --> ALOOP
    E2 --> ALOOP
    E3 --> ALOOP
    ALOOP --> SDK --> LLM
    ALOOP --> TL
    OVR -.->|后处理覆盖| ALOOP
    T1 & T9 --> D3
    T2 --> D3 & D4 & D1
    T3 & T4 --> D4 & D3
    T5 --> D1
    T6 --> D5
    T7 --> D2
    T8 --> D3
    E5 --> D6

    style ALOOP fill:#fef3c7
    style OVR fill:#fecaca
    style D1 fill:#dbeafe
    style LLM fill:#e9d5ff
```

---

## 二、单条序列分类的完整流程

```mermaid
flowchart TD
    Q["📥 用户提交 FASTA<br/>(WEB 或 batch_classify.py)"] --> CK{"缓存命中?<br/>SHA-256 of seq"}
    CK -->|Yes| RET["秒级返回缓存结果"]
    CK -->|No| JOB["生成 job_id<br/>asyncio.create_task"]
    JOB --> LOOP{"Agent 步循环<br/>step ≤ 20"}

    LOOP --> CALL["LLM messages.create()<br/>（含 8× retry 退避）"]
    CALL --> DEC{"stop_reason?"}
    DEC -->|tool_use| TC["执行 tool_use blocks<br/>注入完整序列防截断<br/>capture extracted regions"]
    TC --> LOOP

    DEC -->|end_turn| COMP{"分类完整?<br/>_classification_incomplete()"}
    COMP -->|不完整 & nudge<2| NUDGE["注入 user 消息:<br/>'你停早了，缺 X 工具调用'"] --> LOOP
    COMP -->|完整或耗尽 nudge| EXIT["跳出循环"]

    EXIT --> P1["Layer 1<br/>解析模型最终 JSON<br/>or _build_result_from_logs"]

    P1 --> P2{"family 为空?"}
    P2 -->|Yes| L2["Layer 2 — 通用 BLAST→VMR<br/>top hit 反查 vmr_accessions<br/>填入完整 taxonomy"]
    P2 -->|No| P3
    L2 --> P3{"family = Coronaviridae<br/>且 seq > 20kb?"}

    P3 -->|Yes| L3["Layer 3 — Corona 确定性<br/>未调 corona_pud_classify? → 主动跑<br/>按 rank 强制覆盖 taxonomy/evidence/reasoning<br/>同步抓取 5 个 domain 序列"]
    P3 -->|No| FIN
    L3 --> FIN["💾 写入 cache + 返回<br/>token_usage / extracted_regions / steps"]

    style L2 fill:#dbeafe
    style L3 fill:#fecaca
    style NUDGE fill:#fef3c7
    style RET fill:#d1fae5
```

---

## 三、批量分类时序

```mermaid
sequenceDiagram
    participant U as 用户
    participant C as batch_classify.py
    participant S as FastAPI Server
    participant A as Agent + LLM
    participant T as 工具 + 数据库

    U->>C: python batch_classify.py *.fasta -o out/ --parallel 3
    C->>C: parse 多序列 FASTA
    C->>S: GET /health  (sanity check)

    par 3 路并发 (asyncio.Semaphore)
        C->>S: POST /classify (seq1, model=GLM-5.1)
        S-->>C: job_id1
        loop 每 8s 轮询
            C->>S: GET /result/{job_id1}
            S-->>C: {status, steps[], result?}
        end
    and
        C->>S: POST /classify (seq2)
        Note over S,A: ReAct 循环:<br/>blast_and_compare → corona_pud_classify<br/>→ get_criteria → lookup_taxonomy → JSON
        S->>A: classify_sequence(...)
        loop step
            A->>T: tool_use 调用
            T-->>A: JSON 结果
            A->>A: 累计 token / region 序列
        end
        A->>A: Override 三层后处理
        A-->>S: ClassifyResult
        S-->>C: status=done
    end

    C->>C: write per-seq .txt (含 EXTRACTED DOMAIN SEQUENCES)
    C->>C: write results_summary.xlsx (rich formatting)
    C-->>U: 完成总结 (done/error/avg time/total tokens)
```

---

## 四、知识库与参考库构建（一次性预处理）

```mermaid
graph LR
    subgraph SRC["📚 ICTV 官方源"]
        MSL["MSL40 Excel<br/>16213 species"]
        VMR["VMR Excel<br/>21541 accession→species"]
        REPORT["ICTV Report Chapters<br/>(网页 / PDF)"]
    end

    subgraph SCRIPTS["scripts/ 构建工具"]
        S1["build_taxonomy_db.py"]
        S2["build_vmr_accession_db.py"]
        S3["extract_criteria.py + fetch_genus_criteria.py"]
        S4["download_reference_seqs.py"]
        S5["build_blast_db.py"]
        S6["build_family_hmms.py / build_corona_hmms.py"]
        S7["build_vectordb.py (TF-IDF)"]
    end

    subgraph OUT["data/ 产物"]
        O1["taxonomy.db<br/>species + vmr_accessions 表"]
        O2["criteria.json (32 科)<br/>genus_criteria.json (98 属)"]
        O3["references/Family/sequences.fasta<br/>34 科 6,476 条"]
        O4["db/blastn_ref.* + diamond_ref.dmnd"]
        O5["hmm/*_targets.hmm<br/>24 科 30 区域<br/>CoV_5domains.hmm"]
        O6["vectordb/Family/* TF-IDF"]
    end

    MSL --> S1 --> O1
    VMR --> S2 --> O1
    REPORT --> S3 --> O2
    REPORT -.NCBI Entrez.-> S4 --> O3
    O3 --> S5 --> O4
    O3 --> S6 --> O5
    REPORT --> S7 --> O6
```

---

## 五、关键设计要点速查

| 模块 | 文件 | 职责亮点 |
|---|---|---|
| **Agent 核心** | `backend/agent.py` | 11 工具 ReAct 循环 + 3 层 Override + 模型 nudge + token 追踪 |
| **Coronaviridae 专用** | `backend/tools/corona_pud.py` | DEmARC PUD 5 个复制酶结构域 + ORF1ab 移码检测 + VMR 反查 |
| **HMM 区域提取** | `backend/tools/hmmer.py` | aa + nt 双坐标映射，支持 ORF 反向链 |
| **复合工具** | `compare_query_to_reference` | 一步完成 query+ref 区域提取+对齐+identity，避免多轮 chain |
| **缓存** | `backend/cache.py` | SHA-256 of FASTA → 完整结果，Web 端可逐项删除 |
| **批量驱动** | `scripts/batch_classify.py` | 3 路并发 / 自动重试 / Excel rich text / FASTA 段输出提取序列 |
| **抗幻觉** | `agent.py` 末尾 ~150 行 | L1 模型重跑 + L2 BLAST→VMR + L3 Corona 强制覆盖 |
| **数据完备性** | VMR 21,541 条 → genus/subgenus/species 全权威 | 杜绝 `species LIKE '%accession%'` 反查的旧错误 |

---

## 六、API 端点速查（FastAPI / `backend/main.py`）

| 方法 | 路径 | 功能 |
|---|---|---|
| `GET` | `/health` | 健康检查（families 数、并发槽） |
| `POST` | `/classify` | 提交分类任务，body 接受 `fasta`, `family_hint`, `model` |
| `GET` | `/result/{job_id}` | 轮询任务结果（含 steps/result/token_usage） |
| `GET` | `/stream/{job_id}` | SSE 推流，前端实时显示步骤 |
| `POST` | `/cancel/{job_id}` | 中途取消 asyncio.Task |
| `GET` | `/history` | 历史任务列表（最近若干条） |
| `GET` | `/cache/{seq_hash}` | 按 SHA-256 取缓存结果 |
| `GET` | `/families` | 列出有标准的科 |
| `GET` | `/family/{name}` | 单科详细信息 |
| `GET` | `/family/{name}/summary` | 单科 species/genus 计数概览 |
| `GET` | `/species` | 模糊搜索物种 |
| `GET` | `/` | 静态首页（前端 HTML） |

---

## 七、当前默认 LLM 配置（`run.sh`）

```bash
ANTHROPIC_API_KEY="<SiliconFlow key sk-...>"
ANTHROPIC_BASE_URL="https://api.siliconflow.cn"
CLAUDE_MODEL="Pro/zai-org/GLM-5.1"
```

> ⚠️ **`ANTHROPIC_BASE_URL` 只到域名级**——SDK 自动拼接 `/v1/messages`。任何 Anthropic-compatible 平台（火山 Ark / MiniMax / SiliconFlow）皆同此规则。

> ⚠️ **DNS 解析**：若本机 systemd-resolved 无法解析 API 域名，需在 `/etc/hosts` 固定 IP，**不要走代理**（代理会被云服务侧 rate limit）。

切换模型示例（不必重启 server，使用 `--model` 覆盖）：

```bash
# 走 SiliconFlow 的 Kimi K2.6（疑难序列兜底）
python scripts/batch_classify.py input.fasta -o out/ --model Pro/moonshotai/Kimi-K2-Instruct

# 走火山 Ark 的 GLM-4.7（兼容老接入）
ANTHROPIC_BASE_URL=https://ark.cn-beijing.volces.com/api/coding \
ANTHROPIC_API_KEY=<ark-key> bash run.sh
```
