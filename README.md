# TradeContractAgent
> 建工贸易采购合同 Agentic RAG 智能核验问答 demo

## 项目简介
TradeContractAgent 是面向建筑行业建材贸易采购合同的 Agentic RAG 验证项目，聚焦施工单位的钢材、建材采购贸易合同场景。
针对传统单轮RAG容易**忽略条件型履约分支**的痛点，在 BGE-M3 + bge-reranker-v2-m3 二级检索底座之上搭建Agent调度链路，实现复杂问题拆解、动态Query改写、多轮检索重试、多文档条款比对与答案引用溯源。
在此基础上新增**违约判定专属分支**（任务拆解→责任比对→结论生成）与**推理交叉校验机制**（全局编号一致性校验 + LLM 独立复核闭环），提升合同纠纷问答的准确率与可溯源能力。

> ⚠️ 项目定位：专注于**贸易采购合同**，暂不泛化至总包、劳务、租赁等其他类型工程合同。这类采购合同范式相对统一，核心关注供货、单价、付款节点、质保、违约责任及后续变更补充文件。

## 核心场景
1. **条件分支式履约与违约判定**：对“逾期供货、材料检测不合格”等带前置触发条件的条款做分分支检索推理；对“是否构成违约、由谁担责、如何赔偿”等纠纷问题走违约判定专属分支（义务条款→责任条款→免责条款三要素拆解检索 + 结构化责任比对 + 带溯源结论生成）
2. **推理交叉校验**：违约判定结论在生成前经独立复核节点做对抗性审查（条款理解核对、反方论证、诚实性检查），复核不通过则反馈问题重判，重判后仍不通过则结论强制降级为“暂无法完全判定”
3. **证据不足时动态检索重试**：首轮检索无结果时自动改写检索关键词，重新召回，达到最大轮次后再返回信息缺失
4. **合同定向检索**：FAISS `IndexIDMap2` + `IDSelector` 支持按 `doc_id` 限定检索范围，解决「这份合同的供应商是谁」类定向查询；CLI `--doc` 指定单合同聚焦会话，交互模式跨轮继承范围
5. **多份采购合同横向对比**：同项目多家供应商合同的质保期、违约金上限等条款汇总对比

简单单点查询（合同编号、签约日期、供应商全称）路由至原始单轮RAG，降低时延与Token开销。

## 技术架构
```
用户Query
    ↓
问题路由模块（三分类：simple 单点事实 / breach 违约判定 / complex 复杂问答）
    ↓
┌─────────────────────────────────────────────────────────────┐
│ simple  → 基础RAG（单轮检索 + 引用编号校验）→ END                       │
│ complex → decompose → retrieve ⇄ validate → synthesize → END          │
│ breach  → breach_decompose → retrieve ⇄ validate                       │
│           → breach_compare（责任比对，输出结构化 verdict）              │
│           → breach_verify（独立复核 + 对抗性反方论证）                  │
│              ├─ 通过 → breach_conclude → END                           │
│              └─ 驳回 → 反馈问题回 breach_compare 重判（最多1次）         │
│                        重判后仍不通过 → 结论降级为「暂无法完全判定」     │
└─────────────────────────────────────────────────────────────┘
    ↓
检索底座（复用已调优垂类检索链路）
├─ 粗排：BGE-M3 稠密 + BM25 稀疏（jieba 分词），经 RRF 融合后喂精排
├─ 精排：bge-reranker-v2-m3
└─ 召回文档片段+原文定位ID（用于引用溯源，跨组全局连续编号）
    ↓
生成带引用的答案（引用编号全局连续，与末尾引用清单一一对应）
```

### 主要依赖组件
- 检索模型：`BGE-M3`（稠密粗排）、`bge-reranker-v2-m3`（精排）、`rank_bm25` + `jieba`（稀疏粗排，与 BGE-M3 经 RRF 融合）
- Agent 框架：LangGraph（用于本地demo流程编排）
- LLM：Qwen系列（指令遵循、子问题拆解、SFT推理）
- 向量库：可选用 FAISS / Chroma（本地轻量实验）
- 数据：自建建工贸易合同Query-Doc对、阅读理解SFT样本、IFD Cherry样本

## 关键特性
- ✅ Agent 动态规划，支持多轮检索重试，识别条款版本冲突
- ✅ 违约判定专属分支：三要素拆解（义务/责任/免责）→ 结构化责任比对 → 结论生成
- ✅ 推理交叉校验：引用编号全局连续一致性校验 + LLM 独立复核（对抗性反方论证）闭环
- ✅ 合同定向检索：FAISS `IndexIDMap2` + `IDSelector` 按 `doc_id` 限定范围，`--doc` 跨轮继承
- ✅ 混合检索：BGE-M3 稠密 + BM25 稀疏（jieba 中文分词）粗排，经 RRF 融合后喂 reranker 精排，强化「第 9.4 条」「逾期违约金」等强关键词命中
- ✅ 表格结构化抽取：文字层 PDF 走 pymupdf `find_tables`、扫描件走 OCR 行盒几何重建，表格重建为 markdown 独立 chunk（可检索可引用），单元格结构另存 tables.pkl 旁路
- ✅ 输出答案附带原文片段引用溯源，可定位到合同段落，编号与引用清单严格对齐
- ✅ 评测框架：路由准确率 / 引用召回 / 答案忠实度（LLM-as-judge）/ 违约判定准确率 / 分类型聚合 / 平均耗时

## 快速启动（本地demo）
> 依赖本地 Ollama 提供 LLM（默认 `qwen3.5:9b-q4_K_M`），检索模型缓存到项目 `./models` 目录。
```bash
# 1. 克隆项目
git clone https://github.com/Duandand/trade_contract_agent.git
cd trade_contract_agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备 LLM：安装并启动 Ollama，拉取模型（也可改用其他 Ollama 模型）
ollama pull qwen3.5:9b-q4_K_M
ollama serve   # 若尚未在后台运行

# 4. 下载检索模型：BGE-M3（粗排）+ bge-reranker-v2-m3（精排）到 ./models
python scripts/download_models.py
# 国内网络慢可先 export HF_ENDPOINT=https://hf-mirror.com 再运行

# 5. 放入合同 PDF：将你的采购合同 PDF 放入 ./test_data/（支持文字层 PDF 与扫描件 OCR）
#    ⚠️ 真实合同请勿提交到公开仓库（test_data/ 已在 .gitignore 中排除）

# 6. 文档入库：合同PDF解析、表格结构化抽取、分块、构建向量索引（FAISS IndexIDMap2）+ BM25 稀疏索引
python ingest_contract.py --data_path ./test_data

# 7. 启动问答服务
python main.py                       # 全局交互问答
python main.py --list-docs           # 列出已入库合同 doc_id
python main.py --doc <doc_id> "问题"  # 限定单份合同问答
python main.py --doc <doc_id>        # 单合同聚焦交互（跨轮继承范围）
```

## 目录结构参考
```
trade_contract_agent/
├── scripts/
│   └── download_models.py # 下载 BGE-M3 / bge-reranker-v2-m3 到 ./models
├── src/
│   ├── retriever/         # BGE-M3+BM25 粗排(RRF融合) + bge-reranker精排 + doc_id 定向过滤
│   ├── agent/             # LangGraph Agent调度、子问题拆解、违约判定分支、推理交叉校验
│   ├── pipeline/          # 文档解析、表格结构化抽取、条款分块、向量入库（IndexIDMap2）
│   ├── eval/              # 评测框架 + golden_qa.example.jsonl 示例黄金集 + CSV标注模板
│   └── utils/
├── data/                  # 运行时生成（git 忽略）：vector_store/ 索引 + eval/ 评测报告
├── test_data/             # 自备合同 PDF 样本（git 忽略，不入库）
├── ingest_contract.py     # 入库 CLI
├── main.py                # 问答 CLI（--doc 定向 / --list-docs）
├── run_llm.sh             # Ollama 安装/拉取模型辅助脚本
├── requirements.txt
└── README.md
```

## 数据与隐私说明
本项目开源仅包含**代码与示例配置**，以下内容均不入库（已在 [.gitignore](.gitignore) 中排除）：
- `test_data/`：真实合同 PDF（含真实企业信息），请使用自有合同替换
- `data/`：向量索引与评测报告，均可由 `ingest_contract.py` / 评测脚本重建
- `src/eval/golden_*.jsonl`：基于真实合同人工标注的黄金评测集
- `models/`：检索模型缓存，用 `scripts/download_models.py` 重新下载
- `todo.md` / `record.md`：开发过程记录

评测黄金集请参照 `src/eval/golden_qa.example.jsonl` 的 schema 与 `src/eval/golden_qa_template.csv` 的标注说明自行构建（每条 case 支持 `doc_scope` 限定检索范围、`must_cite` 校验召回条款）。

## 评测方案
- 基础RAG指标：路由准确率、引用召回（must_cite spec 子集匹配）、答案匹配
- Agent新增指标：
  - 答案忠实度（LLM-as-judge 判 supported/unsupported/partial）
  - 违约判定准确率（breach_established + breaching_party 子串匹配）
  - 分类型聚合（simple/breach/complex 各自的路由/召回/忠实指标）
  - 平均端到端耗时
- 黄金集：参照 `src/eval/golden_qa.example.jsonl` 构建自己的黄金集，每条 case 可设 `doc_scope` 限定检索范围评测定向查询
- 运行：`python -m src.eval.run_eval --golden <你的黄金集.jsonl>`（含 judge）或加 `--no-judge`（仅检索/路由，更快）

## 局限性 & 后续方向
1. 局限：当前demo仅针对**建材贸易采购合同**；总包、劳务等其他工程合同文档结构差异大，未做适配；Agent规划本身存在出错风险，依赖前置问题路由做降级；推理交叉校验增加 breach 路径 LLM 调用次数（约 +1~3 次），端到端时延从 24~55s 上升至 90~150s。
2. 后续：扩充合同类型、优化Agent重试策略、跨页表格合并与更多表格版式适配、全链路时延优化、将复核机制按需扩展到 complex 分支。

## 示例问答
**用户提问**：供应商逾期10天才完成供货，合同约定的逾期违约金如何计算？供应商是否构成违约，应承担什么责任？

> Agent逻辑：
> 1. [route] 三分类判定为 `breach`（违约判定/纠纷责任分析）
> 2. [breach_decompose] 拆解为3个判定要素子问题：①供应商按时交货义务及逾期违约情形 ②逾期违约金计算方式/比例 ③逾期导致的合同解除与其他责任
> 3. [retrieve ⇄ validate] 三组子问题各检索5条条款（必要时改写重检一轮），共采纳15条带全局编号 [1]..[15] 的条款上下文
> 4. [breach_compare] 责任比对，输出 verdict：违约成立=True、违约方=乙方、命中2条依据条款、2项缺失事实（具体供货日期、逾期交货金额）
> 5. [breach_verify] 第1次复核 verdict_ok=True（无问题、无反方论点）→ 通过
> 6. [breach_conclude] 生成带溯源结论：乙方构成违约，依据 [10] 条款（每日 5% 违约金、累计逾期超 3 日甲方有权解除合同）；因缺失交货金额无法计算具体数额，标注为「暂无法完全判定金额数值」

问题 top1 命中 粗排→精排分 逾期供货违约责任 条款 9.1 违约金 0.750 → 0.889 混凝土质量不合格 条款 7.4 退货处理 0.666 → 0.872 价格调整条件 条款 1.2.2 调整窗口 0.723 → 0.812 结算货款支付 条款 6.5 支付方式 0.740 → 0.597

## License
MIT

---
