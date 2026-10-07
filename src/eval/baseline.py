"""Basic RAG 基线：用于与 Agentic RAG 做违约判定效果的消融对比。

与 ContractAgent 的差异（即 Agentic 的增量点）：
- 无路由：直接按违约判定处理（task_type 恒为 breach）
- 无子问题拆解：用原始问题做单次检索
- 无证据校验/改写重检：检索一轮即止
- 无独立复核：一次责任比对直接出结论

接口与 ContractAgent.ask 对齐，返回 dict 含 task_type/answer/citations/breach_verdict，
可直接交给 EvalRunner 复用全部指标（引用召回/忠实度/判定准确率/耗时）。
"""
from __future__ import annotations

import logging
from typing import Optional

from ..agent.llm import DEFAULT_MODEL, LLMClient
from ..retriever.retriever import DEFAULT_STORE_DIR, TwoStageRetriever

logger = logging.getLogger("eval")


class BasicBreachRAG:
    """朴素违约判定 RAG：单次检索 → 一次 JSON 判定 → 一次带引用作答。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        store_dir: str = DEFAULT_STORE_DIR,
        top_k: int = 5,
    ) -> None:
        self.llm = LLMClient(model=model)
        self.retriever = TwoStageRetriever(store_dir=store_dir)
        self.top_k = top_k

    def ask(self, query: str, doc_ids: Optional[list[str]] = None) -> dict:
        results = self.retriever.retrieve(query, top_k_final=self.top_k, doc_ids=doc_ids)
        logger.info(
            "[basic] 单轮检索命中 %d 条 | %s",
            len(results),
            "; ".join(r.citation() for r in results[:3]),
        )
        context = TwoStageRetriever.format_context(results)

        # 第一步：基于单次检索上下文直接输出结构化判定
        verdict_prompt = (
            "你是合同违约判定员。基于以下检索到的合同条款上下文，对用户描述的纠纷进行责任比对。\n"
            "比对步骤：\n"
            "1. 从上下文中找出与纠纷相关的条款：义务/违约情形条款、责任条款、免责条款\n"
            "2. 将用户描述的事实与合同约定比对：是否构成违约、责任后果、有无免责事由\n"
            "3. 上下文缺少判定所需信息时不得臆断\n"
            '仅输出JSON：{"breach_established": true/false/null, '
            '"breaching_party": "违约方名称或null", "reasoning": "比对推理"}\n\n'
            f"用户纠纷描述：{query}\n\n合同条款上下文：\n{context}"
        )
        data = self.llm.chat_json([{"role": "user", "content": verdict_prompt}]) or {}
        verdict = {
            "breach_established": data.get("breach_established"),
            "breaching_party": data.get("breaching_party"),
            "matched_clauses": [],
            "missing_facts": [],
            "reasoning": data.get("reasoning", ""),
        }

        # 第二步：基于判定生成带引用的最终答案
        answer_prompt = (
            "你是合同纠纷分析助手。基于以下检索到的合同条款上下文，回答用户的违约纠纷问题。\n"
            "要求：\n- 先给结论（是否构成违约/由谁担责/依据什么条款），再给分析\n"
            "- 引用条款时使用 [编号] 格式（编号对应上下文中的 [1] [2] 等）\n"
            "- 不得编造条款内容，上下文未覆盖的情形如实说明\n\n"
            f"用户纠纷描述：{query}\n\n合同条款上下文：\n{context}"
        )
        answer = self.llm.chat([{"role": "user", "content": answer_prompt}], temperature=0.1)

        return {
            "task_type": "breach",  # 基线无路由环节，恒为 breach
            "answer": answer,
            "citations": [results],
            "breach_verdict": verdict,
        }
