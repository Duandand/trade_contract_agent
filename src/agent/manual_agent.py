"""手搓版 Agent：不依赖 LangGraph，用普通函数 + 循环实现同样的调度链路。

与 src/agent/graph.py 的 LangGraph 版本逻辑一一对应：

    route（简单/复杂路由）
      ├─ 简单 → basic_rag（单轮检索直接生成）→ 结束
      └─ 复杂 → decompose（拆子问题）
                → for 每个子问题:
                      for 轮次 in range(max_rounds):
                          retrieve → validate
                          ├─ 证据充足/轮次耗尽 → 采纳，break
                          └─ 不足 → 改写 query 再检索
                → synthesize（汇总所有子问题上下文生成答案）

用途：对照学习整体流程。节点间不再通过 graph state 传递，
而是直接用局部变量 / 返回字典，数据流向更直白。

用法：
    python -m src.agent.manual_agent "逾期供货的违约金怎么算？"
    或在代码中：from src.agent.manual_agent import ManualAgent; ManualAgent().ask(query)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from ..retriever.retriever import RetrievalResult, TwoStageRetriever
from .llm import DEFAULT_MODEL, LLMClient
from .nodes import DEFAULT_MAX_ROUNDS

logger = logging.getLogger("agent")


class ManualAgent:
    """手写循环版合同问答 Agent，接口与 ContractAgent 保持一致。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        store_dir: Optional[str] = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        top_k: int = 5,
    ) -> None:
        self.llm = LLMClient(model=model)
        kwargs: dict[str, Any] = {"store_dir": store_dir} if store_dir else {}
        self.retriever = TwoStageRetriever(**kwargs)
        self.max_rounds = max_rounds
        self.top_k = top_k

    # ============ 主入口：对应 LangGraph 的图执行 ============
    def ask(self, query: str) -> dict:
        """执行完整问答链路，返回 {"answer": ..., "citations": [...]}。"""
        logger.info("[ask] query=%r", query)

        # 节点1：route
        is_simple = self._route(query)

        if is_simple:
            # 简单路径：单轮 RAG
            return self._basic_rag(query)

        # 复杂路径：Agent 多轮调度
        # 节点2：decompose
        subqueries = self._decompose(query)

        contexts: list[str] = []
        citations: list[list[RetrievalResult]] = []

        for i, subquery in enumerate(subqueries):
            logger.info("[loop] 处理子问题 %d/%d: %r", i + 1, len(subqueries), subquery)
            # 节点3+4：retrieve ⇄ validate（最多 max_rounds 轮）
            context, results = self._retrieve_with_retry(subquery, q_idx=i, total=len(subqueries))
            contexts.append(context)
            citations.append(results)

        # 节点5：synthesize
        answer = self._generate(query, "\n\n".join(c for c in contexts if c))
        logger.info("[synthesize] 答案生成完毕，%d 字", len(answer))
        return {"answer": answer, "citations": citations, "subqueries": subqueries}

    # ============ 节点1：路由 ============
    def _route(self, query: str) -> bool:
        """判断是否简单问题。返回 True=走基础RAG，False=走Agent。"""
        prompt = (
            "你是合同问答路由器。判断用户问题是否为「简单单点事实查询」。\n"
            "简单问题：只需查一个明确字段，如合同编号、签订日期、供应商名称、甲方名称、"
            "项目名称、合同金额等单一事实，不涉及条件判断、多步推理、对比、计算。\n"
            "复杂问题：涉及条件分支（如「逾期供货违约金怎么算」）、多份合同对比、多步推理、"
            "需综合多个条款。\n"
            '仅输出JSON：{"is_simple": true/false, "reason": "简短理由"}\n\n'
            f"用户问题：{query}"
        )
        data = self.llm.chat_json([{"role": "user", "content": prompt}])
        is_simple = bool(data.get("is_simple")) if data else True  # 解析失败降级为简单
        logger.info(
            "[route] is_simple=%s reason=%s -> %s",
            is_simple,
            (data or {}).get("reason", "(解析失败降级)"),
            "basic_rag" if is_simple else "decompose",
        )
        return is_simple

    # ============ 简单路径：基础 RAG ============
    def _basic_rag(self, query: str) -> dict:
        logger.info("[basic_rag] 单轮检索 query=%r", query)
        results = self.retriever.retrieve(query, top_k_final=self.top_k)
        context = TwoStageRetriever.format_context(results)
        answer = self._generate(query, context)
        return {"answer": answer, "citations": [results], "subqueries": [query]}

    # ============ 节点2：子问题拆解 ============
    def _decompose(self, query: str) -> list[str]:
        prompt = (
            "你是合同分析专家。将用户问题拆解为若干可独立检索的子问题，"
            "每个子问题应能用关键词在合同库中检索到相关条款。\n"
            "要求：\n- 覆盖原问题所需全部信息点\n- 简洁、适合向量检索\n"
            "- 原问题若单一检索即可解决，只输出1个\n- 最多5个\n"
            '仅输出JSON：{"subqueries": ["子问题1", "子问题2", ...]}\n\n'
            f"用户问题：{query}"
        )
        data = self.llm.chat_json([{"role": "user", "content": prompt}])
        subs = data.get("subqueries") if data else []
        if not isinstance(subs, list) or not subs:
            subs = [query]  # 拆解失败兜底
        subs = [s for s in subs if isinstance(s, str) and s.strip()][:5] or [query]
        logger.info("[decompose] 拆出 %d 个子问题: %s", len(subs), subs)
        return subs

    # ============ 节点3+4：检索 + 校验重试（核心循环） ============
    def _retrieve_with_retry(
        self, subquery: str, q_idx: int, total: int
    ) -> tuple[str, list[RetrievalResult]]:
        """对单个子问题做「检索→校验→（不足则改写再检索）」，返回最终采纳的上下文与结果。"""
        query = subquery
        for round_idx in range(self.max_rounds):
            # --- retrieve ---
            t0 = time.perf_counter()
            results = self.retriever.retrieve(query, top_k_final=self.top_k)
            context = TwoStageRetriever.format_context(results)
            logger.info(
                "[retrieve] 子问题%d/%d 第%d轮 query=%r 命中%d条 (%.1fs) | %s",
                q_idx + 1, total, round_idx + 1, query, len(results),
                time.perf_counter() - t0,
                "; ".join(f"{r.citation()} score={r.rerank_score:.3f}" for r in results[:3]),
            )

            # --- validate ---
            sufficient, rewritten = self._judge(query, context)
            logger.info(
                "[validate] sufficient=%s rewritten=%r",
                sufficient, rewritten if not sufficient else "",
            )

            if sufficient:
                return context, results  # 证据充足，采纳
            if round_idx + 1 >= self.max_rounds:
                logger.info("[validate] 轮次耗尽，降级采纳当前结果")
                return context, results  # 轮次耗尽，降级采纳

            # 证据不足 → 改写 query，进入下一轮
            logger.info("[validate] 证据不足 -> 改写重检: %r -> %r", query, rewritten or query)
            query = rewritten or query

        return "", []  # 理论不可达

    def _judge(self, subquery: str, context: str) -> tuple[bool, str]:
        """校验证据是否充足，返回 (是否充足, 改写query)。"""
        if not context:
            return False, subquery
        prompt = (
            "你是合同证据校验员。判断提供的检索上下文能否回答子问题。\n"
            "判断标准：上下文是否包含与子问题直接相关的合同条款信息。\n"
            '仅输出JSON：{"sufficient": true/false, '
            '"rewritten_query": "若不足，提供更精准的检索词；若充足则空"}\n\n'
            f"子问题：{subquery}\n\n检索上下文：\n{context}"
        )
        data = self.llm.chat_json([{"role": "user", "content": prompt}])
        if not data:
            return True, ""  # 解析失败默认充足，避免死循环
        return bool(data.get("sufficient")), (data.get("rewritten_query", "") or "").strip()

    # ============ 节点5 / 共用：生成 ============
    def _generate(self, query: str, context: str) -> str:
        if not context:
            return "根据现有合同文本无法确定：未检索到相关条款。"
        prompt = (
            "你是合同问答助手。基于以下检索到的合同条款上下文回答用户问题。\n"
            "要求：\n- 答案必须基于提供的上下文，不得编造\n"
            "- 引用来源时使用 [编号] 格式（编号对应上下文中的 [1] [2] 等）\n"
            "- 信息不足时明确说明「根据现有合同文本无法确定」\n"
            "- 条款存在冲突或版本差异时，指出区别\n\n"
            f"用户问题：{query}\n\n上下文：\n{context}"
        )
        return self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "逾期供货的违约金怎么算？"
    agent = ManualAgent()
    result = agent.ask(q)
    print("\n--- 答案 ---")
    print(result["answer"])
    print("\n--- 引用来源 ---")
    seen = set()
    for group in result["citations"]:
        for r in group:
            key = (r.doc_id, r.clause_label, r.page_start)
            if key not in seen:
                seen.add(key)
                print(f"  {r.citation()} (rerank={r.rerank_score:.3f})")
