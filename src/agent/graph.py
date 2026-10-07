"""LangGraph 编排：构建合同问答 Agent 状态机。

图结构：
  START → route →(simple)→ basic_rag → END
                →(breach)→ breach_decompose → retrieve ⇄ validate → breach_compare → breach_verify → breach_conclude → END
                                     (next/rewrite→retrieve, breach_compare→breach_compare, synthesize→synthesize)
                                     (verify 通过→breach_conclude, 驳回→回 breach_compare 重判,
                                      缺关键条款→breach_supplement_retrieve 定向补检后回 breach_compare；每案最多1次补检)
                →(complex)→ decompose → retrieve ⇄ validate → synthesize → END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..retriever.retriever import DEFAULT_STORE_DIR, TwoStageRetriever
from .llm import DEFAULT_MODEL, LLMClient
from .nodes import DEFAULT_MAX_ROUNDS, DEFAULT_TOP_K, AgentNodes, AgentState


def _route_decision(state: AgentState) -> str:
    task_type = state.get("task_type", "complex")
    if task_type == "breach":
        return "breach_decompose"
    return "basic_rag" if task_type == "simple" else "decompose"


def _validate_decision(state: AgentState) -> str:
    nxt = state.get("_next", "synthesize")
    # next / rewrite 都回到 retrieve
    if nxt in ("next", "rewrite"):
        return "retrieve"
    # 违约判定分支：比对 → 结论；复杂问答：合成
    return "breach_compare" if nxt == "breach_compare" else "synthesize"


def _verify_decision(state: AgentState) -> str:
    nxt = state.get("_next")
    # 补检 → 定向补充检索后重新比对；驳回 → 回 breach_compare 重判；否则 → breach_conclude
    if nxt == "supplement":
        return "breach_supplement_retrieve"
    return "breach_compare" if nxt == "recompare" else "breach_conclude"


def build_graph(nodes: AgentNodes) -> "StateGraph":  # type: ignore[name-defined]
    """根据节点集构建编译后的 LangGraph。"""
    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("route", nodes.route)
    builder.add_node("basic_rag", nodes.basic_rag)
    builder.add_node("breach_decompose", nodes.breach_decompose)
    builder.add_node("decompose", nodes.decompose)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("validate", nodes.validate)
    builder.add_node("breach_compare", nodes.breach_compare)
    builder.add_node("breach_verify", nodes.breach_verify)
    builder.add_node("breach_supplement_retrieve", nodes.breach_supplement_retrieve)
    builder.add_node("breach_conclude", nodes.breach_conclude)
    builder.add_node("synthesize", nodes.synthesize)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        _route_decision,
        {"basic_rag": "basic_rag", "breach_decompose": "breach_decompose", "decompose": "decompose"},
    )
    builder.add_edge("basic_rag", END)
    builder.add_edge("breach_decompose", "retrieve")
    builder.add_edge("decompose", "retrieve")
    builder.add_edge("retrieve", "validate")
    builder.add_conditional_edges(
        "validate",
        _validate_decision,
        {"retrieve": "retrieve", "breach_compare": "breach_compare", "synthesize": "synthesize"},
    )
    builder.add_edge("breach_compare", "breach_verify")
    builder.add_conditional_edges(
        "breach_verify",
        _verify_decision,
        {
            "breach_compare": "breach_compare",
            "breach_supplement_retrieve": "breach_supplement_retrieve",
            "breach_conclude": "breach_conclude",
        },
    )
    builder.add_edge("breach_supplement_retrieve", "breach_compare")
    builder.add_edge("breach_conclude", END)
    builder.add_edge("synthesize", END)
    return builder.compile()


class ContractAgent:
    """端到端合同问答 Agent：路由 → (检索|多轮Agent) → 带引用答案。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        store_dir: str = DEFAULT_STORE_DIR,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.llm = LLMClient(model=model)
        self.retriever = TwoStageRetriever(store_dir=store_dir)
        self.nodes = AgentNodes(self.llm, self.retriever, max_rounds=max_rounds, top_k=top_k)
        self.graph = build_graph(self.nodes)

    def ask(self, query: str, doc_ids: list[str] | None = None) -> dict:
        """同步执行一次问答，返回最终 state（含 answer/citations/breach_verdict）。

        Args:
            query: 用户问题。
            doc_ids: 限定本次问答的合同范围（doc_id 列表）。None=全局检索。
                交互模式下由 main.py --doc 设定并跨轮继承。
        """
        initial = {"query": query, "max_rounds": self.nodes.max_rounds}
        if doc_ids:
            initial["current_doc_ids"] = doc_ids
        return self.graph.invoke(initial)

    def stream(self, query: str, doc_ids: list[str] | None = None):
        """流式输出各节点更新，便于调试。"""
        initial = {"query": query, "max_rounds": self.nodes.max_rounds}
        if doc_ids:
            initial["current_doc_ids"] = doc_ids
        for chunk in self.graph.stream(initial, stream_mode="updates"):
            yield chunk
