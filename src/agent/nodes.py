"""LangGraph Agent 节点与状态定义。

流程：
- 简单: route → basic_rag → END
- 违约判定: route → breach_decompose → retrieve ⇄ validate → breach_compare → breach_verify → breach_conclude → END
  （breach_verify 不通过时：缺关键条款→breach_supplement_retrieve 定向补检后重比对；
  其他→反馈问题回 breach_compare 重判；各最多 1 次，仍不通过则结论降级为「暂无法完全判定」）
- 复杂: route → decompose → retrieve ⇄ validate → synthesize → END
- validate 充足或耗尽轮次 → 进入下一子问题或比对/合成
- validate 不足且仍有轮次 → 改写 query 回 retrieve
所有节点为纯函数，返回需更新的 state 字段。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import TypedDict

from ..retriever.retriever import RetrievalResult, TwoStageRetriever
from .llm import LLMClient

logger = logging.getLogger("agent")

DEFAULT_MAX_ROUNDS = 2  # 每个子问题最多检索轮次（原始1 + 改写1）
MAX_VERIFY = 2  # 违约判定最多复核次数（初次1次 + 重判/补检后1次）
DEFAULT_TOP_K = 8  # 每路检索精排返回数（5→8：9.4 等正确条款实测排在 #6-8 被截断）
VERDICT_MAX_TOKENS = 3072  # 比对/复核 JSON 输出预算（条款族扩展后上下文增大，默认 2048 会截断）
FAMILY_EXPAND_CAP = 8  # 条款族扩展全案总上界（防止多个子问题重复触发导致上下文膨胀）
COMPARE_CONTEXT_CAP = 18  # compare/conclude 全局上下文条数上界（防 5 子问题累积致输出超长、JSON 解析失败）


class AgentState(TypedDict, total=False):
    # 输入
    query: str
    max_rounds: int
    current_doc_ids: list[str]  # 本次会话限定的合同 doc_id 范围；None/空=全局检索
    # 路由
    task_type: str  # simple / breach / complex
    # 子问题调度
    subqueries: list[str]
    sub_idx: int
    current_query: str
    round: int
    # 当前检索中间结果
    current_results: list[RetrievalResult]
    # 累计
    citations: list[list[RetrievalResult]]  # 每个子问题采纳的检索结果，分组顺序即全局编号顺序
    # 违约判定分支
    breach_verdict: dict  # breach_compare 的结构化责任比对结果
    verify_round: int  # 已执行的复核次数（0=尚未复核）
    verify_issues: list[str]  # 复核驳回的问题清单，反馈给 breach_compare 重判
    breach_review: dict  # breach_verify 的复核结论（passed/issues/counter_arguments/force_caveat）
    supplement_query: str  # 复核要求补充检索的条款要点（verify → supplement_retrieve）
    supplement_done: bool  # 是否已用过一次补充检索（每案最多 1 次，控制成本）
    family_extra_count: int  # 条款族扩展全案已补入条数（全局预算上界，防止上下文膨胀）
    # 输出
    answer: str
    # 路由标记（validate → 下一步）
    _next: str


_REF_RE = re.compile(r"\[(\d+)\]")
_CLAUSE_SUB_RE = re.compile(r"^(\d+)\.(\d+)$")


def _check_refs(text: str, n_total: int, where: str) -> list[int]:
    """校验文本中引用的 [N] 编号是否落在合法范围 1..n_total。

    返回非法编号列表；存在非法引用时输出告警日志。
    """
    refs = {int(m) for m in _REF_RE.findall(text or "")}
    invalid = sorted(r for r in refs if r < 1 or r > n_total)
    if invalid:
        logger.warning("[%s] 引用编号越界 %s，合法范围 1-%d", where, invalid, n_total)
    return invalid


class AgentNodes:
    """持有 llm / retriever 的节点集合，方法作为 LangGraph 节点注册。"""

    def __init__(
        self,
        llm: LLMClient,
        retriever: TwoStageRetriever,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.max_rounds = max_rounds
        self.top_k = top_k

    # ---------- 简单/违约判定/复杂 路由 ----------
    def route(self, state: AgentState) -> dict:
        query = state["query"]
        logger.info("[route] query=%r", query)
        # 合同名解析：query 显式提到某份合同 doc_id 时填 current_doc_ids，
        # 使后续检索限定在该合同范围。「这份」「它」等代词在单轮 CLI 无法消解，
        # 需 main.py --doc 显式指定（交互模式继承）或多轮记忆（功能 #7）。
        scope = state.get("current_doc_ids") or []
        if not scope:
            known_docs = sorted({c["doc_id"] for c in self.retriever.store.chunks})
            matched = [d for d in known_docs if d and d in query]
            if matched:
                scope = matched
                logger.info("[route] query 命中合同范围 %s", scope)
        prompt = (
            "你是合同问答路由器。将用户问题分类为以下三类之一：\n"
            "1. simple：简单单点事实查询，只需查一个明确字段，如合同编号、签订日期、"
            "供应商名称、甲方名称、项目名称、合同金额等单一事实，不涉及条件判断、多步推理、对比。\n"
            "2. breach：违约判定/纠纷责任分析。用户描述了违约情形或纠纷事实，"
            "询问是否构成违约、由谁担责、如何赔偿、违约金/赔偿金怎么算、能否解除合同等。\n"
            "3. complex：其他复杂问题，涉及条件分支、多份合同对比、多步推理、"
            "需综合多个条款，但不属于违约判定。\n"
            '仅输出JSON：{"task_type": "simple/breach/complex", "reason": "简短理由"}\n\n'
            f"用户问题：{query}"
        )
        data = self.llm.chat_json([{"role": "user", "content": prompt}])
        task_type = data.get("task_type") if data else None
        if task_type not in ("simple", "breach", "complex"):
            task_type = "complex"  # 解析失败降级为复杂，走完整检索管线
        logger.info(
            "[route] task_type=%s reason=%s scope=%s -> %s",
            task_type,
            (data or {}).get("reason", "(解析失败降级)"),
            scope or "(全局)",
            {"simple": "basic_rag", "breach": "breach_decompose", "complex": "decompose"}[task_type],
        )
        out = {"task_type": task_type}
        if scope:
            out["current_doc_ids"] = scope
        return out

    # ---------- 基础 RAG（简单问题降级路径） ----------
    def basic_rag(self, state: AgentState) -> dict:
        query = state["query"]
        doc_ids = state.get("current_doc_ids") or None
        logger.info("[basic_rag] 单轮检索 query=%r scope=%s", query, doc_ids or "(全局)")
        results = self.retriever.retrieve(query, top_k_final=self.top_k, doc_ids=doc_ids)
        logger.info(
            "[basic_rag] 命中 %d 条 | %s",
            len(results),
            "; ".join(r.citation() for r in results[:3]),
        )
        context = TwoStageRetriever.format_context(results)
        answer = self._generate(query, context)
        _check_refs(answer, len(results), "basic_rag")
        return {"answer": answer, "citations": [results]}

    # ---------- 子问题拆解 ----------
    def decompose(self, state: AgentState) -> dict:
        query = state["query"]
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
            subs = [query]  # 拆解失败兜底：原问题作为唯一子问题
        subs = [s for s in subs if isinstance(s, str) and s.strip()][:5] or [query]
        logger.info("[decompose] 拆出 %d 个子问题: %s", len(subs), subs)
        return {
            "subqueries": subs,
            "sub_idx": 0,
            "round": 0,
            "current_query": subs[0],
            "citations": [],
        }

    # ---------- 检索 ----------
    @staticmethod
    def _merge_reranked(
        a: list[RetrievalResult], b: list[RetrievalResult], k: int
    ) -> list[RetrievalResult]:
        """合并两路检索：按 chunk_id 去重（保留 rerank 高分），降序取前 k。

        两路使用同一 reranker，分数跨调用可比；chunk_id 相同时取较高分。
        """
        by_id: dict[str, RetrievalResult] = {}
        for r in list(a) + list(b):
            old = by_id.get(r.chunk_id)
            if old is None or r.rerank_score > old.rerank_score:
                by_id[r.chunk_id] = r
        merged = sorted(by_id.values(), key=lambda r: r.rerank_score, reverse=True)[:k]
        for i, r in enumerate(merged, 1):
            r.rank = i
        return merged

    def _expand_clause_family(
        self,
        results: list[RetrievalResult],
        doc_ids: list[str] | None,
        already_used: int = 0,
    ) -> tuple[list[RetrievalResult], int]:
        """条款族扩展：已检出同一「条」≥2 个子条款（如 9.1/9.2/9.3）时，
        把该条其余子条款（如 9.4）补入上下文。

        违约判定按条编号组织（第九条违约责任含 9.1–9.13），reranker 对
        特定问句常把兄弟子条款排到目标子条款之前（实测 9.4 在不可抗力问句
        下排 #10+），法律审阅逻辑也要求通读整章；补全同族子条款可消除该类漏检。
        扩展块 rerank_score=0，排在检出块之后。

        全案总预算 FAMILY_EXPAND_CAP：already_used 为此前各次检索已补入的条数，
        本次只补剩余额度，返回 (扩展后结果, 本次新增条数)。
        """
        if not results or already_used >= FAMILY_EXPAND_CAP:
            return results, 0
        root_children: dict[str, set[str]] = {}
        for r in results:
            m = _CLAUSE_SUB_RE.match(r.clause_label or "")
            if m:
                root_children.setdefault(m.group(1), set()).add(r.clause_label)
        trigger_roots = {root for root, labels in root_children.items() if len(labels) >= 2}
        if not trigger_roots:
            return results, 0
        scope = set(doc_ids) if doc_ids else None
        existing = {r.chunk_id for r in results}
        # 候选扩展块：按「家族已命中数降序 → 条号数值升序」排序。
        # 违约类问题里第九条（违约责任）通常命中最多，优先补全；同条内 9.4 早于 9.13，
        # 顺序靠前可确保最常用的子条款被纳入
        family_size = {root: len(labels) for root, labels in root_children.items()}

        def _sort_key(c: dict) -> tuple:
            m = _CLAUSE_SUB_RE.match(c.get("clause_label", ""))
            root = m.group(1)
            return (-family_size.get(root, 0), int(root), int(m.group(2)))

        candidates = []
        for c in self.retriever.store.chunks:
            if scope is not None and c["doc_id"] not in scope:
                continue
            m = _CLAUSE_SUB_RE.match(c.get("clause_label", ""))
            if m and m.group(1) in trigger_roots and c["chunk_id"] not in existing:
                candidates.append(c)
        candidates.sort(key=_sort_key)
        # 全案预算：只取剩余额度，避免多家族在多个子问题上重复扩展导致上下文膨胀
        budget = FAMILY_EXPAND_CAP - already_used
        extras: list[RetrievalResult] = []
        for c in candidates[:budget]:
            extras.append(
                RetrievalResult(
                    chunk_id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    file_name=c["file_name"],
                    page_start=c["page_start"],
                    page_end=c["page_end"],
                    clause_label=c["clause_label"],
                    text=c["text"],
                    coarse_score=0.0,
                    rerank_score=0.0,
                    rank=0,
                )
            )
        if extras:
            logger.info(
                "[retrieve] 条款族扩展补入 %d 条 (第%s条，全案累计 %d/%d)",
                len(extras),
                ",".join(sorted(trigger_roots)),
                already_used + len(extras),
                FAMILY_EXPAND_CAP,
            )
        return results + extras, len(extras)

    def retrieve(self, state: AgentState) -> dict:
        q = state["current_query"]
        doc_ids = state.get("current_doc_ids") or None
        round_idx = state.get("round", 0)
        logger.info(
            "[retrieve] 子问题%d/%d 第%d轮 query=%r scope=%s",
            state.get("sub_idx", 0) + 1,
            len(state.get("subqueries", [])),
            round_idx + 1,
            q,
            doc_ids or "(全局)",
        )
        t0 = time.perf_counter()
        results = self.retriever.retrieve(q, top_k_final=self.top_k, doc_ids=doc_ids)
        # 双路检索（仅首轮）：拆解后的空壳子问题会稀释语义（实测把 9.4 从
        # #4 拉到 #8），用原始问题再检一路合并，防止关键词/实体信息丢失
        original_query = state.get("query", "")
        dual_extra = 0
        if round_idx == 0 and original_query and original_query.strip() != q.strip():
            results_orig = self.retriever.retrieve(
                original_query, top_k_final=self.top_k, doc_ids=doc_ids
            )
            before = {r.chunk_id for r in results}
            results = self._merge_reranked(results, results_orig, self.top_k)
            dual_extra = sum(1 for r in results if r.chunk_id not in before)
        # 违约判定：条款族扩展（同条 ≥2 个子条款已命中 → 补全该条其余子条款，全案预算 8 条）
        added = 0
        if state.get("task_type") == "breach":
            used = state.get("family_extra_count", 0)
            results, added = self._expand_clause_family(results, doc_ids, used)
        logger.info(
            "[retrieve] 命中 %d 条 (%.1fs%s) | %s",
            len(results),
            time.perf_counter() - t0,
            f", 原始问题路补入 {dual_extra} 条" if dual_extra else "",
            "; ".join(f"{r.citation()} score={r.rerank_score:.3f}" for r in results[:3]),
        )
        return {"current_results": results}

    # ---------- 证据校验 + 调度决策 ----------
    def validate(self, state: AgentState) -> dict:
        subquery = state["current_query"]
        results = state.get("current_results", [])
        context = TwoStageRetriever.format_context(results)
        round_idx = state.get("round", 0)
        max_rounds = state.get("max_rounds", self.max_rounds)

        sufficient, rewritten = self._judge(subquery, context) if results else (False, subquery)
        logger.info(
            "[validate] 子问题%d 第%d轮 sufficient=%s rewritten=%r",
            state.get("sub_idx", 0) + 1,
            round_idx + 1,
            sufficient,
            rewritten if not sufficient else "",
        )

        citations = list(state.get("citations", []))

        # 充足 或 已耗尽轮次 → 采纳当前结果，推进到下一子问题
        if sufficient or round_idx + 1 >= max_rounds:
            # 跨组去重：多个子问题（及双路检索）会反复命中同一条款，
            # 只在首次出现的组保留，控制比对/合成的上下文规模（空组后续会被跳过）
            seen = {r.chunk_id for g in citations for r in g}
            fresh = [r for r in results if r.chunk_id not in seen]
            if len(fresh) < len(results):
                logger.info(
                    "[validate] 子问题%d 跨组去重 %d→%d 条",
                    state.get("sub_idx", 0) + 1,
                    len(results),
                    len(fresh),
                )
            citations.append(fresh)
            # 条款族扩展预算按「实际采纳的扩展块」计数（rank==0 为扩展块标记），
            # 未采纳的轮次（证据不足走改写重检）不占用预算
            adopted_extra = sum(1 for r in fresh if r.rank == 0)
            count_update = (
                {"family_extra_count": state.get("family_extra_count", 0) + adopted_extra}
                if adopted_extra
                else {}
            )
            next_idx = state.get("sub_idx", 0) + 1
            subs = state.get("subqueries", [])
            exhausted = not sufficient and round_idx + 1 >= max_rounds
            # 全部子问题完成后，按任务类型进入比对（违约判定）或合成（复杂问答）
            final_next = "breach_compare" if state.get("task_type") == "breach" else "synthesize"
            if next_idx < len(subs):
                logger.info(
                    "[validate] %s -> 推进子问题%d/%d: %r",
                    "轮次耗尽(降级采纳)" if exhausted else "证据充足",
                    next_idx + 1,
                    len(subs),
                    subs[next_idx],
                )
                return {
                    "citations": citations,
                    "sub_idx": next_idx,
                    "round": 0,
                    "current_query": subs[next_idx],
                    "_next": "next",
                    **count_update,
                }
            return {
                "citations": citations,
                "sub_idx": next_idx,
                "_next": final_next,
                **count_update,
            }

        # 不足且仍有轮次 → 改写 query 重检
        logger.info("[validate] 证据不足 -> 改写重检: %r -> %r", subquery, rewritten or subquery)
        return {
            "current_query": rewritten or subquery,
            "round": round_idx + 1,
            "_next": "rewrite",
        }

    # ---------- 违约判定：任务拆解 ----------
    def breach_decompose(self, state: AgentState) -> dict:
        query = state["query"]
        prompt = (
            "你是合同违约判定专家。为判定用户描述的纠纷是否构成违约、责任如何承担，"
            "需要从合同中检索判定依据。将所需信息拆解为若干可独立检索的子问题，"
            "围绕以下三类判定要素：\n"
            "1. 义务与违约情形条款：合同对相关义务的约定、何种行为构成违约\n"
            "2. 违约责任条款：违约后的责任承担方式（违约金、赔偿、解除权等）及计算方式\n"
            "3. 免责与例外条款：不可抗力、免责事由、例外情形\n"
            "根据用户纠纷聚焦的具体主题（如供货逾期、质量不合格、付款拖欠）细化检索词。\n"
            "要求：\n- 每个子问题覆盖一类信息点，简洁、适合向量检索\n"
            "- 最多5个，与纠纷无关的要素可省略\n"
            '仅输出JSON：{"subqueries": ["子问题1", "子问题2", ...]}\n\n'
            f"用户纠纷描述：{query}"
        )
        data = self.llm.chat_json([{"role": "user", "content": prompt}])
        subs = data.get("subqueries") if data else []
        if not isinstance(subs, list) or not subs:
            subs = [query]  # 拆解失败兜底：原问题作为唯一子问题
        subs = [s for s in subs if isinstance(s, str) and s.strip()][:5] or [query]
        logger.info("[breach_decompose] 拆出 %d 个判定要素子问题: %s", len(subs), subs)
        return {
            "subqueries": subs,
            "sub_idx": 0,
            "round": 0,
            "current_query": subs[0],
            "citations": [],
            "verify_round": 0,
            "verify_issues": [],
        }

    # ---------- 违约判定：责任比对 ----------
    def breach_compare(self, state: AgentState) -> dict:
        query = state["query"]
        context, n_total, n_used = self._breach_context(state)
        logger.info(
            "[breach_compare] 基于 %d/%d 条条款上下文进行责任比对", n_used, n_total
        )
        prompt = (
            "你是合同违约判定员。基于合同条款上下文，对用户描述的纠纷进行责任比对。\n"
            "定性框架（务必遵守）：\n"
            "- 用户描述的行为是【假定成立的事实】（as alleged），你的任务是法律定性，"
            "不是事实查证。不得要求用户提供所述行为实际发生的证据"
            "（如检测报告、现场记录、时间线证明、主观动机说明），这类内容一律禁止列入 missing_facts。\n"
            "- 用户以「因……存在争议/有纠纷」为由描述一方行为时：争议本身不构成"
            "中止履行的合同依据，不得把「争议孰是孰非、相对方是否真有过错、"
            "争议解决进展」列为 missing_facts，更不得据此输出 null。"
            "例：「因货款支付存在争议，乙方单方面停止混凝土供应，乙方是否违约？」"
            "——停供是假定事实，若合同（如约定期限/争议期间不得中断供货、"
            "不得擅自停止供货的条款）禁止该行为，直接 claim_established=true、"
            "违约方为停供方；只有合同明确赋予该方在该情形下中止履行权利时才可判 false。\n"
            "- 先锚定主张对象：识别用户在主张【哪一方】违约（claimed_party），"
            "判断【该主张】是否成立（claim_established）。判定字段只回答这个主张；"
            "即使另一方存在其他违约行为，也不得改变对当前主张的判定。\n"
            "比对步骤：\n"
            "1. 从上下文中找出与纠纷相关的条款：义务/违约情形条款、责任条款、免责条款，"
            "记录其 [编号] 与要点\n"
            "2. 将用户假定的事实与合同约定逐条比对：是否符合违约情形、"
            "对应的责任后果、是否存在合同约定的免责事由\n"
            "2a. 【期限/宽限期】涉及付款期限、宽限期、异议期、整改期的，"
            "必须在 reasoning 中显式列出时间线：义务起算点 → 用户给出的经过时长 → "
            "与合同约定的期限（含宽限期）逐一比较。尚在约定期限/宽限期内的行为不构成违约，"
            "不得跳过这一步直接定性。期限一律以用户给出的时间锚点起算"
            "（如「付款到期后第N个月」即从付款到期日起算），不得把合同生效日期、"
            "具体付款节点界定等用户未提及的事项列为 missing_facts。"
            "宽限期是合同预先给予的期限利益（条款本身即授权），不得要求用户另行举证"
            "「对方已同意」或存在补充协议。"
            "示例（必须按此处理）：合同写「如甲方出现资金困难，乙方同意给予3个月的"
            "付款宽限期」，该约定在合同生效时即已授权；用户称资金困难且到期后第2个月"
            "仍未付的，直接结论「在3个月宽限期内，不构成违约」，"
            "禁止以「需乙方另行同意/需通知/需补充协议/待核实」为由输出 true 或 null\n"
            "2b. 【履行顺序与抗辩权】仅当用户描述的事实中存在对应线索时，才检查履行先后约定"
            "（如先开票后付款）或条款但书/例外（如「乙方未履行义务的除外」）："
            "负先履行义务一方未履行时，另一方拒绝或暂缓履行属于约定抗辩，不构成违约。"
            "用户完全未提及的环节（如未提发票状态）不得假设抗辩存在，"
            "不得列入 missing_facts，也不得以「需进一步核实」为由回避定性\n"
            "3. missing_facts 仅允许列：定性或责任计算所必需、且用户未提供的信息"
            "（如违约金计算基数金额、合同明确要求的前置通知）。用户已给出的信息"
            "（逾期天数、金额、行为本身等）视为已知事实\n"
            "4. 合同条款未约定的免责事由（如不可抗力）不得作为免责依据，"
            "也不得要求用户提供合同外证明（如气象证明）。严禁援引合同外的法律条文"
            "（如《民法典》关于不可抗力的规定）、交易惯例或学理来填补合同空白——"
            "判定只以给定合同文本为准。合同未约定免责时，免责主张即不成立，"
            "直接回归适用的责任条款判定违约成立及责任\n"
            "5. 【claim_established 的两种问句形态】"
            "①用户主张「某方构成违约」：该字段=该违约主张是否成立；"
            "②用户问「某违约方能否以不可抗力/对方原因等主张免责或抗辩」："
            "先按责任条款判定其基础违约是否成立，再判断免责是否有【合同内】依据——"
            "基础违约成立且合同未支持免责的，claim_established=true、"
            "breaching_party/claimed_party 均为该违约方，reasoning 中写明"
            "「免责/抗辩主张不成立，仍构成违约」。两种形态都不得输出 null 来回避定性\n"
            '仅输出JSON：{"claimed_party": "被主张违约方（甲方/乙方）", '
            '"grace_period_check": "期限/宽限期检查（不涉及填空字符串）：按'
            '「起算点=用户给的时间锚点；经过时长=用户给的时长；约定期限=合同条款值；'
            '结论=在内/超期」四要素填写。例如「起算=付款到期日；经过=第2个月；'
            '约定=6.5给予3个月宽限期；结论=在内，不构成违约」。'
            '结论只允许填「在内，不构成违约」或「超期，构成违约」，'
            '禁止填「待核实/无法认定/需对方同意」等回避表述", '
            '"defense_check": "履行顺序/条款但书抗辩检查（无线索填空字符串）：'
            '仅依据用户已描述的事实填写，例如「用户称乙方未开票，6.3约定先开票后付款，'
            '甲方拒付属约定抗辩，不构成违约」；用户未提及的环节不得填写猜测", '
            '"claim_established": true/false/null, '
            '"matched_clauses": ["[编号] 条款要点"], '
            '"missing_facts": ["缺失事实"], '
            '"other_party_note": "另一方的其他违约行为，没有则空字符串", '
            '"reasoning": "逐条比对推理（控制在300字以内，只保留关键比对链）"}\n'
            "字段一致性要求（强制）：\n"
            "- grace_period_check 结论为「在内」的，claim_established 必须为 false；\n"
            "- defense_check 认定约定抗辩成立的，claim_established 必须为 false；\n"
            "- 二者为空字符串且无合同内免责依据时，才按 matched_clauses 的责任条款定性。\n"
            "（事实充分时 claim_established 给 true/false；仅当缺少步骤3所列"
            "用户确实未提供的关键信息时才给 null）\n\n"
            f"用户纠纷描述：{query}\n\n合同条款上下文：\n{context}"
        )
        # 复核驳回后重判：把问题清单注入 prompt，要求逐条修正
        feedback = [str(x) for x in (state.get("verify_issues") or []) if str(x).strip()]
        if feedback:
            prompt += "\n\n上一轮判定被独立复核驳回，存在以下问题，本轮务必逐条修正：\n" + "\n".join(
                f"- {x}" for x in feedback
            )
            logger.info("[breach_compare] 复核反馈重判，%d 条问题待修正", len(feedback))
        data = self.llm.chat_json(
            [{"role": "user", "content": prompt}], max_tokens=VERDICT_MAX_TOKENS
        )
        if data:
            # 主张锚定规范化：判定字段统一回答「用户主张的那一方是否违约」
            claim_est = data.get("claim_established")
            claimed_party = data.get("claimed_party")
            if isinstance(claim_est, bool):
                breach_established: object = claim_est
                breaching_party: object = claimed_party if claim_est else None
            else:
                breach_established = data.get("breach_established")
                breaching_party = data.get("breaching_party")
            verdict = {
                "breach_established": breach_established,
                "breaching_party": breaching_party,
                "claimed_party": claimed_party,
                "claim_established": claim_est,
                "grace_period_check": str(data.get("grace_period_check", "") or ""),
                "defense_check": str(data.get("defense_check", "") or ""),
                "matched_clauses": data.get("matched_clauses", []) or [],
                "missing_facts": data.get("missing_facts", []) or [],
                "other_party_note": data.get("other_party_note", "") or "",
                "reasoning": data.get("reasoning", ""),
            }
            # 程序化一致性兜底：模型自己的检查字段已得「期限内在/抗辩成立」结论，
            # 却仍给出 true/null 时，以检查字段为准强制改为 false（防止字段间自相矛盾）
            grace_text = verdict["grace_period_check"]
            defense_text = verdict["defense_check"]
            grace_within = "在内" in grace_text and "超期" not in grace_text
            # 宽限期回避检测：合同含宽限期、用户给了相对时长，模型却以「需对方同意/
            # 待核实/无法认定」回避——按 prompt 既定规则（条款本身即授权）视为在内
            grace_hedged = (
                "宽限" in grace_text
                and bool(re.search(r"第\s*\d+\s*个?月", query))
                and any(k in grace_text for k in ("待核实", "无法认定", "无法直接", "需乙方同意", "需对方同意", "同意权"))
            )
            defense_holds = ("抗辩" in defense_text or "先开票" in defense_text or "先后" in defense_text) and (
                "不构成违约" in defense_text or "不违约" in defense_text
            )
            if grace_hedged:
                logger.info("[breach_compare] 宽限期回避兜底触发：%s", grace_text[:120])
            if isinstance(claim_est, bool) and claim_est and (grace_within or grace_hedged or defense_holds):
                logger.info(
                    "[breach_compare] 一致性兜底：grace_within=%s defense_holds=%s，claim true→false",
                    grace_within,
                    defense_holds,
                )
                claim_est = False
                verdict["claim_established"] = False
                verdict["breach_established"] = False
                verdict["breaching_party"] = None
        else:  # 比对失败兜底：不给出结论并打 parse_error，复核必须打回重判（不得放行）
            verdict = {
                "breach_established": None,
                "breaching_party": None,
                "claimed_party": None,
                "claim_established": None,
                "grace_period_check": "",
                "defense_check": "",
                "matched_clauses": [],
                "missing_facts": [],
                "other_party_note": "",
                "reasoning": "比对分析解析失败，请基于上下文人工判断",
                "parse_error": True,
            }
            logger.error("[breach_compare] 结构化判定 JSON 解析失败（重试后仍失败）")
        logger.info(
            "[breach_compare] 违约成立=%s 违约方=%s 被主张方=%s 命中条款=%d条 缺失事实=%d项%s",
            verdict["breach_established"],
            verdict["breaching_party"],
            verdict.get("claimed_party"),
            len(verdict["matched_clauses"]),
            len(verdict["missing_facts"]),
            " parse_error" if verdict.get("parse_error") else "",
        )
        _check_refs(" ".join(map(str, verdict["matched_clauses"])), n_used, "breach_compare")
        return {"breach_verdict": verdict}

    # ---------- 违约判定：独立复核（推理交叉校验 + 对抗性反方论证） ----------
    def breach_verify(self, state: AgentState) -> dict:
        query = state["query"]
        verdict = state.get("breach_verdict", {})
        context = self._global_context(state)
        round_idx = state.get("verify_round", 0)
        prompt = (
            "你是合同违约判定的独立复核员。用挑剔的眼光审查以下责任比对结果，不预设它正确。\n"
            "审查维度（仅以下情况才判 verdict_ok=false）：\n"
            "1. 条款理解错误：matched_clauses 中 [编号] 对应的条款原文与 reasoning 的断言矛盾，"
            "或存在断章取义、甲乙方责任搞反、比例/数值/天数引用错误。\n"
            "2. 对抗性检验：站在被主张违约方立场找能推翻或削弱判定的理由——"
            "条款例外情形、事实与违约构成要件不匹配。但合同条款未约定的免责事由"
            "（如不可抗力）不得作为推翻依据。\n"
            "3. 期限与履行抗辩（只能基于用户【已给出】的事实审查，严禁自行增设事实）：\n"
            "   (a) 宽限期/付款期限：用户给出相对时长（如「到期后第2个月」）时，"
            "以到期日为锚点直接与约定期限比较；在宽限期内判不违约才是正确结论。"
            "宽限期是合同预先授予的期限利益，要求举证「对方另行同意/补充协议」"
            "或要求提供具体到期日，均属错误打回。\n"
            "   (b) 履行顺序抗辩（如先开票后付款、条款但书）：仅当用户描述中"
            "存在相关事实线索（如明确说乙方未开票、未交付）时才可纳入。"
            "用户完全未提及的事项，严禁以「若该事实存在则…」「需核实发票/XX状态」"
            "为由打回比对结果或要求输出 null——用用户未提及的假设去推翻一个"
            "在已给事实上成立的结论，本身就是复核错误。\n"
            "   (c) 仅当 compare 确实命中 (a)(b) 所述错误时才判 false 并给 corrected；"
            "若 compare 的定性在用户已给事实上成立，verdict_ok=true。\n"
            "4. 合同外依据：比对结果若援引合同文本之外的法律条文（如《民法典》不可抗力）、"
            "交易惯例作为免责/抗辩依据，或以「合同未排除法定免责」为由支持免责，必须判 false，"
            "并在 corrected 中改为按合同责任条款判定。若用户问的是违约方能否免责，"
            "而合同无免责约定，正确结论是基础违约成立、免责不成立（breach_established=true，"
            "breaching_party=违约方），不得反过来输出 false/null。\n"
            "5. 检查字段一致性：grace_period_check 已结论「在内」却输出 true/null、"
            "或 defense_check 已认定约定抗辩成立（不构成违约）却输出 true/null 的，"
            "必须判 false，corrected 中改为 breach_established=false、breaching_party=null。"
            "若 grace_period_check 以「需核实具体到期日/起算点」为由不给结论，而用户"
            "已给相对时长（如到期后第N个月），也属错误，corrected 直接按期限比较定性。\n"
            "重要判定标准：\n"
            "- 用户在问题中已明确描述的事实（如逾期天数、金额、行为等）应视为充分，"
            "不得以「缺失关键事实」为由判 false。\n"
            "- missing_facts 中列出的待补充事实，若用户问题中已有对应信息，不得作为降级依据。\n"
            "- 仅当 missing_facts 中的事实确实不在用户问题描述中、且为判定结论所必需、"
            "且上下文条款也无法覆盖时，才可判 false。\n"
            "- 轻微问题不影响结论时 verdict_ok=true，仍在 issues 中列出。\n"
            "- matched_clauses 为空仅属于呈现瑕疵：只要 claim_established/breaching_party "
            "的定性与上下文条款实质一致（结论生成环节会自行引用条款），不得仅因 "
            "matched_clauses 为空就判 false；在 issues 中提示即可。\n"
            "补充检索触发（need_retrieval）：\n"
            "- 当且仅当当前上下文缺少解决纠纷的关键条款（例如比对引用的条款适用前提"
            "不匹配、可能存在更直接的责任条款未被检索到）时，置 need_retrieval=true，"
            "并用一句适合检索的话在 retrieval_need 中描述要补找的条款要点"
            "（如「单纯逾期交货的违约金计算条款」）。\n"
            "- 正确条款已在上下文中、只是推理用错时，不要触发补检，直接在 corrected 中修正。\n"
            '仅输出JSON：{"verdict_ok": true/false, "issues": ["问题"], '
            '"counter_arguments": ["反方论点"], "need_retrieval": false, '
            '"retrieval_need": "", '
            '"corrected": {"breach_established": ..., "breaching_party": ..., '
            '"matched_clauses": [...], "missing_facts": [...], "reasoning": ...}}\n'
            "（verdict_ok=false 且 need_retrieval=false 时必须给出 corrected 修正版）\n\n"
            f"用户纠纷描述：{query}\n\n"
            f"待复核的责任比对结果：\n{json.dumps(verdict, ensure_ascii=False)}\n\n"
            f"合同条款上下文：\n{context}"
        )
        # parse_error 拦截：比对结果本身无效，强制打回重判，绝不放行或降级
        if verdict.get("parse_error"):
            ok = False
            issues = ["责任比对结果 JSON 解析失败，未形成有效判定，必须重新输出结构化比对结果"]
            counters: list[str] = []
            corrected = None
            need_retrieval, retrieval_need = False, ""
            logger.info("[breach_verify] 第%d次复核 拦截 parse_error 判定，强制重判", round_idx + 1)
        else:
            data = self.llm.chat_json(
                [{"role": "user", "content": prompt}], max_tokens=VERDICT_MAX_TOKENS
            )
            if not data:  # 复核本身解析失败：信任比对结果（compare 自身解析失败另有 parse_error 拦截）
                ok, issues, counters, corrected = True, [], [], None
                need_retrieval, retrieval_need = False, ""
                logger.warning("[breach_verify] 第%d次复核 JSON 解析失败，维持比对结果", round_idx + 1)
            else:
                ok = bool(data.get("verdict_ok"))
                issues = [str(x) for x in (data.get("issues") or []) if str(x).strip()]
                counters = [str(x) for x in (data.get("counter_arguments") or []) if str(x).strip()]
                corrected = data.get("corrected") if isinstance(data.get("corrected"), dict) else None
                need_retrieval = bool(data.get("need_retrieval"))
                retrieval_need = str(data.get("retrieval_need") or "").strip()
        logger.info(
            "[breach_verify] 第%d次复核 verdict_ok=%s 问题=%d 反方论点=%d 补检=%s",
            round_idx + 1,
            ok,
            len(issues),
            len(counters),
            retrieval_need if (need_retrieval and retrieval_need) else "否",
        )

        if ok or round_idx + 1 >= MAX_VERIFY:
            force_caveat = not ok  # 重判后仍不通过 → 结论强制降级
            review = {
                "passed": ok,
                "issues": issues,
                "counter_arguments": counters,
                "force_caveat": force_caveat,
                "round": round_idx + 1,
            }
            updates: dict = {"breach_review": review, "_next": "conclude"}
            if force_caveat and corrected:
                updates["breach_verdict"] = corrected  # 采纳复核修正版，结论仍需降级表述
                logger.info("[breach_verify] 采纳复核修正版判定")
            logger.info(
                "[breach_verify] %s -> breach_conclude",
                "复核通过" if ok else "复核仍未通过(结论降级)",
            )
            return updates

        # 不通过且可补检（每案最多1次）→ 带条款要点回检索环节补充证据，再重新比对
        if need_retrieval and retrieval_need and not state.get("supplement_done"):
            logger.info("[breach_verify] 复核不通过 -> 补充检索: %s", retrieval_need)
            return {
                "supplement_query": retrieval_need,
                "supplement_done": True,
                "verify_round": round_idx + 1,
                "verify_issues": [f"请结合补充检索到的条款重新比对：{'; '.join(issues)}"],
                "breach_review": {
                    "passed": False,
                    "issues": issues,
                    "counter_arguments": counters,
                    "force_caveat": False,
                    "round": round_idx + 1,
                },
                "_next": "supplement",
            }

        # 不通过 → 带问题清单回 breach_compare 重判
        logger.info("[breach_verify] 复核不通过 -> 反馈问题重判: %s", issues)
        return {
            "verify_issues": issues,
            "verify_round": round_idx + 1,
            "breach_review": {
                "passed": False,
                "issues": issues,
                "counter_arguments": counters,
                "force_caveat": False,
                "round": round_idx + 1,
            },
            "_next": "recompare",
        }

    # ---------- 违约判定：复核驱动的补充检索（verify → retrieve → compare） ----------
    def breach_supplement_retrieve(self, state: AgentState) -> dict:
        """复核发现关键条款缺失时，按 retrieval_need 定向补检一轮，作为新证据组追加。"""
        q = state["supplement_query"]
        doc_ids = state.get("current_doc_ids") or None
        logger.info("[supplement] 定向补检 query=%r scope=%s", q, doc_ids or "(全局)")
        t0 = time.perf_counter()
        results = self.retriever.retrieve(q, top_k_final=self.top_k, doc_ids=doc_ids)
        used = state.get("family_extra_count", 0)
        results, _added = self._expand_clause_family(results, doc_ids, used)
        citations = list(state.get("citations", []))
        seen = {r.chunk_id for g in citations for r in g}
        fresh = [r for r in results if r.chunk_id not in seen]
        citations.append(fresh)
        adopted_extra = sum(1 for r in fresh if r.rank == 0)
        logger.info(
            "[supplement] 命中 %d 条（新增 %d 条，%.1fs）| %s",
            len(results),
            len(fresh),
            time.perf_counter() - t0,
            "; ".join(f"{r.citation()} score={r.rerank_score:.3f}" for r in fresh[:3]),
        )
        updates: dict = {"citations": citations}
        if adopted_extra:
            updates["family_extra_count"] = used + adopted_extra
        return updates

    # ---------- 违约判定：结论生成（带溯源引用） ----------
    def breach_conclude(self, state: AgentState) -> dict:
        query = state["query"]
        verdict = state.get("breach_verdict", {})
        context, n_total, n_used = self._breach_context(state)
        logger.info(
            "[breach_conclude] 汇总 %d/%d 条上下文 + 比对结果生成判定结论",
            n_used,
            n_total,
        )
        prompt = (
            "你是合同纠纷分析助手。基于以下责任比对结果与合同条款上下文，"
            "输出违约判定结论。\n"
            "要求：\n"
            "- 先给结论（用户主张的违约是否成立/由谁担责/依据什么条款），再给比对分析过程\n"
            "- 结论必须与责任比对结果的判定字段一致：claim_established=false 时"
            "明确回答「该违约主张不成立」；用户询问违约方能否免责而 claim_established=true 时，"
            "明确回答「合同未约定该免责情形，免责主张不成立，违约仍成立」并给出责任承担；"
            "other_party_note 中另一方的其他违约行为"
            "只能在末尾「补充说明」中呈现，不得改变或冲淡主结论\n"
            "- 引用条款时使用 [编号] 格式（编号对应上下文中的 [1] [2] 等）\n"
            "- 比对结果中 missing_facts 非空时，先判断这些事实是否已在用户的问题描述中给出。"
            "若用户已提供（如逾期天数、金额、行为等），直接据此给出确定性结论；"
            "仅当确实缺少用户未提供且上下文条款也无法覆盖的关键事实时，"
            "才说明「暂无法完全判定」并列出需补充的事实及对应条款\n"
            "- 合同条款未约定的免责事由（如不可抗力）不得作为免责依据，"
            "也不得要求合同外证明；严禁援引合同外法律条文（如《民法典》不可抗力规定）"
            "或以「合同未排除法定免责」为由支持免责；合同无免责约定时应明确告知"
            "「合同未约定该免责情形」，并回归适用的责任条款给出判定\n"
            "- 存在合同约定的免责事由或条款冲突时指出；但仅当用户主动提出免责/不可抗力"
            "等主张时才讨论该话题，用户未问时不要主动展开，也不得在答案中提及"
            "「不得援引合同外法律」之类的内部审查规则或任何法律名称\n"
            "- 只回答用户所问行为对应的责任：附条件后果（如「逾期超3日可解除合同」）"
            "在用户事实未表明条件成就时，不得当作已触发的结论陈述，至多提示该条件\n"
            "- 引用条款编号前必须核对该编号对应的条款原文确实支持你的断言，"
            "断言与条款内容不匹配时宁可不引用，严禁把 A 条款（如质量不合格责任）"
            "套到 B 行为（如逾期供货）上\n"
            "- 责任比对结果是内部结构化数据（含 claimed_party、grace_period_check、"
            "defense_check 等英文键）：答案中严禁出现这些字段名、JSON 原文或"
            "「责任比对结果显示」等系统措辞，只能用中文自然转述其结论\n"
            "- 不得编造条款内容，条款未覆盖的情形如实说明\n\n"
            f"用户纠纷描述：{query}\n\n"
            f"责任比对结果：{json.dumps(verdict, ensure_ascii=False)}\n\n"
            f"合同条款上下文：\n{context}"
        )
        # 复核结论注入：未通过 → 强制降级；通过但有保留意见 → 结论末尾如实呈现
        review = state.get("breach_review") or {}
        caveats = [str(x) for x in (review.get("issues", []) + review.get("counter_arguments", [])) if str(x).strip()]
        if review.get("force_caveat"):
            if verdict.get("breach_established") is None:
                # 无有效判定：硬性中立模板，禁止模型自行读条款后给出确定性结论
                # （教训：模型曾在降级路径自信地把 6.5 宽限期条款张冠李戴，写出错误定性）
                prompt += (
                    "\n\n【最高优先级约束】责任比对未形成有效判定（breach_established=null），"
                    "你不得输出任何「构成违约/不构成违约」的确定性结论，也不得替任何一方"
                    "完成定性推理（例如不得自行套用宽限期、抗辩权等条款下结论）。\n"
                    "输出必须满足：\n"
                    "1. 首句固定为「现有信息不足以作出确定性违约判定」；\n"
                    "2. 其后仅客观说明：复核发现的未解决问题、与纠纷相关的条款要点"
                    "（只复述条款原文含义，不套用定性）、需要补充的事实；\n"
                    "3. 不得援引责任比对结果 matched_clauses 之外的条款作定性依据。\n"
                    "未解决问题：\n" + "\n".join(f"- {x}" for x in caveats)
                )
            else:
                prompt += (
                    "\n\n独立复核仍未通过，存在以下未解决问题。请逐条评估：若问题属于实质性"
                    "逻辑错误（条款理解错误、甲乙方搞反、比例/天数计算错误），结论应以"
                    "「暂无法完全判定」为核心表述；若问题仅为事实补充类（用户已在问题描述中"
                    "提供或上下文可覆盖），可给出确定性判定但附保留意见。列明争议点、"
                    "双方论点及对应条款编号：\n"
                    + "\n".join(f"- {x}" for x in caveats)
                )
            logger.info("[breach_conclude] 复核未通过，结论降级为暂无法完全判定")
        elif caveats:
            prompt += (
                "\n\n独立复核已通过，但存在以下保留意见，请在结论末尾单列「保留意见」小节如实呈现：\n"
                + "\n".join(f"- {x}" for x in caveats)
            )
        answer = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
        _check_refs(answer, n_used, "breach_conclude")
        logger.info("[breach_conclude] 判定结论生成完毕，%d 字", len(answer))
        return {"answer": answer}

    # ---------- 合成 ----------
    def synthesize(self, state: AgentState) -> dict:
        query = state["query"]
        context = self._global_context(state)
        n_total = sum(len(g) for g in state.get("citations", []))
        logger.info("[synthesize] 汇总 %d 条上下文生成答案", n_total)
        answer = self._generate(query, context)
        _check_refs(answer, n_total, "synthesize")
        logger.info("[synthesize] 答案生成完毕，%d 字", len(answer))
        return {"answer": answer}

    # ---------- 内部工具 ----------
    def _global_context(self, state: AgentState) -> str:
        """按 citations 分组重建全局连续编号的上下文。

        各子问题检索结果跨组连续编号（[1]..[N]），保证 LLM 引用的编号与
        main.py 最终渲染的引用清单一一对应，消除两套编号体系的错位。
        """
        return TwoStageRetriever.format_context_grouped(state.get("citations", []))

    @staticmethod
    def _capped_citation_groups(
        state: AgentState, cap: int = COMPARE_CONTEXT_CAP
    ) -> list:
        """跨子问题组按 rerank_score 取全局 top-cap，保留分组结构。

        多子问题检索累积可达 25-30 条，compare 长上下文 + 长 JSON 输出会反复
        触发解析失败。正确条款（如 6.5/9.4/9.7）精排分均在 0.9+，按分截断安全。
        """
        groups = state.get("citations", []) or []
        flat: list[tuple[float, int, int]] = []
        seen: set = set()
        for gi, group in enumerate(groups):
            for k, r in enumerate(group):
                if r.chunk_id in seen:
                    continue
                seen.add(r.chunk_id)
                score = getattr(r, "rerank_score", None)
                if score is None:
                    score = getattr(r, "coarse_score", None) or 0.0
                flat.append((float(score), gi, k))
        if len(flat) <= cap:
            return groups
        keep = {(gi, k) for _, gi, k in sorted(flat, key=lambda x: x[0], reverse=True)[:cap]}
        out = []
        for gi, group in enumerate(groups):
            ng = [r for k, r in enumerate(group) if (gi, k) in keep]
            if ng:
                out.append(ng)
        return out

    def _breach_context(self, state: AgentState) -> tuple[str, int, int]:
        """违约链路（compare/conclude）专用上下文：全局截断到 COMPARE_CONTEXT_CAP。"""
        groups = self._capped_citation_groups(state)
        n_total = sum(len(g) for g in (state.get("citations", []) or []))
        n_used = sum(len(g) for g in groups)
        return TwoStageRetriever.format_context_grouped(groups), n_total, n_used

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
        sufficient = bool(data.get("sufficient"))
        rewritten = data.get("rewritten_query", "") or ""
        return sufficient, rewritten.strip()

    def _generate(self, query: str, context: str) -> str:
        """基于上下文生成带引用的答案。"""
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
