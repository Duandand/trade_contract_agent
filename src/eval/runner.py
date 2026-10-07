"""评测运行器：跑黄金集 → 聚合指标 → 生成报告。"""
from __future__ import annotations

import logging
import time
from typing import Optional

from ..agent import DEFAULT_MODEL, ContractAgent
from ..agent.llm import LLMClient
from ..retriever.retriever import DEFAULT_STORE_DIR, RetrievalResult, TwoStageRetriever
from .judge import judge_faithfulness
from .metrics import citation_recall, routing_correct, verdict_match
from .schema import CaseResult, EvalReport, GoldenCase, load_golden

logger = logging.getLogger("eval")

class EvalRunner:
    def __init__(
        self,
        agent: ContractAgent,
        llm_for_judge: Optional[LLMClient] = None,
        use_judge: bool = True,
    ) -> None:
        self.agent = agent
        self.llm_for_judge = llm_for_judge
        self.use_judge = use_judge

    @classmethod
    def from_defaults(
        cls,
        model: Optional[str] = None,
        store_dir: Optional[str] = None,
        use_judge: bool = True,
    ) -> "EvalRunner":
        agent = ContractAgent(model=model or DEFAULT_MODEL, store_dir=store_dir or DEFAULT_STORE_DIR)
        judge_llm = LLMClient(model=model or DEFAULT_MODEL) if use_judge else None
        return cls(agent=agent, llm_for_judge=judge_llm, use_judge=use_judge)

    def run_case(self, case: GoldenCase) -> CaseResult:
        result = CaseResult(case=case, actual_task_type="?", elapsed=0.0)
        citations_grouped: list[list[RetrievalResult]] = []
        t0 = time.perf_counter()
        try:
            # doc_scope 限定检索范围（定向查询场景）；None=全局
            doc_ids = case.doc_scope or None
            state = self.agent.ask(case.query, doc_ids=doc_ids)
            result.actual_task_type = state.get("task_type", "?")
            citations_grouped = state.get("citations", []) or []
            flat = [r for g in citations_grouped for r in g]
            result.retrieved_chunk_ids = [r.chunk_id for r in flat]
            result.retrieved_citations = [
                {
                    "doc_id": r.doc_id,
                    "clause_label": r.clause_label,
                    "page_start": r.page_start,
                    "page_end": r.page_end,
                }
                for r in flat
            ]
            result.answer = state.get("answer", "")
            if case.task_type == "breach":
                result.verdict = state.get("breach_verdict")
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            logger.error("[eval] case %s 执行失败: %s", case.id, result.error)
        finally:
            result.elapsed = time.perf_counter() - t0

        # 指标
        result.citation_recall = citation_recall(result.retrieved_citations, case.must_cite)
        result.routing_correct = routing_correct(case.task_type, result.actual_task_type)
        if case.task_type == "breach":
            result.verdict_correct = verdict_match(case.expected_verdict, result.verdict)

        # LLM-as-judge
        if self.use_judge and self.llm_for_judge is not None and result.answer:
            context = TwoStageRetriever.format_context_grouped(citations_grouped)
            try:
                j = judge_faithfulness(case.query, result.answer, context, self.llm_for_judge)
                result.faithfulness_pass = j["pass"]
                result.faithfulness_reasoning = j["reasoning"]
            except Exception as e:
                logger.warning("[eval] case %s LLM 判定失败: %s", case.id, e)
                result.faithfulness_pass = None
                result.faithfulness_reasoning = f"judge error: {e}"
        return result

    def run(self, golden_path: str) -> EvalReport:
        cases = load_golden(golden_path)
        logger.info("[eval] 加载 %d 条黄金用例", len(cases))
        results: list[CaseResult] = []
        for i, case in enumerate(cases, 1):
            logger.info("[eval] (%d/%d) 跑用例 %s task=%s", i, len(cases), case.id, case.task_type)
            r = self.run_case(case)
            results.append(r)
            logger.info(
                "[eval] 完成 %s: route=%s recall=%.2f faith=%s verdict=%s (%.1fs)",
                case.id,
                r.routing_correct,
                r.citation_recall,
                r.faithfulness_pass,
                r.verdict_correct,
                r.elapsed,
            )
        return self._aggregate(results)

    def _aggregate(self, results: list[CaseResult]) -> EvalReport:
        total = len(results)
        routing_acc = sum(1 for r in results if r.routing_correct) / total if total else 0.0
        recall_vals = [r.citation_recall for r in results if r.citation_recall >= 0]
        avg_recall = sum(recall_vals) / len(recall_vals) if recall_vals else 0.0
        faith_vals = [r.faithfulness_pass for r in results if r.faithfulness_pass is not None]
        faith_pass = (sum(1 for v in faith_vals if v) / len(faith_vals)) if faith_vals else None
        breach_results = [r for r in results if r.case.task_type == "breach"]
        breach_correct = (
            sum(1 for r in breach_results if r.verdict_correct) / len(breach_results)
            if breach_results else None
        )
        avg_latency = sum(r.elapsed for r in results) / total if total else 0.0
        errors = sum(1 for r in results if r.error)

        by_type: dict[str, dict] = {}
        for r in results:
            t = r.case.task_type
            v = by_type.setdefault(t, {"n": 0, "routing": 0, "recall_sum": 0.0, "recall_n": 0, "faith": 0, "faith_n": 0})
            v["n"] += 1
            v["routing"] += int(r.routing_correct)
            if r.citation_recall >= 0:
                v["recall_sum"] += r.citation_recall
                v["recall_n"] += 1
            if r.faithfulness_pass is not None:
                v["faith_n"] += 1
                v["faith"] += int(r.faithfulness_pass)
        by_type_summary = {
            t: {
                "n": v["n"],
                "routing_acc": v["routing"] / v["n"] if v["n"] else 0.0,
                "avg_recall": v["recall_sum"] / v["recall_n"] if v["recall_n"] else 0.0,
                "faith_pass": v["faith"] / v["faith_n"] if v["faith_n"] else None,
            }
            for t, v in by_type.items()
        }

        metrics = {
            "total": total,
            "routing_accuracy": routing_acc,
            "avg_citation_recall": avg_recall,
            "faithfulness_pass_rate": faith_pass,
            "breach_verdict_accuracy": breach_correct,
            "avg_latency_s": avg_latency,
            "errors": errors,
            "by_type": by_type_summary,
        }
        return EvalReport(
            total=total,
            metrics=metrics,
            case_results=[r.to_dict() for r in results],
        )

def render_markdown(report: EvalReport) -> str:
    m = report.metrics
    lines = [
        "# 评测报告",
        "",
        f"- 总用例数：{m['total']}",
        f"- 路由准确率：{m['routing_accuracy']:.2%}",
        f"- 平均引用召回：{m['avg_citation_recall']:.2%}",
    ]
    if m["faithfulness_pass_rate"] is not None:
        lines.append(f"- 答案忠实度通过率：{m['faithfulness_pass_rate']:.2%}")
    else:
        lines.append("- 答案忠实度通过率：N/A")
    if m["breach_verdict_accuracy"] is not None:
        lines.append(f"- 违约判定准确率：{m['breach_verdict_accuracy']:.2%}")
    else:
        lines.append("- 违约判定准确率：N/A")
    lines += [
        f"- 平均耗时：{m['avg_latency_s']:.1f}s",
        f"- 执行错误数：{m['errors']}",
        "",
        "## 按任务类型",
        "",
        "| 类型 | n | 路由 | 召回 | 忠实 |",
        "|---|---|---|---|---|",
    ]
    for t, v in m["by_type"].items():
        faith = f"{v['faith_pass']:.2%}" if v["faith_pass"] is not None else "N/A"
        lines.append(f"| {t} | {v['n']} | {v['routing_acc']:.2%} | {v['avg_recall']:.2%} | {faith} |")
    lines += ["", "## 明细", ""]
    for cr in report.case_results:
        case = cr["case"]
        lines.append(f"### {case['id']} `{case['task_type']}`")
        lines.append(f"- 问题：{case['query']}")
        lines.append(
            f"- 路由：期望 {case['task_type']} / 实际 {cr['actual_task_type']} → "
            f"{'✅' if cr['routing_correct'] else '❌'}"
        )
        if cr["citation_recall"] >= 0:
            lines.append(f"- 引用召回：{cr['citation_recall']:.2%}")
        if cr["faithfulness_pass"] is not None:
            mark = "✅" if cr["faithfulness_pass"] else "❌"
            lines.append(f"- 忠实度：{mark} {cr['faithfulness_reasoning']}")
        if cr["verdict_correct"] is not None:
            lines.append(f"- 违约判定：{'✅' if cr['verdict_correct'] else '❌'}")
        if cr["error"]:
            lines.append(f"- ❌ 执行错误：{cr['error']}")
        lines.append(f"- 耗时：{cr['elapsed']:.1f}s")
        lines.append("")
    return "\n".join(lines)