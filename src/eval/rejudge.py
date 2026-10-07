"""忠实度补判：对已生成的评测报告重新执行 LLM-as-judge 并回写。

背景：judge prompt 模板曾含未转义花括号导致 .format 抛 KeyError，
历史上跑出的报告 faithfulness 全为 null。本脚本利用报告中保存的
retrieved_chunk_ids + chunks.pkl 重建 judge 上下文（编号顺序与
runner 的 format_context_grouped 扁平化顺序一致），补判后回写报告，
并重算 faithfulness 聚合指标，无需重跑昂贵的 agent 管线。

用法：
    python -m src.eval.rejudge data/eval/report_breach_basic.json
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle

from ..agent.llm import LLMClient
from ..retriever.retriever import TABLE_SNIPPET_CHARS
from .judge import judge_faithfulness

logger = logging.getLogger("eval")

MAX_CHARS = 500


def _load_chunks(store_dir: str) -> dict[str, dict]:
    with open(f"{store_dir}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return {c["chunk_id"]: c for c in chunks}


def _citation(c: dict) -> str:
    page_start, page_end = c.get("page_start"), c.get("page_end", c.get("page_start"))
    page = f"p{page_start}" if page_start == page_end else f"p{page_start}-{page_end}"
    label = c.get("clause_label", "")
    if label.startswith("表格"):
        return f"《{c['doc_id']}》{label} ({page})"
    label = f"条款 {label}" if label and label != "首部" else "首部"
    return f"《{c['doc_id']}》{label} ({page})"


def rebuild_context(chunk_ids: list[str], by_id: dict[str, dict]) -> str:
    """按扁平顺序重建编号 [1..N] 的 judge 上下文（与 runner 的分组连续编号一致）。"""
    parts = []
    for n, cid in enumerate(chunk_ids, 1):
        c = by_id.get(cid)
        if not c:
            continue
        max_chars = max(MAX_CHARS, TABLE_SNIPPET_CHARS) if c.get("clause_label", "").startswith("表格") else MAX_CHARS
        parts.append(f"[{n}] {_citation(c)}\n{c['text'][:max_chars]}")
    return "\n\n".join(parts)


def rejudge(report_path: str, store_dir: str) -> None:
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    by_id = _load_chunks(store_dir)
    llm = LLMClient()

    results = report["case_results"]
    for i, cr in enumerate(results, 1):
        if not cr.get("answer") or cr.get("error"):
            continue
        context = rebuild_context(cr.get("retrieved_chunk_ids", []), by_id)
        try:
            j = judge_faithfulness(cr["case"]["query"], cr["answer"], context, llm)
            cr["faithfulness_pass"] = j["pass"]
            cr["faithfulness_reasoning"] = j["reasoning"]
        except Exception as e:  # noqa: BLE001 - 单条失败不影响整体补判
            logger.warning("[rejudge] case %s 判定失败: %s", cr["case"]["id"], e)
            cr["faithfulness_pass"] = None
            cr["faithfulness_reasoning"] = f"judge error: {e}"
        logger.info("[rejudge] (%d/%d) %s faith=%s", i, len(results), cr["case"]["id"], cr["faithfulness_pass"])

    # 重算聚合：总体 + 按类型
    faith_vals = [c["faithfulness_pass"] for c in results if c.get("faithfulness_pass") is not None]
    report["metrics"]["faithfulness_pass_rate"] = (
        sum(1 for v in faith_vals if v) / len(faith_vals) if faith_vals else None
    )
    by_type: dict[str, list[bool]] = {}
    for c in results:
        if c.get("faithfulness_pass") is not None:
            by_type.setdefault(c["case"]["task_type"], []).append(c["faithfulness_pass"])
    for t, vals in by_type.items():
        if t in report["metrics"].get("by_type", {}):
            report["metrics"]["by_type"][t]["faith_pass"] = sum(vals) / len(vals)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("[rejudge] 已回写 %s | faithfulness=%s", report_path, report["metrics"]["faithfulness_pass_rate"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description="对评测报告补判忠实度并回写")
    ap.add_argument("report", help="评测报告 JSON 路径（原地回写）")
    ap.add_argument("--store_dir", default="data/vector_store", help="向量库目录（读 chunks.pkl）")
    args = ap.parse_args()
    rejudge(args.report, args.store_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
