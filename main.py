"""TradeContractAgent CLI：交互式合同问答 demo。

用法：
    python main.py                              # 全局交互问答
    python main.py "问题"                       # 全局单次问答
    python main.py --doc <doc_id> "问题"         # 限定单份合同问答
    python main.py --doc <doc_id>               # 限定单份合同交互问答
    python main.py --list-docs                  # 列出已入库的合同 doc_id
"""
from __future__ import annotations

import argparse
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
# 压掉第三方库的噪音日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from src.agent import ContractAgent


def _print_citations(citations: list) -> None:
    seen: set = set()
    n = 0  # 全局编号，与 LLM 上下文中的 [N] 一致（跨组连续）；去重仅跳过打印，不重排编号
    printed = 0
    for group in citations:
        for r in group:
            n += 1
            key = (r.doc_id, r.clause_label, r.page_start)
            if key in seen:
                continue
            seen.add(key)
            printed += 1
            print(f"[{n}] {r.citation()} (rerank={r.rerank_score:.3f})")
    if printed == 0:
        print("(无引用)")


def _run_once(agent: ContractAgent, query: str, doc_ids: list[str] | None) -> None:
    t0 = time.perf_counter()
    try:
        result = agent.ask(query, doc_ids=doc_ids)
    except Exception as e:
        print(f"[错误] {e}")
        return
    elapsed = time.perf_counter() - t0
    print("\n--- 答案 ---")
    print(result.get("answer", "(无答案)"))
    print("\n--- 引用来源 ---")
    _print_citations(result.get("citations", []))
    print(f"\n(耗时 {elapsed:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description="贸易合同 Agentic RAG 问答 demo", allow_abbrev=False)
    ap.add_argument("query", nargs="*", help="问题（不提供则进入交互模式）")
    ap.add_argument("--doc", help="限定本次问答的合同 doc_id（可用 --list-docs 查询）")
    ap.add_argument("--list-docs", action="store_true", help="列出已入库的合同 doc_id 后退出")
    args = ap.parse_args()

    print("初始化检索底座与 Agent（首次加载模型较慢，请稍候）...")
    agent = ContractAgent()

    if args.list_docs:
        docs = sorted({c["doc_id"] for c in agent.retriever.store.chunks})
        print(f"已入库 {len(docs)} 份合同：")
        for d in docs:
            print(f"  {d}")
        return

    doc_ids = [args.doc] if args.doc else None
    if doc_ids:
        print(f"限定合同范围：{doc_ids[0]}")

    query = " ".join(args.query).strip()
    if query:
        _run_once(agent, query, doc_ids)
        return

    print("就绪。输入问题开始问答，Ctrl+D 退出。")
    while True:
        try:
            q = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if not q:
            continue
        _run_once(agent, q, doc_ids)


if __name__ == "__main__":
    main()
