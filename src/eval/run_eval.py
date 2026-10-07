"""评测 CLI：跑黄金集 → JSON + Markdown 报告。

用法：
    python run_eval.py
    python run_eval.py --golden my_golden.jsonl --output r.json --md r.md --no-judge
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("agent").setLevel(logging.INFO)

from src.eval.runner import EvalRunner, render_markdown

def main() -> int:
    ap = argparse.ArgumentParser(description="合同问答 Agent 评测：跑黄金集并生成指标报告")
    ap.add_argument("--golden", default="src/eval/golden_qa.example.jsonl", help="黄金集 jsonl 路径")
    ap.add_argument("--output", default="data/eval/report.json", help="JSON 报告输出路径")
    ap.add_argument("--md", default="data/eval/report.md", help="Markdown 报告输出路径")
    ap.add_argument("--model", default=None, help="Ollama 模型名（默认 DEFAULT_MODEL）")
    ap.add_argument("--store_dir", default=None, help="向量库目录")
    ap.add_argument("--mode", choices=["agent", "basic"], default="agent",
                    help="agent=完整 Agentic RAG；basic=BasicRAG 基线（单次检索一次判定，无拆解/复核）")
    ap.add_argument("--no-judge", action="store_true", help="跳过 LLM-as-judge（更快，仅看检索/路由指标）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.mode == "basic":
        from ..agent import DEFAULT_MODEL
        from ..agent.llm import LLMClient
        from ..retriever.retriever import DEFAULT_STORE_DIR
        from .baseline import BasicBreachRAG
        agent = BasicBreachRAG(
            model=args.model or DEFAULT_MODEL,
            store_dir=args.store_dir or DEFAULT_STORE_DIR,
        )
        judge_llm = LLMClient(model=args.model or DEFAULT_MODEL) if not args.no_judge else None
        runner = EvalRunner(agent=agent, llm_for_judge=judge_llm, use_judge=not args.no_judge)
    else:
        runner = EvalRunner.from_defaults(
            model=args.model,
            store_dir=args.store_dir,
            use_judge=not args.no_judge,
        )
    report = runner.run(args.golden)
    report.save(args.output)
    with open(args.md, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    print(f"\n[eval] JSON 报告：{args.output}")
    print(f"[eval] Markdown 报告：{args.md}")
    m = report.metrics
    print(f"[eval] 路由准确率：{m['routing_accuracy']:.2%} | 引用召回：{m['avg_citation_recall']:.2%}")
    return 0

if __name__ == "__main__":
    sys.exit(main())