"""评测框架：黄金集 schema + 指标 + LLM-as-judge + 运行器 + 标注转换。"""
from .schema import CaseResult, EvalReport, GoldenCase, load_golden
from .metrics import citation_recall, routing_correct, verdict_match
from .judge import judge_faithfulness
from .runner import EvalRunner, render_markdown
from .csv_to_jsonl import csv_to_jsonl

__all__ = [
    "GoldenCase",
    "CaseResult",
    "EvalReport",
    "load_golden",
    "citation_recall",
    "routing_correct",
    "verdict_match",
    "judge_faithfulness",
    "EvalRunner",
    "render_markdown",
    "csv_to_jsonl",
]