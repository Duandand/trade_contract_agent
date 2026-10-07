"""评测数据结构与黄金集 schema。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class GoldenCase:
    id: str
    query: str
    task_type: str
    # doc_scope: 限定本 case 检索范围（doc_id 列表）；None/空=全局检索。
    # 用于评测「这份合同的供应商是谁」这类定向查询——runner 把它传给 agent.ask(doc_ids=...)，
    # must_cite 不必再写 doc_id，只校验「是否命中首部 chunk」即可。
    doc_scope: list[str] = field(default_factory=list)
    # must_cite: 每条 spec 是 {doc_id?, clause_label?, page_start?} 的子集；
    # 检索结果命中所有 spec 指定字段即视为该 spec 被覆盖
    must_cite: list[dict] = field(default_factory=list)
    expected_answer_points: list[str] = field(default_factory=list)
    # breach 专用：期望的 {breach_established, breaching_party}
    expected_verdict: Optional[dict] = None
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenCase":
        return cls(
            id=d["id"],
            query=d["query"],
            task_type=d["task_type"],
            doc_scope=d.get("doc_scope", []),
            must_cite=d.get("must_cite", []),
            expected_answer_points=d.get("expected_answer_points", []),
            expected_verdict=d.get("expected_verdict"),
            note=d.get("note", ""),
        )

@dataclass
class CaseResult:
    case: GoldenCase
    actual_task_type: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    # 每条 {doc_id, clause_label, page_start, page_end}
    retrieved_citations: list[dict] = field(default_factory=list)
    answer: str = ""
    verdict: Optional[dict] = None
    elapsed: float = 0.0
    # 指标
    citation_recall: float = 0.0  # 0..1；无 must_cite 时为 -1
    routing_correct: bool = False
    verdict_correct: Optional[bool] = None  # 仅 breach
    faithfulness_pass: Optional[bool] = None  # LLM-as-judge
    faithfulness_reasoning: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class EvalReport:
    total: int
    metrics: dict
    case_results: list[dict]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

def load_golden(path: str) -> list[GoldenCase]:
    """读取黄金集 jsonl（每行一个 JSON 对象；空行/# 开头跳过）。"""
    cases: list[GoldenCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON 解析失败: {e}") from e
            cases.append(GoldenCase.from_dict(obj))
    if not cases:
        raise ValueError(f"黄金集 {path} 为空")
    return cases