"""评测指标：纯函数，不依赖 LLM。"""
from __future__ import annotations

from typing import Optional

def _match_spec(spec: dict, citation: dict) -> bool:
    """spec 中指定的字段必须全部命中检索结果才算覆盖。"""
    for k in ("doc_id", "clause_label"):
        if k in spec and spec[k] is not None:
            if str(spec[k]) != str(citation.get(k, "")):
                return False
    if "page_start" in spec and spec["page_start"] is not None:
        try:
            target = int(spec["page_start"])
            cs = int(citation.get("page_start", target))
            ce = int(citation.get("page_end", cs))
            if not (cs <= target <= ce):
                return False
        except (TypeError, ValueError):
            return False
    return True

def citation_recall(retrieved_citations: list[dict], must_cite: list[dict]) -> float:
    """must_cite 中有多少 spec 至少被一条检索结果命中。空 must_cite 返回 -1（不参与）。"""
    if not must_cite:
        return -1.0
    covered = sum(
        1 for spec in must_cite if any(_match_spec(spec, c) for c in retrieved_citations)
    )
    return covered / len(must_cite)

def routing_correct(expected: str, actual: str) -> bool:
    return expected == actual

def verdict_match(expected: Optional[dict], actual: Optional[dict]) -> Optional[bool]:
    """breach 判定匹配：breach_established 一致 + breaching_party 一致（允许子串包含）。

    双方一致判定「不违约」时直接通过：不成立场景下违约方字段无意义，
    模型可能输出 null/"无"/"双方均不违约" 等，不应因文本差异误判。
    """
    if not expected:
        return None
    if not actual:
        return False
    if expected.get("breach_established") != actual.get("breach_established"):
        return False
    if expected.get("breach_established") is False:
        return True
    exp_party = (expected.get("breaching_party") or "").strip().lower()
    act_party = (actual.get("breaching_party") or "").strip().lower()
    if not exp_party and not act_party:
        return True
    if exp_party and act_party and (exp_party in act_party or act_party in exp_party):
        return True
    return False