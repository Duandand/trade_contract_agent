"""LLM-as-judge：faithfulness（答案是否被引用支撑）。"""
from __future__ import annotations

import logging

from ..agent.llm import LLMClient

logger = logging.getLogger("eval")

# 注意：模板中的 JSON 示例花括号需双写转义（.format 会把单花括号当替换字段）
_JUDGE_PROMPT = """你是合同问答评测员。判断「答案」是否被「引用上下文」完整支撑，即答案中的每个事实/数字/条款结论都能在引用上下文中找到依据。

判定规则：
- supported：答案所有关键事实都能在引用中找到依据，无臆断、无外部知识补全
- unsupported：存在答案中的关键事实/数字/结论在引用中找不到依据，或与引用矛盾
- partial：部分支撑，但有个别非核心事实超出引用范围

仅输出JSON：{{"label": "supported/unsupported/partial", "reasoning": "简短说明，指出哪条事实无支撑"}}

用户问题：{query}

引用上下文：
{context}

答案：
{answer}
"""

def judge_faithfulness(
    query: str,
    answer: str,
    context: str,
    llm: LLMClient,
) -> dict:
    """返回 {label, reasoning, pass}。pass = label == supported。"""
    if not answer or not context:
        return {"label": "unsupported", "reasoning": "答案或上下文为空", "pass": False}
    prompt = _JUDGE_PROMPT.format(query=query, context=context, answer=answer)
    data = llm.chat_json([{"role": "user", "content": prompt}])
    if not data:
        return {"label": "unknown", "reasoning": "判定 JSON 解析失败，默认不通过", "pass": False}
    label = (data.get("label") or "").strip().lower()
    reasoning = (data.get("reasoning") or "").strip()
    return {"label": label, "reasoning": reasoning, "pass": label == "supported"}