"""Ollama Qwen 封装：统一 chat 接口 + JSON 提取。

模型默认 qwen3.5:9b-q4_K_M（本地 Ollama 服务，需先 `ollama serve`）。
qwen3.5 支持 thinking，响应可能含 <think>...</think>，提取前先剥离。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import ollama

DEFAULT_MODEL = "qwen3.5:9b-q4_K_M"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

logger = logging.getLogger("agent")


class LLMClient:
    """对 ollama.chat 的轻封装，提供纯文本与 JSON 两种调用。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.options = {"temperature": temperature, "num_predict": max_tokens}

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """同步对话，返回去除 <think> 后的纯文本。"""
        opts = dict(self.options)
        if temperature is not None:
            opts["temperature"] = temperature
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        # think=False 关闭 qwen3.5 thinking，避免思考内容在 thinking 字段
        # 占满 num_predict 导致 content 为空（合同长 prompt 会触发深度思考）
        resp = ollama.chat(model=self.model, messages=messages, options=opts, think=False)
        return self._strip_think(resp["message"]["content"])

    def chat_json(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        retries: int = 1,
        max_tokens: Optional[int] = None,
    ) -> Optional[dict]:
        """对话并从响应中提取 JSON 对象，失败时追加纠错提示重试，最终失败返回 None。"""
        attempt = 0
        convo = list(messages)
        while True:
            text = self.chat(convo, temperature=temperature, max_tokens=max_tokens)
            data = self._extract_json(text)
            if data is not None:
                return data
            attempt += 1
            if attempt > retries:
                return None
            logger.warning("[llm] JSON 解析失败，第 %d 次重试", attempt)
            convo = convo + [
                {"role": "assistant", "content": text[:2000]},
                {
                    "role": "user",
                    "content": (
                        "你上一次的输出无法被解析为合法 JSON。请重新输出，"
                        "且只能包含一个合法 JSON 对象：不要使用 ```代码块标记、"
                        "不要输出注释或任何解释文字，确保所有引号、括号配对完整。"
                    ),
                },
            ]

    @staticmethod
    def _strip_think(text: str) -> str:
        return _THINK_RE.sub("", text).strip()

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        text = LLMClient._strip_think(text)
        # 1) 优先匹配 ```json ... ``` 代码块
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # 2) 取第一个 { 到最后一个 } 之间内容
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except json.JSONDecodeError:
                pass
        return None
