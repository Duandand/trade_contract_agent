"""bge-reranker-v2-m3 精排封装。

定位：二级检索的精排器。对粗排召回的候选片段与 query 逐对打分重排，
显著提升条款级相关性。与 embedder 共用同一份模型缓存目录。
"""
from __future__ import annotations

import os
from typing import List

from FlagEmbedding import FlagReranker

from .embedder import DEFAULT_MODEL_DIR

DEFAULT_RERANKER_NAME = "BAAI/bge-reranker-v2-m3"


class BGEReranker:
    """bge-reranker-v2-m3 精排器。"""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_NAME,
        model_dir: str = DEFAULT_MODEL_DIR,
        use_fp16: bool = False,
    ) -> None:
        os.environ.setdefault("HF_HOME", model_dir)
        os.environ.setdefault("HF_HUB_CACHE", model_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", model_dir)
        self.model_name = model_name
        self._reranker = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(
        self,
        query: str,
        texts: List[str],
        normalize: bool = True,
        max_length: int = 1024,
    ) -> List[float]:
        """对 (query, text) 对打分。

        Args:
            normalize: True 返回 sigmoid 归一化后的 [0,1] 概率分；
                       False 返回原始 logit。检索重排通常用 normalize=True。
        Returns: 与 texts 等长的分数列表。
        """
        if not texts:
            return []
        pairs = [[query, t] for t in texts]
        scores = self._reranker.compute_score(pairs, normalize=normalize, max_length=max_length)
        # 单条输入时 compute_score 返回 float，统一成 list
        if isinstance(scores, float):
            scores = [scores]
        return [float(s) for s in scores]
