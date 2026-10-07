"""BM25 稀疏检索索引（中文 jieba 分词）。

与 FAISS 向量库并列落盘 `bm25.pkl`，tokens 与 chunks 下标一一对应。
检索时与 BGE-M3 粗排结果经 RRF 融合后喂 reranker（见 retriever.py）。

为什么必须 jieba：合同是中文，`rank_bm25` 默认按空格切分对中文无效
（整段会被当成一个 token）。jieba 分词后「逾期」「违约金」「第 9.4 条」
等强关键词才能在 BM25 词频统计里命中。
"""
from __future__ import annotations

import pickle
from typing import List, Optional

import jieba  # noqa: F401  导入即触发 jieba 初始化
import numpy as np
from rank_bm25 import BM25Okapi

BM25_FILE = "bm25.pkl"


class BM25Store:
    """BM25Okapi 索引；token 列表与 chunks 按下标对齐。

    只负责「打分」：给定 query 返回每个 chunk 的 BM25 分数数组。
    doc_ids 过滤与 top-k 截断交给调用方（retriever）处理，保持单一职责。
    """

    def __init__(self, tokens: List[List[str]]) -> None:
        self.tokens = tokens
        self.n = len(tokens)
        # 空 corpus 会让 BM25Okapi 内部除零，惰性构造
        self.bm25: Optional[BM25Okapi] = BM25Okapi(tokens) if tokens else None

    def __len__(self) -> int:
        return self.n

    @classmethod
    def build(cls, texts: List[str]) -> "BM25Store":
        """对每段文本 jieba 分词后构造 BM25Okapi。"""
        tokens = [jieba.lcut(t) for t in texts]
        return cls(tokens)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.tokens, f)

    @classmethod
    def load(cls, path: str, expected_len: Optional[int] = None) -> "BM25Store":
        """从 bm25.pkl 加载 tokens。

        Args:
            path: bm25.pkl 路径。
            expected_len: 若给定且与 chunks 长度不一致则抛错，提示重灌。
        """
        with open(path, "rb") as f:
            tokens = pickle.load(f)
        if expected_len is not None and len(tokens) != expected_len:
            raise ValueError(
                f"BM25 索引长度 {len(tokens)} 与 chunks 长度 {expected_len} 不一致，"
                f"请重新跑 ingest_contract.py 构建 BM25 索引"
            )
        return cls(tokens)

    def get_scores(self, query: str) -> np.ndarray:
        """返回 shape (n,) 的 BM25 分数数组，下标对齐 chunks。

        空索引或 query 无可分词 token 时返回全零数组。调用方负责 doc_ids
        过滤与 top-k 截断——这样 BM25Store 只关心稀疏打分，过滤与融合策略
        集中在 retriever，便于后续替换 RRF 为加权融合等其它策略。
        """
        if self.bm25 is None:
            return np.zeros(self.n, dtype="float32")
        q_tokens = jieba.lcut(query)
        if not q_tokens:
            return np.zeros(self.n, dtype="float32")
        return np.asarray(self.bm25.get_scores(q_tokens), dtype="float32")
