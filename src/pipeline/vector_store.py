"""FAISS 向量库：索引构建、持久化、加载。

使用 IndexIDMap2(IndexFlatIP)（内积）+ L2 归一化向量 = 余弦相似度。
IDMap2 让每个向量带显式 int64 id（此处用 chunks 列表下标），支持：
  - 检索时按 IDSelector 过滤（按 doc_id 限定检索范围）
  - remove_ids 删除（未来增量入库，见 todo #9）
旁路 chunks.pkl 保存与向量一一对应的元数据（含原文与定位信息），用于引用溯源。
"""
from __future__ import annotations

import os
import pickle
from typing import List, Optional

# 线程数限制已在 src/__init__.py（包入口）统一设置，此处不再重复。
import faiss
import numpy as np

INDEX_FILE = "index.faiss"
META_FILE = "chunks.pkl"


class VectorStore:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        # IndexIDMap2 包装 IndexFlatIP：支持显式 id + IDSelector 过滤 + remove_ids
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.chunks: List[dict] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, vecs: np.ndarray, metas: List[dict]) -> None:
        if vecs.shape[0] == 0:
            return
        if vecs.shape[1] != self.dim:
            raise ValueError(f"向量维度 {vecs.shape[1]} 与索引维度 {self.dim} 不一致")
        # 用 chunks 列表下标作为 id，与 metas 一一对齐；IDMap2 要求 int64 且唯一
        start = self.index.ntotal
        ids = np.arange(start, start + vecs.shape[0], dtype="int64")
        self.index.add_with_ids(np.ascontiguousarray(vecs, dtype="float32"), ids)
        self.chunks.extend(metas)

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, INDEX_FILE))
        with open(os.path.join(dir_path, META_FILE), "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, dir_path: str) -> "VectorStore":
        store = cls()
        store.index = faiss.read_index(os.path.join(dir_path, INDEX_FILE))
        with open(os.path.join(dir_path, META_FILE), "rb") as f:
            store.chunks = pickle.load(f)
        return store

    def ids_in_scope(self, doc_ids: Optional[List[str]]) -> Optional[np.ndarray]:
        """把 doc_id 列表转成 FAISS id（chunks 下标）数组；None 或无命中返回 None。

        retrieve 时若返回 None 则走全局检索；否则构造 IDSelectorBatch 限定范围。
        """
        if not doc_ids:
            return None
        target = set(doc_ids)
        ids = [i for i, c in enumerate(self.chunks) if c.get("doc_id") in target]
        if not ids:
            return None
        return np.array(ids, dtype="int64")
