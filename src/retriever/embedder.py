"""BGE-M3 稠密向量编码器封装。

定位：检索底座的粗排编码器，文档入库（pipeline.ingest）与查询检索（retriever）
共用同一实例与同一份模型缓存，保证 doc/query 向量空间一致。
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
from FlagEmbedding import BGEM3FlagModel

# 项目根目录与模型缓存（复用已下载的 ./models）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_MODEL_NAME = "BAAI/bge-m3"


class BGEM3Embedder:
    """BGE-M3 稠密向量编码器（粗排）。

    输出 L2 归一化的 1024 维向量，可直接用内积（IndexFlatIP）做余弦相似度检索。
    """

    DENSE_DIM = 1024

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_dir: str = DEFAULT_MODEL_DIR,
        use_fp16: bool = False,
    ) -> None:
        # 复用项目本地缓存，避免重复下载
        os.environ.setdefault("HF_HOME", model_dir)
        os.environ.setdefault("HF_HUB_CACHE", model_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", model_dir)
        self.model_name = model_name
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.DENSE_DIM), dtype="float32")
        out = self._model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vecs = np.asarray(out["dense_vecs"], dtype="float32")
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs

    @property
    def dim(self) -> int:
        return self.DENSE_DIM
