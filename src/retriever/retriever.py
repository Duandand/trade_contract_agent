"""二级检索：粗排(BGE-M3 稠密 + BM25 稀疏，RRF 融合) → bge-reranker-v2-m3 精排。

复用 ingest 落盘的 FAISS 库（index.faiss + chunks.pkl）与 BM25 索引（bm25.pkl）。
每个结果携带粗排/精排双分数与原文定位元数据，供 Agent 引用溯源。
支持按 doc_id 限定检索范围（粗排期 IDSelector 过滤），用于「这份合同」类定向查询。

混合检索动机：合同里「第 9.4 条」「逾期」「违约金」是强关键词，纯稠密易混入
「付款/结算」语义近邻；BM25 按词频精准命中关键词，两路经 RRF 融合后取长补短，
再喂 reranker 精排。bm25.pkl 不存在时自动退化为纯 BGE 粗排（向后兼容旧库）。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from ..pipeline.vector_store import VectorStore, INDEX_FILE
from .bm25_store import BM25Store, BM25_FILE
from .embedder import BGEM3Embedder, DEFAULT_MODEL_DIR
from .reranker import BGEReranker

logger = logging.getLogger("retriever")

# RRF 融合常数，Cormack et al. 2009 默认值；rank 越靠前贡献越大。
RRF_K = 60

DEFAULT_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "vector_store",
)


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    file_name: str
    page_start: int
    page_end: int
    clause_label: str
    text: str
    coarse_score: float  # 粗排分（混合检索 RRF 融合分；无 BM25 索引时为 BGE-M3 余弦相似度）
    rerank_score: float  # reranker 归一化分
    rank: int

    def to_dict(self) -> dict:
        return asdict(self)

    def citation(self) -> str:
        """人类可读引用串，例如：《合同名》条款 1.2.2 (p2-3)；表格为《合同名》表格p2#1 (p2)。"""
        page = f"p{self.page_start}" if self.page_start == self.page_end else f"p{self.page_start}-{self.page_end}"
        if self.clause_label.startswith("表格"):
            return f"《{self.doc_id}》{self.clause_label} ({page})"
        label = f"条款 {self.clause_label}" if self.clause_label and self.clause_label != "首部" else "首部"
        return f"《{self.doc_id}》{label} ({page})"


# 表格 chunk 携带完整 markdown 表格，截断会切掉表尾数据行，放宽到 900 字符
TABLE_SNIPPET_CHARS = 900


def _snippet(r: "RetrievalResult", max_chars: int) -> str:
    if r.clause_label.startswith("表格"):
        max_chars = max(max_chars, TABLE_SNIPPET_CHARS)
    return r.text[:max_chars]


class TwoStageRetriever:
    """BGE-M3 粗排 + bge-reranker-v2-m3 精排。"""

    def __init__(
        self,
        store_dir: str = DEFAULT_STORE_DIR,
        model_dir: str = DEFAULT_MODEL_DIR,
        embedder: Optional[BGEM3Embedder] = None,
        reranker: Optional[BGEReranker] = None,
    ) -> None:
        if not os.path.exists(os.path.join(store_dir, INDEX_FILE)):
            raise FileNotFoundError(f"未找到 FAISS 索引：{os.path.join(store_dir, INDEX_FILE)}，请先跑 ingest_contract.py")
        self.store = VectorStore.load(store_dir)
        self.embedder = embedder or BGEM3Embedder(model_dir=model_dir)
        self.reranker = reranker or BGEReranker(model_dir=model_dir)
        # BM25 稀疏索引（可选）：与 BGE-M3 粗排经 RRF 融合。
        # 不存在（旧库未重灌）或长度不一致时退化为纯 BGE 粗排，不强制重灌。
        bm25_path = os.path.join(store_dir, BM25_FILE)
        self.bm25_store: Optional[BM25Store] = None
        if os.path.exists(bm25_path):
            try:
                self.bm25_store = BM25Store.load(bm25_path, expected_len=len(self.store.chunks))
                logger.info("已加载 BM25 索引，启用混合检索（BGE-M3 + BM25 RRF 融合）")
            except Exception as e:
                logger.warning("BM25 索引加载失败，退化为纯 BGE 粗排：%s", e)
                self.bm25_store = None
        else:
            logger.info(
                "未找到 %s，使用纯 BGE 粗排（重新跑 ingest_contract.py 可启用混合检索）",
                bm25_path,
            )

    def retrieve(
        self,
        query: str,
        top_k_final: int = 5,
        top_k_coarse: int = 20,
        doc_ids: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """两阶段检索：粗排(BGE-M3 + BM25 RRF 融合) → reranker 精排。

        Args:
            query: 用户问题。
            top_k_final: 最终返回的精排结果数。
            top_k_coarse: 粗排召回的候选数（≥ top_k_final）。
            doc_ids: 限定检索范围（按合同 doc_id 过滤）；None 表示全局检索。
                粗排期 BGE 走 IDSelectorBatch、BM25 走下标屏蔽，两路范围一致，
                避免把 reranker 配额浪费给无关合同。
        """
        if self.store.index.ntotal == 0:
            return []
        if top_k_coarse < top_k_final:
            top_k_coarse = top_k_final
        k = min(top_k_coarse, self.store.index.ntotal)

        # doc_ids → chunks 下标集合（BGE 用 IDSelector、BM25 用下标屏蔽）
        scope_ids = self.store.ids_in_scope(doc_ids)
        if scope_ids is not None:
            k = min(k, len(scope_ids))
            if k == 0:
                return []

        # 1a) BGE-M3 粗排（可选 doc_id 过滤）
        qv = np.ascontiguousarray(self.embedder.encode([query]), dtype="float32")
        if scope_ids is not None:
            sel = faiss.IDSelectorBatch(scope_ids)
            params = faiss.SearchParameters(sel=sel)
            bge_scores, bge_idx = self.store.index.search(qv, k, params=params)
        else:
            bge_scores, bge_idx = self.store.index.search(qv, k)
        # FAISS 返回已按相似度降序；rank 即在返回序列中的位置
        bge_candidates: List[Tuple[int, float, int]] = []  # (chunk_idx, cosine_score, rank)
        for rank, (s, i) in enumerate(zip(bge_scores[0], bge_idx[0])):
            if 0 <= i < len(self.store.chunks):
                bge_candidates.append((int(i), float(s), rank))

        use_hybrid = self.bm25_store is not None and self.bm25_store.n > 0

        if use_hybrid:
            # 1b) BM25 粗排：全量打分后按 doc_ids 屏蔽非 scope 位置
            bm25_arr = np.asarray(self.bm25_store.get_scores(query), dtype="float64")
            if scope_ids is not None:
                masked = np.full(self.bm25_store.n, -np.inf, dtype="float64")
                masked[scope_ids] = bm25_arr[scope_ids]
                bm25_arr = masked
            # 降序取前 k，仅保留 > 0（有词项重叠）的候选；0 分候选不进 RRF
            n_bm25 = min(k, self.bm25_store.n)
            bm25_order = np.argsort(bm25_arr)[::-1][:n_bm25]
            bm25_candidates: List[Tuple[int, float, int]] = []  # (chunk_idx, bm25_score, rank)
            for rank, i in enumerate(bm25_order):
                s = bm25_arr[i]
                if s > 0:
                    bm25_candidates.append((int(i), float(s), rank))

            # 1c) RRF 融合：score = sum 1/(RRF_K+rank_bge) + 1/(RRF_K+rank_bm25)
            #     只出现在一路的候选仅取该路贡献；两路都命中的候选取和（互相印证）
            fused: Dict[int, float] = {}
            for idx, _, rank in bge_candidates:
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
            for idx, _, rank in bm25_candidates:
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
            fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)
            top = fused_sorted[:top_k_coarse]
            cand_idx = [idx for idx, _ in top]
            c_scores = [s for _, s in top]  # RRF 融合分
        else:
            # 退化：纯 BGE 粗排，coarse_score 为 BGE-M3 余弦相似度
            bge_top = bge_candidates[:top_k_coarse]
            cand_idx = [idx for idx, _, _ in bge_top]
            c_scores = [s for _, s, _ in bge_top]

        if not cand_idx:
            return []
        candidates = [self.store.chunks[i] for i in cand_idx]

        # 2) bge-reranker-v2-m3 精排
        texts = [c["text"] for c in candidates]
        r_scores = self.reranker.rerank(query, texts, normalize=True)

        # 按精排分降序
        order = sorted(range(len(candidates)), key=lambda j: r_scores[j], reverse=True)
        results: List[RetrievalResult] = []
        for rank, j in enumerate(order[:top_k_final]):
            c = candidates[j]
            results.append(
                RetrievalResult(
                    chunk_id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    file_name=c["file_name"],
                    page_start=c["page_start"],
                    page_end=c["page_end"],
                    clause_label=c["clause_label"],
                    text=c["text"],
                    coarse_score=c_scores[j],
                    rerank_score=r_scores[j],
                    rank=rank + 1,
                )
            )
        return results

    @staticmethod
    def format_context(results: List[RetrievalResult], max_chars_per_chunk: int = 500) -> str:
        """把检索结果拼成带编号引用的上下文，供 LLM prompt 使用。"""
        if not results:
            return ""
        parts: List[str] = []
        for r in results:
            parts.append(f"[{r.rank}] {r.citation()}\n{_snippet(r, max_chars_per_chunk)}")
        return "\n\n".join(parts)

    @staticmethod
    def format_context_grouped(
        groups: List[List[RetrievalResult]], max_chars_per_chunk: int = 500
    ) -> str:
        """多组检索结果拼成一个上下文，编号跨组全局连续。

        与 format_context 的区别：后者每组各自从 [1] 编号，多组拼接会产生重复
        编号，导致 LLM 引用的 [N] 与最终引用清单对不上。本方法从 1 连续编号，
        保证 LLM 引用的 [N] 与 main.py 渲染的引用清单一一对应。空组跳过。
        """
        parts: List[str] = []
        n = 0
        for group in groups:
            for r in group:
                n += 1
                parts.append(f"[{n}] {r.citation()}\n{_snippet(r, max_chars_per_chunk)}")
        return "\n\n".join(parts)
