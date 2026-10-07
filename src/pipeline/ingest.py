"""文档入库编排：PDF 解析 → 表格抽取 → 条款分块 → BGE-M3 向量化 → FAISS 入库 + BM25 索引。

落盘产物（均在 store_dir）：
  - index.faiss / chunks.pkl：稠密支路，BGE-M3 向量 + 元数据（下标对齐）
  - bm25.pkl：稀疏支路，jieba 分词后的 token 列表（与 chunks 下标对齐）
  - tables.pkl：表格结构化旁路存储（markdown + 单元格，供未来工具调用）
检索时两路粗排经 RRF 融合后喂 reranker，见 retriever.py。
表格抽取见 table_extractor.py：markdown 表格生成独立 chunk 可检索可引用，
原始碎片行从正文流剔除避免重复索引。

被 ingest_contract.py (CLI) 调用，也可作为库函数直接调用。
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

from ..retriever.bm25_store import BM25Store, BM25_FILE
from ..retriever.embedder import BGEM3Embedder, DEFAULT_MODEL_DIR
from .chunker import Chunk, ClauseChunker
from .pdf_parser import PDFParser
from .table_extractor import TableExtractor, save_tables
from .vector_store import VectorStore

DEFAULT_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "vector_store",
)


def _build_ocr():
    """惰性构造 RapidOCR，按需下载其内置 PP-OCR 模型。"""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _doc_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def ingest(
    data_path: str,
    store_dir: str = DEFAULT_STORE_DIR,
    model_dir: str = DEFAULT_MODEL_DIR,
    max_len: int = 600,
    overlap_lines: int = 2,
    embed_batch: int = 32,
    dpi: int = 200,
    ocr: Optional[object] = None,
) -> VectorStore:
    """对 data_path 下所有 PDF 做解析、分块、向量化并落盘。

    Args:
        data_path: PDF 所在目录或单个 PDF 文件。
        store_dir: 向量库落盘目录。
        model_dir: BGE-M3 模型缓存目录。
        max_len / overlap_lines: 分块参数。
        embed_batch: BGE-M3 批大小。
        dpi: 扫描页 OCR 渲染分辨率。
        ocr: 已构造的 RapidOCR 实例（None 则按需构造，文字 PDF 不触发）。
    """
    pdf_paths = _collect_pdfs(data_path)
    if not pdf_paths:
        raise FileNotFoundError(f"未在 {data_path} 找到 PDF")

    ocr_engine = ocr  # 不提前构造；遇到扫描件再建
    parser = PDFParser(ocr_engine=ocr_engine, dpi=dpi)
    chunker = ClauseChunker(max_len=max_len, overlap_lines=overlap_lines)
    table_extractor = TableExtractor()
    embedder = BGEM3Embedder(model_dir=model_dir)

    all_chunks: List[Chunk] = []
    all_tables: List[dict] = []
    print(f"[ingest] 共 {len(pdf_paths)} 个 PDF，开始解析与分块 ...")
    for path in pdf_paths:
        doc_id = _doc_id_from_path(path)
        t0 = time.time()
        pages = parser.parse(path)
        # 遇到扫描件且尚未构造 OCR 时，补构造后重解析当前文档
        if pages and any(p.is_ocr for p in pages) and parser.ocr is None:
            raise RuntimeError(
                f"PDF {doc_id} 为扫描件但未提供 OCR 引擎，请构造 RapidOCR 后传入。"
            )
        if not pages or all(not p.text.strip() for p in pages):
            print(f"  - {doc_id}: 解析后无文本，跳过")
            continue
        # 表格结构化抽取：markdown 表格生成独立 chunk，原始碎片行从正文流剔除
        tables = table_extractor.extract(path, pages, doc_id=doc_id, file_name=os.path.basename(path))
        all_tables.extend(t.to_dict() for t in tables)
        chunks = chunker.chunk(pages, doc_id=doc_id, file_name=os.path.basename(path))
        chunks.extend(chunker.build_table_chunks(tables, doc_id=doc_id, start_idx=len(chunks)))
        all_chunks.extend(chunks)
        ocr_flag = "OCR" if pages and pages[0].is_ocr else "text"
        tbl_flag = f"，{len(tables)} 张表格" if tables else ""
        print(f"  - {doc_id}: {len(pages)} 页 / {len(chunks)} 块{tbl_flag} [{ocr_flag}] ({time.time()-t0:.1f}s)")

    if not all_chunks:
        raise RuntimeError("所有文档解析分块后无可用 chunk，入库终止。")

    texts = [c.text for c in all_chunks]
    print(f"[ingest] 共 {len(texts)} 个 chunk，开始 BGE-M3 向量化 ...")
    t0 = time.time()
    vecs = embedder.encode(texts, batch_size=embed_batch)
    print(f"[ingest] 向量化完成，shape={vecs.shape} ({time.time()-t0:.1f}s)")

    store = VectorStore(dim=embedder.dim)
    metas = [c.to_dict() for c in all_chunks]
    store.add(vecs, metas)
    store.save(store_dir)
    print(f"[ingest] FAISS 索引已落盘：{store_dir}（共 {len(store)} 条）")

    # BM25 稀疏索引：jieba 分词后落盘 bm25.pkl，tokens 与 chunks 下标对齐。
    # 检索时与 BGE-M3 粗排经 RRF 融合后喂 reranker，弥补稠密检索对合同
    # 强关键词（「第 9.4 条」「逾期」「违约金」）命中不稳的问题。
    print(f"[ingest] 构建 BM25 索引（jieba 分词）...")
    t0 = time.time()
    bm25_store = BM25Store.build(texts)
    bm25_path = os.path.join(store_dir, BM25_FILE)
    bm25_store.save(bm25_path)
    print(f"[ingest] BM25 索引已落盘：{bm25_path}（共 {len(bm25_store)} 条，{time.time()-t0:.1f}s）")

    # 表格结构化旁路存储：markdown + 单元格，独立于向量库，供结构化查询/工具调用
    tables_path = save_tables(all_tables, store_dir)
    print(f"[ingest] 表格旁路存储已落盘：{tables_path}（共 {len(all_tables)} 张）")
    return store


def _collect_pdfs(data_path: str) -> List[str]:
    if os.path.isfile(data_path):
        return [data_path] if data_path.lower().endswith(".pdf") else []
    return sorted(
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if f.lower().endswith(".pdf")
    )
