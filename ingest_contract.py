"""文档入库 CLI。

用法：
    python ingest_contract.py --data_path ./test_data --store_dir ./data/vector_store
"""
from __future__ import annotations

import argparse
import sys

from src.pipeline.ingest import DEFAULT_STORE_DIR, ingest, _build_ocr


def main() -> int:
    ap = argparse.ArgumentParser(description="贸易采购合同 PDF 解析 + 分块 + 向量入库")
    ap.add_argument("--data_path", default="./test_data", help="PDF 文件或目录")
    ap.add_argument("--store_dir", default=DEFAULT_STORE_DIR, help="向量库落盘目录")
    ap.add_argument("--model_dir", default=None, help="BGE-M3 模型缓存目录（默认项目 ./models）")
    ap.add_argument("--max_len", type=int, default=600, help="单 chunk 最大字符数")
    ap.add_argument("--overlap_lines", type=int, default=2, help="长条款滑窗重叠行数")
    ap.add_argument("--embed_batch", type=int, default=32, help="BGE-M3 编码批大小")
    ap.add_argument("--dpi", type=int, default=200, help="扫描页 OCR 渲染分辨率")
    ap.add_argument("--no_ocr", action="store_true", help="禁用 OCR（仅处理文字层 PDF）")
    args = ap.parse_args()

    ocr = None if args.no_ocr else _build_ocr()
    kwargs = dict(
        data_path=args.data_path,
        store_dir=args.store_dir,
        max_len=args.max_len,
        overlap_lines=args.overlap_lines,
        embed_batch=args.embed_batch,
        dpi=args.dpi,
        ocr=ocr,
    )
    if args.model_dir:
        kwargs["model_dir"] = args.model_dir

    ingest(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
