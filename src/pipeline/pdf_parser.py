"""PDF 解析：扫描件走 OCR，文字 PDF 直取文本层，统一输出按页结构（含行级 bbox）。

行级 bbox 用于后续引用溯源，可定位到合同某页某区域。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pymupdf


@dataclass
class LineInfo:
    text: str
    bbox: List[float]  # [x0, y0, x1, y1]，OCR 模式下坐标已按页像素缩放
    score: Optional[float] = None  # OCR 置信度，文字层模式为 None


@dataclass
class PageContent:
    page_no: int  # 1-based
    text: str
    lines: List[LineInfo] = field(default_factory=list)
    is_ocr: bool = False


def _has_text_layer(doc: "pymupdf.Document", min_spans: int = 20) -> bool:
    """判断 PDF 是否有可用文字层；扫描件基本没有，需走 OCR。"""
    spans = 0
    for page in doc:
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            for l in b.get("lines", []):
                spans += len(l.get("spans", []))
        if spans >= min_spans:
            return True
    return spans >= min_spans


class PDFParser:
    """统一 PDF 解析器。

    Args:
        ocr_engine: RapidOCR 实例；为 None 时仅支持文字层 PDF。
        dpi: 渲染扫描页的分辨率，200 基本够用，复杂小字可调到 300。
    """

    def __init__(self, ocr_engine=None, dpi: int = 200) -> None:
        self.ocr = ocr_engine
        self.dpi = dpi

    def parse(self, path: str) -> List[PageContent]:
        doc = pymupdf.open(path)
        try:
            use_ocr = self.ocr is not None and not _has_text_layer(doc)
            return [
                self._ocr_page(page, i + 1) if use_ocr else self._text_page(page, i + 1)
                for i, page in enumerate(doc)
            ]
        finally:
            doc.close()

    def _text_page(self, page, page_no: int) -> PageContent:
        d = page.get_text("dict")
        lines: List[LineInfo] = []
        parts: List[str] = []
        for b in d.get("blocks", []):
            for l in b.get("lines", []):
                t = "".join(s.get("text", "") for s in l.get("spans", []))
                if t.strip():
                    lines.append(LineInfo(text=t, bbox=list(l.get("bbox", [0, 0, 0, 0]))))
                    parts.append(t)
        return PageContent(page_no=page_no, text="\n".join(parts), lines=lines, is_ocr=False)

    def _ocr_page(self, page, page_no: int) -> PageContent:
        mat = pymupdf.Matrix(self.dpi / 72.0, self.dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if img.shape[2] == 4:  # 极少情况出现 4 通道
            img = img[:, :, :3]

        result, _ = self.ocr(img)
        lines: List[LineInfo] = []
        parts: List[str] = []
        if result:
            for box, text, score in result:
                text = text.strip()
                if not text:
                    continue
                pts = np.asarray(box).reshape(-1, 2)
                x0, y0 = pts[:, 0].min(), pts[:, 1].min()
                x1, y1 = pts[:, 0].max(), pts[:, 1].max()
                lines.append(LineInfo(text=text, bbox=[float(x0), float(y0), float(x1), float(y1)], score=float(score)))
                parts.append(text)
        return PageContent(page_no=page_no, text="\n".join(parts), lines=lines, is_ocr=True)
