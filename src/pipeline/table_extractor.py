"""表格结构化抽取（todo #3）：PDF 表格 → markdown 表格 + table_records 旁路存储。

动机：合同里的价格表/付款节点表被 pdf_parser 拆成逐单元格的碎片行，
条款分块后「单价 390」「819000」散落为独立行，违约金比例/金额类查询
无法命中完整语义。本模块在 chunker 之前把表格重建为 markdown，交由
ClauseChunker 生成独立表格 chunk（可检索、可引用），原始碎片行从正文
流中剔除避免重复索引；结构化单元格另存 tables.pkl 供未来工具调用。

双路径：
  - 文字层 PDF：pymupdf 内置 page.find_tables()（免装 pdfplumber/camelot，
    且与 pdf_parser 行 bbox 同一坐标系，剔除碎片行可靠）。
  - 扫描件（当前 test_data 全部是图片型 PDF）：基于 OCR 行盒几何重建——
    行聚类成 band、锚定 band 对判定列结构、向下扩展 region，重建 markdown。
    启发式规则，换来的是当前全扫描件语料上的实际收益；无法识别时静默跳过，
    行为与旧版一致（碎片行照旧进条款块），不会更差。

定位说明：文字层 bbox 为 PDF 坐标（points）；扫描件 bbox 为 OCR 像素坐标
（dpi 缩放）。仅用于溯源参考，两者不混用。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pymupdf

from .pdf_parser import LineInfo, PageContent

logger = logging.getLogger("table_extractor")

# ---- OCR 重建阈值（单位：OCR 像素，按行盒相对尺寸自适应） ----
BAND_JOIN_OVERLAP = 0.5   # 行加入当前 band 所需的垂直重叠比例（占该行高度）
TABLE_LIKE_MIN_SEGS = 3   # band 内 ≥3 个水平隔离的片段才视为「表格行」
SEG_GAP_FACTOR = 0.4      # 片段水平间距 ≥ 0.4×行高中位数才认为真正分列（表头短标签间距小）
CONT_MAX_SEGS = 2         # ≤2 个片段的 band 视为「续行/合计行」候选
CONT_GAP_FACTOR = 4.0     # 续行与表格 region 底部距离 ≤ 4×行高中位数才并入
TERM_WIDTH_RATIO = 0.25   # 单片段宽度 > 25% region 跨度 → 判为正文行，终止 region
MIN_ROWS = 3              # region 至少 3 行（表头+数据+1）才算表格
MIN_COLS = 3              # 至少 3 个非空列


@dataclass
class TableRecord:
    """一张表格的结构化结果，旁路存储于 tables.pkl。"""

    doc_id: str
    file_name: str
    page_no: int  # 1-based
    table_index: int  # 页内序号，1-based
    mode: str  # "text"（文字层 find_tables）/ "ocr"（扫描件行盒重建）
    n_rows: int
    n_cols: int
    bbox: List[float]  # [x0, y0, x1, y1]，文字层为 PDF 坐标，OCR 为像素坐标
    markdown: str
    rows: List[List[str]] = field(default_factory=list)  # 原始单元格

    def to_dict(self) -> dict:
        return asdict(self)


def build_markdown(rows: List[List[str]]) -> str:
    """单元格二维列表 → markdown 表格；首行作表头。管道符/换行转义防破表。"""
    if not rows:
        return ""
    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    n_cols = max(len(r) for r in rows)
    lines = ["| " + " | ".join(esc(c) for c in r + [""] * (n_cols - len(r))) + " |" for r in rows]
    lines.insert(1, "|" + "|".join([" --- "] * n_cols) + "|")
    return "\n".join(lines)


class TableExtractor:
    """PDF 表格抽取器：文字层走 find_tables，扫描件走 OCR 行盒重建。"""

    def __init__(self, min_rows: int = MIN_ROWS, min_cols: int = MIN_COLS) -> None:
        self.min_rows = min_rows
        self.min_cols = min_cols

    def extract(
        self, path: str, pages: List[PageContent], doc_id: str, file_name: str
    ) -> List[TableRecord]:
        """抽取表格并从 pages 中剔除已入表的碎片行（就地修改）。

        表格来源页的原始行若保留，会与表格 chunk 重复索引同一批数字，
        因此抽取成功后把对应行从 PageContent.lines/text 中移除。
        """
        if not pages:
            return []
        # 路径判定与 PDFParser 一致（pages[0].is_ocr 由解析器决定），保证行 bbox
        # 坐标系与抽取逻辑匹配：文字层走 find_tables，扫描件走 OCR 行盒重建。
        if pages[0].is_ocr:
            records = self._extract_ocr(pages, doc_id, file_name)
        else:
            doc = pymupdf.open(path)
            try:
                records = self._extract_text_layer(doc, pages, doc_id, file_name)
            finally:
                doc.close()
        if records:
            logger.info("%s: 抽取到 %d 张表格", doc_id, len(records))
        return records

    # ---------- 路径一：文字层 PDF ----------

    def _extract_text_layer(
        self, doc: "pymupdf.Document", pages: List[PageContent], doc_id: str, file_name: str
    ) -> List[TableRecord]:
        records: List[TableRecord] = []
        for page in doc:
            page_no = page.number + 1
            try:
                finder = page.find_tables()
            except Exception as e:  # 个别页线条异常不影响整体入库
                logger.warning("%s p%d find_tables 失败：%s", doc_id, page_no, e)
                continue
            if not finder or not finder.tables:
                continue
            pc = pages[page.number] if page.number < len(pages) else None
            removed: Set[int] = set()  # 待剔除的 pages 行下标
            for t_idx, table in enumerate(finder.tables, start=1):
                rows = [
                    ["" if c is None else str(c).strip() for c in row]
                    for row in table.extract()
                ]
                rows = [r for r in rows if any(c for c in r)]
                if len(rows) < self.min_rows:
                    continue
                n_cols = max(len(r) for r in rows)
                if n_cols < self.min_cols:
                    continue
                bbox = [float(v) for v in table.bbox]
                if pc is not None:
                    removed |= {id(pc.lines[i]) for i in self._lines_in_bbox(pc.lines, bbox, margin=2.0)}
                records.append(
                    TableRecord(
                        doc_id=doc_id,
                        file_name=file_name,
                        page_no=page_no,
                        table_index=t_idx,
                        mode="text",
                        n_rows=len(rows),
                        n_cols=n_cols,
                        bbox=bbox,
                        markdown=build_markdown(rows),
                        rows=rows,
                    )
                )
            if pc is not None and removed:
                self._drop_lines(pc, removed)
        return records

    @staticmethod
    def _lines_in_bbox(lines: List[LineInfo], bbox: Sequence[float], margin: float = 0.0) -> Set[int]:
        """返回中心点落在 bbox（外扩 margin）内的行下标。"""
        x0, y0, x1, y1 = bbox
        out: Set[int] = set()
        for i, ln in enumerate(lines):
            bx = ln.bbox
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            if x0 - margin <= cx <= x1 + margin and y0 - margin <= cy <= y1 + margin:
                out.add(i)
        return out

    # ---------- 路径二：扫描件 OCR 行盒重建 ----------

    def _extract_ocr(
        self, pages: List[PageContent], doc_id: str, file_name: str
    ) -> List[TableRecord]:
        records: List[TableRecord] = []
        for pc in pages:
            if len(pc.lines) < TABLE_LIKE_MIN_SEGS * 2:
                continue
            bands = self._cluster_bands(pc.lines)
            page_tables = 0  # 页内序号按页重置
            for region_idx in self._detect_regions(bands):
                rows, used = self._build_rows(bands, region_idx)
                if len(rows) < self.min_rows:
                    continue
                n_cols = max(len(r) for r in rows)
                # 剔除空列（全空列不进 markdown/rows）
                keep = [j for j in range(n_cols) if any(r[j].strip() for r in rows)]
                rows = [[r[j] for j in keep] for r in rows]
                if len(keep) < self.min_cols:
                    continue
                page_tables += 1
                xs0 = min(ln.bbox[0] for ln in used)
                ys0 = min(ln.bbox[1] for ln in used)
                xs1 = max(ln.bbox[2] for ln in used)
                ys1 = max(ln.bbox[3] for ln in used)
                records.append(
                    TableRecord(
                        doc_id=doc_id,
                        file_name=file_name,
                        page_no=pc.page_no,
                        table_index=page_tables,
                        mode="ocr",
                        n_rows=len(rows),
                        n_cols=len(keep),
                        bbox=[float(xs0), float(ys0), float(xs1), float(ys1)],
                        markdown=build_markdown(rows),
                        rows=rows,
                    )
                )
                self._drop_lines(pc, {id(ln) for ln in used})
        return records

    @staticmethod
    def _cluster_bands(lines: List[LineInfo]) -> List[dict]:
        """按垂直重叠把行聚成视觉行 band；band 内片段按 x 排序。

        换行单元格（如「型号规」+「格」）垂直交叠归入同一 band；紧凑相邻行
        （如合计行与备注行）仅少量重叠时保持独立 band，靠后续续行规则处理。
        """
        order = sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        bands: List[dict] = []
        for ln in order:
            y0, y1 = ln.bbox[1], ln.bbox[3]
            h = max(y1 - y0, 1.0)
            placed = False
            for band in reversed(bands):
                by0, by1 = band["y0"], band["y1"]
                overlap = min(y1, by1) - max(y0, by0)
                span = by1 - by0
                # 双守卫：与 band 重叠 ≥50% 自身高度，且 ≥30% band 跨度——
                # 防止竖排大标签（如「合同主要内容」，高几百 px）把邻近行吸进同 band
                if overlap >= BAND_JOIN_OVERLAP * h and overlap >= 0.3 * span:
                    band["lines"].append(ln)
                    band["y0"] = min(by0, y0)
                    band["y1"] = max(by1, y1)
                    placed = True
                    break
                if by1 < y0:  # band 按起始 y 有序，越过即不再回看
                    break
            if not placed:
                bands.append({"y0": y0, "y1": y1, "lines": [ln]})
        for band in bands:
            band["lines"].sort(key=lambda ln: ln.bbox[0])
        bands.sort(key=lambda b: b["y0"])
        return bands

    @staticmethod
    def _median_height(bands: List[dict]) -> float:
        hs = [b["y1"] - b["y0"] for b in bands]
        hs = [h for h in hs if h > 0]
        if not hs:
            return 1.0
        hs.sort()
        return hs[len(hs) // 2]

    @staticmethod
    def _is_table_like(band: dict, med_h: float) -> bool:
        """band 是否像表格行：≥3 个片段且相邻片段水平间距 ≥ 0.8×行高中位数。"""
        segs = band["lines"]
        if len(segs) < TABLE_LIKE_MIN_SEGS:
            return False
        min_gap = SEG_GAP_FACTOR * med_h
        for a, b in zip(segs, segs[1:]):
            if b.bbox[0] - a.bbox[2] < min_gap:
                return False
        return True

    @staticmethod
    def _columns_from(segments: List[LineInfo]) -> List[List[float]]:
        """把片段 x 区间聚合成列区间（交叠或近邻合并）。"""
        ivs = sorted([[ln.bbox[0], ln.bbox[2]] for ln in segments])
        cols: List[List[float]] = []
        for x0, x1 in ivs:
            if cols and x0 <= cols[-1][1] + 1.0:  # 相邻片段天然同列，容差 1px
                cols[-1][1] = max(cols[-1][1], x1)
            else:
                cols.append([x0, x1])
        return cols

    @classmethod
    def _assign_col(cls, ln: LineInfo, cols: List[List[float]]) -> Optional[int]:
        """按最大 x 重叠把片段分到列；与任何列无重叠返回 None。"""
        best, best_ov = None, 0.0
        for j, (cx0, cx1) in enumerate(cols):
            ov = min(ln.bbox[2], cx1) - max(ln.bbox[0], cx0)
            if ov > best_ov:
                best, best_ov = j, ov
        return best

    @classmethod
    def _aligned(cls, b1: dict, b2: dict) -> bool:
        """两个 band 列结构一致：b2 每个片段都能唯一映射到 b1 的一个列区间。"""
        cols = cls._columns_from(b1["lines"])
        if len(cols) < TABLE_LIKE_MIN_SEGS:
            return False
        mapped: Set[int] = set()
        for ln in b2["lines"]:
            j = cls._assign_col(ln, cols)
            if j is None or j in mapped:
                return False
            mapped.add(j)
        return True

    def _detect_regions(self, bands: List[dict]) -> List[List[int]]:
        """扫描 band 序列，返回每个表格 region 的 band 下标列表。

        锚定：两个列结构一致的 table-like band（中间允许夹 ≤2 片段的续行）。
        扩展：后续 band 为列结构兼容的表格行，或近距离续行；遇到整行正文
        （单片段过宽）、无法分列的 band 或距离过远的续行即终止。
        """
        med_h = self._median_height(bands)
        n = len(bands)
        regions: List[List[int]] = []
        i = 0
        while i < n - 1:
            if not self._is_table_like(bands[i], med_h):
                i += 1
                continue
            # 向后找锚定配对（跳过 ≤2 片段续行）
            j, gap = i + 1, 0.0
            while j < n:
                gap = bands[j]["y0"] - bands[j - 1]["y1"]
                if self._is_table_like(bands[j], med_h):
                    break
                if len(bands[j]["lines"]) > CONT_MAX_SEGS or gap > CONT_GAP_FACTOR * med_h:
                    j = -1
                    break
                j += 1
            if j == -1 or j >= n or not self._aligned(bands[i], bands[j]):
                i += 1
                continue
            region = list(range(i, j + 1))
            # 自锚点向下扩展
            k = j + 1
            while k < n:
                band, gap = bands[k], bands[k]["y0"] - bands[k - 1]["y1"]
                segs = band["lines"]
                width = segs[0].bbox[2] - segs[0].bbox[0] if len(segs) == 1 else 0.0
                span = max(ln.bbox[2] for bi in region for ln in bands[bi]["lines"]) - min(
                    ln.bbox[0] for bi in region for ln in bands[bi]["lines"]
                )
                if len(segs) == 1 and width > TERM_WIDTH_RATIO * span:
                    break  # 整行正文，region 结束
                if self._is_table_like(band, med_h) and self._aligned(bands[j], band):
                    region.append(k)
                    j = k
                elif len(segs) <= CONT_MAX_SEGS and gap <= CONT_GAP_FACTOR * med_h:
                    region.append(k)  # 续行/合计行：列归属在 _build_rows 决定
                else:
                    break
                k += 1
            if len(region) >= 2:
                regions.append(region)
                i = k
            else:
                i += 1
        return regions

    def _build_rows(
        self, bands: List[dict], region: List[int]
    ) -> Tuple[List[List[str]], List[LineInfo]]:
        """region 内 band → (单元格二维列表, 已入表的行)。

        全局列模型取自 region 内所有 table-like band 的片段区间；分不到列的
        片段不消费（如表格左侧竖排「合同主要内容」标签，留在正文流）。
        续行（≤2 片段）与上一行垂直交叠（间距 ≤0）时并入上一行（换行单元格，
        如「型号规」+「格」），否则独立成行（合计/备注行）。
        """
        tl_bands = [b for b in (bands[i] for i in region) if len(b["lines"]) >= TABLE_LIKE_MIN_SEGS]
        # 列模型只取锚定前两行（表头+首数据行）：合计行的大跨度单元格
        # （如「陆拾陆万…」横跨数列）若参与建模会把相邻列并成一列
        cols = self._columns_from([ln for b in tl_bands[:2] for ln in b["lines"]])

        merged_rows: List[dict] = []
        used: List[LineInfo] = []
        for i in region:
            band = bands[i]
            cells: Dict[int, str] = {}
            for ln in band["lines"]:
                j = self._assign_col(ln, cols)
                if j is None:
                    continue
                cells[j] = (cells.get(j, "") + ln.text).strip()
                used.append(ln)
            if not cells:
                continue
            if len(band["lines"]) <= CONT_MAX_SEGS and merged_rows:
                prev = merged_rows[-1]
                if band["y0"] - prev["band"]["y1"] <= 0:  # 垂直交叠 → 换行单元格
                    for j, t in cells.items():
                        prev["cells"][j] = (prev["cells"].get(j, "") + t).strip()
                    prev["band"]["y1"] = max(prev["band"]["y1"], band["y1"])
                    continue
            merged_rows.append({"cells": cells, "band": band})
        if not merged_rows:
            return [], used
        n_cols = max(max(r["cells"]) for r in merged_rows) + 1
        return [[r["cells"].get(j, "") for j in range(n_cols)] for r in merged_rows], used

    # ---------- 公共工具 ----------

    @staticmethod
    def _drop_lines(pc: PageContent, keys: Set[int]) -> None:
        """从 PageContent 中剔除已入表的行（按 id(LineInfo) 匹配）并重建 text。"""
        if not keys:
            return
        kept = [ln for ln in pc.lines if id(ln) not in keys]
        if len(kept) != len(pc.lines):
            pc.lines = kept
            pc.text = "\n".join(ln.text for ln in kept)


TABLES_FILE = "tables.pkl"


def save_tables(records: List[dict], store_dir: str) -> str:
    """表格旁路存储落盘（无论是否为空都覆盖，避免旧文件误导）。"""
    import pickle

    os.makedirs(store_dir, exist_ok=True)
    path = os.path.join(store_dir, TABLES_FILE)
    with open(path, "wb") as f:
        pickle.dump(records, f)
    return path


def load_tables(store_dir: str) -> List[dict]:
    """读取 tables.pkl（不存在返回空列表），供未来结构化查询/工具调用使用。"""
    import pickle

    path = os.path.join(store_dir, TABLES_FILE)
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return pickle.load(f)
