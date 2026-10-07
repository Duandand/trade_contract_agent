"""条款感知分块器。

策略：
1. 把所有页的行按顺序串成 (page_no, text) 序列。
2. 用「第X条」标记切分条款组；首部到第一条之前归为「首部」。
3. 每个条款组若文本 ≤ max_len 则整块作为一个 chunk；否则按行滑窗切子块，保留 overlap。
4. 每个 chunk 带定位元数据（页码范围、条款标签、char 长度），供引用溯源使用。

这样单点查询与条款级检索都能定位到合同具体段落。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from typing import List, Tuple

from .pdf_parser import PageContent
from .table_extractor import TableRecord

# 匹配条款标题：① 第X条 ② 编号小节（如 1.2 / 1.2.4，至少含一个点）
# 用捕获组提取标题 token 作为 clause_label，供引用溯源定位。
CLAUSE_RE = re.compile(
    r"^(第[一二三四五六七八九十百零〇两\d]+条|\d{1,2}(?:\.\d{1,2}){1,2})(?![\.\d])"
)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    file_name: str
    page_start: int
    page_end: int
    clause_label: str  # 如 "第五条"；首部为 "首部"
    text: str
    char_len: int

    def to_dict(self) -> dict:
        return asdict(self)


class ClauseChunker:
    def __init__(self, max_len: int = 600, overlap_lines: int = 2) -> None:
        self.max_len = max_len
        self.overlap_lines = overlap_lines

    def chunk(self, pages: List[PageContent], doc_id: str, file_name: str) -> List[Chunk]:
        seq: List[Tuple[int, str]] = []  # (page_no, line_text)
        for p in pages:
            for ln in p.lines:
                if ln.text.strip():
                    seq.append((p.page_no, ln.text.strip()))

        groups: List[Tuple[str, List[Tuple[int, str]]]] = self._group_by_clause(seq)
        chunks: List[Chunk] = []
        idx = 0
        for label, group in groups:
            for sub_lines in self._split_group(group):
                text = "\n".join(t for _, t in sub_lines)
                if not text.strip():
                    continue
                pnos = [pn for pn, _ in sub_lines]
                chunks.append(
                    Chunk(
                        chunk_id=self._make_id(doc_id, idx),
                        doc_id=doc_id,
                        file_name=file_name,
                        page_start=min(pnos),
                        page_end=max(pnos),
                        clause_label=label,
                        text=text,
                        char_len=len(text),
                    )
                )
                idx += 1
        return chunks

    def _group_by_clause(self, seq: List[Tuple[int, str]]) -> List[Tuple[str, List[Tuple[int, str]]]]:
        groups: List[Tuple[str, List]] = []
        current_label, current = "首部", []
        for item in seq:
            _, text = item
            m = CLAUSE_RE.match(text)
            if m:
                if current:
                    groups.append((current_label, current))
                current_label = m.group(1)
                current = [item]
            else:
                current.append(item)
        if current:
            groups.append((current_label, current))
        return groups

    def _split_group(self, group: List[Tuple[int, str]]):
        """按 max_len 切子块；超出时按行滑窗，保留 overlap_lines 行重叠。"""
        full = "\n".join(t for _, t in group)
        if len(full) <= self.max_len:
            return [group] if group else []

        sub_groups: List[List[Tuple[int, str]]] = []
        cur: List[Tuple[int, str]] = []
        cur_len = 0
        for item in group:
            ln_len = len(item[1]) + 1
            if cur and cur_len + ln_len > self.max_len:
                sub_groups.append(cur)
                # 保留尾部 overlap 行
                tail = cur[-self.overlap_lines :] if self.overlap_lines > 0 else []
                cur = list(tail)
                cur_len = sum(len(t[1]) + 1 for t in cur)
            cur.append(item)
            cur_len += ln_len
        if cur:
            sub_groups.append(cur)
        return sub_groups

    def build_table_chunks(
        self, tables: List[TableRecord], doc_id: str, start_idx: int
    ) -> List[Chunk]:
        """表格 → 独立 chunk（不参与条款分组与滑窗，保证表格结构完整）。

        clause_label 形如「表格p2#1」，引用展示为《合同》表格p2#1 (p2)。
        start_idx 延续该文档条款 chunk 的序号，保证 chunk_id 稳定且不冲突。
        表格即使超过 max_len 也整块保留（拆分会破坏表格语义，BGE-M3 支持 8k token）。
        """
        chunks: List[Chunk] = []
        for offset, t in enumerate(tables):
            text = f"【表格】{t.markdown}"
            chunks.append(
                Chunk(
                    chunk_id=self._make_id(doc_id, start_idx + offset),
                    doc_id=doc_id,
                    file_name=t.file_name,
                    page_start=t.page_no,
                    page_end=t.page_no,
                    clause_label=f"表格p{t.page_no}#{t.table_index}",
                    text=text,
                    char_len=len(text),
                )
            )
        return chunks

    @staticmethod
    def _make_id(doc_id: str, idx: int) -> str:
        raw = f"{doc_id}#{idx}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
