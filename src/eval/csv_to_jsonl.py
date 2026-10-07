"""把法律人员填写的标注 CSV 转成 runner 可读的 golden_qa.jsonl。

用法（项目根目录）：
    python -m src.eval.csv_to_jsonl src/eval/golden_qa_template.csv -o src/eval/golden_qa.jsonl

CSV 格式见 golden_qa_template.csv 头部说明。本脚本：
  - 跳过 # 开头的注释行和空行
  - 解析 must_cite 的「条款号@页码」格式（多个用 | 分隔）
  - 解析 expected_answer_points（用；分隔）
  - 仅 breach 类且 breach_established/breaching_party 都填才构造 expected_verdict
  - 输出 utf-8 jsonl，每行一个 GoldenCase JSON
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from typing import Optional


def _parse_must_cite(raw: str) -> list[dict]:
    """解析「条款号@页码|条款号@页码」为 must_cite spec 列表。

    页码可省略或写范围(5-6)；范围取起始页作为 page_start。
    """
    if not raw:
        return []
    specs: list[dict] = []
    for part in raw.split("|"):
        part = part.strip()
        if not part:
            continue
        # 拆「条款号@页码」
        if "@" in part:
            label, page = part.rsplit("@", 1)
            label = label.strip()
            page = page.strip()
        else:
            label, page = part, ""
        spec: dict = {"clause_label": label}
        # 页码：单数字或范围 5-6，取起始页
        if page:
            m = re.match(r"(\d+)", page)
            if m:
                spec["page_start"] = int(m.group(1))
        specs.append(spec)
    return specs


def _parse_list(raw: str, sep: str = "；") -> list[str]:
    """用分隔符拆字符串为列表（中文分号或英文分号兼容）。"""
    if not raw:
        return []
    # 兼容中英文分号
    raw = raw.replace(";", "；")
    return [s.strip() for s in raw.split(sep) if s.strip()]


def _parse_bool(raw: str) -> Optional[bool]:
    raw = raw.strip().lower()
    if raw in ("true", "是", "1", "成立"):
        return True
    if raw in ("false", "否", "0", "不成立"):
        return False
    if raw in ("null", "", "无法判定", "不确定"):
        return None
    return None


def csv_to_jsonl(csv_path: str) -> list[dict]:
    """读标注 CSV 返回 GoldenCase dict 列表。"""
    cases: list[dict] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        # 跳过 # 开头注释行（csv reader 需要先过滤）
        lines = [ln for ln in f if not ln.lstrip().startswith("#") and ln.strip()]
    reader = csv.DictReader(lines)
    if not reader.fieldnames:
        raise ValueError("CSV 无表头，请参照 golden_qa_template.csv 填写")
    for i, row in enumerate(reader, 2):
        cid = (row.get("id") or "").strip()
        query = (row.get("query") or "").strip()
        task_type = (row.get("task_type") or "").strip()
        if not cid or not query or not task_type:
            print(f"[csv_to_jsonl] 第{i}行缺必填字段(id/query/task_type)，跳过", file=sys.stderr)
            continue
        if task_type not in ("simple", "breach", "complex"):
            print(f"[csv_to_jsonl] 第{i}行 task_type={task_type} 非法，跳过", file=sys.stderr)
            continue
        doc_scope = [s.strip() for s in (row.get("doc_scope") or "").split("|") if s.strip()]
        must_cite = _parse_must_cite(row.get("must_cite") or "")
        answer_points = _parse_list(row.get("expected_answer_points") or "")
        case: dict = {
            "id": cid,
            "query": query,
            "task_type": task_type,
            "doc_scope": doc_scope,
            "must_cite": must_cite,
            "expected_answer_points": answer_points,
            "note": (row.get("note") or "").strip(),
        }
        # breach 类构造 expected_verdict
        if task_type == "breach":
            be = _parse_bool(row.get("expected_verdict_breach_established") or "")
            party = (row.get("expected_verdict_breaching_party") or "").strip()
            if be is not None or party:
                case["expected_verdict"] = {
                    "breach_established": be,
                    "breaching_party": party or None,
                }
        cases.append(case)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description="标注 CSV → golden_qa.jsonl 转换")
    ap.add_argument("csv_path", help="标注 CSV 路径（如 src/eval/golden_qa_template.csv）")
    ap.add_argument("-o", "--output", default="src/eval/golden_qa.jsonl", help="输出 jsonl 路径")
    args = ap.parse_args()

    cases = csv_to_jsonl(args.csv_path)
    with open(args.output, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[csv_to_jsonl] 转换完成：{len(cases)} 条 case → {args.output}")
    # 校验摘要
    by_type = {}
    for c in cases:
        by_type[c["task_type"]] = by_type.get(c["task_type"], 0) + 1
    print(f"[csv_to_jsonl] 类型分布：{by_type}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
