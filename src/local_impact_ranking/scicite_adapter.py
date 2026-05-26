from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any


def convert_scicite_jsonl(input_path: str | Path, output_path: str | Path) -> int:
    papers: OrderedDict[str, dict[str, Any]] = OrderedDict()
    source = Path(input_path)
    destination = Path(output_path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            citing_id = _required(record, "citingPaperId", line_number)
            cited_id = _optional_string(record.get("citedPaperId"))
            if cited_id is None or cited_id.lower() == "none":
                continue
            context = _context_text(record, line_number)
            paper = papers.setdefault(citing_id, _empty_paper(citing_id))
            _add_reference_once(paper, cited_id)
            paper["citation_contexts"].append(
                {
                    "reference_id": cited_id,
                    "sentence": context,
                    "section": _optional_string(record.get("sectionName")),
                    "label": _optional_string(record.get("label")),
                    "source_line": line_number,
                }
            )
            paper["metadata"]["scicite_labels"].append(_optional_string(record.get("label")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for paper in papers.values():
            handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
    return len(papers)


def _empty_paper(citing_id: str) -> dict[str, Any]:
    return {
        "paper_id": citing_id,
        "title": f"SciCite citing paper {citing_id}",
        "references": [],
        "citation_contexts": [],
        "metadata": {
            "source_benchmark": "scicite",
            "scicite_labels": [],
        },
    }


def _add_reference_once(paper: dict[str, Any], cited_id: str) -> None:
    if any(reference["reference_id"] == cited_id for reference in paper["references"]):
        return
    paper["references"].append(
        {
            "reference_id": cited_id,
            "title": f"SciCite cited paper {cited_id}",
            "external_ids": {"scicite": cited_id},
        }
    )


def _required(record: dict[str, Any], key: str, line_number: int) -> str:
    value = record.get(key)
    if value is None or value == "":
        raise ValueError(f"SciCite record on line {line_number} is missing {key}")
    return str(value)


def _context_text(record: dict[str, Any], line_number: int) -> str:
    value = record.get("context", record.get("string"))
    if value is None or value == "":
        raise ValueError(f"SciCite record on line {line_number} is missing context/string")
    return str(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)
