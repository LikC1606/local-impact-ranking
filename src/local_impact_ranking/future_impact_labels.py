from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from local_impact_ranking.csv_safety import sanitize_csv_row
from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.models import Paper
from local_impact_ranking.normalize import normalize_paper


LABEL_FIELDNAMES = [
    "paper_id",
    "future_impact",
    "field",
    "year",
    "title",
]


def derive_future_impact_labels(
    input_path: str | Path,
    *,
    start_year: int,
    end_year: int,
    label_column: str = "future_impact",
    normalization: str = "raw",
) -> dict[str, Any]:
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")
    normalization = _normalize_mode(normalization)
    papers = [normalize_paper(paper) for paper in load_papers_jsonl(input_path)]
    labels = []
    skipped = []
    for paper in papers:
        yearly_counts = _yearly_citation_counts(paper)
        if not yearly_counts:
            skipped.append(
                {
                    "paper_id": paper.paper_id,
                    "reason": "missing_yearly_citation_history",
                }
            )
            continue
        window_counts = {
            year: count
            for year, count in yearly_counts.items()
            if start_year <= year <= end_year
        }
        if not window_counts:
            skipped.append(
                {
                    "paper_id": paper.paper_id,
                    "reason": "no_citations_in_requested_window",
                }
            )
            continue
        labels.append(
            {
                "paper_id": paper.paper_id,
                label_column: float(sum(window_counts.values())),
                f"{label_column}_raw": float(sum(window_counts.values())),
                "field": paper.field or "unknown",
                "year": paper.year,
                "title": paper.title,
            }
        )
    _apply_label_normalization(labels, label_column=label_column, mode=normalization)
    return {
        "summary": {
            "paper_count": len(papers),
            "label_count": len(labels),
            "skipped_count": len(skipped),
            "start_year": start_year,
            "end_year": end_year,
            "label_column": label_column,
            "normalization": normalization,
            "label_definition": _label_definition(
                start_year,
                end_year,
                normalization=normalization,
            ),
        },
        "labels": labels,
        "skipped": skipped[:100],
    }


def write_future_impact_labels(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    start_year: int,
    end_year: int,
    label_column: str = "future_impact",
    normalization: str = "raw",
) -> dict[str, Any]:
    report = derive_future_impact_labels(
        input_path,
        start_year=start_year,
        end_year=end_year,
        label_column=label_column,
        normalization=normalization,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        destination / "future_impact_labels.jsonl",
        report["labels"],
    )
    _write_csv(
        destination / "future_impact_labels.csv",
        report["labels"],
        label_column=label_column,
    )
    (destination / "future_impact_labels_manifest.json").write_text(
        json.dumps(
            {
                "summary": report["summary"],
                "skipped": report["skipped"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _yearly_citation_counts(paper: Paper) -> dict[int, float]:
    sources = [
        _paper_extra(paper, "citation_counts_by_year"),
        _paper_extra(paper, "citations_by_year"),
        _paper_extra(paper, "yearly_citation_counts"),
        _paper_extra(paper, "citation_history"),
        paper.metadata.get("citation_counts_by_year"),
        paper.metadata.get("citations_by_year"),
        paper.metadata.get("yearly_citation_counts"),
        paper.metadata.get("citation_history"),
    ]
    for source in sources:
        counts = _parse_yearly_counts(source)
        if counts:
            return counts
    return {}


def _parse_yearly_counts(value: Any) -> dict[int, float]:
    if isinstance(value, dict):
        counts = {}
        for raw_year, raw_count in value.items():
            year = _int_or_none(raw_year)
            count = _float_or_none(raw_count)
            if year is not None and count is not None:
                counts[year] = count
        return counts
    if isinstance(value, list):
        counts = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            year = _int_or_none(
                item.get("year")
                or item.get("publication_year")
                or item.get("cited_year")
            )
            count = _float_or_none(
                item.get("count")
                or item.get("citation_count")
                or item.get("citations")
                or item.get("cited_by_count")
            )
            if year is not None and count is not None:
                counts[year] = count
        return counts
    return {}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, label_column: str) -> None:
    fieldnames = [
        "paper_id",
        label_column,
        f"{label_column}_raw",
        "field",
        "year",
        "title",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                sanitize_csv_row({field: row.get(field) for field in fieldnames})
            )


def _apply_label_normalization(
    labels: list[dict[str, Any]],
    *,
    label_column: str,
    mode: str,
) -> None:
    raw_column = f"{label_column}_raw"
    if mode == "raw":
        return
    if mode == "log1p":
        for row in labels:
            row[label_column] = math.log1p(float(row.get(raw_column, 0.0) or 0.0))
        return
    if mode == "field_year_percentile":
        groups: dict[tuple[str, Any], list[dict[str, Any]]] = {}
        for row in labels:
            key = (str(row.get("field") or "unknown"), row.get("year"))
            groups.setdefault(key, []).append(row)
        for group_rows in groups.values():
            group_rows.sort(key=lambda row: float(row.get(raw_column, 0.0) or 0.0))
            if len(group_rows) == 1:
                group_rows[0][label_column] = 1.0
                continue
            for index, row in enumerate(group_rows):
                row[label_column] = index / (len(group_rows) - 1)
        return
    raise ValueError(f"Unsupported normalization: {mode}")


def _normalize_mode(value: str) -> str:
    mode = value.replace("-", "_").strip().lower()
    if mode not in {"raw", "log1p", "field_year_percentile"}:
        raise ValueError(
            "normalization must be one of raw, log1p, or field_year_percentile"
        )
    return mode


def _label_definition(
    start_year: int,
    end_year: int,
    *,
    normalization: str,
) -> str:
    base = f"future citation counts from {start_year} through {end_year}"
    if normalization == "raw":
        return f"sum of yearly citation counts from {start_year} through {end_year}"
    if normalization == "log1p":
        return f"log1p sum of yearly citation counts from {start_year} through {end_year}"
    if normalization == "field_year_percentile":
        return f"field/year percentile of {base}"
    return base


def _paper_extra(paper: Paper, key: str) -> Any:
    extra = getattr(paper, "__pydantic_extra__", None) or {}
    return extra.get(key)


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(number, 0.0)
