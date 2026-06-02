from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.models import Paper
from local_impact_ranking.normalize import normalize_paper


YEARLY_KEYS = (
    "citation_counts_by_year",
    "citations_by_year",
    "yearly_citation_counts",
    "citation_history",
)


def apply_citation_cutoff(
    input_path: str | Path,
    *,
    cutoff_year: int,
) -> dict[str, Any]:
    papers = [normalize_paper(paper) for paper in load_papers_jsonl(input_path)]
    updated_papers = []
    updated_histories = 0
    updated_counts = 0
    for paper in papers:
        updated, history_changed, count_changed = _apply_cutoff_to_paper(
            paper,
            cutoff_year=cutoff_year,
        )
        updated_papers.append(updated)
        updated_histories += int(history_changed)
        updated_counts += int(count_changed)
    return {
        "summary": {
            "paper_count": len(updated_papers),
            "cutoff_year": cutoff_year,
            "updated_citation_history_count": updated_histories,
            "updated_citation_count_count": updated_counts,
        },
        "papers": updated_papers,
    }


def write_citation_cutoff(
    input_path: str | Path,
    output_path: str | Path,
    *,
    cutoff_year: int,
) -> dict[str, Any]:
    report = apply_citation_cutoff(input_path, cutoff_year=cutoff_year)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for paper in report["papers"]:
            handle.write(json.dumps(paper.model_dump(), ensure_ascii=False) + "\n")
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(report["summary"], indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def audit_citation_cutoff(
    input_path: str | Path,
    *,
    cutoff_year: int,
) -> dict[str, Any]:
    papers = [normalize_paper(paper) for paper in load_papers_jsonl(input_path)]
    future_examples = []
    papers_with_history = 0
    papers_with_future_years = 0
    future_year_count = 0
    cutoff_metadata_count = 0
    for paper in papers:
        metadata = paper.metadata
        if metadata.get("citation_cutoff_year") is not None:
            cutoff_metadata_count += 1
        history = _first_yearly_counts(metadata)
        if not history:
            continue
        papers_with_history += 1
        future_years = [year for year in sorted(history) if year > cutoff_year]
        if future_years:
            papers_with_future_years += 1
            future_year_count += len(future_years)
            if len(future_examples) < 20:
                future_examples.append(
                    {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "future_years": future_years,
                    }
                )
    summary = {
        "paper_count": len(papers),
        "cutoff_year": cutoff_year,
        "papers_with_yearly_citation_history": papers_with_history,
        "papers_with_future_years": papers_with_future_years,
        "future_year_count": future_year_count,
        "citation_cutoff_year_coverage": (
            cutoff_metadata_count / len(papers) if papers else 0.0
        ),
    }
    return {
        "summary": summary,
        "decision": _audit_decision(summary),
        "future_year_examples": future_examples,
    }


def write_citation_cutoff_audit(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    cutoff_year: int,
) -> dict[str, Any]:
    report = audit_citation_cutoff(input_path, cutoff_year=cutoff_year)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "citation_cutoff_audit.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "citation_cutoff_audit.md").write_text(
        render_citation_cutoff_audit_markdown(report),
        encoding="utf-8",
    )
    return report


def render_citation_cutoff_audit_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    decision = report["decision"]
    lines = [
        "# Citation Cutoff Audit",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| status | {decision['status']} |",
        f"| reason | {decision['reason']} |",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {_format_value(value)} |")
    lines.extend(
        [
            "",
            "## Future-Year Examples",
            "",
            "| Paper ID | Future years | Title |",
            "| --- | --- | --- |",
        ]
    )
    for row in report.get("future_year_examples", []):
        lines.append(
            f"| {row['paper_id']} | {','.join(str(year) for year in row['future_years'])} | {row['title']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _apply_cutoff_to_paper(
    paper: Paper,
    *,
    cutoff_year: int,
) -> tuple[Paper, bool, bool]:
    record = paper.model_dump()
    metadata = dict(record.get("metadata") or {})
    history_changed = False
    cutoff_total: float | None = None
    for key in YEARLY_KEYS:
        counts = _parse_yearly_counts(metadata.get(key))
        if not counts:
            continue
        filtered = {
            year: count
            for year, count in sorted(counts.items())
            if year <= cutoff_year
        }
        metadata[key] = {str(year): count for year, count in filtered.items()}
        cutoff_total = float(sum(filtered.values()))
        history_changed = True
        break
    metadata["citation_cutoff_year"] = cutoff_year
    count_changed = False
    if cutoff_total is not None:
        metadata["citation_count"] = cutoff_total
        external_impact = dict(metadata.get("external_impact") or {})
        if "openalex_cited_by_count" in external_impact:
            external_impact["openalex_cited_by_count"] = cutoff_total
        if "opencitations_cited_by_count" in external_impact:
            external_impact["opencitations_cited_by_count"] = min(
                float(external_impact["opencitations_cited_by_count"]),
                cutoff_total,
            )
        metadata["external_impact"] = external_impact
        record["citation_count"] = cutoff_total
        count_changed = True
    record["metadata"] = metadata
    return Paper.model_validate(record), history_changed, count_changed


def _first_yearly_counts(metadata: dict[str, Any]) -> dict[int, float]:
    for key in YEARLY_KEYS:
        counts = _parse_yearly_counts(metadata.get(key))
        if counts:
            return counts
    return {}


def _audit_decision(summary: dict[str, Any]) -> dict[str, str]:
    if int(summary["papers_with_future_years"]) > 0:
        return {
            "status": "failed",
            "reason": "Citation histories contain years after the requested cutoff.",
        }
    if float(summary["citation_cutoff_year_coverage"]) < 1.0:
        return {
            "status": "warning",
            "reason": "No future-year citation leakage was found, but not every paper records citation_cutoff_year metadata.",
        }
    return {
        "status": "passed",
        "reason": "No future-year citation leakage was found and cutoff metadata is present.",
    }


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


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
