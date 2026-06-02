from __future__ import annotations

from collections import Counter
from typing import Any

from local_impact_ranking.models import Paper


MIN_YEAR = 1800
MAX_YEAR = 2100


def validation_report(papers: list[Paper]) -> dict[str, Any]:
    issues = []
    paper_id_counts = Counter(paper.paper_id for paper in papers)
    for paper_id, count in sorted(paper_id_counts.items()):
        if count > 1:
            issues.append(
                _issue(
                    "error",
                    "duplicate_paper_id",
                    paper_id,
                    f"paper_id appears {count} times",
                )
            )

    for paper in papers:
        if not paper.references:
            issues.append(
                _issue(
                    "warning",
                    "empty_reference_list",
                    paper.paper_id,
                    "paper has no candidate references; evidence coverage may be weaker",
                )
            )
        if not paper.title:
            issues.append(
                _issue("warning", "missing_title", paper.paper_id, "paper title is empty")
            )
        if paper.year is not None and not (MIN_YEAR <= paper.year <= MAX_YEAR):
            issues.append(
                _issue(
                    "error",
                    "invalid_year",
                    paper.paper_id,
                    f"paper year {paper.year} is outside supported range {MIN_YEAR}-{MAX_YEAR}",
                )
            )

        reference_ids = [
            reference.reference_id
            for reference in paper.references
            if reference.reference_id
        ]
        reference_counts = Counter(reference_ids)
        for reference_id, count in sorted(reference_counts.items()):
            if count > 1:
                issues.append(
                    _issue(
                        "error",
                        "duplicate_reference_id",
                        paper.paper_id,
                        f"reference_id {reference_id} appears {count} times in one paper",
                    )
                )

    severity_counts = dict(Counter(issue["severity"] for issue in issues))
    return {
        "record_count": len(papers),
        "summary": {
            "issue_count": len(issues),
            "severity_counts": severity_counts,
            "can_run": severity_counts.get("error", 0) == 0,
        },
        "issues": issues,
    }


def render_validation_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    severity_counts = summary.get("severity_counts", {})
    lines = [
        "# Validation Report",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Record count | {report.get('record_count', 0)} |",
        f"| Issue count | {summary.get('issue_count', 0)} |",
        f"| Errors | {severity_counts.get('error', 0)} |",
        f"| Warnings | {severity_counts.get('warning', 0)} |",
        f"| Can run | {_yes_no(summary.get('can_run'))} |",
        "",
        "## Issues",
        "",
        "| Severity | Code | Paper ID | Message |",
        "| --- | --- | --- | --- |",
    ]
    for issue in report.get("issues", []):
        lines.append(
            f"| {_text(issue.get('severity'))} | {_text(issue.get('code'))} | "
            f"{_text(issue.get('paper_id'))} | {_text(issue.get('message'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _issue(severity: str, code: str, paper_id: str, message: str) -> dict[str, str]:
    return {
        "severity": severity,
        "code": code,
        "paper_id": paper_id,
        "message": message,
    }


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
