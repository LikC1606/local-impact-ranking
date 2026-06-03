from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from local_impact_ranking.csv_safety import sanitize_csv_row
from local_impact_ranking.impact_fusion import true_citation_evidence
from local_impact_ranking.models import Paper
from local_impact_ranking.openalex_enrichment import paper_doi


QUERY_ID_KEYS = (
    "doi",
    "openalex",
    "semantic_scholar",
    "pmid",
    "pmcid",
)


CITATION_COMPLETENESS_FIELDS = [
    "paper_id",
    "title",
    "has_doi",
    "has_openalex_id",
    "has_semantic_scholar_id",
    "has_pmid",
    "has_pmcid",
    "query_identifier_count",
    "missing_query_ids",
    "preferred_count",
    "preferred_source",
    "source_count",
    "positive_source_count",
    "min_source_count",
    "max_source_count",
    "source_spread",
    "source_spread_ratio",
    "source_agreement",
    "source_conflict_flag",
    "source_counts",
    "zero_citation_status",
    "can_treat_zero_as_confirmed",
    "completeness_score",
    "coverage_confidence",
    "is_life_science_record",
    "recommended_next_source_actions",
    "completeness_grade",
    "completeness_reason",
    "recommended_action",
]


def citation_completeness_audit_report(papers: list[Paper]) -> dict[str, Any]:
    rows = [citation_completeness_paper_row(paper) for paper in papers]
    return {
        "summary": _summary(rows),
        "papers": rows,
        "claim_boundary": (
            "Multi-source citation evidence reduces missed-citation risk but does not "
            "prove exhaustive citation coverage. Treat missing and single-source counts "
            "as conservative evidence with disclosure."
        ),
        "method": {
            "citation_count_selection": (
                "Use the maximum compatible true-citation count across available "
                "sources, never the sum, because sources can record the same citing "
                "article."
            ),
            "zero_rule": (
                "A zero count is treated as confirmed only when at least two independent "
                "numeric sources report zero. Single-source zero and no-source cases are "
                "kept as missing-risk evidence."
            ),
            "risk_grades": [
                "no_query_id",
                "not_found_in_sources",
                "single_source_zero",
                "single_source_only",
                "multi_source_zero",
                "multi_source_compatible",
                "source_conflict",
                "external_identity_mismatch",
            ],
        },
    }


def citation_completeness_paper_row(paper: Paper) -> dict[str, Any]:
    identifiers = _identifier_status(paper)
    evidence = true_citation_evidence(paper)
    source_counts = dict(evidence.get("per_source") or {})
    source_count = int(evidence.get("source_count") or 0)
    preferred_count = evidence.get("preferred_count")
    grade = _completeness_grade(identifiers, evidence)
    score = _completeness_score(grade)
    is_life_science = _is_life_science_record(paper, identifiers)
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "has_doi": identifiers["doi"],
        "has_openalex_id": identifiers["openalex"],
        "has_semantic_scholar_id": identifiers["semantic_scholar"],
        "has_pmid": identifiers["pmid"],
        "has_pmcid": identifiers["pmcid"],
        "query_identifier_count": sum(1 for present in identifiers.values() if present),
        "missing_query_ids": [
            key for key in QUERY_ID_KEYS if not identifiers.get(key, False)
        ],
        "preferred_count": preferred_count,
        "preferred_source": evidence.get("preferred_source", "missing"),
        "source_count": source_count,
        "positive_source_count": int(evidence.get("positive_source_count") or 0),
        "min_source_count": evidence.get("min_count"),
        "max_source_count": evidence.get("max_count"),
        "source_spread": evidence.get("source_spread"),
        "source_spread_ratio": evidence.get("source_spread_ratio"),
        "source_agreement": evidence.get("source_agreement", "missing"),
        "source_conflict_flag": bool(evidence.get("source_conflict_flag", False)),
        "source_counts": source_counts,
        "zero_citation_status": _zero_citation_status(evidence),
        "can_treat_zero_as_confirmed": _can_treat_zero_as_confirmed(evidence),
        "completeness_score": score,
        "coverage_confidence": _coverage_confidence(score),
        "is_life_science_record": is_life_science,
        "recommended_next_source_actions": _recommended_next_source_actions(
            grade,
            is_life_science=is_life_science,
        ),
        "completeness_grade": grade,
        "completeness_reason": _completeness_reason(grade),
        "recommended_action": _recommended_action(grade),
    }


def write_citation_completeness_audit(
    papers: list[Paper],
    output_dir: str | Path,
) -> dict[str, Any]:
    report = citation_completeness_audit_report(papers)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "citation_completeness_audit.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "citation_completeness_audit.md").write_text(
        render_citation_completeness_audit_markdown(report),
        encoding="utf-8",
    )
    write_citation_completeness_papers_csv(
        report["papers"],
        destination / "citation_completeness_papers.csv",
    )
    return report


def write_citation_completeness_papers_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CITATION_COMPLETENESS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                sanitize_csv_row(
                    {
                        field: _csv_value(row.get(field))
                        for field in CITATION_COMPLETENESS_FIELDS
                    }
                )
            )


def render_citation_completeness_audit_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Citation Completeness Audit",
        "",
        "This report checks whether citation counts are supported by enough external evidence. It does not prove exhaustive citation coverage.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "paper_count",
        "citation_evidence_coverage",
        "multi_source_coverage",
        "positive_citation_evidence_coverage",
        "mean_completeness_score",
        "papers_without_query_ids",
        "papers_without_numeric_citation_evidence",
        "papers_with_single_source_only",
        "papers_with_source_conflict",
        "papers_with_confirmed_multi_source_zero",
    ):
        lines.append(f"| {key} | {_format_value(summary.get(key))} |")

    grade_counts = summary.get("completeness_grade_counts", {})
    if isinstance(grade_counts, dict) and grade_counts:
        lines.extend(
            [
                "",
                "## Risk Grades",
                "",
                "| Grade | Paper count |",
                "| --- | ---: |",
            ]
        )
        for grade, count in sorted(grade_counts.items()):
            lines.append(f"| {grade} | {count} |")

    confidence_counts = summary.get("coverage_confidence_counts", {})
    if isinstance(confidence_counts, dict) and confidence_counts:
        lines.extend(
            [
                "",
                "## Coverage Confidence",
                "",
                "| Confidence | Paper count |",
                "| --- | ---: |",
            ]
        )
        for confidence, count in sorted(confidence_counts.items()):
            lines.append(f"| {confidence} | {count} |")

    action_counts = summary.get("recommended_next_source_action_counts", {})
    if isinstance(action_counts, dict) and action_counts:
        lines.extend(
            [
                "",
                "## Recommended Source Actions",
                "",
                "| Action | Paper count |",
                "| --- | ---: |",
            ]
        )
        for action, count in sorted(action_counts.items()):
            lines.append(f"| {action} | {count} |")

    source_coverage = summary.get("source_coverage", {})
    if isinstance(source_coverage, dict) and source_coverage:
        lines.extend(
            [
                "",
                "## Source Evidence",
                "",
                "| Source | Paper count | Paper coverage |",
                "| --- | ---: | ---: |",
            ]
        )
        for source, values in sorted(source_coverage.items()):
            if not isinstance(values, dict):
                continue
            lines.append(
                "| "
                f"{source} | "
                f"{_format_value(values.get('paper_count'))} | "
                f"{_format_value(values.get('paper_coverage'))} |"
            )

    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Do not sum citation counts across sources; use the maximum supported source count to avoid duplicate citing articles.",
            "- Treat multi-source zero as a stronger zero signal than single-source zero or missing evidence.",
            "- Use source conflicts and missing identifiers as disclosure flags before reporting impact scores.",
            "",
            f"Claim boundary: {report.get('claim_boundary', '')}",
            "",
        ]
    )
    return "\n".join(lines)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paper_count = len(rows)
    grade_counts = Counter(str(row.get("completeness_grade")) for row in rows)
    confidence_counts = Counter(str(row.get("coverage_confidence")) for row in rows)
    action_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for row in rows:
        per_source = row.get("source_counts")
        if not isinstance(per_source, dict):
            continue
        for source in per_source:
            source_counts[source] += 1
        actions = row.get("recommended_next_source_actions")
        if isinstance(actions, list):
            for action in actions:
                action_counts[str(action)] += 1
    return {
        "paper_count": paper_count,
        "papers_with_numeric_citation_evidence": sum(
            1 for row in rows if int(row.get("source_count") or 0) > 0
        ),
        "papers_without_numeric_citation_evidence": sum(
            1 for row in rows if int(row.get("source_count") or 0) == 0
        ),
        "papers_with_multi_source_evidence": sum(
            1 for row in rows if int(row.get("source_count") or 0) >= 2
        ),
        "papers_with_positive_citation_evidence": sum(
            1 for row in rows if int(row.get("positive_source_count") or 0) > 0
        ),
        "papers_without_query_ids": sum(
            1 for row in rows if int(row.get("query_identifier_count") or 0) == 0
        ),
        "papers_with_single_source_only": grade_counts.get("single_source_only", 0),
        "papers_with_source_conflict": grade_counts.get("source_conflict", 0),
        "papers_with_confirmed_multi_source_zero": grade_counts.get(
            "multi_source_zero",
            0,
        ),
        "life_science_record_count": sum(
            1 for row in rows if bool(row.get("is_life_science_record"))
        ),
        "mean_completeness_score": _mean(
            [
                float(row.get("completeness_score") or 0.0)
                for row in rows
            ]
        ),
        "citation_evidence_coverage": _ratio(
            sum(1 for row in rows if int(row.get("source_count") or 0) > 0),
            paper_count,
        ),
        "multi_source_coverage": _ratio(
            sum(1 for row in rows if int(row.get("source_count") or 0) >= 2),
            paper_count,
        ),
        "positive_citation_evidence_coverage": _ratio(
            sum(1 for row in rows if int(row.get("positive_source_count") or 0) > 0),
            paper_count,
        ),
        "query_identifier_coverage": _ratio(
            sum(1 for row in rows if int(row.get("query_identifier_count") or 0) > 0),
            paper_count,
        ),
        "completeness_grade_counts": dict(sorted(grade_counts.items())),
        "coverage_confidence_counts": dict(sorted(confidence_counts.items())),
        "recommended_next_source_action_counts": dict(sorted(action_counts.items())),
        "source_coverage": {
            source: {
                "paper_count": count,
                "paper_coverage": _ratio(count, paper_count),
            }
            for source, count in sorted(source_counts.items())
        },
    }


def _identifier_status(paper: Paper) -> dict[str, bool]:
    return {
        "doi": paper_doi(paper) is not None,
        "openalex": _has_identifier(paper, "openalex", "openalex_id"),
        "semantic_scholar": _has_identifier(
            paper,
            "semantic_scholar",
            "semantic_scholar_id",
            "s2",
            "s2_id",
            "s2_paper_id",
            "paperId",
        ),
        "pmid": _has_identifier(paper, "pmid", "pubmed", "pubmed_id"),
        "pmcid": _has_identifier(paper, "pmcid", "pmc", "pmc_id"),
    }


def _has_identifier(paper: Paper, *keys: str) -> bool:
    metadata = paper.metadata if isinstance(paper.metadata, dict) else {}
    external_ids = metadata.get("external_ids")
    if isinstance(external_ids, dict):
        for key in keys:
            if external_ids.get(key):
                return True
    extra = getattr(paper, "__pydantic_extra__", None) or {}
    for key in keys:
        if extra.get(key):
            return True
        if metadata.get(key):
            return True
    return False


def _is_life_science_record(paper: Paper, identifiers: dict[str, bool]) -> bool:
    if identifiers.get("pmid") or identifiers.get("pmcid"):
        return True
    field = str(paper.field or "").lower()
    venue = str(paper.venue or "").lower()
    haystack = f"{field} {venue}"
    return any(
        token in haystack
        for token in (
            "life science",
            "biomed",
            "biolog",
            "medicine",
            "medical",
            "clinical",
            "health",
            "pubmed",
        )
    )


def _completeness_grade(
    identifiers: dict[str, bool],
    evidence: dict[str, Any],
) -> str:
    if evidence.get("preferred_source") == "external_identity_mismatch":
        return "external_identity_mismatch"
    source_count = int(evidence.get("source_count") or 0)
    max_count = evidence.get("max_count")
    if source_count == 0:
        if not any(identifiers.values()):
            return "no_query_id"
        return "not_found_in_sources"
    if source_count == 1:
        if max_count == 0.0:
            return "single_source_zero"
        return "single_source_only"
    if max_count == 0.0:
        return "multi_source_zero"
    if evidence.get("source_conflict_flag"):
        return "source_conflict"
    return "multi_source_compatible"


def _zero_citation_status(evidence: dict[str, Any]) -> str:
    source_count = int(evidence.get("source_count") or 0)
    max_count = evidence.get("max_count")
    if source_count == 0:
        return "unknown_no_source"
    if max_count != 0.0:
        return "not_zero"
    if source_count >= 2:
        return "confirmed_multi_source_zero"
    return "weak_single_source_zero"


def _can_treat_zero_as_confirmed(evidence: dict[str, Any]) -> bool:
    return int(evidence.get("source_count") or 0) >= 2 and evidence.get("max_count") == 0.0


def _completeness_score(grade: str) -> float:
    return {
        "external_identity_mismatch": 0.0,
        "no_query_id": 0.05,
        "not_found_in_sources": 0.15,
        "single_source_zero": 0.25,
        "single_source_only": 0.45,
        "source_conflict": 0.55,
        "multi_source_zero": 0.80,
        "multi_source_compatible": 0.90,
    }.get(grade, 0.0)


def _coverage_confidence(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "moderate"
    return "low"


def _recommended_next_source_actions(
    grade: str,
    *,
    is_life_science: bool,
) -> list[str]:
    actions = {
        "external_identity_mismatch": ["fix_external_identity"],
        "no_query_id": ["resolve_query_identifier"],
        "not_found_in_sources": ["retry_multi_source_enrichment"],
        "single_source_zero": ["add_second_independent_source"],
        "single_source_only": ["add_second_independent_source"],
        "source_conflict": [
            "inspect_source_conflict",
            "query_high_recall_sources",
        ],
        "multi_source_zero": [],
        "multi_source_compatible": [],
    }.get(grade, ["review_citation_evidence"])
    if is_life_science and grade not in {
        "multi_source_zero",
        "multi_source_compatible",
    }:
        actions = [*actions, "query_pubmed_union_sources"]
    return actions


def _completeness_reason(grade: str) -> str:
    return {
        "no_query_id": "No DOI/OpenAlex/Semantic Scholar/PMID/PMCID identifier is available for reliable citation lookup.",
        "not_found_in_sources": "The paper has at least one query identifier but no numeric citation evidence was found in the enriched record.",
        "single_source_zero": "Only one numeric source reports zero citations; this is weaker than multi-source zero evidence.",
        "single_source_only": "Only one numeric citation source is available, so missed citations cannot be cross-checked.",
        "multi_source_zero": "At least two numeric sources report zero citations.",
        "multi_source_compatible": "Multiple numeric sources are available and their counts are compatible.",
        "source_conflict": "Multiple numeric sources disagree enough to indicate possible undercoverage or source lag.",
        "external_identity_mismatch": "External source identity did not match the local paper title, so the count is excluded.",
    }.get(grade, "Unknown citation completeness grade.")


def _recommended_action(grade: str) -> str:
    return {
        "no_query_id": "Add DOI, PMID/PMCID, OpenAlex ID, or Semantic Scholar ID before treating citation counts as complete.",
        "not_found_in_sources": "Retry enrichment with OpenAlex, Semantic Scholar, Crossref, Europe PMC/iCite/NCBI, OpenCitations, DataCite, OpenAIRE, and Wikidata where applicable.",
        "single_source_zero": "Check at least one more independent source before reporting this as a true zero.",
        "single_source_only": "Add a second independent citation source before making high-confidence claims.",
        "multi_source_zero": "Report as a supported zero with the source list attached.",
        "multi_source_compatible": "Report with normal source disclosure.",
        "source_conflict": "Use the maximum supported count for scoring, but disclose source spread and inspect high-impact papers manually if possible.",
        "external_identity_mismatch": "Fix identifier matching or remove the mismatched external record.",
    }.get(grade, "Review citation evidence manually.")


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if value is None:
        return ""
    return str(value)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)
