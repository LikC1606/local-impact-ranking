from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable

from local_impact_ranking.csv_safety import sanitize_csv_row
from local_impact_ranking.models import Paper
from local_impact_ranking.openalex_enrichment import paper_doi


THRESHOLDS = {
    "strong_external_query_id_coverage": 0.80,
    "usable_external_query_id_coverage": 0.50,
    "strong_citation_count_coverage": 0.70,
    "usable_citation_count_coverage": 0.40,
    "strong_field_year_percentile_coverage": 0.60,
    "usable_field_year_percentile_coverage": 0.25,
    "strong_external_source_coverage": 0.60,
    "usable_external_source_coverage": 0.25,
    "strong_graph_edge_count": 1,
}


def external_evidence_readiness_report(papers: list[Paper]) -> dict[str, Any]:
    paper_count = len(papers)
    identifier_coverage = {
        "doi_coverage": _coverage(papers, lambda paper: paper_doi(paper) is not None),
        "openalex_id_coverage": _coverage(papers, _has_openalex_id),
        "external_query_id_coverage": _coverage(
            papers,
            lambda paper: paper_doi(paper) is not None or _has_openalex_id(paper),
        ),
    }
    source_coverage = {
        "openalex_enriched_coverage": _coverage(
            papers,
            lambda paper: bool(paper.metadata.get("openalex_enriched")),
        ),
        "opencitations_enriched_coverage": _coverage(
            papers,
            lambda paper: bool(paper.metadata.get("opencitations_enriched")),
        ),
        "bip_enriched_coverage": _coverage(papers, _has_bip_source),
        "semantic_scholar_enriched_coverage": _coverage(
            papers,
            _has_semantic_scholar_source,
        ),
    }
    signal_coverage = {
        "citation_count_coverage": _coverage(papers, _has_citation_count),
        "yearly_citation_history_coverage": _coverage(papers, _has_yearly_citation_history),
        "field_year_percentile_coverage": _coverage(
            papers,
            lambda paper: _external_number(
                paper,
                "field_year_percentile",
                "field_year_citation_percentile",
                "citation_percentile",
                "openalex_percentile",
            )
            is not None,
        ),
        "bip_impact_evidence_coverage": _coverage(papers, _has_bip_impact_evidence),
        "positive_bip_impact_evidence_coverage": _coverage(
            papers,
            _has_positive_bip_impact_evidence,
        ),
        "influential_citation_evidence_coverage": _coverage(
            papers,
            _has_influential_citation_evidence,
        ),
        "positive_influential_citation_evidence_coverage": _coverage(
            papers,
            _has_positive_influential_citation_evidence,
        ),
        "patent_evidence_coverage": _coverage(
            papers,
            lambda paper: _external_number(
                paper,
                "patent_citations",
                "patent_citation_count",
                "patent_count",
            )
            is not None,
        ),
        "positive_patent_evidence_coverage": _coverage(
            papers,
            lambda paper: _positive_external_number(
                paper,
                "patent_citations",
                "patent_citation_count",
                "patent_count",
            ),
        ),
        "policy_evidence_coverage": _coverage(
            papers,
            lambda paper: _external_number(
                paper,
                "policy_citations",
                "policy_citation_count",
                "policy_mentions",
            )
            is not None,
        ),
        "positive_policy_evidence_coverage": _coverage(
            papers,
            lambda paper: _positive_external_number(
                paper,
                "policy_citations",
                "policy_citation_count",
                "policy_mentions",
            ),
        ),
        "artifact_evidence_coverage": _coverage(papers, _has_artifact_evidence),
        "positive_artifact_evidence_coverage": _coverage(
            papers,
            _has_positive_artifact_evidence,
        ),
        "media_evidence_coverage": _coverage(papers, _has_media_evidence),
        "positive_media_evidence_coverage": _coverage(
            papers,
            _has_positive_media_evidence,
        ),
        "award_evidence_coverage": _coverage(papers, _has_award_evidence),
        "positive_award_evidence_coverage": _coverage(
            papers,
            _has_positive_award_evidence,
        ),
    }
    graph_evidence = _graph_evidence(papers)
    paper_coverage_summary = _paper_coverage_summary(
        external_evidence_paper_coverage_rows(papers)
    )
    decision = _decision(identifier_coverage, source_coverage, signal_coverage, graph_evidence)
    recommendations = _recommendations(
        identifier_coverage,
        source_coverage,
        signal_coverage,
        graph_evidence,
    )
    return {
        "paper_count": paper_count,
        "identifier_coverage": identifier_coverage,
        "source_coverage": source_coverage,
        "signal_coverage": signal_coverage,
        "graph_evidence": graph_evidence,
        "paper_coverage_summary": paper_coverage_summary,
        "decision": decision,
        "recommendations": recommendations,
        "thresholds": THRESHOLDS,
        "claim_boundary": "High-credibility claims require high external identifier and evidence coverage; otherwise report dataset-local exploratory scores with disclosure.",
    }


def render_external_evidence_readiness_markdown(report: dict[str, Any]) -> str:
    decision = report.get("decision", {})
    lines = [
        "# External Evidence Readiness",
        "",
        "## Decision",
        "",
        f"| Status | {_text(decision.get('status'))} |",
        f"| Can claim high credibility | {_yes_no(decision.get('can_claim_high_credibility'))} |",
        f"| Reason | {_text(decision.get('reason'))} |",
        "",
        "## Coverage",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for section in (
        "identifier_coverage",
        "source_coverage",
        "signal_coverage",
        "graph_evidence",
    ):
        values = report.get(section, {})
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            lines.append(f"| {key} | {_format_value(value)} |")
    paper_coverage_summary = report.get("paper_coverage_summary", {})
    if isinstance(paper_coverage_summary, dict) and paper_coverage_summary:
        lines.extend(
            [
                "",
                "## Per-Paper Evidence Gaps",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
            ]
        )
        for key in (
            "paper_count",
            "fully_covered_paper_count",
            "fully_covered_paper_coverage",
            "papers_with_any_missing_evidence",
            "papers_with_any_missing_evidence_coverage",
        ):
            lines.append(f"| {key} | {_format_value(paper_coverage_summary.get(key))} |")
        top_missing = paper_coverage_summary.get("top_missing_evidence", [])
        if isinstance(top_missing, list) and top_missing:
            lines.extend(
                [
                    "",
                    "| Missing evidence | Paper count | Paper coverage |",
                    "| --- | ---: | ---: |",
                ]
            )
            for item in top_missing:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "| "
                    f"{_text(item.get('label'))} | "
                    f"{_format_value(item.get('paper_count'))} | "
                    f"{_format_value(item.get('paper_coverage'))} |"
                )
    lines.extend(["", "## Recommendations", ""])
    recommendations = report.get("recommendations") or []
    if recommendations:
        lines.extend(f"- {recommendation}" for recommendation in recommendations)
    else:
        lines.append("- No external evidence readiness blockers detected.")
    lines.extend(
        [
            "",
            "## Thresholds",
            "",
            "| Threshold | Value |",
            "| --- | ---: |",
        ]
    )
    thresholds = report.get("thresholds", {})
    if isinstance(thresholds, dict):
        for key, value in thresholds.items():
            lines.append(f"| {key} | {_format_value(value)} |")
    lines.append("")
    return "\n".join(lines)


PAPER_COVERAGE_FIELDS = [
    "paper_id",
    "title",
    "has_doi",
    "has_openalex_id",
    "has_openalex_enrichment",
    "has_opencitations_enrichment",
    "has_citation_count",
    "has_yearly_citation_history",
    "has_field_year_percentile",
    "has_bip_impact_evidence",
    "has_positive_bip_impact_evidence",
    "has_influential_citation_evidence",
    "has_positive_influential_citation_evidence",
    "has_patent_evidence",
    "has_positive_patent_evidence",
    "has_policy_evidence",
    "has_positive_policy_evidence",
    "has_artifact_evidence",
    "has_positive_artifact_evidence",
    "has_media_evidence",
    "has_positive_media_evidence",
    "has_award_evidence",
    "has_positive_award_evidence",
    "has_internal_reference_edges",
    "has_opencitations_incoming_edges",
    "materialized_internal_edge_count",
    "missing_evidence",
]


def external_evidence_paper_coverage_rows(papers: list[Paper]) -> list[dict[str, Any]]:
    paper_ids = {paper.paper_id for paper in papers}
    rows = []
    for paper in papers:
        has_doi = paper_doi(paper) is not None
        has_openalex_id = _has_openalex_id(paper)
        has_openalex_enrichment = bool(paper.metadata.get("openalex_enriched"))
        has_opencitations_enrichment = bool(paper.metadata.get("opencitations_enriched"))
        has_citation_count = _has_citation_count(paper)
        has_yearly_history = _has_yearly_citation_history(paper)
        has_field_year_percentile = (
            _external_number(
                paper,
                "field_year_percentile",
                "field_year_citation_percentile",
                "citation_percentile",
                "openalex_percentile",
            )
            is not None
        )
        has_bip_impact = _has_bip_impact_evidence(paper)
        has_positive_bip_impact = _has_positive_bip_impact_evidence(paper)
        has_influential_citation = _has_influential_citation_evidence(paper)
        has_positive_influential_citation = (
            _has_positive_influential_citation_evidence(paper)
        )
        has_patent = (
            _external_number(
                paper,
                "patent_citations",
                "patent_citation_count",
                "patent_count",
            )
            is not None
        )
        has_positive_patent = _positive_external_number(
            paper,
            "patent_citations",
            "patent_citation_count",
            "patent_count",
        )
        has_policy = (
            _external_number(
                paper,
                "policy_citations",
                "policy_citation_count",
                "policy_mentions",
            )
            is not None
        )
        has_positive_policy = _positive_external_number(
            paper,
            "policy_citations",
            "policy_citation_count",
            "policy_mentions",
        )
        has_artifact = _has_artifact_evidence(paper)
        has_positive_artifact = _has_positive_artifact_evidence(paper)
        has_media = _has_media_evidence(paper)
        has_positive_media = _has_positive_media_evidence(paper)
        has_award = _has_award_evidence(paper)
        has_positive_award = _has_positive_award_evidence(paper)
        has_internal_edges = any(
            reference.reference_id in paper_ids for reference in paper.references
        )
        has_opencitations_edges = (
            isinstance(paper.metadata.get("opencitations_incoming_edges"), list)
            and bool(paper.metadata.get("opencitations_incoming_edges"))
        )
        materialized_edge_count = int(
            float(paper.metadata.get("opencitations_materialized_edge_count") or 0)
        )
        missing = _missing_evidence_labels(
            {
                "doi_or_openalex_id": has_doi or has_openalex_id,
                "openalex_enrichment": has_openalex_enrichment,
                "opencitations_enrichment": has_opencitations_enrichment,
                "citation_count": has_citation_count,
                "yearly_citation_history": has_yearly_history,
                "field_year_percentile": has_field_year_percentile,
                "bip_impact": has_bip_impact,
                "influential_citation": has_influential_citation,
                "internal_reference_or_opencitations_edge": (
                    has_internal_edges or has_opencitations_edges
                ),
            }
        )
        rows.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "has_doi": has_doi,
                "has_openalex_id": has_openalex_id,
                "has_openalex_enrichment": has_openalex_enrichment,
                "has_opencitations_enrichment": has_opencitations_enrichment,
                "has_citation_count": has_citation_count,
                "has_yearly_citation_history": has_yearly_history,
                "has_field_year_percentile": has_field_year_percentile,
                "has_bip_impact_evidence": has_bip_impact,
                "has_positive_bip_impact_evidence": has_positive_bip_impact,
                "has_influential_citation_evidence": has_influential_citation,
                "has_positive_influential_citation_evidence": has_positive_influential_citation,
                "has_patent_evidence": has_patent,
                "has_positive_patent_evidence": has_positive_patent,
                "has_policy_evidence": has_policy,
                "has_positive_policy_evidence": has_positive_policy,
                "has_artifact_evidence": has_artifact,
                "has_positive_artifact_evidence": has_positive_artifact,
                "has_media_evidence": has_media,
                "has_positive_media_evidence": has_positive_media,
                "has_award_evidence": has_award,
                "has_positive_award_evidence": has_positive_award,
                "has_internal_reference_edges": has_internal_edges,
                "has_opencitations_incoming_edges": has_opencitations_edges,
                "materialized_internal_edge_count": materialized_edge_count,
                "missing_evidence": ";".join(missing),
            }
        )
    return rows


def write_external_evidence_paper_coverage_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_COVERAGE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                sanitize_csv_row(
                    {
                        field: _csv_value(row.get(field))
                        for field in PAPER_COVERAGE_FIELDS
                    }
                )
            )


def _paper_coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paper_count = len(rows)
    missing_counts: dict[str, int] = {}
    fully_covered_count = 0
    papers_with_missing = 0
    for row in rows:
        labels = [
            label
            for label in str(row.get("missing_evidence") or "").split(";")
            if label
        ]
        if labels:
            papers_with_missing += 1
        else:
            fully_covered_count += 1
        for label in labels:
            missing_counts[label] = missing_counts.get(label, 0) + 1
    missing_rates = {
        label: _ratio(count, paper_count)
        for label, count in sorted(missing_counts.items())
    }
    top_missing = [
        {
            "label": label,
            "paper_count": count,
            "paper_coverage": _ratio(count, paper_count),
        }
        for label, count in sorted(
            missing_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    return {
        "paper_count": paper_count,
        "fully_covered_paper_count": fully_covered_count,
        "fully_covered_paper_coverage": _ratio(fully_covered_count, paper_count),
        "papers_with_any_missing_evidence": papers_with_missing,
        "papers_with_any_missing_evidence_coverage": _ratio(
            papers_with_missing,
            paper_count,
        ),
        "missing_evidence_counts": dict(sorted(missing_counts.items())),
        "missing_evidence_rates": missing_rates,
        "top_missing_evidence": top_missing[:10],
    }


def _missing_evidence_labels(requirements: dict[str, bool]) -> list[str]:
    return [label for label, present in requirements.items() if not present]


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return _yes_no(value)
    return value


def _decision(
    identifier_coverage: dict[str, float],
    source_coverage: dict[str, float],
    signal_coverage: dict[str, float],
    graph_evidence: dict[str, Any],
) -> dict[str, Any]:
    external_query = identifier_coverage["external_query_id_coverage"]
    source = max(
        source_coverage["openalex_enriched_coverage"],
        source_coverage["opencitations_enriched_coverage"],
    )
    citation = signal_coverage["citation_count_coverage"]
    percentile = signal_coverage["field_year_percentile_coverage"]
    graph_edges = int(graph_evidence["internal_reference_edge_count"])
    if (
        external_query >= THRESHOLDS["strong_external_query_id_coverage"]
        and source >= THRESHOLDS["strong_external_source_coverage"]
        and citation >= THRESHOLDS["strong_citation_count_coverage"]
        and percentile >= THRESHOLDS["strong_field_year_percentile_coverage"]
        and graph_edges >= THRESHOLDS["strong_graph_edge_count"]
    ):
        return {
            "status": "ready_for_high_credibility_scoring",
            "can_claim_high_credibility": True,
            "reason": "External identifier, source, citation, percentile, and graph evidence coverage are strong enough for high-credibility dataset-local scoring.",
        }
    if (
        external_query >= THRESHOLDS["usable_external_query_id_coverage"]
        or citation >= THRESHOLDS["usable_citation_count_coverage"]
        or percentile >= THRESHOLDS["usable_field_year_percentile_coverage"]
        or graph_edges >= THRESHOLDS["strong_graph_edge_count"]
    ):
        return {
            "status": "usable_with_disclosure",
            "can_claim_high_credibility": False,
            "reason": "External evidence is usable but incomplete; disclose coverage limits and keep reportability gates visible.",
        }
    return {
        "status": "insufficient_external_evidence",
        "can_claim_high_credibility": False,
        "reason": "External evidence coverage is too sparse for high-credibility influence claims.",
    }


def _recommendations(
    identifier_coverage: dict[str, float],
    source_coverage: dict[str, float],
    signal_coverage: dict[str, float],
    graph_evidence: dict[str, Any],
) -> list[str]:
    recommendations = []
    if (
        identifier_coverage["external_query_id_coverage"]
        < THRESHOLDS["strong_external_query_id_coverage"]
    ):
        recommendations.append("Increase DOI/OpenAlex ID coverage")
    if (
        source_coverage["openalex_enriched_coverage"]
        < THRESHOLDS["strong_external_source_coverage"]
    ):
        recommendations.append("Run enrich-openalex on DOI/OpenAlex-matched records")
    if (
        source_coverage["opencitations_enriched_coverage"]
        < THRESHOLDS["strong_external_source_coverage"]
    ):
        recommendations.append("Run enrich-opencitations with --materialize-internal-edges")
    if (
        signal_coverage["field_year_percentile_coverage"]
        < THRESHOLDS["strong_field_year_percentile_coverage"]
    ):
        recommendations.append("Add field/year percentile evidence for more papers")
    if int(graph_evidence["internal_reference_edge_count"]) < THRESHOLDS[
        "strong_graph_edge_count"
    ]:
        if float(graph_evidence.get("internal_citation_audit_coverage") or 0.0) >= 0.95:
            recommendations.append("Internal citation audit found no dataset-internal edges")
        else:
            recommendations.append("Add or materialize dataset-internal citation edges")
    return recommendations


def _graph_evidence(papers: list[Paper]) -> dict[str, Any]:
    paper_ids = {paper.paper_id for paper in papers}
    internal_reference_edge_count = sum(
        1
        for paper in papers
        for reference in paper.references
        if reference.reference_id in paper_ids
    )
    materialized_internal_edge_count = sum(
        int(float(paper.metadata.get("opencitations_materialized_edge_count") or 0))
        + int(float(paper.metadata.get("internal_citation_materialized_edge_count") or 0))
        for paper in papers
    )
    opencitations_incoming_edge_papers = sum(
        1
        for paper in papers
        if isinstance(paper.metadata.get("opencitations_incoming_edges"), list)
        and bool(paper.metadata.get("opencitations_incoming_edges"))
    )
    internal_citation_audit_papers = sum(
        1
        for paper in papers
        if bool(paper.metadata.get("internal_citation_matching_attempted"))
    )
    openalex_reference_audit_papers = sum(
        1
        for paper in papers
        if bool(paper.metadata.get("internal_citation_openalex_references_checked"))
    )
    paper_count = len(papers)
    return {
        "internal_reference_edge_count": internal_reference_edge_count,
        "internal_reference_edge_density": round(
            internal_reference_edge_count / paper_count if paper_count else 0.0,
            6,
        ),
        "materialized_internal_edge_count": materialized_internal_edge_count,
        "opencitations_incoming_edge_paper_coverage": round(
            opencitations_incoming_edge_papers / paper_count if paper_count else 0.0,
            6,
        ),
        "internal_citation_audit_coverage": round(
            internal_citation_audit_papers / paper_count if paper_count else 0.0,
            6,
        ),
        "openalex_reference_audit_coverage": round(
            openalex_reference_audit_papers / paper_count if paper_count else 0.0,
            6,
        ),
    }


def _has_openalex_id(paper: Paper) -> bool:
    external_ids = paper.metadata.get("external_ids")
    if isinstance(external_ids, dict) and external_ids.get("openalex"):
        return True
    extra = getattr(paper, "__pydantic_extra__", None) or {}
    return bool(extra.get("openalex"))


def _has_citation_count(paper: Paper) -> bool:
    extra = getattr(paper, "__pydantic_extra__", None) or {}
    return _numeric_or_none(extra.get("citation_count")) is not None or _numeric_or_none(
        paper.metadata.get("citation_count")
    ) is not None


def _has_yearly_citation_history(paper: Paper) -> bool:
    for key in (
        "citation_counts_by_year",
        "citations_by_year",
        "yearly_citation_counts",
        "citation_history",
    ):
        value = paper.metadata.get(key)
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, list) and value:
            return True
    if _known_zero_citations(paper):
        return True
    return False


def _known_zero_citations(paper: Paper) -> bool:
    extra = getattr(paper, "__pydantic_extra__", None) or {}
    for value in (
        extra.get("citation_count"),
        paper.metadata.get("citation_count"),
        _external_number(
            paper,
            "openalex_cited_by_count",
            "opencitations_cited_by_count",
            "semantic_scholar_citation_count",
        ),
    ):
        number = _numeric_or_none(value)
        if number is not None:
            return number == 0.0
    return False


def _has_artifact_evidence(paper: Paper) -> bool:
    return any(
        _external_number(paper, key) is not None
        for key in _artifact_keys()
    )


def _has_positive_artifact_evidence(paper: Paper) -> bool:
    return any(
        _positive_external_number(paper, key)
        for key in _artifact_keys()
    )


def _has_media_evidence(paper: Paper) -> bool:
    return any(_external_number(paper, key) is not None for key in _media_keys())


def _has_positive_media_evidence(paper: Paper) -> bool:
    return any(_positive_external_number(paper, key) for key in _media_keys())


def _has_award_evidence(paper: Paper) -> bool:
    return any(_external_number(paper, key) is not None for key in _award_keys())


def _has_positive_award_evidence(paper: Paper) -> bool:
    return any(_positive_external_number(paper, key) for key in _award_keys())


def _has_bip_source(paper: Paper) -> bool:
    return bool(paper.metadata.get("bip_enriched")) or _has_bip_impact_evidence(paper)


def _has_semantic_scholar_source(paper: Paper) -> bool:
    return bool(paper.metadata.get("semantic_scholar_enriched")) or (
        _has_influential_citation_evidence(paper)
    )


def _has_bip_impact_evidence(paper: Paper) -> bool:
    return any(_external_number(paper, key) is not None for key in _bip_keys())


def _has_positive_bip_impact_evidence(paper: Paper) -> bool:
    return any(_positive_external_number(paper, key) for key in _bip_keys())


def _has_influential_citation_evidence(paper: Paper) -> bool:
    return any(
        _external_number(paper, key) is not None
        for key in _influential_citation_keys()
    )


def _has_positive_influential_citation_evidence(paper: Paper) -> bool:
    return any(
        _positive_external_number(paper, key)
        for key in _influential_citation_keys()
    )


def _artifact_keys() -> tuple[str, ...]:
    return (
        "artifact_adoption",
        "artifact_adoption_count",
        "software_mentions",
        "software_citations",
        "github_stars",
        "repository_stars",
        "dataset_reuses",
        "dataset_citations",
        "model_downloads",
        "dataset_downloads",
        "huggingface_downloads",
    )


def _bip_keys() -> tuple[str, ...]:
    return (
        "bip_impact_percentile",
        "bip_percentile",
        "bip_field_year_percentile",
        "bip_impact_score",
        "bip_score",
        "bip_influence_score",
        "bip_impact_class",
        "bip_class",
        "bip_pagerank",
        "bip_page_rank",
        "bip_att_rank",
        "bip_attrank",
        "bip_ram",
        "bip_ram_score",
    )


def _influential_citation_keys() -> tuple[str, ...]:
    return (
        "semantic_scholar_influential_citation_count",
        "s2_influential_citation_count",
        "influential_citation_count",
        "highly_influential_citation_count",
    )


def _media_keys() -> tuple[str, ...]:
    return (
        "media_mentions",
        "news_mentions",
        "press_mentions",
        "altmetric_mentions",
    )


def _award_keys() -> tuple[str, ...]:
    return (
        "award_count",
        "awards",
        "best_paper_awards",
        "prize_count",
    )


def _external_number(paper: Paper, *keys: str) -> float | None:
    sources = [
        paper.metadata.get("external_impact"),
        paper.metadata.get("impact"),
        paper.metadata.get("metrics"),
        paper.metadata,
        getattr(paper, "__pydantic_extra__", None) or {},
    ]
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = _numeric_or_none(source.get(key))
            if value is not None:
                return value
    return None


def _positive_external_number(paper: Paper, *keys: str) -> bool:
    value = _external_number(paper, *keys)
    return value is not None and value > 0.0


def _numeric_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(number, 0.0)


def _coverage(items: list[Paper], predicate: Callable[[Paper], bool]) -> float:
    if not items:
        return 0.0
    return round(sum(1 for item in items if predicate(item)) / len(items), 6)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return _yes_no(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return _text(value)


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
