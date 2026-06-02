from __future__ import annotations

import csv
import json
import ast
from collections import Counter
from pathlib import Path
from typing import Any


CLAIM_SCOPE = "dataset-local paper impact index"


def impact_claim_certificate(
    run_dir: str | Path,
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    root = Path(run_dir)
    score_path = _find_score_table(root)
    scores = _load_score_rows(score_path) if score_path is not None else []
    project_readiness = _applicable_project_readiness(
        _read_json(root / "reports" / "project_readiness.json"),
        score_path,
    )
    external_evidence = _first_json(
        root / "external_evidence_readiness.json",
        root / "reports" / "external_evidence_readiness.json",
    )
    model_report = _read_json(root / "model_impact_report.json")
    impact_validity = _first_json(
        root / "impact_validity" / "impact_validity_evaluation.json",
        root / "reports" / "impact_validity_evaluation.json",
    )
    claim = _claim_decision(
        scores=scores,
        project_readiness=project_readiness,
        external_evidence=external_evidence,
        model_report=model_report,
        impact_validity=impact_validity,
    )
    return {
        "run_dir": str(root),
        "claim_scope": _claim_scope(project_readiness),
        "claim": claim,
        "score_summary": _score_summary(scores),
        "evidence": _evidence_summary(
            project_readiness=project_readiness,
            external_evidence=external_evidence,
            model_report=model_report,
            impact_validity=impact_validity,
            score_path=score_path,
        ),
        "top_papers": _top_papers(scores, top_n=top_n),
        "recommended_next_actions": _recommended_next_actions(
            claim,
            scores=scores,
            project_readiness=project_readiness,
            external_evidence=external_evidence,
            model_report=model_report,
            impact_validity=impact_validity,
        ),
        "limitations": [
            "The certificate reports dataset-local impact inside the supplied corpus.",
            "A high numeric score is not the same as validated impact accuracy unless future-window or external labels support it.",
            "Global scientific-impact claims require globally calibrated citation/adoption evidence.",
        ],
    }


def _applicable_project_readiness(
    project_readiness: dict[str, Any],
    score_path: Path | None,
) -> dict[str, Any]:
    if not project_readiness or score_path is None or score_path.name != "impact_scores.csv":
        return project_readiness
    decision = project_readiness.get("decision", {})
    failed_checks = set(project_readiness.get("failed_checks") or [])
    artifacts = project_readiness.get("artifacts", {})
    missing_required = int(_float(artifacts.get("missing_required_count", 0)) or 0)
    if (
        decision.get("status") == "blocked"
        and missing_required > 0
        and "paper_ranking_present" in failed_checks
    ):
        return {}
    return project_readiness


def write_impact_claim_certificate(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    certificate = impact_claim_certificate(run_dir, top_n=top_n)
    destination = Path(output_dir) if output_dir is not None else _default_output_dir(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "impact_claim_certificate.json").write_text(
        json.dumps(certificate, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "impact_claim_certificate.md").write_text(
        render_impact_claim_certificate_markdown(certificate),
        encoding="utf-8",
    )
    return certificate


def render_impact_claim_certificate_markdown(certificate: dict[str, Any]) -> str:
    claim = certificate.get("claim", {})
    score_summary = certificate.get("score_summary", {})
    evidence = certificate.get("evidence", {})
    lines = [
        "# Impact Claim Certificate",
        "",
        "## Claim",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| status | {_text(claim.get('status'))} |",
        f"| can_report_dataset_local_score | {_yes_no(claim.get('can_report_dataset_local_score'))} |",
        f"| can_claim_validated_impact_accuracy | {_yes_no(claim.get('can_claim_validated_impact_accuracy'))} |",
        f"| can_claim_high_credibility | {_yes_no(claim.get('can_claim_high_credibility'))} |",
        f"| confidence | {_number(claim.get('confidence'))} |",
        f"| scope | {_text(certificate.get('claim_scope'))} |",
        f"| reason | {_text(claim.get('reason'))} |",
        "",
        "## Score Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| paper_count | {score_summary.get('paper_count', 0)} |",
        f"| reportable_paper_count | {score_summary.get('reportable_paper_count', 0)} |",
        f"| exploratory_paper_count | {score_summary.get('exploratory_paper_count', 0)} |",
        f"| low_confidence_paper_count | {score_summary.get('low_confidence_paper_count', 0)} |",
        f"| average_validity_confidence | {_number(score_summary.get('average_validity_confidence'))} |",
        f"| min_score | {_number(score_summary.get('min_score'))} |",
        f"| max_score | {_number(score_summary.get('max_score'))} |",
        "",
        *(_reportability_reason_markdown_rows(score_summary)),
        *(_reportability_gap_action_markdown_rows(score_summary)),
        "## Evidence",
        "",
        "| Evidence | Value |",
        "| --- | --- |",
        f"| score_table | `{_text(evidence.get('score_table'))}` |",
        f"| project_readiness_status | {_text(evidence.get('project_readiness_status'))} |",
        f"| evidence_level | {_text(evidence.get('evidence_level'))} |",
        f"| external_evidence_status | {_text(evidence.get('external_evidence_status'))} |",
        f"| external_query_id_coverage | {_number(evidence.get('external_query_id_coverage'))} |",
        f"| field_year_percentile_coverage | {_number(evidence.get('field_year_percentile_coverage'))} |",
        f"| materialized_internal_edge_count | {evidence.get('materialized_internal_edge_count', 0)} |",
        f"| validated_impact_claim_status | {_text(evidence.get('validated_impact_claim_status'))} |",
        f"| impact_validity_status | {_text(evidence.get('impact_validity_status'))} |",
        f"| impact_validity_significant_vs_random | {_yes_no(evidence.get('impact_validity_significant_vs_random'))} |",
        f"| impact_credibility_status | {_text(evidence.get('impact_credibility_status'))} |",
        f"| impact_credibility_grade | {_text(evidence.get('impact_credibility_grade'))} |",
        f"| high_validation_credibility | {_yes_no(evidence.get('high_validation_credibility'))} |",
        f"| final_score_stability_status | {_text(evidence.get('final_score_stability_status'))} |",
        f"| final_score_stability_samples | {_text(evidence.get('final_score_stability_samples'))} |",
        f"| final_score_stability_mean_top_n_retention_rate | {_number(evidence.get('final_score_stability_mean_top_n_retention_rate'))} |",
        f"| final_score_stability_unstable_paper_count | {_text(evidence.get('final_score_stability_unstable_paper_count'))} |",
        "",
        "## Top Papers",
        "",
        "| Rank | Paper ID | Score | Validity Confidence | Evidence Tier | Reportability | Final Stability | Final Rank Span |",
        "| ---: | --- | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for index, paper in enumerate(certificate.get("top_papers", []), start=1):
        lines.append(
            f"| {index} | {_text(paper.get('paper_id'))} | "
            f"{_number(paper.get('score'))} | "
            f"{_number(paper.get('validity_confidence'))} | "
            f"{_text(paper.get('credible_evidence_tier'))} | "
            f"{_text(paper.get('reportability_status'))} | "
            f"{_text(paper.get('final_score_stability_band'))} | "
            f"{_text(paper.get('final_score_stability_rank_span'))} |"
        )
    lines.extend(["", "## Recommended Next Actions", ""])
    lines.extend(_list_rows(certificate.get("recommended_next_actions", [])))
    lines.extend(["", "## Limitations", ""])
    lines.extend(_list_rows(certificate.get("limitations", [])))
    lines.append("")
    return "\n".join(lines)


def _find_score_table(root: Path) -> Path | None:
    candidates = [
        root / "impact_scores.csv",
        root / "rankings" / "papers.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def _load_score_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _claim_decision(
    *,
    scores: list[dict[str, Any]],
    project_readiness: dict[str, Any],
    external_evidence: dict[str, Any],
    model_report: dict[str, Any],
    impact_validity: dict[str, Any],
) -> dict[str, Any]:
    readiness_decision = project_readiness.get("decision", {})
    if readiness_decision.get("status") == "blocked" and not scores:
        return {
            "status": "blocked",
            "can_report_dataset_local_score": False,
            "can_claim_validated_impact_accuracy": False,
            "can_claim_high_credibility": False,
            "confidence": 0.0,
            "reason": readiness_decision.get(
                "reason",
                "Project readiness blocks reporting.",
            ),
        }
    if not scores:
        return {
            "status": "blocked",
            "can_report_dataset_local_score": False,
            "can_claim_validated_impact_accuracy": False,
            "can_claim_high_credibility": False,
            "confidence": 0.0,
            "reason": "No paper score table was found.",
        }

    validated_claim = _validated_claim(model_report, impact_validity)
    can_claim_validated = bool(
        validated_claim.get("can_claim_validated_impact_accuracy", False)
    )
    final_score_stability = _final_score_stability(model_report)
    external_decision = external_evidence.get("decision", {})
    can_claim_high_credibility = bool(
        external_decision.get("can_claim_high_credibility", False)
    )
    confidence = _claim_confidence(scores, validated_claim)
    if _final_score_stability_blocks_claim(final_score_stability):
        status = final_score_stability.get("status")
        reason = final_score_stability.get(
            "reason",
            "Final score bootstrap stability does not support reportable claims.",
        )
        return {
            "status": "exploratory",
            "can_report_dataset_local_score": True,
            "can_claim_validated_impact_accuracy": False,
            "can_claim_high_credibility": False,
            "confidence": min(confidence, 0.4),
            "reason": (
                "Scores are usable only as exploratory dataset-local evidence "
                f"because final score stability is {status}: {reason}"
            ),
        }
    if not _has_reportable_evidence(scores):
        return {
            "status": "exploratory",
            "can_report_dataset_local_score": True,
            "can_claim_validated_impact_accuracy": False,
            "can_claim_high_credibility": False,
            "confidence": min(confidence, 0.4),
            "reason": (
                "No papers are marked reportable; scores are usable only as "
                "exploratory dataset-local evidence."
            ),
        }
    if can_claim_validated and can_claim_high_credibility:
        return {
            "status": "validated",
            "can_report_dataset_local_score": True,
            "can_claim_validated_impact_accuracy": True,
            "can_claim_high_credibility": True,
            "confidence": confidence,
            "reason": validated_claim.get(
                "reason",
                "Validity labels and external evidence support this score.",
            ),
        }
    if _has_reportable_evidence(scores):
        return {
            "status": "evidence_supported",
            "can_report_dataset_local_score": True,
            "can_claim_validated_impact_accuracy": False,
            "can_claim_high_credibility": can_claim_high_credibility,
            "confidence": confidence,
            "reason": (
                "Scores have auditable local evidence, but validated impact accuracy "
                "is not supported by external or future-window labels."
            ),
        }
    return {
        "status": "exploratory",
        "can_report_dataset_local_score": True,
        "can_claim_validated_impact_accuracy": False,
        "can_claim_high_credibility": False,
        "confidence": min(confidence, 0.4),
        "reason": "Scores are usable only as exploratory dataset-local evidence.",
    }


def _evidence_summary(
    *,
    project_readiness: dict[str, Any],
    external_evidence: dict[str, Any],
    model_report: dict[str, Any],
    impact_validity: dict[str, Any],
    score_path: Path | None,
) -> dict[str, Any]:
    readiness_decision = project_readiness.get("decision", {})
    quality = project_readiness.get("quality", {})
    external_quality = quality.get("external_evidence", {})
    impact_quality = quality.get("impact_validity", {})
    external_decision = external_evidence.get("decision", {})
    identifiers = external_evidence.get("identifier_coverage", {})
    signals = external_evidence.get("signal_coverage", {})
    graph = external_evidence.get("graph_evidence", {})
    validated_claim = _validated_claim(model_report, impact_validity)
    final_score_stability = _final_score_stability(model_report)
    scorecard = impact_validity.get("credibility_scorecard", {})
    return {
        "score_table": str(score_path) if score_path is not None else None,
        "project_readiness_status": readiness_decision.get("status"),
        "project_readiness_reason": readiness_decision.get("reason"),
        "evidence_level": quality.get("evidence_level"),
        "external_evidence_status": external_decision.get(
            "status",
            external_quality.get("status"),
        ),
        "can_claim_high_credibility": bool(
            external_decision.get(
                "can_claim_high_credibility",
                external_quality.get("can_claim_high_credibility", False),
            )
        ),
        "external_query_id_coverage": identifiers.get(
            "external_query_id_coverage",
            external_quality.get("external_query_id_coverage", 0.0),
        ),
        "field_year_percentile_coverage": signals.get(
            "field_year_percentile_coverage",
            external_quality.get("field_year_percentile_coverage", 0.0),
        ),
        "materialized_internal_edge_count": graph.get(
            "materialized_internal_edge_count",
            external_quality.get("materialized_internal_edge_count", 0),
        ),
        "internal_citation_audit_coverage": graph.get(
            "internal_citation_audit_coverage",
            external_quality.get("internal_citation_audit_coverage", 0.0),
        ),
        "openalex_reference_audit_coverage": graph.get(
            "openalex_reference_audit_coverage",
            external_quality.get("openalex_reference_audit_coverage", 0.0),
        ),
        "validated_impact_claim_status": validated_claim.get("status"),
        "validated_impact_claim_reason": validated_claim.get("reason"),
        "impact_validity_status": _impact_validity_status(impact_validity)
        or impact_quality.get("status"),
        "impact_validity_significant_vs_random": _impact_validity_significant(
            impact_validity
        )
        or bool(impact_quality.get("significant_vs_random", False)),
        "impact_credibility_status": scorecard.get(
            "status",
            impact_quality.get("credibility_scorecard_status"),
        ),
        "impact_credibility_grade": scorecard.get(
            "grade",
            impact_quality.get("credibility_grade"),
        ),
        "high_validation_credibility": bool(
            scorecard.get(
                "can_claim_high_validation_credibility",
                impact_quality.get("can_claim_high_validation_credibility", False),
            )
        ),
        "final_score_stability_status": final_score_stability.get("status"),
        "final_score_stability_samples": final_score_stability.get("samples"),
        "final_score_stability_mean_top_n_retention_rate": final_score_stability.get(
            "mean_top_n_retention_rate"
        ),
        "final_score_stability_unstable_paper_count": final_score_stability.get(
            "unstable_paper_count"
        ),
    }


def _score_summary(scores: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_scores = [_score_value(row) for row in scores if _score_value(row) is not None]
    confidences = [
        _float(row.get("validity_confidence"))
        for row in scores
        if _float(row.get("validity_confidence")) is not None
    ]
    reportability_counts = Counter(_text(row.get("reportability_status")) for row in scores)
    reportability_reason_counts = Counter(
        _text(row.get("reportability_reason"))
        for row in scores
        if _text(row.get("reportability_reason"))
    )
    reportability_gap_action_counts = Counter()
    for row in scores:
        for action in _parse_reportability_gap_actions(
            row.get("reportability_gap_actions")
        ):
            reportability_gap_action_counts[action] += 1
    tier_counts = Counter(_text(row.get("credible_evidence_tier")) for row in scores)
    family_counts = Counter(_text(row.get("dominant_score_family")) for row in scores)
    final_stability_counts = Counter(
        _text(row.get("final_score_stability_band"))
        for row in scores
        if _text(row.get("final_score_stability_band"))
    )
    missing_counts = Counter()
    for row in scores:
        for item in _parse_missing_high_value_evidence(
            row.get("missing_high_value_evidence")
        ):
            missing_counts[item] += 1
    return {
        "paper_count": len(scores),
        "reportable_paper_count": reportability_counts.get("reportable", 0),
        "exploratory_paper_count": reportability_counts.get("exploratory", 0),
        "low_confidence_paper_count": reportability_counts.get("low_confidence", 0),
        "average_validity_confidence": _mean(confidences),
        "min_score": min(numeric_scores) if numeric_scores else None,
        "max_score": max(numeric_scores) if numeric_scores else None,
        "credible_evidence_tier_counts": dict(tier_counts),
        "reportability_status_counts": dict(reportability_counts),
        "reportability_reason_counts": dict(reportability_reason_counts),
        "reportability_gap_action_counts": dict(reportability_gap_action_counts),
        "dominant_score_family_counts": dict(family_counts),
        "final_score_stability_band_counts": dict(final_stability_counts),
        "missing_high_value_evidence_counts": dict(missing_counts),
    }


def _top_papers(scores: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    ranked = sorted(
        scores,
        key=lambda row: (_score_value(row) is not None, _score_value(row) or 0.0),
        reverse=True,
    )
    rows = []
    for row in ranked[:top_n]:
        rows.append(
            {
                "paper_id": row.get("paper_id") or row.get("id"),
                "title": row.get("title", ""),
                "score": _score_value(row),
                "validity_status": row.get("validity_status"),
                "validity_confidence": _float(row.get("validity_confidence")),
                "credible_evidence_tier": row.get("credible_evidence_tier"),
                "reportability_status": row.get("reportability_status"),
                "reportability_gap_actions": _parse_reportability_gap_actions(
                    row.get("reportability_gap_actions")
                ),
                "dominant_score_family": row.get("dominant_score_family"),
                "final_score_stability_band": row.get("final_score_stability_band"),
                "final_score_stability_rank_span": _int_or_none(
                    row.get("final_score_stability_rank_span")
                ),
                "final_score_stability_score_ci_width": _float(
                    row.get("final_score_stability_score_ci_width")
                ),
                "missing_high_value_evidence": _parse_missing_high_value_evidence(
                    row.get("missing_high_value_evidence")
                ),
            }
        )
    return rows


def _recommended_next_actions(
    claim: dict[str, Any],
    *,
    scores: list[dict[str, Any]],
    project_readiness: dict[str, Any],
    external_evidence: dict[str, Any],
    model_report: dict[str, Any],
    impact_validity: dict[str, Any],
) -> list[str]:
    if claim.get("status") == "blocked":
        return ["Fix blocked readiness checks or provide a paper score table."]
    if claim.get("status") == "validated":
        return [
            "Use the certificate, score table, and evidence disclosures in the final report."
        ]
    actions: list[str] = []
    if not _has_reportable_evidence(scores):
        actions.append(
            "No papers are marked reportable; improve evidence coverage or reportability gates before using the ranking in a public report."
        )
    if not _validated_claim(model_report, impact_validity).get(
        "can_claim_validated_impact_accuracy",
        False,
    ):
        actions.append(
            "Supply future-window or external impact labels and run impact validity evaluation."
        )
    if not _external_can_claim_high_credibility(external_evidence, project_readiness):
        actions.extend(
            _external_evidence_next_actions(
                external_evidence,
                project_readiness,
            )
        )
    if _final_score_stability_blocks_claim(_final_score_stability(model_report)):
        actions.append(
            "Increase pairwise comparison coverage or rerun final-score stability auditing before reporting ranked scores."
        )
    if _top_papers_have_unstable_final_score_stability(scores):
        actions.append(
            "Treat top papers with unstable final-score stability as exploratory until additional pairwise comparisons narrow their rank movement."
        )
    if _low_reportability_share(scores):
        actions.append(
            "Review papers marked exploratory or low_confidence before using the ranking in a public report."
        )
    if not actions:
        actions.append("Use the score table with the certificate limitations attached.")
    return actions


def _external_evidence_next_actions(
    external_evidence: dict[str, Any],
    project_readiness: dict[str, Any],
) -> list[str]:
    quality = project_readiness.get("quality", {}).get("external_evidence", {})
    identifiers = external_evidence.get("identifier_coverage", {})
    sources = external_evidence.get("source_coverage", {})
    signals = external_evidence.get("signal_coverage", {})
    graph = external_evidence.get("graph_evidence", {})

    external_query_id_coverage = _float(
        identifiers.get(
            "external_query_id_coverage",
            quality.get("external_query_id_coverage", 0.0),
        )
    )
    citation_count_coverage = _float(signals.get("citation_count_coverage", 0.0))
    field_year_coverage = _float(
        signals.get("field_year_percentile_coverage", 0.0)
    )
    influential_coverage = _float(
        signals.get("influential_citation_evidence_coverage", 0.0)
    )
    openalex_coverage = _float(sources.get("openalex_enriched_coverage", 0.0))
    opencitations_coverage = _float(
        sources.get("opencitations_enriched_coverage", 0.0)
    )
    semantic_scholar_coverage = _float(
        sources.get("semantic_scholar_enriched_coverage", 0.0)
    )
    graph_edges = int(
        _float(
            graph.get(
                "materialized_internal_edge_count",
                graph.get("internal_reference_edge_count", 0),
            )
        )
        or 0
    )

    actions: list[str] = []
    if external_query_id_coverage < 0.8:
        actions.append(
            "Improve DOI/OpenAlex identifier coverage before making high-credibility claims."
        )
    if openalex_coverage < 0.6 or citation_count_coverage < 0.7 or field_year_coverage < 0.6:
        actions.append(
            "Run OpenAlex/OpenCitations enrichment to strengthen citation counts and field/year percentile evidence."
        )
    if semantic_scholar_coverage < 0.6 or influential_coverage < 0.6:
        actions.append(
            "Run Semantic Scholar enrichment to add influential-citation evidence."
        )
    if graph_edges <= 0:
        if _float(graph.get("internal_citation_audit_coverage", 0.0)) >= 0.95:
            actions.append(
                "Disclose that the internal citation graph was audited and no dataset-internal edges were found."
            )
        else:
            actions.append(
                "Add or materialize dataset-internal citation edges so graph evidence can support the local ranking."
            )
    if not actions:
        actions.append(
            "Attach external-evidence disclosures; high-credibility status is still blocked by the current evidence gate."
        )
    return actions


def _validated_claim(
    model_report: dict[str, Any],
    impact_validity: dict[str, Any],
) -> dict[str, Any]:
    model_claim = model_report.get("summary", {}).get("validated_impact_claim")
    if isinstance(model_claim, dict) and model_claim:
        return model_claim
    status = _impact_validity_status(impact_validity)
    significant = _impact_validity_significant(impact_validity)
    if status == "externally_supported" and significant:
        return {
            "status": "externally_validated",
            "can_claim_validated_impact_accuracy": True,
            "confidence_ceiling": 0.85,
            "reason": "Validity evaluation is statistically supported.",
        }
    if status:
        return {
            "status": status,
            "can_claim_validated_impact_accuracy": False,
            "confidence_ceiling": 0.4,
            "reason": "Validity evaluation does not support a validated accuracy claim.",
        }
    return {
        "status": "not_provided",
        "can_claim_validated_impact_accuracy": False,
        "confidence_ceiling": 0.25,
        "reason": "No future-window or external validity labels were supplied.",
    }


def _final_score_stability(model_report: dict[str, Any]) -> dict[str, Any]:
    summary = model_report.get("summary", {}).get("final_score_stability")
    return summary if isinstance(summary, dict) else {}


def _final_score_stability_blocks_claim(stability: dict[str, Any]) -> bool:
    return str(stability.get("status", "")).lower() in {
        "unstable",
        "insufficient_samples",
    }


def _impact_validity_status(report: dict[str, Any]) -> str | None:
    if not report:
        return None
    return report.get("summary", {}).get("decision", {}).get("status") or report.get(
        "status"
    )


def _impact_validity_significant(report: dict[str, Any]) -> bool:
    return bool(
        report.get("statistical_significance", {}).get(
            "significant_vs_random",
            False,
        )
    )


def _external_can_claim_high_credibility(
    external_evidence: dict[str, Any],
    project_readiness: dict[str, Any],
) -> bool:
    return bool(
        external_evidence.get("decision", {}).get(
            "can_claim_high_credibility",
            project_readiness.get("quality", {})
            .get("external_evidence", {})
            .get("can_claim_high_credibility", False),
        )
    )


def _has_reportable_evidence(scores: list[dict[str, Any]]) -> bool:
    return any(row.get("reportability_status") == "reportable" for row in scores)


def _low_reportability_share(scores: list[dict[str, Any]]) -> bool:
    if not scores:
        return True
    weak_count = sum(
        1
        for row in scores
        if row.get("reportability_status") in {"exploratory", "low_confidence"}
    )
    return (weak_count / len(scores)) > 0.5


def _top_papers_have_unstable_final_score_stability(scores: list[dict[str, Any]]) -> bool:
    for row in sorted(
        scores,
        key=lambda item: (_score_value(item) is not None, _score_value(item) or 0.0),
        reverse=True,
    )[:10]:
        if str(row.get("final_score_stability_band", "")).lower() == "unstable":
            return True
    return False


def _claim_confidence(
    scores: list[dict[str, Any]],
    validated_claim: dict[str, Any],
) -> float:
    confidences = [
        _float(row.get("validity_confidence"))
        for row in scores
        if _float(row.get("validity_confidence")) is not None
    ]
    if confidences:
        base = sum(confidences) / len(confidences)
    else:
        tier_prior = {
            "very_high": 0.85,
            "high": 0.7,
            "moderate": 0.55,
            "low": 0.35,
            "model_only": 0.25,
        }
        priors = [
            tier_prior.get(str(row.get("credible_evidence_tier")), 0.35)
            for row in scores
        ]
        base = sum(priors) / len(priors) if priors else 0.0
    ceiling = _float(validated_claim.get("confidence_ceiling"))
    if ceiling is not None:
        base = min(base, ceiling)
    return round(base, 6)


def _score_value(row: dict[str, Any]) -> float | None:
    for key in (
        "final_impact_score",
        "impact_index_v3",
        "credibility_adjusted_impact_score",
        "final_score",
        "score",
    ):
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _parse_missing_high_value_evidence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return _clean_missing_items(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "[]"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (json.JSONDecodeError, SyntaxError, ValueError):
                continue
            if isinstance(parsed, list | tuple | set):
                return _clean_missing_items(parsed)
    return _clean_missing_items(text.replace(";", ",").split(","))


def _parse_reportability_gap_actions(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return _clean_gap_actions(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "[]"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (json.JSONDecodeError, SyntaxError, ValueError):
                continue
            if isinstance(parsed, list | tuple | set):
                return _clean_gap_actions(parsed)
    return _clean_gap_actions(text.replace("|", ";").split(";"))


def _clean_gap_actions(values: Any) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        item = str(value).strip().strip("'\"")
        if not item or item.lower() in {"none", "null", "[]"}:
            continue
        cleaned.append(item)
    return cleaned


def _clean_missing_items(values: Any) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        item = str(value).strip().strip("'\"")
        if not item or item.lower() in {"none", "null", "[]"}:
            continue
        cleaned.append(item)
    return cleaned


def _claim_scope(project_readiness: dict[str, Any]) -> str:
    return project_readiness.get("claim_scope") or CLAIM_SCOPE


def _default_output_dir(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    if (root / "reports").exists():
        return root / "reports"
    return root


def _first_json(*paths: Path) -> dict[str, Any]:
    for path in paths:
        payload = _read_json(path)
        if payload:
            return payload
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    numeric = _float(value)
    if numeric is None:
        return None
    return int(numeric)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _number(value: Any) -> str:
    numeric = _float(value)
    if numeric is None:
        return ""
    return f"{numeric:.3f}"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _list_rows(values: list[Any]) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- {_text(value)}" for value in values]


def _reportability_reason_markdown_rows(score_summary: dict[str, Any]) -> list[str]:
    reason_counts = score_summary.get("reportability_reason_counts", {})
    if not isinstance(reason_counts, dict) or not reason_counts:
        return []
    rows = [
        "## Reportability Reasons",
        "",
        "| Reason | Paper Count |",
        "| --- | ---: |",
    ]
    for reason, count in sorted(
        reason_counts.items(),
        key=lambda item: (-int(item[1] or 0), str(item[0])),
    ):
        rows.append(f"| {_text(reason)} | {count} |")
    rows.append("")
    return rows


def _reportability_gap_action_markdown_rows(score_summary: dict[str, Any]) -> list[str]:
    action_counts = score_summary.get("reportability_gap_action_counts", {})
    if not isinstance(action_counts, dict) or not action_counts:
        return []
    rows = [
        "## Reportability Gap Actions",
        "",
        "| Action | Paper Count |",
        "| --- | ---: |",
    ]
    for action, count in sorted(
        action_counts.items(),
        key=lambda item: (-int(item[1] or 0), str(item[0])),
    ):
        rows.append(f"| {_text(action)} | {count} |")
    rows.append("")
    return rows
