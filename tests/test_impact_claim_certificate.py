import csv
import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.impact_claim_certificate import (
    impact_claim_certificate,
    render_impact_claim_certificate_markdown,
    write_impact_claim_certificate,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_impact_scores(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_model_score_directory_certificate_marks_unvalidated_evidence_supported(tmp_path):
    _write_impact_scores(
        tmp_path / "impact_scores.csv",
        [
            {
                "rank": 1,
                "paper_id": "p1",
                "title": "Strong paper",
                "final_impact_score": 91.5,
                "validity_status": "unvalidated",
                "validity_confidence": 0.25,
                "credible_evidence_tier": "high",
                "reportability_status": "reportable",
                "dominant_score_family": "evidence",
                "missing_high_value_evidence": "",
            },
            {
                "rank": 2,
                "paper_id": "p2",
                "title": "Weak paper",
                "final_impact_score": 24.0,
                "validity_status": "unvalidated",
                "validity_confidence": 0.25,
                "credible_evidence_tier": "low",
                "reportability_status": "exploratory",
                "dominant_score_family": "model",
                "missing_high_value_evidence": "field_year_percentile",
            },
        ],
    )
    _write_json(
        tmp_path / "model_impact_report.json",
        {
            "summary": {
                "validated_impact_claim": {
                    "status": "unvalidated",
                    "can_claim_validated_impact_accuracy": False,
                    "confidence_ceiling": 0.25,
                    "reason": "No labels.",
                },
                "ranking_quality": {"status": "usable"},
            }
        },
    )
    _write_json(
        tmp_path / "external_evidence_readiness.json",
        {
            "decision": {
                "status": "usable_with_disclosure",
                "can_claim_high_credibility": False,
                "reason": "External evidence is partial.",
            },
            "identifier_coverage": {"external_query_id_coverage": 0.65},
            "signal_coverage": {"field_year_percentile_coverage": 0.45},
            "graph_evidence": {"materialized_internal_edge_count": 3},
        },
    )

    certificate = impact_claim_certificate(tmp_path)

    assert certificate["claim"]["status"] == "evidence_supported"
    assert certificate["claim"]["can_claim_validated_impact_accuracy"] is False
    assert certificate["claim"]["can_claim_high_credibility"] is False
    assert certificate["score_summary"]["paper_count"] == 2
    assert certificate["score_summary"]["reportable_paper_count"] == 1
    assert certificate["top_papers"][0]["paper_id"] == "p1"
    assert "Supply future-window" in certificate["recommended_next_actions"][0]
    markdown = render_impact_claim_certificate_markdown(certificate)
    assert "# Impact Claim Certificate" in markdown
    assert "| status | evidence_supported |" in markdown


def test_certificate_requires_at_least_one_reportable_paper_for_evidence_supported_claim(
    tmp_path,
):
    _write_impact_scores(
        tmp_path / "impact_scores.csv",
        [
            {
                "rank": 1,
                "paper_id": "p1",
                "title": "High evidence but not reportable",
                "final_impact_score": 91.5,
                "validity_status": "unvalidated",
                "validity_confidence": 0.25,
                "credible_evidence_tier": "high",
                "reportability_status": "exploratory",
                "reportability_reason": "Evidence requires disclosure before reporting.",
                "reportability_gap_actions": "['Add patent impact evidence.', 'Add policy impact evidence.']",
                "dominant_score_family": "evidence",
                "missing_high_value_evidence": "",
            },
            {
                "rank": 2,
                "paper_id": "p2",
                "title": "Low evidence paper",
                "final_impact_score": 10.0,
                "validity_status": "unvalidated",
                "validity_confidence": 0.25,
                "credible_evidence_tier": "model_only",
                "reportability_status": "low_confidence",
                "reportability_reason": "Too little positive evidence is available.",
                "reportability_gap_actions": "Add at least two positive non-model evidence signals.; Strengthen evidence tier to high or very_high.",
                "dominant_score_family": "model",
                "missing_high_value_evidence": "field_year_percentile",
            }
        ],
    )

    certificate = impact_claim_certificate(tmp_path)

    assert certificate["claim"]["status"] == "exploratory"
    assert certificate["claim"]["can_report_dataset_local_score"] is True
    assert certificate["claim"]["can_claim_validated_impact_accuracy"] is False
    assert certificate["claim"]["can_claim_high_credibility"] is False
    assert certificate["score_summary"]["reportable_paper_count"] == 0
    assert certificate["score_summary"]["reportability_reason_counts"] == {
        "Evidence requires disclosure before reporting.": 1,
        "Too little positive evidence is available.": 1,
    }
    assert certificate["score_summary"]["reportability_gap_action_counts"] == {
        "Add patent impact evidence.": 1,
        "Add policy impact evidence.": 1,
        "Add at least two positive non-model evidence signals.": 1,
        "Strengthen evidence tier to high or very_high.": 1,
    }
    assert certificate["top_papers"][0]["reportability_gap_actions"] == [
        "Add patent impact evidence.",
        "Add policy impact evidence.",
    ]
    assert "No papers are marked reportable" in certificate["claim"]["reason"]
    assert any("No papers are marked reportable" in action for action in certificate["recommended_next_actions"])
    markdown = render_impact_claim_certificate_markdown(certificate)
    assert "Reportability Reasons" in markdown
    assert "Reportability Gap Actions" in markdown
    assert "Evidence requires disclosure before reporting." in markdown
    assert "Add patent impact evidence." in markdown


def test_certificate_parses_missing_high_value_evidence_lists(tmp_path):
    _write_impact_scores(
        tmp_path / "impact_scores.csv",
        [
            {
                "paper_id": "p1",
                "title": "Paper 1",
                "final_impact_score": 80,
                "validity_confidence": 0.25,
                "credible_evidence_tier": "high",
                "reportability_status": "exploratory",
                "dominant_score_family": "evidence",
                "missing_high_value_evidence": "['patent', 'policy', 'artifact_adoption']",
            },
            {
                "paper_id": "p2",
                "title": "Paper 2",
                "final_impact_score": 70,
                "validity_confidence": 0.25,
                "credible_evidence_tier": "moderate",
                "reportability_status": "exploratory",
                "dominant_score_family": "evidence",
                "missing_high_value_evidence": "patent; policy",
            },
            {
                "paper_id": "p3",
                "title": "Paper 3",
                "final_impact_score": 60,
                "validity_confidence": 0.25,
                "credible_evidence_tier": "moderate",
                "reportability_status": "exploratory",
                "dominant_score_family": "evidence",
                "missing_high_value_evidence": "none",
            },
        ],
    )

    certificate = impact_claim_certificate(tmp_path)

    assert certificate["score_summary"]["missing_high_value_evidence_counts"] == {
        "patent": 2,
        "policy": 2,
        "artifact_adoption": 1,
    }
    assert certificate["top_papers"][0]["missing_high_value_evidence"] == [
        "patent",
        "policy",
        "artifact_adoption",
    ]


def test_certificate_marks_validated_when_labels_and_external_evidence_support_claim(tmp_path):
    _write_impact_scores(
        tmp_path / "impact_scores.csv",
        [
            {
                "paper_id": "p1",
                "title": "Validated paper",
                "final_impact_score": 95,
                "validity_status": "externally_validated",
                "validity_confidence": 0.82,
                "credible_evidence_tier": "very_high",
                "reportability_status": "reportable",
                "dominant_score_family": "evidence",
                "missing_high_value_evidence": "",
            }
        ],
    )
    _write_json(
        tmp_path / "model_impact_report.json",
        {
            "summary": {
                "validated_impact_claim": {
                    "status": "externally_validated",
                    "can_claim_validated_impact_accuracy": True,
                    "confidence_ceiling": 0.9,
                    "reason": "Scores align with future labels.",
                }
            }
        },
    )
    _write_json(
        tmp_path / "external_evidence_readiness.json",
        {
            "decision": {
                "status": "ready_for_high_credibility_scoring",
                "can_claim_high_credibility": True,
                "reason": "External evidence is strong.",
            },
            "identifier_coverage": {"external_query_id_coverage": 0.9},
            "signal_coverage": {"field_year_percentile_coverage": 0.8},
            "graph_evidence": {"materialized_internal_edge_count": 12},
        },
    )

    certificate = impact_claim_certificate(tmp_path)

    assert certificate["claim"]["status"] == "validated"
    assert certificate["claim"]["can_claim_validated_impact_accuracy"] is True
    assert certificate["claim"]["can_claim_high_credibility"] is True
    assert certificate["claim"]["confidence"] == 0.82
    assert certificate["recommended_next_actions"] == [
        "Use the certificate, score table, and evidence disclosures in the final report."
    ]


def test_certificate_downgrades_claim_when_final_score_stability_is_unstable(tmp_path):
    _write_impact_scores(
        tmp_path / "impact_scores.csv",
        [
            {
                "paper_id": "p1",
                "title": "Nominally strong paper",
                "final_impact_score": 95,
                "validity_status": "externally_validated",
                "validity_confidence": 0.82,
                "credible_evidence_tier": "very_high",
                "reportability_status": "reportable",
                "dominant_score_family": "evidence",
                "missing_high_value_evidence": "",
            }
        ],
    )
    _write_json(
        tmp_path / "model_impact_report.json",
        {
            "summary": {
                "validated_impact_claim": {
                    "status": "externally_validated",
                    "can_claim_validated_impact_accuracy": True,
                    "confidence_ceiling": 0.9,
                    "reason": "Scores align with future labels.",
                },
                "final_score_stability": {
                    "status": "unstable",
                    "samples": 80,
                    "mean_top_n_retention_rate": 0.42,
                    "unstable_paper_count": 8,
                    "reason": "Bootstrap resampling changes the top-N final-score set substantially.",
                },
            }
        },
    )
    _write_json(
        tmp_path / "external_evidence_readiness.json",
        {
            "decision": {
                "status": "ready_for_high_credibility_scoring",
                "can_claim_high_credibility": True,
                "reason": "External evidence is strong.",
            }
        },
    )

    certificate = impact_claim_certificate(tmp_path)

    assert certificate["claim"]["status"] == "exploratory"
    assert certificate["claim"]["can_claim_validated_impact_accuracy"] is False
    assert certificate["claim"]["can_claim_high_credibility"] is False
    assert certificate["claim"]["confidence"] <= 0.4
    assert "final score stability is unstable" in certificate["claim"]["reason"]
    assert certificate["evidence"]["final_score_stability_status"] == "unstable"
    assert certificate["evidence"]["final_score_stability_mean_top_n_retention_rate"] == 0.42
    assert any("Increase pairwise comparison coverage" in action for action in certificate["recommended_next_actions"])
    markdown = render_impact_claim_certificate_markdown(certificate)
    assert "final_score_stability_status" in markdown


def test_certificate_exposes_per_paper_final_score_stability_in_top_papers(tmp_path):
    _write_impact_scores(
        tmp_path / "impact_scores.csv",
        [
            {
                "paper_id": "p1",
                "title": "Stable top paper",
                "final_impact_score": 90,
                "validity_confidence": 0.25,
                "credible_evidence_tier": "high",
                "reportability_status": "reportable",
                "dominant_score_family": "evidence",
                "missing_high_value_evidence": "",
                "final_score_stability_band": "stable",
                "final_score_stability_rank_span": "2",
                "final_score_stability_score_ci_width": "4.5",
            },
            {
                "paper_id": "p2",
                "title": "Unstable near-top paper",
                "final_impact_score": 87,
                "validity_confidence": 0.25,
                "credible_evidence_tier": "high",
                "reportability_status": "reportable",
                "dominant_score_family": "evidence",
                "missing_high_value_evidence": "",
                "final_score_stability_band": "unstable",
                "final_score_stability_rank_span": "21",
                "final_score_stability_score_ci_width": "44.0",
            },
        ],
    )
    _write_json(
        tmp_path / "model_impact_report.json",
        {
            "summary": {
                "final_score_stability": {
                    "status": "moderate",
                    "samples": 80,
                    "mean_top_n_retention_rate": 0.82,
                    "unstable_paper_count": 1,
                    "reason": "Some final-score movement is visible.",
                }
            }
        },
    )

    certificate = impact_claim_certificate(tmp_path)

    assert certificate["score_summary"]["final_score_stability_band_counts"] == {
        "stable": 1,
        "unstable": 1,
    }
    assert certificate["top_papers"][0]["final_score_stability_band"] == "stable"
    assert certificate["top_papers"][1]["final_score_stability_band"] == "unstable"
    assert certificate["top_papers"][1]["final_score_stability_rank_span"] == 21
    assert certificate["top_papers"][1]["final_score_stability_score_ci_width"] == 44.0
    assert any("unstable final-score stability" in action for action in certificate["recommended_next_actions"])
    markdown = render_impact_claim_certificate_markdown(certificate)
    assert "Final Stability" in markdown
    assert "unstable" in markdown


def test_certificate_recommends_graph_evidence_when_external_coverage_is_strong(tmp_path):
    _write_impact_scores(
        tmp_path / "impact_scores.csv",
        [
            {
                "paper_id": "p1",
                "title": "Evidence-supported paper",
                "final_impact_score": 80,
                "validity_status": "unvalidated",
                "validity_confidence": 0.25,
                "credible_evidence_tier": "high",
                "reportability_status": "exploratory",
                "dominant_score_family": "evidence",
                "missing_high_value_evidence": "",
            }
        ],
    )
    _write_json(
        tmp_path / "model_impact_report.json",
        {
            "summary": {
                "validated_impact_claim": {
                    "status": "unvalidated",
                    "can_claim_validated_impact_accuracy": False,
                    "confidence_ceiling": 0.25,
                    "reason": "No labels.",
                }
            }
        },
    )
    _write_json(
        tmp_path / "external_evidence_readiness.json",
        {
            "decision": {
                "status": "usable_with_disclosure",
                "can_claim_high_credibility": False,
                "reason": "Internal graph evidence is missing.",
            },
            "identifier_coverage": {"external_query_id_coverage": 0.8},
            "source_coverage": {
                "openalex_enriched_coverage": 0.76,
                "opencitations_enriched_coverage": 0.8,
                "semantic_scholar_enriched_coverage": 0.74,
            },
            "signal_coverage": {
                "citation_count_coverage": 0.8,
                "field_year_percentile_coverage": 0.74,
                "influential_citation_evidence_coverage": 0.74,
            },
            "graph_evidence": {"materialized_internal_edge_count": 0},
        },
    )

    certificate = impact_claim_certificate(tmp_path)

    actions = certificate["recommended_next_actions"]
    assert any("dataset-internal citation edges" in action for action in actions)
    assert not any("DOI/OpenAlex IDs" in action for action in actions)


def test_certificate_discloses_completed_empty_internal_graph_audit(tmp_path):
    _write_impact_scores(
        tmp_path / "impact_scores.csv",
        [
            {
                "paper_id": "p1",
                "title": "Evidence-supported paper",
                "final_impact_score": 80,
                "validity_status": "unvalidated",
                "validity_confidence": 0.25,
                "credible_evidence_tier": "high",
                "reportability_status": "exploratory",
                "dominant_score_family": "evidence",
                "missing_high_value_evidence": "",
            }
        ],
    )
    _write_json(
        tmp_path / "model_impact_report.json",
        {
            "summary": {
                "validated_impact_claim": {
                    "status": "unvalidated",
                    "can_claim_validated_impact_accuracy": False,
                    "confidence_ceiling": 0.25,
                    "reason": "No labels.",
                }
            }
        },
    )
    _write_json(
        tmp_path / "external_evidence_readiness.json",
        {
            "decision": {
                "status": "usable_with_disclosure",
                "can_claim_high_credibility": False,
                "reason": "Internal graph was audited and empty.",
            },
            "identifier_coverage": {"external_query_id_coverage": 0.8},
            "source_coverage": {
                "openalex_enriched_coverage": 0.76,
                "opencitations_enriched_coverage": 0.8,
                "semantic_scholar_enriched_coverage": 0.74,
            },
            "signal_coverage": {
                "citation_count_coverage": 0.8,
                "field_year_percentile_coverage": 0.74,
                "influential_citation_evidence_coverage": 0.74,
            },
            "graph_evidence": {
                "materialized_internal_edge_count": 0,
                "internal_citation_audit_coverage": 1.0,
                "openalex_reference_audit_coverage": 1.0,
            },
        },
    )

    certificate = impact_claim_certificate(tmp_path)

    actions = certificate["recommended_next_actions"]
    assert "Disclose that the internal citation graph was audited and no dataset-internal edges were found." in actions
    assert not any("Add or materialize dataset-internal citation edges" in action for action in actions)


def test_certificate_does_not_block_model_score_table_on_legacy_readiness_block(
    tmp_path,
):
    _write_impact_scores(
        tmp_path / "impact_scores.csv",
        [
            {
                "paper_id": "p1",
                "title": "Evidence-supported paper",
                "final_impact_score": 80,
                "validity_status": "unvalidated",
                "validity_confidence": 0.25,
                "credible_evidence_tier": "high",
                "reportability_status": "exploratory",
                "dominant_score_family": "evidence",
                "missing_high_value_evidence": "",
            }
        ],
    )
    _write_json(
        tmp_path / "reports" / "project_readiness.json",
        {
            "decision": {
                "status": "blocked",
                "can_report": False,
                "reason": "Legacy hard gate failed.",
            }
        },
    )
    _write_json(
        tmp_path / "model_impact_report.json",
        {
            "summary": {
                "validated_impact_claim": {
                    "status": "unvalidated",
                    "can_claim_validated_impact_accuracy": False,
                    "confidence_ceiling": 0.25,
                    "reason": "No labels.",
                }
            }
        },
    )

    certificate = impact_claim_certificate(tmp_path)

    assert certificate["claim"]["status"] == "exploratory"
    assert certificate["claim"]["can_report_dataset_local_score"] is True
    assert "No papers are marked reportable" in certificate["claim"]["reason"]
    assert certificate["evidence"]["project_readiness_status"] == "blocked"


def test_certificate_ignores_full_pipeline_readiness_for_model_score_directory(
    tmp_path,
):
    _write_impact_scores(
        tmp_path / "impact_scores.csv",
        [
            {
                "paper_id": "p1",
                "title": "Evidence-supported paper",
                "final_impact_score": 80,
                "validity_status": "unvalidated",
                "validity_confidence": 0.25,
                "credible_evidence_tier": "high",
                "reportability_status": "exploratory",
                "dominant_score_family": "evidence",
                "missing_high_value_evidence": "",
            }
        ],
    )
    _write_json(
        tmp_path / "reports" / "project_readiness.json",
        {
            "decision": {
                "status": "blocked",
                "can_report": False,
                "reason": "Required artifacts or hard quality gates failed.",
            },
            "failed_checks": ["required_artifacts_present", "paper_ranking_present"],
            "artifacts": {"missing_required_count": 66},
        },
    )
    _write_json(
        tmp_path / "model_impact_report.json",
        {
            "summary": {
                "validated_impact_claim": {
                    "status": "unvalidated",
                    "can_claim_validated_impact_accuracy": False,
                    "confidence_ceiling": 0.25,
                    "reason": "No labels.",
                }
            }
        },
    )

    certificate = impact_claim_certificate(tmp_path)

    assert certificate["claim"]["status"] == "exploratory"
    assert "No papers are marked reportable" in certificate["claim"]["reason"]
    assert certificate["evidence"]["project_readiness_status"] is None
    assert certificate["evidence"]["evidence_level"] is None


def test_run_directory_certificate_uses_project_readiness_and_legacy_rankings(tmp_path):
    _write_impact_scores(
        tmp_path / "rankings" / "papers.csv",
        [
            {
                "rank": 1,
                "id": "r1",
                "title": "Reference",
                "final_score": 100,
            }
        ],
    )
    _write_json(
        tmp_path / "reports" / "project_readiness.json",
        {
            "claim_scope": "dataset-local content-aware impact ranking",
            "decision": {
                "status": "needs_improvement",
                "can_report": True,
                "reason": "Warnings must be disclosed.",
            },
            "quality": {
                "evidence_level": "proxy_only_needs_review",
                "external_evidence": {
                    "status": "usable_with_disclosure",
                    "can_claim_high_credibility": False,
                },
                "impact_validity": {
                    "status": "not_available",
                    "significant_vs_random": False,
                },
            },
            "warnings": ["External evidence coverage is incomplete."],
        },
    )

    certificate = impact_claim_certificate(tmp_path)

    assert certificate["claim"]["status"] == "exploratory"
    assert certificate["claim"]["can_report_dataset_local_score"] is True
    assert certificate["score_summary"]["paper_count"] == 1
    assert certificate["top_papers"][0]["paper_id"] == "r1"
    assert certificate["evidence"]["project_readiness_status"] == "needs_improvement"


def test_write_certificate_and_cli_emit_json_markdown(tmp_path):
    _write_impact_scores(
        tmp_path / "impact_scores.csv",
        [
            {
                "paper_id": "p1",
                "title": "Paper",
                "final_impact_score": 75,
                "validity_status": "unvalidated",
                "validity_confidence": 0.25,
                "credible_evidence_tier": "moderate",
                "reportability_status": "reportable",
                "dominant_score_family": "evidence",
                "missing_high_value_evidence": "",
            }
        ],
    )

    certificate = write_impact_claim_certificate(tmp_path, tmp_path / "certificate")

    assert certificate["claim"]["status"] == "evidence_supported"
    assert (tmp_path / "certificate" / "impact_claim_certificate.json").exists()
    assert (tmp_path / "certificate" / "impact_claim_certificate.md").exists()

    result = CliRunner().invoke(
        app,
        [
            "impact-claim-certificate",
            str(tmp_path),
            "--output",
            str(tmp_path / "cli-certificate"),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote impact claim certificate" in result.stdout
    assert (tmp_path / "cli-certificate" / "impact_claim_certificate.json").exists()
