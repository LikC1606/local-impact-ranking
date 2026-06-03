from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_impact_ranking.artifacts import RUN_ARTIFACTS, artifact_index


CLAIM_SCOPE = "dataset-local content-aware impact ranking"

EVIDENCE_FILES = {
    "paper_ranking": "rankings/papers.csv",
    "scoring_config": "reports/scoring_config.json",
    "judge_audit": "audit/judge_outputs.jsonl",
    "scoring_edges": "audit/scoring_edges.csv",
    "validation": "reports/validation.json",
    "evaluation": "reports/evaluation.json",
    "gold_evaluation": "reports/gold_evaluation.json",
    "gold_candidate_position": "reports/gold_candidate_position.json",
    "gold_baseline_comparison": "reports/gold_baseline_comparison.json",
    "proxy_evaluation": "reports/proxy_evaluation.json",
    "baseline_comparison": "reports/baseline_comparison.json",
    "impact_validity_evaluation": "reports/impact_validity_evaluation.json",
    "citation_cutoff_audit": "reports/citation_cutoff_audit.json",
    "external_evidence_readiness": "reports/external_evidence_readiness.json",
    "external_evidence_paper_coverage": "reports/external_evidence_paper_coverage.csv",
    "citation_completeness_audit": "reports/citation_completeness_audit.json",
    "citation_completeness_papers": "reports/citation_completeness_papers.csv",
    "anomaly_report": "reports/anomaly_report.json",
    "stability_report": "reports/stability_report.json",
    "paper_score_uncertainty": "reports/paper_score_uncertainty_report.json",
    "mvp_summary": "reports/mvp_summary.md",
    "run_manifest": "run_manifest.json",
}


def project_readiness_report(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    required_paths = required_artifacts_for_readiness()
    artifacts = artifact_index(root, required_paths)
    validation = _read_json(root / "reports" / "validation.json")
    evaluation = _read_json(root / "reports" / "evaluation.json")
    data_readiness = _read_json(root / "reports" / "data_readiness.json")
    proxy = _read_json(root / "reports" / "proxy_evaluation.json")
    gold = _read_json(root / "reports" / "gold_evaluation.json")
    gold_position = _read_json(root / "reports" / "gold_candidate_position.json")
    gold_baseline = _read_json(root / "reports" / "gold_baseline_comparison.json")
    stability = _read_json(root / "reports" / "stability_report.json")
    anomaly = _read_json(root / "reports" / "anomaly_report.json")
    scoring = _read_json(root / "reports" / "scoring_config.json")
    finetuning = _read_json(root / "finetuning" / "finetuning_preflight.json")
    baseline = _read_json(root / "reports" / "baseline_comparison.json")
    impact_validity = _read_json(root / "reports" / "impact_validity_evaluation.json")
    citation_cutoff_audit = _read_json(root / "reports" / "citation_cutoff_audit.json")
    external_evidence = _read_json(root / "reports" / "external_evidence_readiness.json")
    citation_completeness = _read_json(
        root / "reports" / "citation_completeness_audit.json"
    )
    manifest = _read_json(root / "run_manifest.json")

    checks = _evaluate_checks(
        artifacts=artifacts,
        validation=validation,
        evaluation=evaluation,
        data_readiness=data_readiness,
        proxy=proxy,
        gold=gold,
        gold_baseline=gold_baseline,
        stability=stability,
        anomaly=anomaly,
        scoring=scoring,
        impact_validity=impact_validity,
        citation_cutoff_audit=citation_cutoff_audit,
        external_evidence=external_evidence,
    )
    impact_validity_quality = _impact_validity_quality(impact_validity)
    citation_cutoff_audit_quality = _citation_cutoff_audit_quality(
        citation_cutoff_audit
    )
    decision = _decision(checks)
    return {
        "run_dir": str(root),
        "claim_scope": CLAIM_SCOPE,
        "decision": decision,
        "passed_checks": checks["passed_checks"],
        "failed_checks": checks["failed_checks"],
        "warning_checks": checks["warning_checks"],
        "blockers": checks["blockers"],
        "warnings": checks["warnings"],
        "quality": {
            "evidence_level": _evidence_level(gold, proxy),
            "data": _data_quality(data_readiness, validation),
            "judge": _judge_quality(evaluation),
            "gold": _gold_quality(gold),
            "gold_candidate_position": _gold_candidate_position_quality(gold_position),
            "gold_baseline_comparison": _gold_baseline_comparison_quality(gold_baseline),
            "proxy": _proxy_quality(proxy),
            "impact_validity": impact_validity_quality,
            "citation_cutoff_audit": citation_cutoff_audit_quality,
            "external_evidence": _external_evidence_quality(external_evidence),
            "citation_completeness": _citation_completeness_quality(
                citation_completeness
            ),
            "stability": _stability_quality(stability),
            "anomaly": _anomaly_quality(anomaly),
            "scoring": {
                "version": scoring.get("version"),
                "credit_mode": scoring.get("credit_mode"),
            },
            "baseline": baseline,
            "finetuning": finetuning.get("decision", {}),
        },
        "artifacts": artifacts,
        "evidence_files": dict(EVIDENCE_FILES),
        "reproducibility": _reproducibility_summary(manifest),
        "recommended_next_actions": _recommended_next_actions(
            checks,
            gold,
            finetuning,
            external_evidence,
            impact_validity,
            citation_cutoff_audit,
        ),
        "limitations": [
            "This report supports dataset-local impact reporting only.",
            "Gold metrics evaluate benchmark reference selection when labels exist; they are not global impact truth.",
            "Proxy metrics and baselines improve auditability but do not replace human validation.",
        ],
    }


def write_project_readiness_report(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    report = project_readiness_report(root)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "project_readiness.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "project_readiness.md").write_text(
        render_project_readiness_markdown(report),
        encoding="utf-8",
    )
    return report


def required_artifacts_for_readiness() -> list[str]:
    excluded = {
        "reports/impact_validity_evaluation.json",
        "reports/impact_validity_evaluation.md",
        "reports/citation_cutoff_audit.json",
        "reports/citation_cutoff_audit.md",
        "reports/project_readiness.json",
        "reports/project_readiness.md",
    }
    return [path for path in RUN_ARTIFACTS if path not in excluded]


def render_project_readiness_markdown(report: dict[str, Any]) -> str:
    decision = report.get("decision", {})
    quality = report.get("quality", {})
    judge = quality.get("judge", {})
    data = quality.get("data", {})
    gold = quality.get("gold", {})
    proxy = quality.get("proxy", {})
    impact_validity = quality.get("impact_validity", {})
    citation_cutoff_audit = quality.get("citation_cutoff_audit", {})
    gold_position = quality.get("gold_candidate_position", {})
    gold_baseline = quality.get("gold_baseline_comparison", {})
    external_evidence = quality.get("external_evidence", {})
    citation_completeness = quality.get("citation_completeness", {})
    stability = quality.get("stability", {})
    anomaly = quality.get("anomaly", {})
    scoring = quality.get("scoring", {})
    finetuning = quality.get("finetuning", {})
    evidence_level = quality.get("evidence_level", "none")
    reproducibility = report.get("reproducibility", {})
    lines = [
        "# Project Readiness",
        "",
        "## Decision",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| status | {_text(decision.get('status'))} |",
        f"| can_report | {_yes_no(decision.get('can_report'))} |",
        f"| claim_scope | {_text(report.get('claim_scope'))} |",
        f"| reason | {_text(decision.get('reason'))} |",
        "",
        "## Quality Signals",
        "",
        "| Signal | Value |",
        "| --- | ---: |",
        f"| paper_count | {data.get('paper_count', 0)} |",
        f"| reference_count | {data.get('reference_count', 0)} |",
        f"| evidence_level | {_text(evidence_level)} |",
        f"| validation_issue_count | {data.get('validation_issue_count', 0)} |",
        f"| judge_results | {judge.get('judge_results', 0)} |",
        f"| parse_success_rate | {_number(judge.get('parse_success_rate'))} |",
        f"| invalid_selection_rate | {_number(judge.get('invalid_selection_rate'))} |",
        f"| top_k_compliance_rate | {_number(judge.get('top_k_compliance_rate'))} |",
        f"| labeled_targets | {gold.get('labeled_targets', 0)} |",
        f"| hit_rate | {_number(gold.get('hit_rate'))} |",
        f"| mean_ndcg | {_number(gold.get('mean_ndcg'))} |",
        f"| gold_position_bias_risk | {_text(gold_position.get('position_bias_risk'))} |",
        f"| gold_front_position_share | {_number(gold_position.get('front_position_share'))} |",
        f"| judge_beats_gold_baselines | {_yes_no(gold_baseline.get('judge_beats_all_baselines'))} |",
        f"| judge_ndcg_delta_vs_best_baseline | {_number(gold_baseline.get('mean_ndcg_delta'))} |",
        f"| judge_ndcg_delta_ci95 | {_ci(gold_baseline.get('mean_ndcg_delta_ci95'))} |",
        f"| statistically_positive_ndcg_delta | {_yes_no(gold_baseline.get('statistically_positive_ndcg_delta'))} |",
        f"| impact_validity_status | {_text(impact_validity.get('status'))} |",
        f"| impact_validity_match_rate | {_number(impact_validity.get('score_label_match_rate'))} |",
        f"| impact_validity_spearman | {_number(impact_validity.get('spearman_correlation'))} |",
        f"| impact_validity_pairwise_accuracy | {_number(impact_validity.get('pairwise_accuracy'))} |",
        f"| impact_validity_ndcg_at_top_k | {_number(impact_validity.get('ndcg_at_top_k'))} |",
        f"| impact_validity_significant_vs_random | {_yes_no(impact_validity.get('significant_vs_random'))} |",
        f"| impact_validity_p_value | {_number(impact_validity.get('p_value'))} |",
        f"| impact_credibility_status | {_text(impact_validity.get('credibility_scorecard_status'))} |",
        f"| impact_credibility_grade | {_text(impact_validity.get('credibility_grade'))} |",
        f"| high_validation_credibility | {_yes_no(impact_validity.get('can_claim_high_validation_credibility'))} |",
        f"| citation_cutoff_audit_status | {_text(citation_cutoff_audit.get('status'))} |",
        f"| citation_cutoff_future_year_count | {citation_cutoff_audit.get('future_year_count', 0)} |",
        f"| citation_cutoff_metadata_coverage | {_number(citation_cutoff_audit.get('citation_cutoff_year_coverage'))} |",
        f"| external_evidence_status | {_text(external_evidence.get('status'))} |",
        f"| can_claim_high_credibility | {_yes_no(external_evidence.get('can_claim_high_credibility'))} |",
        f"| external_query_id_coverage | {_number(external_evidence.get('external_query_id_coverage'))} |",
        f"| openalex_enriched_coverage | {_number(external_evidence.get('openalex_enriched_coverage'))} |",
        f"| opencitations_enriched_coverage | {_number(external_evidence.get('opencitations_enriched_coverage'))} |",
        f"| field_year_percentile_coverage | {_number(external_evidence.get('field_year_percentile_coverage'))} |",
        f"| materialized_internal_edge_count | {external_evidence.get('materialized_internal_edge_count', 0)} |",
        f"| citation_evidence_coverage | {_number(citation_completeness.get('citation_evidence_coverage'))} |",
        f"| citation_multi_source_coverage | {_number(citation_completeness.get('multi_source_coverage'))} |",
        f"| citation_mean_completeness_score | {_number(citation_completeness.get('mean_completeness_score'))} |",
        f"| citation_source_conflict_count | {citation_completeness.get('papers_with_source_conflict', 0)} |",
        f"| citation_papers_without_query_ids | {citation_completeness.get('papers_without_query_ids', 0)} |",
        f"| unsupported_selection_rate | {_number(proxy.get('unsupported_selection_rate'))} |",
        f"| leave_one_target_out_average_retention_rate | {_number(stability.get('leave_one_target_out_average_retention_rate'))} |",
        f"| highest_paper_single_edge_dominance_share | {_number(anomaly.get('highest_paper_single_edge_dominance_share'))} |",
        f"| self_citation_risk_level | {_text(anomaly.get('self_citation_risk_level'))} |",
        f"| self_citation_credit_share | {_number(anomaly.get('self_citation_credit_share'))} |",
        "",
        "## Scoring And Training",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| scoring_version | {_text(scoring.get('version'))} |",
        f"| credit_mode | {_text(scoring.get('credit_mode'))} |",
        f"| finetuning_next_step | {_text(finetuning.get('recommended_next_step'))} |",
        f"| can_start_dpo_smoke | {_yes_no(finetuning.get('can_start_dpo_smoke'))} |",
        "",
        "## Reproducibility",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| created_at | {_text(reproducibility.get('created_at'))} |",
        f"| input_path | {_text(reproducibility.get('input_path'))} |",
        f"| input_sha256 | {_text(reproducibility.get('input_sha256'))} |",
        f"| normalized_sha256 | {_text(reproducibility.get('normalized_sha256'))} |",
        f"| normalized_record_count | {_text(reproducibility.get('normalized_record_count'))} |",
        f"| top_k | {_text(reproducibility.get('top_k'))} |",
        f"| limit | {_text(reproducibility.get('limit'))} |",
        f"| judge_name | {_text(reproducibility.get('judge_name'))} |",
        f"| judge_model | {_text(reproducibility.get('judge_model'))} |",
        f"| prompt_version | {_text(reproducibility.get('prompt_version'))} |",
        f"| scoring_version | {_text(reproducibility.get('scoring_version'))} |",
        f"| git_revision | {_text(reproducibility.get('git_revision'))} |",
        f"| git_dirty | {_yes_no(reproducibility.get('git_dirty'))} |",
        "",
        "## Checks",
        "",
        "### Passed",
        "",
        *_list_rows(report.get("passed_checks", [])),
        "",
        "### Failed",
        "",
        *_list_rows(report.get("failed_checks", [])),
        "",
        "### Warnings",
        "",
        *_list_rows(report.get("warning_checks", [])),
        "",
        "## Blockers",
        "",
        *_list_rows(report.get("blockers", [])),
        "",
        "## Recommended Next Actions",
        "",
        *_list_rows(report.get("recommended_next_actions", [])),
        "",
        "## Evidence Files",
        "",
        "| Evidence | Path |",
        "| --- | --- |",
    ]
    for name, path in report.get("evidence_files", {}).items():
        lines.append(f"| {_text(name)} | `{_text(path)}` |")
    lines.extend(["", "## Limitations", ""])
    lines.extend(_list_rows(report.get("limitations", [])))
    lines.append("")
    return "\n".join(lines)


def _evaluate_checks(
    *,
    artifacts: dict[str, Any],
    validation: dict[str, Any],
    evaluation: dict[str, Any],
    data_readiness: dict[str, Any],
    proxy: dict[str, Any],
    gold: dict[str, Any],
    gold_baseline: dict[str, Any],
    stability: dict[str, Any],
    anomaly: dict[str, Any],
    scoring: dict[str, Any],
    impact_validity: dict[str, Any],
    citation_cutoff_audit: dict[str, Any],
    external_evidence: dict[str, Any],
) -> dict[str, list[str]]:
    passed_checks: list[str] = []
    failed_checks: list[str] = []
    warning_checks: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []

    def require(name: str, condition: bool, blocker: str) -> None:
        if condition:
            passed_checks.append(name)
        else:
            failed_checks.append(name)
            blockers.append(blocker)

    def warn(name: str, condition: bool, warning: str) -> None:
        if condition:
            passed_checks.append(name)
        else:
            warning_checks.append(name)
            warnings.append(warning)

    missing_count = int(artifacts.get("missing_count", 0) or 0)
    require(
        "required_artifacts_present",
        missing_count == 0,
        f"Missing required artifacts: {missing_count}",
    )
    validation_summary = validation.get("summary", {})
    require(
        "validation_can_run",
        bool(validation_summary.get("can_run")),
        "Validation failed; fix input data errors before reporting.",
    )
    require(
        "judge_outputs_present",
        int(evaluation.get("judge_results", 0) or 0) > 0,
        "No judge results are available.",
    )
    require(
        "parse_success_rate_high",
        _float(evaluation.get("parse_success_rate")) >= 0.98,
        "Judge parse success rate is below 0.98.",
    )
    require(
        "invalid_selection_rate_zero",
        _float(evaluation.get("invalid_selection_rate")) == 0.0,
        "Judge selected at least one invalid reference ID.",
    )
    require(
        "top_k_compliance_exact",
        _float(evaluation.get("top_k_compliance_rate")) >= 1.0,
        "Judge outputs violate the configured top-k limit.",
    )
    require(
        "scoring_config_present",
        bool(scoring.get("version")) and bool(scoring.get("credit_mode")),
        "Scoring config is missing version or credit mode.",
    )
    require(
        "paper_ranking_present",
        _artifact_exists(artifacts, "rankings/papers.csv"),
        "Paper ranking CSV is missing.",
    )

    ranking_scope = data_readiness.get("ranking_scope", {})
    warn(
        "paper_ranking_scope_formal",
        ranking_scope.get("paper_ranking") == "formal",
        "Paper ranking scope is exploratory; improve ID/title/reference coverage.",
    )
    warn(
        "author_ranking_scope_formal",
        ranking_scope.get("author_ranking") == "formal",
        "Author ranking is exploratory because author metadata coverage is low.",
    )
    warn(
        "institution_ranking_scope_formal",
        ranking_scope.get("institution_ranking") == "formal",
        "Institution ranking is exploratory because affiliation coverage is low.",
    )
    warn(
        "gold_or_proxy_evidence_available",
        _has_gold_evidence(gold) or _has_proxy_evidence(proxy),
        "No gold labeled targets are available; report proxy and baseline evidence only.",
    )
    warn(
        "proxy_support_reasonable",
        _float(proxy.get("unsupported_selection_rate")) <= 0.4,
        "Unsupported selection rate is high; inspect selected-reference evidence.",
    )
    if _has_gold_evidence(gold) and gold_baseline:
        judge_vs_best = gold_baseline.get("judge_vs_best_baseline", {})
        warn(
            "gold_baseline_delta_statistically_positive",
            bool(judge_vs_best.get("statistically_positive_ndcg_delta", False)),
            "Judge beats the best simple baseline in mean, but the bootstrap NDCG delta is not statistically positive.",
        )
    external_decision = external_evidence.get("decision", {})
    warn(
        "external_evidence_ready_for_high_credibility",
        bool(external_decision.get("can_claim_high_credibility")),
        "External evidence coverage is incomplete; report scores with external-evidence disclosure and avoid high-credibility claims.",
    )
    if _has_impact_validity_evidence(impact_validity):
        impact_decision = impact_validity.get("summary", {}).get("decision", {})
        significance = impact_validity.get("statistical_significance", {})
        warn(
            "impact_validity_statistically_supported",
            impact_decision.get("status") in {
                "externally_supported",
                "usable_with_disclosure",
            }
            and bool(significance.get("significant_vs_random", False)),
            "Impact validity results are not statistically supported against the random baseline; do not claim validated impact accuracy.",
        )
        warn(
            "impact_validity_match_coverage_reasonable",
            _float(
                impact_validity.get("summary", {}).get(
                    "score_label_match_rate",
                    1.0,
                )
            )
            >= 0.5,
            "Impact validity label coverage is low; matched labels cover less than half of the scored ranking table.",
        )
        cutoff_decision = citation_cutoff_audit.get("decision", {})
        require(
            "citation_cutoff_audit_no_future_leakage",
            cutoff_decision.get("status") != "failed",
            "Citation cutoff audit found post-cutoff citation years; regenerate cutoff scoring input before reporting future-window validity.",
        )
        warn(
            "citation_cutoff_audit_available_for_impact_validity",
            bool(citation_cutoff_audit) and bool(cutoff_decision),
            "Impact validity evidence is present, but citation cutoff audit is missing; run audit-citation-cutoff before claiming future-window validity.",
        )
        if cutoff_decision and cutoff_decision.get("status") != "failed":
            warn(
                "citation_cutoff_audit_passed",
                cutoff_decision.get("status") == "passed",
                "Citation cutoff audit did not fail, but cutoff metadata is incomplete; disclose the cutoff-audit warning.",
            )

    leave_one_out = stability.get("leave_one_target_out", {})
    warn(
        "stability_reasonable",
        _float(leave_one_out.get("average_top_n_retention_rate")) >= 0.5,
        "Leave-one-target-out stability is weak; rankings may be sensitive to corpus coverage.",
    )
    dominance = _highest_paper_single_edge_share(anomaly)
    warn(
        "single_edge_dominance_not_extreme",
        dominance < 0.75,
        "Top paper scores are dominated by one citation edge; report as fragile.",
    )
    self_citation = anomaly.get("self_citation_risk", {})
    warn(
        "self_citation_risk_not_high",
        self_citation.get("risk_level") != "high",
        "Selected-reference credit has high self-citation concentration; disclose this before reporting the ranking.",
    )
    return {
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "warning_checks": warning_checks,
        "blockers": blockers,
        "warnings": warnings,
    }


def _decision(checks: dict[str, list[str]]) -> dict[str, Any]:
    if checks["failed_checks"]:
        return {
            "status": "blocked",
            "can_report": False,
            "reason": "Required artifacts or hard quality gates failed.",
        }
    if checks["warning_checks"]:
        return {
            "status": "needs_improvement",
            "can_report": True,
            "reason": "Core run is usable, but warnings must be disclosed.",
        }
    return {
        "status": "reportable",
        "can_report": True,
        "reason": "Core artifacts and quality gates passed.",
    }


def _recommended_next_actions(
    checks: dict[str, list[str]],
    gold: dict[str, Any],
    finetuning: dict[str, Any],
    external_evidence: dict[str, Any],
    impact_validity: dict[str, Any],
    citation_cutoff_audit: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    failed = set(checks["failed_checks"])
    warnings = set(checks["warning_checks"])
    if "required_artifacts_present" in failed:
        actions.append("Re-run the pipeline and regenerate missing artifacts.")
    if "validation_can_run" in failed:
        actions.append("Fix validation errors in the input dataset before reporting.")
    if {"parse_success_rate_high", "invalid_selection_rate_zero", "top_k_compliance_exact"} & failed:
        actions.append("Improve or constrain the judge until output reliability gates pass.")
    if int(gold.get("labeled_targets", 0) or 0) <= 0:
        actions.append(
            "Add benchmark gold labels when possible, such as PreScience positives."
        )
    if "proxy_support_reasonable" in warnings:
        actions.append("Inspect unsupported selections and add citation contexts or stronger candidates.")
    if "gold_baseline_delta_statistically_positive" in warnings:
        actions.append("Increase held-out benchmark size or improve the judge until baseline delta CI is positive.")
    if "stability_reasonable" in warnings:
        actions.append("Run on a larger or more representative corpus to improve stability.")
    if "single_edge_dominance_not_extreme" in warnings:
        actions.append("Review single-edge dominated papers before using them in a public ranking.")
    if "external_evidence_ready_for_high_credibility" in warnings:
        actions.extend(_external_evidence_next_actions(external_evidence))
    if "impact_validity_statistically_supported" in warnings:
        actions.append(
            "Do not claim validated impact accuracy; report the score as dataset-local exploratory evidence until future-window validity beats the random baseline."
        )
    if "impact_validity_match_coverage_reasonable" in warnings:
        actions.append(
            "Improve entity matching or label coverage before using the validity report as high-credibility evidence."
        )
    if "citation_cutoff_audit_no_future_leakage" in failed:
        actions.append(
            "Regenerate the scoring input with apply-citation-cutoff, then rerun audit-citation-cutoff before using future-window validity claims."
        )
    if "citation_cutoff_audit_available_for_impact_validity" in warnings:
        actions.append(
            "Run audit-citation-cutoff on the cutoff scoring input and attach citation_cutoff_audit.json/md to the run reports."
        )
    if "citation_cutoff_audit_passed" in warnings:
        actions.append(
            "Improve cutoff metadata coverage or disclose the citation cutoff audit warning with the validity report."
        )
    decision = finetuning.get("decision", {})
    if decision.get("can_start_dpo_smoke"):
        actions.append("Run a small held-out DPO smoke only after preserving train/dev/test splits.")
    if not actions:
        actions.append("Use the run artifacts and claim boundaries in the final report.")
    return actions


def _artifact_exists(artifacts: dict[str, Any], relative_path: str) -> bool:
    return any(
        artifact.get("path") == relative_path and artifact.get("exists")
        for artifact in artifacts.get("artifacts", [])
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _data_quality(
    data_readiness: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    validation_summary = validation.get("summary", {})
    return {
        "paper_count": data_readiness.get("paper_count", 0),
        "reference_count": data_readiness.get("reference_count", 0),
        "ranking_scope": data_readiness.get("ranking_scope", {}),
        "validation_issue_count": validation_summary.get("issue_count", 0),
        "validation_can_run": validation_summary.get("can_run", False),
    }


def _judge_quality(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "judge_results": evaluation.get("judge_results", 0),
        "parse_success_rate": evaluation.get("parse_success_rate", 0.0),
        "error_rate": evaluation.get("error_rate", 0.0),
        "invalid_selection_rate": evaluation.get("invalid_selection_rate", 0.0),
        "empty_selection_rate": evaluation.get("empty_selection_rate", 0.0),
        "duplicate_selection_rate": evaluation.get("duplicate_selection_rate", 0.0),
        "top_k_compliance_rate": evaluation.get("top_k_compliance_rate", 0.0),
    }


def _gold_quality(gold: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_source": gold.get("label_source"),
        "labeled_targets": gold.get("labeled_targets", 0),
        "hit_rate": gold.get("hit_rate", 0.0),
        "selection_precision": gold.get("selection_precision", 0.0),
        "selection_recall": gold.get("selection_recall", 0.0),
        "mean_reciprocal_rank": gold.get("mean_reciprocal_rank", 0.0),
        "mean_ndcg": gold.get("mean_ndcg", 0.0),
    }


def _gold_candidate_position_quality(gold_position: dict[str, Any]) -> dict[str, Any]:
    return {
        "labeled_targets": gold_position.get("labeled_targets", 0),
        "candidate_ordering": gold_position.get("candidate_ordering"),
        "front_position_share": gold_position.get("front_position_share", 0.0),
        "mean_best_gold_position": gold_position.get("mean_best_gold_position", 0.0),
        "mean_gold_position": gold_position.get("mean_gold_position", 0.0),
        "position_bias_risk": gold_position.get("position_bias_risk"),
    }


def _gold_baseline_comparison_quality(gold_baseline: dict[str, Any]) -> dict[str, Any]:
    judge_vs_best = gold_baseline.get("judge_vs_best_baseline", {})
    decision = gold_baseline.get("decision", {})
    return {
        "best_baseline_by_mean_ndcg": gold_baseline.get(
            "best_baseline_by_mean_ndcg"
        ),
        "hit_rate_delta": judge_vs_best.get("hit_rate_delta", 0.0),
        "mean_ndcg_delta": judge_vs_best.get("mean_ndcg_delta", 0.0),
        "mean_ndcg_delta_ci95": judge_vs_best.get(
            "mean_ndcg_delta_ci95",
            [0.0, 0.0],
        ),
        "statistically_positive_ndcg_delta": judge_vs_best.get(
            "statistically_positive_ndcg_delta",
            False,
        ),
        "judge_beats_all_baselines": decision.get(
            "judge_beats_all_baselines",
            False,
        ),
        "needs_model_or_prompt_improvement": decision.get(
            "needs_model_or_prompt_improvement",
            False,
        ),
    }


def _proxy_quality(proxy: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_references": proxy.get("selected_references", 0),
        "unsupported_selection_rate": proxy.get("unsupported_selection_rate", 0.0),
        "context_supported_selection_rate": proxy.get(
            "context_supported_selection_rate",
            0.0,
        ),
        "method_context_selection_rate": proxy.get("method_context_selection_rate", 0.0),
        "bibliometric_top_n_support_rate": proxy.get(
            "bibliometric_top_n_support_rate",
            0.0,
        ),
    }


def _impact_validity_quality(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    decision = summary.get("decision", {})
    significance = report.get("statistical_significance", {})
    scorecard = report.get("credibility_scorecard", {})
    return {
        "status": decision.get("status", "not_available"),
        "reason": decision.get("reason", ""),
        "matched_paper_count": summary.get("matched_paper_count", 0),
        "ranking_paper_count": summary.get("ranking_paper_count", 0),
        "label_paper_count": summary.get("label_paper_count", 0),
        "score_label_match_rate": summary.get("score_label_match_rate", 0.0),
        "label_match_rate": summary.get("label_match_rate", 0.0),
        "spearman_correlation": summary.get("spearman_correlation", 0.0),
        "kendall_tau_b": summary.get("kendall_tau_b", 0.0),
        "pairwise_accuracy": summary.get("pairwise_accuracy", 0.0),
        "ndcg_at_top_k": summary.get("ndcg_at_top_k", 0.0),
        "top_k_precision": summary.get("top_k_precision", 0.0),
        "top_k_recall": summary.get("top_k_recall", 0.0),
        "significance_status": significance.get("status", "not_available"),
        "significant_vs_random": bool(
            significance.get("significant_vs_random", False)
        ),
        "tested_metric": significance.get("tested_metric"),
        "p_value": significance.get("p_value", 1.0),
        "permutation_null_mean": significance.get("permutation_null_mean", 0.0),
        "credibility_scorecard_status": scorecard.get("status"),
        "credibility_grade": scorecard.get("grade"),
        "can_claim_high_validation_credibility": bool(
            scorecard.get("can_claim_high_validation_credibility", False)
        ),
    }


def _citation_cutoff_audit_quality(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    decision = report.get("decision", {})
    return {
        "status": decision.get("status", "not_available"),
        "reason": decision.get("reason", ""),
        "paper_count": summary.get("paper_count", 0),
        "cutoff_year": summary.get("cutoff_year"),
        "papers_with_yearly_citation_history": summary.get(
            "papers_with_yearly_citation_history",
            0,
        ),
        "papers_with_future_years": summary.get("papers_with_future_years", 0),
        "future_year_count": summary.get("future_year_count", 0),
        "citation_cutoff_year_coverage": summary.get(
            "citation_cutoff_year_coverage",
            0.0,
        ),
    }


def _external_evidence_quality(external_evidence: dict[str, Any]) -> dict[str, Any]:
    decision = external_evidence.get("decision", {})
    identifiers = external_evidence.get("identifier_coverage", {})
    sources = external_evidence.get("source_coverage", {})
    signals = external_evidence.get("signal_coverage", {})
    graph = external_evidence.get("graph_evidence", {})
    return {
        "status": decision.get("status", "unknown"),
        "can_claim_high_credibility": bool(
            decision.get("can_claim_high_credibility", False)
        ),
        "reason": decision.get("reason", ""),
        "external_query_id_coverage": identifiers.get(
            "external_query_id_coverage",
            0.0,
        ),
        "doi_coverage": identifiers.get("doi_coverage", 0.0),
        "openalex_id_coverage": identifiers.get("openalex_id_coverage", 0.0),
        "openalex_enriched_coverage": sources.get("openalex_enriched_coverage", 0.0),
        "opencitations_enriched_coverage": sources.get(
            "opencitations_enriched_coverage",
            0.0,
        ),
        "citation_count_coverage": signals.get("citation_count_coverage", 0.0),
        "field_year_percentile_coverage": signals.get(
            "field_year_percentile_coverage",
            0.0,
        ),
        "internal_reference_edge_count": graph.get("internal_reference_edge_count", 0),
        "materialized_internal_edge_count": graph.get(
            "materialized_internal_edge_count",
            0,
        ),
    }


def _citation_completeness_quality(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    return {
        "citation_evidence_coverage": summary.get("citation_evidence_coverage", 0.0),
        "multi_source_coverage": summary.get("multi_source_coverage", 0.0),
        "mean_completeness_score": summary.get("mean_completeness_score", 0.0),
        "positive_citation_evidence_coverage": summary.get(
            "positive_citation_evidence_coverage",
            0.0,
        ),
        "papers_without_query_ids": summary.get("papers_without_query_ids", 0),
        "papers_without_numeric_citation_evidence": summary.get(
            "papers_without_numeric_citation_evidence",
            0,
        ),
        "papers_with_single_source_only": summary.get(
            "papers_with_single_source_only",
            0,
        ),
        "papers_with_source_conflict": summary.get("papers_with_source_conflict", 0),
        "papers_with_confirmed_multi_source_zero": summary.get(
            "papers_with_confirmed_multi_source_zero",
            0,
        ),
    }


def _external_evidence_next_actions(external_evidence: dict[str, Any]) -> list[str]:
    recommendations = external_evidence.get("recommendations")
    if isinstance(recommendations, list) and recommendations:
        return [str(item) for item in recommendations]
    return [
        "Run enrich-openalex and enrich-opencitations, then regenerate external evidence readiness.",
    ]


def _reproducibility_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    input_info = manifest.get("input", {})
    run = manifest.get("run", {})
    judge = manifest.get("judge", {})
    prompt = manifest.get("prompt", {})
    scoring = manifest.get("scoring", {})
    code = manifest.get("code", {})
    return {
        "created_at": manifest.get("created_at"),
        "input_path": input_info.get("path"),
        "input_sha256": input_info.get("sha256"),
        "normalized_sha256": input_info.get("normalized_sha256"),
        "normalized_record_count": input_info.get("normalized_record_count"),
        "top_k": run.get("top_k"),
        "limit": run.get("limit"),
        "credit_mode": run.get("credit_mode"),
        "judge_audit": run.get("judge_audit"),
        "judge_name": judge.get("name"),
        "judge_model": judge.get("model"),
        "judge_temperature": judge.get("temperature"),
        "judge_max_tokens": judge.get("max_tokens"),
        "serving_backend": judge.get("serving_backend"),
        "model_path": judge.get("model_path"),
        "prompt_version": prompt.get("version"),
        "scoring_version": scoring.get("version"),
        "git_revision": code.get("git_revision"),
        "git_dirty": code.get("git_dirty"),
    }


def _evidence_level(gold: dict[str, Any], proxy: dict[str, Any]) -> str:
    if _has_gold_evidence(gold):
        return "gold_validated"
    if _has_proxy_evidence(proxy):
        if _float(proxy.get("unsupported_selection_rate")) <= 0.4:
            return "proxy_supported"
        return "proxy_only_needs_review"
    return "no_external_or_proxy_evidence"


def _has_gold_evidence(gold: dict[str, Any]) -> bool:
    return int(gold.get("labeled_targets", 0) or 0) > 0


def _has_proxy_evidence(proxy: dict[str, Any]) -> bool:
    if int(proxy.get("selected_references", 0) or 0) > 0:
        return True
    return any(
        _float(proxy.get(key)) > 0.0
        for key in (
            "context_supported_selection_rate",
            "method_context_selection_rate",
            "bibliometric_top_n_support_rate",
        )
    )


def _has_impact_validity_evidence(report: dict[str, Any]) -> bool:
    summary = report.get("summary", {})
    if int(summary.get("matched_paper_count", 0) or 0) > 0:
        return True
    decision = summary.get("decision", {})
    return bool(decision.get("status")) and decision.get("status") != "not_available"


def _stability_quality(stability: dict[str, Any]) -> dict[str, Any]:
    leave_one_out = stability.get("leave_one_target_out", {})
    return {
        "leave_one_target_out_average_retention_rate": leave_one_out.get(
            "average_top_n_retention_rate",
            0.0,
        ),
        "leave_one_target_out_min_retention_rate": leave_one_out.get(
            "min_top_n_retention_rate",
            0.0,
        ),
    }


def _anomaly_quality(anomaly: dict[str, Any]) -> dict[str, Any]:
    missing = anomaly.get("missing_metadata", {})
    self_citation = anomaly.get("self_citation_risk", {})
    return {
        "references_missing_title_count": missing.get("references_missing_title_count", 0),
        "references_missing_author_count": missing.get("references_missing_author_count", 0),
        "references_missing_affiliation_count": missing.get(
            "references_missing_affiliation_count",
            0,
        ),
        "highest_paper_single_edge_dominance_share": _highest_paper_single_edge_share(
            anomaly
        ),
        "self_citation_risk_level": self_citation.get("risk_level", "not_available"),
        "self_citation_edge_rate": self_citation.get("self_citation_edge_rate", 0.0),
        "self_citation_credit_share": self_citation.get(
            "self_citation_credit_share",
            0.0,
        ),
    }


def _highest_paper_single_edge_share(anomaly: dict[str, Any]) -> float:
    rows = anomaly.get("score_dominance", {}).get("paper_single_edge_dominance", [])
    if not rows:
        return 0.0
    return _float(rows[0].get("max_single_edge_share"))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _number(value: Any) -> str:
    return f"{_float(value):.3f}"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _ci(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "[0.000, 0.000]"
    return f"[{_number(value[0])}, {_number(value[1])}]"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _list_rows(rows: list[Any]) -> list[str]:
    if not rows:
        return ["- none"]
    return [f"- {_text(row)}" for row in rows]
