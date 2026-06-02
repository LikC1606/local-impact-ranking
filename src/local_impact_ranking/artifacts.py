from __future__ import annotations

from pathlib import Path
from typing import Any


RUN_ARTIFACTS = [
    "rankings/papers.csv",
    "rankings/authors.csv",
    "rankings/institutions.csv",
    "rankings/fields.csv",
    "rankings/years.csv",
    "audit/judge_outputs.jsonl",
    "audit/prompts.jsonl",
    "audit/scoring_edges.csv",
    "baselines/citation_count.csv",
    "baselines/pagerank.csv",
    "baselines/field_year_normalized_citation.csv",
    "reports/validation.json",
    "reports/validation.md",
    "reports/artifact_index.json",
    "reports/evaluation.json",
    "reports/evaluation.md",
    "reports/gold_evaluation.json",
    "reports/gold_evaluation.md",
    "reports/gold_candidate_position.json",
    "reports/gold_candidate_position.md",
    "reports/gold_baseline_comparison.json",
    "reports/gold_baseline_comparison.md",
    "reports/context_coverage.json",
    "reports/data_readiness.json",
    "reports/external_evidence_readiness.json",
    "reports/external_evidence_readiness.md",
    "reports/external_evidence_paper_coverage.csv",
    "reports/impact_validity_evaluation.json",
    "reports/impact_validity_evaluation.md",
    "reports/impact_credibility_scorecard.json",
    "reports/impact_credibility_scorecard.md",
    "reports/citation_cutoff_audit.json",
    "reports/citation_cutoff_audit.md",
    "reports/baseline_comparison.json",
    "reports/proxy_evaluation.json",
    "reports/anomaly_report.json",
    "reports/scoring_config.json",
    "reports/stability_report.json",
    "reports/paper_stability_report.json",
    "reports/paper_stability_report.csv",
    "reports/paper_score_uncertainty_report.json",
    "reports/paper_score_uncertainty_report.csv",
    "reports/paper_explanations.json",
    "reports/entity_explanations.json",
    "reports/paper_evidence_report.json",
    "reports/paper_evidence_report.csv",
    "reports/score_cards.json",
    "reports/score_cards.md",
    "reports/entity_resolution_report.json",
    "reports/entity_resolution_report.md",
    "reports/mvp_summary.json",
    "reports/mvp_summary.md",
    "reports/benchmark_summary.json",
    "reports/benchmark_summary.md",
    "reports/project_readiness.json",
    "reports/project_readiness.md",
    "review/human_audit_sample.csv",
    "finetuning/format_sft.jsonl",
    "finetuning/proxy_positive_sft.jsonl",
    "finetuning/pairwise_preferences.jsonl",
    "finetuning/dpo_preferences.jsonl",
    "finetuning/context_hard_negative_dpo.jsonl",
    "finetuning/method_context_contrastive_dpo.jsonl",
    "finetuning/gold_hard_negative_dpo.jsonl",
    "finetuning/merged_dpo_preferences.jsonl",
    "finetuning/finetuning_readiness.json",
    "finetuning/finetuning_readiness.md",
    "finetuning/finetuning_preflight.json",
    "finetuning/finetuning_preflight.md",
    "finetuning/llamafactory_dataset_info.json",
    "finetuning/llamafactory_sft.yaml",
    "finetuning/llamafactory_dpo.yaml",
    "run_manifest.json",
]


OPTIONAL_RUN_ARTIFACTS = {
    "reports/impact_validity_evaluation.json",
    "reports/impact_validity_evaluation.md",
    "reports/impact_credibility_scorecard.json",
    "reports/impact_credibility_scorecard.md",
    "reports/citation_cutoff_audit.json",
    "reports/citation_cutoff_audit.md",
}


def artifact_index(
    root: str | Path,
    relative_paths: list[str],
    *,
    optional_paths: set[str] | None = None,
) -> dict[str, Any]:
    base = Path(root)
    optional = optional_paths or set()
    artifacts = []
    for relative_path in relative_paths:
        path = base / relative_path
        exists = path.exists()
        required = relative_path not in optional
        artifacts.append(
            {
                "path": relative_path,
                "exists": exists,
                "size_bytes": path.stat().st_size if exists and path.is_file() else 0,
                "required": required,
            }
        )
    required_artifacts = [artifact for artifact in artifacts if artifact["required"]]
    optional_artifacts = [artifact for artifact in artifacts if not artifact["required"]]
    missing_required_count = sum(
        1 for artifact in required_artifacts if not artifact["exists"]
    )
    missing_optional_count = sum(
        1 for artifact in optional_artifacts if not artifact["exists"]
    )
    return {
        "artifact_count": len(artifacts),
        "required_artifact_count": len(required_artifacts),
        "optional_artifact_count": len(optional_artifacts),
        "missing_count": missing_required_count,
        "missing_required_count": missing_required_count,
        "missing_optional_count": missing_optional_count,
        "artifacts": artifacts,
    }
