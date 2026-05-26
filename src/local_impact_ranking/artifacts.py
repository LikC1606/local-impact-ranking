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
    "reports/evaluation.json",
    "reports/evaluation.md",
    "reports/gold_evaluation.json",
    "reports/gold_evaluation.md",
    "reports/context_coverage.json",
    "reports/data_readiness.json",
    "reports/baseline_comparison.json",
    "reports/proxy_evaluation.json",
    "reports/anomaly_report.json",
    "reports/scoring_config.json",
    "reports/stability_report.json",
    "reports/paper_explanations.json",
    "reports/entity_explanations.json",
    "reports/mvp_summary.json",
    "reports/mvp_summary.md",
    "reports/benchmark_summary.json",
    "reports/benchmark_summary.md",
    "review/human_audit_sample.csv",
    "finetuning/format_sft.jsonl",
    "finetuning/proxy_positive_sft.jsonl",
    "finetuning/pairwise_preferences.jsonl",
    "finetuning/dpo_preferences.jsonl",
    "finetuning/gold_hard_negative_dpo.jsonl",
    "finetuning/llamafactory_dataset_info.json",
    "finetuning/llamafactory_sft.yaml",
    "finetuning/llamafactory_dpo.yaml",
    "run_manifest.json",
]


def artifact_index(root: str | Path, relative_paths: list[str]) -> dict[str, Any]:
    base = Path(root)
    artifacts = []
    for relative_path in relative_paths:
        path = base / relative_path
        exists = path.exists()
        artifacts.append(
            {
                "path": relative_path,
                "exists": exists,
                "size_bytes": path.stat().st_size if exists and path.is_file() else 0,
            }
        )
    return {
        "artifact_count": len(artifacts),
        "missing_count": sum(1 for artifact in artifacts if not artifact["exists"]),
        "artifacts": artifacts,
    }
