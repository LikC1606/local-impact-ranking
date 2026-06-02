from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_impact_ranking.citation_cutoff import write_citation_cutoff_audit
from local_impact_ranking.future_impact_labels import write_future_impact_labels
from local_impact_ranking.impact_validity_evaluation import (
    write_impact_validity_evaluation,
)


def write_impact_backtest(
    *,
    enriched_input: str | Path,
    cutoff_input: str | Path,
    rankings: str | Path,
    output_dir: str | Path,
    cutoff_year: int,
    future_start_year: int,
    future_end_year: int,
    score_column: str | None = None,
    label_column: str = "future_impact",
    label_normalization: str = "field_year_percentile",
    top_fraction: float = 0.1,
    min_label_delta: float = 0.0,
    bootstrap_samples: int = 200,
    permutation_samples: int = 200,
    random_seed: int = 0,
) -> dict[str, Any]:
    destination = Path(output_dir)
    reports_dir = destination / "reports"
    labels_dir = destination / "future-labels"
    reports_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cutoff_audit = write_citation_cutoff_audit(
        cutoff_input,
        reports_dir,
        cutoff_year=cutoff_year,
    )
    future_labels = write_future_impact_labels(
        enriched_input,
        labels_dir,
        start_year=future_start_year,
        end_year=future_end_year,
        label_column=label_column,
        normalization=label_normalization,
    )
    validity = write_impact_validity_evaluation(
        rankings,
        labels_dir / "future_impact_labels.jsonl",
        reports_dir,
        score_column=score_column,
        label_column=label_column,
        top_fraction=top_fraction,
        min_label_delta=min_label_delta,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        random_seed=random_seed,
    )
    summary = {
        "cutoff_year": cutoff_year,
        "future_start_year": future_start_year,
        "future_end_year": future_end_year,
        "label_normalization": future_labels["summary"]["normalization"],
        "cutoff_audit_status": cutoff_audit["decision"]["status"],
        "future_label_count": future_labels["summary"]["label_count"],
        "future_label_skipped_count": future_labels["summary"]["skipped_count"],
        "validity_status": validity["summary"]["decision"]["status"],
        "validity_matched_paper_count": validity["summary"]["matched_paper_count"],
        "validity_spearman_correlation": validity["summary"][
            "spearman_correlation"
        ],
        "validity_pairwise_accuracy": validity["summary"]["pairwise_accuracy"],
        "validity_significant_vs_random": validity.get(
            "statistical_significance",
            {},
        ).get("significant_vs_random", False),
    }
    manifest = {
        "summary": summary,
        "inputs": {
            "enriched_input": str(enriched_input),
            "cutoff_input": str(cutoff_input),
            "rankings": str(rankings),
        },
        "outputs": {
            "citation_cutoff_audit_json": "reports/citation_cutoff_audit.json",
            "citation_cutoff_audit_md": "reports/citation_cutoff_audit.md",
            "future_impact_labels_jsonl": "future-labels/future_impact_labels.jsonl",
            "future_impact_labels_csv": "future-labels/future_impact_labels.csv",
            "impact_validity_evaluation_json": (
                "reports/impact_validity_evaluation.json"
            ),
            "impact_validity_evaluation_md": "reports/impact_validity_evaluation.md",
            "impact_credibility_scorecard_json": (
                "reports/impact_credibility_scorecard.json"
            ),
            "impact_credibility_scorecard_md": (
                "reports/impact_credibility_scorecard.md"
            ),
        },
    }
    (destination / "impact_backtest_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
