from __future__ import annotations

import csv
import heapq
import json
import math
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

import httpx

from local_impact_ranking.external_evidence_readiness import (
    external_evidence_paper_coverage_rows,
    external_evidence_readiness_report,
    render_external_evidence_readiness_markdown,
    write_external_evidence_paper_coverage_csv,
)
from local_impact_ranking.impact_fusion import (
    EVIDENCE_FIRST_WEIGHTS,
    IMPACT_INDEX_V2_WEIGHTS,
    IMPACT_INDEX_V3_WEIGHTS,
    compute_impact_index_v2,
)
from local_impact_ranking.impact_claim_certificate import write_impact_claim_certificate
from local_impact_ranking.impact_validity_evaluation import (
    write_impact_validity_evaluation,
)
from local_impact_ranking.impact_taxonomy import ranking_dimension_coverage
from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.models import Paper
from local_impact_ranking.normalize import normalize_paper


PAIRWISE_ABSTRACT_CHAR_BUDGET = 1800
PAIRWISE_TITLE_CHAR_BUDGET = 240
PAIRWISE_SHORT_FIELD_CHAR_BUDGET = 120
DIRECT_ABSTRACT_CHAR_BUDGET = 3000
DIRECT_TITLE_CHAR_BUDGET = 360
DIRECT_SHORT_FIELD_CHAR_BUDGET = 160


def model_score_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    timeout: float = 120.0,
    temperature: float = 0.0,
    max_tokens: int | None = 512,
    max_pairs: int | None = None,
    pair_budget: str = "manual",
    target_paper_coverage: float = 1.0,
    comparison_rounds: int = 1,
    shuffle_pairs: bool = False,
    pair_sampling: str = "sequential",
    seed: int = 0,
    concurrency: int = 1,
    direct_score: bool = False,
    calibration_report_path: str | Path | None = None,
    fusion_weights_path: str | Path | None = None,
    bidirectional_pairwise: bool = False,
    bidirectional_confidence_margin: float = 0.0,
    pairwise_aggregation_method: str = "rank_centrality",
    validity_labels_path: str | Path | None = None,
    validity_label_column: str = "future_impact",
    validity_bootstrap_samples: int = 200,
    validity_permutation_samples: int = 200,
    pairwise_stability_samples: int = 0,
    pairwise_stability_top_n: int = 20,
    pairwise_stability_seed: int = 17,
    final_score_stability_samples: int = 0,
    final_score_stability_top_n: int = 20,
    final_score_stability_seed: int = 23,
    pairwise_judgments_path: str | Path | None = None,
    resume_pairwise_judgments_path: str | Path | None = None,
    pairwise_journal_path: str | Path | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    if pairwise_judgments_path is not None and resume_pairwise_judgments_path is not None:
        raise ValueError(
            "pairwise_judgments_path and resume_pairwise_judgments_path are mutually exclusive"
        )
    papers = [normalize_paper(paper) for paper in load_papers_jsonl(input_path)]
    pair_budget_report = pair_budget_plan(
        len(papers),
        requested_max_pairs=max_pairs,
        pair_budget=pair_budget,
        target_paper_coverage=target_paper_coverage,
    )
    effective_max_pairs = pair_budget_report["effective_max_pairs"]
    pairwise_journal_append = (
        pairwise_journal_path is not None
        and resume_pairwise_judgments_path is not None
        and Path(pairwise_journal_path) == Path(resume_pairwise_judgments_path)
    )
    _prepare_pairwise_journal(pairwise_journal_path, append=pairwise_journal_append)
    calibration_model = _load_empirical_calibration(calibration_report_path)
    (
        fusion_weights,
        evidence_weights,
        impact_index_v3_weights,
        fusion_weight_source,
    ) = _load_fusion_weights(fusion_weights_path)
    direct_scores = (
        generate_model_direct_scores(
            papers,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            concurrency=concurrency,
            client=client,
        )
        if direct_score
        else []
    )
    if pairwise_judgments_path is not None:
        judgments = load_pairwise_judgments_jsonl(pairwise_judgments_path)
        pairwise_judgment_source = {
            "type": "file",
            "path": str(pairwise_judgments_path),
            "count": len(judgments),
        }
    elif resume_pairwise_judgments_path is not None:
        existing_judgments = load_pairwise_judgments_jsonl(resume_pairwise_judgments_path)
        generated_judgments = generate_model_pairwise_judgments(
            papers,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            max_pairs=effective_max_pairs,
            comparison_rounds=comparison_rounds,
            shuffle_pairs=shuffle_pairs,
            pair_sampling=pair_sampling,
            seed=seed,
            concurrency=concurrency,
            bidirectional_pairwise=bidirectional_pairwise,
            bidirectional_confidence_margin=bidirectional_confidence_margin,
            skip_existing_judgments=existing_judgments,
            pairwise_journal_path=pairwise_journal_path,
            client=client,
        )
        judgments = sorted(
            [*existing_judgments, *generated_judgments],
            key=_pairwise_judgment_sort_key,
        )
        pairwise_judgment_source = {
            "type": "resume",
            "path": str(resume_pairwise_judgments_path),
            "existing_count": len(existing_judgments),
            "generated_count": len(generated_judgments),
            "count": len(judgments),
        }
    else:
        judgments = generate_model_pairwise_judgments(
            papers,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            max_pairs=effective_max_pairs,
            comparison_rounds=comparison_rounds,
            shuffle_pairs=shuffle_pairs,
            pair_sampling=pair_sampling,
            seed=seed,
            concurrency=concurrency,
            bidirectional_pairwise=bidirectional_pairwise,
            bidirectional_confidence_margin=bidirectional_confidence_margin,
            pairwise_journal_path=pairwise_journal_path,
            client=client,
        )
        pairwise_judgment_source = {
            "type": "model",
            "path": None,
            "count": len(judgments),
        }
    if calibration_model is not None:
        _attach_calibrated_pairwise_confidences(judgments, calibration_model)
    pairwise_rankings = aggregate_pairwise_impact_index(
        papers,
        judgments,
        pairwise_aggregation_method=pairwise_aggregation_method,
    )
    if direct_scores:
        _attach_direct_scores(pairwise_rankings, direct_scores)
    fusion = compute_impact_index_v2(
        papers,
        pairwise_rankings,
        weights=fusion_weights,
        evidence_weights=evidence_weights,
        impact_index_v3_weights=impact_index_v3_weights,
    )
    fusion["summary"]["weight_source"] = fusion_weight_source
    rankings = fusion["rankings"]
    decisive_judgment_count = sum(
        judgment.get("decisive", judgment["parseable"]) for judgment in judgments
    )
    ranking_quality = pairwise_ranking_quality_report(
        papers,
        judgments,
        decisive_judgment_count=decisive_judgment_count,
    )
    pairwise_stability = pairwise_bootstrap_stability_report(
        papers,
        judgments,
        pairwise_rankings,
        samples=pairwise_stability_samples,
        seed=pairwise_stability_seed,
        top_n=pairwise_stability_top_n,
        pairwise_aggregation_method=pairwise_aggregation_method,
        extra_tracked_paper_ids=[
            str(row.get("paper_id")) for row in rankings[:pairwise_stability_top_n]
        ],
    )
    final_score_stability = final_score_bootstrap_stability_report(
        papers,
        judgments,
        rankings,
        samples=final_score_stability_samples,
        seed=final_score_stability_seed,
        top_n=final_score_stability_top_n,
        pairwise_aggregation_method=pairwise_aggregation_method,
        weights=fusion_weights,
        evidence_weights=evidence_weights,
        impact_index_v3_weights=impact_index_v3_weights,
    )
    external_readiness = external_evidence_readiness_report(papers)
    external_paper_coverage_rows = external_evidence_paper_coverage_rows(papers)
    _attach_ranking_quality(rankings, ranking_quality)
    _attach_pairwise_stability(rankings, pairwise_stability)
    _attach_final_score_stability(rankings, final_score_stability)
    report = {
        "summary": {
            "paper_count": len(papers),
            "direct_scoring_enabled": direct_score,
            "direct_score_count": len(direct_scores),
            "parseable_direct_score_count": sum(
                row["parseable"] for row in direct_scores
            ),
            "direct_score_quality": direct_score_quality_report(direct_scores)
            if direct_score
            else {"status": "disabled"},
            "judgment_count": len(judgments),
            "parseable_judgment_count": sum(judgment["parseable"] for judgment in judgments),
            "parseable_rate": _safe_divide(
                sum(judgment["parseable"] for judgment in judgments),
                len(judgments),
            ),
            "scoring_method": "impact_index_v3_evidence_first",
            "model_pairwise_scoring_method": _pairwise_scoring_method_name(
                pairwise_aggregation_method
            ),
            "fusion_scoring_enabled": True,
            "fusion": fusion["summary"],
            "pair_budget": pair_budget_report,
            "comparison_rounds": comparison_rounds,
            "shuffle_pairs": shuffle_pairs,
            "pair_sampling": pair_sampling,
            "seed": seed,
            "concurrency": concurrency,
            "bidirectional_pairwise": bidirectional_pairwise,
            "bidirectional_confidence_margin": bidirectional_confidence_margin,
            "pairwise_aggregation_method": pairwise_aggregation_method,
            "pairwise_judgment_source": pairwise_judgment_source,
            "pairwise_journal": _pairwise_journal_report(
                pairwise_journal_path,
                append=pairwise_journal_append,
            ),
            "decisive_judgment_count": decisive_judgment_count,
            "position_consistency_counts": _position_consistency_counts(judgments),
            "ranking_quality": ranking_quality,
            "pairwise_stability": pairwise_stability.get("summary", {}),
            "final_score_stability": final_score_stability.get("summary", {}),
            "calibration_model": _calibration_model_report(calibration_model),
            "impact_validity_evaluation": {
                "status": "not_provided",
                "reason": "No validity labels were supplied to model-score-dataset.",
            },
            "external_evidence_readiness": external_readiness,
            "impact_dimension_coverage": ranking_dimension_coverage(rankings),
        },
        "rankings": rankings,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if direct_scores:
        _write_jsonl(destination / "direct_scores.jsonl", direct_scores)
        _write_direct_scores_csv(destination / "direct_scores.csv", direct_scores)
    _write_jsonl(destination / "pairwise_judgments.jsonl", judgments)
    _write_jsonl(destination / "impact_scores.jsonl", rankings)
    _write_impact_scores_csv(destination / "impact_scores.csv", rankings)
    report["summary"]["impact_validity_evaluation"] = _write_impact_validity_if_requested(
        destination,
        validity_labels_path=validity_labels_path,
        label_column=validity_label_column,
        bootstrap_samples=validity_bootstrap_samples,
        permutation_samples=validity_permutation_samples,
    )
    report["summary"]["validated_impact_claim"] = _attach_validity_confidence(
        rankings,
        report["summary"]["impact_validity_evaluation"],
    )
    _write_jsonl(destination / "impact_scores.jsonl", rankings)
    _write_impact_scores_csv(destination / "impact_scores.csv", rankings)
    (destination / "external_evidence_readiness.json").write_text(
        json.dumps(external_readiness, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "external_evidence_readiness.md").write_text(
        render_external_evidence_readiness_markdown(external_readiness),
        encoding="utf-8",
    )
    write_external_evidence_paper_coverage_csv(
        external_paper_coverage_rows,
        destination / "external_evidence_paper_coverage.csv",
    )
    (destination / "pairwise_stability.json").write_text(
        json.dumps(pairwise_stability, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_pairwise_stability_csv(
        destination / "pairwise_stability.csv",
        pairwise_stability,
    )
    (destination / "pairwise_stability.md").write_text(
        render_pairwise_stability_markdown(pairwise_stability),
        encoding="utf-8",
    )
    (destination / "final_score_stability.json").write_text(
        json.dumps(final_score_stability, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_final_score_stability_csv(
        destination / "final_score_stability.csv",
        final_score_stability,
    )
    (destination / "final_score_stability.md").write_text(
        render_final_score_stability_markdown(final_score_stability),
        encoding="utf-8",
    )
    (destination / "model_impact_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "model_impact_report.md").write_text(
        render_model_impact_report_markdown(report),
        encoding="utf-8",
    )
    impact_score_cards = impact_score_cards_from_rankings(rankings)
    (destination / "impact_score_cards.json").write_text(
        json.dumps(impact_score_cards, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "impact_score_cards.md").write_text(
        render_impact_score_cards_markdown(impact_score_cards),
        encoding="utf-8",
    )
    write_impact_claim_certificate(destination, destination)
    if (destination / "reports").exists():
        write_impact_claim_certificate(destination, destination / "reports")
    return report


def model_score_preflight_report(
    input_path: str | Path,
    *,
    max_pairs: int | None = None,
    pair_budget: str = "manual",
    target_paper_coverage: float = 1.0,
    comparison_rounds: int = 1,
    shuffle_pairs: bool = False,
    pair_sampling: str = "sequential",
    seed: int = 0,
    direct_score: bool = False,
    bidirectional_pairwise: bool = False,
    resume_pairwise_judgments_path: str | Path | None = None,
) -> dict[str, Any]:
    papers = [normalize_paper(paper) for paper in load_papers_jsonl(input_path)]
    budget = pair_budget_plan(
        len(papers),
        requested_max_pairs=max_pairs,
        pair_budget=pair_budget,
        target_paper_coverage=target_paper_coverage,
    )
    plan = pairwise_comparison_plan(
        papers,
        comparison_rounds=comparison_rounds,
        shuffle_pairs=shuffle_pairs,
        pair_sampling=pair_sampling,
        seed=seed,
        max_pairs=budget["effective_max_pairs"],
    )
    planned_keys = {_pairwise_plan_item_key(item) for item in plan}
    existing_judgments = (
        load_pairwise_judgments_jsonl(resume_pairwise_judgments_path)
        if resume_pairwise_judgments_path is not None
        else []
    )
    existing_keys = {
        key
        for judgment in existing_judgments
        for key in [_pairwise_judgment_key(judgment)]
        if key is not None
    }
    completed_planned_pair_count = len(planned_keys & existing_keys)
    missing_pair_count = len(plan) - completed_planned_pair_count
    pairwise_request_multiplier = 2 if bidirectional_pairwise else 1
    pairwise_model_request_count = missing_pair_count * pairwise_request_multiplier
    direct_model_request_count = len(papers) if direct_score else 0
    external_readiness = external_evidence_readiness_report(papers)
    pair_plan_quality = _pairwise_plan_quality_report(papers, plan)
    return {
        "input_path": str(input_path),
        "paper_count": len(papers),
        "pair_plan": {
            "total_possible_pair_count": len(papers) * (len(papers) - 1) // 2,
            "planned_pair_count": len(plan),
            "comparison_rounds": comparison_rounds,
            "pair_sampling": pair_sampling,
            "shuffle_pairs": shuffle_pairs,
            "seed": seed,
            "budget": budget,
        },
        "pair_plan_quality": pair_plan_quality,
        "resume": {
            "path": str(resume_pairwise_judgments_path)
            if resume_pairwise_judgments_path is not None
            else None,
            "existing_judgment_count": len(existing_judgments),
            "completed_planned_pair_count": completed_planned_pair_count,
            "missing_pair_count": missing_pair_count,
        },
        "request_estimate": {
            "pairwise_request_multiplier": pairwise_request_multiplier,
            "pairwise_model_request_count": pairwise_model_request_count,
            "direct_model_request_count": direct_model_request_count,
            "total_model_request_count": pairwise_model_request_count
            + direct_model_request_count,
        },
        "external_evidence_readiness": {
            "status": external_readiness.get("decision", {}).get("status"),
            "can_claim_high_credibility": external_readiness.get("decision", {}).get(
                "can_claim_high_credibility",
                False,
            ),
            "external_query_id_coverage": external_readiness.get(
                "identifier_coverage",
                {},
            ).get("external_query_id_coverage", 0.0),
            "field_year_percentile_coverage": external_readiness.get(
                "signal_coverage",
                {},
            ).get("field_year_percentile_coverage", 0.0),
        },
        "recommendations": _model_score_preflight_recommendations(
            missing_pair_count=missing_pair_count,
            resume_pairwise_judgments_path=resume_pairwise_judgments_path,
            pair_plan_quality=pair_plan_quality,
        ),
    }


def _pairwise_plan_quality_report(
    papers: list[Paper],
    plan: list[dict[str, Any]],
) -> dict[str, Any]:
    comparisons = {paper.paper_id: 0 for paper in papers}
    adjacency = {paper.paper_id: set() for paper in papers}
    for item in plan:
        paper_a = item["paper_a"]
        paper_b = item["paper_b"]
        if paper_a.paper_id not in comparisons or paper_b.paper_id not in comparisons:
            continue
        comparisons[paper_a.paper_id] += 1
        comparisons[paper_b.paper_id] += 1
        adjacency[paper_a.paper_id].add(paper_b.paper_id)
        adjacency[paper_b.paper_id].add(paper_a.paper_id)

    paper_count = len(papers)
    compared_paper_count = sum(1 for count in comparisons.values() if count > 0)
    uncompared_paper_count = paper_count - compared_paper_count
    min_comparisons = min(comparisons.values(), default=0)
    max_comparisons = max(comparisons.values(), default=0)
    mean_comparisons = _safe_divide(sum(comparisons.values()), paper_count)
    minimum_recommended_comparisons_per_paper = (
        2 if paper_count >= 3 else int(paper_count > 1)
    )
    graph_quality = _pairwise_graph_quality(adjacency)
    if paper_count == 0:
        status = "missing"
        reason = "No papers were provided."
    elif uncompared_paper_count > 0:
        status = "exploratory"
        reason = (
            f"The planned pairwise graph covers only {compared_paper_count} of "
            f"{paper_count} papers; increase pair budget or target coverage."
        )
    elif graph_quality["connected_component_count"] > 1:
        status = "exploratory"
        reason = (
            f"The planned pairwise graph has "
            f"{graph_quality['connected_component_count']} connected components; "
            "Rank Centrality scores across disconnected components will not be comparable."
        )
    elif min_comparisons < minimum_recommended_comparisons_per_paper:
        status = "limited"
        reason = (
            "The planned pairwise graph is connected, but at least one paper has "
            f"only {min_comparisons} planned comparison(s)."
        )
    else:
        status = "reportable"
        reason = "The planned pairwise graph covers all papers and is connected."

    return {
        "status": status,
        "reason": reason,
        "planned_pair_count": len(plan),
        "minimum_connected_pair_count": max(paper_count - 1, 0),
        "compared_paper_count": compared_paper_count,
        "uncompared_paper_count": uncompared_paper_count,
        "paper_coverage_rate": round(
            compared_paper_count / paper_count if paper_count else 0.0,
            6,
        ),
        "min_planned_comparisons_per_paper": min_comparisons,
        "max_planned_comparisons_per_paper": max_comparisons,
        "mean_planned_comparisons_per_paper": round(mean_comparisons, 6),
        "minimum_recommended_comparisons_per_paper": minimum_recommended_comparisons_per_paper,
        **graph_quality,
    }


def write_model_score_preflight(
    input_path: str | Path,
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    report = model_score_preflight_report(input_path, **kwargs)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "model_score_preflight.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "model_score_preflight.md").write_text(
        render_model_score_preflight_markdown(report),
        encoding="utf-8",
    )
    return report


def render_model_score_preflight_markdown(report: dict[str, Any]) -> str:
    pair_plan = report.get("pair_plan", {})
    pair_plan_quality = report.get("pair_plan_quality", {})
    resume = report.get("resume", {})
    requests = report.get("request_estimate", {})
    external = report.get("external_evidence_readiness", {})
    lines = [
        "# Model Score Preflight",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| paper_count | {report.get('paper_count', 0)} |",
        f"| total_possible_pair_count | {pair_plan.get('total_possible_pair_count', 0)} |",
        f"| planned_pair_count | {pair_plan.get('planned_pair_count', 0)} |",
        f"| completed_planned_pair_count | {resume.get('completed_planned_pair_count', 0)} |",
        f"| missing_pair_count | {resume.get('missing_pair_count', 0)} |",
        f"| pairwise_request_multiplier | {requests.get('pairwise_request_multiplier', 1)} |",
        f"| pairwise_model_request_count | {requests.get('pairwise_model_request_count', 0)} |",
        f"| direct_model_request_count | {requests.get('direct_model_request_count', 0)} |",
        f"| total_model_request_count | {requests.get('total_model_request_count', 0)} |",
        f"| external_evidence_status | {external.get('status', '')} |",
        f"| external_query_id_coverage | {external.get('external_query_id_coverage', 0.0)} |",
        f"| field_year_percentile_coverage | {external.get('field_year_percentile_coverage', 0.0)} |",
        "",
        "## Pair Plan Quality",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| status | {pair_plan_quality.get('status', '')} |",
        f"| reason | {pair_plan_quality.get('reason', '')} |",
        f"| minimum_connected_pair_count | {pair_plan_quality.get('minimum_connected_pair_count', 0)} |",
        f"| compared_paper_count | {pair_plan_quality.get('compared_paper_count', 0)} |",
        f"| paper_coverage_rate | {pair_plan_quality.get('paper_coverage_rate', 0.0)} |",
        f"| connected_component_count | {pair_plan_quality.get('connected_component_count', 0)} |",
        f"| largest_connected_component_coverage | {pair_plan_quality.get('largest_connected_component_coverage', 0.0)} |",
        f"| isolated_paper_count | {pair_plan_quality.get('isolated_paper_count', 0)} |",
        f"| min_planned_comparisons_per_paper | {pair_plan_quality.get('min_planned_comparisons_per_paper', 0)} |",
        f"| minimum_recommended_comparisons_per_paper | {pair_plan_quality.get('minimum_recommended_comparisons_per_paper', 0)} |",
        "",
        "## Recommendations",
        "",
    ]
    lines.extend(f"- {item}" for item in report.get("recommendations", []))
    lines.append("")
    return "\n".join(lines)


def _model_score_preflight_recommendations(
    *,
    missing_pair_count: int,
    resume_pairwise_judgments_path: str | Path | None,
    pair_plan_quality: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    if missing_pair_count <= 0:
        recommendations.append(
            "All planned pairwise judgments are already present; use --pairwise-judgments for replay."
        )
    elif resume_pairwise_judgments_path is not None:
        recommendations.append(
            "Run model-score-dataset with --resume-pairwise-judgments and --pairwise-journal to generate only missing judgments."
        )
    else:
        recommendations.append(
            "Run model-score-dataset with --pairwise-journal so interrupted long runs can be resumed."
        )

    status = str(pair_plan_quality.get("status", ""))
    if status == "exploratory":
        recommendations.append(
            "Increase --max-pairs, --target-paper-coverage, or use --pair-sampling coverage until the planned pair graph covers all papers and is connected."
        )
    elif status == "limited":
        recommendations.append(
            "Increase --max-pairs or --comparison-rounds so every paper has the recommended minimum planned comparisons."
        )
    return recommendations


def generate_model_direct_scores(
    papers: list[Paper],
    *,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    timeout: float = 120.0,
    temperature: float = 0.0,
    max_tokens: int | None = 32,
    concurrency: int = 1,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    owns_client = client is None
    http_client = client or httpx.Client()
    try:
        if concurrency == 1:
            rows = [
                _complete_direct_score_item(
                    index,
                    paper,
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    timeout=timeout,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    client=http_client,
                )
                for index, paper in enumerate(papers)
            ]
        else:
            rows = []
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(
                        _complete_direct_score_item,
                        index,
                        paper,
                        base_url=base_url,
                        model=model,
                        api_key=api_key,
                        timeout=timeout,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        client=http_client,
                    )
                    for index, paper in enumerate(papers)
                ]
                for future in as_completed(futures):
                    rows.append(future.result())
    finally:
        if owns_client and hasattr(http_client, "close"):
            http_client.close()
    rows = sorted(rows, key=lambda row: int(row["paper_index"]))
    for row in rows:
        row.pop("paper_index", None)
    return rows


def generate_model_pairwise_judgments(
    papers: list[Paper],
    *,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    timeout: float = 120.0,
    temperature: float = 0.0,
    max_tokens: int | None = 512,
    max_pairs: int | None = None,
    comparison_rounds: int = 1,
    shuffle_pairs: bool = False,
    pair_sampling: str = "sequential",
    seed: int = 0,
    concurrency: int = 1,
    bidirectional_pairwise: bool = False,
    bidirectional_confidence_margin: float = 0.0,
    skip_existing_judgments: list[dict[str, Any]] | None = None,
    pairwise_journal_path: str | Path | None = None,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if bidirectional_confidence_margin < 0.0:
        raise ValueError("bidirectional_confidence_margin must be non-negative")
    owns_client = client is None
    http_client = client or httpx.Client()
    try:
        plan = pairwise_comparison_plan(
            papers,
            comparison_rounds=comparison_rounds,
            shuffle_pairs=shuffle_pairs,
            pair_sampling=pair_sampling,
            seed=seed,
            max_pairs=max_pairs,
        )
        if skip_existing_judgments:
            existing_keys = {
                _pairwise_judgment_key(judgment)
                for judgment in skip_existing_judgments
                if _pairwise_judgment_key(judgment) is not None
            }
            plan = [
                item
                for item in plan
                if _pairwise_plan_item_key(item) not in existing_keys
            ]
        if concurrency == 1:
            judgments = []
            for item in plan:
                judgment = _complete_plan_item(
                    item,
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    timeout=timeout,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    bidirectional_pairwise=bidirectional_pairwise,
                    bidirectional_confidence_margin=bidirectional_confidence_margin,
                    client=http_client,
                )
                _append_pairwise_journal(pairwise_journal_path, judgment)
                judgments.append(judgment)
        else:
            judgments = []
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(
                        _complete_plan_item,
                        item,
                        base_url=base_url,
                        model=model,
                        api_key=api_key,
                        timeout=timeout,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        bidirectional_pairwise=bidirectional_pairwise,
                        bidirectional_confidence_margin=bidirectional_confidence_margin,
                        client=http_client,
                    )
                    for item in plan
                ]
                for future in as_completed(futures):
                    judgment = future.result()
                    _append_pairwise_journal(pairwise_journal_path, judgment)
                    judgments.append(judgment)
    finally:
        if owns_client and hasattr(http_client, "close"):
            http_client.close()
    return sorted(judgments, key=lambda judgment: int(judgment["pair_index"]))


def load_pairwise_judgments_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    judgments: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Pairwise judgments row {line_number} is not valid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Pairwise judgments row {line_number} must be a JSON object"
                )
            judgments.append(row)
    return judgments


def _prepare_pairwise_journal(path: str | Path | None, *, append: bool) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not append:
        destination.write_text("", encoding="utf-8")


def _append_pairwise_journal(
    path: str | Path | None,
    judgment: dict[str, Any],
) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(judgment, ensure_ascii=False) + "\n")


def _pairwise_journal_report(
    path: str | Path | None,
    *,
    append: bool,
) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "path": str(path),
        "append": append,
    }


def _pairwise_plan_item_key(item: dict[str, Any]) -> tuple[int, str, str]:
    paper_a = item["paper_a"]
    paper_b = item["paper_b"]
    left, right = sorted((str(paper_a.paper_id), str(paper_b.paper_id)))
    return int(item.get("round", 1)), left, right


def _pairwise_judgment_key(judgment: dict[str, Any]) -> tuple[int, str, str] | None:
    paper_a_id = judgment.get("paper_a_id")
    paper_b_id = judgment.get("paper_b_id")
    if not paper_a_id or not paper_b_id:
        return None
    left, right = sorted((str(paper_a_id), str(paper_b_id)))
    return int(judgment.get("round", 1) or 1), left, right


def _pairwise_judgment_sort_key(judgment: dict[str, Any]) -> tuple[int, int, str, str]:
    key = _pairwise_judgment_key(judgment)
    round_number, left, right = key if key is not None else (0, "", "")
    try:
        pair_index = int(judgment.get("pair_index", 0) or 0)
    except (TypeError, ValueError):
        pair_index = 0
    return round_number, pair_index, left, right


def _complete_direct_score_item(
    paper_index: int,
    paper: Paper,
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    temperature: float,
    max_tokens: int | None,
    client: Any,
) -> dict[str, Any]:
    prompt = build_direct_impact_score_prompt(paper)
    prediction = _complete_pairwise_prompt(
        prompt,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        structured_choice=direct_score_choices(),
        request_logprobs=False,
        client=client,
    )
    prediction_text = _completion_text(prediction)
    score = parse_direct_impact_score(prediction_text)
    return {
        "paper_index": paper_index,
        "paper_id": paper.paper_id,
        "title": paper.title,
        "direct_score": score,
        "parseable": score is not None,
        "prediction": prediction_text,
        "model": model,
    }


def _complete_plan_item(
    item: dict[str, Any],
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    temperature: float,
    max_tokens: int | None,
    bidirectional_pairwise: bool = False,
    bidirectional_confidence_margin: float = 0.0,
    client: Any,
) -> dict[str, Any]:
    pair_index = int(item["pair_index"])
    round_number = int(item["round"])
    paper_a = item["paper_a"]
    paper_b = item["paper_b"]
    forward = _complete_oriented_pairwise_item(
        paper_a,
        paper_b,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        client=client,
    )
    if not bidirectional_pairwise:
        return _pairwise_judgment_record(
            pair_index=pair_index,
            round_number=round_number,
            paper_a=paper_a,
            paper_b=paper_b,
            selected=forward,
            model=model,
        )

    reverse = _complete_oriented_pairwise_item(
        paper_b,
        paper_a,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        client=client,
    )
    return _bidirectional_pairwise_judgment_record(
        pair_index=pair_index,
        round_number=round_number,
        paper_a=paper_a,
        paper_b=paper_b,
        forward=forward,
        reverse=reverse,
        model=model,
        confidence_margin=bidirectional_confidence_margin,
    )


def pairwise_comparison_plan(
    papers: list[Paper],
    *,
    comparison_rounds: int = 1,
    shuffle_pairs: bool = False,
    pair_sampling: str = "sequential",
    seed: int = 0,
    max_pairs: int | None = None,
) -> list[dict[str, Any]]:
    if comparison_rounds < 1:
        raise ValueError("comparison_rounds must be at least 1")
    if pair_sampling not in {"sequential", "coverage"}:
        raise ValueError("pair_sampling must be 'sequential' or 'coverage'")
    base_pairs = list(combinations(papers, 2))
    rng = random.Random(seed)
    plan: list[dict[str, Any]] = []
    for round_number in range(1, comparison_rounds + 1):
        if pair_sampling == "coverage":
            round_pairs = _coverage_ordered_pairs(papers, rng=rng if shuffle_pairs else None)
        else:
            round_pairs = list(base_pairs)
            if shuffle_pairs:
                rng.shuffle(round_pairs)
        for paper_a, paper_b in round_pairs:
            if max_pairs is not None and len(plan) >= max_pairs:
                return plan
            plan.append(
                {
                    "pair_index": len(plan) + 1,
                    "round": round_number,
                    "paper_a": paper_a,
                    "paper_b": paper_b,
                }
            )
    return plan


def pair_budget_plan(
    paper_count: int,
    *,
    requested_max_pairs: int | None = None,
    pair_budget: str = "manual",
    target_paper_coverage: float = 1.0,
) -> dict[str, Any]:
    if paper_count < 0:
        raise ValueError("paper_count must be non-negative")
    if pair_budget not in {"manual", "auto"}:
        raise ValueError("pair_budget must be 'manual' or 'auto'")
    if target_paper_coverage <= 0.0 or target_paper_coverage > 1.0:
        raise ValueError("target_paper_coverage must be in the range (0, 1]")

    total_pairs = paper_count * (paper_count - 1) // 2
    if pair_budget == "manual":
        effective_max_pairs = requested_max_pairs
        reason = (
            "Manual pair budget uses all planned pairs."
            if requested_max_pairs is None
            else f"Manual pair budget limited comparisons to {requested_max_pairs} pairs."
        )
    else:
        target_papers = math.ceil(paper_count * target_paper_coverage)
        effective_max_pairs = math.ceil(target_papers / 2)
        effective_max_pairs = min(effective_max_pairs, total_pairs)
        reason = (
            f"Auto pair budget selected {effective_max_pairs} pairs to cover about "
            f"{target_paper_coverage * 100:.1f}% of {paper_count} papers."
        )

    return {
        "mode": pair_budget,
        "requested_max_pairs": requested_max_pairs,
        "effective_max_pairs": effective_max_pairs,
        "target_paper_coverage": target_paper_coverage,
        "paper_count": paper_count,
        "reason": reason,
    }


def _coverage_ordered_pairs(
    papers: list[Paper],
    *,
    rng: random.Random | None = None,
) -> list[tuple[Paper, Paper]]:
    if len(papers) < 2:
        return []

    remaining = list(papers)
    ordered: list[tuple[Paper, Paper]] = []
    used_keys: set[tuple[str, str]] = set()
    degrees: Counter[str] = Counter()

    while len(remaining) >= 2:
        left = remaining.pop(0)
        right = remaining.pop(0)
        ordered.append((left, right))
        used_keys.add(_paper_pair_key(left, right))
        degrees[left.paper_id] += 1
        degrees[right.paper_id] += 1
    if remaining:
        left = remaining.pop(0)
        right = papers[0]
        ordered.append((left, right))
        used_keys.add(_paper_pair_key(left, right))
        degrees[left.paper_id] += 1
        degrees[right.paper_id] += 1
    if rng is not None:
        rng.shuffle(ordered)

    for left, right in _coverage_bridge_pairs(papers, used_keys=used_keys):
        ordered.append((left, right))
        used_keys.add(_paper_pair_key(left, right))
        degrees[left.paper_id] += 1
        degrees[right.paper_id] += 1

    ordered.extend(
        _degree_balanced_remaining_pairs(
            papers,
            used_keys=used_keys,
            degrees=degrees,
            rng=rng,
        )
    )
    return ordered


def _coverage_bridge_pairs(
    papers: list[Paper],
    *,
    used_keys: set[tuple[str, str]],
) -> list[tuple[Paper, Paper]]:
    bridges: list[tuple[Paper, Paper]] = []
    for index in range(1, len(papers) - 1, 2):
        left = papers[index]
        right = papers[index + 1]
        if _paper_pair_key(left, right) in used_keys:
            continue
        bridges.append((left, right))
    return bridges


def _degree_balanced_remaining_pairs(
    papers: list[Paper],
    *,
    used_keys: set[tuple[str, str]],
    degrees: Counter[str],
    rng: random.Random | None,
) -> list[tuple[Paper, Paper]]:
    base_pairs = [
        (left, right)
        for left, right in combinations(papers, 2)
        if _paper_pair_key(left, right) not in used_keys
    ]
    if rng is not None:
        rng.shuffle(base_pairs)

    heap: list[tuple[int, int, int, str, str, Paper, Paper]] = []
    for order, (left, right) in enumerate(base_pairs):
        left_degree = degrees[left.paper_id]
        right_degree = degrees[right.paper_id]
        heapq.heappush(
            heap,
            (
                max(left_degree, right_degree),
                left_degree + right_degree,
                order,
                left.paper_id,
                right.paper_id,
                left,
                right,
            ),
        )

    ordered: list[tuple[Paper, Paper]] = []
    emitted_keys: set[tuple[str, str]] = set()
    while heap:
        _, _, order, _, _, left, right = heapq.heappop(heap)
        pair_key = _paper_pair_key(left, right)
        if pair_key in emitted_keys:
            continue
        left_degree = degrees[left.paper_id]
        right_degree = degrees[right.paper_id]
        current_key = (
            max(left_degree, right_degree),
            left_degree + right_degree,
            order,
            left.paper_id,
            right.paper_id,
            left,
            right,
        )
        if heap and current_key[:2] > heap[0][:2]:
            heapq.heappush(heap, current_key)
            continue
        ordered.append((left, right))
        emitted_keys.add(pair_key)
        degrees[left.paper_id] += 1
        degrees[right.paper_id] += 1
    return ordered


def _paper_pair_key(paper_a: Paper, paper_b: Paper) -> tuple[str, str]:
    return tuple(sorted((paper_a.paper_id, paper_b.paper_id)))


def _complete_oriented_pairwise_item(
    paper_a: Paper,
    paper_b: Paper,
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    temperature: float,
    max_tokens: int | None,
    client: Any,
) -> dict[str, Any]:
    prompt = build_pairwise_impact_prompt(paper_a, paper_b)
    prediction = _complete_pairwise_prompt(
        prompt,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        structured_choice=["Answer: A", "Answer: B"],
        request_logprobs=True,
        client=client,
    )
    prediction_text = _completion_text(prediction)
    winner_side = parse_pairwise_winner(prediction_text)
    return {
        "prediction": prediction_text,
        "winner_side": winner_side,
        **_pairwise_confidence_info(prediction, winner_side),
    }


def _pairwise_judgment_record(
    *,
    pair_index: int,
    round_number: int,
    paper_a: Paper,
    paper_b: Paper,
    selected: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    winner_side = selected["winner_side"]
    winner_id, loser_id = _winner_loser_ids(winner_side, paper_a, paper_b)
    return {
        "pair_index": pair_index,
        "round": round_number,
        "paper_a_id": paper_a.paper_id,
        "paper_b_id": paper_b.paper_id,
        "winner_side": winner_side.upper() if winner_side else None,
        "winner_id": winner_id,
        "loser_id": loser_id,
        "parseable": winner_side is not None,
        "prediction": selected["prediction"],
        "model": model,
        "confidence": selected["confidence"],
        "confidence_method": selected["confidence_method"],
        "choice_logprob_margin": selected["choice_logprob_margin"],
    }


def _bidirectional_pairwise_judgment_record(
    *,
    pair_index: int,
    round_number: int,
    paper_a: Paper,
    paper_b: Paper,
    forward: dict[str, Any],
    reverse: dict[str, Any],
    model: str,
    confidence_margin: float,
) -> dict[str, Any]:
    forward_side = forward["winner_side"]
    reverse_side = reverse["winner_side"]
    reverse_mapped_side = _opposite_side(reverse_side)
    selected_side, selected_source = _select_bidirectional_side(
        forward_side,
        reverse_mapped_side,
        forward["confidence"],
        reverse["confidence"],
        confidence_margin=confidence_margin,
    )
    position_consistency = (
        "consistent"
        if forward_side is not None
        and reverse_mapped_side is not None
        and forward_side == reverse_mapped_side
        else "conflict"
        if forward_side is not None and reverse_mapped_side is not None
        else "unparseable"
    )
    decisive = selected_side is not None
    winner_id, loser_id = _winner_loser_ids(selected_side, paper_a, paper_b)
    confidence = _bidirectional_confidence(forward, reverse, selected_source)
    confidence_method = (
        "bidirectional_consistent"
        if selected_source == "consistent"
        else "bidirectional_confidence"
        if selected_source in {"forward_confidence", "reverse_confidence"}
        else "bidirectional_conflict"
    )
    return {
        "pair_index": pair_index,
        "round": round_number,
        "paper_a_id": paper_a.paper_id,
        "paper_b_id": paper_b.paper_id,
        "winner_side": selected_side.upper() if selected_side else None,
        "winner_id": winner_id,
        "loser_id": loser_id,
        "parseable": decisive,
        "prediction": forward["prediction"],
        "model": model,
        "confidence": confidence,
        "confidence_method": confidence_method,
        "choice_logprob_margin": _max_optional(
            forward.get("choice_logprob_margin"),
            reverse.get("choice_logprob_margin"),
        ),
        "bidirectional": True,
        "decisive": decisive,
        "forward_winner_side": forward_side.upper() if forward_side else None,
        "reverse_winner_side": reverse_side.upper() if reverse_side else None,
        "reverse_mapped_winner_side": reverse_mapped_side.upper()
        if reverse_mapped_side
        else None,
        "position_consistency": position_consistency,
        "selected_source": selected_source,
        "forward_prediction": forward["prediction"],
        "reverse_prediction": reverse["prediction"],
        "forward_confidence": forward["confidence"],
        "reverse_confidence": reverse["confidence"],
    }


def _select_bidirectional_side(
    forward_side: str | None,
    reverse_mapped_side: str | None,
    forward_confidence: float,
    reverse_confidence: float,
    *,
    confidence_margin: float,
) -> tuple[str | None, str]:
    if forward_side is not None and forward_side == reverse_mapped_side:
        return forward_side, "consistent"
    if forward_side is None and reverse_mapped_side is None:
        return None, "unparseable"
    if forward_side is None:
        return reverse_mapped_side, "reverse_only"
    if reverse_mapped_side is None:
        return forward_side, "forward_only"
    if forward_confidence > reverse_confidence + confidence_margin:
        return forward_side, "forward_confidence"
    if reverse_confidence > forward_confidence + confidence_margin:
        return reverse_mapped_side, "reverse_confidence"
    return None, "conflict_tie"


def _winner_loser_ids(
    winner_side: str | None,
    paper_a: Paper,
    paper_b: Paper,
) -> tuple[str | None, str | None]:
    if winner_side == "a":
        return paper_a.paper_id, paper_b.paper_id
    if winner_side == "b":
        return paper_b.paper_id, paper_a.paper_id
    return None, None


def _opposite_side(side: str | None) -> str | None:
    if side == "a":
        return "b"
    if side == "b":
        return "a"
    return None


def _bidirectional_confidence(
    forward: dict[str, Any],
    reverse: dict[str, Any],
    selected_source: str,
) -> float:
    if selected_source == "conflict_tie":
        return 0.0
    if selected_source == "forward_confidence":
        return float(forward["confidence"])
    if selected_source == "reverse_confidence":
        return float(reverse["confidence"])
    return _max_optional(forward.get("confidence"), reverse.get("confidence")) or 0.5


def _elo_pairwise_ratings(
    papers: list[Paper],
    judgments: list[dict[str, Any]],
    *,
    initial_rating: float,
    k_factor: float,
) -> dict[str, float]:
    ratings = {paper.paper_id: initial_rating for paper in papers}
    for judgment in judgments:
        winner_id = judgment.get("winner_id")
        loser_id = judgment.get("loser_id")
        if winner_id not in ratings or loser_id not in ratings:
            continue
        winner_rating = ratings[winner_id]
        loser_rating = ratings[loser_id]
        expected_winner = 1.0 / (1.0 + 10 ** ((loser_rating - winner_rating) / 400.0))
        expected_loser = 1.0 - expected_winner
        ratings[winner_id] = winner_rating + k_factor * (1.0 - expected_winner)
        ratings[loser_id] = loser_rating + k_factor * (0.0 - expected_loser)
    return ratings


def _rank_centrality_impact_values(
    papers: list[Paper],
    judgments: list[dict[str, Any]],
    *,
    damping: float = 0.05,
    iterations: int = 200,
    tolerance: float = 1e-12,
) -> tuple[dict[str, float], dict[str, float]]:
    paper_ids = [paper.paper_id for paper in papers]
    paper_id_set = set(paper_ids)
    if not paper_ids:
        return {}, {}

    pair_totals: defaultdict[tuple[str, str], float] = defaultdict(float)
    directed_wins: defaultdict[tuple[str, str], float] = defaultdict(float)
    neighbors: dict[str, set[str]] = {paper_id: set() for paper_id in paper_ids}
    for judgment in judgments:
        winner_id = judgment.get("winner_id")
        loser_id = judgment.get("loser_id")
        if winner_id not in paper_id_set or loser_id not in paper_id_set:
            continue
        weight = _pairwise_judgment_weight(judgment)
        if weight <= 0.0:
            continue
        pair_key = tuple(sorted((str(winner_id), str(loser_id))))
        pair_totals[pair_key] += weight
        directed_wins[(str(winner_id), str(loser_id))] += weight
        neighbors[str(winner_id)].add(str(loser_id))
        neighbors[str(loser_id)].add(str(winner_id))

    if not pair_totals:
        neutral = {paper_id: 50.0 for paper_id in paper_ids}
        uniform = 1.0 / len(paper_ids)
        return neutral, {paper_id: uniform for paper_id in paper_ids}

    d_max = max((len(value) for value in neighbors.values()), default=0)
    if d_max <= 0:
        neutral = {paper_id: 50.0 for paper_id in paper_ids}
        uniform = 1.0 / len(paper_ids)
        return neutral, {paper_id: uniform for paper_id in paper_ids}

    index_by_id = {paper_id: index for index, paper_id in enumerate(paper_ids)}
    transitions: dict[int, list[tuple[int, float]]] = {index: [] for index in range(len(paper_ids))}
    for pair_key, total in pair_totals.items():
        left, right = pair_key
        if total <= 0.0:
            continue
        left_index = index_by_id[left]
        right_index = index_by_id[right]
        right_over_left = directed_wins.get((right, left), 0.0) / total
        left_over_right = directed_wins.get((left, right), 0.0) / total
        transitions[left_index].append((right_index, right_over_left / d_max))
        transitions[right_index].append((left_index, left_over_right / d_max))

    n = len(paper_ids)
    distribution = [1.0 / n for _ in paper_ids]
    damping = min(max(damping, 0.0), 1.0)
    for _ in range(iterations):
        next_distribution = [damping / n for _ in paper_ids]
        for source_index, source_probability in enumerate(distribution):
            raw_outgoing = sum(probability for _, probability in transitions[source_index])
            for target_index, probability in transitions[source_index]:
                next_distribution[target_index] += (
                    source_probability * (1.0 - damping) * probability
                )
            self_probability = max(1.0 - raw_outgoing, 0.0)
            next_distribution[source_index] += (
                source_probability * (1.0 - damping) * self_probability
            )
        delta = sum(
            abs(next_distribution[index] - distribution[index])
            for index in range(n)
        )
        distribution = next_distribution
        if delta < tolerance:
            break

    probabilities = {
        paper_id: distribution[index]
        for paper_id, index in index_by_id.items()
    }
    return _normalize_rating_to_100(probabilities), probabilities


def _pairwise_judgment_weight(judgment: dict[str, Any]) -> float:
    value = judgment.get("calibrated_confidence")
    if value is None:
        value = judgment.get("confidence")
    if value is None:
        return 1.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(number):
        return 1.0
    return min(max(number, 0.0), 1.0)


def _pairwise_scoring_method_name(pairwise_aggregation_method: str) -> str:
    if pairwise_aggregation_method == "rank_centrality":
        return "rank_centrality_v1"
    if pairwise_aggregation_method == "elo":
        return "pairwise_model_elo_v1"
    return str(pairwise_aggregation_method)


def aggregate_pairwise_impact_index(
    papers: list[Paper],
    judgments: list[dict[str, Any]],
    *,
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
    pairwise_aggregation_method: str = "rank_centrality",
) -> list[dict[str, Any]]:
    if pairwise_aggregation_method not in {"rank_centrality", "elo"}:
        raise ValueError("pairwise_aggregation_method must be 'rank_centrality' or 'elo'")
    ratings = _elo_pairwise_ratings(
        papers,
        judgments,
        initial_rating=initial_rating,
        k_factor=k_factor,
    )
    if pairwise_aggregation_method == "rank_centrality":
        impact_values, rank_centrality_probabilities = _rank_centrality_impact_values(
            papers,
            judgments,
        )
    else:
        impact_values = _normalize_rating_to_100(ratings)
        rank_centrality_probabilities = {}
    wins = {paper.paper_id: 0 for paper in papers}
    losses = {paper.paper_id: 0 for paper in papers}
    comparisons = {paper.paper_id: 0 for paper in papers}
    confidence_sums = {paper.paper_id: 0.0 for paper in papers}
    calibrated_confidence_sums = {paper.paper_id: 0.0 for paper in papers}
    calibrated_confidence_counts = {paper.paper_id: 0 for paper in papers}
    metadata = {paper.paper_id: paper for paper in papers}

    for judgment in judgments:
        winner_id = judgment.get("winner_id")
        loser_id = judgment.get("loser_id")
        if winner_id not in metadata or loser_id not in metadata:
            continue
        wins[winner_id] += 1
        losses[loser_id] += 1
        comparisons[winner_id] += 1
        comparisons[loser_id] += 1
        pairwise_confidence = float(judgment.get("confidence", 0.5))
        confidence_sums[winner_id] += pairwise_confidence
        confidence_sums[loser_id] += pairwise_confidence
        calibrated_pairwise_confidence = judgment.get("calibrated_confidence")
        if calibrated_pairwise_confidence is not None:
            calibrated_pairwise_confidence = float(calibrated_pairwise_confidence)
            calibrated_confidence_sums[winner_id] += calibrated_pairwise_confidence
            calibrated_confidence_sums[loser_id] += calibrated_pairwise_confidence
            calibrated_confidence_counts[winner_id] += 1
            calibrated_confidence_counts[loser_id] += 1

    max_comparisons = max(comparisons.values(), default=0)
    rows = []
    for paper_id, rating in ratings.items():
        paper = metadata[paper_id]
        coverage_confidence = _confidence(comparisons[paper_id], max_comparisons)
        mean_pairwise_confidence = _safe_divide_float(
            confidence_sums[paper_id],
            comparisons[paper_id],
            default=0.5,
        )
        mean_calibrated_pairwise_confidence = (
            _safe_divide_float(
                calibrated_confidence_sums[paper_id],
                calibrated_confidence_counts[paper_id],
                default=0.5,
            )
            if calibrated_confidence_counts[paper_id]
            else None
        )
        confidence_source = (
            "calibrated_pairwise"
            if mean_calibrated_pairwise_confidence is not None
            else "raw_pairwise"
        )
        confidence_pairwise_component = (
            mean_calibrated_pairwise_confidence
            if mean_calibrated_pairwise_confidence is not None
            else mean_pairwise_confidence
        )
        rows.append(
            {
                "paper_id": paper_id,
                "title": paper.title,
                "impact_index": round(impact_values[paper_id], 6),
                "elo_rating": round(rating, 6),
                "rank_centrality_probability": round(
                    rank_centrality_probabilities.get(paper_id, 0.0),
                    12,
                )
                if rank_centrality_probabilities
                else None,
                "pairwise_aggregation_method": pairwise_aggregation_method,
                "wins": wins[paper_id],
                "losses": losses[paper_id],
                "comparisons": comparisons[paper_id],
                "coverage_confidence": round(coverage_confidence, 6),
                "mean_pairwise_confidence": round(mean_pairwise_confidence, 6),
                "mean_calibrated_pairwise_confidence": round(
                    mean_calibrated_pairwise_confidence,
                    6,
                )
                if mean_calibrated_pairwise_confidence is not None
                else None,
                "confidence": round(
                    coverage_confidence * confidence_pairwise_component,
                    6,
                ),
                "confidence_source": confidence_source,
            }
        )
    rows.sort(key=lambda row: (-row["impact_index"], row["paper_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def pairwise_bootstrap_stability_report(
    papers: list[Paper],
    judgments: list[dict[str, Any]],
    base_rankings: list[dict[str, Any]],
    *,
    samples: int = 100,
    seed: int = 17,
    top_n: int = 20,
    pairwise_aggregation_method: str = "rank_centrality",
    extra_tracked_paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    decisive_judgments = [
        judgment
        for judgment in judgments
        if judgment.get("winner_id") and judgment.get("loser_id")
        and judgment.get("decisive", judgment.get("parseable", True))
    ]
    tracked_rankings = sorted(
        base_rankings,
        key=lambda row: (int(row.get("rank") or 10**9), str(row.get("paper_id", ""))),
    )[: max(top_n, 0)]
    base_by_id = {str(row["paper_id"]): row for row in base_rankings}
    tracked_ids = [str(row["paper_id"]) for row in tracked_rankings]
    base_top_ids = set(tracked_ids)
    for paper_id in extra_tracked_paper_ids or []:
        paper_id = str(paper_id)
        if paper_id in base_by_id and paper_id not in tracked_ids:
            tracked_ids.append(paper_id)
    if samples <= 0:
        return _disabled_pairwise_stability_report(
            top_n=top_n,
            seed=seed,
            decisive_judgment_count=len(decisive_judgments),
            reason="Pairwise stability bootstrap was disabled.",
        )
    if not tracked_ids or not decisive_judgments:
        return _disabled_pairwise_stability_report(
            top_n=top_n,
            seed=seed,
            decisive_judgment_count=len(decisive_judgments),
            reason="No tracked papers or decisive pairwise judgments were available.",
            status="insufficient_samples",
        )

    rng = random.Random(seed)
    score_samples: dict[str, list[float]] = {paper_id: [] for paper_id in tracked_ids}
    rank_samples: dict[str, list[int]] = {paper_id: [] for paper_id in tracked_ids}
    top_n_retention_rates: list[float] = []
    missing_rank = len(base_rankings) + 1
    effective_top_n = max(len(base_top_ids), 1)

    for _sample_index in range(samples):
        sampled_judgments = [
            rng.choice(decisive_judgments) for _ in decisive_judgments
        ]
        sampled_rankings = aggregate_pairwise_impact_index(
            papers,
            sampled_judgments,
            pairwise_aggregation_method=pairwise_aggregation_method,
        )
        sampled_rank_by_id = {
            str(row["paper_id"]): int(row.get("rank") or missing_rank)
            for row in sampled_rankings
        }
        sampled_score_by_id = {
            str(row["paper_id"]): float(row.get("impact_index") or 0.0)
            for row in sampled_rankings
        }
        sampled_top_ids = {
            str(row["paper_id"]) for row in sampled_rankings[:effective_top_n]
        }
        top_n_retention_rates.append(
            len(base_top_ids & sampled_top_ids) / effective_top_n
        )
        for paper_id in tracked_ids:
            score_samples[paper_id].append(sampled_score_by_id.get(paper_id, 0.0))
            rank_samples[paper_id].append(sampled_rank_by_id.get(paper_id, missing_rank))

    rows = []
    for paper_id in tracked_ids:
        scores = sorted(score_samples[paper_id])
        ranks = rank_samples[paper_id]
        score_ci_low = _quantile(scores, 0.025)
        score_ci_high = _quantile(scores, 0.975)
        best_rank = min(ranks) if ranks else int(base_by_id[paper_id].get("rank") or 0)
        worst_rank = max(ranks) if ranks else int(base_by_id[paper_id].get("rank") or 0)
        rank_span = worst_rank - best_rank
        top_n_membership_rate = (
            sum(1 for rank in ranks if rank <= effective_top_n) / len(ranks)
            if ranks
            else 0.0
        )
        base_score = float(base_by_id[paper_id].get("impact_index") or 0.0)
        rows.append(
            {
                "paper_id": paper_id,
                "title": base_by_id[paper_id].get("title", ""),
                "base_rank": int(base_by_id[paper_id].get("rank") or 0),
                "base_model_pairwise_score": round(base_score, 6),
                "score_mean": round(sum(scores) / len(scores), 6) if scores else base_score,
                "score_ci_low": round(score_ci_low, 6),
                "score_ci_high": round(score_ci_high, 6),
                "score_ci_width": round(score_ci_high - score_ci_low, 6),
                "best_rank": best_rank,
                "worst_rank": worst_rank,
                "rank_span": rank_span,
                "top_n_membership_rate": round(top_n_membership_rate, 6),
                "stability_band": _pairwise_stability_band(
                    samples=samples,
                    rank_span=rank_span,
                    top_n_membership_rate=top_n_membership_rate,
                ),
            }
        )

    mean_retention = (
        sum(top_n_retention_rates) / len(top_n_retention_rates)
        if top_n_retention_rates
        else 0.0
    )
    status, reason = _pairwise_stability_status(samples, mean_retention, rows)
    return {
        "summary": {
            "method": "pairwise_judgment_bootstrap",
            "status": status,
            "reason": reason,
            "samples": samples,
            "seed": seed,
            "top_n": top_n,
            "tracked_paper_count": len(rows),
            "decisive_judgment_count": len(decisive_judgments),
            "mean_top_n_retention_rate": round(mean_retention, 6),
            "stable_paper_count": sum(
                1 for row in rows if row["stability_band"] == "stable"
            ),
            "unstable_paper_count": sum(
                1 for row in rows if row["stability_band"] == "unstable"
            ),
        },
        "papers": rows,
        "limitations": [
            "Bootstrap stability measures sensitivity to sampled pairwise judgments, not correctness against true future impact.",
            "Stable model-pairwise rankings can still be wrong if the local judge is biased.",
            "The configured top-N base model-pairwise papers are tracked; optional extra tracked papers do not change the top-N retention denominator.",
        ],
    }


def final_score_bootstrap_stability_report(
    papers: list[Paper],
    judgments: list[dict[str, Any]],
    base_rankings: list[dict[str, Any]],
    *,
    samples: int = 100,
    seed: int = 23,
    top_n: int = 20,
    pairwise_aggregation_method: str = "rank_centrality",
    weights: dict[str, float] | None = None,
    evidence_weights: dict[str, float] | None = None,
    impact_index_v3_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    decisive_judgments = [
        judgment
        for judgment in judgments
        if judgment.get("winner_id") and judgment.get("loser_id")
        and judgment.get("decisive", judgment.get("parseable", True))
    ]
    tracked_rankings = sorted(
        base_rankings,
        key=lambda row: (int(row.get("rank") or 10**9), str(row.get("paper_id", ""))),
    )[: max(top_n, 0)]
    tracked_ids = [str(row["paper_id"]) for row in tracked_rankings]
    base_by_id = {str(row["paper_id"]): row for row in base_rankings}
    if samples <= 0:
        return _disabled_final_score_stability_report(
            top_n=top_n,
            seed=seed,
            decisive_judgment_count=len(decisive_judgments),
            reason="Final score stability bootstrap was disabled.",
        )
    if not tracked_ids or not decisive_judgments:
        return _disabled_final_score_stability_report(
            top_n=top_n,
            seed=seed,
            decisive_judgment_count=len(decisive_judgments),
            reason="No tracked papers or decisive pairwise judgments were available.",
            status="insufficient_samples",
        )

    rng = random.Random(seed)
    score_samples: dict[str, list[float]] = {paper_id: [] for paper_id in tracked_ids}
    rank_samples: dict[str, list[int]] = {paper_id: [] for paper_id in tracked_ids}
    top_n_retention_rates: list[float] = []
    missing_rank = len(base_rankings) + 1
    base_top_ids = set(tracked_ids)
    effective_top_n = max(len(tracked_ids), 1)

    for _sample_index in range(samples):
        sampled_judgments = [
            rng.choice(decisive_judgments) for _ in decisive_judgments
        ]
        sampled_pairwise_rankings = aggregate_pairwise_impact_index(
            papers,
            sampled_judgments,
            pairwise_aggregation_method=pairwise_aggregation_method,
        )
        sampled_fusion = compute_impact_index_v2(
            papers,
            sampled_pairwise_rankings,
            weights=weights,
            evidence_weights=evidence_weights,
            impact_index_v3_weights=impact_index_v3_weights,
        )
        sampled_rankings = sampled_fusion["rankings"]
        sampled_rank_by_id = {
            str(row["paper_id"]): int(row.get("rank") or missing_rank)
            for row in sampled_rankings
        }
        sampled_score_by_id = {
            str(row["paper_id"]): float(row.get("final_impact_score") or 0.0)
            for row in sampled_rankings
        }
        sampled_top_ids = {
            str(row["paper_id"]) for row in sampled_rankings[:effective_top_n]
        }
        top_n_retention_rates.append(
            len(base_top_ids & sampled_top_ids) / effective_top_n
        )
        for paper_id in tracked_ids:
            score_samples[paper_id].append(sampled_score_by_id.get(paper_id, 0.0))
            rank_samples[paper_id].append(sampled_rank_by_id.get(paper_id, missing_rank))

    rows = []
    for paper_id in tracked_ids:
        scores = sorted(score_samples[paper_id])
        ranks = rank_samples[paper_id]
        score_ci_low = _quantile(scores, 0.025)
        score_ci_high = _quantile(scores, 0.975)
        best_rank = min(ranks) if ranks else int(base_by_id[paper_id].get("rank") or 0)
        worst_rank = max(ranks) if ranks else int(base_by_id[paper_id].get("rank") or 0)
        rank_span = worst_rank - best_rank
        top_n_membership_rate = (
            sum(1 for rank in ranks if rank <= effective_top_n) / len(ranks)
            if ranks
            else 0.0
        )
        base_score = float(base_by_id[paper_id].get("final_impact_score") or 0.0)
        rows.append(
            {
                "paper_id": paper_id,
                "title": base_by_id[paper_id].get("title", ""),
                "base_rank": int(base_by_id[paper_id].get("rank") or 0),
                "base_final_impact_score": round(base_score, 6),
                "score_mean": round(sum(scores) / len(scores), 6) if scores else base_score,
                "score_ci_low": round(score_ci_low, 6),
                "score_ci_high": round(score_ci_high, 6),
                "score_ci_width": round(score_ci_high - score_ci_low, 6),
                "best_rank": best_rank,
                "worst_rank": worst_rank,
                "rank_span": rank_span,
                "top_n_membership_rate": round(top_n_membership_rate, 6),
                "stability_band": _final_score_stability_band(
                    samples=samples,
                    rank_span=rank_span,
                    top_n_membership_rate=top_n_membership_rate,
                    score_ci_width=score_ci_high - score_ci_low,
                ),
            }
        )

    mean_retention = (
        sum(top_n_retention_rates) / len(top_n_retention_rates)
        if top_n_retention_rates
        else 0.0
    )
    status, reason = _final_score_stability_status(samples, mean_retention, rows)
    return {
        "summary": {
            "method": "final_score_pairwise_bootstrap",
            "status": status,
            "reason": reason,
            "samples": samples,
            "seed": seed,
            "top_n": top_n,
            "tracked_paper_count": len(rows),
            "decisive_judgment_count": len(decisive_judgments),
            "mean_top_n_retention_rate": round(mean_retention, 6),
            "stable_paper_count": sum(
                1 for row in rows if row["stability_band"] == "stable"
            ),
            "unstable_paper_count": sum(
                1 for row in rows if row["stability_band"] == "unstable"
            ),
        },
        "papers": rows,
        "limitations": [
            "Final score bootstrap varies only pairwise judgments; citation and external evidence are held fixed.",
            "This is a stability audit for the dataset-local scoring pipeline, not validation against true future impact.",
            "Stable final scores can still be wrong if fixed external evidence or model judgments are biased.",
        ],
    }


def _disabled_final_score_stability_report(
    *,
    top_n: int,
    seed: int,
    decisive_judgment_count: int,
    reason: str,
    status: str = "disabled",
) -> dict[str, Any]:
    return {
        "summary": {
            "method": "final_score_pairwise_bootstrap",
            "status": status,
            "reason": reason,
            "samples": 0,
            "seed": seed,
            "top_n": top_n,
            "tracked_paper_count": 0,
            "decisive_judgment_count": decisive_judgment_count,
            "mean_top_n_retention_rate": None,
            "stable_paper_count": 0,
            "unstable_paper_count": 0,
        },
        "papers": [],
        "limitations": [
            "Final score bootstrap stability was not computed for this run.",
        ],
    }


def _disabled_pairwise_stability_report(
    *,
    top_n: int,
    seed: int,
    decisive_judgment_count: int,
    reason: str,
    status: str = "disabled",
) -> dict[str, Any]:
    return {
        "summary": {
            "method": "pairwise_judgment_bootstrap",
            "status": status,
            "reason": reason,
            "samples": 0,
            "seed": seed,
            "top_n": top_n,
            "tracked_paper_count": 0,
            "decisive_judgment_count": decisive_judgment_count,
            "mean_top_n_retention_rate": None,
            "stable_paper_count": 0,
            "unstable_paper_count": 0,
        },
        "papers": [],
        "limitations": [
            "Pairwise bootstrap stability was not computed for this run.",
        ],
    }


def _pairwise_stability_status(
    samples: int,
    mean_top_n_retention_rate: float,
    rows: list[dict[str, Any]],
) -> tuple[str, str]:
    if samples < 20:
        return (
            "insufficient_samples",
            "Fewer than 20 bootstrap samples were used.",
        )
    unstable_count = sum(1 for row in rows if row["stability_band"] == "unstable")
    if mean_top_n_retention_rate >= 0.85 and unstable_count == 0:
        return (
            "stable",
            "Bootstrap resampling retains the top-N model-pairwise set with high consistency.",
        )
    if mean_top_n_retention_rate >= 0.65:
        return (
            "moderate",
            "Bootstrap resampling shows some pairwise-ranking movement; disclose stability limits.",
        )
    return (
        "unstable",
        "Bootstrap resampling changes the top-N model-pairwise set substantially.",
    )


def _pairwise_stability_band(
    *,
    samples: int,
    rank_span: int,
    top_n_membership_rate: float,
) -> str:
    if samples < 20:
        return "insufficient_samples"
    if top_n_membership_rate >= 0.90 and rank_span <= 5:
        return "stable"
    if top_n_membership_rate >= 0.70 and rank_span <= 15:
        return "moderate"
    return "unstable"


def _final_score_stability_status(
    samples: int,
    mean_top_n_retention_rate: float,
    rows: list[dict[str, Any]],
) -> tuple[str, str]:
    if samples < 20:
        return (
            "insufficient_samples",
            "Fewer than 20 bootstrap samples were used.",
        )
    unstable_count = sum(1 for row in rows if row["stability_band"] == "unstable")
    if mean_top_n_retention_rate >= 0.90 and unstable_count == 0:
        return (
            "stable",
            "Bootstrap resampling retains the top-N final-score set with high consistency.",
        )
    if mean_top_n_retention_rate >= 0.70:
        return (
            "moderate",
            "Bootstrap resampling shows some final-score movement; disclose stability limits.",
        )
    return (
        "unstable",
        "Bootstrap resampling changes the top-N final-score set substantially.",
    )


def _final_score_stability_band(
    *,
    samples: int,
    rank_span: int,
    top_n_membership_rate: float,
    score_ci_width: float,
) -> str:
    if samples < 20:
        return "insufficient_samples"
    if top_n_membership_rate >= 0.90 and rank_span <= 5 and score_ci_width <= 20.0:
        return "stable"
    if top_n_membership_rate >= 0.70 and rank_span <= 15 and score_ci_width <= 40.0:
        return "moderate"
    return "unstable"


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    probability = min(max(probability, 0.0), 1.0)
    index = (len(values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[int(index)]
    lower_value = values[lower]
    upper_value = values[upper]
    return lower_value + (upper_value - lower_value) * (index - lower)


def build_pairwise_impact_prompt(paper_a: Paper, paper_b: Paper) -> str:
    title_a = _truncate_text_for_prompt(
        paper_a.title or "",
        PAIRWISE_TITLE_CHAR_BUDGET,
    )
    title_b = _truncate_text_for_prompt(
        paper_b.title or "",
        PAIRWISE_TITLE_CHAR_BUDGET,
    )
    venue_a = _truncate_text_for_prompt(
        paper_a.venue or "",
        PAIRWISE_SHORT_FIELD_CHAR_BUDGET,
    )
    venue_b = _truncate_text_for_prompt(
        paper_b.venue or "",
        PAIRWISE_SHORT_FIELD_CHAR_BUDGET,
    )
    abstract_a = _truncate_text_for_prompt(
        paper_a.abstract or "",
        PAIRWISE_ABSTRACT_CHAR_BUDGET,
    )
    abstract_b = _truncate_text_for_prompt(
        paper_b.abstract or "",
        PAIRWISE_ABSTRACT_CHAR_BUDGET,
    )
    return (
        "Which paper has higher scholarly impact within this dataset? "
        "Consider likely influence, methodological contribution, adoption potential, "
        "and breadth of downstream use. Use only the provided metadata.\n\n"
        f"Paper A ID: {paper_a.paper_id}\n"
        f"Title: {title_a}\n"
        f"Year: {paper_a.year or ''}\n"
        f"Venue: {venue_a}\n"
        f"Abstract: {abstract_a}\n\n"
        f"Paper B ID: {paper_b.paper_id}\n"
        f"Title: {title_b}\n"
        f"Year: {paper_b.year or ''}\n"
        f"Venue: {venue_b}\n"
        f"Abstract: {abstract_b}\n\n"
        "Return exactly one line: Answer: A or Answer: B"
    )


def build_direct_impact_score_prompt(paper: Paper) -> str:
    title = _truncate_text_for_prompt(
        paper.title or "",
        DIRECT_TITLE_CHAR_BUDGET,
    )
    venue = _truncate_text_for_prompt(
        paper.venue or "",
        DIRECT_SHORT_FIELD_CHAR_BUDGET,
    )
    field = _truncate_text_for_prompt(
        paper.field or "",
        DIRECT_SHORT_FIELD_CHAR_BUDGET,
    )
    abstract = _truncate_text_for_prompt(
        paper.abstract or "",
        DIRECT_ABSTRACT_CHAR_BUDGET,
    )
    return (
        "Score the likely scholarly impact of this paper within the user's local "
        "dataset on a 0 to 100 scale. Use only the provided metadata. "
        "A higher score means stronger likely influence, methodological contribution, "
        "adoption potential, and breadth of downstream use. Do not compare against "
        "global citation counts unless they are provided.\n\n"
        f"Paper ID: {paper.paper_id}\n"
        f"Title: {title}\n"
        f"Year: {paper.year or ''}\n"
        f"Venue: {venue}\n"
        f"Field: {field}\n"
        f"Abstract: {abstract}\n\n"
        "Return exactly one line using one of these calibrated bins: "
        "Score: 0, Score: 20, Score: 40, Score: 60, Score: 80, Score: 100"
    )


def _truncate_text_for_prompt(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 32:
        return cleaned[:max_chars]
    marker = " ... [truncated] ... "
    side_budget = max((max_chars - len(marker)) // 2, 1)
    return cleaned[:side_budget].rstrip() + marker + cleaned[-side_budget:].lstrip()


def direct_score_choices() -> list[str]:
    return [
        "Score: 0",
        "Score: 20",
        "Score: 40",
        "Score: 60",
        "Score: 80",
        "Score: 100",
    ]


def parse_pairwise_winner(prediction: str | None) -> str | None:
    if prediction is None:
        return None
    answer_matches = list(re.finditer(r"\bAnswer\s*:\s*([AB])\b", prediction, re.I))
    if answer_matches:
        return answer_matches[-1].group(1).lower()
    paper_matches = list(re.finditer(r"\bPaper\s+([AB])\b", prediction, re.I))
    if paper_matches:
        return paper_matches[-1].group(1).lower()
    return None


def parse_direct_impact_score(prediction: str | None) -> float | None:
    if prediction is None:
        return None
    score_match = re.search(r"\bScore\s*:\s*(-?\d+(?:\.\d+)?)\b", prediction, re.I)
    if score_match is None:
        score_match = re.search(r"\b(-?\d+(?:\.\d+)?)\s*/\s*100\b", prediction)
    if score_match is None:
        return None
    score = float(score_match.group(1))
    if score < 0.0 or score > 100.0:
        return None
    return score


def direct_score_quality_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        float(row["direct_score"])
        for row in rows
        if row.get("direct_score") is not None and row.get("parseable")
    ]
    if not rows:
        return {
            "status": "missing",
            "unique_score_count": 0,
            "score_range": 0.0,
            "reason": "Direct scoring was enabled but produced no rows.",
        }
    if not scores:
        return {
            "status": "unparseable",
            "unique_score_count": 0,
            "score_range": 0.0,
            "reason": "No direct model scores were parseable; treat as diagnostic only.",
        }
    unique_score_count = len(set(scores))
    score_range = max(scores) - min(scores)
    if unique_score_count <= 1 or math.isclose(score_range, 0.0):
        return {
            "status": "degenerate",
            "unique_score_count": unique_score_count,
            "score_range": round(score_range, 6),
            "reason": "Direct model scores collapsed to a single value; treat as diagnostic only.",
        }
    return {
        "status": "usable",
        "unique_score_count": unique_score_count,
        "score_range": round(score_range, 6),
        "reason": "Direct model scores have variation; use as an initial prior, not final calibration.",
    }


def pairwise_ranking_quality_report(
    papers: list[Paper],
    judgments: list[dict[str, Any]],
    *,
    decisive_judgment_count: int | None = None,
) -> dict[str, Any]:
    comparisons = {paper.paper_id: 0 for paper in papers}
    adjacency = {paper.paper_id: set() for paper in papers}
    decisive_count = 0
    for judgment in judgments:
        if not judgment.get("decisive", judgment.get("parseable", False)):
            continue
        winner_id = judgment.get("winner_id")
        loser_id = judgment.get("loser_id")
        if winner_id not in comparisons or loser_id not in comparisons:
            continue
        comparisons[winner_id] += 1
        comparisons[loser_id] += 1
        adjacency[winner_id].add(loser_id)
        adjacency[loser_id].add(winner_id)
        decisive_count += 1
    if decisive_judgment_count is not None:
        decisive_count = decisive_judgment_count

    paper_count = len(papers)
    compared_paper_count = sum(1 for count in comparisons.values() if count > 0)
    uncompared_paper_count = paper_count - compared_paper_count
    paper_coverage_rate = _safe_divide(compared_paper_count, paper_count)
    min_comparisons = min(comparisons.values(), default=0)
    max_comparisons = max(comparisons.values(), default=0)
    mean_comparisons = _safe_divide(sum(comparisons.values()), paper_count)
    minimum_recommended_decisive_judgments = paper_count
    minimum_recommended_comparisons_per_paper = 2 if paper_count >= 3 else int(paper_count > 1)
    graph_quality = _pairwise_graph_quality(adjacency)

    if paper_count == 0:
        status = "missing"
        reason = "No papers were provided."
    elif uncompared_paper_count > 0:
        status = "exploratory"
        reason = (
            f"Only {compared_paper_count} of {paper_count} papers have decisive "
            "pairwise comparisons; increase --max-pairs or --comparison-rounds "
            "before treating the index as a full ranking."
        )
    elif graph_quality["connected_component_count"] > 1:
        status = "exploratory"
        reason = (
            f"The decisive pairwise comparison graph has "
            f"{graph_quality['connected_component_count']} connected components; "
            "Rank Centrality scores across disconnected components are not comparable."
        )
    elif min_comparisons < minimum_recommended_comparisons_per_paper:
        status = "limited"
        reason = (
            "All papers are connected, but at least one paper has only "
            f"{min_comparisons} decisive comparison(s); use at least "
            f"{minimum_recommended_comparisons_per_paper} per paper for a more "
            "stable index."
        )
    elif decisive_count < minimum_recommended_decisive_judgments:
        status = "limited"
        reason = (
            f"All papers were compared at least once, but only {decisive_count} "
            f"decisive judgments were made for {paper_count} papers; use more "
            "comparisons for a more stable index."
        )
    else:
        status = "reportable"
        reason = "All papers have decisive pairwise comparison coverage."

    return {
        "status": status,
        "reason": reason,
        "compared_paper_count": compared_paper_count,
        "uncompared_paper_count": uncompared_paper_count,
        "paper_coverage_rate": round(paper_coverage_rate, 6),
        "min_comparisons_per_paper": min_comparisons,
        "max_comparisons_per_paper": max_comparisons,
        "mean_comparisons_per_paper": round(mean_comparisons, 6),
        "decisive_judgment_count": decisive_count,
        "minimum_recommended_decisive_judgments": minimum_recommended_decisive_judgments,
        "minimum_recommended_comparisons_per_paper": minimum_recommended_comparisons_per_paper,
        **graph_quality,
    }


def _pairwise_graph_quality(adjacency: dict[str, set[str]]) -> dict[str, Any]:
    if not adjacency:
        return {
            "connected_component_count": 0,
            "largest_connected_component_paper_count": 0,
            "largest_connected_component_coverage": 0.0,
            "isolated_paper_count": 0,
        }
    visited: set[str] = set()
    component_sizes: list[int] = []
    for paper_id in adjacency:
        if paper_id in visited:
            continue
        stack = [paper_id]
        visited.add(paper_id)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        component_sizes.append(size)
    largest = max(component_sizes, default=0)
    paper_count = len(adjacency)
    return {
        "connected_component_count": len(component_sizes),
        "largest_connected_component_paper_count": largest,
        "largest_connected_component_coverage": round(
            largest / paper_count if paper_count else 0.0,
            6,
        ),
        "isolated_paper_count": sum(1 for neighbors in adjacency.values() if not neighbors),
    }


def render_model_impact_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Model Impact Index",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in report["summary"].items():
        if key in {"pair_budget", "ranking_quality"} and isinstance(value, dict):
            for quality_key, quality_value in value.items():
                lines.append(f"| {key}.{quality_key} | {quality_value} |")
        elif key == "fusion" and isinstance(value, dict):
            lines.append(f"| fusion.version | {value.get('version', '')} |")
            lines.append(f"| fusion.claim_scope | {value.get('claim_scope', '')} |")
            reportability = value.get("reportability", {})
            if isinstance(reportability, dict):
                lines.append(
                    f"| fusion.reportability.reportable_rate | "
                    f"{reportability.get('reportable_rate', 0.0)} |"
                )
                lines.append(
                    f"| fusion.reportability.low_confidence_rate | "
                    f"{reportability.get('low_confidence_rate', 0.0)} |"
                )
            for signal_name, signal in value.get("signal_availability", {}).items():
                lines.append(
                    f"| fusion.{signal_name}.coverage_rate | "
                    f"{signal.get('coverage_rate', 0.0)} |"
                )
            evidence_first = value.get("evidence_first", {})
            if isinstance(evidence_first, dict):
                lines.append(
                    f"| fusion.evidence_first.high_or_better_rate | "
                    f"{evidence_first.get('high_or_better_rate', 0.0)} |"
                )
                lines.append(
                    f"| fusion.evidence_first.external_adoption_coverage_rate | "
                    f"{evidence_first.get('external_adoption_coverage_rate', 0.0)} |"
                )
                signal_audit = evidence_first.get("evidence_signal_audit", {})
                if isinstance(signal_audit, dict):
                    lines.append(
                        "| fusion.evidence_signal_audit.mean_positive_evidence_signal_count | "
                        f"{signal_audit.get('mean_positive_evidence_signal_count', 0.0)} |"
                    )
                    lines.append(
                        "| fusion.evidence_signal_audit.mean_zero_evidence_signal_count | "
                        f"{signal_audit.get('mean_zero_evidence_signal_count', 0.0)} |"
                    )
                    lines.append(
                        "| fusion.evidence_signal_audit.papers_with_no_positive_evidence | "
                        f"{signal_audit.get('papers_with_no_positive_evidence', 0)} |"
                    )
        elif key == "external_evidence_readiness" and isinstance(value, dict):
            decision = value.get("decision", {})
            lines.append(
                "| external_evidence_readiness.status | "
                f"{decision.get('status', '')} |"
            )
            lines.append(
                "| external_evidence_readiness.can_claim_high_credibility | "
                f"{decision.get('can_claim_high_credibility', False)} |"
            )
            identifier = value.get("identifier_coverage", {})
            lines.append(
                "| external_evidence_readiness.external_query_id_coverage | "
                f"{identifier.get('external_query_id_coverage', 0.0)} |"
            )
            graph = value.get("graph_evidence", {})
            lines.append(
                "| external_evidence_readiness.internal_reference_edge_count | "
                f"{graph.get('internal_reference_edge_count', 0)} |"
            )
        elif key == "impact_validity_evaluation" and isinstance(value, dict):
            lines.append(f"| Impact Validity.status | {value.get('status', '')} |")
            lines.append(f"| Impact Validity.reason | {value.get('reason', '')} |")
            lines.append(
                "| Impact Validity.score_label_match_rate | "
                f"{value.get('score_label_match_rate', '')} |"
            )
            lines.append(
                "| Impact Validity.significant_vs_random | "
                f"{value.get('significant_vs_random', '')} |"
            )
        elif key == "validated_impact_claim" and isinstance(value, dict):
            lines.append(
                f"| validated_impact_claim.status | {value.get('status', '')} |"
            )
            lines.append(
                "| validated_impact_claim.can_claim_validated_impact_accuracy | "
                f"{value.get('can_claim_validated_impact_accuracy', False)} |"
            )
            lines.append(
                "| validated_impact_claim.confidence_ceiling | "
                f"{value.get('confidence_ceiling', '')} |"
            )
        elif key == "pairwise_stability" and isinstance(value, dict):
            lines.append(f"| pairwise_stability.status | {value.get('status', '')} |")
            lines.append(f"| pairwise_stability.samples | {value.get('samples', 0)} |")
            lines.append(
                "| pairwise_stability.mean_top_n_retention_rate | "
                f"{value.get('mean_top_n_retention_rate', '')} |"
            )
            lines.append(
                "| pairwise_stability.unstable_paper_count | "
                f"{value.get('unstable_paper_count', 0)} |"
            )
        elif key == "final_score_stability" and isinstance(value, dict):
            lines.append(
                f"| final_score_stability.status | {value.get('status', '')} |"
            )
            lines.append(
                f"| final_score_stability.samples | {value.get('samples', 0)} |"
            )
            lines.append(
                "| final_score_stability.mean_top_n_retention_rate | "
                f"{value.get('mean_top_n_retention_rate', '')} |"
            )
            lines.append(
                "| final_score_stability.unstable_paper_count | "
                f"{value.get('unstable_paper_count', 0)} |"
            )
        elif key == "impact_dimension_coverage" and isinstance(value, dict):
            for dimension, dimension_report in value.items():
                lines.append(
                    f"| impact_dimension.{dimension}.coverage_rate | "
                    f"{dimension_report.get('coverage_rate', 0.0)} |"
                )
        else:
            lines.append(f"| {key} | {value} |")
    external_readiness = report["summary"].get("external_evidence_readiness", {})
    if isinstance(external_readiness, dict):
        decision = external_readiness.get("decision", {})
        identifier = external_readiness.get("identifier_coverage", {})
        source = external_readiness.get("source_coverage", {})
        signal = external_readiness.get("signal_coverage", {})
        graph = external_readiness.get("graph_evidence", {})
        paper_coverage = external_readiness.get("paper_coverage_summary", {})
        lines.extend(
            [
                "",
                "## External Evidence Readiness",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| status | {decision.get('status', '')} |",
                f"| can_claim_high_credibility | {decision.get('can_claim_high_credibility', False)} |",
                f"| external_query_id_coverage | {identifier.get('external_query_id_coverage', 0.0)} |",
                f"| openalex_enriched_coverage | {source.get('openalex_enriched_coverage', 0.0)} |",
                f"| opencitations_enriched_coverage | {source.get('opencitations_enriched_coverage', 0.0)} |",
                f"| citation_count_coverage | {signal.get('citation_count_coverage', 0.0)} |",
                f"| field_year_percentile_coverage | {signal.get('field_year_percentile_coverage', 0.0)} |",
                f"| internal_reference_edge_count | {graph.get('internal_reference_edge_count', 0)} |",
                f"| materialized_internal_edge_count | {graph.get('materialized_internal_edge_count', 0)} |",
            ]
        )
        if isinstance(paper_coverage, dict):
            lines.extend(
                [
                    f"| fully_covered_paper_coverage | {paper_coverage.get('fully_covered_paper_coverage', 0.0)} |",
                    f"| papers_with_any_missing_evidence | {paper_coverage.get('papers_with_any_missing_evidence', 0)} |",
                    f"| papers_with_any_missing_evidence_coverage | {paper_coverage.get('papers_with_any_missing_evidence_coverage', 0.0)} |",
                ]
            )
            top_missing = paper_coverage.get("top_missing_evidence", [])
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
                        f"| {item.get('label', '')} | "
                        f"{item.get('paper_count', 0)} | "
                        f"{item.get('paper_coverage', 0.0)} |"
                    )
        lines.append("")
    dimension_coverage = report["summary"].get("impact_dimension_coverage", {})
    fusion_summary = report["summary"].get("fusion", {})
    evidence_first_summary = (
        fusion_summary.get("evidence_first", {})
        if isinstance(fusion_summary, dict)
        else {}
    )
    signal_audit = (
        evidence_first_summary.get("evidence_signal_audit", {})
        if isinstance(evidence_first_summary, dict)
        else {}
    )
    if isinstance(signal_audit, dict) and signal_audit:
        lines.extend(
            [
                "",
                "## Evidence Signal Audit",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| mean_positive_evidence_signal_count | {signal_audit.get('mean_positive_evidence_signal_count', 0.0)} |",
                f"| mean_zero_evidence_signal_count | {signal_audit.get('mean_zero_evidence_signal_count', 0.0)} |",
                f"| papers_with_no_positive_evidence | {signal_audit.get('papers_with_no_positive_evidence', 0)} |",
                f"| papers_with_zero_only_evidence | {signal_audit.get('papers_with_zero_only_evidence', 0)} |",
                "",
                "| Evidence count type | Count | Paper count |",
                "| --- | ---: | ---: |",
            ]
        )
        for count, paper_count in signal_audit.get(
            "positive_signal_count_distribution",
            {},
        ).items():
            lines.append(f"| positive | {count} | {paper_count} |")
        for count, paper_count in signal_audit.get(
            "zero_signal_count_distribution",
            {},
        ).items():
            lines.append(f"| zero | {count} | {paper_count} |")
        for count, paper_count in signal_audit.get(
            "available_signal_count_distribution",
            {},
        ).items():
            lines.append(f"| available | {count} | {paper_count} |")
        interpretation = signal_audit.get("interpretation")
        if interpretation:
            lines.extend(["", str(interpretation)])
    if isinstance(dimension_coverage, dict) and dimension_coverage:
        lines.extend(
            [
                "",
                "## Impact Dimension Coverage",
                "",
                "| Dimension | Claim Type | Covered Papers | Coverage | Claim Boundary |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for dimension, dimension_report in sorted(dimension_coverage.items()):
            lines.append(
                f"| {dimension} | "
                f"{dimension_report.get('claim_type', '')} | "
                f"{dimension_report.get('covered_paper_count', 0)} | "
                f"{float(dimension_report.get('coverage_rate', 0.0)):.3f} | "
                f"{dimension_report.get('claim_boundary', '')} |"
            )
    pairwise_stability = report["summary"].get("pairwise_stability", {})
    if isinstance(pairwise_stability, dict) and pairwise_stability:
        lines.extend(
            [
                "",
                "## Pairwise Stability",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| status | {pairwise_stability.get('status', '')} |",
                f"| samples | {pairwise_stability.get('samples', 0)} |",
                f"| top_n | {pairwise_stability.get('top_n', '')} |",
                f"| mean_top_n_retention_rate | {pairwise_stability.get('mean_top_n_retention_rate', '')} |",
                f"| stable_paper_count | {pairwise_stability.get('stable_paper_count', 0)} |",
                f"| unstable_paper_count | {pairwise_stability.get('unstable_paper_count', 0)} |",
                f"| reason | {pairwise_stability.get('reason', '')} |",
            ]
        )
    final_score_stability = report["summary"].get("final_score_stability", {})
    if isinstance(final_score_stability, dict) and final_score_stability:
        lines.extend(
            [
                "",
                "## Final Score Stability",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| status | {final_score_stability.get('status', '')} |",
                f"| samples | {final_score_stability.get('samples', 0)} |",
                f"| top_n | {final_score_stability.get('top_n', '')} |",
                f"| mean_top_n_retention_rate | {final_score_stability.get('mean_top_n_retention_rate', '')} |",
                f"| stable_paper_count | {final_score_stability.get('stable_paper_count', 0)} |",
                f"| unstable_paper_count | {final_score_stability.get('unstable_paper_count', 0)} |",
                f"| reason | {final_score_stability.get('reason', '')} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Top Papers",
            "",
            "| Rank | Paper ID | Impact Index v3 | Evidence Score | Model Score | External Adoption | Life Science Translation | Citation | Relative Citation | Graph | Context | Effective Citation | Influential Citation | Momentum | Positive Evidence | Zero Evidence | Evidence Tier | Uncertainty | Reportability | Direct Score | Wins | Losses | Confidence | Title |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["rankings"][:20]:
        direct_score = row.get("direct_score")
        direct_score_text = (
            f"{float(direct_score):.3f}" if isinstance(direct_score, int | float) else ""
        )
        lines.append(
            f"| {row['rank']} | {row['paper_id']} | "
            f"{float(row.get('impact_index_v3', row.get('impact_index_v2', 0.0))):.3f} | "
            f"{float(row.get('evidence_supported_score', 0.0)):.3f} | "
            f"{float(row.get('model_supported_score', row.get('impact_index', 0.0))):.3f} | "
            f"{float(row.get('external_adoption_score', 0.0)):.3f} | "
            f"{float(row.get('life_science_translation_score', 0.0)):.3f} | "
            f"{float(row.get('field_year_normalized_citation_score', 0.0)):.3f} | "
            f"{float(row.get('relative_citation_ratio_score', 0.0)):.3f} | "
            f"{float(row.get('citation_graph_score', 0.0)):.3f} | "
            f"{float(row.get('citation_context_quality_score', 0.0)):.3f} | "
            f"{float(row.get('effective_citation_score', 0.0)):.3f} | "
            f"{float(row.get('influential_citation_score', 0.0)):.3f} | "
            f"{float(row.get('temporal_momentum_score', 0.0)):.3f} | "
            f"{int(row.get('positive_evidence_signal_count') or 0)} | "
            f"{int(row.get('zero_evidence_signal_count') or 0)} | "
            f"{row.get('credible_evidence_tier', row.get('evidence_quality_flag', ''))} | "
            f"{row.get('uncertainty_band', '')} | "
            f"{row.get('reportability_status', '')} | "
            f"{direct_score_text} | {row['wins']} | {row['losses']} | "
            f"{float(row.get('confidence', 0.0)):.3f} | "
            f"{row['title']} |"
        )
    lines.extend(
        [
            "",
            "## Compatibility Top Papers",
            "",
            "| Rank | Paper ID | Impact Index v2 | Pairwise Index | Citation | Relative Citation | Graph | Context | Influential Citation | Momentum | Evidence | Uncertainty | Reportability | Direct Score | Wins | Losses | Confidence | Coverage Confidence | Mean Pairwise Confidence | Title |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["rankings"][:20]:
        direct_score = row.get("direct_score")
        direct_score_text = (
            f"{float(direct_score):.3f}" if isinstance(direct_score, int | float) else ""
        )
        lines.append(
            f"| {row['rank']} | {row['paper_id']} | "
            f"{float(row.get('impact_index_v2', 0.0)):.3f} | "
            f"{float(row.get('impact_index', 0.0)):.3f} | "
            f"{float(row.get('field_year_normalized_citation_score', 0.0)):.3f} | "
            f"{float(row.get('relative_citation_ratio_score', 0.0)):.3f} | "
            f"{float(row.get('citation_graph_score', 0.0)):.3f} | "
            f"{float(row.get('citation_context_quality_score', 0.0)):.3f} | "
            f"{float(row.get('influential_citation_score', 0.0)):.3f} | "
            f"{float(row.get('temporal_momentum_score', 0.0)):.3f} | "
            f"{row.get('evidence_quality_flag', '')} | "
            f"{row.get('uncertainty_band', '')} | "
            f"{row.get('reportability_status', '')} | "
            f"{direct_score_text} | {row['wins']} | {row['losses']} | "
            f"{float(row.get('confidence', 0.0)):.3f} | "
            f"{float(row.get('coverage_confidence', 0.0)):.3f} | "
            f"{float(row.get('mean_pairwise_confidence', 0.0)):.3f} | "
            f"{row['title']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Gaps",
            "",
            "| Rank | Paper ID | credible_evidence_tier | dominant_score_family | missing_high_value_evidence | Title |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["rankings"][:20]:
        missing = row.get("missing_high_value_evidence", [])
        if isinstance(missing, list):
            missing_text = ", ".join(str(item) for item in missing) or "none"
        else:
            missing_text = str(missing) if missing else "none"
        lines.append(
            f"| {row['rank']} | {row['paper_id']} | "
            f"{row.get('credible_evidence_tier', '')} | "
            f"{row.get('dominant_score_family', '')} | "
            f"{missing_text} | {row['title']} |"
        )
    lines.extend(
        [
            "",
            "## Scoring Note",
            "",
            "`impact_index_v3` is the raw evidence-first score. `final_impact_score` is the credibility-adjusted score used for sorting and reporting; it discounts model-only or weak-evidence rows.",
            "`impact_index_v2` is retained as the earlier compatibility score for audit and ablation.",
            "`impact_index` is the model-assisted pairwise score. The default aggregation is order-independent Rank Centrality; `elo_rating` remains available as an audit field.",
            "OpenAlex/BIP/Semantic Scholar compatible external evidence can feed field/year percentile, BIP impact, and influential-citation score components when supplied.",
            "Life-science fields such as NIH RCR, PubMed citation counts, clinical-trial citations, guideline mentions, drug-approval mentions, protocol reuse, and reagent/assay reuse are recorded as auditable citation, translation, policy/guideline, and artifact-adoption evidence.",
            "Effective citation fields classify incoming internal citation contexts into guideline, clinical, method, dataset/tool, supporting, negative/contrasting, background, or unknown roles.",
            "Missing signal weights are redistributed per paper, and `evidence_quality_flag`, `uncertainty_band`, and `reportability_status` mark weak or sensitive scores instead of hiding sparse data.",
            "Direct scores are diagnostic local-impact priors and are not used as the final score.",
            "`Confidence` combines comparison coverage and mean pairwise model confidence. If an empirical calibration report is supplied, calibrated pairwise confidence is used for the final confidence field.",
            "When logprobs are unavailable, raw pairwise confidence defaults to 0.5.",
            "",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def impact_score_cards_from_rankings(
    rankings: list[dict[str, Any]],
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    cards = [_impact_score_card(row) for row in rankings[:top_n]]
    return {
        "card_count": len(cards),
        "top_n": top_n,
        "cards": cards,
        "limitations": [
            "Cards explain dataset-local impact scores, not global scientific impact.",
            "Missing evidence lowers reportability and is exposed rather than imputed.",
        ],
    }


def render_impact_score_cards_markdown(report: dict[str, Any]) -> str:
    lines = ["# Impact Score Cards", ""]
    cards = report.get("cards", [])
    if not cards:
        lines.extend(["No impact score cards available.", ""])
        return "\n".join(lines)
    for card in cards:
        components = card.get("component_scores", {})
        raw = card.get("raw_evidence", {})
        missing = card.get("missing_high_value_evidence") or []
        missing_text = ", ".join(str(item) for item in missing) if missing else "none"
        gap_actions = card.get("reportability_gap_actions") or []
        gap_actions_text = (
            "; ".join(str(item) for item in gap_actions) if gap_actions else "none"
        )
        lines.extend(
            [
                f"## {card.get('rank')}. {card.get('title') or card.get('paper_id')}",
                "",
                "| Field | Value |",
                "| --- | ---: |",
                f"| paper_id | {card.get('paper_id', '')} |",
                f"| final_impact_score | {_markdown_number(card.get('final_impact_score'))} |",
                f"| impact_index_v3 | {_markdown_number(card.get('impact_index_v3'))} |",
                f"| evidence_supported_score | {_markdown_number(card.get('evidence_supported_score'))} |",
                f"| model_supported_score | {_markdown_number(card.get('model_supported_score'))} |",
                f"| positive_evidence_signal_count | {card.get('positive_evidence_signal_count', 0)} |",
                f"| zero_evidence_signal_count | {card.get('zero_evidence_signal_count', 0)} |",
                f"| credible_evidence_tier | {card.get('credible_evidence_tier', '')} |",
                f"| reportability_status | {card.get('reportability_status', '')} |",
                f"| reportability_gap_count | {card.get('reportability_gap_count', 0)} |",
                f"| reportability_gap_actions | {gap_actions_text} |",
                f"| confidence | {_markdown_number(card.get('confidence'))} |",
                f"| citation_outlier_ratio | {_markdown_number(raw.get('citation_outlier_ratio'))} |",
                f"| missing_high_value_evidence | {missing_text} |",
                "",
                "| Component | Score | Raw |",
                "| --- | ---: | ---: |",
                f"| Citation | {_markdown_number(components.get('field_year_normalized_citation'))} | {_markdown_number(raw.get('citation_count'))} |",
                f"| Relative Citation | {_markdown_number(components.get('relative_citation_ratio'))} | {_markdown_number(raw.get('relative_citation_ratio'))} |",
                f"| Graph | {_markdown_number(components.get('citation_graph'))} | {_markdown_number(raw.get('citation_graph'))} |",
                f"| Context | {_markdown_number(components.get('citation_context_quality'))} | {_markdown_number(raw.get('citation_context_quality'))} |",
                f"| Effective Citation | {_markdown_number(components.get('effective_citation'))} | {_markdown_number(raw.get('effective_citation'))} |",
                f"| Influential Citation | {_markdown_number(components.get('influential_citation'))} | {_markdown_number(raw.get('influential_citation_count'))} |",
                f"| Momentum | {_markdown_number(components.get('temporal_momentum'))} | {_markdown_number(raw.get('temporal_momentum'))} |",
                f"| External Adoption | {_markdown_number(components.get('external_adoption'))} | {_markdown_number(raw.get('external_adoption'))} |",
                f"| Life Science Translation | {_markdown_number(components.get('life_science_translation'))} | {_markdown_number(raw.get('life_science_translation'))} |",
                f"| BIP | {_markdown_number(components.get('bip'))} | {_markdown_number(raw.get('bip_impact'))} |",
                f"| Media | {_markdown_number(components.get('media'))} | {_markdown_number(raw.get('media_mentions'))} |",
                f"| Award | {_markdown_number(components.get('award'))} | {_markdown_number(raw.get('award_count'))} |",
                "",
            ]
        )
    if report.get("limitations"):
        lines.extend(["## Limitations", ""])
        for limitation in report.get("limitations", []):
            lines.append(f"- {limitation}")
        lines.append("")
    return "\n".join(lines)


def _impact_score_card(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": row.get("rank"),
        "paper_id": row.get("paper_id"),
        "title": row.get("title"),
        "final_impact_score": row.get("final_impact_score"),
        "impact_index_v3": row.get("impact_index_v3"),
        "impact_index_v2": row.get("impact_index_v2"),
        "credibility_adjusted_impact_score": row.get(
            "credibility_adjusted_impact_score"
        ),
        "credibility_adjustment_factor": row.get("credibility_adjustment_factor"),
        "evidence_supported_score": row.get("evidence_supported_score"),
        "model_supported_score": row.get("model_supported_score"),
        "positive_evidence_signal_count": row.get("positive_evidence_signal_count", 0),
        "zero_evidence_signal_count": row.get("zero_evidence_signal_count", 0),
        "credible_evidence_tier": row.get("credible_evidence_tier"),
        "dominant_score_family": row.get("dominant_score_family"),
        "uncertainty_band": row.get("uncertainty_band"),
        "reportability_status": row.get("reportability_status"),
        "reportability_reason": row.get("reportability_reason"),
        "reportability_gap_count": row.get("reportability_gap_count", 0),
        "reportability_gap_actions": row.get("reportability_gap_actions", []),
        "confidence": row.get("confidence"),
        "missing_high_value_evidence": row.get("missing_high_value_evidence", []),
        "component_scores": {
            "field_year_normalized_citation": row.get(
                "field_year_normalized_citation_score",
                0.0,
            ),
            "relative_citation_ratio": row.get("relative_citation_ratio_score", 0.0),
            "citation_graph": row.get("citation_graph_score", 0.0),
            "citation_context_quality": row.get(
                "citation_context_quality_score",
                0.0,
            ),
            "effective_citation": row.get("effective_citation_score", 0.0),
            "influential_citation": row.get("influential_citation_score", 0.0),
            "temporal_momentum": row.get("temporal_momentum_score", 0.0),
            "external_adoption": row.get("external_adoption_score", 0.0),
            "life_science_translation": row.get(
                "life_science_translation_score",
                0.0,
            ),
            "bip": row.get("bip_impact_score", 0.0),
            "media": row.get("media_impact_score", 0.0),
            "award": row.get("award_impact_score", 0.0),
        },
        "raw_evidence": {
            "citation_count": row.get("citation_count_raw"),
            "citation_outlier_flag": row.get("citation_outlier_flag"),
            "citation_outlier_ratio": row.get("citation_outlier_ratio"),
            "field_year_normalized_citation": row.get(
                "field_year_normalized_citation_raw"
            ),
            "relative_citation_ratio": row.get("relative_citation_ratio_raw"),
            "relative_citation_rate": row.get("relative_citation_rate_raw"),
            "relative_expected_citation_rate": row.get(
                "relative_expected_citation_rate_raw"
            ),
            "citation_graph": row.get("citation_graph_raw"),
            "citation_context_quality": row.get("citation_context_quality_raw"),
            "effective_citation": row.get("effective_citation_raw"),
            "effective_citation_edge_count": row.get(
                "effective_citation_edge_count"
            ),
            "effective_citation_context_count": row.get(
                "effective_citation_context_count"
            ),
            "effective_citation_count": row.get("effective_citation_count"),
            "effective_citation_guideline_count": row.get(
                "effective_citation_guideline_count"
            ),
            "effective_citation_clinical_count": row.get(
                "effective_citation_clinical_count"
            ),
            "effective_citation_method_count": row.get(
                "effective_citation_method_count"
            ),
            "effective_citation_dataset_or_tool_count": row.get(
                "effective_citation_dataset_or_tool_count"
            ),
            "effective_citation_background_count": row.get(
                "effective_citation_background_count"
            ),
            "effective_citation_unknown_count": row.get(
                "effective_citation_unknown_count"
            ),
            "effective_citation_coverage": row.get("effective_citation_coverage"),
            "effective_citation_evidence_level": row.get(
                "effective_citation_evidence_level"
            ),
            "influential_citation_count": row.get("influential_citation_count_raw"),
            "temporal_momentum": row.get("temporal_momentum_raw"),
            "external_adoption": row.get("external_adoption_raw"),
            "life_science_translation": row.get("life_science_translation_raw"),
            "bip_impact": row.get("bip_impact_raw"),
            "external_field_year_percentile": row.get(
                "external_field_year_percentile_raw"
            ),
            "patent_citations": row.get("patent_citations_raw"),
            "policy_citations": row.get("policy_citations_raw"),
            "artifact_adoption": row.get("artifact_adoption_raw"),
            "media_mentions": row.get("media_mentions_raw"),
            "award_count": row.get("award_count_raw"),
        },
    }


def _markdown_number(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.3f}"


def _complete_pairwise_prompt(
    prompt: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    temperature: float,
    max_tokens: int | None,
    structured_regex: str | None = None,
    structured_choice: list[str] | None = None,
    request_logprobs: bool = False,
    client: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if structured_regex is not None:
        payload["structured_outputs"] = {"regex": structured_regex}
    if structured_choice is not None:
        payload["structured_outputs"] = {"choice": structured_choice}
    if request_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = min(len(structured_choice or []), 5)
    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    response_payload = response.json()
    choice = response_payload["choices"][0]
    return {
        "content": str(choice["message"]["content"]).strip(),
        "logprobs": choice.get("logprobs"),
    }


def _normalize_rating_to_100(ratings: dict[str, float]) -> dict[str, float]:
    if not ratings:
        return {}
    minimum = min(ratings.values())
    maximum = max(ratings.values())
    if math.isclose(maximum, minimum):
        return {paper_id: 50.0 for paper_id in ratings}
    return {
        paper_id: (rating - minimum) / (maximum - minimum) * 100.0
        for paper_id, rating in ratings.items()
    }


def _confidence(comparisons: int, max_comparisons: int) -> float:
    if max_comparisons <= 0:
        return 0.0
    return comparisons / max_comparisons


def _position_consistency_counts(judgments: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for judgment in judgments:
        value = judgment.get("position_consistency")
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _completion_text(prediction: str | dict[str, Any] | None) -> str | None:
    if prediction is None:
        return None
    if isinstance(prediction, dict):
        return str(prediction.get("content", "")).strip()
    return str(prediction).strip()


def _pairwise_confidence_info(
    prediction: str | dict[str, Any] | None,
    winner_side: str | None,
) -> dict[str, Any]:
    if winner_side is None:
        return {
            "confidence": 0.0,
            "confidence_method": "unparseable",
            "choice_logprob_margin": None,
        }
    margin = _choice_logprob_margin(prediction)
    if margin is None:
        return {
            "confidence": 0.5,
            "confidence_method": "unavailable",
            "choice_logprob_margin": None,
        }
    return {
        "confidence": round(_logistic(margin), 6),
        "confidence_method": "choice_logprob_margin",
        "choice_logprob_margin": round(margin, 6),
    }


def _choice_logprob_margin(prediction: str | dict[str, Any] | None) -> float | None:
    if not isinstance(prediction, dict):
        return None
    best_a = _best_choice_logprob(prediction.get("logprobs"), "A")
    best_b = _best_choice_logprob(prediction.get("logprobs"), "B")
    if best_a is None or best_b is None:
        return None
    return abs(best_a - best_b)


def _max_optional(*values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return max(numbers)


def _best_choice_logprob(logprobs: Any, label: str) -> float | None:
    candidates: list[float] = []
    for item in _logprob_content_items(logprobs):
        token = str(item.get("token", "")).strip()
        if token == label and item.get("logprob") is not None:
            candidates.append(float(item["logprob"]))
        for top_item in item.get("top_logprobs") or []:
            top_token = str(top_item.get("token", "")).strip()
            if top_token == label and top_item.get("logprob") is not None:
                candidates.append(float(top_item["logprob"]))
    if not candidates:
        return None
    return max(candidates)


def _logprob_content_items(logprobs: Any) -> list[dict[str, Any]]:
    if not isinstance(logprobs, dict):
        return []
    content = logprobs.get("content")
    if not isinstance(content, list):
        return []
    return [item for item in content if isinstance(item, dict)]


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _safe_divide_float(numerator: float, denominator: int, *, default: float) -> float:
    return numerator / denominator if denominator else default


def _attach_direct_scores(
    rankings: list[dict[str, Any]],
    direct_scores: list[dict[str, Any]],
) -> None:
    by_paper_id = {row["paper_id"]: row for row in direct_scores}
    for row in rankings:
        score_row = by_paper_id.get(row["paper_id"])
        row["direct_score"] = score_row.get("direct_score") if score_row else None
        row["direct_score_parseable"] = bool(score_row and score_row.get("parseable"))


def _attach_ranking_quality(
    rankings: list[dict[str, Any]],
    ranking_quality: dict[str, Any],
) -> None:
    for row in rankings:
        row["ranking_quality_status"] = ranking_quality["status"]
        row["ranking_quality_reason"] = ranking_quality["reason"]


def _attach_pairwise_stability(
    rankings: list[dict[str, Any]],
    stability: dict[str, Any],
) -> None:
    summary = stability.get("summary", {})
    by_paper_id = {row.get("paper_id"): row for row in stability.get("papers", [])}
    for row in rankings:
        stability_row = by_paper_id.get(row.get("paper_id"))
        row["pairwise_stability_status"] = summary.get("status")
        row["pairwise_stability_band"] = (
            stability_row.get("stability_band") if stability_row else "not_tracked"
        )
        row["pairwise_stability_score_ci_low"] = (
            stability_row.get("score_ci_low") if stability_row else None
        )
        row["pairwise_stability_score_ci_high"] = (
            stability_row.get("score_ci_high") if stability_row else None
        )
        row["pairwise_stability_score_ci_width"] = (
            stability_row.get("score_ci_width") if stability_row else None
        )
        row["pairwise_stability_best_rank"] = (
            stability_row.get("best_rank") if stability_row else None
        )
        row["pairwise_stability_worst_rank"] = (
            stability_row.get("worst_rank") if stability_row else None
        )
        row["pairwise_stability_rank_span"] = (
            stability_row.get("rank_span") if stability_row else None
        )
        row["pairwise_stability_top_n_membership_rate"] = (
            stability_row.get("top_n_membership_rate") if stability_row else None
        )


def _attach_final_score_stability(
    rankings: list[dict[str, Any]],
    stability: dict[str, Any],
) -> None:
    summary = stability.get("summary", {})
    by_paper_id = {row.get("paper_id"): row for row in stability.get("papers", [])}
    for row in rankings:
        stability_row = by_paper_id.get(row.get("paper_id"))
        row["final_score_stability_status"] = summary.get("status")
        row["final_score_stability_band"] = (
            stability_row.get("stability_band") if stability_row else "not_tracked"
        )
        row["final_score_stability_score_ci_low"] = (
            stability_row.get("score_ci_low") if stability_row else None
        )
        row["final_score_stability_score_ci_high"] = (
            stability_row.get("score_ci_high") if stability_row else None
        )
        row["final_score_stability_score_ci_width"] = (
            stability_row.get("score_ci_width") if stability_row else None
        )
        row["final_score_stability_best_rank"] = (
            stability_row.get("best_rank") if stability_row else None
        )
        row["final_score_stability_worst_rank"] = (
            stability_row.get("worst_rank") if stability_row else None
        )
        row["final_score_stability_rank_span"] = (
            stability_row.get("rank_span") if stability_row else None
        )
        row["final_score_stability_top_n_membership_rate"] = (
            stability_row.get("top_n_membership_rate") if stability_row else None
        )


def _load_empirical_calibration(
    calibration_report_path: str | Path | None,
) -> dict[str, Any] | None:
    if calibration_report_path is None:
        return None
    source = Path(calibration_report_path)
    report = json.loads(source.read_text(encoding="utf-8"))
    bins = []
    for row in report.get("calibration", {}).get("bins", []):
        if row.get("count", 0) <= 0 or row.get("accuracy") is None:
            continue
        bins.append(
            {
                "lower": float(row["lower"]),
                "upper": float(row["upper"]),
                "accuracy": float(row["accuracy"]),
                "count": int(row["count"]),
            }
        )
    return {
        "method": "empirical_bin_accuracy",
        "source_path": str(source),
        "bins": bins,
    }


def _calibrated_confidence(
    confidence: float | None,
    calibration_model: dict[str, Any] | None,
) -> float | None:
    if confidence is None or calibration_model is None:
        return None
    bins = calibration_model.get("bins", [])
    for index, row in enumerate(bins):
        lower = float(row["lower"])
        upper = float(row["upper"])
        is_last = index == len(bins) - 1
        if lower <= confidence < upper or (is_last and lower <= confidence <= upper):
            return round(float(row["accuracy"]), 6)
    return None


def _attach_calibrated_pairwise_confidences(
    judgments: list[dict[str, Any]],
    calibration_model: dict[str, Any],
) -> None:
    for judgment in judgments:
        calibrated_confidence = _calibrated_confidence(
            judgment.get("confidence"),
            calibration_model,
        )
        judgment["calibrated_confidence"] = calibrated_confidence
        judgment["calibrated_confidence_method"] = (
            calibration_model["method"] if calibrated_confidence is not None else None
        )


def _calibration_model_report(
    calibration_model: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if calibration_model is None:
        return None
    return {
        "method": calibration_model["method"],
        "source_path": calibration_model["source_path"],
        "bin_count": len(calibration_model.get("bins", [])),
    }


def _write_impact_validity_if_requested(
    destination: Path,
    *,
    validity_labels_path: str | Path | None,
    label_column: str,
    bootstrap_samples: int,
    permutation_samples: int,
) -> dict[str, Any]:
    if validity_labels_path is None:
        return {
            "status": "not_provided",
            "reason": "No validity labels were supplied to model-score-dataset.",
        }
    label_path = Path(validity_labels_path)
    report = write_impact_validity_evaluation(
        destination / "impact_scores.csv",
        label_path,
        destination / "impact_validity",
        label_column=label_column,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
    )
    summary = report.get("summary", {})
    decision = summary.get("decision", {})
    significance = report.get("statistical_significance", {})
    return {
        "status": decision.get("status", "unknown"),
        "reason": decision.get("reason", ""),
        "label_path": str(label_path),
        "label_column": label_column,
        "matched_paper_count": summary.get("matched_paper_count", 0),
        "score_label_match_rate": summary.get("score_label_match_rate", 0.0),
        "spearman_correlation": summary.get("spearman_correlation"),
        "kendall_tau_b": summary.get("kendall_tau_b"),
        "pairwise_accuracy": summary.get("pairwise_accuracy"),
        "ndcg_at_top_k": summary.get("ndcg_at_top_k"),
        "significant_vs_random": significance.get("significant_vs_random"),
        "p_value": significance.get("p_value"),
        "report_path": str(destination / "impact_validity" / "impact_validity_evaluation.json"),
    }


def _attach_validity_confidence(
    rankings: list[dict[str, Any]],
    validity: dict[str, Any],
) -> dict[str, Any]:
    status = str(validity.get("status", "unknown"))
    if status == "not_provided":
        claim = {
            "status": "unvalidated",
            "can_claim_validated_impact_accuracy": False,
            "confidence_ceiling": 0.25,
            "reason": "No future-window or external validity labels were supplied.",
        }
    elif status == "externally_supported" and bool(validity.get("significant_vs_random")):
        claim = {
            "status": "externally_validated",
            "can_claim_validated_impact_accuracy": True,
            "confidence_ceiling": 0.90,
            "reason": "Scores are statistically supported against supplied external/future labels.",
        }
    elif status == "usable_with_disclosure":
        claim = {
            "status": "partially_validated",
            "can_claim_validated_impact_accuracy": False,
            "confidence_ceiling": 0.60,
            "reason": "Validity labels show some signal, but not enough for a validated accuracy claim.",
        }
    elif status == "insufficient_validation_data":
        claim = {
            "status": "insufficient_validation_data",
            "can_claim_validated_impact_accuracy": False,
            "confidence_ceiling": 0.40,
            "reason": "Validity labels were supplied but matched sample size is too small.",
        }
    else:
        claim = {
            "status": "not_validated",
            "can_claim_validated_impact_accuracy": False,
            "confidence_ceiling": 0.35,
            "reason": "Supplied validity labels do not support a validated impact claim.",
        }

    for row in rankings:
        row["validity_status"] = claim["status"]
        row["validity_confidence"] = round(
            min(
                float(claim["confidence_ceiling"]),
                _row_evidence_confidence_prior(row),
            ),
            6,
        )
    return claim


def _row_evidence_confidence_prior(row: dict[str, Any]) -> float:
    tier = str(row.get("credible_evidence_tier", "model_only"))
    tier_prior = {
        "very_high": 0.95,
        "high": 0.80,
        "moderate": 0.60,
        "low": 0.40,
        "model_only": 0.25,
    }
    return tier_prior.get(tier, 0.25)


def _load_fusion_weights(
    path: str | Path | None,
) -> tuple[
    dict[str, float] | None,
    dict[str, float] | None,
    dict[str, float] | None,
    dict[str, Any],
]:
    if path is None:
        return None, None, None, {"source": "default", "source_path": None}

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Fusion weights file must contain a JSON object")

    component_weights_payload = payload.get("best_component_weights")
    if isinstance(component_weights_payload, dict):
        component_weights = _parse_weight_map(
            component_weights_payload,
            expected_signals=("model_pairwise", *EVIDENCE_FIRST_WEIGHTS),
            label="Final component weight",
        )
        normalized_components = _normalize_positive_weights(
            component_weights,
            label="Final component weights",
        )
        model_weight = normalized_components["model_pairwise"]
        evidence_weight = sum(
            normalized_components[signal] for signal in EVIDENCE_FIRST_WEIGHTS
        )
        if evidence_weight > 0.0:
            evidence_weights = {
                signal: normalized_components[signal] / evidence_weight
                for signal in EVIDENCE_FIRST_WEIGHTS
            }
        else:
            evidence_weights = {signal: 0.0 for signal in EVIDENCE_FIRST_WEIGHTS}
        impact_index_v3_weights = {
            "evidence_supported": evidence_weight,
            "model_supported": model_weight,
        }
        return None, evidence_weights, impact_index_v3_weights, {
            "source": "fusion_calibration.best_component_weights",
            "source_path": str(source),
            "weight_profile": "impact_index_v3_final_components",
        }

    weights_payload = payload.get("best_weights") if isinstance(payload, dict) else None
    if not isinstance(weights_payload, dict):
        raise ValueError("Fusion weights file must contain a best_weights object")

    provided_signals = set(weights_payload)
    if provided_signals == set(IMPACT_INDEX_V2_WEIGHTS):
        weights = _normalize_positive_weights(
            _parse_weight_map(
                weights_payload,
                expected_signals=tuple(IMPACT_INDEX_V2_WEIGHTS),
                label="Fusion weight",
            ),
            label="Fusion weights",
        )
        return weights, None, None, {
            "source": "fusion_calibration.best_weights",
            "source_path": str(source),
            "weight_profile": "impact_index_v2_compatibility",
        }

    if provided_signals == set(EVIDENCE_FIRST_WEIGHTS):
        evidence_weights = _normalize_positive_weights(
            _parse_weight_map(
                weights_payload,
                expected_signals=tuple(EVIDENCE_FIRST_WEIGHTS),
                label="Evidence-first weight",
            ),
            label="Evidence-first weights",
        )
        impact_index_v3_payload = payload.get("best_impact_index_v3_weights")
        impact_index_v3_weights = (
            _normalize_positive_weights(
                _parse_weight_map(
                    impact_index_v3_payload,
                    expected_signals=tuple(IMPACT_INDEX_V3_WEIGHTS),
                    label="ImpactIndex v3 layer weight",
                ),
                label="ImpactIndex v3 layer weights",
            )
            if isinstance(impact_index_v3_payload, dict)
            else None
        )
        return None, evidence_weights, impact_index_v3_weights, {
            "source": "fusion_calibration.best_weights",
            "source_path": str(source),
            "weight_profile": "impact_index_v3_evidence_first",
        }

    _raise_invalid_weight_keys(
        provided_signals,
        expected_options=(
            set(IMPACT_INDEX_V2_WEIGHTS),
            set(EVIDENCE_FIRST_WEIGHTS),
            {"model_pairwise", *EVIDENCE_FIRST_WEIGHTS},
        ),
    )


def _parse_weight_map(
    payload: dict[str, Any],
    *,
    expected_signals: tuple[str, ...],
    label: str,
) -> dict[str, float]:
    expected = set(expected_signals)
    provided = set(payload)
    if provided != expected:
        _raise_invalid_weight_keys(provided, expected_options=(expected,))
    weights: dict[str, float] = {}
    for signal in expected_signals:
        value = payload[signal]
        if isinstance(value, bool):
            raise ValueError(f"{label} for {signal} must be numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} for {signal} must be numeric") from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{label} for {signal} must be finite and non-negative")
        weights[signal] = numeric
    return weights


def _normalize_positive_weights(
    weights: dict[str, float],
    *,
    label: str,
) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError(f"{label} must have a positive total")
    return {signal: value / total for signal, value in weights.items()}


def _raise_invalid_weight_keys(
    provided_signals: set[str],
    *,
    expected_options: tuple[set[str], ...],
) -> None:
    expected_union = set().union(*expected_options)
    missing = sorted(min(expected_options, key=lambda option: len(option - provided_signals)) - provided_signals)
    extra = sorted(provided_signals - expected_union)
    problems = []
    if missing:
        problems.append(f"missing signals: {', '.join(missing)}")
    if extra:
        problems.append(f"unknown signals: {', '.join(extra)}")
    if not problems:
        expected_text = " or ".join(
            ", ".join(sorted(option)) for option in expected_options
        )
        problems.append(f"expected one of: {expected_text}")
    raise ValueError("Invalid fusion weights: " + "; ".join(problems))


def render_pairwise_stability_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Pairwise Stability",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| status | {summary.get('status', '')} |",
        f"| method | {summary.get('method', '')} |",
        f"| samples | {summary.get('samples', 0)} |",
        f"| top_n | {summary.get('top_n', '')} |",
        f"| decisive_judgment_count | {summary.get('decisive_judgment_count', 0)} |",
        f"| mean_top_n_retention_rate | {summary.get('mean_top_n_retention_rate', '')} |",
        f"| stable_paper_count | {summary.get('stable_paper_count', 0)} |",
        f"| unstable_paper_count | {summary.get('unstable_paper_count', 0)} |",
        f"| reason | {summary.get('reason', '')} |",
        "",
        "## Tracked Papers",
        "",
        "| Paper ID | Base Rank | Base Score | Score CI Low | Score CI High | Rank Span | Top-N Membership | Stability | Title |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report.get("papers", []):
        lines.append(
            f"| {row.get('paper_id', '')} | "
            f"{row.get('base_rank', '')} | "
            f"{_markdown_number(row.get('base_model_pairwise_score'))} | "
            f"{_markdown_number(row.get('score_ci_low'))} | "
            f"{_markdown_number(row.get('score_ci_high'))} | "
            f"{row.get('rank_span', '')} | "
            f"{_markdown_number(row.get('top_n_membership_rate'))} | "
            f"{row.get('stability_band', '')} | "
            f"{row.get('title', '')} |"
        )
    limitations = report.get("limitations", [])
    if limitations:
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {item}" for item in limitations)
    lines.append("")
    return "\n".join(lines)


def _write_pairwise_stability_csv(path: Path, report: dict[str, Any]) -> None:
    fieldnames = [
        "paper_id",
        "title",
        "base_rank",
        "base_model_pairwise_score",
        "score_mean",
        "score_ci_low",
        "score_ci_high",
        "score_ci_width",
        "best_rank",
        "worst_rank",
        "rank_span",
        "top_n_membership_rate",
        "stability_band",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("papers", []):
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def render_final_score_stability_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Final Score Stability",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| status | {summary.get('status', '')} |",
        f"| method | {summary.get('method', '')} |",
        f"| samples | {summary.get('samples', 0)} |",
        f"| top_n | {summary.get('top_n', '')} |",
        f"| decisive_judgment_count | {summary.get('decisive_judgment_count', 0)} |",
        f"| mean_top_n_retention_rate | {summary.get('mean_top_n_retention_rate', '')} |",
        f"| stable_paper_count | {summary.get('stable_paper_count', 0)} |",
        f"| unstable_paper_count | {summary.get('unstable_paper_count', 0)} |",
        f"| reason | {summary.get('reason', '')} |",
        "",
        "## Tracked Papers",
        "",
        "| Paper ID | Base Rank | base_final_impact_score | Score CI Low | Score CI High | Rank Span | Top-N Membership | Stability | Title |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report.get("papers", []):
        lines.append(
            f"| {row.get('paper_id', '')} | "
            f"{row.get('base_rank', '')} | "
            f"{_markdown_number(row.get('base_final_impact_score'))} | "
            f"{_markdown_number(row.get('score_ci_low'))} | "
            f"{_markdown_number(row.get('score_ci_high'))} | "
            f"{row.get('rank_span', '')} | "
            f"{_markdown_number(row.get('top_n_membership_rate'))} | "
            f"{row.get('stability_band', '')} | "
            f"{row.get('title', '')} |"
        )
    limitations = report.get("limitations", [])
    if limitations:
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {item}" for item in limitations)
    lines.append("")
    return "\n".join(lines)


def _write_final_score_stability_csv(path: Path, report: dict[str, Any]) -> None:
    fieldnames = [
        "paper_id",
        "title",
        "base_rank",
        "base_final_impact_score",
        "score_mean",
        "score_ci_low",
        "score_ci_high",
        "score_ci_width",
        "best_rank",
        "worst_rank",
        "rank_span",
        "top_n_membership_rate",
        "stability_band",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("papers", []):
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_impact_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "paper_id",
        "title",
        "impact_index_v3",
        "impact_index_v2",
        "final_impact_score",
        "validity_status",
        "validity_confidence",
        "credibility_adjusted_impact_score",
        "credibility_adjustment_factor",
        "credibility_adjustment_reason",
        "source_kind",
        "source_quality_flag",
        "source_quality_adjustment_factor",
        "source_quality_reason",
        "pre_source_quality_final_impact_score",
        "evidence_supported_score",
        "model_supported_score",
        "impact_index",
        "model_pairwise_rank",
        "model_pairwise_score",
        "field_year_normalized_citation_score",
        "relative_citation_ratio_score",
        "citation_graph_score",
        "citation_context_quality_score",
        "effective_citation_score",
        "influential_citation_score",
        "temporal_momentum_score",
        "external_adoption_score",
        "external_field_year_percentile_score",
        "patent_impact_score",
        "policy_impact_score",
        "artifact_adoption_score",
        "media_impact_score",
        "award_impact_score",
        "bip_impact_score",
        "life_science_translation_score",
        "model_pairwise_raw",
        "field_year_normalized_citation_raw",
        "relative_citation_ratio_raw",
        "relative_citation_rate_raw",
        "relative_expected_citation_rate_raw",
        "citation_count_raw",
        "citation_outlier_flag",
        "citation_outlier_ratio",
        "citation_outlier_basis",
        "citation_graph_raw",
        "citation_context_quality_raw",
        "effective_citation_raw",
        "effective_citation_edge_count",
        "effective_citation_context_count",
        "effective_citation_count",
        "effective_citation_guideline_count",
        "effective_citation_clinical_count",
        "effective_citation_method_count",
        "effective_citation_dataset_or_tool_count",
        "effective_citation_supporting_count",
        "effective_citation_negative_count",
        "effective_citation_background_count",
        "effective_citation_unknown_count",
        "effective_citation_coverage",
        "effective_citation_evidence_level",
        "influential_citation_count_raw",
        "temporal_momentum_raw",
        "external_adoption_raw",
        "external_field_year_percentile_raw",
        "patent_citations_raw",
        "policy_citations_raw",
        "artifact_adoption_raw",
        "media_mentions_raw",
        "award_count_raw",
        "bip_impact_raw",
        "life_science_translation_raw",
        "fusion_signal_count",
        "fusion_available_weight",
        "evidence_first_signal_count",
        "positive_evidence_signal_count",
        "zero_evidence_signal_count",
        "fusion_weight_model_pairwise",
        "fusion_weight_field_year_normalized_citation",
        "fusion_weight_citation_graph",
        "fusion_weight_citation_context_quality",
        "fusion_weight_temporal_momentum",
        "dominant_signal",
        "max_signal_contribution_share",
        "score_ci_low",
        "score_ci_high",
        "score_ci_width",
        "score_interval_method",
        "uncertainty_band",
        "evidence_quality_flag",
        "evidence_quality_reason",
        "credible_evidence_tier",
        "dominant_score_family",
        "missing_high_value_evidence",
        "reportability_status",
        "reportability_reason",
        "reportability_gap_count",
        "reportability_gap_actions",
        "reportability_basis",
        "legacy_reportability_status",
        "legacy_reportability_reason",
        "direct_score",
        "direct_score_parseable",
        "elo_rating",
        "rank_centrality_probability",
        "pairwise_aggregation_method",
        "wins",
        "losses",
        "comparisons",
        "coverage_confidence",
        "mean_pairwise_confidence",
        "mean_calibrated_pairwise_confidence",
        "confidence",
        "confidence_source",
        "ranking_quality_status",
        "ranking_quality_reason",
        "pairwise_stability_status",
        "pairwise_stability_band",
        "pairwise_stability_score_ci_low",
        "pairwise_stability_score_ci_high",
        "pairwise_stability_score_ci_width",
        "pairwise_stability_best_rank",
        "pairwise_stability_worst_rank",
        "pairwise_stability_rank_span",
        "pairwise_stability_top_n_membership_rate",
        "final_score_stability_status",
        "final_score_stability_band",
        "final_score_stability_score_ci_low",
        "final_score_stability_score_ci_high",
        "final_score_stability_score_ci_width",
        "final_score_stability_best_rank",
        "final_score_stability_worst_rank",
        "final_score_stability_rank_span",
        "final_score_stability_top_n_membership_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_direct_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "paper_id",
        "title",
        "direct_score",
        "parseable",
        "prediction",
        "model",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
