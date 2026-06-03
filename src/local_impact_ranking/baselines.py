from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_impact_ranking.csv_safety import sanitize_csv_row
from local_impact_ranking.models import Paper


@dataclass(frozen=True)
class BibliometricBaselines:
    citation_count: dict[str, float]
    pagerank: dict[str, float]
    field_year_normalized_citation: dict[str, float]


def compute_bibliometric_baselines(
    papers: list[Paper],
    *,
    pagerank_iterations: int = 30,
    damping: float = 0.85,
) -> BibliometricBaselines:
    return BibliometricBaselines(
        citation_count=_citation_count_scores(papers),
        pagerank=_pagerank_scores(papers, iterations=pagerank_iterations, damping=damping),
        field_year_normalized_citation=_field_year_normalized_citation_scores(papers),
    )


def export_baseline_rankings(baselines: BibliometricBaselines, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_score_table(destination / "citation_count.csv", baselines.citation_count)
    _write_score_table(destination / "pagerank.csv", baselines.pagerank)
    _write_score_table(
        destination / "field_year_normalized_citation.csv",
        baselines.field_year_normalized_citation,
    )


def summarize_baseline_comparison(
    llm_scores: dict[str, float],
    baselines: BibliometricBaselines,
    *,
    final_scores: dict[str, float] | None = None,
    top_n: int = 100,
) -> dict[str, Any]:
    llm_top = set(_top_ids(llm_scores, top_n))
    final_scores = final_scores or {}
    final_top = set(_top_ids(final_scores, top_n))
    return {
        "top_n": top_n,
        "baselines": {
            "citation_count": _comparison_summary(
                llm_top,
                baselines.citation_count,
                top_n,
                final_top=final_top,
                final_scores=final_scores,
            ),
            "pagerank": _comparison_summary(
                llm_top,
                baselines.pagerank,
                top_n,
                final_top=final_top,
                final_scores=final_scores,
            ),
            "field_year_normalized_citation": _comparison_summary(
                llm_top,
                baselines.field_year_normalized_citation,
                top_n,
                final_top=final_top,
                final_scores=final_scores,
            ),
        },
        "metric_contract": {
            "task": "Compare LLM-selected influential-reference paper scores against local bibliometric baselines.",
            "dataset": "Current input paper dataset.",
            "split": "Full input set used by this run.",
            "metric_direction": "higher_is_better",
            "baseline_metrics": [
                "citation_count",
                "pagerank",
                "field_year_normalized_citation",
            ],
        },
    }


def _citation_count_scores(papers: list[Paper]) -> dict[str, float]:
    return {paper.paper_id: _citation_count(paper) for paper in papers}


def _field_year_normalized_citation_scores(papers: list[Paper]) -> dict[str, float]:
    citation_counts = _citation_count_scores(papers)
    buckets: defaultdict[tuple[str, int], list[float]] = defaultdict(list)
    for paper in papers:
        if paper.field is None or paper.year is None:
            continue
        buckets[(paper.field, paper.year)].append(citation_counts[paper.paper_id])

    bucket_means = {
        key: sum(values) / len(values)
        for key, values in buckets.items()
        if values and sum(values) > 0
    }
    scores: dict[str, float] = {}
    for paper in papers:
        count = citation_counts[paper.paper_id]
        key = (paper.field, paper.year)
        mean = bucket_means.get(key)
        scores[paper.paper_id] = count / mean if mean else 0.0
    return scores


def _pagerank_scores(
    papers: list[Paper],
    *,
    iterations: int,
    damping: float,
) -> dict[str, float]:
    paper_ids = {paper.paper_id for paper in papers}
    if not paper_ids:
        return {}
    outgoing: dict[str, set[str]] = {}
    for paper in papers:
        outgoing[paper.paper_id] = {
            reference.reference_id
            for reference in paper.references
            if reference.reference_id in paper_ids
        }

    node_count = len(paper_ids)
    scores = {paper_id: 1.0 / node_count for paper_id in paper_ids}
    teleport = (1.0 - damping) / node_count

    for _ in range(iterations):
        next_scores = {paper_id: teleport for paper_id in paper_ids}
        dangling_mass = sum(
            scores[paper_id] for paper_id, targets in outgoing.items() if not targets
        )
        dangling_credit = damping * dangling_mass / node_count
        for paper_id in paper_ids:
            next_scores[paper_id] += dangling_credit
        for source_id, targets in outgoing.items():
            if not targets:
                continue
            credit = damping * scores[source_id] / len(targets)
            for target_id in targets:
                next_scores[target_id] += credit
        scores = next_scores
    return scores


def _citation_count(paper: Paper) -> float:
    top_level = _extra_value(paper, "citation_count")
    if top_level is not None:
        return float(top_level)
    metadata_value = paper.metadata.get("citation_count")
    if metadata_value is not None:
        return float(metadata_value)
    return 0.0


def _extra_value(paper: Paper, key: str) -> Any | None:
    extra = getattr(paper, "__pydantic_extra__", None) or {}
    return extra.get(key)


def _comparison_summary(
    llm_top: set[str],
    baseline_scores: dict[str, float],
    top_n: int,
    *,
    final_top: set[str] | None = None,
    final_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    baseline_top = set(_top_ids(baseline_scores, top_n))
    final_top = final_top or set()
    final_scores = final_scores or {}
    return {
        "top_n_overlap": len(llm_top & baseline_top),
        "top_n_overlap_rate": len(llm_top & baseline_top) / len(llm_top) if llm_top else 0.0,
        "raw_top_n_overlap": len(llm_top & baseline_top),
        "raw_top_n_overlap_rate": len(llm_top & baseline_top) / len(llm_top)
        if llm_top
        else 0.0,
        "final_top_n_overlap": len(final_top & baseline_top),
        "final_top_n_overlap_rate": len(final_top & baseline_top) / len(final_top)
        if final_top
        else 0.0,
        "final_spearman": _spearman(final_scores, baseline_scores),
        "final_kendall_tau": _kendall_tau(final_scores, baseline_scores),
    }


def _top_ids(scores: dict[str, float], top_n: int) -> list[str]:
    return [
        identifier
        for identifier, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
            :top_n
        ]
    ]


def _spearman(first_scores: dict[str, float], second_scores: dict[str, float]) -> float:
    identifiers = sorted(set(first_scores) & set(second_scores))
    if len(identifiers) < 2:
        return 0.0
    first_ranks = _rank_map(first_scores, identifiers)
    second_ranks = _rank_map(second_scores, identifiers)
    n = len(identifiers)
    squared_diffs = sum(
        (first_ranks[identifier] - second_ranks[identifier]) ** 2
        for identifier in identifiers
    )
    return 1.0 - (6.0 * squared_diffs) / (n * (n * n - 1))


def _kendall_tau(first_scores: dict[str, float], second_scores: dict[str, float]) -> float:
    identifiers = sorted(set(first_scores) & set(second_scores))
    if len(identifiers) < 2:
        return 0.0
    first_ranks = _rank_map(first_scores, identifiers)
    second_ranks = _rank_map(second_scores, identifiers)
    concordant = 0
    discordant = 0
    for left_index, left_id in enumerate(identifiers):
        for right_id in identifiers[left_index + 1 :]:
            first_delta = first_ranks[left_id] - first_ranks[right_id]
            second_delta = second_ranks[left_id] - second_ranks[right_id]
            product = first_delta * second_delta
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def _rank_map(scores: dict[str, float], identifiers: list[str]) -> dict[str, int]:
    ordered = [
        identifier
        for identifier, _score in sorted(
            ((identifier, scores[identifier]) for identifier in identifiers),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    return {identifier: rank for rank, identifier in enumerate(ordered, start=1)}


def _write_score_table(path: Path, scores: dict[str, float]) -> None:
    rows = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "id", "score"])
        writer.writeheader()
        for rank, (identifier, score) in enumerate(rows, start=1):
            writer.writerow(
                sanitize_csv_row(
                    {"rank": rank, "id": identifier, "score": f"{score:.6f}"}
                )
            )
