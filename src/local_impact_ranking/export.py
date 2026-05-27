from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from local_impact_ranking.judge import JudgeResult
from local_impact_ranking.models import Paper
from local_impact_ranking.scoring import InfluenceScores


def export_rankings_csv(scores: InfluenceScores, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_score_table(
        destination / "papers.csv",
        scores.paper_scores,
        score_details=scores.paper_score_details,
    )
    _write_score_table(destination / "authors.csv", scores.author_scores)
    _write_score_table(destination / "institutions.csv", scores.institution_scores)
    _write_score_table(destination / "fields.csv", scores.field_scores)
    _write_score_table(destination / "years.csv", scores.year_scores)


def export_scoring_edges_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_paper_id",
        "reference_id",
        "rank",
        "rank_weight",
        "context_count",
        "context_weight",
        "credit",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "target_paper_id": row["target_paper_id"],
                    "reference_id": row["reference_id"],
                    "rank": row["rank"],
                    "rank_weight": f"{row['rank_weight']:.6f}",
                    "context_count": row["context_count"],
                    "context_weight": f"{row['context_weight']:.6f}",
                    "credit": f"{row['credit']:.6f}",
                }
            )


def _write_score_table(
    path: Path,
    scores: dict[str, float],
    *,
    score_details: dict[str, dict[str, float]] | None = None,
) -> None:
    rows = _score_rows(scores, score_details)
    max_score = max((score for _identifier, score in rows), default=0.0)
    total = len(rows)
    detail_fields = _score_detail_fields(score_details)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "id",
                "score",
                *detail_fields,
                "normalized_score",
                "percentile",
            ],
        )
        writer.writeheader()
        for rank, (identifier, score) in enumerate(rows, start=1):
            normalized_score = (score / max_score * 100.0) if max_score > 0 else 0.0
            percentile = ((total - rank + 1) / total * 100.0) if total else 0.0
            writer.writerow(
                {
                    "rank": rank,
                    "id": identifier,
                    "score": f"{score:.6f}",
                    **_score_detail_row(identifier, score_details),
                    "normalized_score": f"{normalized_score:.6f}",
                    "percentile": f"{percentile:.6f}",
                }
            )


def _score_rows(
    scores: dict[str, float],
    score_details: dict[str, dict[str, float]] | None,
) -> list[tuple[str, float]]:
    identifiers = set(scores)
    if score_details is not None:
        identifiers.update(score_details)
    sort_field = _sort_score_field(score_details)
    return sorted(
        ((identifier, scores.get(identifier, 0.0)) for identifier in identifiers),
        key=lambda item: (
            -(
                score_details.get(item[0], {}).get(sort_field, item[1])
                if score_details is not None
                else item[1]
            ),
            item[0],
        ),
    )


def _score_detail_row(
    identifier: str,
    score_details: dict[str, dict[str, float]] | None,
) -> dict[str, str]:
    if score_details is None:
        return {}
    details = score_details.get(identifier, {})
    row = {}
    for field in _score_detail_fields(score_details):
        value = details.get(field, "")
        if field in _text_score_detail_fields():
            row[field] = str(value)
        else:
            row[field] = f"{float(value or 0.0):.6f}"
    return row


def _score_detail_fields(score_details: dict[str, dict[str, float]] | None) -> list[str]:
    if score_details is None:
        return []
    preferred_order = [
        "core_score",
        "expanded_score",
        "context_score",
        "graph_score",
        "field_year_percentile",
        "max_single_edge_share",
        "final_score",
        "confidence_flag",
        "confidence_reason",
    ]
    present = {field for details in score_details.values() for field in details}
    ordered = [field for field in preferred_order if field in present]
    ordered.extend(sorted(present - set(preferred_order)))
    return ordered


def _text_score_detail_fields() -> set[str]:
    return {"confidence_flag", "confidence_reason"}


def _sort_score_field(score_details: dict[str, dict[str, float]] | None) -> str | None:
    if score_details is None:
        return None
    if any("final_score" in details for details in score_details.values()):
        return "final_score"
    return None


def export_human_audit_sample(
    papers: list[Paper],
    judge_results: list[JudgeResult],
    path: str | Path,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    papers_by_id = {paper.paper_id: paper for paper in papers}
    results_by_id = {result.target_paper_id: result for result in judge_results}
    fieldnames = [
        "target_paper_id",
        "target_title",
        "reference_id",
        "reference_title",
        "reference_year",
        "reference_venue",
        "rank",
        "confidence",
        "valid",
        "rationale",
        "human_importance_label",
        "human_notes",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for paper in papers:
            result = results_by_id.get(paper.paper_id)
            if result is None:
                continue
            references_by_id = {
                reference.reference_id: reference for reference in paper.references
            }
            for selection in sorted(result.selected_references, key=lambda item: item.rank):
                reference = references_by_id.get(selection.reference_id)
                writer.writerow(
                    {
                        "target_paper_id": paper.paper_id,
                        "target_title": paper.title,
                        "reference_id": selection.reference_id,
                        "reference_title": reference.title if reference else "",
                        "reference_year": reference.year if reference and reference.year else "",
                        "reference_venue": reference.venue if reference and reference.venue else "",
                        "rank": selection.rank,
                        "confidence": f"{selection.confidence:.6f}",
                        "valid": selection.valid,
                        "rationale": selection.rationale,
                        "human_importance_label": "",
                        "human_notes": "",
                    }
                )
