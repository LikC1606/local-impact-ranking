from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from local_impact_ranking.csv_safety import sanitize_csv_row
from local_impact_ranking.model_impact_index import aggregate_pairwise_impact_index
from local_impact_ranking.models import Paper
from local_impact_ranking.pairwise_evaluation import write_pairwise_evaluation
from local_impact_ranking.sciimpact_adapter import _parse_pair
from local_impact_ranking.sciimpact_model_eval import _side_from_prediction


def write_sciimpact_index_evaluation(
    sft_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions = _load_predictions(predictions_path)
    papers_by_id: dict[str, Paper] = {}
    labels = []
    judgments = []
    skipped = []

    with Path(sft_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = _parse_pair(json.loads(stripped), line_number)
            except ValueError as exc:
                skipped.append({"source_line": line_number, "error": str(exc)})
                continue
            paper_a = parsed["winner"] if _winner_side(parsed) == "a" else parsed["loser"]
            paper_b = parsed["winner"] if _winner_side(parsed) == "b" else parsed["loser"]
            paper_a_model = _paper_from_parsed(paper_a)
            paper_b_model = _paper_from_parsed(paper_b)
            papers_by_id[paper_a_model.paper_id] = paper_a_model
            papers_by_id[paper_b_model.paper_id] = paper_b_model
            labels.append(
                {
                    "pair_id": f"sciimpact-{line_number}",
                    "winner_id": str(parsed["winner"]["paper_id"]),
                    "loser_id": str(parsed["loser"]["paper_id"]),
                    "impact_dimension": parsed.get("impact_dimension", "unknown"),
                    "field": "unknown",
                    "source_line": line_number,
                }
            )
            prediction = predictions.get(line_number)
            predicted_side = _side_from_prediction(prediction)
            if predicted_side == "a":
                winner_id = paper_a_model.paper_id
                loser_id = paper_b_model.paper_id
            elif predicted_side == "b":
                winner_id = paper_b_model.paper_id
                loser_id = paper_a_model.paper_id
            else:
                winner_id = None
                loser_id = None
            judgments.append(
                {
                    "pair_index": line_number,
                    "paper_a_id": paper_a_model.paper_id,
                    "paper_b_id": paper_b_model.paper_id,
                    "winner_side": predicted_side.upper() if predicted_side else None,
                    "winner_id": winner_id,
                    "loser_id": loser_id,
                    "parseable": predicted_side is not None,
                    "prediction": prediction,
                    "model": "sciimpact-predictions",
                }
            )

    papers = list(papers_by_id.values())
    rankings = aggregate_pairwise_impact_index(papers, judgments)
    _write_jsonl(destination / "pairwise_judgments.jsonl", judgments)
    _write_jsonl(destination / "pairwise_labels.jsonl", labels)
    _write_jsonl(destination / "impact_scores.jsonl", rankings)
    _write_impact_scores_csv(destination / "impact_scores.csv", rankings)
    pairwise_report = write_pairwise_evaluation(
        destination / "impact_scores.csv",
        destination / "pairwise_labels.jsonl",
        destination / "pairwise_evaluation",
        score_column="impact_index",
    )
    report = {
        "summary": {
            "paper_count": len(papers),
            "label_count": len(labels),
            "judgment_count": len(judgments),
            "parseable_judgment_count": sum(judgment["parseable"] for judgment in judgments),
            "skipped_source_count": len(skipped),
        },
        "pairwise_evaluation": pairwise_report,
        "skipped_sources": skipped[:20],
    }
    (destination / "sciimpact_index_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "sciimpact_index_evaluation.md").write_text(
        render_sciimpact_index_evaluation_markdown(report),
        encoding="utf-8",
    )
    return report


def render_sciimpact_index_evaluation_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    pairwise_summary = report["pairwise_evaluation"]["summary"]
    return "\n".join(
        [
            "# SciImpact Index Evaluation",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| paper_count | {summary['paper_count']} |",
            f"| label_count | {summary['label_count']} |",
            f"| judgment_count | {summary['judgment_count']} |",
            f"| parseable_judgment_count | {summary['parseable_judgment_count']} |",
            f"| pairwise_accuracy | {pairwise_summary['pairwise_accuracy']:.3f} |",
            f"| coverage_rate | {pairwise_summary['coverage_rate']:.3f} |",
            "",
        ]
    )


def _load_predictions(path: str | Path) -> dict[int, str]:
    predictions: dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            source_line = int(record.get("source_line", line_number))
            prediction = record.get("prediction") or record.get("answer") or record.get("text")
            if prediction is not None:
                predictions[source_line] = str(prediction)
    return predictions


def _winner_side(parsed: dict[str, Any]) -> str:
    return str(parsed["winner"].get("metadata", {}).get("sciimpact_source_side"))


def _paper_from_parsed(record: dict[str, Any]) -> Paper:
    return Paper.model_validate(
        {
            "paper_id": record["paper_id"],
            "title": record.get("title") or "Untitled SciImpact paper",
            "abstract": record.get("abstract"),
            "year": record.get("year"),
            "references": [],
            "metadata": record.get("metadata", {}),
        }
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_impact_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "paper_id",
        "title",
        "impact_index",
        "elo_rating",
        "wins",
        "losses",
        "comparisons",
        "confidence",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                sanitize_csv_row({field: row[field] for field in fieldnames})
            )
