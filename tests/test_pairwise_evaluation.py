import csv
import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.pairwise_evaluation import (
    evaluate_pairwise_labels,
    write_pairwise_evaluation,
)


def write_rankings(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "id", "score", "final_score"])
        writer.writeheader()
        writer.writerow({"rank": 1, "id": "p-a", "score": "0.0", "final_score": "90.0"})
        writer.writerow({"rank": 2, "id": "p-b", "score": "0.0", "final_score": "70.0"})
        writer.writerow({"rank": 3, "id": "p-c", "score": "0.0", "final_score": "80.0"})
        writer.writerow({"rank": 4, "id": "p-d", "score": "0.0", "final_score": "80.0"})


def write_labels(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "pair_id": "pair-1",
            "winner_id": "p-a",
            "loser_id": "p-b",
            "impact_dimension": "citation",
            "field": "AI",
        },
        {
            "pair_id": "pair-2",
            "winner_id": "p-b",
            "loser_id": "p-c",
            "impact_dimension": "societal",
            "field": "AI",
        },
        {
            "pair_id": "pair-3",
            "winner_id": "p-c",
            "loser_id": "p-d",
            "impact_dimension": "citation",
            "field": "Biology",
        },
        {
            "pair_id": "pair-4",
            "winner_id": "p-a",
            "loser_id": "missing",
            "impact_dimension": "societal",
            "field": "Biology",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_evaluate_pairwise_labels_counts_correct_wrong_ties_and_coverage(tmp_path):
    rankings = tmp_path / "rankings" / "papers.csv"
    labels = tmp_path / "pairwise_labels.jsonl"
    write_rankings(rankings)
    write_labels(labels)

    report = evaluate_pairwise_labels(rankings, labels)

    assert report["summary"]["total_pairs"] == 4
    assert report["summary"]["covered_pairs"] == 3
    assert report["summary"]["uncovered_pairs"] == 1
    assert report["summary"]["correct_pairs"] == 1
    assert report["summary"]["wrong_pairs"] == 1
    assert report["summary"]["tie_pairs"] == 1
    assert report["summary"]["coverage_rate"] == 0.75
    assert report["summary"]["pairwise_accuracy"] == 1 / 3
    assert report["summary"]["non_tie_pairwise_accuracy"] == 0.5
    assert report["by_dimension"]["citation"]["pairwise_accuracy"] == 0.5
    assert report["by_dimension"]["societal"]["coverage_rate"] == 0.5
    assert report["by_field"]["AI"]["pairwise_accuracy"] == 0.5
    assert report["by_field"]["Biology"]["tie_rate"] == 1.0
    assert report["examples"][0]["pair_id"] == "pair-1"


def test_write_pairwise_evaluation_writes_json_and_markdown(tmp_path):
    rankings = tmp_path / "rankings" / "papers.csv"
    labels = tmp_path / "pairwise_labels.jsonl"
    output = tmp_path / "eval"
    write_rankings(rankings)
    write_labels(labels)

    report = write_pairwise_evaluation(rankings, labels, output)

    assert report["summary"]["covered_pairs"] == 3
    assert (output / "pairwise_evaluation.json").exists()
    markdown = (output / "pairwise_evaluation.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Pairwise Impact Evaluation\n")
    assert "| pairwise_accuracy | 0.333 |" in markdown
    assert "| citation | 2 | 2 | 0.500 |" in markdown


def test_cli_evaluate_pairwise_impact_writes_report(tmp_path):
    rankings = tmp_path / "rankings" / "papers.csv"
    labels = tmp_path / "pairwise_labels.jsonl"
    output = tmp_path / "eval"
    write_rankings(rankings)
    write_labels(labels)

    result = CliRunner().invoke(
        app,
        [
            "evaluate-pairwise-impact",
            "--rankings",
            str(rankings),
            "--labels",
            str(labels),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "pairwise_accuracy=0.333" in result.output
    assert (output / "pairwise_evaluation.json").exists()
