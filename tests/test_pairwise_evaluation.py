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
    assert report["credibility_scorecard"]["status"] == "insufficient_validation_data"
    assert report["statistical_significance"]["status"] == "insufficient_validation_data"


def test_write_pairwise_evaluation_writes_json_and_markdown(tmp_path):
    rankings = tmp_path / "rankings" / "papers.csv"
    labels = tmp_path / "pairwise_labels.jsonl"
    output = tmp_path / "eval"
    write_rankings(rankings)
    write_labels(labels)

    report = write_pairwise_evaluation(rankings, labels, output)

    assert report["summary"]["covered_pairs"] == 3
    assert (output / "pairwise_evaluation.json").exists()
    assert (output / "pairwise_credibility_scorecard.json").exists()
    assert (output / "pairwise_credibility_scorecard.md").exists()
    markdown = (output / "pairwise_evaluation.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Pairwise Impact Evaluation\n")
    assert "| pairwise_accuracy | 0.333 |" in markdown
    assert "| citation | 2 | 2 | 0.500 |" in markdown
    scorecard_md = (output / "pairwise_credibility_scorecard.md").read_text(
        encoding="utf-8"
    )
    assert scorecard_md.startswith("# Pairwise Credibility Scorecard\n")


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
    assert "credibility=" in result.output
    assert (output / "pairwise_evaluation.json").exists()


def test_pairwise_credibility_scorecard_marks_strong_benchmark_supported(tmp_path):
    rankings = tmp_path / "rankings.csv"
    labels = tmp_path / "pairwise_labels.jsonl"
    with rankings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "final_score"])
        writer.writeheader()
        for index in range(80):
            writer.writerow({"paper_id": f"p{index:02d}", "final_score": 100 - index})
    rows = []
    for index in range(40):
        rows.append(
            {
                "pair_id": f"pair-{index}",
                "winner_id": f"p{index:02d}",
                "loser_id": f"p{index + 40:02d}",
                "impact_dimension": "citation" if index < 20 else "patent",
                "field": "AI" if index < 20 else "Bio",
            }
        )
    labels.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = evaluate_pairwise_labels(rankings, labels, score_column="final_score")

    assert report["summary"]["pairwise_accuracy"] == 1.0
    assert report["statistical_significance"]["significant_vs_random"] is True
    scorecard = report["credibility_scorecard"]
    assert scorecard["status"] == "high_credibility"
    assert scorecard["grade"] == "A"
    assert scorecard["checks"]["statistical_support"]["status"] == "pass"
    assert scorecard["checks"]["slice_stability"]["status"] == "pass"


def test_pairwise_credibility_scorecard_rejects_random_like_pairs(tmp_path):
    rankings = tmp_path / "rankings.csv"
    labels = tmp_path / "pairwise_labels.jsonl"
    with rankings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "final_score"])
        writer.writeheader()
        for index in range(80):
            writer.writerow({"paper_id": f"p{index:02d}", "final_score": 100 - index})
    rows = []
    for index in range(40):
        if index % 2 == 0:
            winner_id = f"p{index:02d}"
            loser_id = f"p{index + 40:02d}"
        else:
            winner_id = f"p{index + 40:02d}"
            loser_id = f"p{index:02d}"
        rows.append(
            {
                "pair_id": f"pair-{index}",
                "winner_id": winner_id,
                "loser_id": loser_id,
                "impact_dimension": "citation",
                "field": "AI",
            }
        )
    labels.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = evaluate_pairwise_labels(rankings, labels, score_column="final_score")

    assert report["summary"]["non_tie_pairwise_accuracy"] == 0.5
    assert report["statistical_significance"]["significant_vs_random"] is False
    scorecard = report["credibility_scorecard"]
    assert scorecard["status"] == "not_credible"
    assert scorecard["checks"]["statistical_support"]["status"] == "fail"
