import csv
import json

from typer.testing import CliRunner

import local_impact_ranking.cli as cli
from local_impact_ranking.cli import app
from local_impact_ranking.sciimpact_index_eval import write_sciimpact_index_evaluation


def sft_row(title_a: str, title_b: str, answer: str, dimension: str = "citation") -> dict:
    return {
        "conversations": [
            {"from": "system", "value": f"Which paper has more {dimension}?"},
            {
                "from": "human",
                "value": (
                    f"Paper A: Title: {title_a} Year: 2024 Abstract: {title_a}. "
                    f"Paper B: Title: {title_b} Year: 2024 Abstract: {title_b}. "
                    f"Which paper has more {dimension}?"
                ),
            },
            {"from": "gpt", "value": f"Paper {answer} has more {dimension}."},
        ]
    }


def test_write_sciimpact_index_evaluation_aggregates_predictions_to_scores(tmp_path):
    sft = tmp_path / "sample.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "index_eval"
    rows = [
        sft_row("Alpha", "Beta", "A"),
        sft_row("Alpha", "Gamma", "A"),
        sft_row("Beta", "Gamma", "A"),
    ]
    sft.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"source_line": 1, "prediction": "Answer: A"}),
                json.dumps({"source_line": 2, "prediction": "Answer: A"}),
                json.dumps({"source_line": 3, "prediction": "Answer: A"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = write_sciimpact_index_evaluation(sft, predictions, output)

    assert report["summary"]["paper_count"] == 3
    assert report["summary"]["parseable_judgment_count"] == 3
    assert report["pairwise_evaluation"]["summary"]["pairwise_accuracy"] == 1.0
    with (output / "impact_scores.csv").open(newline="", encoding="utf-8") as handle:
        score_rows = list(csv.DictReader(handle))
    assert score_rows[0]["paper_id"].startswith("sciimpact:paper:")
    assert float(score_rows[0]["impact_index"]) == 100.0
    assert (output / "pairwise_labels.jsonl").exists()
    assert (output / "pairwise_evaluation" / "pairwise_evaluation.json").exists()


def test_cli_evaluate_sciimpact_index_writes_report(monkeypatch, tmp_path):
    sft = tmp_path / "sample.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "index_eval"
    sft.write_text(json.dumps(sft_row("Alpha", "Beta", "A")) + "\n", encoding="utf-8")
    predictions.write_text(
        json.dumps({"source_line": 1, "prediction": "Answer: A"}) + "\n",
        encoding="utf-8",
    )

    def fake_write_sciimpact_index_evaluation(*args, **kwargs):
        output.mkdir(parents=True)
        return {
            "summary": {"paper_count": 2},
            "pairwise_evaluation": {
                "summary": {"pairwise_accuracy": 1.0, "coverage_rate": 1.0}
            },
        }

    monkeypatch.setattr(
        cli,
        "write_sciimpact_index_evaluation",
        fake_write_sciimpact_index_evaluation,
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate-sciimpact-index",
            "--sft",
            str(sft),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "pairwise_accuracy=1.000" in result.output
