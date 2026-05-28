import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.sciimpact_model_eval import (
    evaluate_sciimpact_predictions,
    generate_sciimpact_predictions,
)


def write_sft(path):
    rows = [
        {
            "conversations": [
                {"from": "system", "value": "Which paper has more citations?"},
                {
                    "from": "human",
                    "value": (
                        "Paper A: Title: Low Abstract: Narrow. "
                        "Paper B: Title: High Abstract: Broad. "
                        "Which paper has more citations?"
                    ),
                },
                {"from": "gpt", "value": "Paper B has more citations."},
            ]
        },
        {
            "conversations": [
                {"from": "system", "value": "Which paper won the Best Paper award?"},
                {
                    "from": "human",
                    "value": (
                        "Paper A: Title: Winner Abstract: Strong. "
                        "Paper B: Title: Loser Abstract: Weak. "
                        "Which paper should win the Best Paper award?"
                    ),
                },
                {"from": "gpt", "value": "Paper A won the best paper award."},
            ]
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_evaluate_sciimpact_predictions_counts_correct_wrong_and_missing(tmp_path):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "eval"
    write_sft(sft)
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"source_line": 1, "prediction": "Paper B has more citations."}),
                json.dumps({"source_line": 2, "prediction": "Paper B won the best paper award."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_sciimpact_predictions(sft, predictions, output)

    assert report["summary"]["total_examples"] == 2
    assert report["summary"]["covered_examples"] == 2
    assert report["summary"]["correct_examples"] == 1
    assert report["summary"]["wrong_examples"] == 1
    assert report["summary"]["accuracy"] == 0.5
    assert report["by_dimension"]["citation"]["accuracy"] == 1.0
    assert report["by_dimension"]["award"]["accuracy"] == 0.0
    assert (output / "sciimpact_model_eval.json").exists()
    assert (output / "sciimpact_model_eval.md").exists()


def test_cli_evaluate_sciimpact_predictions_writes_report(tmp_path):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "eval"
    write_sft(sft)
    predictions.write_text(
        json.dumps({"source_line": 1, "prediction": "Paper B has more citations."}) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate-sciimpact-predictions",
            "--sft",
            str(sft),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "accuracy=1.000" in result.output
    assert "coverage_rate=0.500" in result.output


def test_generate_sciimpact_predictions_writes_openai_compatible_answers(tmp_path):
    class FakeClient:
        def __init__(self):
            self.payloads = []

        def post(self, url, headers, json, timeout):
            self.payloads.append(json)

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "choices": [
                            {"message": {"content": "Paper B has more citations."}}
                        ]
                    }

            return Response()

    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_sft(sft)
    client = FakeClient()

    manifest = generate_sciimpact_predictions(
        sft,
        predictions,
        base_url="http://localhost:8000/v1",
        model="local-impact",
        client=client,
        limit=1,
    )

    assert manifest["written_prediction_count"] == 1
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "source_line": 1,
            "prediction": "Paper B has more citations.",
            "model": "local-impact",
        }
    ]
    assert "Paper A:" in client.payloads[0]["messages"][1]["content"]
    assert client.payloads[0]["messages"][0]["role"] == "system"


def test_cli_generate_sciimpact_predictions_writes_predictions_with_fake_client(
    monkeypatch,
    tmp_path,
):
    sft = tmp_path / "sft.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_sft(sft)

    def fake_generate(*args, **kwargs):
        predictions.write_text(
            json.dumps(
                {
                    "source_line": 1,
                    "prediction": "Paper B has more citations.",
                    "model": kwargs["model"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"written_prediction_count": 1, "skipped_existing_count": 0}

    import local_impact_ranking.cli as cli

    monkeypatch.setattr(cli, "generate_sciimpact_predictions", fake_generate)

    result = CliRunner().invoke(
        app,
        [
            "generate-sciimpact-predictions",
            "--sft",
            str(sft),
            "--output",
            str(predictions),
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "local-impact",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 1 SciImpact predictions" in result.output
