import csv
from pathlib import Path

from typer.testing import CliRunner

from local_impact_ranking.cli import app


def test_context_example_demonstrates_context_weighted_score(tmp_path):
    input_path = Path("examples/context_papers.jsonl")
    output_dir = tmp_path / "context-run"

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--top-k",
            "1",
            "--credit-mode",
            "context_weighted",
        ],
    )

    assert result.exit_code == 0
    with (output_dir / "rankings" / "papers.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "rank": "1",
            "id": "r1",
            "score": "1.500000",
            "core_score": "1.000000",
            "expanded_score": "1.500000",
            "normalized_score": "100.000000",
            "percentile": "100.000000",
        }
    ]
