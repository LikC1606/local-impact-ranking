import csv
import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.sciimpact_baseline import (
    score_sciimpact_text_baseline,
    write_sciimpact_text_baseline_rankings,
)


def write_papers(path):
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "low",
                        "title": "Small note",
                        "abstract": "A narrow description.",
                        "references": [],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "high",
                        "title": "Open benchmark dataset for scalable neural network evaluation",
                        "abstract": (
                            "We introduce a large open-source benchmark dataset and "
                            "state-of-the-art method used for broad reproducible evaluation."
                        ),
                        "references": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_score_sciimpact_text_baseline_rewards_impact_cues():
    low = score_sciimpact_text_baseline(
        title="Small note",
        abstract="A narrow description.",
        item_type="paper",
        impact_dimension="citation",
    )
    high = score_sciimpact_text_baseline(
        title="Open benchmark dataset for scalable neural network evaluation",
        abstract=(
            "We introduce a large open-source benchmark dataset and "
            "state-of-the-art method used for broad reproducible evaluation."
        ),
        item_type="paper",
        impact_dimension="citation",
    )

    assert high > low


def test_write_sciimpact_text_baseline_rankings_outputs_pairwise_compatible_csv(tmp_path):
    papers = tmp_path / "papers.jsonl"
    output = tmp_path / "rankings.csv"
    write_papers(papers)

    manifest = write_sciimpact_text_baseline_rankings(papers, output)

    assert manifest["paper_count"] == 2
    assert manifest["output_path"] == str(output)
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert rows[0]["id"] == "high"
    assert float(rows[0]["final_score"]) > float(rows[1]["final_score"])
    assert rows[0]["baseline_name"] == "sciimpact_text_prior"


def test_cli_sciimpact_text_baseline_writes_rankings(tmp_path):
    papers = tmp_path / "papers.jsonl"
    output = tmp_path / "rankings.csv"
    write_papers(papers)

    result = CliRunner().invoke(
        app,
        [
            "sciimpact-text-baseline",
            "--input",
            str(papers),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote SciImpact text baseline rankings for 2 papers" in result.output
    assert output.exists()
