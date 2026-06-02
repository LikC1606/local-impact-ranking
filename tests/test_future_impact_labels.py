import csv
import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.future_impact_labels import (
    derive_future_impact_labels,
    write_future_impact_labels,
)


def test_derive_future_impact_labels_uses_year_window_citation_increments(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "A",
                        "field": "AI",
                        "references": [],
                        "metadata": {
                            "citation_counts_by_year": {
                                "2020": 3,
                                "2021": 5,
                                "2022": 7,
                                "2023": 11,
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "B",
                        "field": "AI",
                        "references": [],
                        "metadata": {
                            "citations_by_year": [
                                {"year": 2021, "count": 2},
                                {"year": 2022, "count": 4},
                                {"year": 2023, "count": 8},
                            ]
                        },
                    }
                ),
                json.dumps({"paper_id": "p3", "title": "No history", "references": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = derive_future_impact_labels(input_path, start_year=2021, end_year=2022)

    assert report["summary"] == {
        "paper_count": 3,
        "label_count": 2,
        "skipped_count": 1,
        "start_year": 2021,
        "end_year": 2022,
        "label_column": "future_impact",
        "normalization": "raw",
        "label_definition": "sum of yearly citation counts from 2021 through 2022",
    }
    by_id = {row["paper_id"]: row for row in report["labels"]}
    assert by_id["p1"]["future_impact"] == 12.0
    assert by_id["p2"]["future_impact"] == 6.0
    assert report["skipped"][0]["paper_id"] == "p3"


def test_derive_future_impact_labels_can_field_year_percentile_normalize(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    records = []
    for index, count in enumerate([1, 10, 100], start=1):
        records.append(
            {
                "paper_id": f"ai{index}",
                "title": f"AI {index}",
                "field": "AI",
                "year": 2020,
                "references": [],
                "metadata": {"citation_counts_by_year": {"2023": count}},
            }
        )
    for index, count in enumerate([2, 20], start=1):
        records.append(
            {
                "paper_id": f"bio{index}",
                "title": f"Bio {index}",
                "field": "Bio",
                "year": 2020,
                "references": [],
                "metadata": {"citation_counts_by_year": {"2023": count}},
            }
        )
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    report = derive_future_impact_labels(
        input_path,
        start_year=2023,
        end_year=2023,
        normalization="field_year_percentile",
    )

    assert report["summary"]["normalization"] == "field_year_percentile"
    assert report["summary"]["label_definition"] == (
        "field/year percentile of future citation counts from 2023 through 2023"
    )
    by_id = {row["paper_id"]: row for row in report["labels"]}
    assert by_id["ai1"]["future_impact_raw"] == 1.0
    assert by_id["ai1"]["future_impact"] == 0.0
    assert by_id["ai2"]["future_impact"] == 0.5
    assert by_id["ai3"]["future_impact"] == 1.0
    assert by_id["bio1"]["future_impact"] == 0.0
    assert by_id["bio2"]["future_impact"] == 1.0


def test_write_future_impact_labels_writes_jsonl_csv_and_manifest(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "labels"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "A",
                "field": "AI",
                "references": [],
                "metadata": {"citation_counts_by_year": {"2021": 5}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = write_future_impact_labels(
        input_path,
        output_dir,
        start_year=2021,
        end_year=2021,
        normalization="log1p",
    )

    assert report["summary"]["label_count"] == 1
    assert report["summary"]["normalization"] == "log1p"
    assert (output_dir / "future_impact_labels.jsonl").exists()
    assert (output_dir / "future_impact_labels.csv").exists()
    assert (output_dir / "future_impact_labels_manifest.json").exists()
    with (output_dir / "future_impact_labels.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert round(float(rows[0]["future_impact"]), 6) == 1.791759
    assert rows[0]["future_impact_raw"] == "5.0"


def test_cli_derive_future_impact_labels_writes_outputs(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_dir = tmp_path / "labels"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "A",
                "references": [],
                "metadata": {"citation_counts_by_year": {"2021": 5}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "derive-future-impact-labels",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--start-year",
            "2021",
            "--end-year",
            "2021",
            "--normalization",
            "field-year-percentile",
        ],
    )

    assert result.exit_code == 0
    assert "labels=1 skipped=0" in result.output
    assert "normalization=field_year_percentile" in result.output
    assert (output_dir / "future_impact_labels.jsonl").exists()
