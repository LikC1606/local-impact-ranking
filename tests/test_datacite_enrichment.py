import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.datacite_enrichment import (
    DATACITE_DOIS_URL,
    enrich_papers_with_datacite,
)


class DataCiteClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, params=None):
        self.requests.append({"url": url, "params": params or {}})
        return DataCiteResponse(self.payloads.get(url))

    def close(self):
        self.closed = True


class DataCiteResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 404 if payload is None else int(payload.get("__status_code__", 200))

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_enrich_papers_with_datacite_adds_usage_and_citation_evidence(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.datacite.jsonl"
    summary_path = tmp_path / "datacite_summary.json"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Dataset paper",
                "doi": "10.5281/zenodo.12345",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = DataCiteClient(
        {
            f"{DATACITE_DOIS_URL}/10.5281%2Fzenodo.12345": {
                "data": {
                    "id": "10.5281/zenodo.12345",
                    "attributes": {
                        "doi": "10.5281/zenodo.12345",
                        "titles": [{"title": "Dataset paper"}],
                        "publisher": "Zenodo",
                        "publicationYear": 2025,
                        "types": {"resourceTypeGeneral": "Dataset"},
                        "citationCount": 4,
                        "viewCount": 100,
                        "downloadCount": 20,
                    },
                }
            }
        }
    )

    summary = enrich_papers_with_datacite(
        input_path,
        output_path,
        summary_output=summary_path,
        client=client,
    )

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    metadata = record["metadata"]
    external_impact = metadata["external_impact"]
    assert summary["matched_count"] == 1
    assert summary["doi_query_count"] == 1
    assert metadata["datacite_enriched"] is True
    assert metadata["datacite_title"] == "Dataset paper"
    assert metadata["datacite_publisher"] == "Zenodo"
    assert metadata["datacite_resource_type"] == "Dataset"
    assert metadata["external_ids"]["doi"] == "10.5281/zenodo.12345"
    assert external_impact["datacite_citation_count"] == 4.0
    assert external_impact["datacite_view_count"] == 100.0
    assert external_impact["datacite_download_count"] == 20.0
    assert record["citation_count"] == 4.0
    assert record["year"] == 2025
    assert json.loads(summary_path.read_text(encoding="utf-8"))["matched_count"] == 1


def test_enrich_papers_with_datacite_preserves_404_record(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.datacite.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Article DOI",
                "doi": "10.1117/not-in-datacite",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = DataCiteClient({})

    summary = enrich_papers_with_datacite(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["matched_count"] == 0
    assert summary["unmatched_count"] == 1
    assert record["paper_id"] == "p1"
    assert "datacite_enriched" not in record.get("metadata", {})


def test_enrich_papers_with_datacite_can_resume_from_existing_output(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.datacite.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Already enriched",
                        "doi": "10.5281/already",
                        "references": [],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Needs enrichment",
                        "doi": "10.5281/needs",
                        "references": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Already enriched",
                "doi": "10.5281/already",
                "references": [],
                "citation_count": 9,
                "metadata": {"datacite_enriched": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = DataCiteClient(
        {
            f"{DATACITE_DOIS_URL}/10.5281%2Fneeds": {
                "data": {
                    "attributes": {
                        "doi": "10.5281/needs",
                        "titles": [{"title": "Needs enrichment"}],
                        "citationCount": 12,
                    }
                }
            }
        }
    )

    summary = enrich_papers_with_datacite(
        input_path,
        output_path,
        client=client,
        resume=True,
    )

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["resumed_count"] == 1
    assert summary["matched_count"] == 1
    assert len(client.requests) == 1
    assert client.requests[0]["url"].endswith("10.5281%2Fneeds")
    assert [record["paper_id"] for record in records] == ["p1", "p2"]
    assert records[0]["citation_count"] == 9
    assert records[1]["citation_count"] == 12.0


def test_cli_enrich_datacite_writes_enriched_jsonl(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.datacite.jsonl"
    summary_path = tmp_path / "datacite_summary.json"
    input_path.write_text("", encoding="utf-8")

    def fake_enrich(*args, **kwargs):
        assert args[0] == input_path
        assert args[1] == output_path
        assert kwargs["summary_output"] == summary_path
        assert kwargs["timeout"] == 9.0
        output_path.write_text(
            json.dumps({"paper_id": "p1", "title": "x", "references": []}) + "\n",
            encoding="utf-8",
        )
        return {"matched_count": 1, "unmatched_count": 0}

    monkeypatch.setattr("local_impact_ranking.cli.enrich_papers_with_datacite", fake_enrich)

    result = CliRunner().invoke(
        app,
        [
            "enrich-datacite",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--timeout",
            "9",
        ],
    )

    assert result.exit_code == 0
    assert "DataCite-enriched" in result.output
