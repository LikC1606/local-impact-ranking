import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.crossref_data_citations_enrichment import (
    CROSSREF_DATA_CITATIONS_URL,
    enrich_papers_with_crossref_data_citations,
)


class CrossrefDataCitationsClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, params=None):
        params = params or {}
        self.requests.append({"url": url, "params": params})
        return CrossrefDataCitationsResponse(
            self.payloads.get((url, params.get("object-id")))
        )

    def close(self):
        self.closed = True


class CrossrefDataCitationsResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 404 if payload is None else int(payload.get("__status_code__", 200))

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_enrich_papers_with_crossref_data_citations_adds_count(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.crossref_data_citations.jsonl"
    summary_path = tmp_path / "crossref_data_citations_summary.json"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Dataset DOI paper",
                "doi": "10.5281/zenodo.123",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = CrossrefDataCitationsClient(
        {
            (
                CROSSREF_DATA_CITATIONS_URL,
                "10.5281/zenodo.123",
            ): {
                "message": {
                    "total-results": 7,
                    "items-per-page": 0,
                    "items": [],
                }
            }
        }
    )

    summary = enrich_papers_with_crossref_data_citations(
        input_path,
        output_path,
        summary_output=summary_path,
        client=client,
    )

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    metadata = record["metadata"]
    external_impact = metadata["external_impact"]
    assert summary["matched_count"] == 1
    assert summary["positive_count"] == 1
    assert client.requests[0]["url"] == CROSSREF_DATA_CITATIONS_URL
    assert client.requests[0]["params"]["object-id"] == "10.5281/zenodo.123"
    assert client.requests[0]["params"]["rows"] == 0
    assert metadata["crossref_data_citations_enriched"] is True
    assert external_impact["crossref_data_citation_count"] == 7.0
    assert external_impact["crossref_data_citations_total_results"] == 7.0
    assert record["citation_count"] == 7.0
    assert json.loads(summary_path.read_text(encoding="utf-8"))["positive_count"] == 1


def test_enrich_papers_with_crossref_data_citations_records_zero(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.crossref_data_citations.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "No data citations",
                "doi": "10.1038/no-data-cites",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = CrossrefDataCitationsClient(
        {
            (
                CROSSREF_DATA_CITATIONS_URL,
                "10.1038/no-data-cites",
            ): {"message": {"total-results": 0, "items": []}}
        }
    )

    summary = enrich_papers_with_crossref_data_citations(
        input_path,
        output_path,
        client=client,
    )

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    external_impact = record["metadata"]["external_impact"]
    assert summary["matched_count"] == 1
    assert summary["positive_count"] == 0
    assert summary["zero_count"] == 1
    assert external_impact["crossref_data_citation_count"] == 0.0


def test_enrich_papers_with_crossref_data_citations_can_resume_existing_rows(
    tmp_path,
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.crossref_data_citations.jsonl"
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
                "metadata": {"crossref_data_citations_enriched": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = CrossrefDataCitationsClient(
        {
            (
                CROSSREF_DATA_CITATIONS_URL,
                "10.5281/needs",
            ): {"message": {"total-results": 3, "items": []}}
        }
    )

    summary = enrich_papers_with_crossref_data_citations(
        input_path,
        output_path,
        client=client,
        resume=True,
    )

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert summary["resume"] is True
    assert summary["resumed_count"] == 1
    assert summary["matched_count"] == 1
    assert summary["positive_count"] == 1
    assert len(client.requests) == 1
    assert client.requests[0]["params"]["object-id"] == "10.5281/needs"
    assert records[0]["citation_count"] == 9
    assert records[1]["citation_count"] == 3.0


def test_cli_enrich_crossref_data_citations_writes_enriched_jsonl(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.crossref_data_citations.jsonl"
    summary_path = tmp_path / "crossref_data_citations_summary.json"
    input_path.write_text("", encoding="utf-8")

    def fake_enrich(*args, **kwargs):
        assert args[0] == input_path
        assert args[1] == output_path
        assert kwargs["summary_output"] == summary_path
        assert kwargs["timeout"] == 8.0
        assert kwargs["resume"] is True
        output_path.write_text(
            json.dumps({"paper_id": "p1", "title": "x", "references": []}) + "\n",
            encoding="utf-8",
        )
        return {"matched_count": 1, "positive_count": 1, "unmatched_count": 0}

    monkeypatch.setattr(
        "local_impact_ranking.cli.enrich_papers_with_crossref_data_citations",
        fake_enrich,
    )

    result = CliRunner().invoke(
        app,
        [
            "enrich-crossref-data-citations",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--timeout",
            "8",
            "--resume",
        ],
    )

    assert result.exit_code == 0
    assert "Crossref Data Citations-enriched" in result.output
