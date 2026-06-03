import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.wikidata_enrichment import (
    WIKIDATA_SPARQL_URL,
    enrich_papers_with_wikidata,
)


class WikidataClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, params=None, headers=None):
        params = params or {}
        headers = headers or {}
        self.requests.append({"url": url, "params": params, "headers": headers})
        query = params.get("query", "")
        key = "p1" if "10.1038/example" in query else "missing"
        return WikidataResponse(self.payloads.get(key))

    def close(self):
        self.closed = True


class WikidataResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 404 if payload is None else int(payload.get("__status_code__", 200))

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_enrich_papers_with_wikidata_adds_citation_count_and_entity(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.wikidata.jsonl"
    summary_path = tmp_path / "wikidata_summary.json"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Wikidata article",
                "doi": "10.1038/example",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = WikidataClient(
        {
            "p1": {
                "results": {
                    "bindings": [
                        {
                            "work": {"value": "http://www.wikidata.org/entity/Q123"},
                            "workLabel": {"value": "Wikidata article"},
                            "count": {"value": "5"},
                        }
                    ]
                }
            }
        }
    )

    summary = enrich_papers_with_wikidata(
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
    assert client.requests[0]["url"] == WIKIDATA_SPARQL_URL
    assert client.requests[0]["headers"]["Accept"] == "application/sparql-results+json"
    assert metadata["wikidata_enriched"] is True
    assert metadata["wikidata_entity"] == "Q123"
    assert metadata["wikidata_title"] == "Wikidata article"
    assert external_impact["wikidata_cited_by_count"] == 5.0
    assert record["citation_count"] == 5.0
    assert json.loads(summary_path.read_text(encoding="utf-8"))["matched_count"] == 1


def test_enrich_papers_with_wikidata_preserves_unmatched_record(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.wikidata.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Missing Wikidata",
                "doi": "10.1038/missing",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = WikidataClient({"missing": {"results": {"bindings": []}}})

    summary = enrich_papers_with_wikidata(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["matched_count"] == 0
    assert summary["unmatched_count"] == 1
    assert "wikidata_enriched" not in record.get("metadata", {})


def test_enrich_papers_with_wikidata_can_resume_existing_rows(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.wikidata.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Already enriched",
                        "doi": "10.1038/already",
                        "references": [],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Wikidata article",
                        "doi": "10.1038/example",
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
                "doi": "10.1038/already",
                "references": [],
                "citation_count": 4,
                "metadata": {"wikidata_enriched": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = WikidataClient(
        {
            "p1": {
                "results": {
                    "bindings": [
                        {
                            "work": {"value": "http://www.wikidata.org/entity/Q123"},
                            "workLabel": {"value": "Wikidata article"},
                            "count": {"value": "5"},
                        }
                    ]
                }
            }
        }
    )

    summary = enrich_papers_with_wikidata(
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
    assert len(client.requests) == 1
    assert "10.1038/example" in client.requests[0]["params"]["query"]
    assert records[0]["citation_count"] == 4
    assert records[1]["citation_count"] == 5.0


def test_cli_enrich_wikidata_writes_enriched_jsonl(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.wikidata.jsonl"
    summary_path = tmp_path / "wikidata_summary.json"
    input_path.write_text("", encoding="utf-8")

    def fake_enrich(*args, **kwargs):
        assert args[0] == input_path
        assert args[1] == output_path
        assert kwargs["summary_output"] == summary_path
        assert kwargs["timeout"] == 5.0
        assert kwargs["resume"] is True
        output_path.write_text(
            json.dumps({"paper_id": "p1", "title": "x", "references": []}) + "\n",
            encoding="utf-8",
        )
        return {"matched_count": 1, "unmatched_count": 0, "positive_count": 1}

    monkeypatch.setattr("local_impact_ranking.cli.enrich_papers_with_wikidata", fake_enrich)

    result = CliRunner().invoke(
        app,
        [
            "enrich-wikidata",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--timeout",
            "5",
            "--resume",
        ],
    )

    assert result.exit_code == 0
    assert "Wikidata-enriched" in result.output
