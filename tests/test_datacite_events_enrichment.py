import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.datacite_events_enrichment import (
    DATACITE_EVENTS_URL,
    enrich_papers_with_datacite_events,
)


class DataCiteEventsClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, params=None):
        params = params or {}
        self.requests.append({"url": url, "params": params})
        return DataCiteEventsResponse(self.payloads.get((url, params.get("obj-id"))))

    def close(self):
        self.closed = True


class DataCiteEventsResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 404 if payload is None else int(payload.get("__status_code__", 200))

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_enrich_papers_with_datacite_events_counts_only_citation_relations(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.datacite_events.jsonl"
    summary_path = tmp_path / "datacite_events_summary.json"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Cited article",
                "doi": "10.1038/example",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = DataCiteEventsClient(
        {
            (
                DATACITE_EVENTS_URL,
                "https://doi.org/10.1038/example",
            ): {
                "meta": {
                    "total": 11,
                    "sources": [
                        {"id": "crossref", "count": 8},
                        {"id": "datacite-crossref", "count": 3},
                    ],
                    "relation-types": [
                        {"id": "references", "count": 8},
                        {"id": "cites", "count": 2},
                        {"id": "is-supplement-to", "count": 1},
                    ],
                }
            }
        }
    )

    summary = enrich_papers_with_datacite_events(
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
    assert summary["zero_count"] == 0
    assert client.requests[0]["url"] == DATACITE_EVENTS_URL
    assert client.requests[0]["params"]["obj-id"] == "https://doi.org/10.1038/example"
    assert client.requests[0]["params"]["page[size]"] == 0
    assert metadata["datacite_events_enriched"] is True
    assert metadata["datacite_events_obj_id"] == "https://doi.org/10.1038/example"
    assert external_impact["datacite_events_cited_by_count"] == 10.0
    assert external_impact["datacite_events_total_event_count"] == 11.0
    assert external_impact["datacite_events_references_count"] == 8.0
    assert external_impact["datacite_events_cites_count"] == 2.0
    assert external_impact["datacite_events_crossref_count"] == 8.0
    assert external_impact["datacite_events_datacite_crossref_count"] == 3.0
    assert record["citation_count"] == 10.0
    assert json.loads(summary_path.read_text(encoding="utf-8"))["positive_count"] == 1


def test_enrich_papers_with_datacite_events_records_verified_zero(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.datacite_events.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Uncited article",
                "doi": "10.1038/uncited",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = DataCiteEventsClient(
        {
            (
                DATACITE_EVENTS_URL,
                "https://doi.org/10.1038/uncited",
            ): {"meta": {"total": 0, "relation-types": [], "sources": []}}
        }
    )

    summary = enrich_papers_with_datacite_events(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    external_impact = record["metadata"]["external_impact"]
    assert summary["matched_count"] == 1
    assert summary["positive_count"] == 0
    assert summary["zero_count"] == 1
    assert external_impact["datacite_events_cited_by_count"] == 0.0
    assert external_impact["datacite_events_total_event_count"] == 0.0


def test_enrich_papers_with_datacite_events_can_resume_from_existing_output(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.datacite_events.jsonl"
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
                        "title": "Needs enrichment",
                        "doi": "10.1038/needs",
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
                "citation_count": 9,
                "metadata": {"datacite_events_enriched": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = DataCiteEventsClient(
        {
            (
                DATACITE_EVENTS_URL,
                "https://doi.org/10.1038/needs",
            ): {
                "meta": {
                    "total": 12,
                    "relation-types": [{"id": "references", "count": 12}],
                    "sources": [],
                }
            }
        }
    )

    summary = enrich_papers_with_datacite_events(
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
    assert client.requests[0]["params"]["obj-id"] == "https://doi.org/10.1038/needs"
    assert [record["paper_id"] for record in records] == ["p1", "p2"]
    assert records[0]["citation_count"] == 9
    assert records[1]["citation_count"] == 12.0


def test_cli_enrich_datacite_events_writes_enriched_jsonl(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.datacite_events.jsonl"
    summary_path = tmp_path / "datacite_events_summary.json"
    input_path.write_text("", encoding="utf-8")

    def fake_enrich(*args, **kwargs):
        assert args[0] == input_path
        assert args[1] == output_path
        assert kwargs["summary_output"] == summary_path
        assert kwargs["timeout"] == 6.0
        output_path.write_text(
            json.dumps({"paper_id": "p1", "title": "x", "references": []}) + "\n",
            encoding="utf-8",
        )
        return {"matched_count": 1, "unmatched_count": 0, "positive_count": 1}

    monkeypatch.setattr(
        "local_impact_ranking.cli.enrich_papers_with_datacite_events",
        fake_enrich,
    )

    result = CliRunner().invoke(
        app,
        [
            "enrich-datacite-events",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--timeout",
            "6",
        ],
    )

    assert result.exit_code == 0
    assert "DataCite Events-enriched" in result.output
