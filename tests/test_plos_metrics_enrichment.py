import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.plos_metrics_enrichment import (
    PLOS_SEARCH_URL,
    enrich_papers_with_plos_metrics,
)


class PLOSClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, params=None):
        params = params or {}
        self.requests.append({"url": url, "params": params})
        return PLOSResponse(self.payloads.get((url, params.get("q"))))

    def close(self):
        self.closed = True


class PLOSResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 404 if payload is None else int(payload.get("__status_code__", 200))

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_enrich_papers_with_plos_metrics_adds_usage_evidence(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.plos_metrics.jsonl"
    summary_path = tmp_path / "plos_metrics_summary.json"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Neural Substrate of Cold-Seeking Behavior in Endotoxin Shock",
                "doi": "10.1371/journal.pone.0000001",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    query = 'id:"10.1371/journal.pone.0000001"'
    client = PLOSClient(
        {
            (
                PLOS_SEARCH_URL,
                query,
            ): {
                "response": {
                    "numFound": 1,
                    "docs": [
                        {
                            "id": "10.1371/journal.pone.0000001",
                            "title_display": (
                                "Neural Substrate of Cold-Seeking Behavior in Endotoxin Shock"
                            ),
                            "journal": "PLoS ONE",
                            "publication_date": "2006-12-20T00:00:00Z",
                            "counter_total_all": 29355,
                            "counter_total_month": 11,
                        }
                    ],
                }
            }
        }
    )

    summary = enrich_papers_with_plos_metrics(
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
    assert client.requests[0]["url"] == PLOS_SEARCH_URL
    assert client.requests[0]["params"]["q"] == query
    assert metadata["plos_metrics_enriched"] is True
    assert metadata["plos_title"] == "Neural Substrate of Cold-Seeking Behavior in Endotoxin Shock"
    assert metadata["plos_journal"] == "PLoS ONE"
    assert external_impact["plos_counter_total_all"] == 29355.0
    assert external_impact["plos_counter_total_month"] == 11.0
    assert external_impact["article_usage_count"] == 29355.0
    assert json.loads(summary_path.read_text(encoding="utf-8"))["positive_count"] == 1


def test_enrich_papers_with_plos_metrics_preserves_unmatched_record(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.plos_metrics.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Not PLOS",
                "doi": "10.1038/not-plos",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = PLOSClient(
        {(PLOS_SEARCH_URL, 'id:"10.1038/not-plos"'): {"response": {"numFound": 0, "docs": []}}}
    )

    summary = enrich_papers_with_plos_metrics(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["matched_count"] == 0
    assert summary["unmatched_count"] == 1
    assert "plos_metrics_enriched" not in record.get("metadata", {})


def test_enrich_papers_with_plos_metrics_can_resume_existing_rows(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.plos_metrics.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Already enriched",
                        "doi": "10.1371/journal.pone.already",
                        "references": [],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Needs PLOS metrics",
                        "doi": "10.1371/journal.pone.needs",
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
                "doi": "10.1371/journal.pone.already",
                "references": [],
                "metadata": {"plos_metrics_enriched": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = PLOSClient(
        {
            (
                PLOS_SEARCH_URL,
                'id:"10.1371/journal.pone.needs"',
            ): {
                "response": {
                    "docs": [{"counter_total_all": 10, "counter_total_month": 1}]
                }
            }
        }
    )

    summary = enrich_papers_with_plos_metrics(
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
    assert client.requests[0]["params"]["q"] == 'id:"10.1371/journal.pone.needs"'
    assert records[1]["metadata"]["external_impact"]["article_usage_count"] == 10.0


def test_cli_enrich_plos_metrics_writes_enriched_jsonl(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.plos_metrics.jsonl"
    summary_path = tmp_path / "plos_metrics_summary.json"
    input_path.write_text("", encoding="utf-8")

    def fake_enrich(*args, **kwargs):
        assert args[0] == input_path
        assert args[1] == output_path
        assert kwargs["summary_output"] == summary_path
        assert kwargs["timeout"] == 4.0
        assert kwargs["resume"] is True
        output_path.write_text(
            json.dumps({"paper_id": "p1", "title": "x", "references": []}) + "\n",
            encoding="utf-8",
        )
        return {"matched_count": 1, "positive_count": 1, "unmatched_count": 0}

    monkeypatch.setattr("local_impact_ranking.cli.enrich_papers_with_plos_metrics", fake_enrich)

    result = CliRunner().invoke(
        app,
        [
            "enrich-plos-metrics",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--timeout",
            "4",
            "--resume",
        ],
    )

    assert result.exit_code == 0
    assert "PLOS metrics-enriched" in result.output
