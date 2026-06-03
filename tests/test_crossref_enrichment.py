import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.crossref_enrichment import (
    CROSSREF_WORKS_URL,
    enrich_papers_with_crossref,
)


class CrossrefClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, params=None):
        self.requests.append({"url": url, "params": params or {}})
        return CrossrefResponse(self.payloads.get(url))

    def close(self):
        self.closed = True


class CrossrefResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 404 if payload is None else int(payload.get("__status_code__", 200))

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_enrich_papers_with_crossref_adds_cited_by_count_and_title(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.crossref.jsonl"
    summary_path = tmp_path / "crossref_summary.json"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Example paper",
                "doi": "https://doi.org/10.123/ABC",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = CrossrefClient(
        {
            f"{CROSSREF_WORKS_URL}/10.123%2FABC": {
                "message": {
                    "DOI": "10.123/ABC",
                    "title": ["Example paper"],
                    "publisher": "Example Publisher",
                    "published": {"date-parts": [[2024, 5, 1]]},
                    "is-referenced-by-count": 12,
                }
            }
        }
    )

    summary = enrich_papers_with_crossref(
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
    assert client.requests[0]["url"] == f"{CROSSREF_WORKS_URL}/10.123%2FABC"
    assert metadata["crossref_enriched"] is True
    assert metadata["crossref_title"] == "Example paper"
    assert metadata["crossref_publisher"] == "Example Publisher"
    assert metadata["external_ids"]["doi"] == "10.123/ABC"
    assert external_impact["crossref_is_referenced_by_count"] == 12.0
    assert external_impact["crossref_cited_by_count"] == 12.0
    assert record["citation_count"] == 12.0
    assert record["year"] == 2024
    assert json.loads(summary_path.read_text(encoding="utf-8"))["matched_count"] == 1


def test_enrich_papers_with_crossref_skips_missing_doi(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.crossref.jsonl"
    input_path.write_text(
        json.dumps({"paper_id": "p1", "title": "No DOI", "references": []}) + "\n",
        encoding="utf-8",
    )
    client = CrossrefClient({})

    summary = enrich_papers_with_crossref(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["queryable_count"] == 0
    assert summary["unmatched_count"] == 1
    assert client.requests == []
    assert record["paper_id"] == "p1"


def test_enrich_papers_with_crossref_can_resume_from_existing_output(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.crossref.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Already enriched",
                        "doi": "10.123/already",
                        "references": [],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Needs enrichment",
                        "doi": "10.123/needs",
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
                "doi": "10.123/already",
                "references": [],
                "citation_count": 9,
                "metadata": {"crossref_enriched": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = CrossrefClient(
        {
            f"{CROSSREF_WORKS_URL}/10.123%2Fneeds": {
                "message": {
                    "DOI": "10.123/needs",
                    "title": ["Needs enrichment"],
                    "is-referenced-by-count": 12,
                }
            }
        }
    )

    summary = enrich_papers_with_crossref(
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
    assert client.requests[0]["url"].endswith("10.123%2Fneeds")
    assert [record["paper_id"] for record in records] == ["p1", "p2"]
    assert records[0]["citation_count"] == 9
    assert records[1]["citation_count"] == 12.0


def test_enrich_papers_with_crossref_preserves_404_record(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.crossref.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Missing DOI",
                "doi": "10.999/missing",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = CrossrefClient({})

    summary = enrich_papers_with_crossref(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["matched_count"] == 0
    assert summary["unmatched_count"] == 1
    assert record["paper_id"] == "p1"
    assert "crossref_enriched" not in record.get("metadata", {})


def test_cli_enrich_crossref_writes_enriched_jsonl(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.crossref.jsonl"
    summary_path = tmp_path / "crossref_summary.json"
    input_path.write_text("", encoding="utf-8")

    def fake_enrich(*args, **kwargs):
        assert args[0] == input_path
        assert args[1] == output_path
        assert kwargs["summary_output"] == summary_path
        assert kwargs["timeout"] == 11.0
        output_path.write_text(
            json.dumps({"paper_id": "p1", "title": "x", "references": []}) + "\n",
            encoding="utf-8",
        )
        return {"matched_count": 1, "unmatched_count": 0}

    monkeypatch.setattr("local_impact_ranking.cli.enrich_papers_with_crossref", fake_enrich)

    result = CliRunner().invoke(
        app,
        [
            "enrich-crossref",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--timeout",
            "11",
        ],
    )

    assert result.exit_code == 0
    assert "Crossref-enriched" in result.output
