import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.openaire_enrichment import (
    OPENAIRE_RESEARCH_PRODUCTS_URL,
    enrich_papers_with_openaire,
)


class OpenAIREClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, params=None):
        self.requests.append({"url": url, "params": params or {}})
        key = (url, (params or {}).get("pid"))
        return OpenAIREResponse(self.payloads.get(key))

    def close(self):
        self.closed = True


class OpenAIREResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 404 if payload is None else int(payload.get("__status_code__", 200))

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_enrich_papers_with_openaire_adds_citation_impact_indicators(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.openaire.jsonl"
    summary_path = tmp_path / "openaire_summary.json"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "OpenAIRE paper",
                "doi": "10.123/openaire",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenAIREClient(
        {
            (
                OPENAIRE_RESEARCH_PRODUCTS_URL,
                "10.123/openaire",
            ): {
                "results": [
                    {
                        "id": "openaire____::abc",
                        "mainTitle": "OpenAIRE paper",
                        "publicationDate": "2024-03-02",
                        "type": "publication",
                        "publisher": "Open Journal",
                        "indicators": {
                            "citationImpact": {
                                "citationCount": 7,
                                "influence": 0.12,
                                "popularity": 0.34,
                                "impulse": 0.56,
                                "citationClass": "C2",
                            }
                        },
                    }
                ]
            }
        }
    )

    summary = enrich_papers_with_openaire(
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
    assert client.requests[0]["url"] == OPENAIRE_RESEARCH_PRODUCTS_URL
    assert client.requests[0]["params"]["pid"] == "10.123/openaire"
    assert metadata["openaire_enriched"] is True
    assert metadata["openaire_id"] == "openaire____::abc"
    assert metadata["openaire_title"] == "OpenAIRE paper"
    assert metadata["openaire_type"] == "publication"
    assert metadata["openaire_publisher"] == "Open Journal"
    assert external_impact["openaire_citation_count"] == 7.0
    assert external_impact["openaire_influence"] == 0.12
    assert external_impact["openaire_popularity"] == 0.34
    assert external_impact["openaire_impulse"] == 0.56
    assert external_impact["openaire_citation_class"] == "C2"
    assert record["citation_count"] == 7.0
    assert record["year"] == 2024
    assert json.loads(summary_path.read_text(encoding="utf-8"))["matched_count"] == 1


def test_enrich_papers_with_openaire_preserves_no_result_record(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.openaire.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Missing OpenAIRE",
                "doi": "10.123/missing",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenAIREClient(
        {(OPENAIRE_RESEARCH_PRODUCTS_URL, "10.123/missing"): {"results": []}}
    )

    summary = enrich_papers_with_openaire(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["matched_count"] == 0
    assert summary["unmatched_count"] == 1
    assert record["paper_id"] == "p1"
    assert "openaire_enriched" not in record.get("metadata", {})


def test_enrich_papers_with_openaire_can_resume_from_existing_output(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.openaire.jsonl"
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
                "metadata": {"openaire_enriched": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenAIREClient(
        {
            (
                OPENAIRE_RESEARCH_PRODUCTS_URL,
                "10.123/needs",
            ): {
                "results": [
                    {
                        "id": "openaire____::needs",
                        "mainTitle": "Needs enrichment",
                        "indicators": {"citationImpact": {"citationCount": 12}},
                    }
                ]
            }
        }
    )

    summary = enrich_papers_with_openaire(
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
    assert client.requests[0]["params"]["pid"] == "10.123/needs"
    assert [record["paper_id"] for record in records] == ["p1", "p2"]
    assert records[0]["citation_count"] == 9
    assert records[1]["citation_count"] == 12.0


def test_cli_enrich_openaire_writes_enriched_jsonl(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.openaire.jsonl"
    summary_path = tmp_path / "openaire_summary.json"
    input_path.write_text("", encoding="utf-8")

    def fake_enrich(*args, **kwargs):
        assert args[0] == input_path
        assert args[1] == output_path
        assert kwargs["summary_output"] == summary_path
        assert kwargs["timeout"] == 7.0
        output_path.write_text(
            json.dumps({"paper_id": "p1", "title": "x", "references": []}) + "\n",
            encoding="utf-8",
        )
        return {"matched_count": 1, "unmatched_count": 0}

    monkeypatch.setattr("local_impact_ranking.cli.enrich_papers_with_openaire", fake_enrich)

    result = CliRunner().invoke(
        app,
        [
            "enrich-openaire",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--timeout",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert "OpenAIRE-enriched" in result.output
