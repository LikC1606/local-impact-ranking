import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.europepmc_enrichment import (
    EUROPEPMC_SEARCH_URL,
    enrich_papers_with_europepmc,
)


class EuropePMCClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, params=None):
        self.requests.append({"url": url, "params": params or {}})
        query = (params or {}).get("query", "")
        return EuropePMCResponse(self.payloads.get(query))

    def close(self):
        self.closed = True


class EuropePMCResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 404 if payload is None else 200

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class TimeoutEuropePMCClient:
    def __init__(self):
        self.requests = []

    def get(self, url, params=None):
        self.requests.append({"url": url, "params": params or {}})
        raise TimeoutError("read timed out")


def test_enrich_papers_with_europepmc_resolves_doi_pmid_and_cited_by_count(
    tmp_path,
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.europepmc.jsonl"
    summary_path = tmp_path / "europepmc_summary.json"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "42030147",
                "title": "Initial T cell receptor transgenic cell precursor frequency dictates critical aspects of the CD8(+) T cell response to infection.",
                "doi": "10.1016/j.immuni.2007.04.013",
                "references": [],
                "metadata": {"external_ids": {"pmid": "42030147"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = EuropePMCClient(
        {
            'DOI:"10.1016/j.immuni.2007.04.013"': {
                "resultList": {
                    "result": [
                        {
                            "id": "17555991",
                            "source": "MED",
                            "pmid": "17555991",
                            "pmcid": "PMC1989155",
                            "doi": "10.1016/j.immuni.2007.04.013",
                            "title": "Initial T cell receptor transgenic cell precursor frequency dictates critical aspects of the CD8(+) T cell response to infection.",
                            "journalTitle": "Immunity",
                            "pubYear": "2007",
                            "citedByCount": 334,
                        }
                    ]
                }
            }
        }
    )

    summary = enrich_papers_with_europepmc(
        input_path,
        output_path,
        summary_output=summary_path,
        client=client,
    )

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    metadata = record["metadata"]
    external_ids = metadata["external_ids"]
    external_impact = metadata["external_impact"]
    assert summary["source"] == "europepmc"
    assert summary["matched_count"] == 1
    assert summary["pmid_replaced_count"] == 1
    assert summary["doi_query_count"] == 1
    assert client.requests[0]["url"] == EUROPEPMC_SEARCH_URL
    assert external_ids["pmid"] == "17555991"
    assert external_ids["pmcid"] == "PMC1989155"
    assert external_ids["europepmc"] == "MED:17555991"
    assert metadata["europepmc_replaced_pmid"] == {
        "old": "42030147",
        "new": "17555991",
    }
    assert metadata["europepmc_match_method"] == "doi"
    assert metadata["europepmc_enriched"] is True
    assert external_impact["europepmc_cited_by_count"] == 334.0
    assert record["citation_count"] == 334.0
    assert json.loads(summary_path.read_text(encoding="utf-8"))["matched_count"] == 1


def test_enrich_papers_with_europepmc_preserves_unmatched_record(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.europepmc.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Unmatched paper",
                "doi": "10.999/missing",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = EuropePMCClient({"DOI:\"10.999/missing\"": {"resultList": {"result": []}}})

    summary = enrich_papers_with_europepmc(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["matched_count"] == 0
    assert summary["unmatched_count"] == 1
    assert record["paper_id"] == "p1"
    assert "europepmc_enriched" not in record.get("metadata", {})


def test_enrich_papers_with_europepmc_can_resume_from_existing_output(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.europepmc.jsonl"
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
                "metadata": {"europepmc_enriched": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = EuropePMCClient(
        {
            'DOI:"10.123/needs"': {
                "resultList": {
                    "result": [
                        {
                            "id": "123",
                            "source": "MED",
                            "pmid": "123",
                            "doi": "10.123/needs",
                            "title": "Needs enrichment",
                            "citedByCount": 12,
                        }
                    ]
                }
            }
        }
    )

    summary = enrich_papers_with_europepmc(
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
    assert client.requests[0]["params"]["query"] == 'DOI:"10.123/needs"'
    assert [record["paper_id"] for record in records] == ["p1", "p2"]
    assert records[0]["citation_count"] == 9
    assert records[1]["citation_count"] == 12.0


def test_enrich_papers_with_europepmc_skips_timeout_query(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.europepmc.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Timeout paper",
                "doi": "10.999/timeout",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = TimeoutEuropePMCClient()

    summary = enrich_papers_with_europepmc(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["matched_count"] == 0
    assert summary["unmatched_count"] == 1
    assert summary["query_error_count"] == 1
    assert record["paper_id"] == "p1"


def test_cli_enrich_europepmc_writes_enriched_jsonl(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.europepmc.jsonl"
    summary_path = tmp_path / "europepmc_summary.json"
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

    monkeypatch.setattr(
        "local_impact_ranking.cli.enrich_papers_with_europepmc",
        fake_enrich,
    )

    result = CliRunner().invoke(
        app,
        [
            "enrich-europepmc",
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
    assert "Europe PMC-enriched" in result.output
