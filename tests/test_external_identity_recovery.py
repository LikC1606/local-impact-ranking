import json

from typer.testing import CliRunner

from local_impact_ranking.cli import app
from local_impact_ranking.external_identity_recovery import (
    recover_external_identities,
)


class OpenAlexRecoveryClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, params=None):
        self.requests.append({"url": url, "params": params or {}})
        return OpenAlexRecoveryResponse(self.payloads.get(url))

    def close(self):
        self.closed = True


class OpenAlexRecoveryResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 404 if payload is None else 200

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_recover_external_identities_recovers_mismatched_paper_from_title_search(
    tmp_path,
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.recovered.jsonl"
    summary_path = tmp_path / "identity_recovery_summary.json"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Resting-State Coactivation Patterns of Language Reorganization in Brain Tumors",
                "citation_count": 75,
                "references": [],
                "metadata": {
                    "europepmc_title": "TbCAPs: A toolbox for co-activation pattern analysis.",
                    "external_ids": {
                        "doi": "10.1016/j.neuroimage.2020.116621",
                        "pmid": "32058000",
                        "openalex": "https://openalex.org/W2979924736",
                    },
                    "external_impact": {
                        "crossref_is_referenced_by_count": 75,
                        "europepmc_cited_by_count": 68,
                        "ncbi_pubmed_citedin_count": 34,
                        "datacite_citation_count": 4,
                        "openaire_citation_count": 7,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenAlexRecoveryClient(
        {
            "https://api.openalex.org/works": {
                "results": [
                    {
                        "id": "https://openalex.org/W777",
                        "doi": "https://doi.org/10.1155/np/1421115",
                        "display_name": "Resting-State Coactivation Patterns of Language Reorganization in Brain Tumors",
                        "publication_year": 2026,
                        "cited_by_count": 0,
                        "counts_by_year": [],
                        "citation_normalized_percentile": {"value": 0.12},
                    }
                ]
            }
        }
    )

    summary = recover_external_identities(
        input_path,
        output_path,
        summary_output=summary_path,
        client=client,
    )

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    metadata = record["metadata"]
    external_ids = metadata["external_ids"]
    external_impact = metadata["external_impact"]
    recovery = metadata["external_identity_recovery"]
    assert summary["mismatch_count"] == 1
    assert summary["recovered_count"] == 1
    assert external_ids["doi"] == "https://doi.org/10.1155/np/1421115"
    assert external_ids["openalex"] == "https://openalex.org/W777"
    assert "pmid" not in external_ids
    assert metadata["rejected_external_ids"][0]["external_ids"]["pmid"] == "32058000"
    assert metadata["openalex_title"] == (
        "Resting-State Coactivation Patterns of Language Reorganization in Brain Tumors"
    )
    assert external_impact["openalex_cited_by_count"] == 0.0
    assert "crossref_is_referenced_by_count" not in external_impact
    assert "europepmc_cited_by_count" not in external_impact
    assert "ncbi_pubmed_citedin_count" not in external_impact
    assert "datacite_citation_count" not in external_impact
    assert "openaire_citation_count" not in external_impact
    assert record["citation_count"] == 0.0
    assert recovery["status"] == "recovered"
    assert recovery["source"] == "openalex_title_search"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["recovered_count"] == 1


def test_recover_external_identities_quarantines_unresolved_mismatch(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.recovered.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Specific target title",
                "citation_count": 75,
                "references": [],
                "metadata": {
                    "europepmc_title": "Different external article",
                    "external_ids": {
                        "doi": "10.123/wrong",
                        "pmid": "111",
                    },
                    "external_impact": {
                        "crossref_is_referenced_by_count": 75,
                        "icite_citation_count": 70,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenAlexRecoveryClient(
        {
            "https://api.openalex.org/works": {
                "results": [
                    {
                        "id": "https://openalex.org/W999",
                        "doi": "https://doi.org/10.555/other",
                        "display_name": "Another unrelated work",
                        "cited_by_count": 99,
                    }
                ]
            }
        }
    )

    summary = recover_external_identities(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    metadata = record["metadata"]
    assert summary["mismatch_count"] == 1
    assert summary["unresolved_count"] == 1
    assert metadata["external_ids"] == {}
    assert metadata["rejected_external_ids"][0]["external_ids"]["doi"] == "10.123/wrong"
    assert metadata["external_impact"] == {}
    assert "citation_count" not in record
    assert metadata["external_identity_recovery"]["status"] == "unresolved"


def test_recover_external_identities_removes_openalex_field_from_unresolved_mismatch(
    tmp_path,
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.recovered.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Specific target title",
                "field": "wrong external field",
                "references": [],
                "metadata": {
                    "field_source": "openalex_primary_topic",
                    "openalex_title": "Different external article",
                    "external_ids": {"openalex": "https://openalex.org/W1"},
                    "external_impact": {"openalex_cited_by_count": 99},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenAlexRecoveryClient({})

    recover_external_identities(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert "field" not in record
    assert "field_source" not in record["metadata"]


def test_recover_external_identities_preserves_matched_record(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.recovered.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Matching title",
                "references": [],
                "metadata": {
                    "europepmc_title": "Matching title",
                    "external_ids": {"doi": "10.123/correct"},
                    "external_impact": {"crossref_is_referenced_by_count": 5},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenAlexRecoveryClient({})

    summary = recover_external_identities(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["mismatch_count"] == 0
    assert client.requests == []
    assert record["metadata"]["external_ids"]["doi"] == "10.123/correct"
    assert record["metadata"]["external_impact"]["crossref_is_referenced_by_count"] == 5


def test_recover_external_identities_can_resume_existing_rows(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.recovered.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Already fixed",
                        "references": [],
                        "metadata": {"europepmc_title": "Wrong external title"},
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Needs recovery",
                        "references": [],
                        "metadata": {"europepmc_title": "Different external title"},
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
                "title": "Already fixed",
                "references": [],
                "citation_count": 9,
                "metadata": {
                    "external_identity_recovery": {"status": "recovered"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenAlexRecoveryClient(
        {
            "https://api.openalex.org/works": {
                "results": [
                    {
                        "id": "https://openalex.org/W222",
                        "doi": "https://doi.org/10.123/needs",
                        "display_name": "Needs recovery",
                        "publication_year": 2026,
                        "cited_by_count": 2,
                    }
                ]
            }
        }
    )

    summary = recover_external_identities(
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
    assert summary["mismatch_count"] == 1
    assert summary["recovered_count"] == 1
    assert len(client.requests) == 1
    assert records[0]["citation_count"] == 9
    assert records[1]["metadata"]["external_identity_recovery"]["status"] == "recovered"


def test_cli_recover_external_identities_writes_recovered_jsonl(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.recovered.jsonl"
    summary_path = tmp_path / "identity_recovery_summary.json"
    input_path.write_text("", encoding="utf-8")

    def fake_recover(*args, **kwargs):
        assert args[0] == input_path
        assert args[1] == output_path
        assert kwargs["summary_output"] == summary_path
        assert kwargs["timeout"] == 13.0
        assert kwargs["min_title_similarity"] == 0.91
        assert kwargs["resume"] is True
        output_path.write_text(
            json.dumps({"paper_id": "p1", "title": "x", "references": []}) + "\n",
            encoding="utf-8",
        )
        return {"recovered_count": 1, "unresolved_count": 0}

    monkeypatch.setattr(
        "local_impact_ranking.cli.recover_external_identities",
        fake_recover,
    )

    result = CliRunner().invoke(
        app,
        [
            "recover-external-identities",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--timeout",
            "13",
            "--min-title-similarity",
            "0.91",
            "--resume",
        ],
    )

    assert result.exit_code == 0
    assert "identity-recovered" in result.output
