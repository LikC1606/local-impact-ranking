import json

from typer.testing import CliRunner

from local_impact_ranking import ncbi_citation_enrichment as ncbi
from local_impact_ranking.cli import app
from local_impact_ranking.ncbi_citation_enrichment import (
    NCBI_ELINK_URL,
    _elink_counts_by_id,
    _fetch_elink_counts,
    enrich_papers_with_ncbi_citations,
)


class NCBIClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, params=None):
        self.requests.append({"url": url, "params": params or {}})
        id_value = (params or {}).get("id")
        if isinstance(id_value, (list, tuple)):
            id_value = ",".join(str(value) for value in id_value)
        key = (
            (params or {}).get("dbfrom"),
            (params or {}).get("linkname"),
            str(id_value),
        )
        return NCBIResponse(self.payloads.get(key))

    def close(self):
        self.closed = True


class NCBIResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 404 if payload is None else 200

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class TimeoutNCBIClient:
    def __init__(self):
        self.requests = []

    def get(self, url, params=None):
        self.requests.append({"url": url, "params": params or {}})
        raise TimeoutError("read timed out")


def test_enrich_papers_with_ncbi_citations_adds_pubmed_and_pmc_cited_by_counts(
    tmp_path,
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.ncbi.jsonl"
    summary_path = tmp_path / "ncbi_summary.json"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "PubMed article",
                "references": [],
                "metadata": {
                    "external_ids": {"pmid": "17555991", "pmcid": "PMC1989155"}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = NCBIClient(
        {
            ("pubmed", "pubmed_pubmed_citedin", "17555991"): {
                "linksets": [
                    {
                        "linksetdbs": [
                            {
                                "linkname": "pubmed_pubmed_citedin",
                                "links": ["1", "2", "2"],
                            }
                        ]
                    }
                ]
            },
            ("pmc", "pmc_pmc_citedby", "1989155"): {
                "linksets": [
                    {
                        "linksetdbs": [
                            {"linkname": "pmc_pmc_citedby", "links": ["10", "11"]}
                        ]
                    }
                ]
            },
        }
    )

    summary = enrich_papers_with_ncbi_citations(
        input_path,
        output_path,
        summary_output=summary_path,
        client=client,
    )

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    external_impact = record["metadata"]["external_impact"]
    assert summary["source"] == "ncbi_elink"
    assert summary["matched_count"] == 1
    assert summary["pubmed_query_count"] == 1
    assert summary["pmc_query_count"] == 1
    assert client.requests[0]["url"] == NCBI_ELINK_URL
    assert external_impact["ncbi_pubmed_citedin_count"] == 2.0
    assert external_impact["ncbi_pmc_cited_by_count"] == 2.0
    assert record["citation_count"] == 2.0
    assert record["metadata"]["ncbi_citation_enriched"] is True
    assert json.loads(summary_path.read_text(encoding="utf-8"))["matched_count"] == 1


def test_enrich_papers_with_ncbi_citations_batches_pubmed_ids(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.ncbi.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "One",
                        "references": [],
                        "metadata": {"external_ids": {"pmid": "111"}},
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Two",
                        "references": [],
                        "metadata": {"external_ids": {"pmid": "222"}},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = NCBIClient(
        {
            ("pubmed", "pubmed_pubmed_citedin", "111,222"): {
                "linksets": [
                    {
                        "ids": ["111"],
                        "linksetdbs": [
                            {"linkname": "pubmed_pubmed_citedin", "links": ["1"]}
                        ],
                    },
                    {
                        "ids": ["222"],
                        "linksetdbs": [
                            {
                                "linkname": "pubmed_pubmed_citedin",
                                "links": ["2", "3"],
                            }
                        ],
                    },
                ]
            }
        }
    )

    summary = enrich_papers_with_ncbi_citations(input_path, output_path, client=client)

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert summary["pubmed_query_count"] == 1
    assert len(client.requests) == 1
    assert client.requests[0]["params"]["id"] == ["111", "222"]
    assert records[0]["metadata"]["external_impact"]["ncbi_pubmed_citedin_count"] == 1.0
    assert records[1]["metadata"]["external_impact"]["ncbi_pubmed_citedin_count"] == 2.0


def test_enrich_papers_with_ncbi_citations_batches_pmc_ids_as_repeated_params(
    tmp_path,
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.ncbi.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "PMC One",
                        "references": [],
                        "metadata": {"external_ids": {"pmcid": "PMC1989155"}},
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "PMC Two",
                        "references": [],
                        "metadata": {"external_ids": {"pmcid": "PMC13055105"}},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = NCBIClient(
        {
            ("pmc", "pmc_pmc_citedby", "13055105,1989155"): {
                "linksets": [
                    {
                        "ids": ["13055105"],
                        "linksetdbs": [
                            {"linkname": "pmc_pmc_citedby", "links": []}
                        ],
                    },
                    {
                        "ids": ["1989155"],
                        "linksetdbs": [
                            {
                                "linkname": "pmc_pmc_citedby",
                                "links": ["10", "11", "11"],
                            }
                        ],
                    },
                ]
            }
        }
    )

    enrich_papers_with_ncbi_citations(input_path, output_path, client=client)

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert client.requests[0]["params"]["id"] == ["13055105", "1989155"]
    assert records[0]["metadata"]["external_impact"]["ncbi_pmc_cited_by_count"] == 2.0
    assert records[1]["metadata"]["external_impact"]["ncbi_pmc_cited_by_count"] == 0.0


def test_elink_counts_by_id_ignores_aggregate_multi_id_linksets():
    counts = _elink_counts_by_id(
        {
            "linksets": [
                {
                    "ids": ["1989155", "13055105"],
                    "linksetdbs": [
                        {
                            "linkname": "pmc_pmc_citedby",
                            "links": [str(index) for index in range(246)],
                        }
                    ],
                }
            ]
        },
        "pmc_pmc_citedby",
        ["13055105", "1989155"],
    )

    assert counts == {"13055105": 0.0, "1989155": 0.0}


def test_fetch_elink_counts_chunks_repeated_id_requests(monkeypatch):
    monkeypatch.setattr(ncbi, "NCBI_ELINK_BATCH_SIZE", 2)
    client = NCBIClient(
        {
            ("pubmed", "pubmed_pubmed_citedin", "111,222"): {
                "linksets": [
                    {
                        "ids": ["111"],
                        "linksetdbs": [
                            {"linkname": "pubmed_pubmed_citedin", "links": ["1"]}
                        ],
                    },
                    {
                        "ids": ["222"],
                        "linksetdbs": [
                            {"linkname": "pubmed_pubmed_citedin", "links": ["2", "3"]}
                        ],
                    },
                ]
            },
            ("pubmed", "pubmed_pubmed_citedin", "333"): {
                "linksets": [
                    {
                        "ids": ["333"],
                        "linksetdbs": [
                            {"linkname": "pubmed_pubmed_citedin", "links": ["4"]}
                        ],
                    }
                ]
            },
        }
    )

    counts = _fetch_elink_counts(
        client,
        dbfrom="pubmed",
        linkname="pubmed_pubmed_citedin",
        identifiers=["111", "222", "333"],
    )

    assert [request["params"]["id"] for request in client.requests] == [
        ["111", "222"],
        ["333"],
    ]
    assert counts == {"111": 1.0, "222": 2.0, "333": 1.0}


def test_enrich_papers_with_ncbi_citations_does_not_promote_pmc_count_to_generic_citations(
    tmp_path,
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.ncbi.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "PMC audit-only article",
                "citation_count": 6,
                "references": [],
                "metadata": {
                    "external_ids": {"pmid": "111", "pmcid": "PMC999"},
                    "external_impact": {"crossref_is_referenced_by_count": 6},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = NCBIClient(
        {
            ("pubmed", "pubmed_pubmed_citedin", "111"): {
                "linksets": [
                    {
                        "ids": ["111"],
                        "linksetdbs": [
                            {"linkname": "pubmed_pubmed_citedin", "links": []}
                        ],
                    }
                ]
            },
            ("pmc", "pmc_pmc_citedby", "999"): {
                "linksets": [
                    {
                        "ids": ["999"],
                        "linksetdbs": [
                            {
                                "linkname": "pmc_pmc_citedby",
                                "links": [str(index) for index in range(246)],
                            }
                        ],
                    }
                ]
            },
        }
    )

    enrich_papers_with_ncbi_citations(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    external_impact = record["metadata"]["external_impact"]
    assert external_impact["ncbi_pubmed_citedin_count"] == 0.0
    assert external_impact["ncbi_pmc_cited_by_count"] == 246.0
    assert record["citation_count"] == 6
    assert record["metadata"].get("citation_count") != 246.0


def test_enrich_papers_with_ncbi_citations_can_resume_without_refetching_ids(
    tmp_path,
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.ncbi.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Already enriched",
                        "references": [],
                        "metadata": {"external_ids": {"pmid": "111"}},
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Needs enrichment",
                        "references": [],
                        "metadata": {"external_ids": {"pmid": "222"}},
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
                "references": [],
                "citation_count": 4,
                "metadata": {"ncbi_citation_enriched": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = NCBIClient(
        {
            ("pubmed", "pubmed_pubmed_citedin", "222"): {
                "linksets": [
                    {
                        "ids": ["222"],
                        "linksetdbs": [
                            {
                                "linkname": "pubmed_pubmed_citedin",
                                "links": ["2", "3"],
                            }
                        ],
                    }
                ]
            }
        }
    )

    summary = enrich_papers_with_ncbi_citations(
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
    assert summary["skipped_count"] == 0
    assert len(client.requests) == 1
    assert client.requests[0]["params"]["id"] == ["222"]
    assert records[0]["citation_count"] == 4
    assert records[1]["citation_count"] == 2.0


def test_enrich_papers_with_ncbi_citations_skips_missing_identifiers(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.ncbi.jsonl"
    input_path.write_text(
        json.dumps({"paper_id": "p1", "title": "No IDs", "references": []}) + "\n",
        encoding="utf-8",
    )
    client = NCBIClient({})

    summary = enrich_papers_with_ncbi_citations(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["skipped_count"] == 1
    assert summary["matched_count"] == 0
    assert client.requests == []
    assert "ncbi_citation_enriched" not in record.get("metadata", {})


def test_enrich_papers_with_ncbi_citations_skips_timeout_query(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.ncbi.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "PubMed article",
                "references": [],
                "metadata": {"external_ids": {"pmid": "17555991"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = TimeoutNCBIClient()

    summary = enrich_papers_with_ncbi_citations(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["matched_count"] == 0
    assert summary["query_error_count"] == 1
    assert "ncbi_citation_enriched" not in record.get("metadata", {})


def test_cli_enrich_ncbi_citations_writes_enriched_jsonl(monkeypatch, tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.ncbi.jsonl"
    summary_path = tmp_path / "ncbi_summary.json"
    input_path.write_text("", encoding="utf-8")

    def fake_enrich(*args, **kwargs):
        assert args[0] == input_path
        assert args[1] == output_path
        assert kwargs["summary_output"] == summary_path
        assert kwargs["timeout"] == 13.0
        assert kwargs["resume"] is True
        output_path.write_text(
            json.dumps({"paper_id": "p1", "title": "x", "references": []}) + "\n",
            encoding="utf-8",
        )
        return {"matched_count": 1, "skipped_count": 0}

    monkeypatch.setattr(
        "local_impact_ranking.cli.enrich_papers_with_ncbi_citations",
        fake_enrich,
    )

    result = CliRunner().invoke(
        app,
        [
            "enrich-ncbi-citations",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--timeout",
            "13",
            "--resume",
        ],
    )

    assert result.exit_code == 0
    assert "NCBI citation-enriched" in result.output
