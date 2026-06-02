import json

from local_impact_ranking.impact_fusion import compute_impact_index_v2
from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.opencitations_enrichment import (
    enrich_papers_with_opencitations,
)


class OpenCitationsClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, headers=None):
        self.requests.append({"url": url, "headers": headers or {}})
        payload = self.payloads.get(url)
        return OpenCitationsResponse(payload)

    def close(self):
        self.closed = True


class OpenCitationsResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 404 if payload is None else 200

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_enrich_papers_with_opencitations_adds_counts_and_internal_edges(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Cited paper",
                        "doi": "10.1000/cited",
                        "references": [],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Citing paper",
                        "doi": "10.1000/citing",
                        "references": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenCitationsClient(
        {
            "https://api.opencitations.net/index/v2/citation-count/doi:10.1000%2Fcited": [
                {"count": "3"}
            ],
            "https://api.opencitations.net/index/v2/reference-count/doi:10.1000%2Fcited": [
                {"count": "12"}
            ],
            "https://api.opencitations.net/index/v2/citations/doi:10.1000%2Fcited": [
                {
                    "oci": "oci-1",
                    "citing": "doi:10.1000/citing openalex:W2",
                    "cited": "doi:10.1000/cited",
                    "creation": "2024-01-02",
                    "journal_sc": "no",
                    "author_sc": "yes",
                },
                {
                    "oci": "oci-2",
                    "citing": "doi:10.1000/external",
                    "cited": "doi:10.1000/cited",
                },
            ],
            "https://api.opencitations.net/index/v2/citation-count/doi:10.1000%2Fciting": [
                {"count": "1"}
            ],
            "https://api.opencitations.net/index/v2/reference-count/doi:10.1000%2Fciting": [
                {"count": "20"}
            ],
            "https://api.opencitations.net/index/v2/citations/doi:10.1000%2Fciting": [],
        }
    )

    summary = enrich_papers_with_opencitations(
        input_path,
        output_path,
        token="secret-token",
        client=client,
        include_edges=True,
    )

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {record["paper_id"]: record for record in records}
    p1_metadata = by_id["p1"]["metadata"]
    p2_metadata = by_id["p2"]["metadata"]

    assert summary["matched_count"] == 2
    assert summary["internal_incoming_edge_count"] == 1
    assert summary["external_incoming_edge_count"] == 1
    assert client.requests[0]["headers"]["authorization"] == "secret-token"
    assert by_id["p1"]["citation_count"] == 3.0
    assert p1_metadata["opencitations_enriched"] is True
    assert p1_metadata["external_impact"]["opencitations_cited_by_count"] == 3.0
    assert p1_metadata["external_impact"]["opencitations_reference_count"] == 12.0
    assert p1_metadata["external_ids"]["doi"] == "10.1000/cited"
    assert p1_metadata["opencitations_incoming_edges"] == [
        {
            "source_paper_id": "p2",
            "source_doi": "10.1000/citing",
            "target_paper_id": "p1",
            "target_doi": "10.1000/cited",
            "oci": "oci-1",
            "creation": "2024-01-02",
            "journal_self_citation": False,
            "author_self_citation": True,
        }
    ]
    assert p2_metadata["external_impact"]["opencitations_cited_by_count"] == 1.0


def test_enrich_papers_with_opencitations_preserves_unqueryable_rows(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    input_path.write_text(
        json.dumps({"paper_id": "p1", "title": "No DOI", "references": []}) + "\n",
        encoding="utf-8",
    )
    client = OpenCitationsClient({})

    summary = enrich_papers_with_opencitations(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["queryable_count"] == 0
    assert summary["unmatched_count"] == 1
    assert record["metadata"] == {}
    assert client.requests == []


def test_enrich_papers_with_opencitations_can_materialize_internal_graph_edges(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Cited paper",
                        "doi": "10.1000/cited",
                        "references": [],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Citing paper",
                        "doi": "10.1000/citing",
                        "references": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenCitationsClient(
        {
            "https://api.opencitations.net/index/v2/citation-count/doi:10.1000%2Fcited": [
                {"count": "3"}
            ],
            "https://api.opencitations.net/index/v2/reference-count/doi:10.1000%2Fcited": [
                {"count": "12"}
            ],
            "https://api.opencitations.net/index/v2/citations/doi:10.1000%2Fcited": [
                {
                    "oci": "oci-1",
                    "citing": "doi:10.1000/citing",
                    "cited": "doi:10.1000/cited",
                }
            ],
            "https://api.opencitations.net/index/v2/citation-count/doi:10.1000%2Fciting": [
                {"count": "1"}
            ],
            "https://api.opencitations.net/index/v2/reference-count/doi:10.1000%2Fciting": [
                {"count": "20"}
            ],
            "https://api.opencitations.net/index/v2/citations/doi:10.1000%2Fciting": [],
        }
    )

    enrich_papers_with_opencitations(
        input_path,
        output_path,
        client=client,
        materialize_internal_edges=True,
    )

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {record["paper_id"]: record for record in records}
    assert by_id["p2"]["references"] == [
        {
            "reference_id": "p1",
            "title": "Cited paper",
            "doi": "10.1000/cited",
            "external_ids": {"doi": "10.1000/cited"},
            "raw_citation": "OpenCitations internal DOI edge",
        }
    ]
    assert by_id["p2"]["metadata"]["opencitations_materialized_edge_count"] == 1


def test_materialized_internal_edges_feed_citation_graph_score(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Cited paper",
                        "doi": "10.1000/cited",
                        "references": [],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Citing paper",
                        "doi": "10.1000/citing",
                        "references": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenCitationsClient(
        {
            "https://api.opencitations.net/index/v2/citation-count/doi:10.1000%2Fcited": [
                {"count": "3"}
            ],
            "https://api.opencitations.net/index/v2/reference-count/doi:10.1000%2Fcited": [
                {"count": "12"}
            ],
            "https://api.opencitations.net/index/v2/citations/doi:10.1000%2Fcited": [
                {"citing": "doi:10.1000/citing", "cited": "doi:10.1000/cited"}
            ],
            "https://api.opencitations.net/index/v2/citation-count/doi:10.1000%2Fciting": [
                {"count": "1"}
            ],
            "https://api.opencitations.net/index/v2/reference-count/doi:10.1000%2Fciting": [
                {"count": "20"}
            ],
            "https://api.opencitations.net/index/v2/citations/doi:10.1000%2Fciting": [],
        }
    )
    enrich_papers_with_opencitations(
        input_path,
        output_path,
        client=client,
        materialize_internal_edges=True,
    )

    papers = load_papers_jsonl(output_path)
    result = compute_impact_index_v2(
        papers,
        [
            {"paper_id": "p1", "title": "Cited paper", "impact_index": 50, "comparisons": 0},
            {"paper_id": "p2", "title": "Citing paper", "impact_index": 50, "comparisons": 0},
        ],
    )

    by_id = {row["paper_id"]: row for row in result["rankings"]}
    assert result["summary"]["signal_availability"]["citation_graph"]["available"] is True
    assert by_id["p1"]["citation_graph_score"] > by_id["p2"]["citation_graph_score"]
