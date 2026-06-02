import json

from local_impact_ranking.openalex_enrichment import enrich_papers_with_openalex


class OpenAlexClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, params=None):
        self.requests.append({"url": url, "params": params or {}})
        payload = self.payloads.get(url)
        return OpenAlexResponse(payload)

    def close(self):
        self.closed = True


class OpenAlexResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 404 if payload is None else 200

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_enrich_papers_with_openalex_adds_citation_and_external_impact(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Paper",
                "doi": "10.123/example",
                "references": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenAlexClient(
        {
            "https://api.openalex.org/works/doi:10.123%2Fexample": {
                "id": "https://openalex.org/W123",
                "doi": "https://doi.org/10.123/example",
                "publication_year": 2024,
                "primary_topic": {
                    "field": {"display_name": "Medicine"},
                    "subfield": {"display_name": "Oncology"},
                },
                "cited_by_count": 42,
                "counts_by_year": [
                    {"year": 2024, "cited_by_count": 10},
                    {"year": 2025, "cited_by_count": 20},
                ],
                "citation_normalized_percentile": {"value": 0.93},
            }
        }
    )

    summary = enrich_papers_with_openalex(
        input_path,
        output_path,
        email="user@example.com",
        client=client,
    )

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    metadata = record["metadata"]
    assert summary["matched_count"] == 1
    assert summary["unmatched_count"] == 0
    assert client.requests[0]["params"]["mailto"] == "user@example.com"
    assert record["citation_count"] == 42.0
    assert record["year"] == 2024
    assert record["field"] == "Medicine"
    assert metadata["citation_counts_by_year"] == {"2024": 10.0, "2025": 20.0}
    assert metadata["external_impact"]["field_year_percentile"] == 0.93
    assert metadata["external_impact"]["openalex_cited_by_count"] == 42.0
    assert metadata["external_ids"]["openalex"] == "https://openalex.org/W123"
    assert metadata["openalex_enriched"] is True


def test_enrich_papers_with_openalex_uses_openalex_id_and_preserves_unmatched(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Paper",
                        "metadata": {"external_ids": {"openalex": "https://openalex.org/W999"}},
                        "references": [],
                    }
                ),
                json.dumps({"paper_id": "p2", "title": "No ID", "references": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenAlexClient(
        {
            "https://api.openalex.org/works/W999": {
                "id": "https://openalex.org/W999",
                "cited_by_count": 7,
                "counts_by_year": [],
                "citation_normalized_percentile": {"value": 87},
            }
        }
    )

    summary = enrich_papers_with_openalex(input_path, output_path, client=client)

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["queryable_count"] == 1
    assert summary["matched_count"] == 1
    assert summary["unmatched_count"] == 1
    assert records[0]["metadata"]["external_impact"]["field_year_percentile"] == 0.87
    assert records[1]["metadata"] == {}


def test_enrich_papers_with_openalex_falls_back_to_high_confidence_title_search(
    tmp_path,
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Design principles of the cytotoxic CD8+ T cell response",
                "references": [],
                "metadata": {"source_kind": "article_or_extracted_article"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenAlexClient(
        {
            "https://api.openalex.org/works": {
                "results": [
                    {
                        "id": "https://openalex.org/W777",
                        "doi": "https://doi.org/10.1000/title-match",
                        "display_name": (
                            "Design principles of the cytotoxic CD8+ T cell response"
                        ),
                        "cited_by_count": 11,
                        "counts_by_year": [{"year": 2025, "cited_by_count": 4}],
                        "citation_normalized_percentile": {"value": 0.81},
                    }
                ]
            }
        }
    )

    summary = enrich_papers_with_openalex(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["queryable_count"] == 1
    assert summary["matched_count"] == 1
    assert summary["title_query_count"] == 1
    assert summary["title_match_count"] == 1
    assert client.requests[0]["url"] == "https://api.openalex.org/works"
    assert client.requests[0]["params"]["filter"].startswith("title.search:")
    assert client.requests[0]["params"]["per-page"] == 5
    assert record["citation_count"] == 11.0
    assert record["metadata"]["external_ids"]["openalex"] == "https://openalex.org/W777"
    assert record["metadata"]["external_impact"]["field_year_percentile"] == 0.81


def test_enrich_papers_with_openalex_rejects_low_similarity_title_match(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Specific rare paper title",
                "references": [],
                "metadata": {"source_kind": "article_or_extracted_article"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenAlexClient(
        {
            "https://api.openalex.org/works": {
                "results": [
                    {
                        "id": "https://openalex.org/W888",
                        "display_name": "A different article title",
                        "cited_by_count": 99,
                    }
                ]
            }
        }
    )

    summary = enrich_papers_with_openalex(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["matched_count"] == 0
    assert summary["title_query_count"] == 1
    assert summary["title_match_count"] == 0
    assert record["metadata"]["source_kind"] == "article_or_extracted_article"
    assert "citation_count" not in record


def test_enrich_papers_with_openalex_skips_title_search_for_supplement_records(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Supplementary Online Content",
                "references": [],
                "metadata": {"source_kind": "supplement_or_extracted_supplement"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenAlexClient({})

    summary = enrich_papers_with_openalex(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["queryable_count"] == 0
    assert summary["title_query_count"] == 0
    assert client.requests == []
    assert record["metadata"]["source_kind"] == "supplement_or_extracted_supplement"


def test_enrich_papers_with_openalex_sanitizes_title_search_query(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": (
                    "On the Use of the Jackknife to Compare "
                    "$\\mathbf { R } _ { \\mathbf { X Y } }$ Statistics"
                ),
                "references": [],
                "metadata": {"source_kind": "article_or_extracted_article"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = OpenAlexClient({"https://api.openalex.org/works": {"results": []}})

    enrich_papers_with_openalex(input_path, output_path, client=client)

    query = client.requests[0]["params"]["filter"]
    assert query == (
        "title.search:on the use of the jackknife to compare "
        "mathbf r mathbf x y statistics"
    )
    assert "$" not in query
    assert "\\" not in query
