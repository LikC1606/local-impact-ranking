import json

from local_impact_ranking.semantic_scholar_enrichment import (
    enrich_papers_with_semantic_scholar,
)


class SemanticScholarClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, params=None, headers=None):
        self.requests.append(
            {"method": "GET", "url": url, "params": params or {}, "headers": headers or {}}
        )
        payload = self.payloads.get(url)
        return SemanticScholarResponse(payload)

    def post(self, url, json=None, params=None, headers=None):
        self.requests.append(
            {
                "method": "POST",
                "url": url,
                "json": json or {},
                "params": params or {},
                "headers": headers or {},
            }
        )
        payload = self.payloads.get(url)
        return SemanticScholarResponse(payload)

    def close(self):
        self.closed = True


class SemanticScholarResponse:
    def __init__(self, payload):
        self.payload = payload
        if isinstance(payload, dict) and "__status_code__" in payload:
            self.status_code = payload["__status_code__"]
        else:
            self.status_code = 404 if payload is None else 200

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_enrich_papers_with_semantic_scholar_by_doi_adds_influential_evidence(
    tmp_path,
):
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
    client = SemanticScholarClient(
        {
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.123%2Fexample": {
                "paperId": "S2PID",
                "corpusId": 12345,
                "externalIds": {"DOI": "10.123/example"},
                "title": "Paper",
                "year": 2024,
                "citationCount": 17,
                "influentialCitationCount": 3,
                "fieldsOfStudy": ["Medicine", "Biology"],
            }
        }
    )

    summary = enrich_papers_with_semantic_scholar(
        input_path,
        output_path,
        api_key="secret",
        client=client,
    )

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    metadata = record["metadata"]
    external_impact = metadata["external_impact"]
    assert summary["matched_count"] == 1
    assert summary["doi_query_count"] == 1
    assert client.requests[0]["headers"]["x-api-key"] == "secret"
    assert record["citation_count"] == 17.0
    assert record["year"] == 2024
    assert record["field"] == "Medicine"
    assert metadata["semantic_scholar_enriched"] is True
    assert metadata["external_ids"]["semantic_scholar"] == "S2PID"
    assert metadata["external_ids"]["semantic_scholar_corpus_id"] == "12345"
    assert external_impact["semantic_scholar_influential_citation_count"] == 3.0
    assert external_impact["semantic_scholar_citation_count"] == 17.0


def test_enrich_papers_with_semantic_scholar_uses_title_match_for_articles(tmp_path):
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
    client = SemanticScholarClient(
        {
            "https://api.semanticscholar.org/graph/v1/paper/search/match": {
                "paperId": "TITLEPID",
                "title": "Design principles of the cytotoxic CD8+ T cell response",
                "citationCount": 9,
                "influentialCitationCount": 2,
            }
        }
    )

    summary = enrich_papers_with_semantic_scholar(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["title_query_count"] == 1
    assert summary["title_match_count"] == 1
    assert client.requests[0]["url"].endswith("/paper/search/match")
    assert client.requests[0]["params"]["query"] == (
        "design principles of the cytotoxic cd8 t cell response"
    )
    assert record["metadata"]["external_ids"]["semantic_scholar"] == "TITLEPID"
    assert (
        record["metadata"]["external_impact"][
            "semantic_scholar_influential_citation_count"
        ]
        == 2.0
    )


def test_enrich_papers_with_semantic_scholar_batches_identifier_queries(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "One",
                        "doi": "10.123/one",
                        "references": [],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Two",
                        "metadata": {"external_ids": {"semantic_scholar": "S2TWO"}},
                        "references": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = SemanticScholarClient(
        {
            "https://api.semanticscholar.org/graph/v1/paper/batch": [
                {
                    "paperId": "S2ONE",
                    "title": "One",
                    "citationCount": 5,
                    "influentialCitationCount": 1,
                },
                None,
            ]
        }
    )

    summary = enrich_papers_with_semantic_scholar(input_path, output_path, client=client)

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["batch_query_count"] == 1
    assert summary["matched_count"] == 1
    assert summary["unmatched_count"] == 1
    assert len(client.requests) == 1
    assert client.requests[0]["method"] == "POST"
    assert client.requests[0]["json"] == {"ids": ["DOI:10.123/one", "S2TWO"]}
    assert records[0]["metadata"]["external_ids"]["semantic_scholar"] == "S2ONE"
    assert records[1]["paper_id"] == "p2"


def test_enrich_papers_with_semantic_scholar_skips_supplement_title_search(tmp_path):
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
    client = SemanticScholarClient({})

    summary = enrich_papers_with_semantic_scholar(input_path, output_path, client=client)

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary["queryable_count"] == 0
    assert summary["title_query_count"] == 0
    assert client.requests == []
    assert record["metadata"]["source_kind"] == "supplement_or_extracted_supplement"


def test_enrich_papers_with_semantic_scholar_stops_after_rate_limit(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "enriched.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "First",
                        "doi": "10.123/limited",
                        "references": [],
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Second",
                        "doi": "10.123/not-requested",
                        "references": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = SemanticScholarClient(
        {
            "https://api.semanticscholar.org/graph/v1/paper/batch": {"__status_code__": 429}
        }
    )

    summary = enrich_papers_with_semantic_scholar(input_path, output_path, client=client)

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["matched_count"] == 0
    assert summary["rate_limited_count"] == 1
    assert summary["rate_limit_stopped_count"] == 1
    assert summary["unmatched_count"] == 2
    assert len(client.requests) == 1
    assert client.requests[0]["method"] == "POST"
    assert records[0]["paper_id"] == "p1"
    assert records[1]["paper_id"] == "p2"
