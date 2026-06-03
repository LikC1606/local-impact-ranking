import json

import local_impact_ranking.internal_citation_edges as internal_edges
from local_impact_ranking.internal_citation_edges import (
    materialize_internal_citation_edges,
)
from local_impact_ranking.io import load_papers_jsonl


def _write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


class OpenAlexReferenceClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []
        self.closed = False

    def get(self, url, params=None):
        self.requests.append({"url": url, "params": params or {}})
        return OpenAlexReferenceResponse(self.payloads.get(url))

    def close(self):
        self.closed = True


class OpenAlexReferenceResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 404 if payload is None else 200

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_materialize_internal_citation_edges_matches_references_by_doi(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.internal.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(
        input_path,
        [
            {
                "paper_id": "p1",
                "title": "A target paper",
                "doi": "10.1000/TARGET",
                "references": [],
            },
            {
                "paper_id": "p2",
                "title": "A citing paper",
                "references": [
                    {
                        "title": "Citation as originally extracted",
                        "doi": "https://doi.org/10.1000/target",
                    }
                ],
            },
        ],
    )

    summary = materialize_internal_citation_edges(
        input_path,
        output_path,
        summary_output=summary_path,
    )

    papers = {paper.paper_id: paper for paper in load_papers_jsonl(output_path)}
    reference = papers["p2"].references[0]
    assert reference.reference_id == "p1"
    assert reference.title == "Citation as originally extracted"
    assert summary["materialized_internal_edge_count"] == 1
    assert summary["doi_match_count"] == 1
    assert summary["title_match_count"] == 0
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    assert papers["p2"].metadata["internal_citation_matching_attempted"] is True
    assert papers["p2"].metadata["internal_citation_materialized_edge_count"] == 1
    assert papers["p2"].metadata["internal_citation_edges"][0]["match_method"] == "doi"


def test_materialize_internal_citation_edges_matches_references_by_strict_title(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.internal.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "paper_id": "p1",
                "title": "Long term effects of a transparent impact ranking system",
                "references": [],
            },
            {
                "paper_id": "p2",
                "title": "A citing paper",
                "references": [
                    {
                        "title": "Long-term effects of a transparent impact ranking system"
                    }
                ],
            },
        ],
    )

    summary = materialize_internal_citation_edges(
        input_path,
        output_path,
        title_threshold=0.92,
    )

    papers = {paper.paper_id: paper for paper in load_papers_jsonl(output_path)}
    assert papers["p2"].references[0].reference_id == "p1"
    assert summary["materialized_internal_edge_count"] == 1
    assert summary["title_match_count"] == 1


def test_materialize_internal_citation_edges_uses_index_for_exact_title_matches(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.internal.jsonl"
    records = [
        {
            "paper_id": "target",
            "title": "High resolution mapping of immune cell lineage plasticity in tumors",
            "references": [],
        },
        {
            "paper_id": "citing",
            "title": "A citing paper",
            "references": [
                {
                    "title": (
                        "High-resolution mapping of immune cell lineage plasticity "
                        "in tumors"
                    )
                }
            ],
        },
    ]
    for index in range(300):
        records.append(
            {
                "paper_id": f"decoy-{index}",
                "title": f"Unrelated decoy manuscript with synthetic marker {index}",
                "references": [],
            }
        )
    _write_jsonl(input_path, records)

    call_count = 0
    original_similarity = internal_edges._title_similarity

    def counting_similarity(left, right):
        nonlocal call_count
        call_count += 1
        return original_similarity(left, right)

    monkeypatch.setattr(internal_edges, "_title_similarity", counting_similarity)

    summary = materialize_internal_citation_edges(input_path, output_path)

    papers = {paper.paper_id: paper for paper in load_papers_jsonl(output_path)}
    assert papers["citing"].references[0].reference_id == "target"
    assert summary["materialized_internal_edge_count"] == 1
    assert call_count <= 2


def test_materialize_internal_citation_edges_blocks_fuzzy_title_candidates(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.internal.jsonl"
    records = [
        {
            "paper_id": "target",
            "title": (
                "Single cell atlas of inflammatory fibroblast activation "
                "in pancreatic cancer"
            ),
            "references": [],
        },
        {
            "paper_id": "citing",
            "title": "A citing paper",
            "references": [
                {
                    "title": (
                        "Single cell atlas of inflammatory fibroblast activation "
                        "in pancreas cancer"
                    )
                }
            ],
        },
    ]
    for index in range(300):
        records.append(
            {
                "paper_id": f"decoy-{index}",
                "title": f"Background report for unrelated engineering benchmark {index}",
                "references": [],
            }
        )
    _write_jsonl(input_path, records)

    call_count = 0
    original_similarity = internal_edges._title_similarity

    def counting_similarity(left, right):
        nonlocal call_count
        call_count += 1
        return original_similarity(left, right)

    monkeypatch.setattr(internal_edges, "_title_similarity", counting_similarity)

    summary = materialize_internal_citation_edges(
        input_path,
        output_path,
        title_threshold=0.9,
    )

    papers = {paper.paper_id: paper for paper in load_papers_jsonl(output_path)}
    assert papers["citing"].references[0].reference_id == "target"
    assert summary["materialized_internal_edge_count"] == 1
    assert call_count <= 16


def test_materialize_internal_citation_edges_uses_phrase_blocks_for_common_tokens(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.internal.jsonl"
    records = [
        {
            "paper_id": "target",
            "title": (
                "Single cell atlas of inflammatory fibroblast activation "
                "in pancreatic cancer"
            ),
            "references": [],
        },
        {
            "paper_id": "citing",
            "title": "A citing paper",
            "references": [
                {
                    "title": (
                        "Single cell atlas of inflammatory fibroblast activation "
                        "in pancreas cancer"
                    )
                }
            ],
        },
    ]
    for index in range(300):
        records.append(
            {
                "paper_id": f"decoy-{index}",
                "title": (
                    "Inflammatory cancer atlas single fibroblast activation "
                    f"cell benchmark {index}"
                ),
                "references": [],
            }
        )
    _write_jsonl(input_path, records)

    call_count = 0
    original_similarity = internal_edges._title_similarity

    def counting_similarity(left, right):
        nonlocal call_count
        call_count += 1
        return original_similarity(left, right)

    monkeypatch.setattr(internal_edges, "_title_similarity", counting_similarity)

    summary = materialize_internal_citation_edges(
        input_path,
        output_path,
        title_threshold=0.9,
    )

    papers = {paper.paper_id: paper for paper in load_papers_jsonl(output_path)}
    assert papers["citing"].references[0].reference_id == "target"
    assert summary["materialized_internal_edge_count"] == 1
    assert call_count <= 16


def test_materialize_internal_citation_edges_skips_overloaded_phrase_blocks(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.internal.jsonl"
    records = [
        {
            "paper_id": "target",
            "title": (
                "Single cell atlas of inflammatory fibroblast activation "
                "in pancreatic cancer raremarker"
            ),
            "references": [],
        },
        {
            "paper_id": "citing",
            "title": "A citing paper",
            "references": [
                {
                    "title": (
                        "Single cell atlas of inflammatory fibroblast activation "
                        "in pancreas cancer raremarker"
                    )
                }
            ],
        },
    ]
    for index in range(300):
        records.append(
            {
                "paper_id": f"decoy-{index}",
                "title": (
                    "Single cell atlas of inflammatory fibroblast activation "
                    f"in unrelated cancer decoymarker{index}"
                ),
                "references": [],
            }
        )
    _write_jsonl(input_path, records)

    call_count = 0
    original_similarity = internal_edges._title_similarity

    def counting_similarity(left, right):
        nonlocal call_count
        call_count += 1
        return original_similarity(left, right)

    monkeypatch.setattr(internal_edges, "_title_similarity", counting_similarity)

    summary = materialize_internal_citation_edges(
        input_path,
        output_path,
        title_threshold=0.9,
    )

    papers = {paper.paper_id: paper for paper in load_papers_jsonl(output_path)}
    assert papers["citing"].references[0].reference_id == "target"
    assert summary["materialized_internal_edge_count"] == 1
    assert call_count <= 16


def test_materialize_internal_citation_edges_rejects_weak_title_matches(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.internal.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "paper_id": "p1",
                "title": "Deep learning for citation impact prediction",
                "references": [],
            },
            {
                "paper_id": "p2",
                "title": "A citing paper",
                "references": [{"title": "Clinical trial protocol and appendix"}],
            },
        ],
    )

    summary = materialize_internal_citation_edges(input_path, output_path)

    papers = {paper.paper_id: paper for paper in load_papers_jsonl(output_path)}
    assert papers["p2"].references[0].reference_id is None
    assert summary["materialized_internal_edge_count"] == 0
    assert summary["unmatched_reference_count"] == 1


def test_materialize_internal_citation_edges_skips_self_citations(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.internal.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "paper_id": "p1",
                "title": "Self target",
                "doi": "10.1000/self",
                "references": [{"title": "Self target", "doi": "10.1000/self"}],
            }
        ],
    )

    summary = materialize_internal_citation_edges(input_path, output_path)

    papers = load_papers_jsonl(output_path)
    assert papers[0].references[0].reference_id is None
    assert summary["self_reference_skipped_count"] == 1
    assert summary["materialized_internal_edge_count"] == 0


def test_materialize_internal_citation_edges_preserves_existing_internal_edges(tmp_path):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.internal.jsonl"
    _write_jsonl(
        input_path,
        [
            {"paper_id": "p1", "title": "Target", "references": []},
            {
                "paper_id": "p2",
                "title": "Citing",
                "references": [{"reference_id": "p1", "title": "Target"}],
            },
        ],
    )

    summary = materialize_internal_citation_edges(input_path, output_path)

    papers = {paper.paper_id: paper for paper in load_papers_jsonl(output_path)}
    assert papers["p2"].references[0].reference_id == "p1"
    assert summary["existing_internal_edge_count"] == 1
    assert summary["materialized_internal_edge_count"] == 0
    assert papers["p2"].metadata["internal_citation_edge_count"] == 1


def test_materialize_internal_citation_edges_can_use_openalex_referenced_works(
    tmp_path,
):
    input_path = tmp_path / "papers.jsonl"
    output_path = tmp_path / "papers.internal.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "paper_id": "p1",
                "title": "Dataset target",
                "doi": "10.1000/target",
                "references": [],
                "metadata": {"external_ids": {"openalex": "https://openalex.org/W1"}},
            },
            {
                "paper_id": "p2",
                "title": "Dataset citing paper",
                "references": [],
                "metadata": {"external_ids": {"openalex": "https://openalex.org/W2"}},
            },
        ],
    )
    client = OpenAlexReferenceClient(
        {
            "https://api.openalex.org/works/W1": {
                "id": "https://openalex.org/W1",
                "referenced_works": [],
            },
            "https://api.openalex.org/works/W2": {
                "id": "https://openalex.org/W2",
                "referenced_works": ["https://openalex.org/W1"],
            },
        }
    )

    summary = materialize_internal_citation_edges(
        input_path,
        output_path,
        openalex_references=True,
        openalex_client=client,
    )

    papers = {paper.paper_id: paper for paper in load_papers_jsonl(output_path)}
    reference = papers["p2"].references[0]
    assert reference.reference_id == "p1"
    assert reference.title == "Dataset target"
    assert reference.raw_citation == "OpenAlex referenced_works internal edge"
    assert summary["openalex_reference_query_count"] == 2
    assert summary["openalex_reference_match_count"] == 1
    assert summary["materialized_internal_edge_count"] == 1
    assert (
        papers["p2"]
        .metadata["internal_citation_edges"][0]["match_method"]
        == "openalex_referenced_work"
    )
