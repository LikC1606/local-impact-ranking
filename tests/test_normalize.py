from local_impact_ranking.models import Paper
from local_impact_ranking.normalize import (
    normalize_doi,
    normalize_paper,
    normalize_text,
    stable_reference_id,
)


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  A   paper\n title\t") == "A paper title"


def test_normalize_doi_removes_case_insensitive_prefixes_and_trailing_punctuation():
    assert normalize_doi(" DOI:10.1000/ABC. ") == "10.1000/ABC"
    assert normalize_doi("HTTPS://DX.DOI.ORG/10.2000/Mixed;") == "10.2000/Mixed"
    assert normalize_doi("http://doi.org/10.3000/example)") == "10.3000/example"


def test_stable_reference_id_uses_title_year_and_authors():
    reference = {
        "title": " Foundational   Paper ",
        "year": 2020,
        "authors": [{"name": "Grace Hopper"}],
    }

    assert stable_reference_id(reference).startswith("ref-")
    assert stable_reference_id(reference) == stable_reference_id(reference)


def test_normalize_paper_cleans_text_and_fills_reference_ids():
    paper = Paper.model_validate(
        {
            "paper_id": " p1 ",
            "title": " Target   Paper ",
            "authors": [{"name": " Ada   Lovelace ", "affiliations": [" Analytical  U "]}],
            "references": [
                {
                    "title": " Foundational   Paper ",
                    "authors": [{"name": " Grace Hopper ", "affiliations": [" Compiler  Lab "]}],
                    "year": 2020,
                }
            ],
        }
    )

    normalized = normalize_paper(paper)

    assert normalized.paper_id == "p1"
    assert normalized.title == "Target Paper"
    assert normalized.authors[0].name == "Ada Lovelace"
    assert normalized.authors[0].affiliations == ["Analytical U"]
    assert normalized.references[0].reference_id.startswith("ref-")
    assert normalized.references[0].title == "Foundational Paper"


def test_normalize_paper_deduplicates_references_by_doi_and_remaps_contexts():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "references": [
                {
                    "reference_id": "r1",
                    "title": "Foundational Method",
                    "doi": "https://doi.org/10.1000/ABC.",
                },
                {
                    "reference_id": "r2",
                    "title": "Foundational Method",
                    "doi": "10.1000/abc",
                    "authors": [{"name": "Grace Hopper"}],
                    "external_ids": {"openalex": "W123"},
                },
            ],
            "citation_contexts": [
                {
                    "reference_id": " r2 ",
                    "sentence": "We reuse the method from this work.",
                }
            ],
        }
    )

    normalized = normalize_paper(paper)

    assert [reference.reference_id for reference in normalized.references] == ["r1"]
    assert normalized.references[0].authors[0].name == "Grace Hopper"
    assert normalized.references[0].external_ids == {"openalex": "W123"}
    assert normalized.citation_contexts[0].reference_id == "r1"
    assert normalized.metadata["reference_deduplication"] == {
        "original_reference_count": 2,
        "deduplicated_reference_count": 1,
        "removed_reference_count": 1,
        "duplicate_reference_id_map": {"r2": "r1"},
    }


def test_normalize_paper_deduplicates_references_by_title_year_and_authors():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "references": [
                {
                    "reference_id": "r1",
                    "title": "A Long Foundational Paper Title",
                    "year": 2020,
                    "authors": [{"name": "Grace Hopper"}],
                },
                {
                    "reference_id": "r2",
                    "title": " A   Long Foundational Paper Title ",
                    "year": 2020,
                    "authors": [{"name": "Grace   Hopper"}],
                },
            ],
        }
    )

    normalized = normalize_paper(paper)

    assert [reference.reference_id for reference in normalized.references] == ["r1"]
    assert normalized.metadata["reference_deduplication"]["duplicate_reference_id_map"] == {
        "r2": "r1"
    }


def test_normalize_paper_preserves_same_title_references_from_different_years():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "references": [
                {
                    "reference_id": "r1",
                    "title": "A Long Foundational Paper Title",
                    "year": 2020,
                    "authors": [{"name": "Grace Hopper"}],
                },
                {
                    "reference_id": "r2",
                    "title": "A Long Foundational Paper Title",
                    "year": 2021,
                    "authors": [{"name": "Grace Hopper"}],
                },
            ],
        }
    )

    normalized = normalize_paper(paper)

    assert [reference.reference_id for reference in normalized.references] == ["r1", "r2"]
    assert "reference_deduplication" not in normalized.metadata


def test_normalize_paper_records_audit_when_duplicate_ids_do_not_need_remapping():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "references": [
                {"reference_id": "r1", "title": "Reference"},
                {"reference_id": "r1", "title": "Reference", "year": 2020},
            ],
        }
    )

    normalized = normalize_paper(paper)

    assert [reference.reference_id for reference in normalized.references] == ["r1"]
    assert normalized.references[0].year == 2020
    assert normalized.metadata["reference_deduplication"] == {
        "original_reference_count": 2,
        "deduplicated_reference_count": 1,
        "removed_reference_count": 1,
        "duplicate_reference_id_map": {},
    }


def test_normalize_paper_preserves_duplicate_reference_source_evidence():
    paper = Paper.model_validate(
        {
            "paper_id": "p1",
            "title": "Target",
            "references": [
                {
                    "reference_id": "target",
                    "title": "Target Reference",
                    "raw_citation": "Input reference list edge",
                    "internal_citation_match_method": "input_reference",
                },
                {
                    "reference_id": "target",
                    "title": "Target Reference",
                    "raw_citation": "OpenCitations internal DOI edge",
                    "internal_citation_match_method": "opencitations_doi",
                    "internal_citation_match_confidence": 1.0,
                },
            ],
        }
    )

    normalized = normalize_paper(paper)
    reference = normalized.references[0]
    extra = reference.model_extra or {}

    assert reference.raw_citation == "Input reference list edge"
    assert extra["deduplicated_reference_evidence"] == [
        {
            "reference_id": "target",
            "raw_citation": "Input reference list edge",
            "internal_citation_match_method": "input_reference",
        },
        {
            "reference_id": "target",
            "raw_citation": "OpenCitations internal DOI edge",
            "internal_citation_match_method": "opencitations_doi",
            "internal_citation_match_confidence": 1.0,
        },
    ]
