from local_impact_ranking.models import Paper
from local_impact_ranking.normalize import normalize_paper, normalize_text, stable_reference_id


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  A   paper\n title\t") == "A paper title"


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
