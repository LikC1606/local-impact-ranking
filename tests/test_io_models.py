import json

import pytest
from pydantic import ValidationError

from local_impact_ranking.io import convert_papers_csv_to_jsonl, load_papers_jsonl
from local_impact_ranking.models import Paper


def paper_record():
    return {
        "paper_id": "p1",
        "title": "Target Paper",
        "abstract": "A target abstract.",
        "year": 2024,
        "venue": "ICML",
        "field": "Machine Learning",
        "authors": [
            {
                "name": "Ada Lovelace",
                "affiliations": ["Analytical University"],
                "ids": {"orcid": "0000-0001"},
            }
        ],
        "references": [
            {
                "reference_id": "r1",
                "title": "Foundational Paper",
                "authors": [
                    {"name": "Grace Hopper", "affiliations": ["Compiler Institute"]}
                ],
                "year": 2020,
                "venue": "NeurIPS",
                "doi": "10.0000/example",
                "external_ids": {"openalex": "W123"},
                "raw_citation": "Grace Hopper. Foundational Paper. NeurIPS 2020.",
            }
        ],
        "citation_contexts": [
            {
                "reference_id": "r1",
                "sentence": "We build on Foundational Paper.",
                "paragraph": "We build on Foundational Paper for the core method.",
                "section": "Method",
            }
        ],
    }


def test_loads_valid_jsonl_records(tmp_path):
    path = tmp_path / "papers.jsonl"
    path.write_text(json.dumps(paper_record()) + "\n", encoding="utf-8")

    papers = load_papers_jsonl(path)

    assert len(papers) == 1
    assert papers[0].paper_id == "p1"
    assert papers[0].authors[0].affiliations == ["Analytical University"]
    assert papers[0].references[0].external_ids["openalex"] == "W123"
    assert papers[0].citation_contexts[0].reference_id == "r1"
    assert papers[0].citation_contexts[0].section == "Method"


@pytest.mark.parametrize("missing_key", ["paper_id", "title", "references"])
def test_paper_requires_core_fields(missing_key):
    record = paper_record()
    record.pop(missing_key)

    with pytest.raises(ValidationError):
        Paper.model_validate(record)


def test_jsonl_loader_reports_line_number_for_invalid_json(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(paper_record()) + "\n" + "{not json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_papers_jsonl(path)


def test_convert_papers_csv_to_jsonl_writes_valid_local_schema(tmp_path):
    source = tmp_path / "papers.csv"
    output = tmp_path / "papers.jsonl"
    source.write_text(
        "\n".join(
            [
                "paper_id,title,abstract,year,venue,field,authors_json,references_json,citation_contexts_json",
                '"p1","Target","An abstract","2024","ICML","Machine Learning","[{""name"":""Ada"",""affiliations"":[""Analytical University""]}]","[{""reference_id"":""r1"",""title"":""Reference"",""year"":2020,""field"":""AI"",""authors"":[{""name"":""Grace""}]}]","[{""reference_id"":""r1"",""sentence"":""We build on it."",""section"":""Method""}]"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    count = convert_papers_csv_to_jsonl(source, output)

    papers = load_papers_jsonl(output)
    assert count == 1
    assert papers[0].paper_id == "p1"
    assert papers[0].year == 2024
    assert papers[0].authors[0].name == "Ada"
    assert papers[0].references[0].reference_id == "r1"
    assert papers[0].references[0].field == "AI"
    assert papers[0].citation_contexts[0].section == "Method"


def test_convert_papers_csv_to_jsonl_reports_row_number_for_bad_nested_json(tmp_path):
    source = tmp_path / "papers.csv"
    output = tmp_path / "papers.jsonl"
    source.write_text(
        "paper_id,title,references_json\np1,Target,not-json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row 2"):
        convert_papers_csv_to_jsonl(source, output)
