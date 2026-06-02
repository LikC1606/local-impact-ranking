import json
from zipfile import ZipFile

from local_impact_ranking.io import load_papers_jsonl
from local_impact_ranking.mineru_zip_adapter import convert_mineru_zip_to_jsonl


def test_convert_mineru_zip_extracts_numbered_reference_tail_without_heading(tmp_path):
    source = tmp_path / "papers.zip"
    output = tmp_path / "papers.jsonl"
    markdown = """# Design principles of the cytotoxic CD8+ T cell response

Obinna Example and Ada Example

Edited by Example Editor; received November 8, 2025; accepted April 10, 2026.

Cytotoxic T lymphocytes eliminate infected or malignant cells.

89. A. Author, B. Writer, Foundational discovery of cytotoxic response. Nature 12, 1-2 (2020). https://doi.org/10.1000/foundational.
90. C. Researcher et al., A variational deep-learning approach to modeling immune dynamics. Science 8, 20-30 (2021).
"""
    with ZipFile(source, "w") as archive:
        archive.writestr("p1/auto/p1.md", markdown)

    summary = convert_mineru_zip_to_jsonl(source, output)

    papers = load_papers_jsonl(output)
    assert summary["paper_count"] == 1
    assert summary["with_references_count"] == 1
    assert summary["without_references_count"] == 0
    assert papers[0].paper_id == "p1"
    assert papers[0].title == "Design principles of the cytotoxic CD8+ T cell response"
    assert papers[0].year == 2025
    assert [reference.reference_id for reference in papers[0].references] == [
        "p1_ref_001",
        "p1_ref_002",
    ]
    assert papers[0].references[0].doi == "10.1000/foundational"
    assert papers[0].references[0].title == "Foundational discovery of cytotoxic response"


def test_convert_mineru_zip_marks_supplement_without_references(tmp_path):
    source = tmp_path / "papers.zip"
    output = tmp_path / "papers.jsonl"
    markdown = """# Supplementary Material

Figure S1. Supporting experiment.
"""
    with ZipFile(source, "w") as archive:
        archive.writestr("p2/auto/p2.md", markdown)

    summary = convert_mineru_zip_to_jsonl(source, output)

    record = json.loads(output.read_text(encoding="utf-8"))
    assert summary["paper_count"] == 1
    assert summary["with_references_count"] == 0
    assert summary["without_references_count"] == 1
    assert record["references"] == []
    assert record["metadata"]["source_kind"] == "supplement_or_extracted_supplement"
    assert record["metadata"]["extraction_status"] == "no_references_extracted"


def test_convert_mineru_zip_does_not_mark_article_with_supplement_section_as_supplement(
    tmp_path,
):
    source = tmp_path / "papers.zip"
    output = tmp_path / "papers.jsonl"
    markdown = """# Live and heat-treated Lactiplantibacillus plantarum induce distinct responses

This is the target article abstract.

## Supplemental Information

Additional experimental details are available online.

## References

1. Author, A. Article citation. Journal 1, 1-2 (2022).
"""
    with ZipFile(source, "w") as archive:
        archive.writestr("p2b/auto/p2b.md", markdown)

    convert_mineru_zip_to_jsonl(source, output)

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["metadata"]["source_kind"] == "article_or_extracted_article"


def test_convert_mineru_zip_marks_figure_caption_record_as_supplement(tmp_path):
    source = tmp_path / "papers.zip"
    output = tmp_path / "papers.jsonl"
    markdown = """# FigS2: Flow cytometry dot plots of treated cells

Panel A. Supporting figure only.
"""
    with ZipFile(source, "w") as archive:
        archive.writestr("p2c/auto/p2c.md", markdown)

    convert_mineru_zip_to_jsonl(source, output)

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["metadata"]["source_kind"] == "supplement_or_extracted_supplement"


def test_convert_mineru_zip_extracts_heading_references_with_parenthesized_numbers(
    tmp_path,
):
    source = tmp_path / "papers.zip"
    output = tmp_path / "papers.jsonl"
    markdown = """# Hybrid Fabrication of an Electrochemical Electrode

An abstract for the target paper.

## REFERENCES

(1) Min, J.; Hu, J. Fabrication processes of metal-fiber reinforced polymer hybrid components: a review. Adv. Compos. Hybrid Mater. 2022, 5, 651-678.

(2) Singh, M.; Haverinen, H. M. Inkjet Printing Process and Its Applications. Adv. Mater. 2010, 22, 673-685. DOI: 10.1002/adma.200901141.
"""
    with ZipFile(source, "w") as archive:
        archive.writestr("p3/auto/p3.md", markdown)

    summary = convert_mineru_zip_to_jsonl(source, output)

    papers = load_papers_jsonl(output)
    assert summary["with_references_count"] == 1
    assert len(papers[0].references) == 2
    assert papers[0].references[0].title == (
        "Fabrication processes of metal-fiber reinforced polymer hybrid components: "
        "a review"
    )
    assert papers[0].references[1].doi == "10.1002/adma.200901141"


def test_convert_mineru_zip_does_not_use_reference_doi_as_paper_doi(tmp_path):
    source = tmp_path / "papers.zip"
    output = tmp_path / "papers.jsonl"
    markdown = """# Target Without DOI

This paper itself has no DOI in the front matter.

## References

Author, A. (2022). Reference with DOI. Journal 1, 1-2. doi:10.1000/reference-only
"""
    with ZipFile(source, "w") as archive:
        archive.writestr("p4/auto/p4.md", markdown)

    convert_mineru_zip_to_jsonl(source, output)

    record = json.loads(output.read_text(encoding="utf-8"))
    assert "doi" not in record
    assert record["metadata"].get("external_ids") is None
    assert record["references"][0]["doi"] == "10.1000/reference-only"
