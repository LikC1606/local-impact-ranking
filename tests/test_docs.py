from pathlib import Path


def test_input_schema_doc_covers_required_and_benchmark_fields():
    text = Path("docs/input-schema.md").read_text(encoding="utf-8")

    assert "paper_id" in text
    assert "references" in text
    assert "citation_contexts" in text
    assert "gold_reference_ids" in text
    assert "prescience_positive_reference_ids" in text
    assert "convert-csv" in text


def test_readme_links_to_input_schema_doc():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "[docs/input-schema.md](docs/input-schema.md)" in text
