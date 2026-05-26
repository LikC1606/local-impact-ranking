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


def test_scoring_method_doc_covers_versions_weights_and_limits():
    text = Path("docs/scoring-method.md").read_text(encoding="utf-8")

    assert "score_v1_rank_weighted" in text
    assert "score_v2_context_weighted" in text
    assert "rank_weighted" in text
    assert "context_weighted" in text
    assert "flat" in text
    assert "1.0, 0.8, 0.6, 0.4, 0.2" in text
    assert "1.5" in text
    assert "dataset-local" in text


def test_readme_links_to_scoring_method_doc():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "[docs/scoring-method.md](docs/scoring-method.md)" in text
