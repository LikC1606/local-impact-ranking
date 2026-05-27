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
    assert "core_score" in text
    assert "expanded_score" in text
    assert "log2(rank + 1)" in text
    assert "normalized_score" in text
    assert "percentile" in text
    assert "dataset-local" in text


def test_readme_links_to_scoring_method_doc():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "[docs/scoring-method.md](docs/scoring-method.md)" in text


def test_evaluation_method_doc_covers_metrics_and_limits():
    text = Path("docs/evaluation-method.md").read_text(encoding="utf-8")

    assert "parse_success_rate" in text
    assert "invalid_selection_rate" in text
    assert "hit_rate" in text
    assert "selection_precision" in text
    assert "selection_recall" in text
    assert "mean_reciprocal_rank" in text
    assert "mean_ndcg" in text
    assert "gold_candidate_recall" in text
    assert "fixed_count" in text
    assert "broken_count" in text
    assert "judge_beats_all_baselines" in text
    assert "not global scientific impact" in text


def test_readme_links_to_evaluation_method_doc():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "[docs/evaluation-method.md](docs/evaluation-method.md)" in text


def test_benchmark_mapping_doc_covers_plan_command_and_public_sources():
    text = Path("docs/planning/benchmark-mapping.md").read_text(encoding="utf-8")

    assert "benchmarks plan" in text
    assert "https://huggingface.co/datasets/allenai/prescience" in text
    assert "pairwise_accuracy" in text
    assert "project-readiness" in text


def test_readme_documents_benchmark_plan_command():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "benchmarks plan" in text
    assert "ready_now" in text


def test_readme_documents_report_bundle_command():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "report-bundle" in text
    assert "bundle_manifest.json" in text


def test_readme_links_to_publish_checklist():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "[docs/publish-checklist.md](docs/publish-checklist.md)" in text


def test_readme_documents_doctor_command():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "doctor --markdown" in text
    assert "doctor.json" in text


def test_readme_documents_final_operational_commands():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "runs-index" in text
    assert "model-eval" in text
    assert "score-card" in text
    assert "entity-resolution-report" in text
    assert "split-prescience" in text


def test_publish_checklist_documents_github_release_boundaries():
    text = Path("docs/publish-checklist.md").read_text(encoding="utf-8")

    assert "data/" in text
    assert "models/" in text
    assert "runs/" in text
    assert "pytest -q" in text
    assert "report-bundle" in text
    assert "dataset-local" in text
