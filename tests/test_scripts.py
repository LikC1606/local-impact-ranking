from pathlib import Path


def test_run_pilot_script_documents_required_arguments_and_defaults():
    script = Path("scripts/run_pilot.sh").read_text(encoding="utf-8")

    assert 'INPUT="${1:?Usage:' in script
    assert 'OUTPUT="${2:-runs/pilot-100}"' in script
    assert 'LIMIT="${LIMIT:-100}"' in script
    assert 'TOP_K="${TOP_K:-5}"' in script
    assert 'CONCURRENCY="${CONCURRENCY:-1}"' in script
    assert 'BASE_URL="${BASE_URL:-http://localhost:8000/v1}"' in script
    assert 'MODEL="${MODEL:-local-impact-judge}"' in script
    assert 'MAX_TOKENS="${MAX_TOKENS:-512}"' in script


def test_run_pilot_script_checks_server_judges_then_scores_from_audit():
    script = Path("scripts/run_pilot.sh").read_text(encoding="utf-8")

    assert 'scripts/check_model_server.sh "${BASE_URL}"' in script
    assert "--limit" in script
    assert '"${LIMIT}"' in script
    assert "--judge-audit" in script
    assert '--judge-audit "${AUDIT_PATH}" \\\n  --limit "${LIMIT}"' in script
    assert "--strict-validation" in script
    assert "--max-tokens" in script
    assert '"${AUDIT_PATH}"' in script
    assert 'test -s "${OUTPUT}/audit/prompts.jsonl"' in script
    assert 'test -s "${OUTPUT}/audit/scoring_edges.csv"' in script
    assert 'test -s "${OUTPUT}/rankings/fields.csv"' in script
    assert 'test -s "${OUTPUT}/rankings/years.csv"' in script
    assert 'test -s "${OUTPUT}/reports/validation.json"' in script
    assert 'test -s "${OUTPUT}/reports/validation.md"' in script
    assert 'test -s "${OUTPUT}/reports/artifact_index.json"' in script
    assert 'test -s "${OUTPUT}/reports/evaluation.json"' in script
    assert 'test -s "${OUTPUT}/reports/evaluation.md"' in script
    assert 'test -s "${OUTPUT}/reports/gold_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/reports/proxy_evaluation.json"' in script
    assert 'test -f "${OUTPUT}/run_manifest.json"' in script
    assert 'test -s "${OUTPUT}/reports/paper_explanations.json"' in script
    assert 'test -s "${OUTPUT}/reports/entity_explanations.json"' in script
    assert 'test -s "${OUTPUT}/reports/paper_stability_report.json"' in script
    assert 'test -s "${OUTPUT}/reports/paper_stability_report.csv"' in script
    assert 'test -s "${OUTPUT}/reports/paper_score_uncertainty_report.json"' in script
    assert 'test -s "${OUTPUT}/reports/paper_score_uncertainty_report.csv"' in script
    assert 'test -s "${OUTPUT}/reports/paper_evidence_report.json"' in script
    assert 'test -s "${OUTPUT}/reports/paper_evidence_report.csv"' in script
    assert 'test -s "${OUTPUT}/reports/score_cards.md"' in script
    assert 'test -s "${OUTPUT}/reports/entity_resolution_report.md"' in script
    assert 'test -s "${OUTPUT}/reports/mvp_summary.json"' in script
    assert 'test -s "${OUTPUT}/reports/mvp_summary.md"' in script
    assert 'test -s "${OUTPUT}/finetuning/finetuning_readiness.json"' in script
    assert 'test -s "${OUTPUT}/finetuning/finetuning_readiness.md"' in script
    assert 'test -s "${OUTPUT}/finetuning/finetuning_preflight.json"' in script
    assert 'test -s "${OUTPUT}/finetuning/finetuning_preflight.md"' in script
    assert 'test -f "${OUTPUT}/finetuning/context_hard_negative_dpo.jsonl"' in script
    assert 'test -f "${OUTPUT}/finetuning/method_context_contrastive_dpo.jsonl"' in script
    assert 'test -f "${OUTPUT}/finetuning/merged_dpo_preferences.jsonl"' in script
    assert 'report-bundle "${OUTPUT}" --output "${OUTPUT}/report_bundle"' in script
    assert 'test -s "${OUTPUT}/report_bundle/README.md"' in script
    assert 'test -s "${OUTPUT}/report_bundle/bundle_manifest.json"' in script


def test_run_local_smoke_script_runs_heuristic_examples_and_checks_context_weight():
    script = Path("scripts/run_local_smoke.sh").read_text(encoding="utf-8")

    assert 'OUTPUT="${1:-runs/local-smoke}"' in script
    assert 'examples/tiny_papers.jsonl' in script
    assert 'examples/context_papers.jsonl' in script
    assert "--strict-validation" in script
    assert "--credit-mode context_weighted" in script
    assert 'test -s "${OUTPUT}/context/audit/prompts.jsonl"' in script
    assert 'test -s "${OUTPUT}/context/audit/scoring_edges.csv"' in script
    assert 'test -s "${OUTPUT}/context/rankings/fields.csv"' in script
    assert 'test -s "${OUTPUT}/context/rankings/years.csv"' in script
    assert 'test -s "${OUTPUT}/context/reports/evaluation.md"' in script
    assert 'test -s "${OUTPUT}/context/reports/gold_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/context/reports/paper_stability_report.csv"' in script
    assert 'test -s "${OUTPUT}/context/reports/paper_score_uncertainty_report.csv"' in script
    assert 'test -s "${OUTPUT}/context/reports/mvp_summary.md"' in script
    assert 'test -s "${OUTPUT}/context/reports/project_readiness.json"' in script
    assert 'test -s "${OUTPUT}/context/reports/project_readiness.md"' in script
    assert 'test -s "${OUTPUT}/context/reports/paper_evidence_report.csv"' in script
    assert 'test -s "${OUTPUT}/context/reports/score_cards.md"' in script
    assert 'test -s "${OUTPUT}/context/reports/entity_resolution_report.md"' in script
    assert 'test -s "${OUTPUT}/context/finetuning/finetuning_readiness.md"' in script
    assert 'test -s "${OUTPUT}/context/finetuning/finetuning_preflight.md"' in script
    assert 'test -f "${OUTPUT}/context/finetuning/context_hard_negative_dpo.jsonl"' in script
    assert 'test -f "${OUTPUT}/context/finetuning/method_context_contrastive_dpo.jsonl"' in script
    assert 'test -f "${OUTPUT}/context/finetuning/merged_dpo_preferences.jsonl"' in script
    assert 'test -s "${OUTPUT}/context/reports/artifact_index.json"' in script
    assert 'report-bundle "${OUTPUT}/context" --output "${OUTPUT}/context/report_bundle"' in script
    assert 'test -s "${OUTPUT}/context/report_bundle/README.md"' in script
    assert 'test -s "${OUTPUT}/context/report_bundle/bundle_manifest.json"' in script
    assert 'grep -q "1,r1,1.500000" "${OUTPUT}/context/rankings/papers.csv"' in script


def test_run_quickstart_script_runs_no_gpu_end_to_end_demo():
    script = Path("scripts/run_quickstart.sh").read_text(encoding="utf-8")

    assert 'OUTPUT="${1:-runs/quickstart}"' in script
    assert "doctor" in script
    assert '--output "${OUTPUT}/doctor"' in script
    assert "run" in script
    assert "--input examples/context_papers.jsonl" in script
    assert "score-card" in script
    assert "entity-resolution-report" in script
    assert "report-bundle" in script
    assert "runs-index" in script
    assert 'test -s "${OUTPUT}/run/reports/score_cards.md"' in script
    assert 'test -s "${OUTPUT}/run/reports/entity_resolution_report.md"' in script
    assert 'test -s "${OUTPUT}/score_cards/score_cards.md"' in script
    assert 'test -s "${OUTPUT}/entity_resolution/entity_resolution_report.md"' in script
    assert 'test -s "${OUTPUT}/run/report_bundle/bundle_manifest.json"' in script
    assert 'test -s "${OUTPUT}/runs_index/index.md"' in script


def test_github_actions_runs_publish_check_and_quickstart_smoke():
    workflow = Path(".github/workflows/tests.yml").read_text(encoding="utf-8")

    assert "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true" in workflow
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert "scripts/check_publish_ready.sh" in workflow
    assert "scripts/run_quickstart.sh" in workflow
    assert "runs/quickstart" in workflow


def test_compare_model_judges_script_runs_base_candidate_and_reports_deltas():
    script = Path("scripts/compare_model_judges.sh").read_text(encoding="utf-8")

    assert 'INPUT="${1:?Usage:' in script
    assert 'OUTPUT="${2:-runs/model-comparison}"' in script
    assert 'BASE_MODEL="${BASE_MODEL:-local-impact-judge}"' in script
    assert 'CANDIDATE_MODEL="${CANDIDATE_MODEL:-local-impact-judge-candidate}"' in script
    assert 'BASE_AUDIT_PATH="${OUTPUT}/base/audit/judge_outputs.jsonl"' in script
    assert 'CANDIDATE_AUDIT_PATH="${OUTPUT}/candidate/audit/judge_outputs.jsonl"' in script
    assert '--model "${BASE_MODEL}"' in script
    assert '--model "${CANDIDATE_MODEL}"' in script
    assert 'compare-benchmarks' in script
    assert 'compare-gold-targets' in script
    assert 'model-selection-decision' in script
    assert 'test -s "${OUTPUT}/comparison/benchmark_comparison.md"' in script
    assert 'test -s "${OUTPUT}/comparison/target_comparison.json"' in script
    assert 'test -s "${OUTPUT}/comparison/target_changes.csv"' in script
    assert 'test -s "${OUTPUT}/comparison/model_selection_decision.md"' in script


def test_run_benchmark_smoke_script_runs_limited_scicite_pipeline():
    script = Path("scripts/run_benchmark_smoke.sh").read_text(encoding="utf-8")

    assert 'INPUT="${INPUT:-data/scicite/dev.papers.jsonl}"' in script
    assert "--limit" in script
    assert '"${LIMIT}"' in script
    assert "--credit-mode context_weighted" in script
    assert 'test -s "${OUTPUT}/reports/mvp_summary.md"' in script
    assert 'test -s "${OUTPUT}/reports/project_readiness.json"' in script
    assert 'test -s "${OUTPUT}/reports/project_readiness.md"' in script
    assert 'test -s "${OUTPUT}/reports/proxy_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/reports/gold_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/rankings/papers.csv"' in script


def test_run_sciimpact_smoke_script_converts_and_evaluates_pairwise_fixture():
    script = Path("scripts/run_sciimpact_smoke.sh").read_text(encoding="utf-8")

    assert 'INPUT="${INPUT:-examples/sciimpact_pairs.jsonl}"' in script
    assert "benchmarks convert-sciimpact" in script
    assert "evaluate-pairwise-impact" in script
    assert 'test -s "${OUTPUT}/converted/papers.jsonl"' in script
    assert 'test -s "${OUTPUT}/converted/pairwise_labels.jsonl"' in script
    assert 'test -s "${OUTPUT}/rankings/papers.csv"' in script
    assert 'test -s "${OUTPUT}/pairwise_eval/pairwise_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/pairwise_eval/pairwise_evaluation.md"' in script


def test_run_sciimpact_sft_smoke_script_converts_official_sft_sample():
    script = Path("scripts/run_sciimpact_sft_smoke.sh").read_text(encoding="utf-8")

    assert 'INPUT="${INPUT:-data/sciimpact/source/sft_data/paper_impact_sft_test_clean.jsonl}"' in script
    assert 'LIMIT="${LIMIT:-200}"' in script
    assert 'sed -n "1,${LIMIT}p" "${INPUT}" > "${OUTPUT}/input.jsonl"' in script
    assert "benchmarks convert-sciimpact" in script
    assert 'test -s "${OUTPUT}/converted/papers.jsonl"' in script
    assert 'test -s "${OUTPUT}/converted/pairwise_labels.jsonl"' in script
    assert 'test -s "${OUTPUT}/converted/manifest.json"' in script
    assert 'grep -q \'"skipped_pair_count"\' "${OUTPUT}/converted/manifest.json"' in script
    assert 'pair_count="$(wc -l < "${OUTPUT}/converted/pairwise_labels.jsonl" | tr -d \' \')"' in script
    assert 'paper_count}" -lt 1 || "${paper_count}" -gt $((pair_count * 2))' in script


def test_run_sciimpact_sft_baseline_script_converts_scores_and_evaluates():
    script = Path("scripts/run_sciimpact_sft_baseline.sh").read_text(encoding="utf-8")

    assert 'INPUT="${INPUT:-data/sciimpact/source/sft_data/paper_impact_sft_test_clean.jsonl}"' in script
    assert 'LIMIT="${LIMIT:-1000}"' in script
    assert "benchmarks convert-sciimpact" in script
    assert "sciimpact-text-baseline" in script
    assert "evaluate-pairwise-impact" in script
    assert 'test -s "${OUTPUT}/rankings/sciimpact_text_prior.csv"' in script
    assert 'test -s "${OUTPUT}/pairwise_eval/pairwise_evaluation.json"' in script
    assert "pairwise_accuracy=" in script


def test_run_sciimpact_dpo_export_smoke_script_exports_and_validates_preferences():
    script = Path("scripts/run_sciimpact_dpo_export_smoke.sh").read_text(encoding="utf-8")

    assert 'INPUT="${INPUT:-data/sciimpact/source/sft_data/paper_impact_sft_train_clean.jsonl}"' in script
    assert 'LIMIT="${LIMIT:-1000}"' in script
    assert "benchmarks export-sciimpact-dpo" in script
    assert "validate-dpo-data" in script
    assert 'test -s "${OUTPUT}/finetuning/sciimpact_dpo_preferences.jsonl"' in script
    assert 'test -s "${OUTPUT}/finetuning/sciimpact_dpo_manifest.json"' in script
    assert 'test -s "${OUTPUT}/finetuning/dpo_data_validation.json"' in script
    assert "loaded_example_count" in script


def test_run_sciimpact_prediction_eval_smoke_script_evaluates_oracle_predictions():
    script = Path("scripts/run_sciimpact_prediction_eval_smoke.sh").read_text(encoding="utf-8")

    assert 'INPUT="${INPUT:-data/sciimpact/source/sft_data/paper_impact_sft_test_clean.jsonl}"' in script
    assert 'LIMIT="${LIMIT:-200}"' in script
    assert "oracle_predictions.jsonl" in script
    assert "evaluate-sciimpact-predictions" in script
    assert 'test -s "${OUTPUT}/model_eval/sciimpact_model_eval.json"' in script
    assert 'if [[ "${accuracy}" != "1.000" ]]; then' in script


def test_run_sciimpact_model_eval_script_generates_predictions_and_evaluates():
    script = Path("scripts/run_sciimpact_model_eval.sh").read_text(encoding="utf-8")

    assert 'INPUT="${INPUT:-data/sciimpact/source/sft_data/paper_impact_sft_test_clean.jsonl}"' in script
    assert 'BASE_URL="${BASE_URL:-http://localhost:8000/v1}"' in script
    assert 'MODEL="${MODEL:-local-impact-judge}"' in script
    assert 'scripts/check_model_server.sh "${BASE_URL}"' in script
    assert "generate-sciimpact-predictions" in script
    assert "evaluate-sciimpact-predictions" in script
    assert 'test -s "${OUTPUT}/predictions.jsonl"' in script
    assert 'test -s "${OUTPUT}/eval/sciimpact_model_eval.json"' in script


def test_download_prescience_sample_script_streams_from_mirror():
    script = Path("scripts/download_prescience_sample.sh").read_text(encoding="utf-8")

    assert 'HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"' in script
    assert 'TARGETS="${TARGETS:-100}"' in script
    assert "load_dataset('allenai/prescience'" in script
    assert "os._exit(0)" in script


def test_run_prescience_smoke_script_downloads_converts_and_runs_gold_eval():
    script = Path("scripts/run_prescience_smoke.sh").read_text(encoding="utf-8")

    assert 'scripts/download_prescience_sample.sh "${SAMPLE_PATH}"' in script
    assert 'sample_target_count' in script
    assert 'if [[ ! -s "${SAMPLE_PATH}" || "${sample_target_count}" -lt "${TARGETS}" ]]; then' in script
    assert "benchmarks convert-prescience" in script
    assert "--max-candidates" in script
    assert "--shuffle-seed" in script
    assert "--strict-validation" in script
    assert 'test -s "${OUTPUT}/reports/gold_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/reports/gold_evaluation.md"' in script
    assert 'test -s "${OUTPUT}/reports/gold_candidate_position.json"' in script
    assert 'test -s "${OUTPUT}/reports/gold_candidate_position.md"' in script
    assert 'test -s "${OUTPUT}/reports/gold_baseline_comparison.json"' in script
    assert 'test -s "${OUTPUT}/reports/gold_baseline_comparison.md"' in script
    assert 'report-bundle "${OUTPUT}" --output "${OUTPUT}/report_bundle"' in script
    assert 'test -s "${OUTPUT}/report_bundle/README.md"' in script


def test_run_prescience_gpu_benchmark_script_judges_incrementally_then_scores():
    script = Path("scripts/run_prescience_gpu_benchmark.sh").read_text(encoding="utf-8")

    assert 'SAMPLE_PATH="${SAMPLE_PATH:-data/prescience/sample.jsonl}"' in script
    assert 'LIMIT="${LIMIT:-100}"' in script
    assert 'MAX_CANDIDATES="${MAX_CANDIDATES:-32}"' in script
    assert 'BASE_URL="${BASE_URL:-http://localhost:8000/v1}"' in script
    assert 'MODEL="${MODEL:-local-impact-judge}"' in script
    assert 'AUDIT_PATH="${OUTPUT}/audit/judge_outputs.jsonl"' in script
    assert 'scripts/check_model_server.sh "${BASE_URL}"' in script
    assert "benchmarks convert-prescience" in script
    assert "-m local_impact_ranking.cli judge" in script
    assert "--audit-output" in script
    assert '"${AUDIT_PATH}"' in script
    assert "-m local_impact_ranking.cli score" in script
    assert '--judge-audit "${AUDIT_PATH}"' in script
    assert 'test -s "${OUTPUT}/audit/judge_outputs.jsonl"' in script
    assert 'test -s "${OUTPUT}/reports/evaluation.json"' in script
    assert 'test -s "${OUTPUT}/reports/gold_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/reports/mvp_summary.md"' in script
    assert 'test -f "${OUTPUT}/run_manifest.json"' in script


def test_download_scicite_script_downloads_extracts_and_converts_splits():
    script = Path("scripts/download_scicite.sh").read_text(encoding="utf-8")

    assert 'SCICITE_URL="${SCICITE_URL:-https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz}"' in script
    assert 'DEST_DIR="${1:-data/scicite}"' in script
    assert 'curl -L "${SCICITE_URL}" -o "${ARCHIVE_PATH}"' in script
    assert 'tar -xzf "${ARCHIVE_PATH}" -C "${DEST_DIR}/raw"' in script
    assert 'for split in train dev test; do' in script
    assert 'benchmarks convert-scicite' in script
    assert '${split}.papers.jsonl' in script


def test_check_publish_ready_script_blocks_local_outputs_and_runs_tests():
    script = Path("scripts/check_publish_ready.sh").read_text(encoding="utf-8")

    assert "data/.publish-check" in script
    assert "models/.publish-check" in script
    assert "runs/.publish-check" in script
    assert "git check-ignore" in script
    assert "scripts/run_quickstart.sh" in script
    assert "git ls-files data models runs" in script
    assert "pytest -q" in script
    assert "docs/publish-checklist.md" in script
    assert ".github/workflows/tests.yml" in script
    assert "report-bundle" in script
