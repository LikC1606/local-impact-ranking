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


def test_serve_recommended_impact_judge_serves_source_task_dpo_adapter():
    script = Path("scripts/serve_recommended_impact_judge.sh").read_text(encoding="utf-8")

    assert "qwen32b-source-task-sft16k4096-dpo16k1024-128step" in script
    assert 'SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local-impact-judge-source-task-dpo}"' in script
    assert 'SERVER_HOST="${SERVER_HOST:-0.0.0.0}"' in script
    assert 'HOST="${HOST:-0.0.0.0}"' not in script
    assert "--enable-lora" in script
    assert "--lora-modules" in script
    assert 'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"' in script
    assert 'MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"' in script


def test_train_sciimpact_long_context_sft_script_runs_preflight_then_16k_training():
    script = Path("scripts/train_sciimpact_qwen32b_long_context_sft.sh").read_text(
        encoding="utf-8"
    )

    assert 'export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"' in script
    assert 'MAX_LENGTH="${MAX_LENGTH:-16384}"' in script
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"' in script
    assert "inspect-sft-tokenization" in script
    assert '--max-length "${MAX_LENGTH}"' in script
    assert '--max-examples "${MAX_EXAMPLES}"' in script
    assert "train-sft" in script
    assert 'OUTPUT="${OUTPUT:-runs/sciimpact-long-context/checkpoints/qwen32b-sft16k-16k-256step}"' in script
    assert 'MAX_STEPS="${MAX_STEPS:-256}"' in script


def test_train_sciimpact_long_context_dpo_script_continues_from_16k_sft():
    script = Path("scripts/train_sciimpact_qwen32b_long_context_dpo.sh").read_text(
        encoding="utf-8"
    )

    assert 'export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"' in script
    assert 'BASE_ADAPTER="${BASE_ADAPTER:-runs/sciimpact-long-context/checkpoints/qwen32b-sft16k-16k-256step}"' in script
    assert 'TRAIN_FILE="${TRAIN_FILE:-runs/sciimpact-pairwise-dpo-full/train_pairwise_dpo.jsonl}"' in script
    assert 'OUTPUT="${OUTPUT:-runs/sciimpact-long-context/checkpoints/qwen32b-sft16k-16k-dpo16k-256step}"' in script
    assert 'MAX_LENGTH="${MAX_LENGTH:-16384}"' in script
    assert 'MAX_STEPS="${MAX_STEPS:-256}"' in script
    assert 'LEARNING_RATE="${LEARNING_RATE:-1e-6}"' in script
    assert "train-dpo" in script
    assert '--base-adapter "${BASE_ADAPTER}"' in script
    assert '--max-length "${MAX_LENGTH}"' in script


def test_train_sciimpact_weak_slice_dpo_script_validates_candidates_before_training():
    script = Path("scripts/train_sciimpact_weak_slice_dpo.sh").read_text(
        encoding="utf-8"
    )

    assert 'TRAIN_FILE="${TRAIN_FILE:-runs/sciimpact-hard-example-train-p100-dpo64/weak_slice_training_candidates/train_pairwise_dpo.jsonl}"' in script
    assert "qwen32b-source-task-sft16k4096-dpo16k1024-128step" in script
    assert 'MAX_STEPS="${MAX_STEPS:-32}"' in script
    assert 'MAX_EXAMPLES="${MAX_EXAMPLES:-139}"' in script
    assert 'MAX_LENGTH="${MAX_LENGTH:-4096}"' in script
    assert "validate-dpo-data" in script
    assert "train-dpo" in script
    assert '--base-adapter "${BASE_ADAPTER}"' in script
    assert '--train-file "${TRAIN_FILE}"' in script


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


def test_run_impact_demo_script_scores_dataset_without_gpu():
    script = Path("scripts/run_impact_demo.sh").read_text(encoding="utf-8")

    assert 'OUTPUT="${1:-runs/impact-demo}"' in script
    assert "model-score-preflight" in script
    assert "model-score-dataset" in script
    assert "--input examples/impact_demo_papers.jsonl" in script
    assert "--pairwise-judgments examples/impact_demo_pairwise_judgments.jsonl" in script
    assert "--validity-labels examples/impact_demo_future_labels.jsonl" in script
    assert "--report-bundle-output" in script
    assert 'test -s "${OUTPUT}/model-impact/impact_scores.csv"' in script
    assert 'test -s "${OUTPUT}/model-impact/impact_claim_certificate.md"' in script
    assert 'test -s "${OUTPUT}/model-impact/report_bundle/bundle_manifest.json"' in script
    assert 'test -s "${OUTPUT}/demo_status.json"' in script
    assert 'test -s "${OUTPUT}/demo_report.md"' in script
    assert 'claim = certificate.get("claim", {})' in script
    assert "impact claim certificate is missing claim.status or claim.reason" in script
    assert "claim.get('status', '')" in script
    assert "can_claim_high_credibility" in script
    assert 'grep -q "| status | evidence_supported |" "${OUTPUT}/demo_report.md"' in script
    assert "validated_impact_claim.status" in script
    assert "final_impact_score" in script
    assert "Impact Demo Report" in script


def test_run_user_impact_script_scores_user_dataset_with_model_and_bundle():
    script = Path("scripts/run_user_impact.sh").read_text(encoding="utf-8")

    assert 'INPUT="${1:?Usage: scripts/run_user_impact.sh INPUT_JSONL [OUTPUT_DIR]}"' in script
    assert 'OUTPUT="${2:-runs/user-impact}"' in script
    assert 'if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then' in script
    assert "Usage: scripts/run_user_impact.sh INPUT_JSONL [OUTPUT_DIR]" in script
    assert "Common environment variables:" in script
    assert "RUN_PRESET=quick|balanced|full|custom" in script
    assert "MODEL_CREDIBILITY_REPORT" in script
    assert "PAIRWISE_JUDGMENTS" in script
    assert "RESUME_PAIRWISE_JUDGMENTS" in script
    assert "VALIDITY_LABELS" in script
    assert "AUTO_ENRICH=0|1" in script
    assert "AUTO_ENRICH_SOURCES=openalex,europepmc,identity_recovery,europepmc,crossref,crossref_data_citations,icite,ncbi_citations,opencitations,semantic_scholar,datacite,datacite_events,openaire" in script
    assert 'BASE_URL="${BASE_URL:-http://localhost:8000/v1}"' in script
    assert 'MODEL="${MODEL:-local-impact-judge-source-task-dpo}"' in script
    assert 'RUN_PRESET="${RUN_PRESET:-custom}"' in script
    assert 'AUTO_ENRICH="${AUTO_ENRICH:-0}"' in script
    assert 'AUTO_ENRICH_SOURCES="${AUTO_ENRICH_SOURCES:-openalex,europepmc,identity_recovery,europepmc,crossref,crossref_data_citations,icite,ncbi_citations,opencitations,semantic_scholar,datacite,datacite_events,openaire}"' in script
    assert 'AUTO_ENRICH_TIMEOUT="${AUTO_ENRICH_TIMEOUT:-30}"' in script
    assert 'SCORING_INPUT="${INPUT}"' in script
    assert 'AUTO_ENRICH_OUTPUT="${OUTPUT}/auto-enrich"' in script
    assert 'enrich-openalex' in script
    assert 'enrich-europepmc' in script
    assert 'recover-external-identities' in script
    assert 'enrich-crossref' in script
    assert 'enrich-crossref-data-citations' in script
    assert 'enrich-icite' in script
    assert 'enrich-ncbi-citations' in script
    assert 'enrich-opencitations' in script
    assert 'enrich-semantic-scholar' in script
    assert 'enrich-datacite' in script
    assert 'enrich-datacite-events' in script
    assert 'enrich-openaire' in script
    assert 'enrich-plos-metrics' in script
    assert 'next_input="${AUTO_ENRICH_OUTPUT}/papers.openalex.jsonl"' in script
    assert 'next_input="${AUTO_ENRICH_OUTPUT}/papers.europepmc.jsonl"' in script
    assert 'next_input="${AUTO_ENRICH_OUTPUT}/papers.identity_recovered.jsonl"' in script
    assert 'next_input="${AUTO_ENRICH_OUTPUT}/papers.crossref.jsonl"' in script
    assert 'next_input="${AUTO_ENRICH_OUTPUT}/papers.crossref_data_citations.jsonl"' in script
    assert 'next_input="${AUTO_ENRICH_OUTPUT}/papers.icite.jsonl"' in script
    assert 'next_input="${AUTO_ENRICH_OUTPUT}/papers.ncbi_citations.jsonl"' in script
    assert 'next_input="${AUTO_ENRICH_OUTPUT}/papers.opencitations.jsonl"' in script
    assert 'next_input="${AUTO_ENRICH_OUTPUT}/papers.semantic_scholar.jsonl"' in script
    assert 'next_input="${AUTO_ENRICH_OUTPUT}/papers.datacite.jsonl"' in script
    assert 'next_input="${AUTO_ENRICH_OUTPUT}/papers.datacite_events.jsonl"' in script
    assert 'next_input="${AUTO_ENRICH_OUTPUT}/papers.openaire.jsonl"' in script
    assert 'SCORING_INPUT="${auto_enrich_current_input}"' in script
    assert '"${SCORING_INPUT}"' in script
    assert '"quick")' in script
    assert '"balanced")' in script
    assert '"full")' in script
    assert "Unknown RUN_PRESET" in script
    assert 'PAIR_BUDGET="${PAIR_BUDGET:-auto}"' in script
    assert 'PAIR_SAMPLING="${PAIR_SAMPLING:-coverage}"' in script
    assert 'TARGET_PAPER_COVERAGE="${TARGET_PAPER_COVERAGE:-0.25}"' in script
    assert 'BIDIRECTIONAL_PAIRWISE="${BIDIRECTIONAL_PAIRWISE:-1}"' in script
    assert 'CALIBRATION_REPORT="${CALIBRATION_REPORT:-configs/calibration/sciimpact-qwen32b-heldout-p50.json}"' in script
    assert 'MODEL_CREDIBILITY_REPORT="${MODEL_CREDIBILITY_REPORT:-configs/model_credibility/local-impact-judge-source-task-dpo.json}"' in script
    assert 'AUTO_ENRICH="${AUTO_ENRICH:-0}"' in script
    assert 'AUTO_ENRICH_SOURCES="${AUTO_ENRICH_SOURCES:-openalex,europepmc,identity_recovery,europepmc,crossref,crossref_data_citations,icite,ncbi_citations,opencitations,semantic_scholar,datacite,datacite_events,openaire}"' in script
    assert 'AUTO_ENRICH_TIMEOUT="${AUTO_ENRICH_TIMEOUT:-30}"' in script
    assert 'enrich-openalex' in script
    assert 'enrich-europepmc' in script
    assert 'recover-external-identities' in script
    assert 'enrich-crossref' in script
    assert 'enrich-crossref-data-citations' in script
    assert 'enrich-icite' in script
    assert 'enrich-ncbi-citations' in script
    assert 'enrich-opencitations' in script
    assert 'enrich-semantic-scholar' in script
    assert 'enrich-datacite' in script
    assert 'enrich-datacite-events' in script
    assert 'enrich-openaire' in script
    assert 'enrich-plos-metrics' in script
    assert '--input "${SCORING_INPUT}"' in script
    assert 'auto_enrich_enabled = sys.argv[8] == "1"' in script
    assert 'auto_enrich_sources = [source for source in sys.argv[9].split(",") if source]' in script
    assert '"auto_enrich_input": sys.argv[2]' in script
    assert '"scoring_input": scoring_input' in script
    assert "Auto Enrichment" in script
    assert "auto_enrich" in script
    assert "scoring_input" in script
    assert 'scripts/check_model_server.sh "${BASE_URL}"' in script
    assert "validate" in script
    assert "model-score-preflight" in script
    assert "model-score-dataset" in script
    assert "--report-bundle" in script
    assert '--report-bundle-output "${OUTPUT}/model-impact/report_bundle"' in script
    assert "--pairwise-journal" in script
    assert "FUSION_WEIGHTS" in script
    assert 'test -s "${OUTPUT}/validation/validation.json"' in script
    assert 'test -s "${OUTPUT}/preflight/model_score_preflight.json"' in script
    assert 'test -s "${OUTPUT}/model-impact/impact_scores.csv"' in script
    assert 'test -s "${OUTPUT}/model-impact/citation_completeness_audit.md"' in script
    assert 'test -s "${OUTPUT}/model-impact/citation_completeness_papers.csv"' in script
    assert 'test -s "${OUTPUT}/model-impact/impact_claim_certificate.md"' in script
    assert 'test -s "${OUTPUT}/model-impact/report_bundle/bundle_manifest.json"' in script
    assert 'grep -q "validated_impact_claim.status" "${OUTPUT}/model-impact/model_impact_report.md"' in script
    assert 'user_impact_status.json' in script
    assert 'user_impact_summary.md' in script
    assert "User Impact Run Summary" in script
    assert "Evidence Gaps" in script
    assert "Recommended Next Actions" in script
    assert "top_missing_evidence" in script
    assert "report_decision" in script
    assert "report_with_disclosure" in script
    assert "exploratory_only" in script
    assert "validation_issue_count" in script
    assert "validation_error_count" in script
    assert "validation_warning_count" in script
    assert "validation_disclosure_suffix" in script
    assert "final_score_stability_blocks_reporting" in script
    assert "final_score_stability_disclosure_suffix" in script
    assert "no_reportable_scores_block_reporting" in script
    assert "No papers are marked reportable" in script
    assert "Final-score stability is unstable" in script
    assert "Final-score stability is moderate" in script
    assert "Input validation has" in script
    assert "Input validation reports" not in script
    assert "disclose or fix them" not in script
    assert "Validation Issues" in script
    assert "Internal Citation Graph Audit" in script
    assert "Citation Completeness Audit" in script
    assert "citation_completeness_audit" in script
    assert "citation_evidence_coverage" in script
    assert "multi_source_coverage" in script
    assert "papers_with_source_conflict" in script
    assert "papers_without_query_ids" in script
    assert "internal_citation_audit_coverage" in script
    assert "openalex_reference_audit_coverage" in script
    assert 'preflight_args+=(--resume-pairwise-judgments "${PAIRWISE_JUDGMENTS}")' in script
    assert "Pair Plan Quality" in script
    assert "preflight_pair_plan_quality_status" in script
    assert "preflight_pair_plan_connected_component_count" in script
    assert "run_preset" in script
    assert "auto_enrich_enabled" in script
    assert "auto_enrich_sources" in script
    assert "auto_enrich_input" in script
    assert "scoring_input" in script
    assert "auto_enrich_output" in script
    assert "Auto Enrichment" in script
    assert "Model Credibility" in script
    assert "model_credibility_status" in script
    assert "model_credibility_card" in script
    assert "model_credibility_card.md" in script
    assert "model_credibility_model_identity_match" in script
    assert "served_model" in script
    assert "calibration_profile_card" in script
    assert "calibration_status" in script
    assert "calibration_model_identity_match" in script
    assert "can_use_for_report_with_disclosure" in script
    assert "pooled_accuracy" in script
    assert "weak_slices" in script
    assert "report_decision_with_model_credibility" in script
    assert "claim_status" in script
    assert "ranking_quality_status" in script
    assert "dataset-local paper influence" in script
    assert "Source Quality" in script
    assert "excluded_non_article" in script
    assert "reportable_paper_count" in script
    assert "exploratory_paper_count" in script
    assert "low_confidence_paper_count" in script
    assert "Positive Evidence Signals" in script
    assert "Zero Evidence Signals" in script
    assert 'PAIRWISE_STABILITY_SAMPLES="${PAIRWISE_STABILITY_SAMPLES:-0}"' in script
    assert 'FINAL_SCORE_STABILITY_SAMPLES="${FINAL_SCORE_STABILITY_SAMPLES:-0}"' in script
    assert "--pairwise-stability-samples" in script
    assert "--pairwise-stability-top-n" in script
    assert "--pairwise-stability-seed" in script
    assert "--final-score-stability-samples" in script
    assert "--final-score-stability-top-n" in script
    assert "--final-score-stability-seed" in script
    assert "pairwise_stability_status" in script
    assert "final_score_stability_status" in script
    assert "Pairwise Stability" in script
    assert "Final Score Stability" in script
    assert 'test -s "${OUTPUT}/model-impact/pairwise_stability.md"' in script
    assert 'test -s "${OUTPUT}/model-impact/final_score_stability.md"' in script


def test_run_user_impact_auto_enrich_uses_distinct_outputs_for_repeated_sources():
    script = Path("scripts/run_user_impact.sh").read_text(encoding="utf-8")

    assert "declare -A auto_enrich_source_counts=()" in script
    assert 'source_count="${auto_enrich_source_counts[${source}]:-0}"' in script
    assert 'source_count=$((source_count + 1))' in script
    assert 'papers.europepmc.${source_count}.jsonl' in script
    assert 'europepmc_${source_count}_summary.json' in script


def test_run_deliverable_check_script_runs_required_and_optional_delivery_paths():
    script = Path("scripts/run_deliverable_check.sh").read_text(encoding="utf-8")

    assert 'OUTPUT="${1:-runs/deliverable-check}"' in script
    assert 'export PYTHON_BIN' in script
    assert 'scripts/run_quickstart.sh "${OUTPUT}/quickstart"' in script
    assert 'scripts/run_impact_demo.sh "${OUTPUT}/impact-demo"' in script
    assert 'scripts/run_user_impact.sh examples/impact_demo_papers.jsonl "${OUTPUT}/user-impact"' in script
    assert 'scripts/run_sciimpact_smoke.sh "${OUTPUT}/sciimpact-smoke"' in script
    assert "PAIRWISE_JUDGMENTS=examples/impact_demo_pairwise_judgments.jsonl" in script
    assert "VALIDITY_LABELS=examples/impact_demo_future_labels.jsonl" in script
    assert 'CHECK_MODEL="${CHECK_MODEL:-0}"' in script
    assert 'scripts/check_model_server.sh "${BASE_URL}"' in script
    assert "model-score-dataset" in script
    assert "--pair-sampling coverage" in script
    assert "--pair-budget auto" in script
    assert "--bidirectional-pairwise" in script
    assert "configs/calibration/sciimpact-qwen32b-heldout-p50.json" in script
    assert "configs/model_credibility/local-impact-judge-source-task-dpo.json" in script
    assert "--report-bundle" in script
    assert '--report-bundle-output "${OUTPUT}/model-impact/report_bundle"' in script
    assert "backtest-impact-validity" in script
    assert "--label-normalization field-year-percentile" in script
    assert 'test -s "${OUTPUT}/backtest/reports/citation_cutoff_audit.json"' in script
    assert 'test -s "${OUTPUT}/backtest/reports/impact_validity_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/backtest/reports/impact_credibility_scorecard.json"' in script
    assert 'test -s "${OUTPUT}/backtest/future-labels/future_impact_labels.jsonl"' in script
    assert 'test -s "${OUTPUT}/quickstart/run/report_bundle/bundle_manifest.json"' in script
    assert 'test -s "${OUTPUT}/quickstart/run/reports/external_evidence_paper_coverage.csv"' in script
    assert 'test -s "${OUTPUT}/impact-demo/demo_status.json"' in script
    assert 'test -s "${OUTPUT}/impact-demo/demo_report.md"' in script
    assert 'test -s "${OUTPUT}/impact-demo/model-impact/impact_scores.csv"' in script
    assert 'test -s "${OUTPUT}/impact-demo/model-impact/impact_claim_certificate.md"' in script
    assert 'test -s "${OUTPUT}/user-impact/user_impact_status.json"' in script
    assert 'test -s "${OUTPUT}/user-impact/user_impact_summary.md"' in script
    assert 'test -s "${OUTPUT}/user-impact/model-impact/impact_scores.csv"' in script
    assert 'test -s "${OUTPUT}/user-impact/model-impact/model_credibility_card.md"' in script
    assert 'test -s "${OUTPUT}/user-impact/model-impact/report_bundle/model_credibility_card.md"' in script
    assert 'test -s "${OUTPUT}/user-impact/model-impact/calibration_profile_card.md"' in script
    assert 'test -s "${OUTPUT}/user-impact/model-impact/report_bundle/calibration_profile_card.md"' in script
    assert 'grep -q "User Impact Run Summary" "${OUTPUT}/user-impact/user_impact_summary.md"' in script
    assert 'test -s "${OUTPUT}/sciimpact-smoke/pairwise_eval/pairwise_evaluation.json"' in script
    assert 'test -s "${OUTPUT}/sciimpact-smoke/pairwise_eval/pairwise_credibility_scorecard.json"' in script
    assert 'test -s "${OUTPUT}/model-impact/external_evidence_paper_coverage.csv"' in script
    assert 'test -s "${OUTPUT}/model-impact/impact_claim_certificate.md"' in script
    assert 'test -s "${OUTPUT}/model-impact/report_bundle/impact_scores.csv"' in script
    assert 'test -s "${OUTPUT}/model-impact/report_bundle/model_impact_report.md"' in script
    assert 'grep -q "Model Score Dataset" "${OUTPUT}/model-impact/report_bundle/README.md"' in script
    assert 'grep -q "validated_impact_claim.status"' in script
    assert 'grep -q "validity_confidence" "${OUTPUT}/model-impact/impact_scores.csv"' in script
    assert 'grep -q "Impact Claim Certificate" "${OUTPUT}/model-impact/impact_claim_certificate.md"' in script
    assert 'deliverable_status.json' in script
    assert '"user_impact": "passed"' in script


def test_github_actions_runs_no_gpu_reproduction_entrypoint():
    workflow = Path(".github/workflows/tests.yml").read_text(encoding="utf-8")

    assert "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true" in workflow
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert "scripts/reproduce_no_gpu.sh" in workflow
    assert "SKIP_INSTALL=1" in workflow
    assert "runs/reproduce-no-gpu" in workflow


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
    assert 'test -s "${OUTPUT}/pairwise_eval/pairwise_credibility_scorecard.json"' in script
    assert 'test -s "${OUTPUT}/pairwise_eval/pairwise_credibility_scorecard.md"' in script


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
    assert 'test -s "${OUTPUT}/pairwise_eval/pairwise_credibility_scorecard.json"' in script
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
    assert 'test -s "${OUTPUT}/model_eval/sciimpact_model_credibility_scorecard.json"' in script
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
    assert 'test -s "${OUTPUT}/eval/sciimpact_model_credibility_scorecard.json"' in script


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


def test_apply_high_recall_evidence_script_imports_audits_and_optionally_replays():
    script = Path("scripts/apply_high_recall_evidence.sh").read_text(encoding="utf-8")

    assert 'INPUT="${1:?Usage: scripts/apply_high_recall_evidence.sh INPUT_JSONL EVIDENCE_CSV [OUTPUT_DIR]}"' in script
    assert 'EVIDENCE="${2:?Usage: scripts/apply_high_recall_evidence.sh INPUT_JSONL EVIDENCE_CSV [OUTPUT_DIR]}"' in script
    assert 'OUTPUT="${3:-runs/high-recall-evidence}"' in script
    assert "import-external-evidence" in script
    assert "--evidence" in script
    assert "citation-completeness-audit" in script
    assert 'CITATION_COMPLETENESS_BEFORE_OUTPUT="${OUTPUT}/citation-completeness-before"' in script
    assert 'PAIRWISE_JUDGMENTS="${PAIRWISE_JUDGMENTS:-}"' in script
    assert 'if [[ -n "${PAIRWISE_JUDGMENTS}" ]]; then' in script
    assert "model-score-dataset" in script
    assert "--pairwise-judgments" in script
    assert "--report-bundle" in script
    assert 'test -s "${OUTPUT}/import_summary.json"' in script
    assert 'test -s "${OUTPUT}/citation-completeness-before/citation_completeness_audit.json"' in script
    assert 'test -s "${OUTPUT}/citation-completeness/citation_completeness_audit.json"' in script
    assert '"citation_completeness_delta":' in script
    assert 'STATUS_PATH="${OUTPUT}/high_recall_external_evidence_status.json"' in script


def test_check_publish_ready_script_blocks_local_outputs_and_runs_tests():
    script = Path("scripts/check_publish_ready.sh").read_text(encoding="utf-8")

    assert "configs/calibration/sciimpact-qwen32b-heldout-p50.json" in script
    assert "data/.publish-check" in script
    assert "models/.publish-check" in script
    assert "runs/.publish-check" in script
    assert "git check-ignore" in script
    assert "scripts/run_quickstart.sh" in script
    assert "scripts/run_user_impact.sh" in script
    assert "scripts/run_deliverable_check.sh" in script
    assert "scripts/train_sciimpact_weak_slice_dpo.sh" in script
    assert "git ls-files data models runs" in script
    assert "pytest -q" in script
    assert "docs/publish-checklist.md" in script
    assert ".github/workflows/tests.yml" in script
    assert "report-bundle" in script
    assert "run_user_impact.sh" in script
    assert "user_impact_status.json" in script
    assert "user_impact_summary.md" in script
    assert "model_credibility_status" in script
    assert "run_deliverable_check.sh" in script
    assert "rank_centrality" in script
    assert "rank-percentile" in script
    assert "import-external-evidence" in script
    assert "--validity-labels" in script
    assert "influential_citation_score" in script
    assert "bip_impact_score" in script
    assert "impact_index_v3_final_components" in script
    assert "validity_confidence" in script
    assert "validated_impact_claim" in script
    assert "sciimpact_model_credibility_summary" in script
    assert "sciimpact_model_candidate_gate" in script
    assert "sciimpact_model_weak_slice_error_pack" in script
