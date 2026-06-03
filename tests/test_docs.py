from pathlib import Path


def test_input_schema_doc_covers_required_and_benchmark_fields():
    text = Path("docs/input-schema.md").read_text(encoding="utf-8")

    assert "paper_id" in text
    assert "references" in text
    assert "citation_contexts" in text
    assert "external_impact" in text
    assert "enrich-openalex" in text
    assert "enrich-opencitations" in text
    assert "import-external-evidence" in text
    assert "bip_impact_percentile" in text
    assert "semantic_scholar_influential_citation_count" in text
    assert "openalex_cited_by_count" in text
    assert "opencitations_cited_by_count" in text
    assert "patent_citations" in text
    assert "policy_citations" in text
    assert "dataset_reuses" in text
    assert "gold_reference_ids" in text
    assert "prescience_positive_reference_ids" in text
    assert "convert-csv" in text
    assert "citation-completeness-audit" in text


def test_readme_links_to_input_schema_doc():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "[docs/input-schema.md](docs/input-schema.md)" in text
    assert "[docs/citation-data-sources.md](docs/citation-data-sources.md)" in text
    assert "[docs/effective-citation-calculation.md](docs/effective-citation-calculation.md)" in text
    assert "run_impact_demo.sh" in text
    assert "offline replayed pairwise judgments" in text
    assert "impact_claim_certificate.md" in text
    assert "demo_status.json" in text


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
    assert "impact_index_v3" in text
    assert "impact_index_v2" in text
    assert "evidence_supported_score" in text
    assert "final_impact_score" in text
    assert "true_citation_score" in text
    assert "effective_citation_score" in text
    assert "true_citation_source" in text
    assert "true_effective_citation_impact_v1" in text
    assert "external_adoption_score" in text
    assert "influential_citation_score" in text
    assert "bip_impact_score" in text
    assert "impact_index_v3_final_components" in text
    assert "validity_confidence" in text
    assert "validated_impact_claim" in text
    assert "impact_credibility_scorecard" in text
    assert "model_credibility_card" in text
    assert "model_identity_match" in text
    assert "mismatched_profile" in text
    assert "calibration_profile_card" in text
    assert "calibration_model_identity_match" in text
    assert "patent_impact_score" in text
    assert "0.30 * model_pairwise_score" in text
    assert "field_year_normalized_citation_score" in text
    assert "citation_graph_score" in text
    assert "temporal_momentum_score" in text
    assert "score_ci_low" in text
    assert "uncertainty_band" in text
    assert "reportability_status" in text


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
    assert "Current Impact Credibility" in text
    assert "evaluate-current-impact-credibility" in text
    assert "current_impact_credibility.json" in text
    assert "evaluate-impact-validity" in text
    assert "backtest-impact-validity" in text
    assert "derive-future-impact-labels" in text
    assert "apply-citation-cutoff" in text
    assert "audit-citation-cutoff" in text
    assert "citation_cutoff_audit.json" in text
    assert "failed cutoff audit becomes a blocking condition" in text
    assert "bootstrap" in text
    assert "permutation" in text
    assert "significant_vs_random" in text
    assert "score_label_match_rate" in text
    assert "coverage_examples" in text
    assert "unmatched_score_ids" in text
    assert "impact_credibility_scorecard" in text
    assert "outlier robustness" in text
    assert "credibility_scorecard" in text
    assert "pairwise_credibility_scorecard" in text
    assert "sciimpact_model_credibility_scorecard" in text
    assert "sciimpact_model_credibility_summary" in text
    assert "model_credibility_card" in text
    assert "model_identity_match" in text
    assert "calibration_profile_card" in text
    assert "calibration_model_identity_match" in text
    assert "sciimpact_model_candidate_gate" in text
    assert "sciimpact_model_weak_slice_error_pack" in text
    assert "train_pairwise_sft.jsonl" in text
    assert "weak_slice_source_records.jsonl" in text
    assert "train_sciimpact_weak_slice_dpo.sh" in text
    assert "one-sided binomial" in text
    assert "project-readiness" in text
    assert "not statistically supported" in text
    assert "spearman_correlation" in text
    assert "kendall_tau_b" in text
    assert "ndcg_at_top_k" in text
    assert "future-window" in text
    assert "field_year_percentile" in text
    assert "--label-normalization field-year-percentile" in text
    assert "not global scientific impact" in text


def test_readme_links_to_evaluation_method_doc():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "[docs/evaluation-method.md](docs/evaluation-method.md)" in text


def test_benchmark_mapping_doc_covers_plan_command_and_public_sources():
    text = Path("docs/planning/benchmark-mapping.md").read_text(encoding="utf-8")

    assert "benchmarks plan" in text
    assert "OpCitance + NIH iCite Current Impact" in text
    assert "build-opcitance-icite-current-benchmark" in text
    assert "effective_citation_coverage_rate=0.713" in text
    assert "https://huggingface.co/datasets/allenai/prescience" in text
    assert "pairwise_accuracy" in text
    assert "project-readiness" in text


def test_readme_documents_benchmark_plan_command():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "benchmarks plan" in text
    assert "ready_now" in text
    assert "build-opcitance-icite-current-benchmark" in text
    assert "opcitance_icite_benchmark_summary.md" in text


def test_readme_documents_model_score_dataset_final_score():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "model-score-dataset" in text
    assert "model-score-preflight" in text
    assert "evaluate-current-impact-credibility" in text
    assert "current_impact_credibility.json/md" in text
    assert "total_model_request_count" in text
    assert "planned_pair_count" in text
    assert "enrich-openalex" in text
    assert "enrich-opencitations" in text
    assert "external-evidence-readiness" in text
    assert "papers.openalex.jsonl" in text
    assert "papers.opencitations.jsonl" in text
    assert "impact_index_v3" in text
    assert "impact_index_v2" in text
    assert "final_impact_score" in text
    assert "impact_claim_certificate.md" in text
    assert "model_credibility_card.md" in text
    assert "calibration_profile_card.md" in text
    assert "model_identity_match" in text
    assert "calibration_model_identity_match" in text
    assert "validated`, `evidence_supported`, `exploratory`, or `blocked`" in text
    assert "--report-bundle" in text
    assert "--report-bundle-output" in text
    assert "--pairwise-judgments" in text
    assert "pairwise_judgment_source.type=file" in text
    assert "--resume-pairwise-judgments" in text
    assert "--pairwise-journal" in text
    assert "pairwise_judgment_source.type=resume" in text
    assert "generated_count" in text
    assert "JSONL checkpoint" in text
    assert "serve_recommended_impact_judge.sh" in text
    assert "local-impact-judge-source-task-dpo" in text
    assert "credibility-report" in text
    assert "--min-seed-count 3" in text
    assert "--allowed-seed-selected-regression 0.01" in text
    assert "--allowed-slice-regression 0.06" in text
    assert "mean bidirectional selected accuracy" in text
    assert "mean direct accuracy" in text
    assert "0.95 * true_citation_score" in text
    assert "0.05 * effective_citation_score" in text
    assert "true_citation_source" in text
    assert "score_95_true_05_effective" in text
    assert "score_90_true_10_effective" in text
    assert "score_50_true_50_effective" in text
    assert "score_70_true_30_effective" in text
    assert "legacy ablation scores" in text
    assert "0.30 model_pairwise" in text
    assert "calibrate-fusion-weights" in text
    assert "--fusion-weights" in text
    assert "Legacy SciImpact index-eval CSV files" in text
    assert "evidence_quality_flag" in text
    assert "credible_evidence_tier" in text
    assert "missing_high_value_evidence" in text
    assert "reportability_status" in text
    assert "reportability_basis" in text
    assert "legacy_reportability_status" in text
    assert "external_evidence_readiness.json" in text
    assert "citation_completeness_audit.json" in text
    assert "citation-completeness-audit" in text
    assert "evaluate-impact-validity" in text
    assert "backtest-impact-validity" in text
    assert "derive-future-impact-labels" in text
    assert "apply-citation-cutoff" in text
    assert "audit-citation-cutoff" in text
    assert "citation_cutoff_audit.json" in text
    assert "readiness blocks future-window validity claims" in text
    assert "bootstrap confidence intervals" in text
    assert "permutation random baseline" in text
    assert "impact_credibility_scorecard" in text
    assert "outlier robustness checks" in text
    assert "pairwise_credibility_scorecard" in text
    assert "sciimpact_model_credibility_scorecard" in text
    assert "sciimpact_model_credibility_summary" in text
    assert "sciimpact_model_candidate_gate" in text
    assert "sciimpact_model_weak_slice_error_pack" in text
    assert "train_pairwise_dpo.jsonl" in text
    assert "weak_slice_source_records.jsonl" in text
    assert "project-readiness warns" in text
    assert "score_label_match_rate" in text
    assert "coverage_examples" in text
    assert "--label-normalization field-year-percentile" in text


def test_citation_data_sources_doc_covers_source_roles_and_reconciliation():
    text = Path("docs/citation-data-sources.md").read_text(encoding="utf-8")

    assert "OpenAlex" in text
    assert "citation-completeness-audit" in text
    assert "external-evidence-template" in text
    assert "multi_source_compatible" in text
    assert "Semantic Scholar" in text
    assert "OpenCitations" in text
    assert "PubMed/iCite" in text
    assert "Crossref" in text
    assert "Crossref Data Citations" in text
    assert "DataCite Events" in text
    assert "Wikidata" in text
    assert "PLOS Metrics" in text
    assert "uCite" in text
    assert "NIH-OCC" in text
    assert "Scopus" in text
    assert "Web of Science" in text
    assert "Google Scholar" in text
    assert "semantic_scholar_influential_citation_count" in text
    assert "crossref_data_citation_count" in text
    assert "datacite_events_cited_by_count" in text
    assert "wikidata_cited_by_count" in text
    assert "ucite_citation_count" in text
    assert "nih_occ_citation_count" in text
    assert "google_scholar_citation_count" in text
    assert "field_year_normalized_citation_score" in text
    assert "Do not average citation counts" in text
    assert "citation_count_ratio" in text
    assert "Life-Science Preference" in text


def test_effective_citation_calculation_doc_covers_formula_and_outputs():
    text = Path("docs/effective-citation-calculation.md").read_text(encoding="utf-8")

    assert "citing_paper.references[].reference_id == cited_paper.paper_id" in text
    assert "edge_effective_weight = max(role_weight(context_roles_for_edge))" in text
    assert "effective_citation_raw" in text
    assert "tied_rank_percentile(log1p(effective_citation_raw))" in text
    assert "effective_citation_count" in text
    assert "effective_citation_coverage" in text
    assert "effective_citation_evidence_level" in text
    assert "metadata.impact_selected_references" in text
    assert "effective_citation_impact_selected_raw" in text
    assert "per-edge max rule" in text
    assert "guideline" in text
    assert "clinical" in text
    assert "method" in text


def test_readme_documents_long_context_tokenization_preflight():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "inspect-sft-tokenization" in text
    assert "--max-length 16384" in text
    assert "8192" in text
    assert "earlier 1024-token LoRA experiments" in text


def test_readme_documents_report_bundle_command():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "report-bundle" in text
    assert "bundle_manifest.json" in text
    assert "Model Score Dataset" in text
    assert "Model Credibility" in text
    assert "Calibration Profile" in text
    assert "model_credibility_card.json/md" in text
    assert "calibration_profile_card.json/md" in text
    assert "model_identity_match" in text
    assert "calibration_model_identity_match" in text
    assert "bundle_decision" in text
    assert "model_impact_report.json/md" in text
    assert "impact_scores.csv/jsonl" in text


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
    assert "run_quickstart.sh" in text
    assert "scripts/run_deliverable_check.sh" in text
    assert "CHECK_MODEL=1" in text


def test_readme_documents_builtin_calibration_profile():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "model-score-dataset" in text
    assert "AUTO_ENRICH=1" in text
    assert "AUTO_ENRICH_SOURCES=openalex,europepmc,identity_recovery,europepmc,crossref,crossref_data_citations,icite,ncbi_citations,opencitations,semantic_scholar,datacite,datacite_events,openaire" in text
    assert "auto_enrich_status.json" in text
    assert "scoring_input" in text
    assert "enrich-icite" in text
    assert "enrich-europepmc" in text
    assert "identity_recovery" in text
    assert "enrich-crossref" in text
    assert "enrich-crossref-data-citations" in text
    assert "enrich-ncbi-citations" in text
    assert "enrich-datacite" in text
    assert "enrich-datacite-events" in text
    assert "enrich-openaire" in text
    assert "enrich-wikidata" in text
    assert "enrich-plos-metrics" in text
    assert "configs/calibration/sciimpact-qwen32b-heldout-p50.json" in text
    assert "compatible_models" in text
    assert "calibration_profile_card.md" in text
    assert "calibration_model_identity_match" in text
    assert "--validity-labels" in text
    assert "SciImpact-derived" in text


def test_project_summary_report_matches_current_deliverable():
    text = Path("docs/planning/project-summary-report.md").read_text(encoding="utf-8")

    assert "run_user_impact.sh" in text
    assert "user_impact_summary.md" in text
    assert "report_decision" in text
    assert "final_impact_score" in text
    assert "impact_index_v2/v3" in text
    assert "local-impact-judge-source-task-dpo" in text
    assert "model_smoke=passed" in text
    assert "473 passed" in text
    assert "not global scientific impact" in text


def test_docs_document_responsible_metrics_upgrade():
    readme = Path("README.md").read_text(encoding="utf-8")
    scoring = Path("docs/scoring-method.md").read_text(encoding="utf-8")

    assert "relative_citation_ratio_score" in readme
    assert "impact_score_cards.md" in readme
    assert "Impact Dimension Coverage" in readme
    assert "citation outlier diagnostics" in readme
    assert "import-external-evidence" in readme
    assert "external-evidence-template" in readme
    assert "scripts/apply_high_recall_evidence.sh" in readme
    assert "citation_completeness_delta" in readme
    assert "evidence_validation" in readme
    assert "PMID, or PMCID" in readme
    assert "uCite/NIH-OCC biomedical citation counts" in readme
    assert "Scopus/Web of Science/Dimensions/Lens/Google Scholar" in readme
    assert "Influential Citation" in scoring
    assert "BIP" in scoring
    assert "self_citation_risk" in scoring
    assert "citation_outlier_flag" in scoring
    assert "media_impact_score" in scoring
    assert "award_impact_score" in scoring


def test_publish_checklist_documents_github_release_boundaries():
    text = Path("docs/publish-checklist.md").read_text(encoding="utf-8")

    assert "data/" in text
    assert "models/" in text
    assert "runs/" in text
    assert "configs/calibration/" in text
    assert "pytest -q" in text
    assert "report-bundle" in text
    assert "dataset-local" in text


def test_qwen32b_plan_documents_three_seed_promotion_and_limits():
    text = Path("docs/planning/qwen32b-sciimpact-evaluation.md").read_text(
        encoding="utf-8"
    )

    assert "SEED=71" in text
    assert "promote_candidate_for_bidirectional_scoring" in text
    assert "qwen32b-source-task-sft16k4096-dpo16k1024-128step" in text
    assert "Mean bidirectional selected accuracy" in text
    assert "Source-task SFT+DPO" in text
    assert "dataset-local claim boundary" in text
