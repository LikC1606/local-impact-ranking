import math

from local_impact_ranking.impact_fusion import compute_impact_index_v2
from local_impact_ranking.models import Paper


def test_compute_impact_index_v2_fuses_model_citation_graph_context_and_momentum():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Foundational method",
                "year": 2020,
                "field": "AI",
                "citation_count": 1000,
                "metadata": {"citation_counts_by_year": {"2023": 5, "2024": 5, "2025": 5}},
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p2",
                "title": "Adopted benchmark",
                "year": 2020,
                "field": "AI",
                "metadata": {
                    "citation_count": 80,
                    "citation_counts_by_year": {"2023": 20, "2024": 30, "2025": 40},
                },
                "references": [{"reference_id": "p1", "title": "Foundational method"}],
                "citation_contexts": [
                    {
                        "reference_id": "p1",
                        "sentence": "We build on this method as our baseline.",
                        "section": "Methods",
                    }
                ],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p3",
                "title": "Narrow follow-up",
                "year": 2021,
                "field": "AI",
                "citation_count": 8,
                "metadata": {"citation_counts_by_year": {"2023": 1, "2024": 0, "2025": 0}},
                "references": [
                    {"reference_id": "p1", "title": "Foundational method"},
                    {"reference_id": "p2", "title": "Adopted benchmark"},
                ],
                "citation_contexts": [
                    {
                        "reference_id": "p2",
                        "sentence": "We compare against this benchmark.",
                        "section": "Experiments",
                    }
                ],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": 1,
            "paper_id": "p2",
            "title": "Adopted benchmark",
            "impact_index": 100.0,
            "comparisons": 2,
            "confidence": 0.8,
        },
        {
            "rank": 2,
            "paper_id": "p1",
            "title": "Foundational method",
            "impact_index": 60.0,
            "comparisons": 2,
            "confidence": 0.8,
        },
        {
            "rank": 3,
            "paper_id": "p3",
            "title": "Narrow follow-up",
            "impact_index": 0.0,
            "comparisons": 2,
            "confidence": 0.8,
        },
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert result["summary"]["version"] == "true_effective_citation_impact_v1"
    assert result["summary"]["legacy_fusion_version"] == "impact_index_v3_evidence_first"
    assert result["summary"]["signal_availability"]["model_pairwise"]["available_paper_count"] == 3
    assert result["summary"]["signal_availability"]["citation_context_quality"]["available_paper_count"] == 3
    assert by_id["p2"]["impact_index_v2"] > by_id["p3"]["impact_index_v2"]
    assert by_id["p1"]["field_year_normalized_citation_score"] == 100.0
    assert by_id["p1"]["citation_count_raw"] == 1000.0
    assert by_id["p2"]["temporal_momentum_score"] == 100.0
    assert by_id["p1"]["citation_context_quality_score"] == 100.0
    assert by_id["p2"]["fusion_signal_count"] == 5
    assert math.isclose(by_id["p2"]["fusion_available_weight"], 1.0)
    assert by_id["p2"]["fusion_weight_model_pairwise"] == 0.3
    assert by_id["p2"]["fusion_weight_temporal_momentum"] == 0.1
    assert by_id["p2"]["evidence_quality_flag"] in {"moderate", "strong"}
    assert by_id["p2"]["score_interval_method"] == "leave_one_signal_out"
    assert by_id["p2"]["score_ci_low"] <= by_id["p2"]["impact_index_v2"]
    assert by_id["p2"]["score_ci_high"] >= by_id["p2"]["impact_index_v2"]
    assert by_id["p2"]["uncertainty_band"] in {"low", "medium"}
    assert by_id["p2"]["reportability_status"] in {"reportable", "exploratory"}
    assert result["summary"]["reportability"]["paper_count"] == 3


def test_final_impact_score_uses_only_true_and_effective_citations():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "high_true",
                "title": "Highly cited paper",
                "citation_count": 1000,
                "metadata": {
                    "external_impact": {
                        "patent_citations": 0,
                        "policy_citations": 0,
                        "software_mentions": 0,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "effective",
                "title": "Method reused paper",
                "citation_count": 10,
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "model_liked",
                "title": "Model liked but uncited paper",
                "citation_count": 0,
                "metadata": {
                    "external_impact": {
                        "field_year_percentile": 0.99,
                        "patent_citations": 20,
                        "policy_citations": 20,
                        "software_mentions": 100,
                    }
                },
                "references": [
                    {
                        "reference_id": "effective",
                        "title": "Method reused paper",
                    }
                ],
                "citation_contexts": [
                    {
                        "reference_id": "effective",
                        "sentence": "We used this method as the assay protocol.",
                        "section": "Methods",
                    }
                ],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": 1,
            "paper_id": "model_liked",
            "title": "Model liked but uncited paper",
            "impact_index": 100.0,
            "comparisons": 2,
            "confidence": 0.9,
        },
        {
            "rank": 2,
            "paper_id": "high_true",
            "title": "Highly cited paper",
            "impact_index": 20.0,
            "comparisons": 2,
            "confidence": 0.9,
        },
        {
            "rank": 3,
            "paper_id": "effective",
            "title": "Method reused paper",
            "impact_index": 10.0,
            "comparisons": 2,
            "confidence": 0.9,
        },
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert result["summary"]["version"] == "true_effective_citation_impact_v1"
    assert result["summary"]["final_score_formula"] == {
        "true_citation_score": 0.95,
        "effective_citation_score": 0.05,
    }
    assert "weighted combination" in result["summary"]["true_effective_citation"][
        "interpretation"
    ]
    assert "average" not in result["summary"]["true_effective_citation"][
        "interpretation"
    ]
    assert by_id["high_true"]["true_citation_score"] == 100.0
    assert by_id["effective"]["effective_citation_score"] == 100.0
    assert by_id["high_true"]["final_impact_score"] == 95.0
    assert by_id["effective"]["final_impact_score"] == 52.5
    assert by_id["high_true"]["score_95_true_05_effective"] == 95.0
    assert by_id["effective"]["score_90_true_10_effective"] == 55.0
    assert by_id["effective"]["score_50_true_50_effective"] == 75.0
    assert by_id["high_true"]["score_50_true_50_effective"] == 50.0
    assert by_id["model_liked"]["final_impact_score"] == 0.0
    assert result["rankings"][0]["paper_id"] == "high_true"


def test_true_citation_fields_resolve_multi_source_counts_and_conflicts():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "openalex",
                "title": "Semantic Scholar exceeds OpenAlex",
                "citation_count": 90,
                "metadata": {
                    "external_impact": {
                        "openalex_cited_by_count": 50,
                        "semantic_scholar_citation_count": 90,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "opencitations_rescue",
                "title": "OpenAlex missed citations",
                "metadata": {
                    "external_impact": {
                        "openalex_cited_by_count": 0,
                        "semantic_scholar_citation_count": 0,
                        "pubmed_citation_count": 0,
                        "opencitations_cited_by_count": 24,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "all_zero",
                "title": "Known zero citations",
                "metadata": {
                    "external_impact": {
                        "openalex_cited_by_count": 0,
                        "semantic_scholar_citation_count": 0,
                        "icite_citation_count": 0,
                        "crossref_is_referenced_by_count": 0,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "europepmc_ncbi",
                "title": "OpenAlex missed biomedical citations",
                "metadata": {
                    "external_impact": {
                        "openalex_cited_by_count": 0,
                        "europepmc_cited_by_count": 12,
                        "ncbi_pubmed_citedin_count": 15,
                        "ncbi_pmc_cited_by_count": 8,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "datacite_openaire",
                "title": "Repository impact",
                "metadata": {
                    "external_impact": {
                        "datacite_citation_count": 3,
                        "openaire_citation_count": 11,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "datacite_events",
                "title": "DataCite Events impact",
                "metadata": {
                    "external_impact": {
                        "datacite_events_cited_by_count": 12,
                        "wikidata_cited_by_count": 8,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "crossref_data",
                "title": "Crossref Data Citation impact",
                "metadata": {
                    "external_impact": {
                        "crossref_data_citation_count": 13,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "wikidata_only",
                "title": "Wikidata impact",
                "metadata": {
                    "external_impact": {
                        "wikidata_cited_by_count": 9,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "supplied",
                "title": "Supplied generic count",
                "citation_count": 10,
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert by_id["openalex"]["true_citation_raw"] == 90.0
    assert by_id["openalex"]["true_citation_source"] == "semantic_scholar"
    assert by_id["openalex"]["true_citation_openalex_raw"] == 50.0
    assert by_id["openalex"]["true_citation_semantic_scholar_raw"] == 90.0
    assert by_id["openalex"]["true_citation_max_raw"] == 90.0
    assert by_id["openalex"]["true_citation_source_count"] == 2
    assert by_id["openalex"]["true_citation_positive_source_count"] == 2
    assert by_id["openalex"]["true_citation_source_conflict_flag"] is False
    assert by_id["openalex"]["true_citation_source_agreement"] == "compatible"
    assert by_id["opencitations_rescue"]["true_citation_raw"] == 24.0
    assert by_id["opencitations_rescue"]["true_citation_source"] == "opencitations"
    assert by_id["opencitations_rescue"]["true_citation_openalex_raw"] == 0.0
    assert by_id["opencitations_rescue"]["true_citation_opencitations_raw"] == 24.0
    assert by_id["opencitations_rescue"]["true_citation_source_conflict_flag"] is True
    assert by_id["opencitations_rescue"]["true_citation_source_agreement"] == "conflict"
    assert by_id["all_zero"]["true_citation_raw"] == 0.0
    assert by_id["all_zero"]["true_citation_source"] == "multi_source_zero"
    assert by_id["all_zero"]["true_citation_crossref_raw"] == 0.0
    assert by_id["all_zero"]["true_citation_source_count"] == 4
    assert by_id["all_zero"]["true_citation_positive_source_count"] == 0
    assert by_id["all_zero"]["true_citation_source_conflict_flag"] is False
    assert by_id["all_zero"]["true_citation_source_agreement"] == "all_zero"
    assert by_id["europepmc_ncbi"]["true_citation_raw"] == 15.0
    assert by_id["europepmc_ncbi"]["true_citation_source"] == "ncbi_pubmed"
    assert by_id["europepmc_ncbi"]["true_citation_openalex_raw"] == 0.0
    assert by_id["europepmc_ncbi"]["true_citation_europepmc_raw"] == 12.0
    assert by_id["europepmc_ncbi"]["true_citation_ncbi_pubmed_raw"] == 15.0
    assert by_id["europepmc_ncbi"]["true_citation_source_conflict_flag"] is True
    assert by_id["datacite_openaire"]["true_citation_raw"] == 11.0
    assert by_id["datacite_openaire"]["true_citation_source"] == "openaire"
    assert by_id["datacite_openaire"]["true_citation_datacite_raw"] == 3.0
    assert by_id["datacite_openaire"]["true_citation_openaire_raw"] == 11.0
    assert by_id["datacite_openaire"]["true_citation_source_conflict_flag"] is True
    assert by_id["datacite_events"]["true_citation_raw"] == 12.0
    assert by_id["datacite_events"]["true_citation_source"] == "datacite_events"
    assert by_id["datacite_events"]["true_citation_datacite_events_raw"] == 12.0
    assert by_id["datacite_events"]["true_citation_wikidata_raw"] == 8.0
    assert by_id["crossref_data"]["true_citation_raw"] == 13.0
    assert by_id["crossref_data"]["true_citation_source"] == "crossref_data_citations"
    assert by_id["crossref_data"]["true_citation_crossref_data_citations_raw"] == 13.0
    assert by_id["wikidata_only"]["true_citation_raw"] == 9.0
    assert by_id["wikidata_only"]["true_citation_source"] == "wikidata"
    assert by_id["wikidata_only"]["true_citation_wikidata_raw"] == 9.0
    assert by_id["supplied"]["true_citation_raw"] == 10.0
    assert by_id["supplied"]["true_citation_source"] == "supplied_citation_count"
    assert by_id["supplied"]["true_citation_supplied_raw"] == 10.0
    assert by_id["openalex"]["true_citation_score"] == 100.0


def test_true_citation_fields_include_high_recall_external_exports():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "high_recall",
                "title": "High recall imported citation evidence",
                "metadata": {
                    "external_impact": {
                        "openalex_cited_by_count": 12,
                        "ucite_citation_count": 31,
                        "nih_occ_citation_count": 29,
                        "dimensions_citation_count": 42,
                        "scopus_citation_count": 40,
                        "web_of_science_citation_count": 38,
                        "lens_citation_count": 41,
                        "google_scholar_citation_count": 57,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "comparison",
                "title": "Comparison paper",
                "metadata": {
                    "external_impact": {
                        "openalex_cited_by_count": 20,
                        "semantic_scholar_citation_count": 20,
                    }
                },
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert by_id["high_recall"]["true_citation_raw"] == 57.0
    assert by_id["high_recall"]["true_citation_source"] == "google_scholar"
    assert by_id["high_recall"]["true_citation_openalex_raw"] == 12.0
    assert by_id["high_recall"]["true_citation_ucite_raw"] == 31.0
    assert by_id["high_recall"]["true_citation_nih_occ_raw"] == 29.0
    assert by_id["high_recall"]["true_citation_dimensions_raw"] == 42.0
    assert by_id["high_recall"]["true_citation_scopus_raw"] == 40.0
    assert by_id["high_recall"]["true_citation_web_of_science_raw"] == 38.0
    assert by_id["high_recall"]["true_citation_lens_raw"] == 41.0
    assert by_id["high_recall"]["true_citation_google_scholar_raw"] == 57.0
    assert by_id["high_recall"]["true_citation_source_count"] == 8
    assert by_id["high_recall"]["true_citation_positive_source_count"] == 8
    assert by_id["high_recall"]["true_citation_source_conflict_flag"] is True
    assert by_id["high_recall"]["true_citation_source_spread"] == 45.0
    assert by_id["high_recall"]["true_citation_score"] == 100.0


def test_true_citation_blocks_external_counts_when_external_title_mismatches():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "bad_identity",
                "title": "Resting-State Coactivation Patterns of Language Reorganization in Brain Tumors",
                "citation_count": 75,
                "metadata": {
                    "europepmc_title": "TbCAPs: A toolbox for co-activation pattern analysis.",
                    "external_impact": {
                        "openalex_cited_by_count": 3,
                        "semantic_scholar_citation_count": 64,
                        "crossref_is_referenced_by_count": 75,
                        "europepmc_cited_by_count": 68,
                        "ncbi_pubmed_citedin_count": 34,
                    },
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "good_identity",
                "title": "Matching external title",
                "metadata": {
                    "europepmc_title": "Matching external title",
                    "external_impact": {"crossref_is_referenced_by_count": 10},
                },
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert by_id["bad_identity"]["true_citation_raw"] is None
    assert by_id["bad_identity"]["true_citation_score"] == 0.0
    assert by_id["bad_identity"]["true_citation_source"] == "external_identity_mismatch"
    assert by_id["bad_identity"]["true_citation_crossref_raw"] == 75.0
    assert by_id["bad_identity"]["true_citation_identity_status"] == "mismatch"
    assert by_id["bad_identity"]["true_citation_identity_similarity"] < 0.7
    assert "TbCAPs" in by_id["bad_identity"]["true_citation_identity_external_title"]
    assert by_id["bad_identity"]["final_impact_score"] == 0.0
    assert by_id["good_identity"]["true_citation_raw"] == 10.0
    assert by_id["good_identity"]["true_citation_identity_status"] == "matched"


def test_compute_impact_index_v2_redistributes_missing_signals_but_marks_weak_evidence():
    papers = [
        Paper.model_validate({"paper_id": "p1", "title": "One", "references": []}),
        Paper.model_validate({"paper_id": "p2", "title": "Two", "references": []}),
    ]
    pairwise_rankings = [
        {
            "rank": 1,
            "paper_id": "p1",
            "title": "One",
            "impact_index": 90.0,
            "comparisons": 1,
            "confidence": 0.7,
        },
        {
            "rank": 2,
            "paper_id": "p2",
            "title": "Two",
            "impact_index": 10.0,
            "comparisons": 1,
            "confidence": 0.7,
        },
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert by_id["p1"]["impact_index_v2"] == 90.0
    assert by_id["p2"]["impact_index_v2"] == 10.0
    assert by_id["p1"]["fusion_signal_count"] == 1
    assert by_id["p1"]["fusion_available_weight"] == 0.3
    assert by_id["p1"]["fusion_weight_model_pairwise"] == 1.0
    assert by_id["p1"]["fusion_weight_field_year_normalized_citation"] == 0.0
    assert by_id["p1"]["evidence_quality_flag"] == "weak"
    assert "Only 1 impact signal" in by_id["p1"]["evidence_quality_reason"]
    assert by_id["p1"]["uncertainty_band"] == "high"
    assert by_id["p1"]["reportability_status"] == "low_confidence"
    assert by_id["p1"]["dominant_signal"] == "model_pairwise"
    assert by_id["p1"]["max_signal_contribution_share"] == 1.0


def test_compute_impact_index_v2_treats_known_zero_citations_as_zero_momentum_signal():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "zero",
                "title": "Known zero citations",
                "year": 2025,
                "field": "AI",
                "metadata": {"citation_count": 0},
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "positive",
                "title": "Positive citations without yearly history",
                "year": 2025,
                "field": "AI",
                "metadata": {"citation_count": 5},
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": 1,
            "paper_id": "zero",
            "title": "Known zero citations",
            "impact_index": 50.0,
            "comparisons": 1,
            "confidence": 0.7,
        },
        {
            "rank": 2,
            "paper_id": "positive",
            "title": "Positive citations without yearly history",
            "impact_index": 50.0,
            "comparisons": 1,
            "confidence": 0.7,
        },
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert result["summary"]["signal_availability"]["temporal_momentum"][
        "available_paper_count"
    ] == 1
    assert by_id["zero"]["temporal_momentum_raw"] == 0.0
    assert by_id["zero"]["fusion_weight_temporal_momentum"] > 0.0
    assert by_id["positive"]["fusion_weight_temporal_momentum"] == 0.0


def test_compute_impact_index_v3_does_not_let_zero_momentum_inflate_evidence_tier():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "zero",
                "title": "Known zero citations with one external signal",
                "year": 2025,
                "field": "AI",
                "metadata": {
                    "citation_count": 0,
                    "external_impact": {
                        "field_year_percentile": 0.8,
                        "semantic_scholar_influential_citation_count": 0,
                    },
                },
                "references": [],
            }
        )
    ]
    pairwise_rankings = [
        {
            "rank": 1,
            "paper_id": "zero",
            "title": "Known zero citations with one external signal",
            "impact_index": 80.0,
            "comparisons": 1,
            "confidence": 0.7,
        }
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    row = result["rankings"][0]

    assert row["evidence_first_signal_count"] == 3
    assert row["positive_evidence_signal_count"] == 1
    assert row["zero_evidence_signal_count"] == 2
    assert row["temporal_momentum_raw"] == 0.0
    assert row["temporal_momentum_score"] == 0.0
    assert row["external_field_year_percentile_raw"] == 80.0
    assert row["credible_evidence_tier"] == "moderate"
    assert result["summary"]["evidence_first"]["high_or_better_rate"] == 0.0
    signal_audit = result["summary"]["evidence_first"]["evidence_signal_audit"]
    assert signal_audit["positive_signal_count_distribution"] == {"1": 1}
    assert signal_audit["zero_signal_count_distribution"] == {"2": 1}
    assert signal_audit["available_signal_count_distribution"] == {"3": 1}
    assert signal_audit["mean_positive_evidence_signal_count"] == 1.0
    assert signal_audit["mean_zero_evidence_signal_count"] == 2.0
    assert signal_audit["papers_with_no_positive_evidence"] == 0
    assert "do not upgrade the tier" in signal_audit["interpretation"]


def test_compute_impact_index_v3_demotes_supplementary_records_from_ranked_papers():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "article",
                "title": "Real Article",
                "references": [],
                "metadata": {"source_kind": "article_or_extracted_article"},
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "supplement",
                "title": "Supplementary Online Content",
                "references": [],
                "metadata": {"source_kind": "supplement_or_extracted_supplement"},
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": 1,
            "paper_id": "supplement",
            "title": "Supplementary Online Content",
            "impact_index": 100.0,
            "comparisons": 4,
            "confidence": 0.9,
        },
        {
            "rank": 2,
            "paper_id": "article",
            "title": "Real Article",
            "impact_index": 10.0,
            "comparisons": 4,
            "confidence": 0.9,
        },
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert by_id["supplement"]["source_quality_flag"] == "non_article"
    assert by_id["supplement"]["source_quality_adjustment_factor"] == 0.0
    assert by_id["supplement"]["reportability_status"] == "excluded_non_article"
    assert by_id["supplement"]["final_impact_score"] == 0.0
    assert result["rankings"][0]["paper_id"] == "article"


def test_compute_impact_index_v2_uses_citation_signals_when_model_has_no_comparisons():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Cited",
                "year": 2024,
                "field": "AI",
                "citation_count": 20,
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p2",
                "title": "Uncited",
                "year": 2024,
                "field": "AI",
                "citation_count": 1,
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": 1,
            "paper_id": "p1",
            "title": "Cited",
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        },
        {
            "rank": 2,
            "paper_id": "p2",
            "title": "Uncited",
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        },
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert by_id["p1"]["impact_index_v2"] > by_id["p2"]["impact_index_v2"]
    assert by_id["p1"]["fusion_signal_count"] == 1
    assert by_id["p1"]["fusion_weight_model_pairwise"] == 0.0
    assert by_id["p1"]["fusion_weight_field_year_normalized_citation"] == 1.0


def test_compute_impact_index_v2_uses_log_robust_citation_normalization_for_outliers():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "outlier",
                "title": "Extreme Citation Outlier",
                "year": 2024,
                "field": "AI",
                "citation_count": 1_000_000,
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "solid",
                "title": "Solid Paper",
                "year": 2024,
                "field": "AI",
                "citation_count": 1_000,
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "small",
                "title": "Small Paper",
                "year": 2024,
                "field": "AI",
                "citation_count": 10,
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert by_id["outlier"]["field_year_normalized_citation_score"] == 100.0
    assert by_id["solid"]["field_year_normalized_citation_score"] > 45.0
    assert by_id["solid"]["field_year_normalized_citation_score"] < 70.0
    assert by_id["solid"]["reportability_status"] == "low_confidence"
    assert by_id["solid"]["uncertainty_band"] == "high"
    assert by_id["outlier"]["citation_outlier_flag"] is True
    assert by_id["outlier"]["citation_outlier_ratio"] > 100.0
    assert by_id["solid"]["citation_outlier_flag"] is False


def test_citation_component_uses_rank_percentiles_to_avoid_outlier_compression():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "extreme",
                "title": "Extreme Citation Outlier",
                "year": 2024,
                "field": "AI",
                "citation_count": 1_000_000,
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "strong",
                "title": "Strong Non-Outlier Paper",
                "year": 2024,
                "field": "AI",
                "citation_count": 100,
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "near_peer",
                "title": "Nearby Peer Paper",
                "year": 2024,
                "field": "AI",
                "citation_count": 90,
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert by_id["extreme"]["field_year_normalized_citation_score"] == 100.0
    assert by_id["strong"]["field_year_normalized_citation_score"] == 50.0
    assert by_id["near_peer"]["field_year_normalized_citation_score"] == 0.0


def test_compute_impact_index_v3_adds_relative_citation_ratio_signal():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "fast",
                "title": "Fast moving niche paper",
                "year": 2024,
                "field": "AI",
                "citation_count": 50,
                "metadata": {
                    "citation_observation_year": 2026,
                    "expected_citation_rate": 5,
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "broad",
                "title": "Broad field paper",
                "year": 2024,
                "field": "AI",
                "citation_count": 60,
                "metadata": {
                    "citation_observation_year": 2026,
                    "expected_citation_rate": 20,
                },
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert "relative_citation_ratio" in result["summary"]["signal_availability"]
    assert result["summary"]["signal_availability"]["relative_citation_ratio"][
        "available_paper_count"
    ] == 2
    assert by_id["fast"]["relative_citation_ratio_score"] == 100.0
    assert by_id["fast"]["relative_citation_ratio_raw"] > by_id["broad"][
        "relative_citation_ratio_raw"
    ]
    assert by_id["fast"]["relative_expected_citation_rate_raw"] < by_id["broad"][
        "relative_expected_citation_rate_raw"
    ]
    assert by_id["fast"]["relative_expected_citation_rate_raw"] == 5.0


def test_compute_impact_index_v2_adds_v3_evidence_first_external_adoption_layer():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Adopted work",
                "year": 2024,
                "field": "AI",
                "citation_count": 120,
                "metadata": {
                    "external_impact": {
                        "field_year_percentile": 0.95,
                        "patent_citations": 4,
                        "policy_citations": 2,
                        "software_mentions": 20,
                        "dataset_reuses": 8,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "p2",
                "title": "Model liked work",
                "year": 2024,
                "field": "AI",
                "citation_count": 10,
                "metadata": {
                    "external_impact": {
                        "field_year_percentile": 0.35,
                        "patent_citations": 0,
                        "policy_citations": 0,
                        "software_mentions": 1,
                        "dataset_reuses": 0,
                    }
                },
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": 1,
            "paper_id": "p2",
            "title": "Model liked work",
            "impact_index": 100.0,
            "comparisons": 1,
            "confidence": 0.8,
        },
        {
            "rank": 2,
            "paper_id": "p1",
            "title": "Adopted work",
            "impact_index": 10.0,
            "comparisons": 1,
            "confidence": 0.8,
        },
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert result["summary"]["version"] == "true_effective_citation_impact_v1"
    assert result["summary"]["legacy_fusion_version"] == "impact_index_v3_evidence_first"
    assert "external_adoption" in result["summary"]["signal_availability"]
    assert result["summary"]["evidence_first"]["high_or_better_rate"] == 0.5
    assert result["summary"]["evidence_first"]["dominant_score_family_counts"][
        "evidence_supported"
    ] == 1
    assert result["summary"]["evidence_first"]["external_adoption_coverage_rate"] == 1.0
    assert by_id["p1"]["impact_index_v3"] > by_id["p2"]["impact_index_v3"]
    assert by_id["p1"]["evidence_supported_score"] > by_id["p1"]["model_supported_score"]
    assert by_id["p1"]["external_field_year_percentile_score"] == 95.0
    assert by_id["p1"]["patent_impact_score"] == 100.0
    assert by_id["p1"]["policy_impact_score"] == 100.0
    assert by_id["p1"]["artifact_adoption_score"] == 100.0
    assert math.isclose(by_id["p1"]["external_adoption_score"], 98.125)
    assert by_id["p1"]["credible_evidence_tier"] in {"high", "very_high"}
    assert by_id["p1"]["reportability_status"] == "reportable"
    assert by_id["p1"]["reportability_basis"] == "impact_index_v3_evidence_first"
    assert by_id["p1"]["legacy_reportability_status"] in {"low_confidence", "exploratory", "reportable"}
    assert by_id["p1"]["missing_high_value_evidence"] == []
    assert "patent" in by_id["p2"]["missing_high_value_evidence"]
    assert by_id["p2"]["dominant_score_family"] == "model_assisted"
    assert by_id["p2"]["reportability_status"] != "reportable"


def test_compute_impact_index_v3_accepts_media_and_award_external_evidence():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "visible",
                "title": "Visible prize paper",
                "metadata": {
                    "external_impact": {
                        "media_mentions": 12,
                        "award_count": 2,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "quiet",
                "title": "Quiet paper",
                "metadata": {
                    "external_impact": {
                        "media_mentions": 1,
                        "award_count": 0,
                    }
                },
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert by_id["visible"]["media_impact_score"] == 100.0
    assert by_id["visible"]["award_impact_score"] == 100.0
    assert by_id["visible"]["media_mentions_raw"] == 12.0
    assert by_id["visible"]["award_count_raw"] == 2.0
    assert by_id["visible"]["external_adoption_score"] > by_id["quiet"][
        "external_adoption_score"
    ]


def test_compute_impact_index_v3_treats_optional_adoption_gaps_as_disclosure_not_blocker():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "core",
                "title": "Core scientific influence paper",
                "year": 2024,
                "field": "AI",
                "citation_count": 80,
                "metadata": {
                    "expected_citation_rate": 10,
                    "citation_counts_by_year": {"2025": 30},
                    "external_impact": {
                        "field_year_percentile": 0.92,
                        "semantic_scholar_influential_citation_count": 6,
                    },
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "weak",
                "title": "Weak comparator",
                "year": 2024,
                "field": "AI",
                "citation_count": 1,
                "metadata": {"expected_citation_rate": 10},
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": 1,
            "paper_id": "weak",
            "title": "Weak comparator",
            "impact_index": 90.0,
            "comparisons": 1,
            "confidence": 0.8,
        },
        {
            "rank": 2,
            "paper_id": "core",
            "title": "Core scientific influence paper",
            "impact_index": 10.0,
            "comparisons": 1,
            "confidence": 0.8,
        },
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    row = {row["paper_id"]: row for row in result["rankings"]}["core"]

    assert row["credible_evidence_tier"] == "high"
    assert row["dominant_score_family"] == "evidence_supported"
    assert row["reportability_status"] == "reportable"
    assert "disclose optional missing adoption evidence" in row["reportability_reason"]
    assert "patent" in row["missing_high_value_evidence"]
    assert "policy" in row["missing_high_value_evidence"]
    assert "artifact_adoption" in row["missing_high_value_evidence"]
    assert "Add patent impact evidence." not in row["reportability_gap_actions"]
    assert "Add policy impact evidence." not in row["reportability_gap_actions"]
    assert row["reportability_gap_count"] == 0


def test_compute_impact_index_v3_uses_semantic_scholar_influential_citations():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "influential",
                "title": "Influential citation paper",
                "metadata": {
                    "external_impact": {
                        "semantic_scholar_influential_citation_count": 8,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "ordinary",
                "title": "Ordinary citation paper",
                "metadata": {
                    "external_impact": {
                        "semantic_scholar_influential_citation_count": 1,
                    }
                },
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": 1,
            "paper_id": "ordinary",
            "title": "Ordinary citation paper",
            "impact_index": 100.0,
            "comparisons": 1,
            "confidence": 0.8,
        },
        {
            "rank": 2,
            "paper_id": "influential",
            "title": "Influential citation paper",
            "impact_index": 0.0,
            "comparisons": 1,
            "confidence": 0.8,
        },
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert "influential_citation" in result["summary"]["signal_availability"]
    assert by_id["influential"]["influential_citation_score"] == 100.0
    assert by_id["ordinary"]["influential_citation_score"] == 0.0
    assert by_id["influential"]["influential_citation_count_raw"] == 8.0
    assert by_id["influential"]["evidence_first_signal_count"] == 1


def test_compute_impact_index_records_life_science_effective_citations():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "assay",
                "title": "Reusable biomarker assay",
                "year": 2022,
                "field": "Life Sciences",
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "trial",
                "title": "Phase 3 trial using assay",
                "year": 2024,
                "field": "Life Sciences",
                "references": [
                    {"reference_id": "assay", "title": "Reusable biomarker assay"}
                ],
                "citation_contexts": [
                    {
                        "reference_id": "assay",
                        "sentence": "Patients were stratified using the assay protocol described by the prior study.",
                        "section": "Methods",
                    }
                ],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "guideline",
                "title": "Clinical practice guideline",
                "year": 2025,
                "field": "Life Sciences",
                "references": [
                    {"reference_id": "assay", "title": "Reusable biomarker assay"}
                ],
                "citation_contexts": [
                    {
                        "reference_id": "assay",
                        "sentence": "The NCCN guideline recommends this diagnostic biomarker assay for patient selection.",
                        "section": "Guidelines",
                    }
                ],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "review",
                "title": "Background review",
                "year": 2025,
                "field": "Life Sciences",
                "references": [
                    {"reference_id": "assay", "title": "Reusable biomarker assay"}
                ],
                "citation_contexts": [
                    {
                        "reference_id": "assay",
                        "sentence": "Several previous studies have examined biomarkers in this disease.",
                        "section": "Introduction",
                    }
                ],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    row = {row["paper_id"]: row for row in result["rankings"]}["assay"]

    assert row["effective_citation_edge_count"] == 3
    assert row["effective_citation_context_count"] == 3
    assert row["effective_citation_count"] == 2
    assert row["effective_citation_guideline_count"] == 1
    assert row["effective_citation_clinical_count"] == 1
    assert row["effective_citation_method_count"] == 1
    assert row["effective_citation_background_count"] == 1
    assert row["effective_citation_unknown_count"] == 0
    assert row["effective_citation_score"] == 100.0
    assert row["effective_citation_evidence_level"] == "high"
    assert row["effective_citation_coverage"] == "contextualized"


def test_model_unknown_role_does_not_fall_back_to_keyword_rules():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "cited",
                "title": "Cited assay paper",
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "citing",
                "title": "Citing paper",
                "references": [
                    {"reference_id": "cited", "title": "Cited assay paper"}
                ],
                "citation_contexts": [
                    {
                        "reference_id": "cited",
                        "sentence": "We used this method as the assay protocol.",
                        "section": "Methods",
                        "role": "unknown",
                        "role_source": "model",
                        "role_validation_status": "valid",
                    }
                ],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    row = {row["paper_id"]: row for row in result["rankings"]}["cited"]

    assert row["effective_citation_method_count"] == 0
    assert row["effective_citation_unknown_count"] == 1
    assert row["effective_citation_raw"] == 0.1


def test_model_canonical_roles_are_honored_without_keyword_fallback():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "dataset",
                "title": "Reusable dataset paper",
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "contrast",
                "title": "Contrasting result paper",
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "dataset_citing",
                "title": "Paper citing a cached dataset role",
                "references": [
                    {"reference_id": "dataset", "title": "Reusable dataset paper"}
                ],
                "citation_contexts": [
                    {
                        "reference_id": "dataset",
                        "sentence": "This prior work is cited here.",
                        "section": "Related Work",
                        "role": "dataset_or_tool",
                        "role_source": "model",
                    }
                ],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "contrast_citing",
                "title": "Paper citing a cached contrasting role",
                "references": [
                    {"reference_id": "contrast", "title": "Contrasting result paper"}
                ],
                "citation_contexts": [
                    {
                        "reference_id": "contrast",
                        "sentence": "This prior work is cited here.",
                        "section": "Related Work",
                        "role": "negative_or_contrasting",
                        "role_source": "model",
                    }
                ],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert by_id["dataset"]["effective_citation_dataset_or_tool_count"] == 1
    assert by_id["dataset"]["effective_citation_unknown_count"] == 0
    assert by_id["dataset"]["effective_citation_raw"] == 1.5
    assert by_id["contrast"]["effective_citation_negative_count"] == 1
    assert by_id["contrast"]["effective_citation_unknown_count"] == 0
    assert by_id["contrast"]["effective_citation_raw"] == 1.1


def test_compute_impact_index_counts_cross_source_duplicate_edge_once():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "assay",
                "title": "Reusable biomarker assay",
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "trial",
                "title": "Trial using assay",
                "references": [
                    {
                        "reference_id": "assay",
                        "title": "Reusable biomarker assay",
                        "raw_citation": "Input reference list edge",
                    },
                    {
                        "reference_id": "assay",
                        "title": "Reusable biomarker assay",
                        "raw_citation": "OpenCitations internal DOI edge",
                    },
                ],
                "citation_contexts": [
                    {
                        "reference_id": "assay",
                        "sentence": "The assay protocol was used for stratification.",
                        "section": "Methods",
                    }
                ],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    row = {row["paper_id"]: row for row in result["rankings"]}["assay"]

    assert row["effective_citation_edge_count"] == 1
    assert row["effective_citation_context_count"] == 1
    assert row["effective_citation_method_count"] == 1
    assert row["effective_citation_raw"] == 1.6


def test_compute_impact_index_uses_impact_selected_references_as_effective_citations():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "assay",
                "title": "Reusable biomarker assay",
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "trial",
                "title": "Trial using assay",
                "references": [
                    {"reference_id": "assay", "title": "Reusable biomarker assay"}
                ],
                "metadata": {
                    "impact_selected_references": [
                        {
                            "reference_id": "assay",
                            "rank": 1,
                            "confidence": 0.82,
                            "valid": True,
                        }
                    ]
                },
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "background",
                "title": "Background-only article",
                "references": [
                    {"reference_id": "assay", "title": "Reusable biomarker assay"}
                ],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    row = {row["paper_id"]: row for row in result["rankings"]}["assay"]

    assert row["effective_citation_edge_count"] == 2
    assert row["effective_citation_context_count"] == 0
    assert row["effective_citation_unknown_count"] == 2
    assert row["effective_citation_impact_selected_count"] == 1
    assert row["effective_citation_impact_selected_top5_count"] == 1
    assert row["effective_citation_impact_selected_raw"] == 1.0
    assert row["effective_citation_raw"] == 1.1
    assert row["effective_citation_count"] == 1
    assert row["effective_citation_evidence_level"] == "low"


def test_compute_impact_index_does_not_double_count_impact_selection_over_context():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "assay",
                "title": "Reusable biomarker assay",
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "trial",
                "title": "Trial using assay",
                "references": [
                    {"reference_id": "assay", "title": "Reusable biomarker assay"}
                ],
                "citation_contexts": [
                    {
                        "reference_id": "assay",
                        "sentence": "We used the assay protocol from the prior study.",
                        "section": "Methods",
                    }
                ],
                "metadata": {
                    "impact_selected_references": [
                        {"reference_id": "assay", "rank": 1, "valid": True}
                    ]
                },
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    row = {row["paper_id"]: row for row in result["rankings"]}["assay"]

    assert row["effective_citation_method_count"] == 1
    assert row["effective_citation_impact_selected_count"] == 1
    assert row["effective_citation_impact_selected_raw"] == 1.0
    assert row["effective_citation_raw"] == 1.6


def test_compute_impact_index_accepts_life_science_external_evidence_aliases():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "translated",
                "title": "Translated biomedical paper",
                "metadata": {
                    "external_impact": {
                        "pubmed_citation_count": 80,
                        "nih_rcr": 3.2,
                        "guideline_citations": 2,
                        "clinical_trial_citations": 4,
                        "drug_approval_mentions": 1,
                        "protocol_reuses": 3,
                        "reagent_or_assay_reuses": 5,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "ordinary",
                "title": "Ordinary biomedical paper",
                "metadata": {
                    "external_impact": {
                        "pubmed_citation_count": 5,
                        "nih_rcr": 0.8,
                        "guideline_citations": 0,
                        "clinical_trial_citations": 0,
                        "protocol_reuses": 0,
                    }
                },
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert by_id["translated"]["citation_count_raw"] == 80.0
    assert by_id["translated"]["relative_citation_ratio_raw"] == 3.2
    assert by_id["translated"]["policy_citations_raw"] == 2.0
    assert by_id["translated"]["artifact_adoption_raw"] == 8.0
    assert by_id["translated"]["life_science_translation_raw"] == 7.0
    assert by_id["translated"]["life_science_translation_score"] == 100.0
    assert by_id["translated"]["external_adoption_score"] > by_id["ordinary"][
        "external_adoption_score"
    ]


def test_compute_impact_index_v3_uses_bip_external_impact_evidence():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "bip_high",
                "title": "BIP high impact paper",
                "metadata": {"external_impact": {"bip_impact_percentile": 0.92}},
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "bip_low",
                "title": "BIP low impact paper",
                "metadata": {"external_impact": {"bip_impact_percentile": 0.20}},
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": index,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "impact_index": 50.0,
            "comparisons": 0,
            "confidence": 0.0,
        }
        for index, paper in enumerate(papers, start=1)
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert by_id["bip_high"]["bip_impact_score"] == 92.0
    assert by_id["bip_high"]["bip_impact_raw"] == 92.0
    assert by_id["bip_high"]["external_adoption_score"] > by_id["bip_low"][
        "external_adoption_score"
    ]


def test_compute_impact_index_v2_applies_custom_weights_to_v3_evidence_layer():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "citation",
                "title": "Citation supported",
                "year": 2024,
                "field": "AI",
                "citation_count": 100,
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "momentum",
                "title": "Momentum supported",
                "year": 2024,
                "field": "AI",
                "citation_count": 1,
                "metadata": {"citation_counts_by_year": {"2023": 0, "2024": 50}},
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": 1,
            "paper_id": "citation",
            "title": "Citation supported",
            "impact_index": 50.0,
            "comparisons": 1,
            "confidence": 0.8,
        },
        {
            "rank": 2,
            "paper_id": "momentum",
            "title": "Momentum supported",
            "impact_index": 50.0,
            "comparisons": 1,
            "confidence": 0.8,
        },
    ]

    citation_weighted = compute_impact_index_v2(
        papers,
        pairwise_rankings,
        weights={
            "model_pairwise": 0.0,
            "field_year_normalized_citation": 1.0,
            "citation_graph": 0.0,
            "citation_context_quality": 0.0,
            "temporal_momentum": 0.0,
        },
    )
    momentum_weighted = compute_impact_index_v2(
        papers,
        pairwise_rankings,
        weights={
            "model_pairwise": 0.0,
            "field_year_normalized_citation": 0.0,
            "citation_graph": 0.0,
            "citation_context_quality": 0.0,
            "temporal_momentum": 1.0,
        },
    )

    citation_by_id = {row["paper_id"]: row for row in citation_weighted["rankings"]}
    momentum_by_id = {row["paper_id"]: row for row in momentum_weighted["rankings"]}

    assert citation_by_id["citation"]["impact_index_v2"] > citation_by_id["momentum"][
        "impact_index_v2"
    ]
    assert momentum_by_id["momentum"]["impact_index_v2"] > momentum_by_id["citation"][
        "impact_index_v2"
    ]
    assert momentum_weighted["rankings"][0]["paper_id"] == "citation"
    assert momentum_weighted["rankings"][0]["final_score_method"] == (
        "true_effective_citation_impact_v1"
    )


def test_compute_impact_index_v3_does_not_treat_zero_external_counts_as_strong_evidence():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "p1",
                "title": "Sparse external evidence",
                "year": 2024,
                "field": "AI",
                "citation_count": 10,
                "metadata": {
                    "external_impact": {
                        "field_year_percentile": 0.35,
                        "patent_citations": 0,
                        "policy_citations": 0,
                        "dataset_reuses": 1,
                    }
                },
                "references": [],
            }
        )
    ]
    pairwise_rankings = [
        {
            "rank": 1,
            "paper_id": "p1",
            "title": "Sparse external evidence",
            "impact_index": 50.0,
            "comparisons": 1,
            "confidence": 0.8,
        }
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    row = result["rankings"][0]

    assert row["credible_evidence_tier"] == "moderate"
    assert row["reportability_status"] == "exploratory"
    assert "patent" in row["missing_high_value_evidence"]
    assert "policy" in row["missing_high_value_evidence"]
    assert row["reportability_gap_count"] >= 3
    assert "Add patent impact evidence." in row["reportability_gap_actions"]
    assert "Add policy impact evidence." in row["reportability_gap_actions"]
    assert any(
        "Strengthen evidence tier" in action
        for action in row["reportability_gap_actions"]
    )
    assert result["summary"]["evidence_first"]["high_or_better_rate"] == 0.0


def test_compute_impact_index_v3_uses_credibility_adjusted_final_score_for_sorting():
    papers = [
        Paper.model_validate(
            {
                "paper_id": "evidence",
                "title": "Evidence supported paper",
                "year": 2024,
                "field": "AI",
                "citation_count": 80,
                "metadata": {
                    "external_impact": {
                        "field_year_percentile": 0.90,
                        "patent_citations": 3,
                        "policy_citations": 2,
                        "dataset_reuses": 8,
                    }
                },
                "references": [],
            }
        ),
        Paper.model_validate(
            {
                "paper_id": "model_only",
                "title": "Model only paper",
                "references": [],
            }
        ),
    ]
    pairwise_rankings = [
        {
            "rank": 1,
            "paper_id": "model_only",
            "title": "Model only paper",
            "impact_index": 100.0,
            "comparisons": 1,
            "confidence": 0.8,
        },
        {
            "rank": 2,
            "paper_id": "evidence",
            "title": "Evidence supported paper",
            "impact_index": 40.0,
            "comparisons": 1,
            "confidence": 0.8,
        },
    ]

    result = compute_impact_index_v2(papers, pairwise_rankings)
    by_id = {row["paper_id"]: row for row in result["rankings"]}

    assert by_id["model_only"]["impact_index_v3"] > by_id["evidence"]["impact_index_v3"]
    assert by_id["model_only"]["credibility_adjustment_factor"] < by_id["evidence"][
        "credibility_adjustment_factor"
    ]
    assert by_id["model_only"]["final_impact_score"] < by_id["evidence"][
        "final_impact_score"
    ]
    assert result["rankings"][0]["paper_id"] == "evidence"
    assert result["summary"]["evidence_first"]["credibility_adjustment"]["adjusted_sorting"] is False
    assert by_id["evidence"]["final_score_method"] == "true_effective_citation_impact_v1"
